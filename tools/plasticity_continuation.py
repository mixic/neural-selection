#!/usr/bin/env python3
"""Randomized mixed-world plasticity continuations; no third-party dependencies."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess


def summarize_trial(path):
    with Path(path).open() as f: rows = list(csv.DictReader(f))
    groups = {}
    for frozen, name in ((0, 'control'), (1, 'frozen')):
        animals = [r for r in rows if int(r['frozen']) == frozen]
        if not animals:
            raise ValueError('Each trial needs animals in both groups')
        survival = [int(r['alive']) for r in animals]
        followup = [int(r['observed_step']) - int(r['start_step']) for r in animals]
        offspring = [int(r['final_offspring']) - int(r['initial_offspring']) for r in animals]
        groups[name] = dict(n=len(animals), surviving=sum(survival),
            survival_fraction=statistics.mean(survival),
            mean_remaining_life_steps=statistics.mean(followup),
            mean_additional_offspring=statistics.mean(offspring),
            total_additional_offspring=sum(offspring),
            observed_animal_steps=sum(followup),
            deaths_by_cause={str(c): sum(int(r['death_cause']) == c and not int(r['alive'])
                                       for r in animals) for c in range(1, 5)})
    return dict(groups=groups, differences_frozen_minus_control={
        k: groups['frozen'][k] - groups['control'][k] for k in
        ('survival_fraction', 'mean_remaining_life_steps', 'mean_additional_offspring')})


def aggregate(trials):
    # Resample whole trials, not socially interacting animals within a world.
    rng = random.Random(94351)
    result = {}
    for metric in trials[0]['differences_frozen_minus_control']:
        values = [t['differences_frozen_minus_control'][metric] for t in trials]
        ci = None
        if len(values) >= 3:
            means = sorted(statistics.mean(rng.choices(values, k=len(values)))
                           for _ in range(10000))
            ci = [means[249], means[9749]]
        result[metric] = dict(mean=statistics.mean(values),
                             trial_differences=values, bootstrap_95_percent_interval=ci)
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path, action='append',
                   help='Repeat for independent starting worlds; each gets --trials assignments')
    p.add_argument('--executable', type=Path, default=Path('build/neural_selection'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--steps', type=int, default=10000, help='Additional world steps per trial')
    p.add_argument('--seed', type=int, default=1, help='Assignment RNG only; world RNG restored unchanged')
    p.add_argument('--fraction', type=float, default=.5)
    args = p.parse_args()
    if args.trials < 2 or args.steps < 1 or not 0 < args.fraction < 1 or args.seed < 0:
        p.error('Need >=2 trials, positive steps, nonnegative seed and fraction strictly between 0 and 1')
    exe = args.executable.resolve()
    checkpoints = [c.resolve(strict=True) for c in args.checkpoint]
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(protocol='mixed-world-long-term-plasticity-v1', executable=str(exe),
        executable_sha256=digest(exe), checkpoints=[dict(path=str(c), sha256=digest(c)) for c in checkpoints],
        trials_per_checkpoint=args.trials, continuation_steps=args.steps,
        assignment_seed=args.seed, frozen_fraction=args.fraction,
        interpretation='Random assignment conditional on saved worlds. Social interference is allowed; '
        'these are not independent evolved-world replicates. Descendants are untreated and excluded. '
        'Negative frozen-minus-control differences indicate a disadvantage. '
        'Intervals resample trials; they do not capture variation between unprovided starting worlds.')
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    trials=[]
    for world_index, checkpoint in enumerate(checkpoints):
        for index in range(args.trials):
            seed=args.seed + world_index*args.trials + index
            if seed >= 2**64: raise ValueError('Assignment seed overflow')
            directory=args.output/f'world-{world_index:03d}-trial-{index:03d}'
            directory.mkdir()
            output=directory/'world.csv'
            command=[str(exe), '--resume', str(checkpoint), '--output', str(output.resolve()),
                '--checkpoint', str((directory/'continuation.gz').resolve()),
                '--continuation-steps', str(args.steps), '--plasticity-ablation',
                '--ablation-seed', str(seed), '--ablation-fraction', str(args.fraction)]
            (directory/'command.json').write_text(json.dumps(command,indent=2)+'\n')
            with (directory/'simulation.log').open('w') as log:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:
                (args.output/'status.json').write_text(json.dumps(dict(state='failed',trial=str(directory),
                    returncode=process.returncode),indent=2)+'\n')
                raise RuntimeError(f'Trial failed; see {directory}/simulation.log')
            trial=summarize_trial(str(output)+'.plasticity.csv')
            # A signal-driven early stop exits normally; do not mistake that for completion.
            with Path(str(output)+'.plasticity.csv').open() as f:
                rows=list(csv.DictReader(f))
            # If the whole cohort died, CSV follow-up legitimately ends early. The main log still
            # records the actual final world step (including checkpoint messages).
            with output.open() as f: world_rows=list(csv.DictReader(f))
            final=int(world_rows[-1]['step'])
            start=int(rows[0]['start_step'])
            if final != start + args.steps:
                (args.output/'status.json').write_text(json.dumps(dict(state='failed',
                    trial=str(directory),error='Trial interrupted before requested horizon',
                    final_step=final,expected_step=start+args.steps),indent=2)+'\n')
                raise RuntimeError('Trial interrupted before requested horizon')
            trial.update(world=world_index, trial=index, assignment_seed=seed,
                         cohort_file=str(Path(str(output)+'.plasticity.csv').resolve()), final_step=final)
            (directory/'results.json').write_text(json.dumps(trial,indent=2)+'\n')
            trials.append(trial)
            result=dict(plan=plan,completed_trials=len(trials),trials=trials,
                        aggregate=aggregate(trials))
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            (args.output/'status.json').write_text(json.dumps(dict(state='running',completed_trials=len(trials)),indent=2)+'\n')
            print(f'Finished world {world_index}, trial {index}: {trial["differences_frozen_minus_control"]}',flush=True)
    (args.output/'status.json').write_text(json.dumps(dict(state='complete',completed_trials=len(trials)),indent=2)+'\n')


if __name__ == '__main__': main()
