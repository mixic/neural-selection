"""Export genomes, held-out results, median-return replays and training charts."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
from .replay import rollout


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='runs/gym-suite');p.add_argument('--output',default='examples/gym-suite');p.add_argument('--no-gifs',action='store_true');args=p.parse_args()
    root=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);specs=json.loads((root/'plan.json').read_text());records=[]
    os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
    for spec in specs:
        source=root/spec['name'];result_path=source/'results.json'
        if not result_path.exists():continue
        target=out/spec['name'];target.mkdir(exist_ok=True);result=json.loads(result_path.read_text())
        for filename in ('champion.json','initial.json','history.csv','results.json','command.json'):shutil.copy2(source/filename,target/filename)
        for filename in ('sensory-diagnostic.json','goal-diagnostic.json'):
            if (source/filename).exists():shutil.copy2(source/filename,target/filename)
        returns=result['champion_holdout']['returns'];ordered=sorted(range(len(returns)),key=lambda i:returns[i]);index=ordered[len(ordered)//2];seed=result['heldout_seeds'][index]
        replay=None
        if not args.no_gifs:replay=rollout(target/'champion.json',result['config'],seed,target/'rollout.gif')
        (target/'replay.json').write_text(json.dumps({'selection':'upper median held-out episode by return','episode_index':index,'seed':seed,'result':replay},indent=2)+'\n')
        genome=json.loads((target/'champion.json').read_text());record={'name':spec['name'],'env':spec['env'],'result':result,'neurons':len(genome['neurons']),'synapses':len(genome['synapses']),'replay_seed':seed};records.append(record)
    lines=['# Exploratory Gymnasium environment suite','', 'Independent evolutionary searches using the same physiological brain model. Rewards select genomes externally and never enter the brain. Hidden/motor applied current remains zero. Each champion starts every episode from its genome, with no lifetime state inherited.','', 'The table contains **untouched final test seeds**, evaluated on CPU. Each environment received one search run with the recorded budget; this is an exploratory comparison, not an exhaustive tuning study.','', '| Environment | Random return | Best initial return | Evolved return | Goal success | Generations | Brain neurons / synapses |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for record in records:
        r=record['result'];success=f"{r['success_rate']*100:.0f}% ({len(r['heldout_seeds'])} episodes)" if 'success_rate' in r else '—'
        goal=out/record['name']/'goal-diagnostic.json'
        if goal.exists():
            g=json.loads(goal.read_text());success=f"{g['success_rate']*100:.0f}% ({len(g['seeds'])} episodes)"
        lines.append(f"| {record['env']} | {r['random_holdout']['mean']:.3f} | {r['initial_holdout']['mean']:.3f} | **{r['champion_holdout']['mean']:.3f}** | {success} | {r['completed_generations']} | {record['neurons']} / {record['synapses']} |")
    lines+=['','Brain sizes are genome counts, including disabled elements. The complete results retain all per-episode returns.','','## Observations','']
    for record in records:
        diagnostic=out/record['name']/'sensory-diagnostic.json'
        if diagnostic.exists():
            d=json.loads(diagnostic.read_text());c=d['conditions']
            lines += [f"{record['env']} sensory check, on {len(d['seeds'])} separate post-training seeds: normal observations {c['normal']['success_rate']*100:.0f}% success, constant opponent input {c['neutral']['success_rate']*100:.0f}%, first view frozen {c['frozen']['success_rate']*100:.0f}%. These interventions test sensory dependence and do not establish learning within a lifetime.",'']
        goal=out/record['name']/'goal-diagnostic.json'
        if goal.exists():
            g=json.loads(goal.read_text());lines += [f"{record['env']} reached its goal in {g['success_count']}/{len(g['seeds'])} held-out episodes.",'']
        r=record['result']
        if r['champion_holdout']['mean']<=r['random_holdout']['mean']:
            lines += [f"{record['env']} did not beat random actions on held-out seeds. A better validation score did not transfer to the final test set.",'']
    lines+=['## Training curves','','![Validation returns](training.png)','','For the corrected protocol, these are validation returns of each generation’s training-selected winner; validation is reporting only. Historical runs from the cancelled protocol retained validation-selected champions and are contaminated.','','## Replays','','Each GIF shows the **upper-median held-out episode by return**, chosen after final testing. It is not a selected best case. The original Gymnasium rendering displays the world; brains receive only the observations documented in the adapter guide.']
    for record in records:
        name=record['name'];lines += ['',f"### {record['env']}",'',f"![{record['env']}]({name}/rollout.gif)",'',f"Replay seed: `{record['replay_seed']}`. [Genome]({name}/champion.json), [complete test results]({name}/results.json), [training history]({name}/history.csv), [exact command]({name}/command.json)."]
    lines+=['','## Interpretation and limits','','Navigation success does not by itself demonstrate memory or learning within a lifetime. MiniGrid Memory needs a separate cue-dependence/state-ablation analysis before attributing success to memory. A fixed sequence or simple reactive policy can perform well in some benchmarks. These are reduced neuron models; optimized parameters and firing rates may lie outside their experimentally fitted regimes.','','MiniGrid uses the full standard 7×7 partial symbolic view with object/color/state fields. The fixed text mission and compass are removed. There is no privileged global position, full map, goal direction, reward shaping or action mask. The seven native actions remain available. The Memory search/evaluation cap is 196 steps (its native limit is 245); Empty-Random uses its native 144-step limit.','','CUDA performs population evaluation for MiniGrid; classic-control searches use eight CPU workers. Validation and all final testing run on CPU. Test seeds were fixed in the suite plan before training. No test return selected genomes or settings. Each experiment retains resumable checkpoints locally under `runs/gym-suite/`; these large population-state files are not copied into the examples.','','[Adapter guide](../../docs/GYMNASIUM.md) · [Performance measurements](../../docs/GYM_PERFORMANCE.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(records),1,figsize=(9,max(3,len(records)*2.2)),squeeze=False,layout='constrained')
    for ax,record in zip(axes[:,0],records):
        with (out/record['name']/'history.csv').open() as f:history=list(csv.DictReader(f))
        ax.plot([int(row['generation']) for row in history],[float(row['champion_validation']) for row in history],color='#008b7b',lw=2)
        ax.set_title(record['env'],loc='left',fontsize=11);ax.set_ylabel('Validation return');ax.grid(alpha=.2)
    axes[-1,0].set_xlabel('Generation');fig.savefig(out/'training.png',dpi=150);plt.close(fig)
    print(json.dumps({'completed':len(records),'report':str(out/'README.md')}))

if __name__=='__main__':main()
