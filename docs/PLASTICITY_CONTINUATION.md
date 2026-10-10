# Does ongoing plasticity help animals survive and reproduce?

This experiment branches a saved ecological world. It randomly assigns a fixed
fraction (default half) of the animals alive at that checkpoint to frozen
long-term plasticity; the other original animals are controls in the same world.
The saved source world is never overwritten by the experiment.

The intervention preserves each animal's genome, body, existing learned weights,
spine state, homeostatic AMPA scale and ongoing neural state at assignment. It
then stops all implemented synaptic learning profiles (including eligibility
traces and spine growth) and homeostatic AMPA scaling in treated animals.
Spikes, recurrent signaling, sensory adaptation, intrinsic membrane adaptation,
neuromodulator delivery, synaptic release/recovery, short-term facilitation and
depression, movement, costs and reproduction continue normally. Thus “disabled
plasticity” here means **long-term synaptic plasticity**, not every biological
process sometimes called plasticity. No current bias or behavior is added.

Treatment is an experimental record, not a mutation or sensory input. Children
inherit the normal genome and initial neural state through the existing birth
mechanism, have normal plasticity, and are excluded from the original cohort.
An animal slot cannot transfer treatment to a later occupant.

## Repeated continuations

After building the simulator:

```bash
python3 tools/plasticity_continuation.py \
  --checkpoint runs/live-resumable-2048-seed2.csv.checkpoint.gz \
  --output runs/plasticity-comparison \
  --trials 10 --steps 10000 --seed 1
```

`--steps` in this runner counts **additional world steps** from the saved world.
Each step is 16 neural milliseconds. Pick a horizon covering a substantial part
of the original animals' remaining lifetimes. The maximum ordinary lifespan is
10,000 world steps; older animals naturally have less remaining life. Animals
alive at the end are censored, not considered dead.

Repeat `--checkpoint PATH` to include additional independently evolved worlds.
For each checkpoint, trials use different random assignments with the original
saved world RNG state. They do not reseed animals or reset learned neural state.
This measures the effect of the intervention conditional on the starting world.
Repeated assignments from one checkpoint are not independent evolved-world
replicates. Multiple independently evolved checkpoints give broader evidence.
Assignment is a shuffled exact-count sample, independent of simulation RNG.

Each trial produces a simulation log, ordinary world outputs, a resumable
`continuation.gz`, complete `world.csv.plasticity.csv` cohort table, and
`results.json`. The root `plan.json` records checkpoint and executable hashes;
root `results.json` collects trials and differences between groups.
The default runner is sequential and does not stop existing runs.

## Outcomes and interpretation

For the original cohort, we record:

- Survival at the end of the common follow-up horizon and death cause.
- Remaining life observed, capped at the follow-up horizon (a restricted mean).
- Additional offspring produced after assignment, excluding earlier offspring.
- Baseline age, energy, health, generation and allocated brain sizes.

All cohort animals are recorded, even those omitted from the ordinary sampled
lifespan CSV. Deaths are captured before their slots are reclaimed; surviving
parents' offspring counts are captured after births.

The summary reports frozen minus control differences. Negative values indicate
worse survival or reproduction with frozen plasticity. It also reports a
95% bootstrap interval over **whole trials**, never over individual animals.
Intervals require at least three trials and describe assignment variation for
the supplied worlds; they do not establish generality across independently
evolved worlds. Inspect baseline balance and per-world results too.

Animals compete and interact, so treatment can also affect controls through
food availability or predation. This is a mixed-population intervention effect,
not an isolated-animal effect. A disadvantage would support the usefulness of
ongoing plasticity; it would not by itself prove a specific learned behavior
or memory mechanism. A null result can reflect insensitive conditions or
already adequate inherited/previously learned circuits.

## Single continuation and interruption recovery

```bash
build/neural_selection --resume SOURCE.gz \
  --output runs/trial/world.csv --checkpoint runs/trial/continuation.gz \
  --continuation-steps 10000 --plasticity-ablation \
  --ablation-seed 42 --ablation-fraction 0.5

# Resume its original cohort and treatment assignments:
build/neural_selection --resume runs/trial/continuation.gz \
  --continuation-steps 1000
```

Do not specify `--plasticity-ablation` again on an intervention checkpoint.
Assignments, baseline counters and follow-up records are saved inside the
checkpoint. Ordinary version-1 checkpoints remain readable; checkpoints with
the new intervention flag require this version of the simulator. Cohort CSVs
are regenerated from restored records, so post-checkpoint observations cannot
contaminate a resumed trial. Existing output/checkpoint paths are rejected for
new interventions. Fractions 0 and 1 are allowed for simulator diagnostics;
the comparison runner requires animals in both groups.

The runner refuses to reuse an output directory and stops on failures or an
interrupted trial. To finish an interrupted experiment, resume its branch
checkpoint directly and inspect its regenerated cohort table; the batch runner
does not automatically restart or rerandomize it.

## Verification

```bash
build/plasticity_ablation_protocols
python3 tests/plasticity_continuation.py build/neural_selection
python3 tests/checkpoint_roundtrip.py build/neural_selection
```

GPU checks exercise real learning and release functions, homeostatic scaling,
death recording and slot reuse. Integration checks cover global, legacy,
block and event neural paths, frozen learned state with ongoing firing, a
zero-treatment placebo matching normal continuation, resumable interventions,
source protection, and the repeated-trial runner. These checks validate the
experiment machinery; short test runs are not biological outcome evidence.
