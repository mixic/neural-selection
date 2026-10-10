# Evolving brains in Gymnasium

The Gymnasium adapter uses the project's **existing C++ neuron and synapse equations** as an episode policy. A genetic algorithm selects genomes by their mean environment return. There is no backpropagation, and reward is never passed into a sensory channel or neural update.

The default backend evaluates brains on CPU workers. An optional CUDA backend batches brains, retains neural state on the GPU, and steps ordinary Gymnasium environments in persistent groups of CPU workers. It does not use the CUDA world's animals, food, ecology, pool allocator, or checkpoint format. The saved ecological world can remain stopped while these experiments run.

## Setup

From the repository root, with Python 3.10+ and `g++`:

```sh
python3 -m venv .venv-gym
.venv-gym/bin/python -m pip install -r gym_adapter/requirements.txt
```

The adapter compiles `gym_adapter/native.cpp` into `build/libgym_brain.so` on its first use. The CPU backend needs no CUDA Toolkit. `--backend cuda` additionally requires `nvcc` and an NVIDIA GPU; the adapter builds `build/libgym_brain_gpu.so`. Set `NVCC` to a compiler path if necessary, and optionally `GYM_CUDA_ARCH` to a CUDA architecture (default `native`). It includes `src/physiology.cuh` directly, rather than translating the physiological equations into Python. The bridge supports the seven existing membrane profiles, excitation, inhibition, dendritic branch assignment, delays, stochastic release, all five plasticity phenotypes, local modulatory events, and the existing slow AMPA homeostasis.

## First environment: CartPole

```sh
.venv-gym/bin/python -m gym_adapter.evolve \
  --env CartPole-v1 --population 128 --generations 300 \
  --episodes 3 --validation-episodes 12 --test-episodes 100 \
  --hidden 0 --connections 24 --workers 8 \
  --obs-scale 2.4,1,0.05,0.25 \
  --target-return 475 --test-seed 2000000 --output runs/gym-cartpole
```

The scale values specify sensory units for cart position, cart velocity, pole angle, and angular velocity. They increase sensitivity to small pole movements; they do not supply an action or balancing rule. Founders are randomly connected, and can acquire hidden neurons through mutation even when `--hidden 0` is used.

CartPole's actions and episode return are defined by the [official Gymnasium environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/). This command uses the standard reward and a maximum of 500 environment steps per episode.

## Observations and actions

Flattenable observations (`Box`, `Discrete`, fixed `Dict`/`Tuple`, etc.) become signed opponent sensory channels. For each observation dimension, the adapter normalizes it, applies `tanh`, and provides `(1 + value) / 2` and `(1 - value) / 2` to two sensory neurons. **A neutral observation drives both channels equally.** This is the explicit observation encoding, not a hidden or motor current bias.

Finite observation bounds determine default centering and scale; unbounded dimensions default to center zero and scale one. `--obs-scale` supplies explicit positive scales and uses center zero. Sensory neurons receive the same 230 pA maximum direct transduction used for several senses in the world simulation. ORN-profile genes instead use the existing receptor-event model. Hidden and motor cells receive no constant applied current.

Motor outputs are filtered spike counts:

- `Discrete`: the output with greatest filtered activity wins; exact ties, including all-silent outputs, are broken by an episode-seeded random choice. Nonzero `Discrete.start` is respected.
- Finite `Box`: two motor cells represent each action component. Their normalized activity difference is mapped into the action bounds. Silence maps to the midpoint.

The action decoder is fixed and contains no environment-specific controller. Neither decoding nor sensory normalization evolves. The bridge supports up to 32,767 flattened observation dimensions using a separate, wider sensory-channel mapping. It does not alter the ecological simulator's neuronal storage. Very large images would create impractically many sensory cells without a feature encoder. Other action-space types are rejected.

## Use the adapter directly

```python
import gymnasium as gym
import numpy as np
from gym_adapter import SpaceAdapter, SpikingPolicy, random_genome

env = gym.make("Pendulum-v1")
adapter = SpaceAdapter(env.observation_space, env.action_space)
genome = random_genome(adapter.channels, adapter.outputs,
                       np.random.default_rng(1))
with SpikingPolicy(genome, adapter) as policy:
    observation, info = env.reset(seed=1)
    policy.reset(seed=1)
    for _ in range(200):
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
env.close()
```

This example exercises continuous actions with a random brain; it does not claim to solve Pendulum. The trainer accepts `--env` for other registered Gymnasium environments supported by the adapter's space types.

## MiniGrid and the environment suite

```sh
.venv-gym/bin/python -m gym_adapter.evolve \
  --env MiniGrid-Empty-Random-6x6-v0 --backend cuda \
  --population 64 --generations 120 --episodes 4 \
  --validation-episodes 12 --test-episodes 100 --test-seed 4000000 \
  --hidden 8 --connections 96 --max-steps 144 \
  --seed 21 --workers 8 --output runs/minigrid-navigation
```

MiniGrid uses Farama's standard `ImgObsWrapper`: the full **7×7 partially observed symbolic image**, with three integer fields per tile (object, color, state). These are symbolic labels, not RGB pixels. The wrapper removes textual missions and compass direction. This is intended for tasks with a fixed objective, such as Empty, DoorKey or Memory; variable-language tasks need a language encoder. The adapter uses the fields' actual label bounds (10, 5, 2), instead of the generic image byte bound of 255. Labels currently receive an ordinal opponent encoding; categorical one-hot encoding may be a better future representation.

There are 294 sensory cells and seven action outputs before any hidden cells. No map, global position, privileged goal information, reward shaping, or action mask is added. Empty-Random changes starting position and orientation with the episode seed. [MiniGrid's official wrappers](https://minigrid.farama.org/_modules/minigrid/wrappers/) describe these observations; the [Memory task](https://minigrid.farama.org/environments/minigrid/MemoryEnv/) requires remembering the starting cue to choose the matching object at the end of the corridor.

Run the exploratory suite (navigation, Pendulum, Acrobot, MountainCar, Memory):

```sh
.venv-gym/bin/python -m gym_adapter.suite --output runs/gym-suite --workers 8
```

Extend an existing suite to a larger total generation target, rotating environments in chunks:

```sh
.venv-gym/bin/python -m gym_adapter.suite \
  --output runs/gym-suite --generations 5000 --generation-chunk 250
```

Existing populations, champions and genetic RNG state are resumed. Completed earlier results do not block an extension. Interim chunks save a checkpoint without re-running final test episodes; final tests run when an environment reaches the total target. The same original test seeds are reused for direct comparison, so they are no longer an untouched test set for the extended search.

The suite saves its plan, exact commands, logs, genomes, checkpoints and final test results. Every decision advances 16 neural milliseconds by default; this is a configurable policy integration interval and need not equal the environment's physical time step. It uses bounded budgets; a failed or flat search is retained as a result. Re-running the suite skips completed experiments and resumes compatible incomplete checkpoints.

## CPU and CUDA performance

Use `gym_adapter.profile` for observation/native/decoder/environment timing and instrumented C++ phase timing; use `gym_adapter.benchmark` for inference and whole-population comparisons:

```sh
.venv-gym/bin/python -m gym_adapter.profile --episodes 30
.venv-gym/bin/python -m gym_adapter.benchmark --population 128
.venv-gym/bin/python -m gym_adapter.benchmark \
  --env MiniGrid-Empty-Random-6x6-v0 --population 64 --repeats 40
```

The instrumented C++ library is a separate build and is never used by the trainer. Timing instrumentation has overhead; the normal build supplies total phase shares and the instrumented build identifies costs within neural execution. CUDA timings include observation/output transfer and synchronization. End-to-end comparisons use warmed worker pools and include GPU genome upload/setup.

The CUDA kernel assigns one block per brain and parallelizes neurons and synapses. Incoming edges are reduced in original graph order, with barriers between release, neuron integration and plasticity. Brain sizes can differ within a batch; there is no per-brain capacity cutoff. All requested neural ticks run in one kernel, finished episodes are masked, and episode resets replace neural and decoder state.

CPU and GPU use the same equations, integration interval, stochastic RNG algorithms, delays and plasticity. Floating-point math libraries differ: exact short-run spike parity is tested, but long recurrent trajectories can diverge. The benchmark reports whether complete returns match. Training/validation seeds remain disjoint from final test seeds, and **validation and final evaluation use CPU**, including for CUDA-trained populations.

See [the measured performance report](GYM_PERFORMANCE.md). CPU remains useful for tiny brains and short episodes; a fast inference kernel alone does not guarantee faster complete evolution.

## Genome and episode state

Genes specify membrane phenotype, sensory assignment, enabled state, transmitter sign, modulator identity, capacitance and leak multipliers, threshold, and receptor parameters. Synapses specify endpoints, initial conductance, enabled state, delay, dendrite index, release utilization, recovery, facilitation, plasticity phenotype and modulator receptors.

Mutation changes membrane parameters and synaptic weights, rewires connections, changes transmission/modulation and plasticity, and can add, disable or remove hidden cells and connections. Fixed sensory and motor ports remain intact. Arbitrary recurrence, self-connections and parallel connections are permitted. The JSON genome format contains no learned synaptic state.

Every episode resets membrane state, adaptation, synaptic state, spike history, release RNGs and decoder activity. Plasticity may act during an episode, but its resulting weights are not inherited or carried into the next episode. Offspring receive mutated initial genes, not a parent's episode state.

## Selection and evaluation

**Selection uses fresh training episodes only.** Every candidate's training episode receives its own previously unused seed. A deterministic permutation of an episode counter supplies unique seeds in the high half of the 32-bit range; the counter is checkpointed, so resumed chunks do not repeat training seeds. Validation and test seeds use the low half.

Tournament selection and elitism use only the current generation's training returns. Offspring are chosen before validation is evaluated. The current training winner is saved as `champion.json` and measured on a fixed validation set for reporting. Validation does not select parents, retain an archive, determine elitism, or stop the search. `--target-return`, if supplied, refers to training return only. The saved final policy is the last generation's training winner; validation is not used to choose a preferred generation.

The previous selection protocol incorrectly reinserted a validation-selected champion into breeding. Its searches were cancelled, and their results are historical, contaminated evidence. Corrected training refuses to resume those checkpoints. New checkpoints record `selection_protocol: fresh-training-only-v2` and hash the selection implementation as well as the neural model.

Choose an untouched final evaluation range with `--test-seed` when comparing settings across experiments. After evolution, the champion and the best first-generation candidate are evaluated on fresh held-out seeds, alongside a random-action baseline. The test episodes do not select parents or champions. A higher return shows that evolution found a better episode policy; it does **not** establish learning within an individual lifetime.

The physiological core is a reduced model, and evolution can select parameter combinations and firing rates outside the original fitted experimental regimes. Task performance alone does not validate their biological fidelity.

## Saved files and replay

Each experiment directory contains:

| File | Contents |
| --- | --- |
| `history.csv` | Training and validation returns, graph sizes, motor firing and elapsed time |
| `initial.json` | Best first-generation genome by training return |
| `champion.json` | Best validated genome |
| `evolution.json` | Population, champion, genetic RNG and next generation for resuming |
| `results.json` | Settings and initial/champion/random returns on fresh test seeds |

Use `--resume path/to/evolution.json` with the same experiment settings and a larger `--generations` value to continue evolution. This format is separate from the CUDA world's checkpoints.

Replay the champion:

```sh
.venv-gym/bin/python -m gym_adapter.replay \
  runs/gym-cartpole/champion.json \
  --results runs/gym-cartpole/results.json --seed 123
```

To record a GIF, install Pillow and add `--gif runs/gym-cartpole/rollout.gif`. Rendering also requires the chosen environment's render dependencies.

## Export a suite report

Install optional `Pillow` and `matplotlib`, then run:

```sh
.venv-gym/bin/python -m gym_adapter.report_suite
```

This copies champions and complete test results into `examples/gym-suite`, renders median-return held-out episodes as GIFs, and plots the validation curves. `gym_adapter.diagnose` can run post-training sensory ablations on a separate seed set; these diagnostics do not select genomes.

## Adapter checks

```sh
.venv-gym/bin/python -m pip install pytest
.venv-gym/bin/python -m pytest tests/test_gym_adapter.py tests/test_gym_gpu.py -q
```

These check CPU/CUDA spike parity, masked/reset GPU lanes, full MiniGrid sensory mapping, deterministic genetic-search resumption, complete episode resets, genome-only serialization, add/remove mutations, valid recurrent graphs, generic observation spaces, bounded actions and silent unconnected motors.
