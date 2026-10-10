# Gymnasium performance measurements

Measured on the project's WSL workstation with an RTX 3080 (10 GB), eight CPU workers, Python 3.12, NumPy 2.5.3 and Gymnasium 1.3.0. These are exploratory local timings, not a portable speed guarantee. The ecological simulation remained stopped.

## Where the time went

Thirty CartPole episodes, with timers around observation encoding, native inference, decoding and Gymnasium stepping:

| Brain | Observation encoding | Native call and bridge | Decoding/filtering | Environment | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| Saved ten-neuron champion | 18.9% | 38.8% | 26.8% | 14.3% | 1.2% |
| 74 neurons / 256 connections | 6.2% | 76.8% | 9.1% | 4.8% | 3.1% |
| 266 neurons / 1,024 connections | 2.3% | 91.5% | 3.5% | 1.8% | 1.0% |

The larger brains are random graphs, with short unsuccessful episodes. Their timings are not directly comparable as task-solving throughput. Native-call timing includes ctypes pointer preparation and call overhead.

An instrumented C++ build then split neural execution into clearing inputs, synapse delivery, neuron integration, plasticity and homeostasis. **Neuron integration accounted for 73–78% of the timed internal phases**. Delivery and plasticity followed; clearing and homeostasis were small. Instrumentation itself adds overhead, so these are diagnostic phase shares. The membrane integrator uses four substeps per neural millisecond, repeatedly evaluating nonlinear membrane and, for cortical profiles, dendritic terms. These equations and substep counts were retained.

[Original phase timings](../examples/gym-performance/gym-profile-before.json)

## Small-brain CPU improvement

For a two-action environment, comparing the two motor activities directly avoids several NumPy reductions and allocations. A unique winner has the same action and does not consume random numbers; ties retain the original seeded choice. The output pointer is also cached.

| Ten-neuron champion | Measured brain/environment actions per second |
| --- | ---: |
| Before | 15,326 |
| After | 21,676 |

That is **1.41×** in this instrumented outer-loop measurement, with the same episode returns. Larger brains barely benefited because neural work dominates. Batched observation encoding and decoding were subsequently added to the CUDA path.

[After timings](../examples/gym-performance/gym-profile-after.json)

## CUDA iterations

The first prototype gave each GPU thread a complete scalar brain. It was slower than CPU even for inference: too little parallel work and unfavorable memory access. It was replaced with **one cooperating block per brain**, with parallel release, neurons, plasticity and output collection. Inputs to each neuron are accumulated in the original connection order, avoiding nondeterministic atomic floating-point sums.

With 128 brains, the revised kernel measured approximately **7× faster inference for the ten-neuron champion**, and **30–38× for the larger random graphs**, compared with one CPU core. These timings include transfers, synchronization and the Python batch call; they exclude one-time batch construction. Complete evaluations were substantially less impressive while environments still ran serially in the main process. Small CartPole brains remained faster on eight CPU workers.

The next change grouped ordinary Gymnasium environments into persistent worker processes. This parallelizes environment stepping while neural state remains on the GPU.

### Complete MiniGrid evaluation

`MiniGrid-Empty-Random-6x6-v0`, 64 independently initialized brains, two common episode seeds, 100-step benchmark cap, eight warmed environment workers:

| Backend | Population evaluation time | Executed environment steps |
| --- | ---: | ---: |
| Eight CPU policy/environment workers | 1.081 s | 12,447 |
| Batched CUDA brains + eight environment workers | 0.534 s | 12,447 |

The CUDA route was **2.03× faster overall**, and every episode return matched CPU. The same neural batch had about 32× faster inference than a single CPU core. Environment processing, synchronization, resetting and uploading each generation's genomes limit the total gain.

[Complete benchmark](../examples/gym-performance/gym-benchmark-minigrid-pool.json)

### Numerical limitations

GPU and CPU math libraries differ despite disabled CUDA multiply-add contraction and preserved algorithm/order. The ten-neuron and 74-neuron CartPole comparisons matched episode returns; some 266-neuron recurrent graphs diverged. Both paths execute the same physiological model, but **identical long trajectories across devices are not guaranteed**. Short-run spike parity, masked lanes and complete reset are tested. CUDA-trained champions are validated and tested using the CPU backend.

## Reproduce

See [the adapter guide](GYMNASIUM.md) for build requirements. From the repository root:

```sh
.venv-gym/bin/python -m gym_adapter.profile --episodes 30
.venv-gym/bin/python -m gym_adapter.benchmark --population 128
.venv-gym/bin/python -m gym_adapter.benchmark \
  --env MiniGrid-Empty-Random-6x6-v0 --population 64 --repeats 40
```

Run performance tests without simultaneous evolution or other compute workloads. The checked-in intermediate JSON files preserve the unsuccessful scalar prototype, the cooperating-block version, batched observation handling, and parallel-environment iteration. Current code implements the final iteration, so current results need not reproduce intermediate timings.

## Evolved population optimization (October 2026)

Profiling the corrected searches found that CUDA inference, per-step environment
worker communication, and indented checkpoint serialization dominated runtime.
The evolved genomes averaged 361–379 neurons in MiniGrid, but only 15–17 neurons
had an active causal path to a motor output. Similar redundancy existed in the
classic control genomes.

Policy construction now compiles the motor-relevant subgraph, retaining the full
inherited genome. Reachability includes inhibitory and modulatory connections;
neurons and synapses retain their original indices for stochastic reset streams.
Omitted components have no path to any motor. Their internal states are not
computed or exposed by the Gym adapter. Recurrence, plasticity, local homeostasis,
neural equations, and the neural time resolution are retained. Full graphs are
still validated before compilation. `GYM_FULL_GRAPH=1` disables compilation for
reference comparisons.

CUDA evaluation runs each individual's independent episodes concurrently rather
than sequentially. Each episode still receives its own training seed and reset.
The default cooperating block now has 32 threads, with strided processing for
arbitrary graph sizes; `GYM_CUDA_THREADS=128` selects the previous configuration.
For the two supported MiniGrid tasks, a compiled partial-view builder produces
identical symbolic observations without allocating and rotating temporary grids.
Environment actions, rewards, termination, reset RNG, and world dynamics remain
in Gymnasium. `GYM_REFERENCE_OBSERVATION=1` selects the original observation
builder. Other MiniGrid tasks use their original observation builder.

Checkpoints use compact JSON and synchronous serialization of genome lists,
avoiding recursive dataclass copies. They are still saved atomically every
generation, with the same information.

Measured on generation-4,000 populations, four complete generations including
worker startup, validation, breeding, and checkpointing:

| Environment | Reference | Optimized | Throughput gain |
| --- | ---: | ---: | ---: |
| MiniGrid Memory | 8.54 s | 3.48 s | 2.46× |
| Acrobot | 8.19 s | 3.58 s | 2.29× |

Both comparisons produced identical next populations, training winners, validation
scores, genetic RNG states, and training seed cursors. Sixteen tests passed,
including full/compiled motor trace parity with plasticity and modulation,
independent episode batching, reference MiniGrid observation equality, resume,
and validation noninterference. These gains depend on evolved graph sparsity;
densely connected brains may benefit much less. This does not establish a 10×
speedup.

Results and the checkpoint migration audit are saved under
`runs/gym-speedup-verification/` and
`runs/gym-suite-corrected/performance-migration.json`. The migration changes only
the recorded source hash; all five original generation-4,000 checkpoints have
separate backups.
