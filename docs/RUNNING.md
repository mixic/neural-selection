# Running and observing Neural Selection

Requires an NVIDIA GPU, CUDA Toolkit with `nvcc`, CMake, a C++ compiler, and libpng and zlib development headers.

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/neural_selection --steps 2000 --population 512 --seed 1 --food-renewal 0.0006 --output run.csv
```

For a live view, add `--visualize` and open the local address printed by the program (by default, [http://127.0.0.1:8765/](http://127.0.0.1:8765/)):

```sh
./build/neural_selection --population 512 --seed 1 --visualize --output run.csv
```

Live runs have no step limit unless you set `--steps N`. `--steps 0` also selects an unlimited run without the browser. The browser view shows the 2D food map and moving, oriented animals, with pulses for births and deaths and orange flashes at exact bite locations, live population and event counts, global animal and brain-slot usage, a population chart, and details for a clicked animal. Drag to pan, scroll to zoom, and use the controls to pause, resume, change speed, or end the run. At 1× the program targets 20 world steps per second; 4× targets 80; Max runs at GPU speed. The page updates every two world steps. End run writes the final image and brain tables and leaves the page open for inspection; Ctrl+C saves and exits. Use `--port N` if port 8765 is occupied. The server listens only on `127.0.0.1`.

Selecting an animal opens a live brain view below the world. Neurons are arranged by sensory input, internal network, and motor output, with colors for inhibitory transmission and neuromodulators. Zoom, pan, expand, and hover to inspect voltage, sampled firing rate, and input/output counts. The view uses the actual evolved graph, including recurrent and self connections; axon trees terminate at boutons facing spines or inhibitory shaft contacts on the receiving dendritic tree, grouped by the actual simulated dendrite index. Each terminal maps to a real synapse; positions and anatomical shapes are schematic. Hovering a terminal reveals its endpoints, target dendrite, delay, and current weight. Flashes come from changes in recorded spike counters. Travelling pulses follow those axons and stop at their boutons. They are slowed visual cues for presynaptic firing, rather than exact timing or proof of synaptic release. Only the selected brain is downloaded, and observation never changes neural state.

Live runs now save an atomic, compressed checkpoint at step 0, every 10,000 world steps, and on a normal stop. It defaults to `<output>.checkpoint.gz`. To continue after a crash, run `./build/neural_selection --resume run.csv.checkpoint.gz --visualize`. The saved world includes living genomes, learned synaptic weights, neuron voltages and other dynamic state, resource and body state, random generators, and world history counters. Resume restores the saved simulation settings and returns to the last completed checkpoint; steps after it are lost. When continuing into the same output files, history rows beyond that step are removed before new rows are appended. Starting a new run refuses to overwrite an existing checkpoint. `--checkpoint-every N` changes the interval; `--checkpoint file.gz` enables checkpointing for a run without the live view or changes the destination. A checkpoint only loads into a build with a matching state layout and compatible simulation rules.

Use `--profile` for per-stage timing and sampled delivery, neuron, and plasticity kernel times. It synchronizes the GPU to measure stages, so its throughput includes profiling overhead. `--legacy-neural` retains the original separate delivery and plasticity launches for comparisons; the default pipelines plasticity from one tick with delivery for the next while preserving their order. The modulatory event timestamps use alternating tick slots so a new delivery cannot overwrite the previous tick's event during the combined pass.

`--block-brains` selects an experimental GPU path that advances each eligible animal's brain for all 16 neural ticks in one CUDA block, keeping changing state in shared memory. It uses a small-block tier through 80 allocated neurons and 192 synapses, a medium-block tier through 128 neurons and 256 synapses, and the existing global-slot path above either medium limit. These limits only select execution paths; they do not constrain evolution. On the RTX 3080 with 2,048 founders, the original small tier reduced neural work from about 14.1 to 5.8 ms per world step. In a 112-neuron comparison, the medium tier reduced neural work from 17.8 to 9.6 ms. See [EXPERIMENTS.md](../EXPERIMENTS.md) for validation and limits.

`--event-synapses` is a further experimental mode on the block path. It queues spike arrivals and advances plasticity state across quiet intervals without applying its update equations at every neural tick. Neurons still integrate at 1 ms network intervals, so this is a partial event-driven solver. In a matched 2,048-founder benchmark it reduced the neural stage from 6.21 to 5.56 ms per step. It is opt-in while longer-term behavioral equivalence remains unverified.

Architecture `86` targets an RTX 3080; choose the architecture supported by your GPU, or use `-DCMAKE_CUDA_ARCHITECTURES=native` when configuring on the machine that will run the simulation.

Each world step advances the neural simulation by 16 one-millisecond ticks. The program writes population history to the requested CSV, an image to `<output>.png`, and final genomes and synaptic weights to `<output>.neurons.csv` and `<output>.synapses.csv`. The brain tables include allocated inactive genes, global neuron/synapse slot IDs, fixed port numbers (`-1` means hidden), and heritable sensory-channel assignments. White pixels in the image are animals; green, orange, and blue represent the three food types. Runs with the same seed may differ because animals contend for resources and birth slots through GPU atomics.

## Observing behaviour

The simulation also writes cumulative food-choice and movement measurements to `<output>.behavior.csv` every 100 world steps, plus a 10% deterministic sample of completed lifetimes to `<output>.lives.csv`. The live view shows recent hungry food-choice rates. Summarize an active or finished run with:

```sh
python3 tools/analyze_behavior.py run.csv
```

An *opportunity* means an animal starts a step on a stocked food cell; *accepted* means its eat output fires and it successfully harvests that cell. Empty-cell eat attempts are recorded separately to reveal an indiscriminate eat drive. Counts are split by food A/B/C, energy below or at least 16 at the start of the step, and animal age 0–99, 100–299, or 300+ steps. Hungry approach alignment compares movement toward the nearest stocked cell of each type within four cells against distance traveled; +1 is straight toward it, −1 straight away. These observations are never provided to the brain as rewards or fitness signals. A and B have distant directional senses. All three foods have distinct contact chemosensory channels while underfoot, before ingestion; C has no distant cue, so its approach score is a reference. An animal moving onto food senses its identity before it can eat there on the following step. Consumption still produces distinct taste and body-state signals. High acceptance alone is not evidence of selective feeding; compare it with empty-cell attempts and food-specific acceptance.

The history CSV also counts the first 100 world steps of the founding animals: `founder_observed_steps` is the number of living founder animal-steps, and `founder_forward_steps` and the other `founder_*_steps` columns count steps with at least one spike from each motor. The evolving world has no inherited constant-current bias; motor activity must arise from sensory input, synaptic transmission, or intrinsic membrane conductances. Fixed motor port meanings remain unchanged. Earlier runs with inherited motor current are not directly comparable to bias-free runs.

Forward and turning motor spikes drive decaying body-side muscle states with a 200 ms decay time, approximating measured insect body-muscle relaxation. Eat and bite require a spike in the current 16 ms step; they do not inherit that muscle persistence. Muscle states reset at birth and never feed current into a neuron. `--instant-motors` restores the earlier one-step movement rule for comparisons. `--silence-senses` sets external sensory levels to zero for a diagnostic run; olfactory receptor neurons retain their measured spontaneous baseline. `--silence-contact` disables only the three pre-ingestion food contact channels. These flags do not change the genome or add a neural bias. The first-100-step founder counters still count motor spikes, not muscle activity.

The lifetime sample records age at death, cause (1 predation, 2 energy depletion, 3 other health loss, 4 age limit), offspring count, brain size, modulatory and plastic synapse counts, synaptic weight change, and the same food-choice counters. It supports comparisons within the same animal and across brain types. Such associations can suggest adaptation, but age, survival selection, and genotype differences can mimic learning; they do not by themselves prove it.

With inherited constant current first removed, the earlier `--steps 5000 --population 2048 --seed 1` configuration produced no motor spikes or births. All founders died by step 3,800. In a matching 2,000-step trial, all 12,288 active food receptor neurons fired, and 22,563 active hidden neurons fired, but none of the 10,240 motor neurons did. This was a sensory-to-motor propagation gap in that earlier founder configuration; later physiology, genome and muscle changes are described in [EXPERIMENTS.md](../EXPERIMENTS.md).

To inspect one founding animal's motor inputs, add `--trace-animal N` with a founder slot number from `0` through `population - 1`. The run writes `<output>.trace.csv` with excitatory and inhibitory synapse arrivals, released conductance, maximum motor voltage, and motor spikes. Choose a run that ends before that founder dies, since slots can be reused after death.

## Ecology

There are no fixed predator and prey species. Every animal can detect other animals in three directions and has a separate neural bite output. A bite costs energy, injures a co-located animal, and gives that animal a pain signal. A lethal bite transfers 45% of the prey's remaining energy to the attacker and activates a meat-taste signal. Predation therefore transfers existing energy rather than adding a reward signal. Whether animals hunt or avoid others depends on their evolving circuits.

Each active neuron costs 0.00002 energy per world step, one tenth of the previous rate, in addition to a 0.003 body upkeep cost and the costs of movement and biting. These are ecological calibration parameters, not fitted neuronal metabolism. At 45 active neurons, idle upkeep is 0.0039 energy per step; larger brains still cost more, but the selective pressure to shrink them is weaker.

Food renews independently in each empty cell with probability 0.0006 per world step by default; `--food-renewal` changes that probability. The history CSV includes population, births, total deaths, predation deaths, bites, and stocked food cells. With the shared-pool brain layout, a 10,000-step run before deletion mutations and the latest upkeep reduction started at 512 animals, ended with 413, and never reached the 2,048-slot safety ceiling. This is an earlier empirical calibration, not a guaranteed equilibrium for the current settings.

