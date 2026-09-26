# Neural Selection

A CUDA C++ simulation of animals with evolved spiking brains in a shared 2D world. Animals sense food, nearby animals, hunger, fullness, taste, and injury; move, eat, and bite; compete for resources; and reproduce with mutations to their neural genomes. The world supplies no reward or fitness value to the brain. Survival and reproduction follow from body state and resource use.

## Build and run

Requires an NVIDIA GPU, CUDA Toolkit with `nvcc`, CMake, a C++ compiler, and libpng development headers.

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/neural_selection --steps 2000 --population 512 --seed 1 --food-renewal 0.0006 --output run.csv
```

Architecture `86` targets an RTX 3080; change it for another GPU. On this machine, a temporary CUDA 12.9 toolkit under `/tmp/cuda129` was used to build the existing executable because `nvcc` was not installed system-wide. If that toolkit is gone, rebuilding needs another CUDA Toolkit.

Each world step advances the neural simulation by 16 one-millisecond ticks. The program writes population history to the requested CSV, an image to `<output>.png`, and final genomes and synaptic weights to `<output>.neurons.csv` and `<output>.synapses.csv`. White pixels in the image are animals; green, orange, and blue represent the three food types. Runs with the same seed may differ because animals contend for resources and birth slots through GPU atomics.

## Ecology

There are no fixed predator and prey species. Every animal can detect other animals in three directions and has a separate neural bite output. A bite costs energy, injures a co-located animal, and gives that animal a pain signal. A lethal bite transfers 45% of the prey's remaining energy to the attacker and activates a meat-taste signal. Predation therefore transfers existing energy rather than adding a reward signal. Whether animals hunt or avoid others depends on their evolving circuits.

Each active neuron costs 0.0002 energy per world step, in addition to a 0.003 body upkeep cost and the costs of movement and biting. These are ecological calibration parameters, not fitted neuronal metabolism. At roughly 45 active neurons, total idle upkeep is about the previous 0.012 energy per step, while larger brains now need to earn their extra energy use.

Food renews independently in each empty cell with probability 0.0006 per world step by default; `--food-renewal` changes that probability. The history CSV includes population, births, total deaths, predation deaths, bites, and stocked food cells. With the current bite and per-neuron upkeep parameters, a 10,000-step run starting at 512 animals ended with 339 animals and never reached the 2,048-slot safety ceiling. This is an empirical calibration, not a guaranteed equilibrium for every seed or evolved population.

## Neural model

- Each active neuron has a membrane voltage, adaptation current, and excitatory and inhibitory conductances. It emits discrete spikes. The adaptive exponential integrate-and-fire equations and two parameter sets come from fits to recorded cortical neurons in [Naud et al. (2008)](https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Naud08.pdf). The default pyramidal profile also has an inherited, mutable spike aftercurrent fitted to cortical afterpotentials reported by [Haj-Dahmane and Andrade (1997)](https://journals.physiology.org/doi/10.1152/jn.1997.78.4.1983) and the 40 Hz residual voltage in [Sjöström et al. (2001)](https://plasticity.muhc.mcgill.ca/DataPage/PDFs/Sjostrom_Neuron_2001.pdf). Four 0.25 ms integration substeps are used per neural tick.
- Excitatory pyramidal cells route synapses to four mutable dendritic branches. Each branch has a fitted nonlinear input/output curve following the two-site summation pattern measured by [Polsky et al. (2004)](https://neurophysics.ucsd.edu/courses/physics_171/49528.pdf).
- Excitatory synapses have resource depletion, recovery, inherited facilitation parameters, and a mutable number of stochastic release sites, following paired and multi-cell recordings in [Tsodyks and Markram (1997)](https://www.cns.nyu.edu/csh/csh04/Articles/Tsodyks-Markram-97.pdf) and [Markram et al. (1998)](https://web.stanford.edu/class/cs379c/archive/2016/class_messages_listing/articles/MarkrametalPNAS-98.pdf). Inhibitory synapses add a decaying inhibitory conductance.
- Excitatory-to-excitatory synapses have source-specific plasticity profiles. One approximates the cultured hippocampal spike-timing protocol in [Bi and Poo (1998)](https://www.gatsby.ucl.ac.uk/~pel/course_wuhan/papers/bi.poo_1998.pdf); the default cortical layer-5 profile also requires preceding depolarization, as measured by [Sjöström et al. (2001)](https://plasticity.muhc.mcgill.ca/DataPage/PDFs/Sjostrom_Neuron_2001.pdf). NMDA blockade can be applied in protocol checks. Neither profile is a universal law of learning.
- A separate synaptic phenotype can retain short-lived LTP and LTD eligibility traces, converted by locally delivered noradrenaline- or serotonin-like events through matching receptors. This is fitted to the two-pathway recordings of [He et al. (2015)](https://escholarship.org/content/qt290403p6/qt290403p6_noSplash_15961229581c4dab94718b9699f4df0e.pdf). Hidden neurons may inherit and mutate a modulatory transmitter identity; synapses may inherit and mutate receptors and plasticity phenotype. No cortical anatomy is imposed on evolving brains.
- Another mutable synaptic phenotype has delayed dopamine sensitivity and competing depression, fitted to the cell-specific timing and blocker controls in [Shindou et al. (2019)](https://onlinelibrary.wiley.com/doi/10.1111/ejn.13921). This adds no required striatal layout; its eligibility time course is phenomenological.
- An optional synapse phenotype tracks local dopamine-dependent spine enlargement with the timing and inhibitor controls of [Yagishita et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25258080/). This structural state is exported for inspection but does not yet change synaptic transmission or animal behaviour.
- Excitatory pyramidal cells also have a slow postsynaptic AMPA scale shared by their excitatory inputs. Its 48-hour endpoint response is calibrated to [Turrigiano et al. (1998)](https://faculty.washington.edu/somurray/psych545/Turrigiano_Nature_1998.pdf). It changes negligibly in the short world runs shown here.
- The heritable genome stores initial neuron parameters, cell types, topology, connection strength, delays, and synaptic release parameters. Mutation can add or silence hidden neurons and synapses and alter their parameters. Learning changes current synaptic weights; newborns start from inherited initial strengths.

The body has fixed sensory and motor ports. Brain topology is otherwise arbitrary within 69 neuron and 384 synapse slots per animal. Sixteen sensory and five motor ports leave 48 hidden neuron slots; up to 2048 animals fit in the current storage budget. These fixed per-animal capacities are implementation limits, not imposed brain structures.

## Evidence and limits

[EXPERIMENTS.md](EXPERIMENTS.md) lists the original experimental protocols, our results, what was fitted versus checked, and remaining gaps. The protocol checks run with `./build/physiology_protocols` or `ctest`.

These checks establish a few local response patterns, not a validated biological brain. In particular, the dendritic branches are phenomenological rather than explicit membranes, and the model omits detailed ion channels, variable quantum size, synapse-specific molecular machinery, development, and evidence-based sensory transduction. The neuron fits come from mammalian cortex; plasticity checks draw on cortical, cultured hippocampal, and striatal experiments. None validates an insect brain or complex learned behaviour in the evolving world.
