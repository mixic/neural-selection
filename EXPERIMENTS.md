# Experimental checks of the neural simulation

These are comparisons with published experiments on real cells, not just with other simulations. `tests/physiology.cpp` applies each stated protocol to the same neuron and synapse functions used by the GPU world. The full paper links below lead to the researchers' original reports.

## 1. Response to injected current

**Experiment.** [Naud, Marcille, Clopath and Gerstner, 2008, *Biological Cybernetics*](https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Naud08.pdf), Figure 8 and Table 1, recorded cortical cells under step-current injection and fitted the adaptive exponential integrate-and-fire (AdEx) model to individual cells. Figure 8 includes a regular-spiking pyramidal cell (RS, 130 pA) and a continuously non-adapting inhibitory cell (cNA, 150 pA).

**Simulation.** Table 1's nine fitted cell parameters are used directly for each profile. At the currents stated in Figure 8's experimental caption, a 2-second step produced 32 spikes in RS (130 pA), with first and last interspike intervals of 17 and 64 ms; cNA (150 pA) produced 73 spikes, with intervals of 23 and 27 ms. Table 1 lists different fitted *model* currents, 98 and 184 pA, at which our model emitted 22 and 100 spikes respectively. RS remained silent at zero injected current. Four 0.25 ms integration substeps per neural tick improved cNA spike count convergence at 150 pA: a one-substep version gave 69 spikes, four substeps 73, and a 0.1 ms numerical reference 74.

**Interpretation.** The qualitative adapting/non-adapting firing patterns are reproduced. This is a check of an implementation against the original paper's *fitted model and recorded pattern*, not an independent fit to raw experimental traces. The paper does not supply the underlying voltage samples here, so exact spike-time error against the recorded neurons has not been measured. Action-potential shape is intentionally replaced by a reset.

## 2. Paired-pulse synaptic depression

**Experiment.** [Tsodyks and Markram, 1997, *PNAS*](https://pmc.ncbi.nlm.nih.gov/articles/PMC19580/) used patch-clamp recordings from connected neocortical pyramidal neurons and fitted a synaptic-resource model. One reported fit uses utilization `U = 0.55` and recovery time `450 ms`.

**Simulation.** With those fitted values, a second presynaptic spike 50 ms after the first released 0.279 of unit synaptic strength, versus 0.550 on the first (ratio 0.508). After a further 2 seconds, the response recovered to 0.545. The simulation gives each excitatory synapse its own resource state.

**Interpretation.** The depression and recovery direction follows the paired-recording result. These numerical outputs are predictions of the fitted model; they have not been independently compared against digitized EPSP amplitudes from the experiment. Release is deterministic and does not reproduce trial-to-trial failures or vesicle noise.

## 3. Spike-timing-dependent change

**Experiment.** [Bi and Poo, 1998, *Journal of Neuroscience*](https://www.gatsby.ucl.ac.uk/~pel/course_wuhan/papers/bi.poo_1998.pdf) made simultaneous recordings from connected cultured rat hippocampal neurons. Their 60 pairings at 1 Hz showed potentiation when a postsynaptic spike followed excitatory input within about 20 ms, depression for the reverse order, little change outside that window, and no significant change at glutamatergic inputs to GABAergic postsynaptic cells. Their reported mean potentiation in one comparison was 48.4% ± 9.9% SEM and depression was about 18%.

**Simulation.** Sixty +10 ms pairings raised a normalized weight from 1 to 1.415; sixty −10 ms pairings lowered it to 0.836. ±50 ms pairings and +10 ms pairing onto an inhibitory target left it at 1. The timing rule uses *presynaptic arrival* at the synapse rather than presynaptic spike emission.

**Interpretation.** The timing window and sign agree with the experiment, while the magnitudes were **calibrated** to its reported group means. They are therefore not independent validation. The original study also found dependence on initial synaptic strength and NMDA receptors; the model approximates strength dependence but has no explicit receptor or calcium dynamics. Its behaviour near zero time difference, for complex spike trains, and across other species or synapse classes is untested. This local rule should not be read as a universal biological learning rule.

## 4. Inhibitory current

Paired recordings report an approximately −75 mV inhibitory reversal potential ([Tarczy-Hornoch et al., 1998](https://pmc.ncbi.nlm.nih.gov/articles/PMC2230896/)); recordings of fast GABA-A currents support a decay on the order of 10 ms ([Hardie and Pearce, 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674351/)). The protocol check adds 4 nS of inhibitory conductance to a resting cNA cell, verifies hyperpolarization, then verifies that conductance falls to between 1 and 2 nS after 10 ms. These are limited checks of sign and timescale, not a recreation of either full recording protocol.

## Changes driven by these comparisons

The initial simulator used a coarse 5 ms integrate-and-fire tick, ad hoc activation/reset parameters, and globally released chemicals that directly changed eligible synapses. It now uses published AdEx cell fits, 1 ms spike timing with converged 0.25 ms voltage integration, conductance-based input, short-term release depletion, and a source-specific timing rule. The global chemical rule was removed because its particular effects and constants lacked a matching experimental protocol. No direct reward signal enters neural plasticity.

The environment's sensor-current scale, food chemistry, motor mapping, mutation ranges, and reproduction rules are artificial-world design choices, not claims about animal physiology. The current evidence supports the possibility of local activity-dependent synaptic change in this model. It does **not** establish that evolving animals will develop useful memory, adaptive learning, instincts, or ant-like behaviour.

## 5. Thin-dendrite summation

**Experiment.** [Polsky, Mel and Schiller, 2004, *Nature Neuroscience*](https://neurophysics.ucsd.edu/courses/physics_171/49528.pdf) used confocal imaging, focal stimulation at two identified sites, and somatic recordings from rat pyramidal neurons. Weak nearby inputs on one branch summed nearly linearly, intermediate inputs could sum superlinearly, strong inputs saturated, and inputs on different branches summed approximately linearly.

**Simulation.** Excitatory pyramidal cells now have four independently addressed dendritic compartments. A phenomenological sigmoid transforms each branch's decaying conductance before the branches combine at the soma. At weak two-site input, same-branch and separate-branch peak responses were 1.446 and 1.445 mV; at intermediate input, 8.695 and 6.943 mV; at strong input, 22.324 and 25.792 mV. The GPU genome assigns each excitatory synapse a mutable branch.

**Interpretation.** This reproduces the qualitative within- versus between-branch summation pattern. Branch count, conductance threshold, gain, and saturation were chosen for the protocol; they are not experimentally identified channel parameters. No explicit dendritic membrane voltage, calcium signal, NMDA receptor kinetics, or spatial distance within a branch is simulated. The original study observed failures and variability that this stage does not reproduce.
