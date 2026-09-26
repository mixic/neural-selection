# Experimental checks of the neural simulation

These are comparisons with published experiments on real cells, not just with other simulations. `tests/physiology.cpp` applies each stated protocol to the same neuron and synapse functions used by the GPU world. The full paper links below lead to the researchers' original reports.

## 1. Response to injected current

**Experiment.** [Naud, Marcille, Clopath and Gerstner, 2008, *Biological Cybernetics*](https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Naud08.pdf), Figure 8 and Table 1, recorded cortical cells under step-current injection and fitted the adaptive exponential integrate-and-fire (AdEx) model to individual cells. Figure 8 includes a regular-spiking pyramidal cell (RS, 130 pA) and a continuously non-adapting inhibitory cell (cNA, 150 pA).

**Simulation.** Table 1's nine fitted cell parameters are used directly for each profile. At the currents stated in Figure 8's experimental caption, a 2-second step produced 32 spikes in RS (130 pA), with first and last interspike intervals of 17 and 64 ms; cNA (150 pA) produced 73 spikes, with intervals of 23 and 27 ms. Table 1 lists different fitted *model* currents, 98 and 184 pA, at which our model emitted 22 and 100 spikes respectively. RS remained silent at zero injected current. Four 0.25 ms integration substeps per neural tick improved cNA spike count convergence at 150 pA: a one-substep version gave 69 spikes, four substeps 73, and a 0.1 ms numerical reference 74.

**Interpretation.** The qualitative adapting/non-adapting firing patterns are reproduced. This is a check of an implementation against the original paper's *fitted model and recorded pattern*, not an independent fit to raw experimental traces. The paper does not supply the underlying voltage samples here, so exact spike-time error against the recorded neurons has not been measured. Action-potential shape is intentionally replaced by a reset.

## 2. Paired-pulse synaptic depression

**Experiment.** [Tsodyks and Markram, 1997, *PNAS*](https://pmc.ncbi.nlm.nih.gov/articles/PMC19580/) used patch-clamp recordings from connected neocortical pyramidal neurons and fitted a synaptic-resource model. One reported fit uses utilization `U = 0.55` and recovery time `450 ms`.

**Simulation.** With those fitted values, a second presynaptic spike 50 ms after the first released a mean 0.279 of unit synaptic strength across 10,000 independent trials, versus 0.548 on the first (ratio 0.509). After a further 2 seconds, the mean response recovered to 0.546. The simulation gives each excitatory synapse its own resource state.

**Interpretation.** The depression and recovery direction follows the paired-recording result. These numerical outputs are predictions of the fitted model; they have not been independently compared against digitized EPSP amplitudes from the experiment. Individual trials now include quantal release and failures.

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

## 6. Facilitation and depression from one axon

**Experiment.** [Markram, Wang and Tsodyks, 1998, *PNAS*](https://web.stanford.edu/class/cs379c/archive/2016/class_messages_listing/articles/MarkrametalPNAS-98.pdf) simultaneously recorded a pyramidal neuron and two of its targets. During a 30 Hz six-spike train, its connection to another pyramidal neuron depressed while its connection to an interneuron facilitated. The paper gives a fitted facilitating connection with first-spike utilization 0.1, recovery time 30 ms, and facilitation time 1700 ms (Figure 3), and states the pulse-driven utilization equations.

**Simulation.** Each excitatory synapse now has its own running utilization and its own inherited facilitation time. The same six-spike schedule yielded mean sixth/first release ratios of 3.741 for the paper's facilitating profile and 0.134 for our existing depressing profile across 10,000 independent trials. Newly generated excitatory-to-inhibitory synapses start with the published facilitating profile, and their parameters may mutate. The revised CUDA world completed a 300-step run with births.

**Interpretation.** The response direction and target-specific difference are reproduced. The reported ratios are model outputs, not measured values transcribed from the paper; the experiment's voltage traces have not been digitized for an independent amplitude comparison. Individual release is now stochastic, while the expected response follows the paper's phenomenological fit.

## 7. Cortical plasticity needs depolarization and NMDA receptors

**Experiment.** [Sjöström, Turrigiano and Nelson, 2001, *Neuron*](https://plasticity.muhc.mcgill.ca/DataPage/PDFs/Sjostrom_Neuron_2001.pdf) recorded connected layer-5 pyramidal neurons in rat visual cortex. Fifty +10 ms spike pairs at 0.1 Hz failed to potentiate a weak unitary connection. Cooperating inputs or injected depolarization rescued potentiation; 40 Hz bursts were normally effective. Reverse timing produced depression at low rate. NMDA receptor blockade prevented the measured changes. A fitted half-maximum for depolarization-dependent potentiation was 2.1 mV, with a statistical threshold near 2.3 mV.

**Simulation.** Synapses now distinguish two experimentally studied plasticity profiles: the earlier cultured hippocampal protocol and this cortical layer-5 protocol. In the cortical profile, LTP requires recent postsynaptic depolarization of at least 2.3 mV; LTD remains timing-dependent. A 50-pair protocol gave normalized weights of 1.000 for isolated low-rate +10 ms pairs, 1.360 with 3 mV preceding depolarization, 0.861 for −10 ms pairs, and 1.000 when NMDA receptors were blocked in the experimental condition. In an integrated cell-and-synapse protocol, 75 spikes delivered as 40 Hz bursts **with 25 pA background depolarizing current** raised weight to 1.070; 50 sparse pairs remained at 1.000. The GPU world uses the cortical profile initially and computes a short running estimate of preceding somatic depolarization.

**Interpretation.** The low-rate failure, depolarization rescue, reverse-timing depression, and pharmacological blockade are reproduced qualitatively. The 2.3 mV threshold comes from this experiment; the simulator's 10 ms voltage averaging is an approximation. The burst result depends on injected background current: without it, our current single-cell model did *not* build enough residual depolarization to reproduce the paper's 40 Hz result. NMDA blockade is a protocol switch, not a simulated receptor or calcium process. Dopamine-dependent plasticity reported for other cell classes remains outside this cortical profile and is not yet reproduced.

## 8. Variable release and failures

**Experiment.** In [Markram, Wang and Tsodyks, 1998, *PNAS*](https://web.stanford.edu/class/cs379c/archive/2016/class_messages_listing/articles/MarkrametalPNAS-98.pdf), simultaneous recordings from one pyramidal axon and two target cells reported first-response failures in 24% of trials at its interneuron connection and none in 60 trials at its pyramidal connection (Figure 2). Quantal analysis of a small identified sensory-to-motor circuit in [Castellucci and Kandel, 1974, *PNAS*](https://pubmed.ncbi.nlm.nih.gov/4373738/) also supports release in discrete units rather than a perfectly continuous amount.

**Simulation.** Each excitatory connection now has a mutable count of independent effective release sites, each with a ready/depleted state. At a presynaptic arrival, ready sites release probabilistically, become depleted, and recover probabilistically. Fourteen effective sites with the two previously fitted utilization profiles gave first-response failure rates of 22.28% and 0% across 10,000 trials, while retaining the measured directions of facilitation and depression. The GPU gives each synapse its own random state; animal RNG is not shared during spike delivery.

**Interpretation.** This reproduces the failure-rate contrast and produces trial-to-trial quantal variability. The effective site count was **calibrated** to the reported failure probability; it is not an anatomical contact count. We have not compared the full amplitude distribution or coefficient of variation with the experiment. The model also omits spontaneous miniature events and variability in quantum size.

## 9. Slow homeostatic synaptic scaling

**Experiment.** [Turrigiano, Leslie, Desai, Rutherford and Nelson, 1998, *Nature*](https://faculty.washington.edu/somurray/psych545/Turrigiano_Nature_1998.pdf) measured miniature AMPA currents in cultured rat visual-cortex neurons after 48 hours of altered activity. Blocking spikes with TTX increased mean amplitude to 192% ± 16% of control. Blocking GABA-A inhibition with bicuculline initially raised firing to about 265% of control and reduced mean amplitude to 70% ± 4% after 48 hours. Changes were approximately multiplicative across excitatory inputs, developed gradually, and were not significant in bipolar interneurons under activity blockade. The study also found little change in mini-current kinetics or frequency.

**Simulation.** Excitatory pyramidal neurons now maintain a separate *postsynaptic* AMPA scale shared by their excitatory inputs. Its slow empirical response is calibrated to the study's 48-hour endpoints, with 0.4 Hz as the control culture rate. An accelerated protocol that advances the scaling rule one hour at a time gives factors of 1.92 for a continuously silent cell, 1.00 at control firing, 0.70 for a cell held at 2.65 times control firing, and 1.00 for an inhibitory cell. Multiplying two unequal inputs by the same factor preserves their relative strength. The full GPU world completed 2,000 steps and exported finite AMPA scales.

**Interpretation.** The direction, cell-type distinction, multiplicative property, and calibrated endpoints are reproduced. The paper did not report a fixed high firing rate throughout bicuculline treatment; its rate fell during adaptation, whereas the high-rate protocol above holds rate constant. The interpolation over time and between activity conditions is therefore a model assumption, not independently validated physiology. The 2,000-step world run spans only 32 seconds of neural time, so its scales remain near 1 (0.999934–1.00012). Testing the actual 48-hour cellular response requires the accelerated isolated protocol or a much longer neural simulation.
