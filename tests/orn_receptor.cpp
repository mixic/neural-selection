#include "physiology.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace sim;

static double response_rate(NeuronGene gene, float sensory_level,
                            int trials = 100) {
    gene.sensory_channel = 0;
    int spikes = 0;
    for (int trial = 1; trial <= trials; ++trial) {
        auto state = initial_neuron_state(gene, trial);
        float no_synapses[DENDRITES]{};
        for (int t = 0; t < 5000; ++t) {
            float current = receptor_current(gene, state, sensory_level);
            neuron_advance(gene, state, current, no_synapses, 0);
            if (t >= 1000) spikes += state.spike;
        }
    }
    return spikes * 1000.0 / (trials * 4000);
}

int main() {
    auto receptor = orn_receptor_profile();
    double spontaneous = response_rate(receptor, 0);
    double weak = response_rate(receptor, 0.3f);
    double strong = response_rate(receptor, 1);
    auto silent_receptor = receptor;
    silent_receptor.receptor_event_ns = 0;
    double silent = response_rate(silent_receptor, 1);
    std::cout << "ORN receptor current protocol: spontaneous=" << spontaneous
              << "Hz weak=" << weak << "Hz strong=" << strong
              << "Hz receptor-silenced=" << silent << "Hz\n";
    // Kazama & Wilson's 7, 20 and 50 Hz stimulus protocols define the
    // effective-rate targets. Receptor disruption controls the spontaneous
    // drive; it is not interpreted as a full Orco knockout simulation.
    if (std::abs(spontaneous - 7) > 1 ||
        std::abs(weak - 20) > 3 ||
        std::abs(strong - 50) > 5 || silent != 0)
        return EXIT_FAILURE;

    receptor.sensory_channel = 0;
    int first_spikes = 0, second_spikes = 0, coincident = 0;
    float no_synapses[DENDRITES]{};
    for (int trial = 1; trial <= 100; ++trial) {
        auto first = initial_neuron_state(receptor, trial);
        auto second = initial_neuron_state(receptor, trial + 100000);
        for (int t = 0; t < 4000; ++t) {
            float a = receptor_current(receptor, first, 0);
            float b = receptor_current(receptor, second, 0);
            neuron_advance(receptor, first, a, no_synapses, 0);
            neuron_advance(receptor, second, b, no_synapses, 0);
            first_spikes += first.spike;
            second_spikes += second.spike;
            coincident += first.spike && second.spike;
        }
    }
    std::cout << "independent afferents: first=" << first_spikes
              << " second=" << second_spikes
              << " same-ms coincidences=" << coincident << '\n';
    return first_spikes > 0 && second_spikes > 0 &&
           coincident < first_spikes / 20 ? EXIT_SUCCESS : EXIT_FAILURE;
}
