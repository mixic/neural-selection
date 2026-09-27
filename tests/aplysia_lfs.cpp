#include "physiology.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace sim;

static int sensory_evoked_spikes(bool stimulate) {
    // LE-to-LFS fast and slow conductance components from Lieb and Frost
    // (1997), Table 2: 0.0339/0.0033 microSiemens, 12/50 ms opening,
    // 25/170 ms closing. This 20 Hz challenge is a protocol probe, not
    // their measured 4 g skin stimulus or a fit to its spike count.
    constexpr float weight_ns[2] = {33.9f, 3.3f};
    constexpr float opening_ms[2] = {12.0f, 50.0f};
    constexpr float closing_ms[2] = {25.0f, 170.0f};
    float activated[2]{}, open[2]{};
    NeuronGene gene = aplysia_lfs_profile();
    NeuronState state = initial_neuron_state(gene);
    int response_spikes = 0;
    for (int tick = 0; tick < 5000; ++tick) {
        if (stimulate && tick >= 1000 && tick < 1500 && tick % 50 == 0)
            for (int j = 0; j < 2; ++j) activated[j] += 1.0f;
        float excitation_ns = 0;
        for (int j = 0; j < 2; ++j) {
            open[j] += activated[j] / opening_ms[j] - open[j] / closing_ms[j];
            activated[j] *= std::exp(-1.0f / opening_ms[j]);
            excitation_ns += weight_ns[j] * open[j];
        }
        lfs_advance(gene, state, 0, excitation_ns, 0);
        if (tick >= 1000 && tick < 1500) response_spikes += state.spike;
    }
    return response_spikes;
}

int main() {
    NeuronGene gene = aplysia_lfs_profile();
    float no_synapses[DENDRITES]{};
    NeuronState spontaneous = initial_neuron_state(gene);
    for (int t = 0; t < 10000; ++t)
        neuron_advance(gene, spontaneous, 0, no_synapses, 0);

    NeuronState held = initial_neuron_state(gene);
    held.lfs.voltage_mv = gene.rest_mv - 20.0f;
    held.voltage_mv = held.lfs.voltage_mv;
    for (int t = 0; t < 1000; ++t)
        neuron_advance(gene, held, -300.0f, no_synapses, 0);
    float first = held.voltage_mv;
    for (int t = 0; t < 1000; ++t)
        neuron_advance(gene, held, -400.0f, no_synapses, 0);
    float second = held.voltage_mv;
    float resistance_mohm = (first - second) * 10.0f;
    float tau_ms = gene.capacitance_pf / gene.leak_ns;
    int control_spikes = sensory_evoked_spikes(false);
    int evoked_spikes = sensory_evoked_spikes(true);
    std::cout << "LFS: zero-current spikes in 10 s=" << spontaneous.total_spikes
              << " resting reversal=" << gene.rest_mv
              << " mV measured resistance=" << resistance_mohm
              << " MOhm passive time constant=" << tau_ms
              << " ms, control/evoked spikes in 500 ms="
              << control_spikes << '/' << evoked_spikes << '\n';
    // Lieb and Frost (1997), Table 1: -45.1 mV, 65.6 MOhm, 1.20 nF;
    // LFS cells were spontaneously active in the isolated preparation.
    return spontaneous.total_spikes > 0 &&
           std::fabs(gene.rest_mv + 45.1f) < 0.1f &&
           std::fabs(resistance_mohm - 65.6f) < 0.5f &&
           std::fabs(gene.capacitance_pf - 1200.0f) < 0.1f &&
           std::fabs(tau_ms - 78.72f) < 0.2f &&
           held.total_spikes == 0 && evoked_spikes > control_spikes
           ? EXIT_SUCCESS : EXIT_FAILURE;
}
