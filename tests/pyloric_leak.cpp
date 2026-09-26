#include "physiology.cuh"

#include <cmath>
#include <iostream>

using namespace sim;

// Zhao et al. Fig. 4A: voltage-ramp slopes were 149 nS during
// decentralization and 71 nS before it; the difference current reversed at
// -63.6 mV on average. A quiet membrane isolates that measured passive term.
float steady_voltage(float probe_pa, float clamp_ns) {
    NeuronGene g = recorded_profile(1);
    g.capacitance_pf = 628.3f;
    g.leak_ns = 149.0f;
    g.rest_mv = -63.6f;
    g.threshold_mv = 100.0f;
    g.adapt_ns = 0.0f;
    g.spike_adapt_pa = 0.0f;
    NeuronState s = initial_neuron_state(g);
    const float no_synapse[DENDRITES]{};
    for (int t = 0; t < 200; ++t)
        neuron_advance(g, s, probe_pa, no_synapse, 0.0f,
                       clamp_ns, -63.6f);
    return s.voltage_mv;
}

float measured_slope_ns(float clamp_ns) {
    const float low = steady_voltage(-50.0f, clamp_ns);
    const float high = steady_voltage(50.0f, clamp_ns);
    return 100.0f / (high - low);
}

int main() {
    const float decentralized = measured_slope_ns(0.0f);
    const float negative_clamp = measured_slope_ns(-78.0f);
    const float positive_clamp = measured_slope_ns(78.0f);
    const float reversal = steady_voltage(0.0f, -78.0f);
    std::cout << "Measured input conductance (nS): decentralized "
              << decentralized << ", negative clamp " << negative_clamp
              << ", positive clamp " << positive_clamp << '\n';
    std::cout << "Difference-current reversal (mV): " << reversal << '\n';
    if (!(std::fabs(decentralized - 149.0f) < 1.0f &&
          std::fabs(negative_clamp - 71.0f) < 1.0f &&
          positive_clamp > decentralized &&
          std::fabs(reversal + 63.6f) < 0.1f)) {
        std::cerr << "Zhao et al. Fig. 4 passive conductance was not reproduced\n";
        return 1;
    }
}
