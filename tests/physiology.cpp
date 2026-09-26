#include "physiology.cuh"

#include <cmath>
#include <iostream>
#include <vector>

using namespace sim;

std::vector<int> current_step(const NeuronGene &g, float current_pa) {
    NeuronState state = initial_neuron_state(g);
    std::vector<int> spikes;
    for (int t = 0; t < 2000; ++t) {
        neuron_advance(g, state, current_pa, 0, 0);
        if (state.spike) spikes.push_back(t);
        if (!std::isfinite(state.voltage_mv)) return {};
    }
    return spikes;
}

float paired_plasticity(int interval_ms, bool excitatory_target) {
    SynapseGene gene{1.0f, 0.55f, 450.0f, 0, 1, 5, 1, 1};
    SynapseState state = initial_synapse_state(gene);
    for (int t = 0; t < 60000; ++t) {
        int phase = t % 1000;
        bool arrival = phase == 100;
        bool post = phase == 100 + interval_ms;
        synapse_plasticity_step(gene, state, arrival, post, excitatory_target);
    }
    return state.weight_ns;
}

int main() {
    int failures = 0;
    auto require = [&](bool condition, const char *message) {
        if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
    };

    auto rs = current_step(recorded_profile(0), 130);
    auto cna = current_step(recorded_profile(1), 150);
    auto rs_fit = current_step(recorded_profile(0), 98);
    auto cna_fit = current_step(recorded_profile(1), 184);
    auto quiet = current_step(recorded_profile(0), 0);
    require(rs.size() > 4, "recorded RS profile should spike under a 130 pA step");
    require(cna.size() > 4, "recorded cNA profile should spike under a 150 pA step");
    require(rs_fit.size() > 4 && cna_fit.size() > 4,
            "both fitted profiles should spike at their Table 1 model currents");
    require(quiet.empty(), "RS profile should remain silent without injected current");
    if (rs.size() > 4) {
        int first = rs[1] - rs[0], last = rs.back() - rs[rs.size() - 2];
        require(last > first, "RS profile should adapt during sustained current");
        std::cout << "RS: " << rs.size() << " spikes, first/last ISI " << first << '/' << last << " ms\n";
    }
    if (cna.size() > 4) {
        int first = cna[1] - cna[0], last = cna.back() - cna[cna.size() - 2];
        require(std::abs(last - first) < 10, "cNA profile should be nearly non-adapting");
        std::cout << "cNA: " << cna.size() << " spikes, first/last ISI " << first << '/' << last << " ms\n";
    }
    std::cout << "Table 1 fitted-current spike counts: RS " << rs_fit.size()
              << ", cNA " << cna_fit.size() << '\n';

    auto inhibited = initial_neuron_state(recorded_profile(1));
    neuron_advance(recorded_profile(1), inhibited, 0, 0, 4);
    require(inhibited.voltage_mv < recorded_profile(1).rest_mv,
            "recorded inhibitory reversal should hyperpolarize a resting neuron");
    for (int t = 1; t < 10; ++t)
        neuron_advance(recorded_profile(1), inhibited, 0, 0, 0);
    require(inhibited.inh_ns > 1.0f && inhibited.inh_ns < 2.0f,
            "fast inhibitory conductance should decay on a roughly 10 ms scale");

    SynapseGene gene{1.0f, 0.55f, 450.0f, 0, 1, 1, 1, 0};
    auto state = initial_synapse_state(gene);
    float first = synapse_release(gene, state, 0);
    float second = synapse_release(gene, state, 50);
    float recovered = synapse_release(gene, state, 2050);
    require(std::abs(first - 0.55f) < 1e-5f, "first release must match U=0.55");
    require(second < first * 0.7f && second > first * 0.5f,
            "50 ms paired pulse should be depressed");
    require(recovered > first * 0.95f, "release resources should recover after 2 seconds");
    std::cout << "Paired release: " << first << ", " << second
              << " (ratio " << second / first << "), recovered " << recovered << '\n';

    float plus10 = paired_plasticity(10, true);
    float minus10 = paired_plasticity(-10, true);
    float plus50 = paired_plasticity(50, true);
    float minus50 = paired_plasticity(-50, true);
    float gaba_target = paired_plasticity(10, false);
    require(plus10 > 1.3f && plus10 < 1.6f, "pre-before-post should potentiate near recorded range");
    require(minus10 > 0.75f && minus10 < 0.9f, "post-before-pre should depress near recorded range");
    require(std::abs(plus50 - 1) < 1e-5f && std::abs(minus50 - 1) < 1e-5f,
            "50 ms pairing should have no persistent change in this measured protocol");
    require(std::abs(gaba_target - 1) < 1e-5f,
            "glutamatergic input to inhibitory target should not use this STDP rule");
    std::cout << "60-pair STDP weights: +10 " << plus10 << ", -10 " << minus10
              << ", +50 " << plus50 << ", -50 " << minus50
              << ", inhibitory target " << gaba_target << '\n';
    if (failures) return 1;
    std::cout << "All physiology protocol checks passed\n";
}
