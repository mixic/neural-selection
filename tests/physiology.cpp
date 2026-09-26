#include "physiology.cuh"

#include <cmath>
#include <iostream>
#include <vector>

using namespace sim;

std::vector<int> current_step(const NeuronGene &g, float current_pa) {
    NeuronState state = initial_neuron_state(g);
    std::vector<int> spikes;
    const float no_input[DENDRITES]{};
    for (int t = 0; t < 2000; ++t) {
        neuron_advance(g, state, current_pa, no_input, 0);
        if (state.spike) spikes.push_back(t);
        if (!std::isfinite(state.voltage_mv)) return {};
    }
    return spikes;
}

float dendritic_peak(float each_ns, bool same_branch, bool two_inputs = true) {
    NeuronGene g = recorded_profile(0);
    g.threshold_mv = 100.0f; // voltage-clamp-style subthreshold readout
    NeuronState state = initial_neuron_state(g);
    float peak = 0;
    for (int t = 0; t < 30; ++t) {
        float input[DENDRITES]{};
        if (t == 0) {
            input[0] = each_ns;
            if (two_inputs) input[same_branch ? 0 : 1] += each_ns;
        }
        neuron_advance(g, state, 0, input, 0);
        peak = std::max(peak, state.voltage_mv - g.rest_mv);
    }
    return peak;
}

float paired_plasticity(int interval_ms, bool excitatory_target,
                        unsigned char profile = 1, float depol_mv = 0,
                        bool nmda_available = true, int pairs = 60) {
    SynapseGene gene{1.0f, 0.55f, 450.0f, 0, 0, 1, 5, 1, 1};
    gene.plasticity = profile;
    SynapseState state = initial_synapse_state(gene);
    for (int t = 0; t < pairs * 1000; ++t) {
        int phase = t % 1000;
        bool arrival = phase == 100;
        bool post = phase == 100 + interval_ms;
        synapse_plasticity_step(gene, state, arrival, post, excitatory_target,
                                depol_mv, nmda_available);
    }
    return state.weight_ns;
}

float cortical_train(bool bursts, int &evoked_spikes) {
    NeuronGene neuron = recorded_profile(0);
    NeuronState cell = initial_neuron_state(neuron);
    SynapseGene gene{1.0f, 0.55f, 450.0f, 0, 0, 1, 1, 1, 2};
    SynapseState synapse = initial_synapse_state(gene);
    evoked_spikes = 0;
    int eligible_spikes = 0;
    int duration = bursts ? 15000 : 500000;
    for (int t = 0; t < duration; ++t) {
        bool pre = bursts ? (t % 1000 >= 100 && t % 1000 <= 200 &&
                             (t % 1000 - 100) % 25 == 0)
                          : t % 10000 == 100;
        bool post_stim = bursts ? (t % 1000 >= 110 && t % 1000 <= 210 &&
                                   (t % 1000 - 110) % 25 == 0)
                                : t % 10000 == 110;
        float input[DENDRITES]{};
        if (pre) input[0] = 0.2f;
        neuron_advance(neuron, cell, (bursts ? 25.0f : 0.0f)
                       + (post_stim ? 3500.0f : 0.0f), input, 0);
        if (cell.spike) ++evoked_spikes;
        if (cell.spike && cell.recent_depol_mv >= 2.3f) ++eligible_spikes;
        synapse_plasticity_step(gene, synapse, pre, cell.spike, true,
                                cell.recent_depol_mv);
    }
    std::cout << (bursts ? "burst" : "sparse") << " eligible spikes "
              << eligible_spikes << '/' << evoked_spikes << '\n';
    return synapse.weight_ns;
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
    const float no_input[DENDRITES]{};
    neuron_advance(recorded_profile(1), inhibited, 0, no_input, 4);
    require(inhibited.voltage_mv < recorded_profile(1).rest_mv,
            "recorded inhibitory reversal should hyperpolarize a resting neuron");
    for (int t = 1; t < 10; ++t)
        neuron_advance(recorded_profile(1), inhibited, 0, no_input, 0);
    require(inhibited.inh_ns > 1.0f && inhibited.inh_ns < 2.0f,
            "fast inhibitory conductance should decay on a roughly 10 ms scale");

    float weak_separate = dendritic_peak(0.5f, false);
    float weak_same = dendritic_peak(0.5f, true);
    float medium_single = dendritic_peak(2.5f, false, false);
    float medium_same = dendritic_peak(2.5f, true);
    float medium_separate = dendritic_peak(2.5f, false);
    float strong_separate = dendritic_peak(8.0f, false);
    float strong_same = dendritic_peak(8.0f, true);
    require(weak_same > weak_separate * 0.9f && weak_same < weak_separate * 1.1f,
            "weak same-branch inputs should sum approximately linearly");
    require(medium_same > medium_separate * 1.2f,
            "intermediate same-branch inputs should sum superlinearly");
    require(medium_separate > 2.0f * medium_single * 0.85f &&
            medium_separate < 2.0f * medium_single * 1.1f,
            "intermediate between-branch inputs should sum approximately linearly");
    require(strong_same < strong_separate * 0.9f,
            "strong same-branch inputs should saturate");
    std::cout << "Dendritic peak mV: weak same/separate " << weak_same << '/' << weak_separate
              << ", intermediate " << medium_same << '/' << medium_separate
              << ", strong same/separate " << strong_same << '/' << strong_separate << '\n';

    constexpr int trials = 10000;
    SynapseGene gene{1.0f, 0.55f, 450.0f, 0, 0, 1, 1, 1, 0, 0, 14};
    double first = 0, second = 0, recovered = 0;
    for (int trial = 1; trial <= trials; ++trial) {
        auto state = initial_synapse_state(gene, trial);
        first += synapse_release(gene, state, 0);
        second += synapse_release(gene, state, 50);
        recovered += synapse_release(gene, state, 2050);
    }
    first /= trials; second /= trials; recovered /= trials;
    require(std::abs(first - 0.55) < 0.02, "mean first release should match U=0.55");
    require(second < first * 0.7 && second > first * 0.45,
            "mean 50 ms paired pulse should be depressed");
    require(recovered > first * 0.95, "mean release should recover after 2 seconds");
    std::cout << "Paired release means: " << first << ", " << second
              << " (ratio " << second / first << "), recovered " << recovered << '\n';

    SynapseGene facilitating{1.0f, 0.1f, 30.0f, 1700.0f, 0, 1, 1, 1, 0, 0, 14};
    SynapseGene depressing{1.0f, 0.55f, 450.0f, 0, 0, 1, 1, 1, 0, 0, 14};
    double facil_first = 0, facil_sixth = 0, depress_first = 0, depress_sixth = 0;
    int facil_failures = 0, depress_failures = 0;
    for (int trial = 1; trial <= trials; ++trial) {
        auto facil_state = initial_synapse_state(facilitating, trial);
        auto depress_state = initial_synapse_state(depressing, trial + trials);
        for (int pulse = 0; pulse < 6; ++pulse) {
            float f = synapse_release(facilitating, facil_state, pulse * 33);
            float d = synapse_release(depressing, depress_state, pulse * 33);
            if (pulse == 0) {
                facil_first += f; depress_first += d;
                facil_failures += f == 0; depress_failures += d == 0;
            }
            if (pulse == 5) { facil_sixth += f; depress_sixth += d; }
        }
    }
    facil_first /= trials; facil_sixth /= trials;
    depress_first /= trials; depress_sixth /= trials;
    require(facil_sixth > facil_first * 2.0f,
            "30 Hz train should facilitate at fitted interneuron target");
    require(depress_sixth < depress_first * 0.5f,
            "same 30 Hz train should depress at pyramidal target");
    require(facil_failures > trials * 0.20 && facil_failures < trials * 0.26,
            "interneuron target should have about 24% first-response failures");
    require(depress_failures < trials * 0.01,
            "pyramidal target should rarely fail on the first response");
    std::cout << "30 Hz same-axon sixth/first: interneuron "
              << facil_sixth / facil_first << ", pyramidal "
              << depress_sixth / depress_first << "; first failures "
              << (double)facil_failures / trials << '/' << (double)depress_failures / trials << '\n';

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

    float cortical_weak = paired_plasticity(10, true, 2, 0, true, 50);
    float cortical_cooperative = paired_plasticity(10, true, 2, 3, true, 50);
    float cortical_ltd = paired_plasticity(-10, true, 2, 0, true, 50);
    float cortical_blocked = paired_plasticity(10, true, 2, 3, false, 50);
    require(std::abs(cortical_weak - 1) < 1e-5f,
            "low-rate isolated L5 pair should not potentiate");
    require(cortical_cooperative > 1.2f,
            "preceding depolarization should rescue L5 potentiation");
    require(cortical_ltd < 0.9f,
            "reverse timing should depress at low rate without depolarization");
    require(std::abs(cortical_blocked - 1) < 1e-5f,
            "NMDA blockade should prevent L5 potentiation");
    std::cout << "Cortical L5 50-pair weights: weak " << cortical_weak
              << ", cooperative " << cortical_cooperative
              << ", reverse " << cortical_ltd
              << ", NMDA blocked " << cortical_blocked << '\n';
    int burst_spikes = 0, sparse_spikes = 0;
    float burst_weight = cortical_train(true, burst_spikes);
    float sparse_weight = cortical_train(false, sparse_spikes);
    require(burst_spikes == 75 && sparse_spikes == 50,
            "current pulses should evoke the requested cortical spike protocols");
    require(burst_weight > sparse_weight + 0.05f,
            "depolarized 40 Hz bursts should potentiate more than sparse pairs");
    std::cout << "Cortical train weights: depolarized 40 Hz " << burst_weight
              << " (" << burst_spikes << " evoked), 0.1 Hz " << sparse_weight
              << " (" << sparse_spikes << " evoked)\n";

    NeuronGene pyramid = recorded_profile(0);
    NeuronGene interneuron = recorded_profile(1);
    auto quiet_homeo = initial_neuron_state(pyramid);
    auto control_homeo = initial_neuron_state(pyramid);
    auto active_homeo = initial_neuron_state(pyramid);
    auto inhibitory_homeo = initial_neuron_state(interneuron);
    for (int hour = 0; hour < 48; ++hour) {
        homeostatic_advance(pyramid, quiet_homeo, 3600000.0f, 0.0f);
        homeostatic_advance(pyramid, control_homeo, 3600000.0f, 0.4f);
        homeostatic_advance(pyramid, active_homeo, 3600000.0f, 1.06f);
        homeostatic_advance(interneuron, inhibitory_homeo, 3600000.0f, 0.0f);
    }
    float quiet_scale = std::exp(quiet_homeo.log_ampa_scale);
    float control_scale = std::exp(control_homeo.log_ampa_scale);
    float active_scale = std::exp(active_homeo.log_ampa_scale);
    float inhibitory_scale = std::exp(inhibitory_homeo.log_ampa_scale);
    require(std::abs(quiet_scale - 1.92f) < 0.02f,
            "48-hour activity blockade should increase pyramidal AMPA amplitude");
    require(std::abs(active_scale - 0.70f) < 0.02f,
            "48-hour elevated firing should decrease pyramidal AMPA amplitude");
    require(std::abs(control_scale - 1) < 1e-5f &&
            std::abs(inhibitory_scale - 1) < 1e-5f,
            "control activity and inhibitory cells should not scale in this protocol");
    require(std::abs((2.0f * quiet_scale) / (0.5f * quiet_scale) - 4) < 1e-5f,
            "homeostatic scaling should preserve relative synaptic strengths");
    std::cout << "48-hour AMPA scales: silent " << quiet_scale
              << ", control " << control_scale << ", elevated " << active_scale
              << ", interneuron " << inhibitory_scale << '\n';
    if (failures) return 1;
    std::cout << "All physiology protocol checks passed\n";
}
