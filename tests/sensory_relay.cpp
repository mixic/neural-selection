#include "physiology.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace sim;

// Controlled ORN-like spike protocols through the very same release and
// membrane functions that the CUDA world calls. This is a diagnostic assay:
// the cortical cell profile and generic synapse are not an insect PN fit.
static SynapseGene generic_synapse(float weight_ns) {
    SynapseGene g{};
    g.initial_weight_ns = weight_ns;
    g.utilization = 0.55f;
    g.recovery_ms = 450.0f;
    g.release_sites = 14;
    g.active = 1;
    return g;
}

struct SingleResponse { double epsc_pa, epsp_mv, cv; };

static SingleResponse single_axon(SynapseGene synapse, int trials = 1000) {
    auto cell = cortical_l5_profile();
    double current_sum = 0, voltage_sum = 0, current_sq_sum = 0;
    for (int trial = 1; trial <= trials; ++trial) {
        auto state = initial_neuron_state(cell);
        auto release = initial_synapse_state(synapse, trial);
        float conductance = synapse_release(synapse, release, 0);
        float current = conductance * (EXC_REVERSAL_MV - cell.rest_mv);
        float peak = 0;
        for (int t = 0; t < 50; ++t) {
            float excitation[DENDRITES]{};
            if (t == 0) excitation[0] = conductance;
            neuron_advance(cell, state, 0, excitation, 0);
            peak = std::max(peak, state.voltage_mv - cell.rest_mv);
        }
        current_sum += current;
        current_sq_sum += current * current;
        voltage_sum += peak;
    }
    double mean = current_sum / trials;
    return {mean, voltage_sum / trials,
            std::sqrt(std::max(0.0, current_sq_sum / trials - mean * mean)) / mean};
}

static std::vector<int> pulses(int baseline_ms, int challenge_hz, int challenge_ms) {
    std::vector<int> times;
    for (int t = 0; t < baseline_ms; t += 1000 / 7) times.push_back(t);
    for (int t = baseline_ms; t < baseline_ms + challenge_ms;
         t += 1000 / challenge_hz) times.push_back(t);
    return times;
}

struct TrainResponse { double first_pa, basal_end_pa, challenge_first_pa,
                              challenge_last_pa; };

static TrainResponse train(SynapseGene gene, int challenge_hz, int trials = 1000) {
    auto times = pulses(4000, challenge_hz, 500);
    int basal_last = 0;
    while (basal_last + 1 < static_cast<int>(times.size()) && times[basal_last + 1] < 4000)
        ++basal_last;
    double first = 0, basal = 0, start = 0, end = 0;
    for (int trial = 1; trial <= trials; ++trial) {
        auto state = initial_synapse_state(gene, trial);
        for (int i = 0; i < static_cast<int>(times.size()); ++i) {
            double current = synapse_release(gene, state, times[i]) * 65.0;
            if (i == 0) first += current;
            if (i == basal_last) basal += current;
            if (i == basal_last + 1) start += current;
            if (i == static_cast<int>(times.size()) - 1) end += current;
        }
    }
    return {first / trials, basal / trials, start / trials, end / trials};
}

static double recovery_after_pause(SynapseGene gene, int pause_ms,
                                   int trials = 1000) {
    double recovered = 0, first = 0;
    for (int trial = 1; trial <= trials; ++trial) {
        auto state = initial_synapse_state(gene, trial);
        first += synapse_release(gene, state, 0);
        for (int t = 142; t < 4000; t += 142)
            synapse_release(gene, state, t);
        recovered += synapse_release(gene, state, 4000 + pause_ms);
    }
    return recovered / first;
}

static int sensory_spikes(float sense, float bias_pa = 0,
                         int duration_ms = 500) {
    auto gene = cortical_l5_profile();
    auto state = initial_neuron_state(gene);
    float no_input[DENDRITES]{};
    for (int t = 0; t < duration_ms; ++t)
        neuron_advance(gene, state, bias_pa + 230.0f * sense, no_input, 0);
    return state.total_spikes;
}

// Independent inputs with fixed spike times provide a convergence check.
// Each axon has its own release state, as in the GPU world.
static int convergent_spikes(int inputs, int rate_hz, SynapseGene gene,
                            int duration_ms = 500) {
    auto cell = cortical_l5_profile();
    auto state = initial_neuron_state(cell);
    std::vector<SynapseState> axons;
    for (int i = 0; i < inputs; ++i)
        axons.push_back(initial_synapse_state(gene, i + 1));
    for (int t = 0; t < duration_ms; ++t) {
        float excitation[DENDRITES]{};
        for (int i = 0; i < inputs; ++i) {
            // Evenly stagger spikes so this does not assume synchrony.
            int period = 1000 / rate_hz;
            if ((t + i * period / inputs) % period == 0)
                excitation[i % DENDRITES] += synapse_release(gene, axons[i], t);
        }
        neuron_advance(cell, state, 0, excitation, 0);
    }
    return state.total_spikes;
}

struct OdorResponse { int baseline_spikes, odor_spikes; };

static OdorResponse odor_step(int inputs, int odor_hz, SynapseGene gene) {
    auto cell = cortical_l5_profile();
    auto state = initial_neuron_state(cell);
    std::vector<SynapseState> axons;
    for (int i = 0; i < inputs; ++i)
        axons.push_back(initial_synapse_state(gene, i + 1));
    int baseline_spikes = 0, odor_spikes = 0;
    for (int t = 0; t < 4500; ++t) {
        bool odor = t >= 4000;
        int rate = odor ? odor_hz : 7;
        int period = 1000 / rate;
        float excitation[DENDRITES]{};
        for (int i = 0; i < inputs; ++i) {
            int phase = (i * period) / inputs;
            int relative_t = odor ? t - 4000 : t;
            if ((relative_t + phase) % period == 0)
                excitation[i % DENDRITES] += synapse_release(gene, axons[i], t);
        }
        neuron_advance(cell, state, 0, excitation, 0);
        if (state.spike && t >= 3500 && t < 4000) ++baseline_spikes;
        if (state.spike && odor) ++odor_spikes;
    }
    return {baseline_spikes, odor_spikes};
}

int main() {
    std::cout << std::fixed << std::setprecision(3);
    auto generic = generic_synapse(2.25f);
    auto one = single_axon(generic);
    auto matched = generic_synapse(29.0f / (65.0f * 0.55f));
    auto matched_current = single_axon(matched);
    std::cout << "single_generic epsc_pa=" << one.epsc_pa
              << " epsp_mv=" << one.epsp_mv << " current_cv=" << one.cv << '\n';
    std::cout << "single_current_matched epsc_pa=" << matched_current.epsc_pa
              << " epsp_mv=" << matched_current.epsp_mv
              << " current_cv=" << matched_current.cv << '\n';
    for (int hz : {20, 50}) {
        auto response = train(generic, hz);
        std::cout << "train_" << hz << "Hz first_pa=" << response.first_pa
                  << " basal_end_pa=" << response.basal_end_pa
                  << " basal_ratio=" << response.basal_end_pa / response.first_pa
                  << " challenge_first_pa=" << response.challenge_first_pa
                  << " challenge_last_pa=" << response.challenge_last_pa << '\n';
    }
    // Kazama & Wilson's estimates: roughly 0.75 release probability and
    // dozens of sites per ORN axon. Recovery is calibrated to their reported
    // approximately 40% depression after sustained 7 Hz stimulation.
    auto orn = generic_synapse(29.0f / (65.0f * 0.75f));
    orn.utilization = 0.75f;
    orn.recovery_ms = 170.0f;
    orn.release_sites = 50;
    auto orn_single = single_axon(orn);
    auto orn_train = train(orn, 50);
    std::cout << "orn_pn_candidate epsc_pa=" << orn_single.epsc_pa
              << " epsp_mv=" << orn_single.epsp_mv
              << " current_cv=" << orn_single.cv
              << " basal_ratio=" << orn_train.basal_end_pa / orn_train.first_pa
              << " challenge_last_pa=" << orn_train.challenge_last_pa
              << " recovery_1s=" << recovery_after_pause(orn, 1000) << '\n';
    for (float bias : {0.0f, 50.0f, 100.0f})
        for (float level : {0.0f, 0.125f, 0.25f, 0.5f, 1.0f})
            std::cout << "sensory_bias_" << bias << "_level_" << level
                      << " spikes_500ms=" << sensory_spikes(level, bias) << '\n';
    for (int inputs : {1, 2, 4, 8, 16})
        std::cout << "convergence_" << inputs << " spikes_500ms="
                  << convergent_spikes(inputs, 20, generic) << '\n';
    for (int inputs : {16, 32, 64})
        std::cout << "orn_convergence_" << inputs << " spikes_500ms="
                  << convergent_spikes(inputs, 20, orn) << '\n';
    for (int inputs : {16, 32, 64}) for (int hz : {20, 50}) {
        auto response = odor_step(inputs, hz, orn);
        std::cout << "orn_odor_" << inputs << "x" << hz
                  << "Hz pn_baseline_hz=" << response.baseline_spikes * 2
                  << " pn_odor_hz=" << response.odor_spikes * 2 << '\n';
    }
}
