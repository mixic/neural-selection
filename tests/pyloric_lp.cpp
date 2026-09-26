#include "physiology.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace sim;

struct Response {
    int spikes = 0, long_intervals = 0, longest_interval_ms = 0;
    int deeply_hyperpolarized_ms = 0;
};

Response isolated_lp(float injected_na) {
    const auto gene = lobster_lp_profile();
    auto cell = initial_neuron_state(gene);
    const float no_synapse[DENDRITES]{};
    Response result;
    int previous_spike = -1;
    for (int t = 0; t < 20000; ++t) {
        neuron_advance(gene, cell, injected_na * 1000.0f,
                       no_synapse, 0.0f);
        if (!std::isfinite(cell.voltage_mv)) return {};
        if (t < 5000) continue; // original paper records after settling
        result.deeply_hyperpolarized_ms += cell.voltage_mv < -65.0f;
        if (!cell.spike) continue;
        ++result.spikes;
        if (previous_spike >= 0) {
            int interval = t - previous_spike;
            result.long_intervals += interval > 100;
            result.longest_interval_ms =
                std::max(result.longest_interval_ms, interval);
        }
        previous_spike = t;
    }
    return result;
}

int main() {
    auto hyperpolarized = isolated_lp(-4.0f);
    auto control = isolated_lp(0.0f);
    auto depolarized = isolated_lp(1.0f);
    std::cout << "LP current sweep (-4, 0, +1 nA), spike counts: "
              << hyperpolarized.spikes << ", " << control.spikes
              << ", " << depolarized.spikes << '\n';
    std::cout << "Longest silent intervals (ms): "
              << hyperpolarized.longest_interval_ms << ", "
              << control.longest_interval_ms << ", "
              << depolarized.longest_interval_ms << '\n';
    std::cout << "Time below -65 mV (ms of 15000): "
              << hyperpolarized.deeply_hyperpolarized_ms << ", "
              << control.deeply_hyperpolarized_ms << ", "
              << depolarized.deeply_hyperpolarized_ms << '\n';
    bool reproduced = hyperpolarized.spikes > 20 &&
        hyperpolarized.long_intervals >= 5 &&
        hyperpolarized.longest_interval_ms > 1000 &&
        control.spikes > 20 &&
        depolarized.spikes > 20 &&
        depolarized.longest_interval_ms < 500 &&
        depolarized.deeply_hyperpolarized_ms == 0;
    if (!reproduced) {
        std::cerr << "Isolated LP current-sweep pattern was not reproduced\n";
        return 1;
    }
}
