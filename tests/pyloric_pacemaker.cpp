#include "physiology.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace sim;

struct Response {
    int spikes = 0, bursts = 0;
    float lowest_mv = 100, highest_mv = -100;
};

Response run_pacemaker(float ambient_leak_ns, float dynamic_clamp_ns) {
    NeuronGene gene = crab_pacemaker_profile();
    gene.prinz_ambient_leak_ns = ambient_leak_ns;
    NeuronState cell = initial_neuron_state(gene);
    const float no_synapse[DENDRITES]{};
    Response response;
    int last_spike = -1;
    for (int t = 0; t < 20000; ++t) {
        neuron_advance(gene, cell, 0, no_synapse, 0,
                       dynamic_clamp_ns, -68);
        if (!std::isfinite(cell.voltage_mv)) return {};
        if (t < 10000) continue;
        response.lowest_mv = std::min(response.lowest_mv, cell.voltage_mv);
        response.highest_mv = std::max(response.highest_mv, cell.voltage_mv);
        if (cell.spike) {
            ++response.spikes;
            if (last_spike >= 0 && t-last_spike > 100)
                ++response.bursts;
            last_spike = t;
        }
    }
    return response;
}

int main() {
    auto control = run_pacemaker(0, 0);
    auto decentralized = run_pacemaker(80, 0);
    auto rescued = run_pacemaker(80, -80);
    auto positive = run_pacemaker(80, 80);
    std::cout << "Control / high-leak / negative-clamp / positive-clamp "
                 "spikes: " << control.spikes << " / "
              << decentralized.spikes << " / " << rescued.spikes << " / "
              << positive.spikes << '\n';
    std::cout << "Control / rescued burst gaps: "
              << control.bursts << " / " << rescued.bursts << '\n';
    if (!(control.spikes > 50 && control.bursts >= 4 &&
          decentralized.spikes == 0 &&
          rescued.spikes > 50 && rescued.bursts >= 4 &&
          positive.spikes == 0 &&
          rescued.lowest_mv < -60 && rescued.highest_mv > 0)) {
        std::cerr << "Pacemaker leak-dependent bursting was not reproduced\n";
        return 1;
    }
}
