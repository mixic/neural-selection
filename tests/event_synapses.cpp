#include "physiology.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace sim;

static bool close(float a, float b) {
    return std::abs(a - b) <= 2e-4f * std::max({1.0f, std::abs(a), std::abs(b)});
}

static bool same_state(const SynapseState &a, const SynapseState &b) {
    return close(a.weight_ns, b.weight_ns) &&
        close(a.pre_trace, b.pre_trace) && close(a.post_trace, b.post_trace) &&
        close(a.eligible_ltp, b.eligible_ltp) &&
        close(a.eligible_ltd, b.eligible_ltd) &&
        close(a.da_age_ms, b.da_age_ms) &&
        close(a.spine_volume, b.spine_volume) &&
        close(a.structural_age_ms, b.structural_age_ms) &&
        close(a.pre_age_ms, b.pre_age_ms) &&
        close(a.post_age_ms, b.post_age_ms);
}

int main() {
    for (int profile = 1; profile <= 5; ++profile) {
        SynapseGene g{};
        g.plasticity = profile;
        g.initial_weight_ns = 1.0f;
        g.mod_receptors = MOD_NE | MOD_5HT | MOD_DA;
        SynapseState regular = initial_synapse_state(g);
        SynapseState event = regular;
        int last_event = -1;
        for (int tick = 0; tick < 8192; ++tick) {
            bool arrived = tick % 23 == 4 || tick % 137 == 5;
            bool post = tick % 29 == 8 || tick % 137 == 9;
            unsigned char mod = 0;
            if (tick % 67 == 13) mod |= MOD_NE;
            if (tick % 71 == 17) mod |= MOD_5HT;
            if (tick % 73 == 21) mod |= MOD_DA;
            synapse_plasticity_step(g, regular, arrived, post, true, 3.0f,
                                   true, mod);
            if (arrived || post || mod) {
                synapse_plasticity_quiet(g, event, tick - last_event - 1);
                synapse_plasticity_step(g, event, arrived, post, true, 3.0f,
                                       true, mod);
                last_event = tick;
            }
            if (tick % 16 == 15) {
                synapse_plasticity_quiet(g, event, tick - last_event);
                last_event = tick;
                if (!same_state(regular, event)) {
                    std::fprintf(stderr, "profile %d diverged at tick %d\n",
                                 profile, tick);
                    return 1;
                }
            }
        }
    }
    std::puts("Event-driven quiet synapse updates match tick updates across all five plasticity profiles");
}
