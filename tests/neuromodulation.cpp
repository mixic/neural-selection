#include "physiology.cuh"

#include <cmath>
#include <iostream>

using namespace sim;

struct CorticalResult { float pre_post, post_pre; };

// He et al. (2015): two independent L4-to-L2/3 pathways, 200 pairings at
// 10 Hz, one presynaptic event and four postsynaptic spikes at 100 Hz.
CorticalResult cortical_eligibility(unsigned char modulator, int delay_ms,
                                    bool pair = true,
                                    unsigned char receptors = MOD_NE | MOD_5HT) {
    SynapseGene gene{1.0f, 0.55f, 450.0f, 0, 0, 1, 1, 1, 3, 0, 14};
    gene.mod_receptors = receptors;
    SynapseState forward = initial_synapse_state(gene);
    SynapseState reverse = initial_synapse_state(gene);
    int mod_tick = delay_ms < 0 ? -1 : 20000 + delay_ms;
    int duration = mod_tick < 0 ? 20001 : mod_tick + 1;
    for (int t = 0; t < duration; ++t) {
        int phase = t % 100;
        bool training = pair && t < 20000;
        bool post_forward = training && (phase == 10 || phase == 20 ||
                                         phase == 30 || phase == 40);
        bool post_reverse = training && (phase == 0 || phase == 10 ||
                                         phase == 20 || phase == 30);
        unsigned char event = t == mod_tick ? modulator : 0;
        synapse_plasticity_step(gene, forward, training && phase == 0,
                                post_forward, true, 0, true, event);
        synapse_plasticity_step(gene, reverse, training && phase == 40,
                                post_reverse, true, 0, true, event);
    }
    return {forward.weight_ns, reverse.weight_ns};
}

int main() {
    int failures = 0;
    auto require = [&](bool ok, const char *message) {
        if (!ok) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
    };
    CorticalResult alone = cortical_eligibility(0, -1);
    CorticalResult ne = cortical_eligibility(MOD_NE, 0);
    CorticalResult serotonin = cortical_eligibility(MOD_5HT, 0);
    CorticalResult late_ne = cortical_eligibility(MOD_NE, 10000);
    CorticalResult late_serotonin = cortical_eligibility(MOD_5HT, 5000);
    CorticalResult ne_only = cortical_eligibility(MOD_NE, 0, false);
    CorticalResult blocked = cortical_eligibility(MOD_NE, 0, true, MOD_5HT);
    require(std::abs(alone.pre_post - 1) < 1e-5f &&
            std::abs(alone.post_pre - 1) < 1e-5f,
            "conditioning alone should leave both visual-cortex pathways unchanged");
    require(ne.pre_post > 1.2f && ne.pre_post < 1.42f &&
            std::abs(ne.post_pre - 1) < 1e-5f,
            "NE should selectively convert the pre-post trace into LTP");
    require(serotonin.post_pre > 0.64f && serotonin.post_pre < 0.82f &&
            std::abs(serotonin.pre_post - 1) < 1e-5f,
            "serotonin should selectively convert the post-pre trace into LTD");
    require(late_ne.pre_post < 1.1f && late_serotonin.post_pre > 0.9f,
            "eligibility should fade before late modulator application");
    require(std::abs(ne_only.pre_post - 1) < 1e-5f &&
            std::abs(blocked.pre_post - 1) < 1e-5f,
            "modulator without activity or matching receptor should not change weight");
    std::cout << "He 2015: alone " << alone.pre_post << '/' << alone.post_pre
              << ", NE " << ne.pre_post << '/' << ne.post_pre
              << ", 5HT " << serotonin.pre_post << '/' << serotonin.post_pre
              << ", delayed NE/5HT " << late_ne.pre_post << '/'
              << late_serotonin.post_pre << '\n';
    return failures ? 1 : 0;
}
