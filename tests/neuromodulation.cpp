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

// Shindou et al. (2019): 60 corticostriatal pairings at 0.1 Hz, with one
// EPSP 10 ms before a three-spike postsynaptic burst at 100 Hz.
float striatal_eligibility(int dopamine_offset_ms, bool l_type_available,
                          bool nmda_available = true,
                          unsigned char receptors = MOD_DA,
                          bool calcium_permeable_ampa_available = true) {
    SynapseGene gene{1.0f, 0.55f, 450.0f, 0, 0, 1, 1, 1, 4, 0, 14};
    gene.mod_receptors = receptors;
    SynapseState synapse = initial_synapse_state(gene);
    for (int t = 0; t < 600000; ++t) {
        int phase = t % 10000;
        bool pre = phase == 0;
        bool post = phase == 10 || phase == 20 || phase == 30;
        int mod_phase = dopamine_offset_ms < 0 ?
                        10000 + dopamine_offset_ms : dopamine_offset_ms;
        unsigned char event = dopamine_offset_ms != -100000 &&
                              phase == mod_phase ? MOD_DA : 0;
        synapse_plasticity_step(gene, synapse, pre, post, true, 0,
                                nmda_available, event, l_type_available,
                                calcium_permeable_ampa_available);
    }
    return synapse.weight_ns;
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
    float striatum_alone = striatal_eligibility(-100000, true);
    float striatum_nimo = striatal_eligibility(-100000, false);
    float striatum_early = striatal_eligibility(-2000, false);
    float striatum_zero = striatal_eligibility(0, false);
    float striatum_two = striatal_eligibility(2000, false);
    float striatum_four = striatal_eligibility(4000, false);
    float striatum_two_no_nimo = striatal_eligibility(2000, true);
    float striatum_nmda_blocked = striatal_eligibility(2000, false, false);
    float striatum_d1_blocked = striatal_eligibility(2000, false, true, 0);
    float striatum_ampa_blocked = striatal_eligibility(2000, false, true,
                                                       MOD_DA, false);
    require(striatum_alone > 0.62f && striatum_alone < 0.74f,
            "D1 pre-post pairing should produce LTD without dopamine");
    require(std::abs(striatum_nimo - 1) < 1e-5f,
            "blocking L-type calcium current should remove competing LTD");
    require(striatum_two > 1.18f && striatum_two < 1.36f,
            "dopamine two seconds after pairing should unmask LTP");
    require(striatum_early < 1.08f && striatum_zero < 1.08f &&
            striatum_four < 1.10f,
            "early, simultaneous, and four-second dopamine should not potentiate strongly");
    require(striatum_two_no_nimo > striatum_alone + 0.18f &&
            striatum_two_no_nimo < 1.1f,
            "delayed dopamine should counteract LTD when L-type channels remain open");
    require(striatum_nmda_blocked > 1.18f &&
            std::abs(striatum_d1_blocked - 1) < 1e-5f &&
            std::abs(striatum_ampa_blocked - 1) < 1e-5f,
            "striatal LTP should require D1 and calcium-permeable AMPA, not NMDA");
    std::cout << "Shindou 2019: pairing alone " << striatum_alone
              << ", nimodipine " << striatum_nimo
              << ", DA at -2/0/+2/+4 s " << striatum_early << '/'
              << striatum_zero << '/' << striatum_two << '/'
              << striatum_four << ", +2 s no nimodipine "
              << striatum_two_no_nimo << '\n';
    return failures ? 1 : 0;
}
