#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define NS_HD __host__ __device__
#else
#define NS_HD
#endif

namespace sim {
constexpr float DT_MS = 1.0f;
constexpr float EXC_DECAY_MS = 3.0f;
constexpr int DENDRITES = 4;
constexpr float INH_DECAY_MS = 10.0f;
constexpr float EXC_REVERSAL_MV = 0.0f;
constexpr float INH_REVERSAL_MV = -75.0f;
constexpr float LTP_TRACE_MS = 17.0f;
constexpr float LTD_TRACE_MS = 34.0f;
constexpr unsigned char MOD_NE = 1;
constexpr unsigned char MOD_5HT = 2;
constexpr unsigned char MOD_DA = 4;

// Parameter sets fitted to step-current recordings in Naud et al. (2008), Table 1.
// 0: regular-spiking pyramidal cell; 1: continuously non-adapting interneuron.
struct NeuronGene {
    float capacitance_pf, leak_ns, rest_mv, threshold_mv, slope_mv;
    float adapt_ns, adapt_tau_ms, spike_adapt_pa, reset_mv, bias_pa;
    float afterdepol_spike_pa, afterdepol_tau_ms;
    unsigned char type, active, modulator;
};
struct NeuronState {
    float voltage_mv, adaptation_pa, afterdepol_pa, inh_ns, recent_depol_mv, log_ampa_scale;
    float dendrite_exc_ns[DENDRITES];
    int total_spikes;
    unsigned char spike;
};
struct SynapseGene {
    float initial_weight_ns, utilization, recovery_ms, facilitation_ms;
    int pre, post;
    unsigned char delay, active, plasticity, branch, release_sites, mod_receptors;
};
struct SynapseState {
    float weight_ns, running_utilization, pre_trace, post_trace;
    float eligible_ltp, eligible_ltd, da_age_ms;
    float spine_volume, structural_age_ms;
    float pre_age_ms, post_age_ms;
    int64_t last_arrival_tick;
    uint32_t ready_mask, rng_state;
    unsigned char arrived;
};

NS_HD inline float bound(float x, float low, float high) {
    return fminf(high, fmaxf(low, x));
}

NS_HD inline NeuronGene recorded_profile(unsigned char type) {
    if (type == 1) return {59, 2.9f, -62, -42, 3.0f, 1.8f, 16, 61, -54, 0, 0, 0, 1, 1, 0};
    return {104, 4.3f, -65, -52, 0.8f, -0.8f, 88, 65, -53, 0, 0, 0, 0, 1, 0};
}

// Visual-cortex L5 protocol profile. The Naud step-current fit above remains
// unchanged; the inward aftercurrent is fitted to Sjostrom et al.'s residual
// 40 Hz depolarization, with a timescale consistent with cortical fADP recordings.
NS_HD inline NeuronGene cortical_l5_profile() {
    NeuronGene g = recorded_profile(0);
    g.reset_mv = -68.0f;
    g.afterdepol_spike_pa = 120.0f;
    g.afterdepol_tau_ms = 45.0f;
    return g;
}

NS_HD inline NeuronState initial_neuron_state(const NeuronGene &g) {
    NeuronState state{};
    state.voltage_mv = g.rest_mv;
    return state;
}

// Phenomenological thin-branch input/output curve: linear for weak inputs,
// regenerative at intermediate strengths, then saturated. Polsky et al.
// (2004) measured this shape for nearby inputs on one pyramidal dendrite.
// The nS threshold and gain here are fitted to the qualitative protocol,
// not measured molecular channel parameters.
NS_HD inline float dendritic_transfer(float conductance_ns) {
    float regenerative = 5.0f / (1.0f + expf(-(conductance_ns - 4.0f) / 0.4f));
    float offset = 5.0f / (1.0f + expf(10.0f));
    return fminf(11.0f, conductance_ns + regenerative - offset);
}

NS_HD inline float soma_excitation(const NeuronGene &g, const NeuronState &s) {
    float total = 0;
    for (int branch = 0; branch < DENDRITES; ++branch)
        total += g.type == 0 ? dendritic_transfer(s.dendrite_exc_ns[branch])
                             : s.dendrite_exc_ns[branch];
    return total;
}

// Empirical 48-hour AMPA quantal-amplitude response in cultured pyramidal
// cells (Turrigiano et al. 1998). The interpolation between measured conditions
// is an approximation; no activity response is assigned to inhibitory cells.
NS_HD inline void homeostatic_advance(const NeuronGene &g, NeuronState &s,
                                      float elapsed_ms, float observed_hz) {
    if (g.type != 0) return;
    constexpr float target_hz = 0.4f; // control culture mean in Figure 2d
    float ratio = bound(observed_hz / target_hz, 0.0f, 2.65f);
    float change_per_48h = ratio < 1.0f
        ? logf(1.92f) * (1.0f - ratio)
        : logf(0.70f) * ((ratio - 1.0f) / 1.65f);
    s.log_ampa_scale = bound(s.log_ampa_scale +
        change_per_48h * (elapsed_ms / 172800000.0f), -3.0f, 3.0f);
}

// AdEx equations with conductance inputs. Physical units: ms, mV, pA, pF, nS.
NS_HD inline void neuron_advance(const NeuronGene &g, NeuronState &s,
                                 float current_pa, const float *incoming_exc_ns,
                                 float incoming_inh_ns) {
    for (int branch = 0; branch < DENDRITES; ++branch)
        s.dendrite_exc_ns[branch] += incoming_exc_ns[branch];
    // Sustained depolarization immediately preceding a spike, as measured in
    // the cooperativity protocol of Sjostrom et al. (2001).
    float subthreshold = fmaxf(0.0f, fminf(s.voltage_mv, -50.0f) - g.rest_mv);
    s.recent_depol_mv += 0.1f * (subthreshold - s.recent_depol_mv);
    s.inh_ns += incoming_inh_ns;
    s.spike = 0;
    constexpr int substeps = 4;
    constexpr float h = DT_MS / substeps;
    for (int step = 0; step < substeps; ++step) {
        float v = s.voltage_mv;
        s.adaptation_pa += h * (g.adapt_ns * (v - g.rest_mv) - s.adaptation_pa) / g.adapt_tau_ms;
        float exponent = bound((v - g.threshold_mv) / g.slope_mv, -20.0f, 10.0f);
        float dv = g.leak_ns * (g.rest_mv - v)
            + g.leak_ns * g.slope_mv * expf(exponent)
            + soma_excitation(g, s) * (EXC_REVERSAL_MV - v)
            + s.inh_ns * (INH_REVERSAL_MV - v)
            + current_pa + s.afterdepol_pa - s.adaptation_pa;
        s.voltage_mv = v + h * dv / g.capacitance_pf;
        if (s.voltage_mv >= 0.0f) {
            s.spike = 1;
            ++s.total_spikes;
            s.voltage_mv = g.reset_mv;
            s.adaptation_pa += g.spike_adapt_pa;
            s.afterdepol_pa += g.afterdepol_spike_pa;
        }
        if (g.afterdepol_tau_ms > 0)
            s.afterdepol_pa *= expf(-h / g.afterdepol_tau_ms);
        for (int branch = 0; branch < DENDRITES; ++branch)
            s.dendrite_exc_ns[branch] *= expf(-h / EXC_DECAY_MS);
        s.inh_ns *= expf(-h / INH_DECAY_MS);
    }
}

NS_HD inline int site_count(const SynapseGene &g) {
    return g.release_sites ? g.release_sites : 14;
}

NS_HD inline uint32_t random_bits(SynapseState &s) {
    uint32_t x = s.rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s.rng_state = x;
    return x;
}

NS_HD inline float release_uniform(SynapseState &s) {
    return (random_bits(s) >> 8) * (1.0f / 16777216.0f);
}

NS_HD inline SynapseState initial_synapse_state(const SynapseGene &g,
                                                uint32_t seed = 1) {
    uint32_t x = seed ? seed : 1;
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    if (!x) x = 1;
    return {g.initial_weight_ns, 0, 0, 0, 0, 0, -1, 1, -1,
            1000000, 1000000,
            -1000000, (1u << site_count(g)) - 1u, x, 0};
}

// Fitted temporal envelope for dopamine-sensitive eligibility at striatal
// D1-like synapses in Shindou et al. (2019). The molecular state is unresolved.
NS_HD inline float delayed_da_eligibility(float age_ms) {
    if (age_ms < 0 || age_ms > 20000.0f) return 0;
    float x = age_ms / 2000.0f;
    return x * x * x * x * expf(4.0f * (1.0f - x));
}

// Fitted structural-plasticity window after spike pairing in Yagishita et al.
// (2014). This returns a relative susceptibility, not a molecular PKA model.
NS_HD inline float spine_growth_window(float age_ms) {
    if (age_ms < 0 || age_ms > 5000.0f) return 0;
    float x = age_ms / 1000.0f;
    return x * x * x * expf(3.0f * (1.0f - x));
}

// Resource depletion (Tsodyks and Markram 1997), plus the pulse-driven
// utilization increase measured in facilitating connections by Markram,
// Wang and Tsodyks (1998). Returns an excitatory conductance increment.
NS_HD inline float synapse_release(const SynapseGene &g, SynapseState &s, int64_t tick) {
    float recovery_probability = 0;
    if (s.last_arrival_tick > -1000000) {
        float elapsed = (tick - s.last_arrival_tick) * DT_MS;
        recovery_probability = 1.0f - expf(-elapsed / g.recovery_ms);
        if (g.facilitation_ms > 0)
            s.running_utilization *= expf(-elapsed / g.facilitation_ms);
    }
    s.running_utilization = g.facilitation_ms > 0
        ? s.running_utilization + g.utilization * (1.0f - s.running_utilization)
        : g.utilization;
    int released = 0;
    for (int site = 0; site < site_count(g); ++site) {
        uint32_t bit = 1u << site;
        if (!(s.ready_mask & bit) && release_uniform(s) < recovery_probability)
            s.ready_mask |= bit;
        if ((s.ready_mask & bit) && release_uniform(s) < s.running_utilization) {
            s.ready_mask &= ~bit;
            ++released;
        }
    }
    s.last_arrival_tick = tick;
    return s.weight_ns * released / site_count(g);
}

// Plasticity profile 1: cultured hippocampal pairs in Bi and Poo (1998).
// Profile 2: cortical L5 pairs in Sjostrom et al. (2001), which require
// preceding depolarization for LTP. Both depend on NMDA receptors.
NS_HD inline void synapse_plasticity_step(const SynapseGene &g, SynapseState &s,
                                         bool arrived, bool postsynaptic_spike,
                                         bool excitatory_pair,
                                         float preceding_depol_mv = 0,
                                         bool nmda_available = true,
                                         unsigned char mod_event = 0,
                                         bool l_type_available = true,
                                         bool calcium_permeable_ampa_available = true,
                                         bool pka_available = true) {
    if (!g.plasticity || !excitatory_pair) return;
    if (g.plasticity == 5) {
        if (s.structural_age_ms >= 0) {
            s.structural_age_ms += DT_MS;
            if (s.structural_age_ms > 5000.0f) s.structural_age_ms = -1;
        }
        s.pre_age_ms += DT_MS;
        if (arrived) s.pre_age_ms = 0;
        if (nmda_available && postsynaptic_spike && s.pre_age_ms <= 15.0f &&
            s.structural_age_ms < 0)
            s.structural_age_ms = 0;
        if ((mod_event & MOD_DA) && (g.mod_receptors & MOD_DA) &&
            pka_available)
            s.spine_volume = bound(s.spine_volume +
                0.025f * spine_growth_window(s.structural_age_ms), 1.0f, 2.0f);
        return;
    }
    if (g.plasticity == 4) {
        // D1 striatal profile: pre/post conjunction creates a delayed silent
        // dopamine eligibility state and a competing L-type-dependent LTD.
        if (s.da_age_ms >= 0) {
            s.da_age_ms += DT_MS;
            if (s.da_age_ms > 20000.0f) s.da_age_ms = -1;
        }
        s.pre_age_ms += DT_MS;
        if (arrived) s.pre_age_ms = 0;
        if (postsynaptic_spike && s.pre_age_ms <= 15.0f &&
            (s.da_age_ms < 0 || s.da_age_ms > 30.0f)) {
            s.da_age_ms = 0;
            if (l_type_available)
                s.weight_ns -= 0.005383f * g.initial_weight_ns;
        }
        if ((mod_event & MOD_DA) && (g.mod_receptors & MOD_DA) &&
            calcium_permeable_ampa_available) {
            s.weight_ns += 0.00453f * delayed_da_eligibility(s.da_age_ms)
                           * g.initial_weight_ns;
            s.da_age_ms = -1;
        }
        s.weight_ns = bound(s.weight_ns, 0.0f, 2.0f * g.initial_weight_ns);
        return;
    }
    if (!nmda_available) return;
    if (g.plasticity == 3) {
        // He et al. (2015): activity leaves distinct silent LTP/LTD traces.
        // In this visual-cortex profile, local beta2-adrenergic and 5-HT2C
        // receptor activation converts only the matching trace.
        s.eligible_ltp *= expf(-DT_MS / 7200.0f);
        s.eligible_ltd *= expf(-DT_MS / 3600.0f);
        s.pre_age_ms += DT_MS;
        s.post_age_ms += DT_MS;
        if (arrived) {
            if (s.post_age_ms <= 15.0f) s.eligible_ltd += 1.0f;
            s.pre_age_ms = 0;
        }
        if (postsynaptic_spike) {
            if (s.pre_age_ms <= 15.0f) s.eligible_ltp += 1.0f;
            s.post_age_ms = 0;
        }
        if ((mod_event & MOD_NE) && (g.mod_receptors & MOD_NE)) {
            s.weight_ns += 0.0048f * s.eligible_ltp * g.initial_weight_ns;
            s.eligible_ltp = 0;
        }
        if ((mod_event & MOD_5HT) && (g.mod_receptors & MOD_5HT)) {
            s.weight_ns -= 0.0075f * s.eligible_ltd * g.initial_weight_ns;
            s.eligible_ltd = 0;
        }
        s.weight_ns = bound(s.weight_ns, 0.0f, 2.0f * g.initial_weight_ns);
        return;
    }
    s.pre_trace *= expf(-DT_MS / LTP_TRACE_MS);
    s.post_trace *= expf(-DT_MS / LTD_TRACE_MS);
    s.pre_age_ms += DT_MS;
    s.post_age_ms += DT_MS;
    if (arrived) {
        if (s.post_age_ms <= (g.plasticity == 2 ? 50.0f : 20.0f))
            s.weight_ns *= 1.0f - 0.004f * s.post_trace;
        s.pre_trace += 1.0f;
        s.pre_age_ms = 0;
    }
    if (postsynaptic_spike) {
        if (s.pre_age_ms <= 20.0f &&
            (g.plasticity == 1 || preceding_depol_mv >= 2.3f))
            s.weight_ns += 0.016f * s.pre_trace * (2.0f * g.initial_weight_ns - s.weight_ns);
        s.post_trace += 1.0f;
        s.post_age_ms = 0;
    }
    s.weight_ns = bound(s.weight_ns, 0.0f, 2.0f * g.initial_weight_ns);
}
} // namespace sim
