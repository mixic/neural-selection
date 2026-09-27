#pragma once

#include <cmath>
#include <cstdint>
#include "lp_cell.cuh"
#include "prinz_cell.cuh"

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
    float adapt_ns, adapt_tau_ms, spike_adapt_pa, reset_mv;
    float afterdepol_spike_pa, afterdepol_tau_ms;
    unsigned char type, active, modulator;
    // Fast transmitter identity is independent of membrane dynamics.
    unsigned char inhibitory_transmitter = 0;
    LPConductance lp_scale;
    PrinzConductance prinz_g;
    float prinz_ambient_leak_ns;
    // A sensory channel can be read by multiple independently spiking cells.
    // -1 means no direct receptor input; channel numbers are world-defined.
    signed char sensory_channel = -1;
    float receptor_baseline_hz = 0, receptor_gain_hz = 0;
    float receptor_event_ns = 0, receptor_decay_ms = 3;
};
struct LFSCell {
    float voltage_mv, threshold_mv;
    float activated[3], open[3];
};
struct NeuronState {
    float voltage_mv, adaptation_pa, afterdepol_pa, inh_ns, recent_depol_mv, log_ampa_scale;
    float dendrite_exc_ns[DENDRITES];
    union { LPCell lp; PrinzCell prinz; LFSCell lfs; };
    int total_spikes;
    unsigned char spike;
    float receptor_ns;
    uint32_t receptor_rng;
};
struct SynapseGene {
    float initial_weight_ns, utilization, recovery_ms, facilitation_ms;
    int pre, post;
    unsigned char delay, active, plasticity, branch, release_sites, mod_receptors;
    // Optional slow component of observed use-dependent depression. Zero is off.
    float slow_recovery_ms, slow_depletion;
};
struct SynapseState {
    float weight_ns, running_utilization, pre_trace, post_trace;
    float eligible_ltp, eligible_ltd, da_age_ms;
    float spine_volume, structural_age_ms;
    float pre_age_ms, post_age_ms;
    int64_t last_arrival_tick;
    uint64_t ready_mask;
    uint32_t rng_state;
    unsigned char arrived;
    float slow_resource;
};

NS_HD inline float bound(float x, float low, float high) {
    return fminf(high, fmaxf(low, x));
}

NS_HD inline NeuronGene recorded_profile(unsigned char type) {
    if (type == 1) {
        NeuronGene g{59, 2.9f, -62, -42, 3.0f, 1.8f, 16, 61, -54, 0, 0, 1, 1, 0};
        g.inhibitory_transmitter = 1;
        return g;
    }
    return {104, 4.3f, -65, -52, 0.8f, -0.8f, 88, 65, -53, 0, 0, 0, 1, 0};
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

NS_HD inline NeuronGene lobster_lp_profile() {
    NeuronGene g = recorded_profile(1);
    g.type = 2;
    g.lp_scale = {1, 1, 1, 1};
    return g;
}

NS_HD inline NeuronGene crab_pacemaker_profile() {
    NeuronGene g = recorded_profile(1);
    g.type = 3;
    g.prinz_g = prinz_pacemaker_conductance();
    return g;
}

// Effective olfactory receptor neuron. Receptor event statistics are fitted
// against measured spontaneous and odor-driven spike rates in a separate
// protocol; event conductance is not a measured single-channel value.
NS_HD inline NeuronGene orn_receptor_profile() {
    NeuronGene g = recorded_profile(0);
    g.type = 5;
    g.capacitance_pf = 10.0f;
    g.leak_ns = 0.5f;
    g.rest_mv = -60.0f;
    g.threshold_mv = -40.0f;
    g.reset_mv = -60.0f;
    g.adapt_ns = 0;
    g.spike_adapt_pa = 0;
    g.receptor_baseline_hz = 7.0f;
    g.receptor_gain_hz = 43.0f;
    g.receptor_event_ns = 2.0f;
    g.receptor_decay_ms = 3.0f;
    return g;
}

// Effective PN relay profile for the ORN-to-PN protocol. Capacitance and leak
// are fitted to the published unitary EPSP and the author's passive-model
// decay; they are not measurements of a physical PN compartment.
NS_HD inline NeuronGene pn_relay_profile() {
    NeuronGene g = recorded_profile(0);
    g.type = 4;
    g.capacitance_pf = 8.7f;
    g.leak_ns = 8.7f / 12.0f;
    g.rest_mv = -60.0f;
    g.threshold_mv = -40.0f;
    g.reset_mv = -60.0f;
    g.adapt_ns = 0;
    g.spike_adapt_pa = 0;
    return g;
}

// Aplysia LFS siphon motor cell: measured passive properties from Lieb and
// Frost (1997), Table 1. The threshold reset and spike undershoot below use
// Table 2. Their high-current voltage-dependent shunts are not included.
NS_HD inline NeuronGene aplysia_lfs_profile() {
    NeuronGene g = recorded_profile(0);
    g.type = 6;
    g.capacitance_pf = 1200.0f;
    g.leak_ns = 1000.0f / 65.6f;
    g.rest_mv = -45.1f;
    g.threshold_mv = -51.8f;
    return g;
}

NS_HD inline bool excitatory_type(unsigned char type) {
    return type == 0 || type == 4 || type == 5;
}

NS_HD inline NeuronGene initial_profile(unsigned char type) {
    return type == 0 ? cortical_l5_profile() :
           (type == 1 ? recorded_profile(1) :
           (type == 2 ? lobster_lp_profile() :
            (type == 3 ? crab_pacemaker_profile() :
             (type == 4 ? pn_relay_profile() :
              (type == 5 ? orn_receptor_profile() : aplysia_lfs_profile())))));
}


NS_HD inline NeuronState initial_neuron_state(const NeuronGene &g,
                                              uint32_t seed = 1) {
    NeuronState state{};
    state.voltage_mv = g.rest_mv;
    uint32_t x = seed ? seed : 1;
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    state.receptor_rng = x ? x : 1;
    if (g.type == 2) {
        state.lp = initial_lp_cell();
        state.voltage_mv = state.lp.x[15] - 9.342518882f;
    }
    if (g.type == 3) {
        state.prinz = initial_prinz_cell();
        state.voltage_mv = state.prinz.v;
    }
    if (g.type == 6) {
        state.lfs = {};
        state.lfs.voltage_mv = g.rest_mv;
        state.lfs.threshold_mv = g.threshold_mv;
    }
    return state;
}

// Independently sampled receptor events produce a conductance that enters
// the ordinary membrane equation. Spikes still require voltage integration.
NS_HD inline float receptor_current(const NeuronGene &g, NeuronState &s,
                                    float sensory_level) {
    if (g.sensory_channel < 0 || g.receptor_event_ns <= 0) return 0;
    float rate_hz = bound(g.receptor_baseline_hz +
                          g.receptor_gain_hz * sensory_level, 0, 1000);
    s.receptor_rng ^= s.receptor_rng << 13;
    s.receptor_rng ^= s.receptor_rng >> 17;
    s.receptor_rng ^= s.receptor_rng << 5;
    float u = (s.receptor_rng >> 8) * (1.0f / 16777216.0f);
    s.receptor_ns *= expf(-DT_MS / g.receptor_decay_ms);
    if (u < rate_hz * (DT_MS / 1000.0f))
        s.receptor_ns += g.receptor_event_ns;
    return s.receptor_ns * (EXC_REVERSAL_MV - s.voltage_mv);
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

// Partial port of the experimentally fitted LFS integrate-and-fire cell in
// Lieb and Frost (1997). Each step is one millisecond. Conductance weights
// from their Table 2 are converted from microSiemens to nanoSiemens.
NS_HD inline void lfs_advance(const NeuronGene &g, NeuronState &s,
                              float current_pa, float excitation_ns,
                              float inhibition_ns) {
    constexpr float weight_ns[3] = {250.0f, 10.0f, 4.0f};
    constexpr float open_ms[3] = {10.0f, 40.0f, 750.0f};
    constexpr float close_ms[3] = {50.0f, 750.0f, 2000.0f};
    LFSCell &cell = s.lfs;
    cell.threshold_mv = g.threshold_mv +
        (cell.threshold_mv - g.threshold_mv) * expf(-1.0f / 10.0f);
    float undershoot_ns = 0;
    for (int j = 0; j < 3; ++j) {
        cell.open[j] += cell.activated[j] / open_ms[j] -
                        cell.open[j] / close_ms[j];
        cell.activated[j] *= expf(-1.0f / open_ms[j]);
        undershoot_ns += weight_ns[j] * cell.open[j];
    }
    float voltage = cell.voltage_mv;
    float current = g.leak_ns * (g.rest_mv - voltage) +
        excitation_ns * (10.0f - voltage) +
        inhibition_ns * (-60.0f - voltage) +
        undershoot_ns * (-80.0f - voltage) + current_pa;
    cell.voltage_mv = voltage + current / g.capacitance_pf;
    s.spike = cell.voltage_mv >= cell.threshold_mv;
    if (s.spike) {
        cell.threshold_mv = 200.0f;
        for (int j = 0; j < 3; ++j) cell.activated[j] += 1.0f;
        ++s.total_spikes;
    }
    s.voltage_mv = cell.voltage_mv;
}

// AdEx equations with conductance inputs. Physical units: ms, mV, pA, pF, nS.
NS_HD inline void neuron_advance(const NeuronGene &g, NeuronState &s,
                                 float current_pa, const float *incoming_exc_ns,
                                 float incoming_inh_ns,
                                 float clamp_leak_ns = 0.0f,
                                 float clamp_reversal_mv = -68.0f) {
    for (int branch = 0; branch < DENDRITES; ++branch)
        s.dendrite_exc_ns[branch] += incoming_exc_ns[branch];
    // Sustained depolarization immediately preceding a spike, as measured in
    // the cooperativity protocol of Sjostrom et al. (2001).
    float subthreshold = fmaxf(0.0f, fminf(s.voltage_mv, -50.0f) - g.rest_mv);
    s.recent_depol_mv += 0.1f * (subthreshold - s.recent_depol_mv);
    s.inh_ns += incoming_inh_ns;
    s.spike = 0;
    if (g.type == 6) {
        float excitation = 0;
        for (int branch = 0; branch < DENDRITES; ++branch)
            excitation += s.dendrite_exc_ns[branch];
        lfs_advance(g, s, current_pa, excitation, s.inh_ns);
        for (int branch = 0; branch < DENDRITES; ++branch)
            s.dendrite_exc_ns[branch] *= expf(-DT_MS / EXC_DECAY_MS);
        s.inh_ns *= expf(-DT_MS / INH_DECAY_MS);
        return;
    }
    if (g.type == 2) {
        float old_axon = s.lp.x[0];
        float excitation = 0;
        for (int branch = 0; branch < DENDRITES; ++branch)
            excitation += s.dendrite_exc_ns[branch];
        float external_pa = current_pa +
            excitation * (EXC_REVERSAL_MV - s.voltage_mv) +
            s.inh_ns * (INH_REVERSAL_MV - s.voltage_mv) -
            clamp_leak_ns * (s.voltage_mv - clamp_reversal_mv);
        lp_advance(s.lp, external_pa / 1000.0f, g.lp_scale);
        s.voltage_mv = s.lp.x[15] - 9.342518882f;
        s.spike = old_axon < 0 && s.lp.x[0] >= 0;
        s.total_spikes += s.spike;
        for (int branch = 0; branch < DENDRITES; ++branch)
            s.dendrite_exc_ns[branch] *= expf(-DT_MS / EXC_DECAY_MS);
        s.inh_ns *= expf(-DT_MS / INH_DECAY_MS);
        return;
    }
    if (g.type == 3) {
        float excitation = 0;
        for (int branch = 0; branch < DENDRITES; ++branch)
            excitation += s.dendrite_exc_ns[branch];
        const float external_pa = current_pa +
            excitation*(EXC_REVERSAL_MV-s.voltage_mv) +
            s.inh_ns*(INH_REVERSAL_MV-s.voltage_mv);
        s.spike = prinz_advance(s.prinz, g.prinz_g, external_pa,
                                g.prinz_ambient_leak_ns + clamp_leak_ns,
                                clamp_reversal_mv);
        s.total_spikes += s.spike;
        s.voltage_mv = s.prinz.v;
        for (int branch = 0; branch < DENDRITES; ++branch)
            s.dendrite_exc_ns[branch] *= expf(-DT_MS/EXC_DECAY_MS);
        s.inh_ns *= expf(-DT_MS/INH_DECAY_MS);
        return;
    }
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
            + current_pa + s.afterdepol_pa - s.adaptation_pa
            - clamp_leak_ns * (v - clamp_reversal_mv);
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
    return g.release_sites ? (g.release_sites > 64 ? 64 : g.release_sites) : 14;
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
            -1000000, site_count(g) == 64 ? ~uint64_t(0) :
            (uint64_t(1) << site_count(g)) - 1, x, 0, 1};
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
        if (g.slow_recovery_ms > 0)
            s.slow_resource = 1.0f -
                (1.0f - s.slow_resource) * expf(-elapsed / g.slow_recovery_ms);
    }
    s.running_utilization = g.facilitation_ms > 0
        ? s.running_utilization + g.utilization * (1.0f - s.running_utilization)
        : g.utilization;
    int released = 0;
    for (int site = 0; site < site_count(g); ++site) {
        uint64_t bit = uint64_t(1) << site;
        if (!(s.ready_mask & bit) && release_uniform(s) < recovery_probability)
            s.ready_mask |= bit;
        if ((s.ready_mask & bit) && release_uniform(s) < s.running_utilization) {
            s.ready_mask &= ~bit;
            ++released;
        }
    }
    s.last_arrival_tick = tick;
    float output = s.weight_ns * released / site_count(g) * s.slow_resource;
    if (g.slow_recovery_ms > 0)
        s.slow_resource *= 1.0f - g.slow_depletion;
    return output;
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
