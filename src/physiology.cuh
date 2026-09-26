#pragma once

#include <cmath>

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

// Parameter sets fitted to step-current recordings in Naud et al. (2008), Table 1.
// 0: regular-spiking pyramidal cell; 1: continuously non-adapting interneuron.
struct NeuronGene {
    float capacitance_pf, leak_ns, rest_mv, threshold_mv, slope_mv;
    float adapt_ns, adapt_tau_ms, spike_adapt_pa, reset_mv, bias_pa;
    unsigned char type, active;
};
struct NeuronState {
    float voltage_mv, adaptation_pa, inh_ns;
    float dendrite_exc_ns[DENDRITES];
    unsigned char spike;
};
struct SynapseGene {
    float initial_weight_ns, utilization, recovery_ms, facilitation_ms;
    short pre, post;
    unsigned char delay, active, plasticity, branch;
};
struct SynapseState {
    float weight_ns, available, running_utilization, pre_trace, post_trace;
    float pre_age_ms, post_age_ms;
    int last_arrival_tick;
    unsigned char arrived;
};

NS_HD inline float bound(float x, float low, float high) {
    return fminf(high, fmaxf(low, x));
}

NS_HD inline NeuronGene recorded_profile(unsigned char type) {
    if (type == 1) return {59, 2.9f, -62, -42, 3.0f, 1.8f, 16, 61, -54, 0, 1, 1};
    return {104, 4.3f, -65, -52, 0.8f, -0.8f, 88, 65, -53, 0, 0, 1};
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

// AdEx equations with conductance inputs. Physical units: ms, mV, pA, pF, nS.
NS_HD inline void neuron_advance(const NeuronGene &g, NeuronState &s,
                                 float current_pa, const float *incoming_exc_ns,
                                 float incoming_inh_ns) {
    for (int branch = 0; branch < DENDRITES; ++branch)
        s.dendrite_exc_ns[branch] += incoming_exc_ns[branch];
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
            + current_pa - s.adaptation_pa;
        s.voltage_mv = v + h * dv / g.capacitance_pf;
        if (s.voltage_mv >= 0.0f) {
            s.spike = 1;
            s.voltage_mv = g.reset_mv;
            s.adaptation_pa += g.spike_adapt_pa;
        }
        for (int branch = 0; branch < DENDRITES; ++branch)
            s.dendrite_exc_ns[branch] *= expf(-h / EXC_DECAY_MS);
        s.inh_ns *= expf(-h / INH_DECAY_MS);
    }
}

NS_HD inline SynapseState initial_synapse_state(const SynapseGene &g) {
    return {g.initial_weight_ns, 1.0f, 0, 0, 0, 1000000, 1000000, -1000000, 0};
}

// Resource depletion (Tsodyks and Markram 1997), plus the pulse-driven
// utilization increase measured in facilitating connections by Markram,
// Wang and Tsodyks (1998). Returns an excitatory conductance increment.
NS_HD inline float synapse_release(const SynapseGene &g, SynapseState &s, int tick) {
    if (s.last_arrival_tick > -1000000) {
        float elapsed = (tick - s.last_arrival_tick) * DT_MS;
        s.available = 1.0f - (1.0f - s.available) * expf(-elapsed / g.recovery_ms);
        if (g.facilitation_ms > 0)
            s.running_utilization *= expf(-elapsed / g.facilitation_ms);
    }
    s.running_utilization = g.facilitation_ms > 0
        ? s.running_utilization + g.utilization * (1.0f - s.running_utilization)
        : g.utilization;
    float fraction = s.running_utilization * s.available;
    s.available -= fraction;
    s.last_arrival_tick = tick;
    return s.weight_ns * fraction;
}

// Pair-timing rule restricted to excitatory -> excitatory synapses. Time
// constants and target direction follow Bi and Poo (1998); amplitudes are
// calibrated to their 60-pair ±10 ms protocol, not general cellular laws.
NS_HD inline void synapse_plasticity_step(const SynapseGene &g, SynapseState &s,
                                         bool arrived, bool postsynaptic_spike,
                                         bool excitatory_pair) {
    if (!g.plasticity || !excitatory_pair) return;
    s.pre_trace *= expf(-DT_MS / LTP_TRACE_MS);
    s.post_trace *= expf(-DT_MS / LTD_TRACE_MS);
    s.pre_age_ms += DT_MS;
    s.post_age_ms += DT_MS;
    if (arrived) {
        if (s.post_age_ms <= 20.0f)
            s.weight_ns *= 1.0f - 0.004f * s.post_trace;
        s.pre_trace += 1.0f;
        s.pre_age_ms = 0;
    }
    if (postsynaptic_spike) {
        if (s.pre_age_ms <= 20.0f)
            s.weight_ns += 0.016f * s.pre_trace * (2.0f * g.initial_weight_ns - s.weight_ns);
        s.post_trace += 1.0f;
        s.post_age_ms = 0;
    }
    s.weight_ns = bound(s.weight_ns, 0.0f, 2.0f * g.initial_weight_ns);
}
} // namespace sim
