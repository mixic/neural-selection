#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <png.h>
#include "physiology.cuh"
#include "live_view.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace sim {
constexpr int P = 32768;
constexpr int NEURON_CAPACITY = 4194304;
constexpr int SYNAPSE_CAPACITY = 8388608;
constexpr int INITIAL_HIDDEN = 48;
constexpr int INITIAL_SYNAPSES = 180;
constexpr int LIFE_RECORD_CAPACITY = P * 16;
constexpr int LIFE_SAMPLE_INTERVAL = 10;
constexpr int S = 19;
constexpr int M = 5;
constexpr int CONTACT_FOOD_START = 16;
constexpr int W = 128;
constexpr int H = 128;
constexpr int D = 8;
constexpr int HISTORY = D + 1;
constexpr int TICKS = 16;
constexpr int NEURAL_BLOCK = 512;
constexpr int BLOCK_BRAIN_NEURONS = 80;
constexpr int BLOCK_BRAIN_SYNAPSES = 192;
constexpr float BODY_UPKEEP = 0.003f;
constexpr float NEURON_UPKEEP = 0.00002f;
constexpr float BITE_DAMAGE = 2.0f;
constexpr float BITE_ENERGY_COST = 0.02f;
constexpr float MEAT_ENERGY_FRACTION = 0.45f;
// Approximate the ~200 ms contraction decay measured in stimulated insect
// muscle. This is body mechanics, not a somatic neural current.
constexpr float MUSCLE_DECAY_PER_STEP = 0.92311635f; // exp(-16 / 200)
constexpr float MUSCLE_TWITCH_PER_SPIKE = 0.30f;
constexpr float PI = 3.14159265358979323846f;

// Observer-only counters. They never enter sensory input or neural updates.
// Age bins: 0-99, 100-299, and 300+ world steps.
struct Behavior {
    int opportunities[3][2][3];
    int accepted[3][2][3];
    int empty_steps[3][2], empty_eat[3][2];
    float approach_progress[3][3];
    float approach_travel[3][3];
    int first_b_age, offspring, death_cause;
};

struct Animal {
    float x, y, heading, energy, health, fullness, pain;
    float taste[4];
    float muscle[3];
    unsigned long long rng;
    int id, parent_id, generation, age, alive;
    int neuron_head, synapse_head, neuron_count, synapse_count;
    int ports[S + M];
    Behavior behavior;
};
struct LifeRecord {
    Animal body;
    int64_t death_step;
    int active_neurons, active_synapses, modulatory_neurons, plastic_synapses;
    float mean_abs_weight_change;
};
struct BiteEvent {
    int64_t step;
    int attacker_id, prey_id;
    float x, y;
};
struct Device {
    int initial_count;
    int trace_animal;
    bool silence_senses;
    bool silence_contact;
    bool instant_motors;
    Animal *animals;
    NeuronGene *neurons;
    SynapseGene *synapses;
    NeuronState *neuron_state;
    SynapseState *synapse_state;
    float *pending_exc, *pending_inh, *senses;
    // Alternating tick slots keep the previous modulation event readable
    // while the fused synapse pass writes the current tick's event.
    unsigned long long *mod_event_tick;
    unsigned char *spike_history;
    int *neuron_owner, *synapse_owner, *neuron_next, *synapse_next;
    int *occupied_neurons, *occupied_synapses;
    int *occupied_neuron_count, *occupied_synapse_count;
    int *block_animals, *block_animal_count;
    int *neuron_port, *neuron_clone;
    uint32_t *neuron_used, *synapse_used;
    int *active_neuron_counts, *active_synapse_counts;
    unsigned char *bite_intent;
    int *motors, *food, *kind, *cell_head, *cell_next;
    int *trace_exc_arrivals, *trace_exc_releases, *trace_inh_arrivals;
    float *trace_exc_ns, *trace_inh_ns, *trace_peak_mv;
    int *founder_motor_steps, *founder_observed_steps;
    int *births, *deaths, *predation_deaths, *bites, *next_id;
    LifeRecord *life_records;
    int *life_count;
    BiteEvent *bite_events;
    int *bite_event_count;
};

struct OwnedSlot {
    const int *owner;
    __device__ bool operator()(int slot) const { return owner[slot] >= 0; }
};

struct BlockAnimal {
    const Animal *animals;
    __device__ bool operator()(int slot) const {
        return animals[slot].alive == 1 &&
               animals[slot].neuron_count <= BLOCK_BRAIN_NEURONS &&
               animals[slot].synapse_count <= BLOCK_BRAIN_SYNAPSES;
    }
};

struct FallbackOwnedSlot {
    const int *owner;
    const Animal *animals;
    __device__ bool operator()(int slot) const {
        int a = owner[slot];
        return a >= 0 && animals[a].alive == 1 &&
               (animals[a].neuron_count > BLOCK_BRAIN_NEURONS ||
                animals[a].synapse_count > BLOCK_BRAIN_SYNAPSES);
    }
};

struct JsonBuilder {
    std::string value;
    explicit JsonBuilder(size_t capacity) { value.reserve(capacity); }
    JsonBuilder &operator<<(const char *text) { value += text; return *this; }
    JsonBuilder &operator<<(char character) { value += character; return *this; }
    template <typename T, std::enable_if_t<std::is_integral_v<T> &&
                                            !std::is_same_v<T, char>, int> = 0>
    JsonBuilder &operator<<(T number) {
        char buffer[32];
        auto result = std::to_chars(buffer, buffer + sizeof(buffer), number);
        value.append(buffer, result.ptr);
        return *this;
    }
    JsonBuilder &operator<<(float number) { return append_float(number); }
    JsonBuilder &operator<<(double number) { return append_float(number); }
    JsonBuilder &append_float(double number) {
        char buffer[64];
        auto result = std::to_chars(buffer, buffer + sizeof(buffer), number,
                                    std::chars_format::fixed, 2);
        value.append(buffer, result.ptr);
        return *this;
    }
    std::string str() { return std::move(value); }
};

__host__ __device__ inline float clamp(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}
__host__ __device__ inline unsigned long long random64(unsigned long long &s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 2685821657736338717ULL;
}
__host__ __device__ inline float uniform(unsigned long long &s) {
    return (float)((random64(s) >> 40) * (1.0 / 16777216.0));
}
__host__ __device__ inline int randint(unsigned long long &s, int size) {
    return (int)(uniform(s) * size);
}
__host__ __device__ inline unsigned long long seed_for(unsigned long long seed, unsigned long long i) {
    unsigned long long x = seed + 0x9e3779b97f4a7c15ULL * (i + 1);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return (x ^ (x >> 31)) | 1ULL;
}
__device__ inline int cell(float x, float y) {
    int ix = (int)floorf(clamp(x, 0.0f, W - 0.001f));
    int iy = (int)floorf(clamp(y, 0.0f, H - 0.001f));
    return iy * W + ix;
}
__device__ inline bool harvest(int *stock) {
    int old = atomicAdd(stock, 0);
    while (old > 0) {
        int observed = atomicCAS(stock, old, old - 1);
        if (observed == old) return true;
        old = observed;
    }
    return false;
}

// Atomic bitmap allocation lets a genome grow until the shared pool is full.
// No animal owns a fixed range of neuron or synapse slots.
__device__ int claim_slot(uint32_t *used, int capacity, int hint) {
    int words = (capacity + 31) / 32;
    int start = hint % words;
    for (int offset = 0; offset < words; ++offset) {
        int word = (start + offset) % words;
        uint32_t occupied = atomicAdd(&used[word], 0u);
        while (~occupied) {
            int bit = __ffs(~occupied) - 1;
            uint32_t mask = 1u << bit;
            uint32_t observed = atomicCAS(&used[word], occupied,
                                          occupied | mask);
            if (observed == occupied) {
                int slot = word * 32 + bit;
                if (slot < capacity) return slot;
                atomicAnd(&used[word], ~mask);
                return -1;
            }
            occupied = observed;
        }
    }
    return -1;
}

__device__ void release_slot(uint32_t *used, int slot) {
    atomicAnd(&used[slot / 32], ~(1u << (slot % 32)));
}

__device__ int neuron_at(Device d, const Animal &body, int ordinal) {
    int n = body.neuron_head;
    for (int j = 0; j < ordinal && n >= 0; ++j) n = d.neuron_next[n];
    return n;
}

__device__ int synapse_at(Device d, const Animal &body, int ordinal) {
    int e = body.synapse_head;
    for (int j = 0; j < ordinal && e >= 0; ++j) e = d.synapse_next[e];
    return e;
}

__device__ int random_hidden_neuron(Device d, const Animal &body,
                                    unsigned long long &rng) {
    int hidden_count = body.neuron_count - (S + M);
    if (hidden_count <= 0) return -1;
    int ordinal = randint(rng, hidden_count);
    for (int n = body.neuron_head; n >= 0; n = d.neuron_next[n])
        if (d.neuron_port[n] < 0 && ordinal-- == 0) return n;
    return -1;
}

__device__ void initialize_neuron(Device d, int n, int owner, int port,
                                  const NeuronGene &gene, int next,
                                  uint32_t state_seed) {
    d.neuron_owner[n] = owner;
    d.neuron_next[n] = next;
    d.neuron_port[n] = port;
    d.neurons[n] = gene;
    d.neuron_state[n] = initial_neuron_state(gene, state_seed);
    for (int branch = 0; branch < DENDRITES; ++branch)
        d.pending_exc[n * DENDRITES + branch] = 0;
    d.pending_inh[n] = 0;
    for (int k = 0; k < 6; ++k) d.mod_event_tick[n * 6 + k] = 0;
    for (int delay = 0; delay < HISTORY; ++delay)
        d.spike_history[delay * NEURON_CAPACITY + n] = 0;
}

__device__ void initialize_synapse(Device d, int e, int owner,
                                   const SynapseGene &gene, int next,
                                   unsigned long long &rng) {
    d.synapse_owner[e] = owner;
    d.synapse_next[e] = next;
    d.synapses[e] = gene;
    d.synapse_state[e] = initial_synapse_state(gene, (uint32_t)random64(rng));
}

__device__ void release_genome(Device d, Animal &body) {
    for (int e = body.synapse_head; e >= 0;) {
        int next = d.synapse_next[e];
        d.synapse_owner[e] = -1;
        release_slot(d.synapse_used, e);
        e = next;
    }
    for (int n = body.neuron_head; n >= 0;) {
        int next = d.neuron_next[n];
        d.neuron_owner[n] = -1;
        release_slot(d.neuron_used, n);
        n = next;
    }
    body.neuron_head = body.synapse_head = -1;
    body.neuron_count = body.synapse_count = 0;
    for (int p = 0; p < S + M; ++p) body.ports[p] = -1;
}

__global__ void init_world(Device d, unsigned long long seed) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W * H) return;
    auto rng = seed_for(seed, c + P);
    d.kind[c] = randint(rng, 3);
    d.food[c] = uniform(rng) < 0.28f ? 1 : 0;
}

__device__ void random_synapse(SynapseGene &g, Device d, int animal,
                               unsigned long long &rng) {
    Animal body = d.animals[animal];
    int active_sources = 0;
    float target_weight = 0;
    for (int n = body.neuron_head; n >= 0; n = d.neuron_next[n]) {
        NeuronGene gene = d.neurons[n];
        if (!gene.active) continue;
        ++active_sources;
        int port = d.neuron_port[n];
        if (port < 0 || port >= S) target_weight += gene.target_affinity;
    }
    g.pre = body.ports[0];
    int source_choice = randint(rng, active_sources);
    for (int n = body.neuron_head; n >= 0; n = d.neuron_next[n])
        if (d.neurons[n].active && source_choice-- == 0) {
            g.pre = n;
            break;
        }
    g.post = body.ports[S + randint(rng, M)];
    float target_choice = uniform(rng) * target_weight;
    for (int n = body.neuron_head; n >= 0; n = d.neuron_next[n]) {
        NeuronGene gene = d.neurons[n];
        int port = d.neuron_port[n];
        if (!gene.active || (port >= 0 && port < S)) continue;
        target_choice -= gene.target_affinity;
        if (target_choice <= 0) {
            g.post = n;
            break;
        }
    }
    g.delay = (unsigned char)(1 + randint(rng, D));
    g.branch = (unsigned char)randint(rng, DENDRITES);
    g.initial_weight_ns = 0.5f + uniform(rng) * 3.5f;
    g.utilization = 0.55f;
    g.recovery_ms = 450.0f;
    g.facilitation_ms = 0;
    g.release_sites = 14;
    if (d.neurons[g.pre].type == 5) {
        // One experimentally fitted ORN-to-PN release phenotype; the target
        // and all of these heritable parameters remain mutable.
        g.initial_weight_ns = 29.0f / (60.0f * 0.75f);
        g.utilization = 0.75f;
        g.recovery_ms = 50.0f;
        g.release_sites = 50;
        g.slow_recovery_ms = 7500.0f;
        g.slow_depletion = 0.023f;
    }
    if (d.neurons[g.pre].type == 0 && d.neurons[g.post].type == 1) {
        // One fitted pyramidal-to-interneuron profile in Markram et al. (1998).
        g.utilization = 0.1f;
        g.recovery_ms = 30.0f;
        g.facilitation_ms = 1700.0f;
    }
    float plasticity_draw = uniform(rng);
    g.plasticity = plasticity_draw < 0.05f ? 5 :
                   (plasticity_draw < 0.10f ? 4 :
                   (plasticity_draw < 0.20f ? 3 : 2));
    g.mod_receptors = g.plasticity == 3 ? MOD_NE | MOD_5HT :
                      (g.plasticity >= 4 ? MOD_DA : 0);
    g.active = 1;
}

__global__ void init_animals(Device d, int initial, unsigned long long seed) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P) return;
    auto rng = seed_for(seed, a);
    Animal body{};
    body.neuron_head = body.synapse_head = -1;
    for (int p = 0; p < S + M; ++p) body.ports[p] = -1;
    body.rng = rng;
    if (a >= initial) { d.animals[a] = body; return; }
    body.x = uniform(rng) * (W - 1);
    body.y = uniform(rng) * (H - 1);
    body.heading = uniform(rng) * 2.0f * PI;
    body.energy = 14.0f;
    body.health = 10.0f;
    body.id = a;
    body.parent_id = -1;
    body.alive = 1;
    for (int local = 0; local < S + M + INITIAL_HIDDEN; ++local) {
        int n = claim_slot(d.neuron_used, NEURON_CAPACITY,
                           a * (S + M + INITIAL_HIDDEN) + local);
        if (n < 0) {
            release_genome(d, body);
            body.alive = 0;
            d.animals[a] = body;
            return;
        }
        int port = local < S + M ? local : -1;
        float cell_draw = uniform(rng);
        unsigned char type;
        if (port >= 0 && port < 6) type = 5;
        else if (port >= 0 && port < S) type = 0;
        else if (port >= S)
            type = cell_draw < 0.45f ? 0 :
                   (cell_draw < 0.55f ? 1 :
                    (cell_draw < 0.80f ? 4 : 6));
        else
            type = cell_draw < 0.35f ? 0 :
                   (cell_draw < 0.65f ? 1 :
                    (cell_draw < 0.67f ? 2 :
                     (cell_draw < 0.69f ? 3 :
                      (cell_draw < 0.80f ? 4 : 5))));
        NeuronGene g = initial_profile(type);
        if (port < 0 || port >= S) {
            float affinity_draw = uniform(rng);
            g.target_affinity = affinity_draw < 0.015f ? 100.0f :
                (affinity_draw < 0.075f ? 20.0f : 1.0f);
        }
        if (type == 5) g.sensory_channel =
            (signed char)(port >= 0 ? port : randint(rng, 6));
        else if (port >= 0 && port < S) g.sensory_channel =
            (signed char)port;
        if (port < 0 && type != 5) {
            float u = uniform(rng);
            g.modulator = u < 0.04f ? 1 : (u < 0.08f ? 2 :
                          (u < 0.12f ? 3 : 0));
        }
        g.active = port >= 0 || type == 5 || uniform(rng) < 0.5f;
        initialize_neuron(d, n, a, port, g, body.neuron_head,
                          (uint32_t)random64(rng));
        body.neuron_head = n;
        ++body.neuron_count;
        if (port >= 0) body.ports[port] = n;
    }
    d.animals[a] = body;
    for (int local = 0; local < INITIAL_SYNAPSES; ++local) {
        int e = claim_slot(d.synapse_used, SYNAPSE_CAPACITY,
                           a * INITIAL_SYNAPSES + local);
        if (e < 0) {
            release_genome(d, body);
            body.alive = 0;
            d.animals[a] = body;
            return;
        }
        SynapseGene g{};
        random_synapse(g, d, a, rng);
        initialize_synapse(d, e, a, g, body.synapse_head, rng);
        body.synapse_head = e;
        ++body.synapse_count;
    }
    for (int s = 0; s < S; ++s) d.senses[a * S + s] = 0;
    for (int m = 0; m < M; ++m) d.motors[a * M + m] = 0;
    body.rng = rng;
    d.animals[a] = body;
}

__global__ void index_animals(Device d) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P || d.animals[a].alive != 1) return;
    int c = cell(d.animals[a].x, d.animals[a].y);
    d.cell_next[a] = atomicExch(&d.cell_head[c], a);
}

__global__ void update_senses(Device d) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P || d.animals[a].alive != 1) return;
    Animal b = d.animals[a];
    for (int s = 0; s < 6; ++s) {
        float angle = b.heading + (s % 3 - 1) * 0.7f;
        float level = 0;
        for (int range = 1; range <= 4; ++range) {
            int c = cell(b.x + cosf(angle) * range, b.y + sinf(angle) * range);
            if (d.kind[c] == s / 3 && d.food[c] > 0) level += 0.25f;
        }
        d.senses[a * S + s] = clamp(level, 0, 1);
    }
    d.senses[a * S + 6] = clamp((16.0f - b.energy) / 16.0f, 0, 1);
    d.senses[a * S + 7] = clamp(b.fullness / 8.0f, 0, 1);
    d.senses[a * S + 8] = clamp(b.pain / 3.0f, 0, 1);
    for (int k = 0; k < 3; ++k)
        d.senses[a * S + 9 + k] = clamp(b.taste[k], 0, 1);
    for (int direction = 0; direction < 3; ++direction) {
        float angle = b.heading + (direction - 1) * 0.7f;
        float level = 0;
        for (int range = 1; range <= 4; ++range) {
            int c = cell(b.x + cosf(angle) * range,
                         b.y + sinf(angle) * range);
            for (int other = d.cell_head[c]; other >= 0;
                 other = d.cell_next[other])
                if (other != a && d.animals[other].alive == 1)
                    level += 0.25f;
        }
        d.senses[a * S + 12 + direction] = clamp(level, 0, 1);
    }
    d.senses[a * S + 15] = clamp(b.taste[3], 0, 1);
    // Contact chemoreceptors report identity, without an assigned food value.
    // They update after body_step and can inform the next neural decision.
    int contact_cell = cell(b.x, b.y);
    for (int kind = 0; kind < 3; ++kind)
        d.senses[a * S + CONTACT_FOOD_START + kind] =
            d.food[contact_cell] > 0 && d.kind[contact_cell] == kind ? 1.0f : 0.0f;
}

__device__ inline void deliver_synapse(Device d, int a, const SynapseGene &g,
                                       SynapseState &state, int64_t tick) {
    state.arrived = tick >= g.delay &&
        d.spike_history[((tick - g.delay) % HISTORY) * NEURON_CAPACITY + g.pre];
    if (state.arrived) {
        NeuronGene source = d.neurons[g.pre];
        float released = !source.inhibitory_transmitter || source.modulator
            ? synapse_release(g, state, tick) : state.weight_ns;
        int motor = d.neuron_port[g.post] - S;
        bool traced_motor = a == d.trace_animal && motor >= 0 && motor < M;
        if (source.modulator) {
            if (released > 0)
                atomicMax(&d.mod_event_tick[g.post * 6 +
                          (source.modulator - 1) * 2 + (tick & 1)],
                          static_cast<unsigned long long>(tick + 1));
        } else if (!source.inhibitory_transmitter) {
            if (traced_motor) {
                atomicAdd(&d.trace_exc_arrivals[motor], 1);
                if (released > 0) atomicAdd(&d.trace_exc_releases[motor], 1);
                atomicAdd(&d.trace_exc_ns[motor], released);
            }
            float postsynaptic_scale = expf(
                d.neuron_state[g.post].log_ampa_scale);
            atomicAdd(&d.pending_exc[g.post * DENDRITES + g.branch],
                      released * postsynaptic_scale);
        }
        else {
            if (traced_motor) {
                atomicAdd(&d.trace_inh_arrivals[motor], 1);
                atomicAdd(&d.trace_inh_ns[motor], released);
            }
            atomicAdd(&d.pending_inh[g.post], released);
        }
    }
}

__global__ void synapse_delivery(Device d, int64_t tick, int occupied_count) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= occupied_count) return;
    int i = d.occupied_synapses[ordinal];
    int a = d.synapse_owner[i];
    if (a < 0 || d.animals[a].alive != 1) return;
    SynapseGene g = d.synapses[i];
    if (!g.active || d.neuron_owner[g.pre] != a ||
        d.neuron_owner[g.post] != a || !d.neurons[g.pre].active ||
        !d.neurons[g.post].active) return;
    SynapseState state = d.synapse_state[i];
    deliver_synapse(d, a, g, state, tick);
    d.synapse_state[i] = state;
}

__global__ void neuron_step(Device d, int64_t tick, int occupied_count) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= occupied_count) return;
    int i = d.occupied_neurons[ordinal];
    int a = d.neuron_owner[i];
    if (a < 0) return;
    if (d.animals[a].alive != 1) return;
    NeuronGene g = d.neurons[i];
    if (!g.active) return;
    NeuronState state = d.neuron_state[i];
    float exc[DENDRITES];
    for (int branch = 0; branch < DENDRITES; ++branch) {
        int slot = i * DENDRITES + branch;
        exc[branch] = d.pending_exc[slot];
        d.pending_exc[slot] = 0;
    }
    float inh = d.pending_inh[i];
    d.pending_inh[i] = 0;
    int port = d.neuron_port[i];
    float sensory = 0.0f;
    if (g.sensory_channel >= 0 && g.sensory_channel < S) {
        float level = (d.silence_senses ||
                       (d.silence_contact && g.sensory_channel >= CONTACT_FOOD_START))
            ? 0.0f :
            d.senses[a * S + g.sensory_channel];
        sensory = g.type == 5 && g.sensory_channel < 6
            ? receptor_current(g, state, level) : level * 230.0f;
    }
    neuron_advance(g, state, sensory, exc, inh);
    if (a == d.trace_animal && port >= S && port < S + M)
        d.trace_peak_mv[port - S] = fmaxf(d.trace_peak_mv[port - S],
                                         state.voltage_mv);
    if (state.spike && port >= S && port < S + M)
        atomicAdd(&d.motors[a * M + port - S], 1);
    d.spike_history[(tick % HISTORY) * NEURON_CAPACITY + i] = state.spike;
    d.neuron_state[i] = state;
}

__device__ inline void plasticity_synapse(Device d, const SynapseGene &g,
                                          SynapseState &s, int64_t tick) {
    bool excitatory_pair = !d.neurons[g.pre].inhibitory_transmitter &&
        excitatory_type(d.neurons[g.pre].type) &&
        d.neurons[g.pre].modulator == 0 &&
        !d.neurons[g.post].inhibitory_transmitter &&
        excitatory_type(d.neurons[g.post].type);
    int target = g.post * 6 + (tick & 1);
    unsigned char mod_event = 0;
    if (d.mod_event_tick[target] == static_cast<unsigned long long>(tick + 1))
        mod_event |= MOD_NE;
    if (d.mod_event_tick[target + 2] == static_cast<unsigned long long>(tick + 1))
        mod_event |= MOD_5HT;
    if (d.mod_event_tick[target + 4] == static_cast<unsigned long long>(tick + 1))
        mod_event |= MOD_DA;
    synapse_plasticity_step(g, s, s.arrived,
        d.neuron_state[g.post].spike, excitatory_pair,
        d.neuron_state[g.post].recent_depol_mv, true, mod_event);
}

__global__ void synapse_plasticity(Device d, int64_t tick, int occupied_count) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= occupied_count) return;
    int i = d.occupied_synapses[ordinal];
    int a = d.synapse_owner[i];
    if (a < 0 || d.animals[a].alive != 1) return;
    SynapseGene g = d.synapses[i];
    if (!g.active || d.neuron_owner[g.pre] != a ||
        d.neuron_owner[g.post] != a || !d.neurons[g.pre].active ||
        !d.neurons[g.post].active) return;
    SynapseState s = d.synapse_state[i];
    plasticity_synapse(d, g, s, tick);
    d.synapse_state[i] = s;
}

__global__ void synapse_plasticity_delivery(Device d, int64_t previous_tick,
                                            int64_t tick, int occupied_count) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= occupied_count) return;
    int i = d.occupied_synapses[ordinal];
    int a = d.synapse_owner[i];
    if (a < 0 || d.animals[a].alive != 1) return;
    SynapseGene g = d.synapses[i];
    if (!g.active || d.neuron_owner[g.pre] != a ||
        d.neuron_owner[g.post] != a || !d.neurons[g.pre].active ||
        !d.neurons[g.post].active) return;
    SynapseState s = d.synapse_state[i];
    // Matches plasticity(previous_tick) before delivery(tick), with one
    // synapse-state load/store and one launch for the pair of passes.
    plasticity_synapse(d, g, s, previous_tick);
    deliver_synapse(d, a, g, s, tick);
    d.synapse_state[i] = s;
}

struct alignas(16) BlockBrain {
    int neuron_ids[BLOCK_BRAIN_NEURONS];
    int synapse_ids[BLOCK_BRAIN_SYNAPSES];
    int pre[BLOCK_BRAIN_SYNAPSES], post[BLOCK_BRAIN_SYNAPSES];
    int port[BLOCK_BRAIN_NEURONS];
    NeuronState neuron_state[BLOCK_BRAIN_NEURONS];
    SynapseState synapse_state[BLOCK_BRAIN_SYNAPSES];
    float pending_exc[BLOCK_BRAIN_NEURONS * DENDRITES];
    float pending_inh[BLOCK_BRAIN_NEURONS];
    unsigned int mod_bits[BLOCK_BRAIN_NEURONS];
    unsigned char history[HISTORY * BLOCK_BRAIN_NEURONS];
};

__global__ void brain_block_step(Device d, int animal_count, int64_t first_tick) {
    if (blockIdx.x >= animal_count) return;
    int a = d.block_animals[blockIdx.x];
    if (d.animals[a].alive != 1) return;
    extern __shared__ __align__(16) unsigned char storage[];
    BlockBrain &brain = *reinterpret_cast<BlockBrain *>(storage);
    int lane = threadIdx.x;
    int neuron_count = d.animals[a].neuron_count;
    int synapse_count = d.animals[a].synapse_count;
    if (lane == 0) {
        int local = 0;
        for (int n = d.animals[a].neuron_head; n >= 0; n = d.neuron_next[n])
            brain.neuron_ids[local++] = n;
        local = 0;
        for (int e = d.animals[a].synapse_head; e >= 0; e = d.synapse_next[e])
            brain.synapse_ids[local++] = e;
    }
    __syncthreads();
    for (int n = lane; n < neuron_count; n += blockDim.x) {
        int global = brain.neuron_ids[n];
        brain.neuron_state[n] = d.neuron_state[global];
        brain.port[n] = d.neuron_port[global];
        brain.pending_inh[n] = d.pending_inh[global];
        for (int branch = 0; branch < DENDRITES; ++branch)
            brain.pending_exc[n * DENDRITES + branch] =
                d.pending_exc[global * DENDRITES + branch];
        for (int delay = 0; delay < HISTORY; ++delay)
            brain.history[delay * BLOCK_BRAIN_NEURONS + n] =
                d.spike_history[delay * NEURON_CAPACITY + global];
    }
    for (int e = lane; e < synapse_count; e += blockDim.x)
        brain.synapse_state[e] = d.synapse_state[brain.synapse_ids[e]];
    __syncthreads();
    for (int e = lane; e < synapse_count; e += blockDim.x) {
        SynapseGene g = d.synapses[brain.synapse_ids[e]];
        int pre = -1, post = -1;
        for (int n = 0; n < neuron_count; ++n) {
            if (brain.neuron_ids[n] == g.pre) pre = n;
            if (brain.neuron_ids[n] == g.post) post = n;
        }
        brain.pre[e] = pre;
        brain.post[e] = post;
    }
    __syncthreads();
    for (int t = 0; t < TICKS; ++t) {
        int64_t tick = first_tick + t;
        for (int n = lane; n < neuron_count; n += blockDim.x)
            brain.mod_bits[n] = 0;
        __syncthreads();
        for (int e = lane; e < synapse_count; e += blockDim.x) {
            int pre = brain.pre[e], post = brain.post[e];
            if (pre < 0 || post < 0) continue;
            SynapseGene g = d.synapses[brain.synapse_ids[e]];
            if (!g.active || !d.neurons[brain.neuron_ids[pre]].active ||
                !d.neurons[brain.neuron_ids[post]].active) continue;
            SynapseState state = brain.synapse_state[e];
            state.arrived = tick >= g.delay &&
                brain.history[((tick - g.delay) % HISTORY) *
                              BLOCK_BRAIN_NEURONS + pre];
            if (state.arrived) {
                NeuronGene source = d.neurons[brain.neuron_ids[pre]];
                float released = !source.inhibitory_transmitter || source.modulator
                    ? synapse_release(g, state, tick) : state.weight_ns;
                int motor = brain.port[post] - S;
                bool traced_motor = a == d.trace_animal && motor >= 0 && motor < M;
                if (source.modulator) {
                    if (released > 0)
                        atomicOr(&brain.mod_bits[post],
                                 1u << (source.modulator - 1));
                } else if (!source.inhibitory_transmitter) {
                    if (traced_motor) {
                        atomicAdd(&d.trace_exc_arrivals[motor], 1);
                        if (released > 0) atomicAdd(&d.trace_exc_releases[motor], 1);
                        atomicAdd(&d.trace_exc_ns[motor], released);
                    }
                    float scale = expf(brain.neuron_state[post].log_ampa_scale);
                    atomicAdd(&brain.pending_exc[post * DENDRITES + g.branch],
                              released * scale);
                } else {
                    if (traced_motor) {
                        atomicAdd(&d.trace_inh_arrivals[motor], 1);
                        atomicAdd(&d.trace_inh_ns[motor], released);
                    }
                    atomicAdd(&brain.pending_inh[post], released);
                }
            }
            brain.synapse_state[e] = state;
        }
        __syncthreads();
        for (int n = lane; n < neuron_count; n += blockDim.x) {
            NeuronGene g = d.neurons[brain.neuron_ids[n]];
            if (!g.active) continue;
            NeuronState state = brain.neuron_state[n];
            float exc[DENDRITES];
            for (int branch = 0; branch < DENDRITES; ++branch) {
                int slot = n * DENDRITES + branch;
                exc[branch] = brain.pending_exc[slot];
                brain.pending_exc[slot] = 0;
            }
            float inh = brain.pending_inh[n];
            brain.pending_inh[n] = 0;
            float sensory = 0;
            if (g.sensory_channel >= 0 && g.sensory_channel < S) {
                float level = (d.silence_senses ||
                               (d.silence_contact &&
                                g.sensory_channel >= CONTACT_FOOD_START))
                    ? 0.0f : d.senses[a * S + g.sensory_channel];
                sensory = g.type == 5 && g.sensory_channel < 6
                    ? receptor_current(g, state, level) : level * 230.0f;
            }
            neuron_advance(g, state, sensory, exc, inh);
            int port = brain.port[n];
            if (a == d.trace_animal && port >= S && port < S + M)
                d.trace_peak_mv[port - S] = fmaxf(d.trace_peak_mv[port - S],
                                                 state.voltage_mv);
            if (state.spike && port >= S && port < S + M)
                atomicAdd(&d.motors[a * M + port - S], 1);
            brain.history[(tick % HISTORY) * BLOCK_BRAIN_NEURONS + n] = state.spike;
            brain.neuron_state[n] = state;
        }
        __syncthreads();
        for (int e = lane; e < synapse_count; e += blockDim.x) {
            int pre = brain.pre[e], post = brain.post[e];
            if (pre < 0 || post < 0) continue;
            SynapseGene g = d.synapses[brain.synapse_ids[e]];
            if (!g.active || !d.neurons[brain.neuron_ids[pre]].active ||
                !d.neurons[brain.neuron_ids[post]].active) continue;
            SynapseState state = brain.synapse_state[e];
            NeuronGene source = d.neurons[brain.neuron_ids[pre]];
            NeuronGene target = d.neurons[brain.neuron_ids[post]];
            bool excitatory_pair = !source.inhibitory_transmitter &&
                excitatory_type(source.type) && source.modulator == 0 &&
                !target.inhibitory_transmitter && excitatory_type(target.type);
            synapse_plasticity_step(g, state, state.arrived,
                brain.neuron_state[post].spike, excitatory_pair,
                brain.neuron_state[post].recent_depol_mv, true,
                static_cast<unsigned char>(brain.mod_bits[post]));
            brain.synapse_state[e] = state;
        }
        __syncthreads();
    }
    for (int n = lane; n < neuron_count; n += blockDim.x) {
        int global = brain.neuron_ids[n];
        d.neuron_state[global] = brain.neuron_state[n];
        d.pending_inh[global] = brain.pending_inh[n];
        for (int branch = 0; branch < DENDRITES; ++branch)
            d.pending_exc[global * DENDRITES + branch] =
                brain.pending_exc[n * DENDRITES + branch];
        for (int delay = 0; delay < HISTORY; ++delay)
            d.spike_history[delay * NEURON_CAPACITY + global] =
                brain.history[delay * BLOCK_BRAIN_NEURONS + n];
    }
    for (int e = lane; e < synapse_count; e += blockDim.x)
        d.synapse_state[brain.synapse_ids[e]] = brain.synapse_state[e];
}

__global__ void homeostasis_block_animals(Device d, int animal_count) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= animal_count) return;
    int a = d.block_animals[ordinal];
    if (d.animals[a].alive != 1) return;
    float elapsed_ms = fmaxf(16.0f, d.animals[a].age * 16.0f);
    for (int n = d.animals[a].neuron_head; n >= 0; n = d.neuron_next[n]) {
        if (!d.neurons[n].active) continue;
        NeuronState state = d.neuron_state[n];
        float observed_hz = state.total_spikes * 1000.0f / elapsed_ms;
        homeostatic_advance(d.neurons[n], state, 16.0f, observed_hz);
        d.neuron_state[n] = state;
    }
}

__global__ void regrow(Device d, int64_t step, unsigned long long seed,
                       float renewal_probability) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W * H || d.food[c] != 0) return;
    auto rng = seed_for(seed + step, c);
    if (uniform(rng) < renewal_probability) atomicCAS(&d.food[c], 0, 1);
}

__global__ void body_step(Device d) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P || d.animals[a].alive != 1) return;
    Animal b = d.animals[a];
    const int age_bin = b.age < 100 ? 0 : (b.age < 300 ? 1 : 2);
    const bool hungry = b.energy < 16.0f;
    const float old_x = b.x, old_y = b.y;
    float motor_force[M];
    for (int m = 0; m < 3; ++m) {
        motor_force[m] = d.instant_motors
            ? d.motors[a * M + m] / (float)TICKS
            : (b.muscle[m] = fminf(1.0f,
                  b.muscle[m] * MUSCLE_DECAY_PER_STEP +
                  d.motors[a * M + m] * MUSCLE_TWITCH_PER_SPIKE));
    }
    float thrust = motor_force[0];
    float left = motor_force[1];
    float right = motor_force[2];
    // The measured slow body-muscle twitch is not assumed for the feeding
    // apparatus or biting; those actions need a current motor spike.
    bool eat = d.motors[a * M + 3] > 0;
    d.bite_intent[a] = d.motors[a * M + 4] > 0;
    if (b.id < d.initial_count && b.age < 100) {
        atomicAdd(d.founder_observed_steps, 1);
        for (int m = 0; m < M; ++m)
            if (d.motors[a * M + m] > 0)
                atomicAdd(&d.founder_motor_steps[m], 1);
    }
    for (int m = 0; m < M; ++m) d.motors[a * M + m] = 0;
    b.heading += (right - left) * 1.2f;
    b.x = clamp(b.x + cosf(b.heading) * thrust * 1.5f, 0, W - 0.001f);
    b.y = clamp(b.y + sinf(b.heading) * thrust * 1.5f, 0, H - 0.001f);
    float travel = hypotf(b.x - old_x, b.y - old_y);
    if (hungry && travel > 0.0001f) {
        float nearest[3] = {1e9f, 1e9f, 1e9f};
        float target_x[3]{}, target_y[3]{};
        int cx = (int)old_x, cy = (int)old_y;
        for (int dy = -4; dy <= 4; ++dy) {
            int y = cy + dy;
            if (y < 0 || y >= H) continue;
            for (int dx = -4; dx <= 4; ++dx) {
                int x = cx + dx;
                if (x < 0 || x >= W || d.food[y * W + x] <= 0) continue;
                int kind = d.kind[y * W + x];
                float fx = x + 0.5f, fy = y + 0.5f;
                float distance = (fx - old_x) * (fx - old_x) +
                                 (fy - old_y) * (fy - old_y);
                if (distance < nearest[kind]) {
                    nearest[kind] = distance;
                    target_x[kind] = fx;
                    target_y[kind] = fy;
                }
            }
        }
        for (int kind = 0; kind < 3; ++kind) if (nearest[kind] < 1e9f) {
            b.behavior.approach_progress[age_bin][kind] += sqrtf(nearest[kind]) -
                hypotf(target_x[kind] - b.x, target_y[kind] - b.y);
            b.behavior.approach_travel[age_bin][kind] += travel;
        }
    }
    int active_neurons = 0;
    for (int n = b.neuron_head; n >= 0; n = d.neuron_next[n])
        active_neurons += d.neurons[n].active;
    b.energy -= BODY_UPKEEP + active_neurons * NEURON_UPKEEP +
                thrust * 0.02f +
                (d.bite_intent[a] ? BITE_ENERGY_COST : 0.0f);
    b.fullness = fmaxf(0, b.fullness - 0.05f);
    b.pain *= 0.8f;
    for (int k = 0; k < 4; ++k) b.taste[k] *= 0.7f;
    // Feed at the location sensed before this neural decision. Movement may
    // reach another food cell, whose contact identity is sensed next step.
    int food_cell = cell(old_x, old_y);
    bool food_offered = atomicAdd(&d.food[food_cell], 0) > 0;
    if (food_offered) {
        int kind = d.kind[food_cell];
        ++b.behavior.opportunities[age_bin][hungry ? 0 : 1][kind];
    } else {
        ++b.behavior.empty_steps[age_bin][hungry ? 0 : 1];
        b.behavior.empty_eat[age_bin][hungry ? 0 : 1] += eat;
    }
    if (eat) {
        if (harvest(&d.food[food_cell])) {
            int kind = d.kind[food_cell];
            if (!food_offered)
                ++b.behavior.opportunities[age_bin][hungry ? 0 : 1][kind];
            ++b.behavior.accepted[age_bin][hungry ? 0 : 1][kind];
            if (kind == 1 && b.behavior.first_b_age == 0)
                b.behavior.first_b_age = b.age + 1;
            b.taste[kind] = 1;
            b.fullness += 1;
            if (kind == 0) b.energy = fminf(30, b.energy + 4);
            if (kind == 1) { b.energy = fminf(30, b.energy + 2); b.health -= 1.4f; b.pain += 1.4f; }
            if (b.fullness > 8) { b.health -= 0.8f; b.pain += 0.8f; }
        }
    }
    if (b.energy <= 0) { b.energy = 0; b.health -= 0.15f; b.pain += 0.15f; }
    ++b.age;
    if (b.health <= 0 || b.age >= 10000) {
        b.behavior.death_cause = b.age >= 10000 ? 4 : (b.energy <= 0 ? 2 : 3);
        b.alive = 0;
        atomicAdd(d.deaths, 1);
    }
    d.animals[a] = b;
}

__global__ void bite_animals(Device d, int64_t step) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P || d.animals[a].alive != 1 || !d.bite_intent[a]) return;
    int c = cell(d.animals[a].x, d.animals[a].y);
    for (int prey = d.cell_head[c]; prey >= 0; prey = d.cell_next[prey]) {
        if (prey == a || d.animals[prey].alive != 1) continue;
        float previous_health = atomicAdd(&d.animals[prey].health, -BITE_DAMAGE);
        atomicAdd(&d.animals[prey].pain, BITE_DAMAGE);
        atomicAdd(d.bites, 1);
        if (d.bite_events) {
            int event = atomicAdd(d.bite_event_count, 1);
            if (event < 2 * P)
                d.bite_events[event] = {step, d.animals[a].id,
                                        d.animals[prey].id,
                                        d.animals[prey].x, d.animals[prey].y};
        }
        if (previous_health <= BITE_DAMAGE &&
            atomicCAS(&d.animals[prey].alive, 1, -2) == 1) {
            atomicAdd(&d.animals[a].energy,
                      fmaxf(0.0f, d.animals[prey].energy) * MEAT_ENERGY_FRACTION);
            d.animals[a].taste[3] = 1.0f;
            atomicAdd(&d.animals[a].fullness, 1.0f);
        }
        break;
    }
}

__global__ void finish_predation(Device d) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P || d.animals[a].alive != -2) return;
    d.animals[a].behavior.death_cause = 1;
    d.animals[a].alive = 0;
    atomicAdd(d.deaths, 1);
    atomicAdd(d.predation_deaths, 1);
}

__global__ void homeostasis_step(Device d, int occupied_count) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= occupied_count) return;
    int i = d.occupied_neurons[ordinal];
    int a = d.neuron_owner[i];
    if (a < 0) return;
    if (d.animals[a].alive != 1 || !d.neurons[i].active) return;
    NeuronState state = d.neuron_state[i];
    float elapsed_ms = fmaxf(16.0f, d.animals[a].age * 16.0f);
    float observed_hz = state.total_spikes * 1000.0f / elapsed_ms;
    homeostatic_advance(d.neurons[i], state, 16.0f, observed_hz);
    d.neuron_state[i] = state;
}

__global__ void reclaim_dead_genomes(Device d, int64_t step) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P || d.animals[a].alive != 0) return;
    Animal body = d.animals[a];
    if (body.neuron_head < 0 && body.synapse_head < 0) return;
    int record_index = atomicAdd(d.life_count, 1);
    if (record_index < LIFE_RECORD_CAPACITY) {
        LifeRecord record{};
        record.body = body;
        record.death_step = step;
        for (int n = body.neuron_head; n >= 0; n = d.neuron_next[n]) {
            NeuronGene g = d.neurons[n];
            record.active_neurons += g.active;
            record.modulatory_neurons += g.active && g.modulator != 0;
        }
        float total_weight_change = 0;
        for (int e = body.synapse_head; e >= 0; e = d.synapse_next[e]) {
            SynapseGene g = d.synapses[e];
            bool active = g.active && d.neurons[g.pre].active &&
                          d.neurons[g.post].active;
            record.active_synapses += active;
            record.plastic_synapses += active && g.plasticity != 0 &&
                !d.neurons[g.pre].inhibitory_transmitter &&
                excitatory_type(d.neurons[g.pre].type) &&
                d.neurons[g.pre].modulator == 0 &&
                !d.neurons[g.post].inhibitory_transmitter &&
                excitatory_type(d.neurons[g.post].type);
            if (active) total_weight_change +=
                fabsf(d.synapse_state[e].weight_ns - g.initial_weight_ns);
        }
        record.mean_abs_weight_change = record.active_synapses > 0 ?
            total_weight_change / record.active_synapses : 0;
        d.life_records[record_index] = record;
    }
    release_genome(d, body);
    d.animals[a] = body;
}

__global__ void count_brains(Device d) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P) return;
    Animal body = d.animals[a];
    int neurons = 0, synapses = 0;
    if (body.alive == 1) {
        for (int n = body.neuron_head; n >= 0; n = d.neuron_next[n])
            neurons += d.neurons[n].active;
        for (int e = body.synapse_head; e >= 0; e = d.synapse_next[e]) {
            SynapseGene g = d.synapses[e];
            synapses += g.active && d.neurons[g.pre].active &&
                        d.neurons[g.post].active;
        }
    }
    d.active_neuron_counts[a] = neurons;
    d.active_synapse_counts[a] = synapses;
}

__device__ void remove_synapse(Device d, Animal &body, int target) {
    int previous = -1;
    for (int e = body.synapse_head; e >= 0; e = d.synapse_next[e]) {
        if (e != target) {
            previous = e;
            continue;
        }
        int next = d.synapse_next[e];
        if (previous < 0) body.synapse_head = next;
        else d.synapse_next[previous] = next;
        --body.synapse_count;
        d.synapse_owner[e] = -1;
        release_slot(d.synapse_used, e);
        return;
    }
}

__device__ void remove_hidden_neuron(Device d, Animal &body, int target) {
    if (target < 0 || d.neuron_port[target] >= 0) return;
    for (int e = body.synapse_head; e >= 0;) {
        int next = d.synapse_next[e];
        SynapseGene g = d.synapses[e];
        if (g.pre == target || g.post == target) remove_synapse(d, body, e);
        e = next;
    }
    int previous = -1;
    for (int n = body.neuron_head; n >= 0; n = d.neuron_next[n]) {
        if (n != target) {
            previous = n;
            continue;
        }
        int next = d.neuron_next[n];
        if (previous < 0) body.neuron_head = next;
        else d.neuron_next[previous] = next;
        --body.neuron_count;
        d.neuron_owner[n] = -1;
        release_slot(d.neuron_used, n);
        return;
    }
}

__device__ int random_hidden_afferent(Device d, const Animal &body,
                                      unsigned long long &rng) {
    for (int attempt = 0; attempt < 12; ++attempt) {
        int n = random_hidden_neuron(d, body, rng);
        if (n >= 0 && d.neurons[n].sensory_channel >= 0) return n;
    }
    return -1;
}

__device__ void mutate(Device d, int child, unsigned long long &rng) {
    int mutations = 1 + randint(rng, 4);
    for (int m = 0; m < mutations; ++m) {
        int choice = randint(rng, 12);
        Animal body = d.animals[child];
        int n = random_hidden_neuron(d, body, rng);
        int e = body.synapse_count ?
            synapse_at(d, body, randint(rng, body.synapse_count)) : -1;
        if (choice == 0) {
            float action = uniform(rng);
            if (action < 1.0f / 3.0f) {
                int added = claim_slot(d.neuron_used, NEURON_CAPACITY,
                                       child * 131 + body.neuron_count);
                int incoming = added >= 0 ?
                    claim_slot(d.synapse_used, SYNAPSE_CAPACITY,
                               child * 257 + body.synapse_count) : -1;
                int outgoing = incoming >= 0 ?
                    claim_slot(d.synapse_used, SYNAPSE_CAPACITY,
                               child * 257 + body.synapse_count + 1) : -1;
                if (added >= 0 && incoming >= 0 && outgoing >= 0 && e >= 0) {
                    int template_e = e;
                    for (int attempt = 0; attempt < 8; ++attempt) {
                        int candidate = synapse_at(d, body,
                            randint(rng, body.synapse_count));
                        SynapseGene candidate_gene = d.synapses[candidate];
                        if (candidate_gene.active &&
                            d.neurons[candidate_gene.pre].active &&
                            d.neurons[candidate_gene.post].active) {
                            template_e = candidate;
                            break;
                        }
                    }
                    unsigned char type = (unsigned char)randint(rng, 7);
                    NeuronGene g = initial_profile(type);
                    g.modulator = uniform(rng) < 0.12f ?
                        (unsigned char)(1 + randint(rng, 3)) : 0;
                    initialize_neuron(d, added, child, -1, g, body.neuron_head,
                                      (uint32_t)random64(rng));
                    body.neuron_head = added;
                    ++body.neuron_count;
                    SynapseGene first = d.synapses[template_e];
                    SynapseGene second = first;
                    first.post = added;
                    second.pre = added;
                    first.active = second.active = 1;
                    initialize_synapse(d, incoming, child, first,
                                       body.synapse_head, rng);
                    body.synapse_head = incoming;
                    initialize_synapse(d, outgoing, child, second,
                                       body.synapse_head, rng);
                    body.synapse_head = outgoing;
                    body.synapse_count += 2;
                    d.animals[child] = body;
                } else {
                    if (outgoing >= 0) release_slot(d.synapse_used, outgoing);
                    if (incoming >= 0) release_slot(d.synapse_used, incoming);
                    if (added >= 0) release_slot(d.neuron_used, added);
                }
            } else if (action < 2.0f / 3.0f) {
                if (n >= 0) {
                    remove_hidden_neuron(d, body, n);
                    d.animals[child] = body;
                }
            } else if (n >= 0) d.neurons[n].active ^= 1;
        } else if (choice == 1 && n >= 0) {
            auto active = d.neurons[n].active;
            unsigned char type = (unsigned char)randint(rng, 7);
            unsigned char modulator = d.neurons[n].modulator;
            unsigned char inhibitory_transmitter = d.neurons[n].inhibitory_transmitter;
            float target_affinity = d.neurons[n].target_affinity;
            d.neurons[n] = initial_profile(type);
            d.neurons[n].active = active;
            d.neurons[n].modulator = modulator;
            d.neurons[n].inhibitory_transmitter = inhibitory_transmitter;
            d.neurons[n].target_affinity = target_affinity;
        } else if (choice == 2) {
            // Motor ports stay fixed, but their intrinsic excitability can evolve.
            int target = uniform(rng) < 0.2f
                ? body.ports[S + randint(rng, M)] : n;
            if (target < 0) continue;
            auto &g = d.neurons[target];
            if (g.type == 2) {
                float *scales = &g.lp_scale.calcium;
                int channel = randint(rng, 4);
                scales[channel] = clamp(scales[channel] *
                    (0.9f + uniform(rng) * 0.2f), 0.2f, 3.0f);
            } else if (g.type == 3) {
                int channel = randint(rng, 9);
                float &conductance = channel == 8 ?
                    g.prinz_ambient_leak_ns : g.prinz_g.g[channel];
                conductance = clamp((conductance + (channel >= 7 ? 2.0f : 0.0f)) *
                    (0.9f + uniform(rng) * 0.2f), 0.0f,
                    channel >= 7 ? 200.0f : 500000.0f);
            } else if (g.type == 5) {
                int parameter = randint(rng, 4);
                if (parameter == 0) g.receptor_baseline_hz = clamp(
                    g.receptor_baseline_hz + (uniform(rng) - 0.5f) * 2, 0, 30);
                if (parameter == 1) g.receptor_gain_hz = clamp(
                    g.receptor_gain_hz + (uniform(rng) - 0.5f) * 10, 0, 300);
                if (parameter == 2) g.receptor_event_ns = clamp(
                    g.receptor_event_ns * (0.9f + uniform(rng) * 0.2f), 0.5f, 6);
                if (parameter == 3) g.receptor_decay_ms = clamp(
                    g.receptor_decay_ms * (0.9f + uniform(rng) * 0.2f), 1, 20);
            } else {
                g.threshold_mv = clamp(g.threshold_mv + (uniform(rng) - 0.5f) * 2, -60, -35);
            }
        } else if (choice == 3) {
            float action = uniform(rng);
            if (action < 1.0f / 3.0f) {
                int added = claim_slot(d.synapse_used, SYNAPSE_CAPACITY,
                                       child * 257 + body.synapse_count);
                if (added >= 0) {
                    SynapseGene g{};
                    random_synapse(g, d, child, rng);
                    initialize_synapse(d, added, child, g, body.synapse_head, rng);
                    body.synapse_head = added;
                    ++body.synapse_count;
                    d.animals[child] = body;
                }
            } else if (action < 2.0f / 3.0f) {
                if (e >= 0) {
                    remove_synapse(d, body, e);
                    d.animals[child] = body;
                }
            } else if (e >= 0) d.synapses[e].active ^= 1;
        } else if (choice == 4 && e >= 0) {
            auto &g = d.synapses[e];
            random_synapse(g, d, child, rng);
        } else if (choice == 5 && e >= 0) {
            auto &g = d.synapses[e];
            g.initial_weight_ns = clamp(g.initial_weight_ns + (uniform(rng) - 0.5f) * 0.5f, 0.1f, 8);
            g.release_sites = (unsigned char)clamp(
                g.release_sites + (uniform(rng) < 0.5f ? -1 : 1), 1, 64);
        } else if (choice == 6 && e >= 0) {
            auto &g = d.synapses[e];
            g.plasticity = (unsigned char)randint(rng, 6);
            g.mod_receptors = g.plasticity == 3 ?
                (unsigned char)(1 + randint(rng, 3)) :
                (g.plasticity >= 4 ? MOD_DA : 0);
            g.utilization = clamp(g.utilization + (uniform(rng) - 0.5f) * 0.1f, 0.02f, 0.8f);
            g.recovery_ms = clamp(g.recovery_ms + (uniform(rng) - 0.5f) * 50, 20, 1000);
            if (uniform(rng) < 0.15f) {
                g.slow_recovery_ms = g.slow_recovery_ms == 0 ? 7500.0f :
                    (uniform(rng) < 0.1f ? 0.0f :
                     clamp(g.slow_recovery_ms + (uniform(rng) - 0.5f) * 500,
                           1000.0f, 20000.0f));
                g.slow_depletion = g.slow_recovery_ms == 0 ? 0.0f :
                    clamp(g.slow_depletion + (uniform(rng) - 0.5f) * 0.01f,
                          0.0f, 0.1f);
            }
        } else if (choice == 7 && e >= 0) {
            auto &g = d.synapses[e];
            g.delay = (unsigned char)(1 + randint(rng, D));
            g.facilitation_ms = g.facilitation_ms == 0
                ? 500.0f + uniform(rng) * 3500.0f
                : (uniform(rng) < 0.1f ? 0.0f
                   : clamp(g.facilitation_ms + (uniform(rng) - 0.5f) * 200, 100, 4000));
            if (n >= 0) d.neurons[n].spike_adapt_pa = clamp(
                d.neurons[n].spike_adapt_pa + (uniform(rng) - 0.5f) * 5, 0, 150);
        } else if (choice == 8 && n >= 0) {
            auto &g = d.neurons[n];
            if (g.type == 0) {
                g.afterdepol_spike_pa = clamp(g.afterdepol_spike_pa +
                    (uniform(rng) - 0.5f) * 20, 0, 200);
                g.afterdepol_tau_ms = clamp(g.afterdepol_tau_ms +
                    (uniform(rng) - 0.5f) * 10, 10, 150);
            }
        } else if (choice == 9 && n >= 0) {
            if (d.neurons[n].type != 5 && uniform(rng) < 0.5f)
                d.neurons[n].modulator = (unsigned char)randint(rng, 4);
            else
                d.neurons[n].inhibitory_transmitter ^= 1;
        } else if (choice == 10) {
            float action = uniform(rng);
            int afferent = random_hidden_afferent(d, body, rng);
            if (action < 0.5f) {
                int added = claim_slot(d.neuron_used, NEURON_CAPACITY,
                                       child * 131 + body.neuron_count);
                int outgoing = added >= 0 ?
                    claim_slot(d.synapse_used, SYNAPSE_CAPACITY,
                               child * 257 + body.synapse_count) : -1;
                if (added >= 0 && outgoing >= 0) {
                    int target = body.ports[S + randint(rng, M)];
                    for (int attempt = 0; attempt < 8; ++attempt) {
                        int candidate = neuron_at(d, body,
                            randint(rng, body.neuron_count));
                        if (d.neuron_port[candidate] < 0 ||
                            d.neuron_port[candidate] >= S) {
                            target = candidate;
                            break;
                        }
                    }
                    int source = afferent >= 0 && uniform(rng) < 0.5f
                        ? afferent : body.ports[randint(rng, S)];
                    NeuronGene g = d.neurons[source];
                    g.modulator = 0;
                    g.active = 1;
                    initialize_neuron(d, added, child, -1, g, body.neuron_head,
                                      (uint32_t)random64(rng));
                    body.neuron_head = added;
                    ++body.neuron_count;
                    SynapseGene edge{};
                    edge.pre = added;
                    edge.post = target;
                    edge.delay = (unsigned char)(1 + randint(rng, D));
                    edge.branch = (unsigned char)randint(rng, DENDRITES);
                    edge.initial_weight_ns = g.type == 5
                        ? 29.0f / (60.0f * 0.75f) : 1.0f;
                    edge.utilization = g.type == 5 ? 0.75f : 0.55f;
                    edge.recovery_ms = g.type == 5 ? 50.0f : 450.0f;
                    edge.release_sites = g.type == 5 ? 50 : 14;
                    edge.slow_recovery_ms = g.type == 5 ? 7500.0f : 0.0f;
                    edge.slow_depletion = g.type == 5 ? 0.023f : 0.0f;
                    edge.active = 1;
                    initialize_synapse(d, outgoing, child, edge,
                                       body.synapse_head, rng);
                    body.synapse_head = outgoing;
                    ++body.synapse_count;
                    d.animals[child] = body;
                } else {
                    if (outgoing >= 0) release_slot(d.synapse_used, outgoing);
                    if (added >= 0) release_slot(d.neuron_used, added);
                }
            } else if (action < 0.75f && afferent >= 0) {
                remove_hidden_neuron(d, body, afferent);
                d.animals[child] = body;
            } else if (afferent >= 0) {
                auto &g = d.neurons[afferent];
                int channel = randint(rng, S);
                unsigned char active = g.active;
                unsigned char inhibitory_transmitter = g.inhibitory_transmitter;
                float target_affinity = g.target_affinity;
                if (g.type == 5 && channel >= 6) {
                    g = cortical_l5_profile();
                } else if (g.type != 5 && channel < 6) {
                    g = orn_receptor_profile();
                }
                g.sensory_channel = (signed char)channel;
                g.active = active;
                g.inhibitory_transmitter = inhibitory_transmitter;
                g.target_affinity = target_affinity;
            }
        } else if (choice == 11) {
            int target = uniform(rng) < 0.3f
                ? body.ports[S + randint(rng, M)] : n;
            if (target >= 0) {
                auto &g = d.neurons[target];
                g.target_affinity = clamp(g.target_affinity *
                    (uniform(rng) < 0.5f ? 0.5f : 2.0f), 0.125f, 256.0f);
            }
        }
    }
}

__global__ void reproduce(Device d) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P || d.animals[a].alive != 1 || d.animals[a].energy < 22) return;
    Animal parent = d.animals[a];
    auto rng = parent.rng;
    int start = randint(rng, P);
    int child = -1;
    for (int j = 0; j < P; ++j) {
        int candidate = (start + j) % P;
        if (atomicCAS(&d.animals[candidate].alive, 0, -1) == 0) { child = candidate; break; }
    }
    if (child < 0) { d.animals[a].rng = rng; return; }
    Animal newborn{};
    newborn.neuron_head = newborn.synapse_head = -1;
    for (int p = 0; p < S + M; ++p) newborn.ports[p] = -1;
    newborn.x = clamp(parent.x + (uniform(rng) - 0.5f) * 2, 0, W - 0.001f);
    newborn.y = clamp(parent.y + (uniform(rng) - 0.5f) * 2, 0, H - 0.001f);
    newborn.heading = uniform(rng) * 2 * PI;
    newborn.energy = 8;
    newborn.health = 10;
    newborn.generation = parent.generation + 1;
    bool complete = true;
    for (int old = parent.neuron_head; old >= 0; old = d.neuron_next[old]) {
        int n = claim_slot(d.neuron_used, NEURON_CAPACITY,
                           child * 131 + newborn.neuron_count);
        if (n < 0) { complete = false; break; }
        int port = d.neuron_port[old];
        initialize_neuron(d, n, child, port, d.neurons[old], newborn.neuron_head,
                          (uint32_t)random64(rng));
        d.neuron_clone[old] = n;
        newborn.neuron_head = n;
        ++newborn.neuron_count;
        if (port >= 0) newborn.ports[port] = n;
    }
    if (complete) for (int old = parent.synapse_head; old >= 0;
                      old = d.synapse_next[old]) {
        int e = claim_slot(d.synapse_used, SYNAPSE_CAPACITY,
                           child * 257 + newborn.synapse_count);
        if (e < 0) { complete = false; break; }
        SynapseGene g = d.synapses[old];
        g.pre = d.neuron_clone[g.pre];
        g.post = d.neuron_clone[g.post];
        initialize_synapse(d, e, child, g, newborn.synapse_head, rng);
        newborn.synapse_head = e;
        ++newborn.synapse_count;
    }
    if (!complete) {
        release_genome(d, newborn);
        d.animals[child].alive = 0;
        d.animals[a].rng = rng;
        return;
    }
    newborn.id = atomicAdd(d.next_id, 1);
    newborn.parent_id = parent.id;
    newborn.alive = -1;
    d.animals[child] = newborn;
    mutate(d, child, rng);
    newborn = d.animals[child];
    for (int n = newborn.neuron_head; n >= 0; n = d.neuron_next[n])
        d.neuron_state[n] = initial_neuron_state(
            d.neurons[n], (uint32_t)random64(rng));
    for (int e = newborn.synapse_head; e >= 0; e = d.synapse_next[e])
        d.synapse_state[e] = initial_synapse_state(
            d.synapses[e], (uint32_t)random64(rng));
    for (int s = 0; s < S; ++s) d.senses[child * S + s] = 0;
    for (int m = 0; m < M; ++m) d.motors[child * M + m] = 0;
    newborn.rng = rng;
    d.animals[child] = newborn;
    parent.energy -= 10;
    ++parent.behavior.offspring;
    parent.rng = rng;
    d.animals[a] = parent;
    __threadfence();
    atomicExch(&d.animals[child].alive, 1);
    atomicAdd(d.births, 1);
}

void check(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}
template <typename T> void allocate(T *&ptr, size_t count) {
    check(cudaMalloc(&ptr, count * sizeof(T)), "CUDA allocation");
}
template <typename T> void download(std::vector<T> &out, T *ptr, size_t count) {
    out.resize(count);
    check(cudaMemcpy(out.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost), "CUDA download");
}

void write_world(const std::string &filename, Device d) {
    std::vector<int> food, kind;
    std::vector<Animal> animals;
    download(food, d.food, W * H);
    download(kind, d.kind, W * H);
    download(animals, d.animals, P);
    std::vector<unsigned char> image(W * H * 3, 0);
    const int colors[3][3] = {{70, 180, 80}, {200, 85, 55}, {80, 105, 180}};
    for (int i = 0; i < W * H; ++i) if (food[i] > 0)
        for (int k = 0; k < 3; ++k) image[i * 3 + k] = colors[kind[i]][k];
    for (const auto &b : animals) if (b.alive == 1) {
        int i = (int)b.y * W + (int)b.x;
        for (int k = 0; k < 3; ++k) image[i * 3 + k] = 255;
    }
    constexpr int scale = 4;
    std::vector<unsigned char> enlarged(W * H * scale * scale * 3);
    for (int y = 0; y < H * scale; ++y) for (int x = 0; x < W * scale; ++x)
        for (int k = 0; k < 3; ++k)
            enlarged[(y * W * scale + x) * 3 + k] = image[((y / scale) * W + x / scale) * 3 + k];
    png_image png{};
    png.version = PNG_IMAGE_VERSION;
    png.width = W * scale;
    png.height = H * scale;
    png.format = PNG_FORMAT_RGB;
    if (!png_image_write_to_file(&png, filename.c_str(), 0, enlarged.data(), 0, nullptr))
        std::cerr << "World image: " << png.message << '\n';
}

void write_brains(const std::string &prefix, Device d) {
    std::vector<Animal> animals;
    std::vector<NeuronGene> neurons;
    std::vector<SynapseGene> synapses;
    std::vector<SynapseState> states;
    std::vector<NeuronState> neuron_states;
    std::vector<int> neuron_next, synapse_next;
    std::vector<int> neuron_port;
    download(animals, d.animals, P);
    download(neurons, d.neurons, NEURON_CAPACITY);
    download(synapses, d.synapses, SYNAPSE_CAPACITY);
    download(states, d.synapse_state, SYNAPSE_CAPACITY);
    download(neuron_states, d.neuron_state, NEURON_CAPACITY);
    download(neuron_next, d.neuron_next, NEURON_CAPACITY);
    download(synapse_next, d.synapse_next, SYNAPSE_CAPACITY);
    download(neuron_port, d.neuron_port, NEURON_CAPACITY);
    std::ofstream cells(prefix + ".neurons.csv");
    std::ofstream edges(prefix + ".synapses.csv");
    cells << "animal_id,generation,neuron,port,active,type,modulator,inhibitory_transmitter,target_affinity,capacitance_pf,leak_ns,rest_mv,threshold_mv,slope_mv,adapt_ns,adapt_tau_ms,spike_adapt_pa,reset_mv,afterdepol_spike_pa,afterdepol_tau_ms,ampa_scale,lp_calcium_scale,lp_kca_scale,lp_h_scale,lp_leak_scale,prinz_na_ns,prinz_cat_ns,prinz_cas_ns,prinz_a_ns,prinz_kca_ns,prinz_kd_ns,prinz_h_ns,prinz_leak_ns,prinz_ambient_leak_ns,sensory_channel,receptor_baseline_hz,receptor_gain_hz,receptor_event_ns,receptor_decay_ms,receptor_state_ns,total_spikes\n";
    edges << "animal_id,generation,synapse,active,pre,post,branch,delay_ms,initial_weight_ns,learned_weight_ns,spine_volume,utilization,recovery_ms,facilitation_ms,release_sites,plasticity,mod_receptors,slow_recovery_ms,slow_depletion\n";
    for (int a = 0; a < P; ++a) if (animals[a].alive == 1) {
        for (int n = animals[a].neuron_head, visited = 0;
             n >= 0 && visited < animals[a].neuron_count;
             n = neuron_next[n], ++visited) {
            const auto &g = neurons[n];
            cells << animals[a].id << ',' << animals[a].generation << ',' << n << ','
                << neuron_port[n] << ',' << (int)g.active << ','
                << (int)g.type << ',' << (int)g.modulator << ','
                << (int)g.inhibitory_transmitter << ','
                << g.target_affinity << ','
                << g.capacitance_pf << ',' << g.leak_ns << ','
                << g.rest_mv << ',' << g.threshold_mv << ',' << g.slope_mv << ','
                << g.adapt_ns << ',' << g.adapt_tau_ms << ',' << g.spike_adapt_pa << ','
                << g.reset_mv << ','
                << g.afterdepol_spike_pa << ',' << g.afterdepol_tau_ms << ','
                << expf(neuron_states[n].log_ampa_scale) << ','
                << g.lp_scale.calcium << ',' << g.lp_scale.calcium_k << ','
                << g.lp_scale.h << ',' << g.lp_scale.leak;
            for (int channel = 0; channel < 8; ++channel)
                cells << ',' << g.prinz_g.g[channel];
            cells << ',' << g.prinz_ambient_leak_ns << ','
                  << (int)g.sensory_channel << ','
                  << g.receptor_baseline_hz << ',' << g.receptor_gain_hz << ','
                  << g.receptor_event_ns << ',' << g.receptor_decay_ms << ','
                  << neuron_states[n].receptor_ns << ','
                  << neuron_states[n].total_spikes << '\n';
        }
        for (int e = animals[a].synapse_head, visited = 0;
             e >= 0 && visited < animals[a].synapse_count;
             e = synapse_next[e], ++visited) {
            const auto &g = synapses[e];
            edges << animals[a].id << ',' << animals[a].generation << ',' << e << ','
                << (int)g.active << ','
                << g.pre << ',' << g.post << ',' << (int)g.branch << ','
                << (int)g.delay << ',' << g.initial_weight_ns << ','
                << states[e].weight_ns << ',' << states[e].spine_volume << ','
                << g.utilization << ',' << g.recovery_ms << ','
                << g.facilitation_ms << ',' << (int)g.release_sites << ','
                << (int)g.plasticity << ',' << (int)g.mod_receptors << ','
                << g.slow_recovery_ms << ',' << g.slow_depletion << '\n';
        }
    }
}

struct BehaviorTotals {
    uint64_t opportunities[3][2][3]{}, accepted[3][2][3]{};
    uint64_t empty_steps[3][2]{}, empty_eat[3][2]{};
    double approach_progress[3][3]{}, approach_travel[3][3]{};
};

void add_behavior(BehaviorTotals &totals, const Behavior &b) {
    for (int age = 0; age < 3; ++age) {
        for (int hunger = 0; hunger < 2; ++hunger) {
            for (int food = 0; food < 3; ++food) {
                totals.opportunities[age][hunger][food] +=
                    b.opportunities[age][hunger][food];
                totals.accepted[age][hunger][food] +=
                    b.accepted[age][hunger][food];
            }
            totals.empty_steps[age][hunger] += b.empty_steps[age][hunger];
            totals.empty_eat[age][hunger] += b.empty_eat[age][hunger];
        }
        for (int food = 0; food < 3; ++food) {
            totals.approach_progress[age][food] +=
                b.approach_progress[age][food];
            totals.approach_travel[age][food] +=
                b.approach_travel[age][food];
        }
    }
}

void write_behavior_header(std::ostream &out) {
    constexpr const char *ages[] = {"early", "middle", "late"};
    constexpr const char *foods[] = {"a", "b", "c"};
    for (int age = 0; age < 3; ++age) {
        for (int hunger = 0; hunger < 2; ++hunger) {
            for (int food = 0; food < 3; ++food)
                out << ',' << ages[age] << (hunger == 0 ? "_hungry" : "_sated")
                    << "_op_" << foods[food] << ',' << ages[age]
                    << (hunger == 0 ? "_hungry" : "_sated")
                    << "_eat_" << foods[food];
            out << ',' << ages[age] << (hunger == 0 ? "_hungry" : "_sated")
                << "_empty_steps," << ages[age]
                << (hunger == 0 ? "_hungry" : "_sated") << "_empty_eat";
        }
        for (int food = 0; food < 3; ++food)
            out << ',' << ages[age] << "_approach_" << foods[food]
                << ',' << ages[age] << "_travel_" << foods[food];
    }
}

template <typename T> void write_behavior_values(std::ostream &out, const T &b) {
    for (int age = 0; age < 3; ++age) {
        for (int hunger = 0; hunger < 2; ++hunger) {
            for (int food = 0; food < 3; ++food)
                out << ',' << b.opportunities[age][hunger][food]
                    << ',' << b.accepted[age][hunger][food];
            out << ',' << b.empty_steps[age][hunger]
                << ',' << b.empty_eat[age][hunger];
        }
        for (int food = 0; food < 3; ++food)
            out << ',' << b.approach_progress[age][food]
                << ',' << b.approach_travel[age][food];
    }
}

void write_life_record(std::ostream &out, const LifeRecord &record) {
    const Animal &b = record.body;
    out << record.death_step << ',' << b.id << ',' << b.parent_id << ','
        << b.generation << ',' << b.age << ',' << b.behavior.death_cause << ','
        << b.behavior.offspring << ',' << b.neuron_count << ','
        << record.active_neurons << ',' << b.synapse_count << ','
        << record.active_synapses << ',' << record.modulatory_neurons << ','
        << record.plastic_synapses << ',' << record.mean_abs_weight_change << ','
        << b.energy << ',' << b.health << ',' << b.behavior.first_b_age;
    write_behavior_values(out, b.behavior);
    out << '\n';
}

} // namespace sim

volatile std::sig_atomic_t interrupted = 0;
void on_interrupt(int) { interrupted = 1; }

int main(int argc, char **argv) {
    using namespace sim;
    int64_t steps = 2000;
    int initial = 1024;
    int live_port = 8765;
    int trace_animal = -1;
    bool visualize = false, profile = false, legacy_neural = false,
         block_brains = false,
         steps_specified = false, silence_senses = false,
         silence_contact = false,
         instant_motors = false;
    float food_renewal = 0.0006f;
    unsigned long long seed = 1;
    std::string output = "run.csv";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            steps = strtoll(argv[++i], nullptr, 10);
            steps_specified = true;
        }
        else if (!strcmp(argv[i], "--population") && i + 1 < argc) initial = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--food-renewal") && i + 1 < argc)
            food_renewal = strtof(argv[++i], nullptr);
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) output = argv[++i];
        else if (!strcmp(argv[i], "--visualize")) visualize = true;
        else if (!strcmp(argv[i], "--profile")) profile = true;
        else if (!strcmp(argv[i], "--legacy-neural")) legacy_neural = true;
        else if (!strcmp(argv[i], "--block-brains")) block_brains = true;
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) live_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trace-animal") && i + 1 < argc)
            trace_animal = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--silence-senses")) silence_senses = true;
        else if (!strcmp(argv[i], "--silence-contact")) silence_contact = true;
        else if (!strcmp(argv[i], "--instant-motors")) instant_motors = true;
        else { std::cerr << "Usage: neural_selection [--steps N|0] [--population 1..32768] [--seed N] [--food-renewal probability] [--output file.csv] [--visualize] [--profile] [--legacy-neural] [--block-brains] [--port 1..65535] [--trace-animal founder_slot] [--silence-senses] [--silence-contact] [--instant-motors]\n"; return 2; }
    }
    if (visualize && !steps_specified) steps = 0;
    if (steps < 0 || steps > INT64_MAX / TICKS ||
        initial < 1 || initial > P || trace_animal >= initial || trace_animal < -1 ||
        !std::isfinite(food_renewal) ||
        food_renewal < 0 || food_renewal > 1 || live_port < 1 || live_port > 65535) {
        std::cerr << "Invalid steps, population, food renewal, or port\n"; return 2;
    }
    std::signal(SIGINT, on_interrupt);
    std::unique_ptr<LiveView> live;
    if (visualize) {
        try { live = std::make_unique<LiveView>(live_port); }
        catch (const std::exception &error) {
            std::cerr << error.what() << '\n'; return 1;
        }
        std::cout << "Live world: http://127.0.0.1:" << live_port << '/'
                  << "  (Ctrl+C to close after the run)\n" << std::flush;
    }
    int count = 0;
    check(cudaGetDeviceCount(&count), "GPU discovery");
    if (count < 1) { std::cerr << "No NVIDIA GPU found\n"; return 1; }
    Device d{};
    d.initial_count = initial;
    d.trace_animal = trace_animal;
    d.silence_senses = silence_senses;
    d.silence_contact = silence_contact;
    d.instant_motors = instant_motors;
    allocate(d.animals, P);
    allocate(d.neurons, NEURON_CAPACITY); allocate(d.synapses, SYNAPSE_CAPACITY);
    allocate(d.neuron_state, NEURON_CAPACITY);
    allocate(d.synapse_state, SYNAPSE_CAPACITY);
    allocate(d.neuron_owner, NEURON_CAPACITY);
    allocate(d.synapse_owner, SYNAPSE_CAPACITY);
    allocate(d.occupied_neurons, NEURON_CAPACITY);
    allocate(d.occupied_synapses, SYNAPSE_CAPACITY);
    allocate(d.occupied_neuron_count, 1);
    allocate(d.occupied_synapse_count, 1);
    allocate(d.block_animals, P);
    allocate(d.block_animal_count, 1);
    allocate(d.neuron_next, NEURON_CAPACITY);
    allocate(d.synapse_next, SYNAPSE_CAPACITY);
    allocate(d.neuron_port, NEURON_CAPACITY);
    allocate(d.neuron_clone, NEURON_CAPACITY);
    allocate(d.neuron_used, NEURON_CAPACITY / 32);
    allocate(d.synapse_used, SYNAPSE_CAPACITY / 32);
    allocate(d.active_neuron_counts, P);
    allocate(d.active_synapse_counts, P);
    allocate(d.pending_exc, NEURON_CAPACITY * DENDRITES);
    allocate(d.pending_inh, NEURON_CAPACITY);
    allocate(d.mod_event_tick, NEURON_CAPACITY * 6);
    allocate(d.spike_history, HISTORY * NEURON_CAPACITY);
    allocate(d.senses, P * S);
    allocate(d.motors, P * M); allocate(d.bite_intent, P);
    allocate(d.trace_exc_arrivals, M); allocate(d.trace_exc_releases, M);
    allocate(d.trace_inh_arrivals, M); allocate(d.trace_exc_ns, M);
    allocate(d.trace_inh_ns, M); allocate(d.trace_peak_mv, M);
    allocate(d.founder_motor_steps, M);
    allocate(d.founder_observed_steps, 1);
    allocate(d.food, W * H); allocate(d.kind, W * H);
    allocate(d.cell_head, W * H); allocate(d.cell_next, P);
    allocate(d.births, 1); allocate(d.deaths, 1);
    allocate(d.predation_deaths, 1); allocate(d.bites, 1);
    allocate(d.next_id, 1);
    allocate(d.life_records, LIFE_RECORD_CAPACITY);
    allocate(d.life_count, 1);
    if (live) {
        allocate(d.bite_events, 2 * P);
        allocate(d.bite_event_count, 1);
        check(cudaMemset(d.bite_event_count, 0, sizeof(int)), "bite event reset");
    }
    auto slots = thrust::make_counting_iterator<int>(0);
    size_t neuron_select_bytes = 0, synapse_select_bytes = 0;
    size_t animal_select_bytes = 0, fallback_neuron_bytes = 0,
           fallback_synapse_bytes = 0;
    check(cub::DeviceSelect::If(nullptr, neuron_select_bytes, slots,
          d.occupied_neurons, d.occupied_neuron_count, NEURON_CAPACITY,
          OwnedSlot{d.neuron_owner}), "neuron selection size");
    check(cub::DeviceSelect::If(nullptr, synapse_select_bytes, slots,
          d.occupied_synapses, d.occupied_synapse_count, SYNAPSE_CAPACITY,
          OwnedSlot{d.synapse_owner}), "synapse selection size");
    check(cub::DeviceSelect::If(nullptr, animal_select_bytes, slots,
          d.block_animals, d.block_animal_count, P,
          BlockAnimal{d.animals}), "animal selection size");
    check(cub::DeviceSelect::If(nullptr, fallback_neuron_bytes, slots,
          d.occupied_neurons, d.occupied_neuron_count, NEURON_CAPACITY,
          FallbackOwnedSlot{d.neuron_owner, d.animals}), "fallback neuron selection size");
    check(cub::DeviceSelect::If(nullptr, fallback_synapse_bytes, slots,
          d.occupied_synapses, d.occupied_synapse_count, SYNAPSE_CAPACITY,
          FallbackOwnedSlot{d.synapse_owner, d.animals}), "fallback synapse selection size");
    void *selection_temp = nullptr;
    size_t selection_temp_bytes = std::max({neuron_select_bytes, synapse_select_bytes,
        animal_select_bytes, fallback_neuron_bytes, fallback_synapse_bytes});
    check(cudaMalloc(&selection_temp, selection_temp_bytes), "selection workspace");
    if (block_brains)
        check(cudaFuncSetAttribute(brain_block_step,
              cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(BlockBrain)),
              "brain block shared memory");
    check(cudaMemset(d.neuron_owner, 0xff,
                     NEURON_CAPACITY * sizeof(int)), "neuron pool reset");
    check(cudaMemset(d.synapse_owner, 0xff,
                     SYNAPSE_CAPACITY * sizeof(int)), "synapse pool reset");
    check(cudaMemset(d.neuron_used, 0,
                     NEURON_CAPACITY / 32 * sizeof(uint32_t)), "neuron pool reset");
    check(cudaMemset(d.synapse_used, 0,
                     SYNAPSE_CAPACITY / 32 * sizeof(uint32_t)), "synapse pool reset");
    check(cudaMemset(d.births, 0, sizeof(int)), "counter reset");
    check(cudaMemset(d.deaths, 0, sizeof(int)), "counter reset");
    check(cudaMemset(d.predation_deaths, 0, sizeof(int)), "counter reset");
    check(cudaMemset(d.bites, 0, sizeof(int)), "counter reset");
    check(cudaMemset(d.life_count, 0, sizeof(int)), "life record reset");
    check(cudaMemset(d.founder_motor_steps, 0, M * sizeof(int)), "founder activity reset");
    check(cudaMemset(d.trace_exc_arrivals, 0, M * sizeof(int)), "trace reset");
    check(cudaMemset(d.trace_exc_releases, 0, M * sizeof(int)), "trace reset");
    check(cudaMemset(d.trace_inh_arrivals, 0, M * sizeof(int)), "trace reset");
    check(cudaMemset(d.trace_exc_ns, 0, M * sizeof(float)), "trace reset");
    check(cudaMemset(d.trace_inh_ns, 0, M * sizeof(float)), "trace reset");
    float trace_peak_reset[M] = {-100, -100, -100, -100, -100};
    check(cudaMemcpy(d.trace_peak_mv, trace_peak_reset, sizeof(trace_peak_reset),
                     cudaMemcpyHostToDevice), "trace reset");
    check(cudaMemset(d.founder_observed_steps, 0, sizeof(int)), "founder activity reset");
    check(cudaMemcpy(d.next_id, &initial, sizeof(int), cudaMemcpyHostToDevice), "counter init");
    init_world<<<(W * H + 255) / 256, 256>>>(d, seed);
    init_animals<<<(P + 127) / 128, 128>>>(d, initial, seed);
    check(cudaMemset(d.cell_head, 0xff, W * H * sizeof(int)), "spatial reset");
    index_animals<<<(P + 127) / 128, 128>>>(d);
    update_senses<<<(P + 127) / 128, 128>>>(d);
    check(cudaDeviceSynchronize(), "initialization");
    std::ofstream csv(output);
    std::ofstream behavior_csv(output + ".behavior.csv");
    std::ofstream lives_csv(output + ".lives.csv");
    if (!csv || !behavior_csv || !lives_csv) {
        std::cerr << "Cannot open output files\n"; return 1;
    }
    csv << "step,population,births,deaths,predation_deaths,bites,food_cells,max_generation,mean_energy,mean_neurons,mean_synapses,allocated_neurons,allocated_synapses,max_neuron_slots,max_synapse_slots,founder_observed_steps,founder_forward_steps,founder_left_steps,founder_right_steps,founder_eat_steps,founder_bite_steps\n";
    behavior_csv << "step";
    write_behavior_header(behavior_csv);
    behavior_csv << '\n';
    lives_csv << "death_step,id,parent_id,generation,age,death_cause,offspring,"
                 "neuron_count,active_neurons,synapse_count,active_synapses,"
                 "modulatory_neurons,plastic_synapses,mean_abs_weight_change,"
                 "energy,health,first_b_age";
    write_behavior_header(lives_csv);
    lives_csv << '\n';
    BehaviorTotals dead_totals;
    std::deque<std::pair<int64_t, BehaviorTotals>> behavior_history;
    std::vector<Animal> bodies;
    std::vector<LifeRecord> recent_lives;
    std::vector<BiteEvent> recent_bites;
    std::deque<std::pair<uint64_t, BiteEvent>> bite_history;
    uint64_t next_bite_event_id = 1;
    std::vector<int> active_neurons, active_synapses;
    std::vector<int> food, kind;
    std::vector<uint32_t> neuron_used, synapse_used;
    double live_count_ms = 0, live_download_ms = 0, live_format_ms = 0;
    int64_t live_profile_reports = 0;
    auto report = [&](int64_t step, bool log) {
        auto phase_begin = std::chrono::steady_clock::now();
        count_brains<<<(P + 127) / 128, 128>>>(d);
        check(cudaDeviceSynchronize(), "simulation step");
        check(cudaGetLastError(), "CUDA kernel");
        auto phase_counted = std::chrono::steady_clock::now();
        download(bodies, d.animals, P);
        download(active_neurons, d.active_neuron_counts, P);
        download(active_synapses, d.active_synapse_counts, P);
        download(food, d.food, W * H);
        if (live) download(kind, d.kind, W * H);
        auto phase_downloaded = std::chrono::steady_clock::now();
        if (log) {
            download(neuron_used, d.neuron_used, NEURON_CAPACITY / 32);
            download(synapse_used, d.synapse_used, SYNAPSE_CAPACITY / 32);
            int life_count = 0;
            check(cudaMemcpy(&life_count, d.life_count, sizeof(int),
                             cudaMemcpyDeviceToHost), "life count");
            if (life_count > LIFE_RECORD_CAPACITY) {
                std::cerr << "Life record buffer overflow at step " << step << '\n';
                std::exit(1);
            }
            if (life_count > 0) download(recent_lives, d.life_records, life_count);
            for (int i = 0; i < life_count; ++i) {
                const LifeRecord &record = recent_lives[i];
                add_behavior(dead_totals, record.body.behavior);
                if (record.body.id % LIFE_SAMPLE_INTERVAL == 0)
                    write_life_record(lives_csv, record);
            }
            check(cudaMemset(d.life_count, 0, sizeof(int)), "life record reset");
        }
        int alive = 0, max_generation = 0, births = 0, deaths = 0;
        int predation_deaths = 0, bites = 0;
        double energy = 0, neurons = 0, synapses = 0;
        int allocated_neurons = 0, allocated_synapses = 0;
        int max_neuron_slots = 0, max_synapse_slots = 0;
        for (int a = 0; a < P; ++a) if (bodies[a].alive == 1) {
            ++alive; energy += bodies[a].energy;
            max_generation = std::max(max_generation, bodies[a].generation);
            neurons += active_neurons[a];
            synapses += active_synapses[a];
            allocated_neurons += bodies[a].neuron_count;
            allocated_synapses += bodies[a].synapse_count;
            max_neuron_slots = std::max(max_neuron_slots, bodies[a].neuron_count);
            max_synapse_slots = std::max(max_synapse_slots, bodies[a].synapse_count);
        }
        if (log) {
            int occupied_neurons = 0, occupied_synapses = 0;
            for (uint32_t word : neuron_used) occupied_neurons += __builtin_popcount(word);
            for (uint32_t word : synapse_used) occupied_synapses += __builtin_popcount(word);
            if (occupied_neurons != allocated_neurons ||
                occupied_synapses != allocated_synapses) {
                std::cerr << "Global pool accounting mismatch at step " << step << '\n';
                std::exit(1);
            }
        }
        check(cudaMemcpy(&births, d.births, sizeof(int), cudaMemcpyDeviceToHost), "birth count");
        check(cudaMemcpy(&deaths, d.deaths, sizeof(int), cudaMemcpyDeviceToHost), "death count");
        check(cudaMemcpy(&predation_deaths, d.predation_deaths, sizeof(int), cudaMemcpyDeviceToHost), "predation count");
        check(cudaMemcpy(&bites, d.bites, sizeof(int), cudaMemcpyDeviceToHost), "bite count");
        int food_cells = std::count_if(food.begin(), food.end(), [](int amount) { return amount > 0; });
        if (log) {
            int founder_observed_steps = 0;
            int founder_motor_steps[M]{};
            check(cudaMemcpy(&founder_observed_steps, d.founder_observed_steps,
                             sizeof(int), cudaMemcpyDeviceToHost), "founder activity");
            check(cudaMemcpy(founder_motor_steps, d.founder_motor_steps,
                             M * sizeof(int), cudaMemcpyDeviceToHost), "founder activity");
            BehaviorTotals all = dead_totals;
            for (int a = 0; a < P; ++a)
                if (bodies[a].alive == 1) add_behavior(all, bodies[a].behavior);
            behavior_csv << step;
            write_behavior_values(behavior_csv, all);
            behavior_csv << '\n';
            behavior_csv.flush();
            lives_csv.flush();
            behavior_history.emplace_back(step, all);
            while (behavior_history.size() > 1 &&
                   behavior_history.front().first < step - 10000)
                behavior_history.pop_front();
            csv << step << ',' << alive << ',' << births << ',' << deaths << ','
                << predation_deaths << ',' << bites << ',' << food_cells << ','
                << max_generation << ','
                << (alive ? energy / alive : 0) << ',' << (alive ? neurons / alive : 0) << ','
                << (alive ? synapses / alive : 0) << ','
                << allocated_neurons << ',' << allocated_synapses << ','
                << max_neuron_slots << ',' << max_synapse_slots
                << ',' << founder_observed_steps;
            for (int m = 0; m < M; ++m) csv << ',' << founder_motor_steps[m];
            csv << '\n';
            csv.flush();
            std::cout << "step " << step << "  alive " << alive << "  births " << births
                      << "  deaths " << deaths << "  predation " << predation_deaths
                      << "  food " << food_cells << "  max generation " << max_generation << '\n';
        }
        if (live) {
            int event_count = 0;
            check(cudaMemcpy(&event_count, d.bite_event_count, sizeof(int),
                             cudaMemcpyDeviceToHost), "bite event count");
            if (event_count > 2 * P) {
                std::cerr << "Bite event buffer overflow at step " << step << '\n';
                std::exit(1);
            }
            if (event_count > 0) {
                download(recent_bites, d.bite_events, event_count);
                for (const BiteEvent &event : recent_bites)
                    bite_history.emplace_back(next_bite_event_id++, event);
            }
            check(cudaMemset(d.bite_event_count, 0, sizeof(int)), "bite event reset");
            while (!bite_history.empty() &&
                   (bite_history.front().second.step < step - 80 ||
                    bite_history.size() > 1024))
                bite_history.pop_front();
            JsonBuilder json(static_cast<size_t>(alive) * 180 + W * H + 512);
            json << "{\"step\":" << step << ",\"total\":" << steps
                 << ",\"width\":" << W << ",\"height\":" << H
                 << ",\"population\":" << alive << ",\"births\":" << births
                 << ",\"deaths\":" << deaths << ",\"predation\":"
                 << predation_deaths << ",\"bites\":" << bites
                 << ",\"food_cells\":" << food_cells
                 << ",\"mean_energy\":" << (alive ? energy / alive : 0)
                 << ",\"max_generation\":" << max_generation
                 << ",\"food_choice\":[";
            for (int food_type = 0; food_type < 3; ++food_type) {
                if (food_type) json << ',';
                uint64_t accepted = 0, opportunities = 0;
                double progress = 0, travel = 0;
                if (!behavior_history.empty()) {
                    const BehaviorTotals &now = behavior_history.back().second;
                    const BehaviorTotals &then = behavior_history.front().second;
                    for (int age = 0; age < 3; ++age) {
                        accepted += now.accepted[age][0][food_type] -
                                    then.accepted[age][0][food_type];
                        opportunities += now.opportunities[age][0][food_type] -
                                         then.opportunities[age][0][food_type];
                        progress += now.approach_progress[age][food_type] -
                                    then.approach_progress[age][food_type];
                        travel += now.approach_travel[age][food_type] -
                                  then.approach_travel[age][food_type];
                    }
                }
                json << '[' << accepted << ',' << opportunities << ','
                     << progress << ',' << travel << ']';
            }
            uint64_t empty_steps = 0, empty_eat = 0;
            if (!behavior_history.empty()) {
                const BehaviorTotals &now = behavior_history.back().second;
                const BehaviorTotals &then = behavior_history.front().second;
                for (int age = 0; age < 3; ++age) {
                    empty_steps += now.empty_steps[age][0] - then.empty_steps[age][0];
                    empty_eat += now.empty_eat[age][0] - then.empty_eat[age][0];
                }
            }
            json << "],\"empty_choice\":[" << empty_eat << ',' << empty_steps
                 << "],\"bite_events\":[";
            bool first_bite = true;
            for (const auto &[id, event] : bite_history) {
                if (!first_bite) json << ',';
                first_bite = false;
                json << '[' << id << ',' << event.step << ','
                     << event.attacker_id << ',' << event.prey_id << ','
                     << event.x << ',' << event.y << ']';
            }
            json << "],\"animals\":[";
            bool first = true;
            for (int a = 0; a < P; ++a) if (bodies[a].alive == 1) {
                const Animal &b = bodies[a];
                if (!first) json << ',';
                first = false;
                json << '[' << b.id << ',' << b.x << ',' << b.y << ','
                     << b.heading << ',' << b.energy << ',' << b.health << ','
                     << b.age << ',' << b.generation << ',' << b.neuron_count
                     << ',' << b.synapse_count << ',' << active_neurons[a]
                     << ',' << active_synapses[a] << ',' << b.fullness << ','
                     << b.pain << ',' << b.taste[0] << ',' << b.taste[1]
                     << ',' << b.taste[2] << ',' << b.taste[3] << ']';
            }
            json << "],\"food\":\"";
            for (int c = 0; c < W * H; ++c)
                json << (food[c] > 0 ? char('1' + kind[c]) : '0');
            json << "\"}";
            live->publish(json.str());
        }
        if (profile && live && !log) {
            auto phase_end = std::chrono::steady_clock::now();
            auto milliseconds = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            live_count_ms += milliseconds(phase_begin, phase_counted);
            live_download_ms += milliseconds(phase_counted, phase_downloaded);
            live_format_ms += milliseconds(phase_downloaded, phase_end);
            ++live_profile_reports;
        }
    };
    report(0, true);
    int64_t last_step = 0;
    double select_ms = 0, neural_ms = 0, world_ms = 0, report_ms = 0;
    int64_t profile_steps = 0, profile_reports = 0;
    auto simulation_start = std::chrono::steady_clock::now();
    for (int64_t step = 1; steps == 0 || step <= steps; ++step) {
        if (interrupted || (live && live->stop_requested())) break;
        if (live) {
            while (live->paused() && !live->stop_requested() && !interrupted)
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            if (live->stop_requested() || interrupted) break;
        }
        auto frame_start = std::chrono::steady_clock::now();
        if (profile) check(cudaDeviceSynchronize(), "profile start");
        auto select_start = std::chrono::steady_clock::now();
        if (block_brains) {
            check(cub::DeviceSelect::If(selection_temp, selection_temp_bytes, slots,
                  d.block_animals, d.block_animal_count, P,
                  BlockAnimal{d.animals}), "select block brains");
            check(cub::DeviceSelect::If(selection_temp, selection_temp_bytes, slots,
                  d.occupied_neurons, d.occupied_neuron_count, NEURON_CAPACITY,
                  FallbackOwnedSlot{d.neuron_owner, d.animals}),
                  "select fallback neurons");
            check(cub::DeviceSelect::If(selection_temp, selection_temp_bytes, slots,
                  d.occupied_synapses, d.occupied_synapse_count, SYNAPSE_CAPACITY,
                  FallbackOwnedSlot{d.synapse_owner, d.animals}),
                  "select fallback synapses");
        } else {
            check(cub::DeviceSelect::If(selection_temp, selection_temp_bytes, slots,
                  d.occupied_neurons, d.occupied_neuron_count, NEURON_CAPACITY,
                  OwnedSlot{d.neuron_owner}), "select occupied neurons");
            check(cub::DeviceSelect::If(selection_temp, selection_temp_bytes, slots,
                  d.occupied_synapses, d.occupied_synapse_count, SYNAPSE_CAPACITY,
                  OwnedSlot{d.synapse_owner}), "select occupied synapses");
        }
        int neuron_count = 0, synapse_count = 0, block_count = 0;
        check(cudaMemcpy(&neuron_count, d.occupied_neuron_count, sizeof(int),
                         cudaMemcpyDeviceToHost), "occupied neuron count");
        check(cudaMemcpy(&synapse_count, d.occupied_synapse_count, sizeof(int),
                         cudaMemcpyDeviceToHost), "occupied synapse count");
        if (block_brains)
            check(cudaMemcpy(&block_count, d.block_animal_count, sizeof(int),
                             cudaMemcpyDeviceToHost), "block brain count");
        auto neural_start = std::chrono::steady_clock::now();
        if (block_count)
            brain_block_step<<<block_count, 128, sizeof(BlockBrain)>>>(
                d, block_count, (step - 1) * TICKS);
        const bool kernel_sample = profile && !block_brains &&
                                   (step == 1 || step % 100 == 0);
        cudaEvent_t kernel_start[3][TICKS], kernel_end[3][TICKS];
        if (kernel_sample)
            for (int group = 0; group < 3; ++group)
                for (int t = 0; t < TICKS; ++t) {
                    check(cudaEventCreate(&kernel_start[group][t]), "profile event");
                    check(cudaEventCreate(&kernel_end[group][t]), "profile event");
                }
        for (int t = 0; t < TICKS; ++t) {
            int64_t tick = (step - 1) * TICKS + t;
            if (kernel_sample) check(cudaEventRecord(kernel_start[0][t]), "profile delivery");
            if (synapse_count) {
                if (legacy_neural || t == 0)
                    synapse_delivery<<<(synapse_count + NEURAL_BLOCK - 1) / NEURAL_BLOCK,
                                       NEURAL_BLOCK>>>(
                        d, tick, synapse_count);
                else
                    synapse_plasticity_delivery<<<(synapse_count + NEURAL_BLOCK - 1) / NEURAL_BLOCK,
                                                  NEURAL_BLOCK>>>(
                        d, tick - 1, tick, synapse_count);
            }
            if (kernel_sample) check(cudaEventRecord(kernel_end[0][t]), "profile delivery");
            if (kernel_sample) check(cudaEventRecord(kernel_start[1][t]), "profile neuron");
            if (neuron_count)
                neuron_step<<<(neuron_count + NEURAL_BLOCK - 1) / NEURAL_BLOCK,
                              NEURAL_BLOCK>>>(
                    d, tick, neuron_count);
            if (kernel_sample) check(cudaEventRecord(kernel_end[1][t]), "profile neuron");
            if (kernel_sample) check(cudaEventRecord(kernel_start[2][t]), "profile plasticity");
            if (synapse_count && (legacy_neural || t == TICKS - 1))
                synapse_plasticity<<<(synapse_count + NEURAL_BLOCK - 1) / NEURAL_BLOCK,
                                     NEURAL_BLOCK>>>(
                    d, tick, synapse_count);
            if (kernel_sample) check(cudaEventRecord(kernel_end[2][t]), "profile plasticity");
        }
        if (profile) check(cudaDeviceSynchronize(), "profile neural");
        if (kernel_sample) {
            float group_ms[3]{};
            for (int group = 0; group < 3; ++group)
                for (int t = 0; t < TICKS; ++t) {
                    float elapsed = 0;
                    check(cudaEventElapsedTime(&elapsed, kernel_start[group][t],
                                               kernel_end[group][t]), "profile elapsed");
                    group_ms[group] += elapsed;
                    cudaEventDestroy(kernel_start[group][t]);
                    cudaEventDestroy(kernel_end[group][t]);
                }
            std::cerr << "kernel step " << step << " ms synapse_update=" << group_ms[0]
                      << " neuron=" << group_ms[1]
                      << " plasticity_tail=" << group_ms[2] << '\n';
        }
        auto world_start = std::chrono::steady_clock::now();
        body_step<<<(P + 127) / 128, 128>>>(d);
        check(cudaMemsetAsync(d.cell_head, 0xff, W * H * sizeof(int)), "spatial reset");
        index_animals<<<(P + 127) / 128, 128>>>(d);
        bite_animals<<<(P + 127) / 128, 128>>>(d, step);
        finish_predation<<<(P + 127) / 128, 128>>>(d);
        if (neuron_count)
            homeostasis_step<<<(neuron_count + 255) / 256, 256>>>(d, neuron_count);
        if (block_count)
            homeostasis_block_animals<<<(block_count + 127) / 128, 128>>>(
                d, block_count);
        reclaim_dead_genomes<<<(P + 127) / 128, 128>>>(d, step);
        reproduce<<<(P + 127) / 128, 128>>>(d);
        regrow<<<(W * H + 255) / 256, 256>>>(d, step, seed, food_renewal);
        check(cudaMemsetAsync(d.cell_head, 0xff, W * H * sizeof(int)), "spatial reset");
        index_animals<<<(P + 127) / 128, 128>>>(d);
        update_senses<<<(P + 127) / 128, 128>>>(d);
        if (profile) check(cudaDeviceSynchronize(), "profile world");
        auto report_start = std::chrono::steady_clock::now();
        last_step = step;
        bool log = step % 100 == 0 || (steps > 0 && step == steps);
        if (log || (live && step % 2 == 0)) report(step, log);
        if (profile) {
            auto report_end = std::chrono::steady_clock::now();
            auto milliseconds = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            select_ms += milliseconds(select_start, neural_start);
            neural_ms += milliseconds(neural_start, world_start);
            world_ms += milliseconds(world_start, report_start);
            if (log || (live && step % 2 == 0)) {
                report_ms += milliseconds(report_start, report_end);
                ++profile_reports;
            }
            ++profile_steps;
            if (step % 100 == 0 || (steps > 0 && step == steps)) {
                std::cerr << "profile through step " << step
                          << " avg_ms select=" << select_ms / profile_steps
                          << " neural=" << neural_ms / profile_steps
                          << " world=" << world_ms / profile_steps
                          << " report=" << report_ms / profile_reports << '\n';
            }
        }
        if (live && live->speed() > 0) {
            int ms = live->speed() == 1 ? 50 : 12;
            std::this_thread::sleep_until(frame_start + std::chrono::milliseconds(ms));
        }
    }
    if (profile && profile_steps) {
        double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - simulation_start).count();
        double active_ms = select_ms + neural_ms + world_ms + report_ms;
        std::cerr << "profile total steps=" << profile_steps
                  << " wall_steps_per_second=" << profile_steps / wall_seconds
                  << " active_steps_per_second=" << profile_steps * 1000.0 / active_ms
                  << '\n';
        if (live_profile_reports)
            std::cerr << "profile live_report avg_ms count="
                      << live_count_ms / live_profile_reports
                      << " download=" << live_download_ms / live_profile_reports
                      << " format=" << live_format_ms / live_profile_reports << '\n';
    }
    if (last_step > 0 && (steps == 0 || last_step < steps) &&
        last_step % 100 != 0)
        report(last_step, true);
    write_world(output + ".png", d);
    write_brains(output, d);
    if (trace_animal >= 0) {
        int exc_arrivals[M], exc_releases[M], inh_arrivals[M];
        float exc_ns[M], inh_ns[M], peak_mv[M];
        check(cudaMemcpy(exc_arrivals, d.trace_exc_arrivals, sizeof(exc_arrivals),
                         cudaMemcpyDeviceToHost), "trace download");
        check(cudaMemcpy(exc_releases, d.trace_exc_releases, sizeof(exc_releases),
                         cudaMemcpyDeviceToHost), "trace download");
        check(cudaMemcpy(inh_arrivals, d.trace_inh_arrivals, sizeof(inh_arrivals),
                         cudaMemcpyDeviceToHost), "trace download");
        check(cudaMemcpy(exc_ns, d.trace_exc_ns, sizeof(exc_ns),
                         cudaMemcpyDeviceToHost), "trace download");
        check(cudaMemcpy(inh_ns, d.trace_inh_ns, sizeof(inh_ns),
                         cudaMemcpyDeviceToHost), "trace download");
        check(cudaMemcpy(peak_mv, d.trace_peak_mv, sizeof(peak_mv),
                         cudaMemcpyDeviceToHost), "trace download");
        std::ofstream trace(output + ".trace.csv");
        trace << "motor,excitatory_arrivals,nonzero_releases,total_excitatory_ns,"
                 "inhibitory_arrivals,total_inhibitory_ns,peak_voltage_mv,spikes\n";
        const char *names[M] = {"forward", "left", "right", "eat", "bite"};
        for (int m = 0; m < M; ++m) {
            int motor_neuron = bodies[trace_animal].ports[S + m];
            NeuronState state{};
            if (motor_neuron >= 0)
                check(cudaMemcpy(&state, d.neuron_state + motor_neuron,
                                 sizeof(state), cudaMemcpyDeviceToHost), "trace download");
            trace << names[m] << ',' << exc_arrivals[m] << ',' << exc_releases[m]
                  << ',' << exc_ns[m] << ',' << inh_arrivals[m] << ',' << inh_ns[m]
                  << ',' << peak_mv[m] << ',' << state.total_spikes << '\n';
        }
    }
    std::cout << "Wrote " << output << ", world image, and final brain tables\n";
    if (live) {
        live->finish();
        std::cout << "Live view remains open at http://127.0.0.1:"
                  << live_port << "/  Press Ctrl+C to close.\n" << std::flush;
        while (!interrupted)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    cudaFree(d.animals); cudaFree(d.neurons); cudaFree(d.synapses);
    cudaFree(d.neuron_state); cudaFree(d.synapse_state);
    cudaFree(d.neuron_owner); cudaFree(d.synapse_owner);
    cudaFree(d.occupied_neurons); cudaFree(d.occupied_synapses);
    cudaFree(d.occupied_neuron_count); cudaFree(d.occupied_synapse_count);
    cudaFree(d.block_animals); cudaFree(d.block_animal_count);
    cudaFree(selection_temp);
    cudaFree(d.neuron_next); cudaFree(d.synapse_next);
    cudaFree(d.neuron_port); cudaFree(d.neuron_clone);
    cudaFree(d.neuron_used); cudaFree(d.synapse_used);
    cudaFree(d.active_neuron_counts); cudaFree(d.active_synapse_counts);
    cudaFree(d.pending_exc); cudaFree(d.pending_inh); cudaFree(d.mod_event_tick);
    cudaFree(d.spike_history);
    cudaFree(d.bite_intent);
    cudaFree(d.trace_exc_arrivals); cudaFree(d.trace_exc_releases);
    cudaFree(d.trace_inh_arrivals); cudaFree(d.trace_exc_ns);
    cudaFree(d.trace_inh_ns); cudaFree(d.trace_peak_mv);
    cudaFree(d.founder_motor_steps); cudaFree(d.founder_observed_steps);
    cudaFree(d.senses); cudaFree(d.motors);
    cudaFree(d.food); cudaFree(d.kind); cudaFree(d.cell_head); cudaFree(d.cell_next);
    cudaFree(d.births); cudaFree(d.deaths); cudaFree(d.predation_deaths);
    cudaFree(d.bites); cudaFree(d.next_id);
    cudaFree(d.life_records); cudaFree(d.life_count);
    if (live) { cudaFree(d.bite_events); cudaFree(d.bite_event_count); }
}
