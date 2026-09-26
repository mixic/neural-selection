#include <cuda_runtime.h>
#include <png.h>
#include "physiology.cuh"
#include "live_view.hpp"

#include <algorithm>
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
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace sim {
constexpr int P = 2048;
constexpr int NEURON_CAPACITY = 262144;
constexpr int SYNAPSE_CAPACITY = 524288;
constexpr int INITIAL_HIDDEN = 48;
constexpr int INITIAL_SYNAPSES = 120;
constexpr int LIFE_RECORD_CAPACITY = P * 16;
constexpr int LIFE_SAMPLE_INTERVAL = 10;
constexpr int S = 16;
constexpr int M = 5;
constexpr int W = 128;
constexpr int H = 128;
constexpr int D = 8;
constexpr int HISTORY = D + 1;
constexpr int TICKS = 16;
constexpr float BODY_UPKEEP = 0.003f;
constexpr float NEURON_UPKEEP = 0.00002f;
constexpr float BITE_DAMAGE = 2.0f;
constexpr float BITE_ENERGY_COST = 0.02f;
constexpr float MEAT_ENERGY_FRACTION = 0.45f;
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
struct Device {
    Animal *animals;
    NeuronGene *neurons;
    SynapseGene *synapses;
    NeuronState *neuron_state;
    SynapseState *synapse_state;
    float *pending_exc, *pending_inh, *senses;
    unsigned long long *mod_event_tick;
    unsigned char *spike_history;
    int *neuron_owner, *synapse_owner, *neuron_next, *synapse_next;
    int *neuron_port, *neuron_clone;
    uint32_t *neuron_used, *synapse_used;
    int *active_neuron_counts, *active_synapse_counts;
    unsigned char *bite_intent;
    int *motors, *food, *kind, *cell_head, *cell_next;
    int *births, *deaths, *predation_deaths, *bites, *next_id;
    LifeRecord *life_records;
    int *life_count;
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
                                  const NeuronGene &gene, int next) {
    d.neuron_owner[n] = owner;
    d.neuron_next[n] = next;
    d.neuron_port[n] = port;
    d.neurons[n] = gene;
    d.neuron_state[n] = initial_neuron_state(gene);
    for (int branch = 0; branch < DENDRITES; ++branch)
        d.pending_exc[n * DENDRITES + branch] = 0;
    d.pending_inh[n] = 0;
    for (int k = 0; k < 3; ++k) d.mod_event_tick[n * 3 + k] = 0;
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
    g.pre = neuron_at(d, body, randint(rng, body.neuron_count));
    g.post = body.ports[S + randint(rng, M)];
    for (int attempt = 0; attempt < 8; ++attempt) {
        int target = neuron_at(d, body, randint(rng, body.neuron_count));
        if (d.neuron_port[target] >= S || d.neuron_port[target] < 0) {
            g.post = target;
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
        unsigned char type = port >= 0 ? 0 : (unsigned char)randint(rng, 2);
        NeuronGene g = type == 0 ? cortical_l5_profile() : recorded_profile(type);
        if (port < 0) {
            float u = uniform(rng);
            g.modulator = u < 0.04f ? 1 : (u < 0.08f ? 2 :
                          (u < 0.12f ? 3 : 0));
        }
        g.bias_pa = port >= S && port < S + M - 1
            ? 110.0f + uniform(rng) * 90.0f
            : uniform(rng) * 100.0f;
        g.active = port >= 0 || uniform(rng) < 0.5f;
        initialize_neuron(d, n, a, port, g, body.neuron_head);
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
}

__global__ void synapse_delivery(Device d, int64_t tick) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= SYNAPSE_CAPACITY) return;
    int a = d.synapse_owner[i];
    if (a < 0) return;
    if (d.animals[a].alive != 1) return;
    SynapseGene g = d.synapses[i];
    if (!g.active || d.neuron_owner[g.pre] != a ||
        d.neuron_owner[g.post] != a || !d.neurons[g.pre].active ||
        !d.neurons[g.post].active) return;
    SynapseState state = d.synapse_state[i];
    state.arrived = tick >= g.delay &&
        d.spike_history[((tick - g.delay) % HISTORY) * NEURON_CAPACITY + g.pre];
    if (state.arrived) {
        NeuronGene source = d.neurons[g.pre];
        float released = source.type == 0 || source.modulator
            ? synapse_release(g, state, tick) : state.weight_ns;
        if (source.modulator) {
            if (released > 0)
                atomicMax(&d.mod_event_tick[g.post * 3 + source.modulator - 1],
                          static_cast<unsigned long long>(tick + 1));
        } else if (source.type == 0) {
            float postsynaptic_scale = expf(
                d.neuron_state[g.post].log_ampa_scale);
            atomicAdd(&d.pending_exc[g.post * DENDRITES + g.branch],
                      released * postsynaptic_scale);
        }
        else
            atomicAdd(&d.pending_inh[g.post], released);
    }
    d.synapse_state[i] = state;
}

__global__ void neuron_step(Device d, int64_t tick) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NEURON_CAPACITY) return;
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
    float sensory = port >= 0 && port < S ?
        d.senses[a * S + port] * 230.0f : 0.0f;
    neuron_advance(g, state, g.bias_pa + sensory, exc, inh);
    if (state.spike && port >= S && port < S + M)
        atomicAdd(&d.motors[a * M + port - S], 1);
    d.spike_history[(tick % HISTORY) * NEURON_CAPACITY + i] = state.spike;
    d.neuron_state[i] = state;
}

__global__ void synapse_plasticity(Device d, int64_t tick) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= SYNAPSE_CAPACITY) return;
    int a = d.synapse_owner[i];
    if (a < 0) return;
    if (d.animals[a].alive != 1) return;
    SynapseGene g = d.synapses[i];
    if (!g.active || d.neuron_owner[g.pre] != a ||
        d.neuron_owner[g.post] != a || !d.neurons[g.pre].active ||
        !d.neurons[g.post].active) return;
    SynapseState s = d.synapse_state[i];
    bool excitatory_pair = d.neurons[g.pre].type == 0 &&
        d.neurons[g.pre].modulator == 0 &&
        d.neurons[g.post].type == 0;
    int target = g.post * 3;
    unsigned char mod_event = 0;
    if (d.mod_event_tick[target] == static_cast<unsigned long long>(tick + 1))
        mod_event |= MOD_NE;
    if (d.mod_event_tick[target + 1] == static_cast<unsigned long long>(tick + 1))
        mod_event |= MOD_5HT;
    if (d.mod_event_tick[target + 2] == static_cast<unsigned long long>(tick + 1))
        mod_event |= MOD_DA;
    synapse_plasticity_step(g, s, s.arrived,
        d.neuron_state[g.post].spike, excitatory_pair,
        d.neuron_state[g.post].recent_depol_mv, true, mod_event);
    d.synapse_state[i] = s;
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
    float thrust = d.motors[a * M + 0] / (float)TICKS;
    float left = d.motors[a * M + 1] / (float)TICKS;
    float right = d.motors[a * M + 2] / (float)TICKS;
    bool eat = d.motors[a * M + 3] > 0;
    d.bite_intent[a] = d.motors[a * M + 4] > 0;
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
    int food_cell = cell(b.x, b.y);
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

__global__ void bite_animals(Device d) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P || d.animals[a].alive != 1 || !d.bite_intent[a]) return;
    int c = cell(d.animals[a].x, d.animals[a].y);
    for (int prey = d.cell_head[c]; prey >= 0; prey = d.cell_next[prey]) {
        if (prey == a || d.animals[prey].alive != 1) continue;
        float previous_health = atomicAdd(&d.animals[prey].health, -BITE_DAMAGE);
        atomicAdd(&d.animals[prey].pain, BITE_DAMAGE);
        atomicAdd(d.bites, 1);
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

__global__ void homeostasis_step(Device d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NEURON_CAPACITY) return;
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
                d.neurons[g.pre].type == 0 &&
                d.neurons[g.pre].modulator == 0 &&
                d.neurons[g.post].type == 0;
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

__device__ void mutate(Device d, int child, unsigned long long &rng) {
    int mutations = 1 + randint(rng, 4);
    for (int m = 0; m < mutations; ++m) {
        int choice = randint(rng, 10);
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
                    unsigned char type = (unsigned char)randint(rng, 2);
                    NeuronGene g = type == 0 ? cortical_l5_profile() : recorded_profile(type);
                    g.bias_pa = uniform(rng) * 100.0f;
                    g.modulator = uniform(rng) < 0.12f ?
                        (unsigned char)(1 + randint(rng, 3)) : 0;
                    initialize_neuron(d, added, child, -1, g, body.neuron_head);
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
            unsigned char type = (unsigned char)randint(rng, 2);
            unsigned char modulator = d.neurons[n].modulator;
            d.neurons[n] = type == 0 ? cortical_l5_profile() : recorded_profile(type);
            d.neurons[n].active = active;
            d.neurons[n].modulator = modulator;
        } else if (choice == 2 && n >= 0) {
            auto &g = d.neurons[n];
            g.threshold_mv = clamp(g.threshold_mv + (uniform(rng) - 0.5f) * 2, -60, -35);
            g.bias_pa = clamp(g.bias_pa + (uniform(rng) - 0.5f) * 20, -50, 250);
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
                g.release_sites + (uniform(rng) < 0.5f ? -1 : 1), 1, 24);
        } else if (choice == 6 && e >= 0) {
            auto &g = d.synapses[e];
            g.plasticity = (unsigned char)randint(rng, 6);
            g.mod_receptors = g.plasticity == 3 ?
                (unsigned char)(1 + randint(rng, 3)) :
                (g.plasticity >= 4 ? MOD_DA : 0);
            g.utilization = clamp(g.utilization + (uniform(rng) - 0.5f) * 0.1f, 0.02f, 0.8f);
            g.recovery_ms = clamp(g.recovery_ms + (uniform(rng) - 0.5f) * 50, 20, 1000);
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
            d.neurons[n].modulator = (unsigned char)randint(rng, 4);
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
        initialize_neuron(d, n, child, port, d.neurons[old], newborn.neuron_head);
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
        d.neuron_state[n] = initial_neuron_state(d.neurons[n]);
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
    cells << "animal_id,generation,neuron,port,active,type,modulator,capacitance_pf,leak_ns,rest_mv,threshold_mv,slope_mv,adapt_ns,adapt_tau_ms,spike_adapt_pa,reset_mv,bias_pa,afterdepol_spike_pa,afterdepol_tau_ms,ampa_scale\n";
    edges << "animal_id,generation,synapse,active,pre,post,branch,delay_ms,initial_weight_ns,learned_weight_ns,spine_volume,utilization,recovery_ms,facilitation_ms,release_sites,plasticity,mod_receptors\n";
    for (int a = 0; a < P; ++a) if (animals[a].alive == 1) {
        for (int n = animals[a].neuron_head, visited = 0;
             n >= 0 && visited < animals[a].neuron_count;
             n = neuron_next[n], ++visited) {
            const auto &g = neurons[n];
            cells << animals[a].id << ',' << animals[a].generation << ',' << n << ','
                << neuron_port[n] << ',' << (int)g.active << ','
                << (int)g.type << ',' << (int)g.modulator << ','
                << g.capacitance_pf << ',' << g.leak_ns << ','
                << g.rest_mv << ',' << g.threshold_mv << ',' << g.slope_mv << ','
                << g.adapt_ns << ',' << g.adapt_tau_ms << ',' << g.spike_adapt_pa << ','
                << g.reset_mv << ',' << g.bias_pa << ','
                << g.afterdepol_spike_pa << ',' << g.afterdepol_tau_ms << ','
                << expf(neuron_states[n].log_ampa_scale) << '\n';
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
                << (int)g.plasticity << ',' << (int)g.mod_receptors << '\n';
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
    bool visualize = false, steps_specified = false;
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
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) live_port = atoi(argv[++i]);
        else { std::cerr << "Usage: neural_selection [--steps N|0] [--population 1..2048] [--seed N] [--food-renewal probability] [--output file.csv] [--visualize] [--port 1..65535]\n"; return 2; }
    }
    if (visualize && !steps_specified) steps = 0;
    if (steps < 0 || steps > INT64_MAX / TICKS ||
        initial < 1 || initial > P || !std::isfinite(food_renewal) ||
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
    allocate(d.animals, P);
    allocate(d.neurons, NEURON_CAPACITY); allocate(d.synapses, SYNAPSE_CAPACITY);
    allocate(d.neuron_state, NEURON_CAPACITY);
    allocate(d.synapse_state, SYNAPSE_CAPACITY);
    allocate(d.neuron_owner, NEURON_CAPACITY);
    allocate(d.synapse_owner, SYNAPSE_CAPACITY);
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
    allocate(d.mod_event_tick, NEURON_CAPACITY * 3);
    allocate(d.spike_history, HISTORY * NEURON_CAPACITY);
    allocate(d.senses, P * S);
    allocate(d.motors, P * M); allocate(d.bite_intent, P);
    allocate(d.food, W * H); allocate(d.kind, W * H);
    allocate(d.cell_head, W * H); allocate(d.cell_next, P);
    allocate(d.births, 1); allocate(d.deaths, 1);
    allocate(d.predation_deaths, 1); allocate(d.bites, 1);
    allocate(d.next_id, 1);
    allocate(d.life_records, LIFE_RECORD_CAPACITY);
    allocate(d.life_count, 1);
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
    csv << "step,population,births,deaths,predation_deaths,bites,food_cells,max_generation,mean_energy,mean_neurons,mean_synapses,allocated_neurons,allocated_synapses,max_neuron_slots,max_synapse_slots\n";
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
    std::vector<int> active_neurons, active_synapses;
    std::vector<int> food, kind;
    std::vector<uint32_t> neuron_used, synapse_used;
    auto report = [&](int64_t step, bool log) {
        count_brains<<<(P + 127) / 128, 128>>>(d);
        check(cudaDeviceSynchronize(), "simulation step");
        check(cudaGetLastError(), "CUDA kernel");
        download(bodies, d.animals, P);
        download(active_neurons, d.active_neuron_counts, P);
        download(active_synapses, d.active_synapse_counts, P);
        download(food, d.food, W * H);
        if (live) download(kind, d.kind, W * H);
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
                << max_neuron_slots << ',' << max_synapse_slots << '\n';
            csv.flush();
            std::cout << "step " << step << "  alive " << alive << "  births " << births
                      << "  deaths " << deaths << "  predation " << predation_deaths
                      << "  food " << food_cells << "  max generation " << max_generation << '\n';
        }
        if (live) {
            std::ostringstream json;
            json.setf(std::ios::fixed);
            json.precision(2);
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
                 << "],\"animals\":[";
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
    };
    report(0, true);
    int64_t last_step = 0;
    for (int64_t step = 1; steps == 0 || step <= steps; ++step) {
        if (interrupted || (live && live->stop_requested())) break;
        if (live) {
            while (live->paused() && !live->stop_requested() && !interrupted)
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            if (live->stop_requested() || interrupted) break;
        }
        auto frame_start = std::chrono::steady_clock::now();
        for (int t = 0; t < TICKS; ++t) {
            int64_t tick = (step - 1) * TICKS + t;
            synapse_delivery<<<(SYNAPSE_CAPACITY + 255) / 256, 256>>>(d, tick);
            neuron_step<<<(NEURON_CAPACITY + 255) / 256, 256>>>(d, tick);
            synapse_plasticity<<<(SYNAPSE_CAPACITY + 255) / 256, 256>>>(d, tick);
        }
        body_step<<<(P + 127) / 128, 128>>>(d);
        check(cudaMemsetAsync(d.cell_head, 0xff, W * H * sizeof(int)), "spatial reset");
        index_animals<<<(P + 127) / 128, 128>>>(d);
        bite_animals<<<(P + 127) / 128, 128>>>(d);
        finish_predation<<<(P + 127) / 128, 128>>>(d);
        homeostasis_step<<<(NEURON_CAPACITY + 255) / 256, 256>>>(d);
        reclaim_dead_genomes<<<(P + 127) / 128, 128>>>(d, step);
        reproduce<<<(P + 127) / 128, 128>>>(d);
        regrow<<<(W * H + 255) / 256, 256>>>(d, step, seed, food_renewal);
        check(cudaMemsetAsync(d.cell_head, 0xff, W * H * sizeof(int)), "spatial reset");
        index_animals<<<(P + 127) / 128, 128>>>(d);
        update_senses<<<(P + 127) / 128, 128>>>(d);
        last_step = step;
        bool log = step % 100 == 0 || (steps > 0 && step == steps);
        if (log || (live && step % 2 == 0)) report(step, log);
        if (live && live->speed() > 0) {
            int ms = live->speed() == 1 ? 50 : 12;
            std::this_thread::sleep_until(frame_start + std::chrono::milliseconds(ms));
        }
    }
    if (last_step > 0 && (steps == 0 || last_step < steps) &&
        last_step % 100 != 0)
        report(last_step, true);
    write_world(output + ".png", d);
    write_brains(output, d);
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
    cudaFree(d.neuron_next); cudaFree(d.synapse_next);
    cudaFree(d.neuron_port); cudaFree(d.neuron_clone);
    cudaFree(d.neuron_used); cudaFree(d.synapse_used);
    cudaFree(d.active_neuron_counts); cudaFree(d.active_synapse_counts);
    cudaFree(d.pending_exc); cudaFree(d.pending_inh); cudaFree(d.mod_event_tick);
    cudaFree(d.spike_history);
    cudaFree(d.bite_intent);
    cudaFree(d.senses); cudaFree(d.motors);
    cudaFree(d.food); cudaFree(d.kind); cudaFree(d.cell_head); cudaFree(d.cell_next);
    cudaFree(d.births); cudaFree(d.deaths); cudaFree(d.predation_deaths);
    cudaFree(d.bites); cudaFree(d.next_id);
    cudaFree(d.life_records); cudaFree(d.life_count);
}
