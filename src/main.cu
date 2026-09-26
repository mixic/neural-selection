#include <cuda_runtime.h>
#include <png.h>
#include "physiology.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace sim {
constexpr int P = 2048;
constexpr int N = 64;
constexpr int E = 384;
constexpr int S = 12;
constexpr int M = 4;
constexpr int W = 128;
constexpr int H = 128;
constexpr int D = 8;
constexpr int HISTORY = D + 1;
constexpr int TICKS = 16;
constexpr float PI = 3.14159265358979323846f;

struct Animal {
    float x, y, heading, energy, health, fullness, pain;
    float taste[3];
    unsigned long long rng;
    int id, generation, age, alive;
};
struct Device {
    Animal *animals;
    NeuronGene *neurons;
    SynapseGene *synapses;
    NeuronState *neuron_state;
    SynapseState *synapse_state;
    float *pending_exc, *pending_inh, *senses;
    unsigned char *spike_history;
    int *motors, *food, *kind, *births, *deaths, *next_id;
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

__global__ void init_world(Device d, unsigned long long seed) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W * H) return;
    auto rng = seed_for(seed, c + P);
    d.kind[c] = randint(rng, 3);
    d.food[c] = uniform(rng) < 0.28f ? 1 : 0;
}

__device__ void random_synapse(SynapseGene &g, Device d, int animal,
                               unsigned long long &rng) {
    g.pre = (short)randint(rng, N);
    g.post = (short)(S + randint(rng, N - S));
    g.delay = (unsigned char)(1 + randint(rng, D));
    g.branch = (unsigned char)randint(rng, DENDRITES);
    g.initial_weight_ns = 0.5f + uniform(rng) * 3.5f;
    g.utilization = 0.55f;
    g.recovery_ms = 450.0f;
    g.facilitation_ms = 0;
    if (d.neurons[animal * N + g.pre].type == 0 &&
        d.neurons[animal * N + g.post].type == 1) {
        // One fitted pyramidal-to-interneuron profile in Markram et al. (1998).
        g.utilization = 0.1f;
        g.recovery_ms = 30.0f;
        g.facilitation_ms = 1700.0f;
    }
    g.plasticity = 1;
    g.active = 1;
}

__global__ void init_animals(Device d, int initial, unsigned long long seed) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P) return;
    auto rng = seed_for(seed, a);
    Animal body{};
    body.rng = rng;
    if (a < initial) {
        body.x = uniform(rng) * (W - 1);
        body.y = uniform(rng) * (H - 1);
        body.heading = uniform(rng) * 2.0f * PI;
        body.energy = 14.0f;
        body.health = 10.0f;
        body.id = a;
        body.alive = 1;
    }
    for (int n = 0; n < N; ++n) {
        unsigned char type = n < S + M ? 0 : (unsigned char)randint(rng, 2);
        NeuronGene g = recorded_profile(type);
        g.bias_pa = n >= S && n < S + M
            ? 110.0f + uniform(rng) * 90.0f
            : uniform(rng) * 100.0f;
        g.active = (n < S + M || uniform(rng) < 0.5f) ? 1 : 0;
        d.neurons[a * N + n] = g;
        d.neuron_state[a * N + n] = initial_neuron_state(g);
        for (int branch = 0; branch < DENDRITES; ++branch)
            d.pending_exc[(a * N + n) * DENDRITES + branch] = 0;
        d.pending_inh[a * N + n] = 0;
        for (int delay = 0; delay < HISTORY; ++delay)
            d.spike_history[(delay * P + a) * N + n] = 0;
    }
    for (int e = 0; e < E; ++e) {
        SynapseGene g{};
        random_synapse(g, d, a, rng);
        g.active = (e < 120) ? 1 : 0;
        d.synapses[a * E + e] = g;
        d.synapse_state[a * E + e] = initial_synapse_state(g);
    }
    for (int s = 0; s < S; ++s) d.senses[a * S + s] = 0;
    for (int m = 0; m < M; ++m) d.motors[a * M + m] = 0;
    body.rng = rng;
    d.animals[a] = body;
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
}

__global__ void synapse_delivery(Device d, int tick) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * E) return;
    int a = i / E;
    if (d.animals[a].alive != 1) return;
    SynapseGene g = d.synapses[i];
    if (!g.active || !d.neurons[a * N + g.pre].active || !d.neurons[a * N + g.post].active) return;
    SynapseState state = d.synapse_state[i];
    state.arrived = tick >= g.delay &&
        d.spike_history[(((tick - g.delay) % HISTORY) * P + a) * N + g.pre];
    if (state.arrived) {
        float released = d.neurons[a * N + g.pre].type == 0
            ? synapse_release(g, state, tick) : state.weight_ns;
        if (d.neurons[a * N + g.pre].type == 0)
            atomicAdd(&d.pending_exc[(a * N + g.post) * DENDRITES + g.branch], released);
        else
            atomicAdd(&d.pending_inh[a * N + g.post], released);
    }
    d.synapse_state[i] = state;
}

__global__ void neuron_step(Device d, int tick) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * N) return;
    int a = i / N, n = i % N;
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
    float sensory = n < S ? d.senses[a * S + n] * 230.0f : 0.0f;
    neuron_advance(g, state, g.bias_pa + sensory, exc, inh);
    if (state.spike && n >= S && n < S + M)
        atomicAdd(&d.motors[a * M + n - S], 1);
    d.spike_history[((tick % HISTORY) * P + a) * N + n] = state.spike;
    d.neuron_state[i] = state;
}

__global__ void synapse_plasticity(Device d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * E) return;
    int a = i / E;
    if (d.animals[a].alive != 1) return;
    SynapseGene g = d.synapses[i];
    if (!g.active || !d.neurons[a * N + g.pre].active || !d.neurons[a * N + g.post].active) return;
    SynapseState s = d.synapse_state[i];
    bool excitatory_pair = d.neurons[a * N + g.pre].type == 0 &&
        d.neurons[a * N + g.post].type == 0;
    synapse_plasticity_step(g, s, s.arrived,
        d.neuron_state[a * N + g.post].spike, excitatory_pair);
    d.synapse_state[i] = s;
}

__global__ void regrow(Device d, int step, unsigned long long seed) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W * H || d.food[c] != 0) return;
    auto rng = seed_for(seed + step, c);
    if (uniform(rng) < 0.002f) atomicCAS(&d.food[c], 0, 1);
}

__global__ void body_step(Device d) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= P || d.animals[a].alive != 1) return;
    Animal b = d.animals[a];
    float thrust = d.motors[a * M + 0] / (float)TICKS;
    float left = d.motors[a * M + 1] / (float)TICKS;
    float right = d.motors[a * M + 2] / (float)TICKS;
    bool eat = d.motors[a * M + 3] > 0;
    for (int m = 0; m < M; ++m) d.motors[a * M + m] = 0;
    b.heading += (right - left) * 1.2f;
    b.x = clamp(b.x + cosf(b.heading) * thrust * 1.5f, 0, W - 0.001f);
    b.y = clamp(b.y + sinf(b.heading) * thrust * 1.5f, 0, H - 0.001f);
    b.energy -= 0.012f + thrust * 0.02f;
    b.fullness = fmaxf(0, b.fullness - 0.05f);
    b.pain *= 0.8f;
    for (int k = 0; k < 3; ++k) b.taste[k] *= 0.7f;
    if (eat) {
        int c = cell(b.x, b.y);
        if (harvest(&d.food[c])) {
            int kind = d.kind[c];
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
        b.alive = 0;
        atomicAdd(d.deaths, 1);
    }
    d.animals[a] = b;
}

__device__ void mutate(Device d, int child, unsigned long long &rng) {
    int mutations = 1 + randint(rng, 4);
    for (int m = 0; m < mutations; ++m) {
        int choice = randint(rng, 8);
        int n = 16 + randint(rng, N - 16);
        int e = randint(rng, E);
        if (choice == 0) {
            d.neurons[child * N + n].active ^= 1;
        } else if (choice == 1) {
            auto active = d.neurons[child * N + n].active;
            d.neurons[child * N + n] = recorded_profile((unsigned char)randint(rng, 2));
            d.neurons[child * N + n].active = active;
        } else if (choice == 2) {
            auto &g = d.neurons[child * N + n];
            g.threshold_mv = clamp(g.threshold_mv + (uniform(rng) - 0.5f) * 2, -60, -35);
            g.bias_pa = clamp(g.bias_pa + (uniform(rng) - 0.5f) * 20, -50, 250);
        } else if (choice == 3) {
            d.synapses[child * E + e].active ^= 1;
        } else if (choice == 4) {
            auto &g = d.synapses[child * E + e];
            random_synapse(g, d, child, rng);
        } else if (choice == 5) {
            auto &g = d.synapses[child * E + e];
            g.initial_weight_ns = clamp(g.initial_weight_ns + (uniform(rng) - 0.5f) * 0.5f, 0.1f, 8);
        } else if (choice == 6) {
            auto &g = d.synapses[child * E + e];
            g.plasticity ^= 1;
            g.utilization = clamp(g.utilization + (uniform(rng) - 0.5f) * 0.1f, 0.02f, 0.8f);
            g.recovery_ms = clamp(g.recovery_ms + (uniform(rng) - 0.5f) * 50, 20, 1000);
        } else {
            auto &g = d.synapses[child * E + e];
            g.delay = (unsigned char)(1 + randint(rng, D));
            g.facilitation_ms = g.facilitation_ms == 0
                ? 500.0f + uniform(rng) * 3500.0f
                : (uniform(rng) < 0.1f ? 0.0f
                   : clamp(g.facilitation_ms + (uniform(rng) - 0.5f) * 200, 100, 4000));
            d.neurons[child * N + n].spike_adapt_pa = clamp(
                d.neurons[child * N + n].spike_adapt_pa + (uniform(rng) - 0.5f) * 5, 0, 150);
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
    parent.energy -= 10;
    parent.rng = rng;
    d.animals[a] = parent;
    Animal newborn{};
    newborn.x = clamp(parent.x + (uniform(rng) - 0.5f) * 2, 0, W - 0.001f);
    newborn.y = clamp(parent.y + (uniform(rng) - 0.5f) * 2, 0, H - 0.001f);
    newborn.heading = uniform(rng) * 2 * PI;
    newborn.energy = 8;
    newborn.health = 10;
    newborn.generation = parent.generation + 1;
    newborn.id = atomicAdd(d.next_id, 1);
    for (int n = 0; n < N; ++n) {
        d.neurons[child * N + n] = d.neurons[a * N + n];
        d.neuron_state[child * N + n] = initial_neuron_state(d.neurons[child * N + n]);
        for (int branch = 0; branch < DENDRITES; ++branch)
            d.pending_exc[(child * N + n) * DENDRITES + branch] = 0;
        d.pending_inh[child * N + n] = 0;
        for (int delay = 0; delay < HISTORY; ++delay)
            d.spike_history[(delay * P + child) * N + n] = 0;
    }
    for (int e = 0; e < E; ++e)
        d.synapses[child * E + e] = d.synapses[a * E + e];
    mutate(d, child, rng);
    for (int n = 0; n < N; ++n)
        d.neuron_state[child * N + n] = initial_neuron_state(d.neurons[child * N + n]);
    for (int e = 0; e < E; ++e)
        d.synapse_state[child * E + e] = initial_synapse_state(d.synapses[child * E + e]);
    for (int s = 0; s < S; ++s) d.senses[child * S + s] = 0;
    for (int m = 0; m < M; ++m) d.motors[child * M + m] = 0;
    newborn.rng = rng;
    newborn.alive = -1;
    d.animals[child] = newborn;
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
    download(animals, d.animals, P);
    download(neurons, d.neurons, P * N);
    download(synapses, d.synapses, P * E);
    download(states, d.synapse_state, P * E);
    std::ofstream cells(prefix + ".neurons.csv");
    std::ofstream edges(prefix + ".synapses.csv");
    cells << "animal_id,generation,neuron,type,capacitance_pf,leak_ns,rest_mv,threshold_mv,slope_mv,adapt_ns,adapt_tau_ms,spike_adapt_pa,reset_mv,bias_pa\n";
    edges << "animal_id,generation,synapse,pre,post,branch,delay_ms,initial_weight_ns,learned_weight_ns,utilization,recovery_ms,facilitation_ms,plasticity\n";
    for (int a = 0; a < P; ++a) if (animals[a].alive == 1) {
        for (int n = 0; n < N; ++n) {
            const auto &g = neurons[a * N + n];
            if (g.active) cells << animals[a].id << ',' << animals[a].generation << ',' << n << ','
                << (int)g.type << ',' << g.capacitance_pf << ',' << g.leak_ns << ','
                << g.rest_mv << ',' << g.threshold_mv << ',' << g.slope_mv << ','
                << g.adapt_ns << ',' << g.adapt_tau_ms << ',' << g.spike_adapt_pa << ','
                << g.reset_mv << ',' << g.bias_pa << '\n';
        }
        for (int e = 0; e < E; ++e) {
            const auto &g = synapses[a * E + e];
            if (g.active && neurons[a * N + g.pre].active && neurons[a * N + g.post].active)
                edges << animals[a].id << ',' << animals[a].generation << ',' << e << ','
                    << g.pre << ',' << g.post << ',' << (int)g.branch << ',' << (int)g.delay << ',' << g.initial_weight_ns << ','
                    << states[a * E + e].weight_ns << ',' << g.utilization << ','
                    << g.recovery_ms << ',' << g.facilitation_ms << ',' << (int)g.plasticity << '\n';
        }
    }
}

} // namespace sim

int main(int argc, char **argv) {
    using namespace sim;
    int steps = 2000, initial = 1024;
    unsigned long long seed = 1;
    std::string output = "run.csv";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--population") && i + 1 < argc) initial = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) output = argv[++i];
        else { std::cerr << "Usage: neural_selection [--steps N] [--population 1..2048] [--seed N] [--output file.csv]\n"; return 2; }
    }
    if (steps < 1 || initial < 1 || initial > P) { std::cerr << "Invalid steps or population\n"; return 2; }
    int count = 0;
    check(cudaGetDeviceCount(&count), "GPU discovery");
    if (count < 1) { std::cerr << "No NVIDIA GPU found\n"; return 1; }
    Device d{};
    allocate(d.animals, P); allocate(d.neurons, P * N); allocate(d.synapses, P * E);
    allocate(d.neuron_state, P * N); allocate(d.synapse_state, P * E);
    allocate(d.pending_exc, P * N * DENDRITES); allocate(d.pending_inh, P * N);
    allocate(d.spike_history, HISTORY * P * N); allocate(d.senses, P * S);
    allocate(d.motors, P * M);
    allocate(d.food, W * H); allocate(d.kind, W * H);
    allocate(d.births, 1); allocate(d.deaths, 1); allocate(d.next_id, 1);
    check(cudaMemset(d.births, 0, sizeof(int)), "counter reset");
    check(cudaMemset(d.deaths, 0, sizeof(int)), "counter reset");
    check(cudaMemcpy(d.next_id, &initial, sizeof(int), cudaMemcpyHostToDevice), "counter init");
    init_world<<<(W * H + 255) / 256, 256>>>(d, seed);
    init_animals<<<(P + 127) / 128, 128>>>(d, initial, seed);
    update_senses<<<(P + 127) / 128, 128>>>(d);
    check(cudaDeviceSynchronize(), "initialization");
    std::ofstream csv(output);
    if (!csv) { std::cerr << "Cannot open output file\n"; return 1; }
    csv << "step,population,births,deaths,max_generation,mean_energy,mean_neurons,mean_synapses\n";
    std::vector<Animal> bodies;
    std::vector<NeuronGene> genes;
    std::vector<SynapseGene> edges;
    auto report = [&](int step) {
        check(cudaDeviceSynchronize(), "simulation step");
        check(cudaGetLastError(), "CUDA kernel");
        download(bodies, d.animals, P);
        download(genes, d.neurons, P * N);
        download(edges, d.synapses, P * E);
        int alive = 0, max_generation = 0, births = 0, deaths = 0;
        double energy = 0, neurons = 0, synapses = 0;
        for (int a = 0; a < P; ++a) if (bodies[a].alive == 1) {
            ++alive; energy += bodies[a].energy;
            max_generation = std::max(max_generation, bodies[a].generation);
            for (int n = 0; n < N; ++n) neurons += genes[a * N + n].active;
            for (int e = 0; e < E; ++e) if (edges[a * E + e].active &&
                genes[a * N + edges[a * E + e].pre].active &&
                genes[a * N + edges[a * E + e].post].active) ++synapses;
        }
        check(cudaMemcpy(&births, d.births, sizeof(int), cudaMemcpyDeviceToHost), "birth count");
        check(cudaMemcpy(&deaths, d.deaths, sizeof(int), cudaMemcpyDeviceToHost), "death count");
        csv << step << ',' << alive << ',' << births << ',' << deaths << ',' << max_generation << ','
            << (alive ? energy / alive : 0) << ',' << (alive ? neurons / alive : 0) << ','
            << (alive ? synapses / alive : 0) << '\n';
        csv.flush();
        std::cout << "step " << step << "  alive " << alive << "  births " << births
                  << "  deaths " << deaths << "  max generation " << max_generation << '\n';
    };
    report(0);
    for (int step = 1; step <= steps; ++step) {
        for (int t = 0; t < TICKS; ++t) {
            int tick = (step - 1) * TICKS + t;
            synapse_delivery<<<(P * E + 255) / 256, 256>>>(d, tick);
            neuron_step<<<(P * N + 255) / 256, 256>>>(d, tick);
            synapse_plasticity<<<(P * E + 255) / 256, 256>>>(d);
        }
        body_step<<<(P + 127) / 128, 128>>>(d);
        reproduce<<<(P + 127) / 128, 128>>>(d);
        regrow<<<(W * H + 255) / 256, 256>>>(d, step, seed);
        update_senses<<<(P + 127) / 128, 128>>>(d);
        if (step % 100 == 0 || step == steps) report(step);
    }
    write_world(output + ".png", d);
    write_brains(output, d);
    std::cout << "Wrote " << output << ", world image, and final brain tables\n";
    cudaFree(d.animals); cudaFree(d.neurons); cudaFree(d.synapses);
    cudaFree(d.neuron_state); cudaFree(d.synapse_state);
    cudaFree(d.pending_exc); cudaFree(d.pending_inh); cudaFree(d.spike_history);
    cudaFree(d.senses); cudaFree(d.motors);
    cudaFree(d.food); cudaFree(d.kind); cudaFree(d.births); cudaFree(d.deaths); cudaFree(d.next_id);
}
