#include "physiology.cuh"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace sim;

struct Result {
    float peak_mv, final_mv, released_ns;
    int arrivals, spikes;
};

__host__ __device__ Result replay() {
    NeuronGene cell = cortical_l5_profile();
    NeuronState state = initial_neuron_state(cell, 19);
    SynapseGene synapse{};
    synapse.initial_weight_ns = 29.0f / (60.0f * 0.75f);
    synapse.utilization = 0.75f;
    synapse.recovery_ms = 50.0f;
    synapse.release_sites = 50;
    synapse.slow_recovery_ms = 7500.0f;
    synapse.slow_depletion = 0.023f;
    SynapseState release = initial_synapse_state(synapse, 31);
    Result result{cell.rest_mv, cell.rest_mv, 0, 0, 0};
    for (int tick = 0; tick < 2000; ++tick) {
        float excitation[DENDRITES]{};
        if (tick % 137 == 0) {
            excitation[0] = synapse_release(synapse, release, tick);
            result.released_ns += excitation[0];
            ++result.arrivals;
        }
        neuron_advance(cell, state, 0, excitation, 0);
        result.peak_mv = fmaxf(result.peak_mv, state.voltage_mv);
    }
    result.final_mv = state.voltage_mv;
    result.spikes = state.total_spikes;
    return result;
}

__global__ void gpu_replay(Result *result) { *result = replay(); }

int main() {
    Result *device = nullptr;
    if (cudaMalloc(&device, sizeof(Result)) != cudaSuccess) return EXIT_FAILURE;
    gpu_replay<<<1, 1>>>(device);
    Result gpu{};
    cudaError_t status = cudaMemcpy(&gpu, device, sizeof(gpu), cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (status != cudaSuccess) return EXIT_FAILURE;
    Result cpu = replay();
    std::cout << "CPU/GPU relay: arrivals=" << gpu.arrivals
              << " released_ns=" << gpu.released_ns
              << " peak_mv=" << gpu.peak_mv
              << " final_mv=" << gpu.final_mv
              << " spikes=" << gpu.spikes << '\n';
    return gpu.arrivals == cpu.arrivals && gpu.spikes == cpu.spikes &&
           std::fabs(gpu.released_ns - cpu.released_ns) < 0.01f &&
           std::fabs(gpu.peak_mv - cpu.peak_mv) < 0.01f &&
           std::fabs(gpu.final_mv - cpu.final_mv) < 0.01f ? EXIT_SUCCESS
                                                       : EXIT_FAILURE;
}
