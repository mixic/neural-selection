// Exercise the production intervention against learning, release dynamics,
// complete cohort recording, and reuse of an original animal's slot.
#ifdef NDEBUG
#undef NDEBUG
#endif
#define main ecological_application_main
#include "../src/main.cu"
#undef main
#include <cassert>

__global__ void check_intervention(sim::Device d) {
    using namespace sim;
    SynapseGene g{};
    g.pre = 0; g.post = 1; g.plasticity = 1;
    g.initial_weight_ns = 2; g.utilization = 1.0f;
    g.slow_depletion = .2f; g.slow_recovery_ms = 1000;
    g.recovery_ms = 100; g.facilitation_ms = 20; g.release_sites = 1;
    d.neurons[0].type = d.neurons[1].type = 0;
    d.neurons[0].active = d.neurons[1].active = 1;
    d.neuron_state[1].spike = 1;
    d.synapse_state[0] = initial_synapse_state(g, 42);
    d.synapse_state[0].pre_age_ms = 0;
    d.synapse_state[0].arrived = 1;
    SynapseState baseline = d.synapse_state[0];
    plasticity_synapse(d, g, d.synapse_state[0], 0);
    assert(d.synapse_state[0].weight_ns == baseline.weight_ns);
    assert(d.synapse_state[0].pre_age_ms == baseline.pre_age_ms);
    // Every implemented learning profile must freeze its full learning state,
    // including delayed eligibility and structural growth.
    d.neuron_state[1].recent_depol_mv = 10;
    for (int profile = 1; profile <= 5; ++profile) {
        g.plasticity = profile;
        g.mod_receptors = MOD_NE | MOD_5HT | MOD_DA;
        SynapseState before = initial_synapse_state(g, 42);
        before.pre_age_ms = 0; before.arrived = 1;
        before.eligible_ltp = before.eligible_ltd = 2;
        before.structural_age_ms = before.da_age_ms = 1000;
        SynapseState after = before;
        plasticity_synapse(d, g, after, 0);
        assert(after.weight_ns == before.weight_ns &&
               after.spine_volume == before.spine_volume &&
               after.pre_age_ms == before.pre_age_ms &&
               after.eligible_ltp == before.eligible_ltp &&
               after.eligible_ltd == before.eligible_ltd &&
               after.structural_age_ms == before.structural_age_ms &&
               after.da_age_ms == before.da_age_ms);
    }
    g.plasticity = 1;
    float available = d.synapse_state[0].slow_resource;
    synapse_release(g, d.synapse_state[0], 1);
    assert(d.synapse_state[0].slow_resource != available);
    // An untreated animal must still learn under the identical pairing.
    d.plasticity_cohort[0].frozen = 0;
    d.synapse_state[0] = baseline;
    plasticity_synapse(d, g, d.synapse_state[0], 0);
    assert(d.synapse_state[0].weight_ns != baseline.weight_ns);
    d.plasticity_cohort[0].frozen = 1;
    // Original occupant is frozen; a child reusing this slot is not.
    assert(plasticity_frozen(d, 0));
    d.animals[0].id++;
    assert(!plasticity_frozen(d, 0));
    d.animals[0].id--;
}

int main() {
    using namespace sim;
    Device d{}; d.trace_animal = -1;
    allocate(d.animals, P); allocate(d.plasticity_cohort, P);
    allocate(d.neurons, 2); allocate(d.neuron_state, 2);
    allocate(d.neuron_owner, 2); allocate(d.synapse_state, 1);
    allocate(d.mod_event_tick, 12); allocate(d.occupied_neurons, 1);
    cudaMemset(d.animals, 0, P * sizeof(Animal));
    cudaMemset(d.plasticity_cohort, 0, P * sizeof(PlasticityCohort));
    cudaMemset(d.neurons, 0, 2 * sizeof(NeuronGene));
    cudaMemset(d.neuron_state, 0, 2 * sizeof(NeuronState));
    cudaMemset(d.neuron_owner, 0, 2 * sizeof(int));
    cudaMemset(d.mod_event_tick, 0, 12 * sizeof(unsigned long long));
    cudaMemset(d.occupied_neurons, 0, sizeof(int));
    Animal body{}; body.id = 10; body.alive = 1; body.age = 100;
    body.behavior.offspring = 3;
    PlasticityCohort record{}; record.id = 10; record.assigned = record.frozen = 1;
    record.death_step = -1; record.start_step = 500;
    cudaMemcpy(d.animals, &body, sizeof(body), cudaMemcpyHostToDevice);
    cudaMemcpy(d.plasticity_cohort, &record, sizeof(record), cudaMemcpyHostToDevice);
    check_intervention<<<1, 1>>>(d);
    check(cudaDeviceSynchronize(), "intervention checks");
    homeostasis_step<<<1, 1>>>(d, 1);
    NeuronState state{};
    cudaMemcpy(&state, d.neuron_state, sizeof(state), cudaMemcpyDeviceToHost);
    assert(state.log_ampa_scale == 0);
    record.frozen = 0;
    cudaMemcpy(d.plasticity_cohort, &record, sizeof(record), cudaMemcpyHostToDevice);
    homeostasis_step<<<1, 1>>>(d, 1);
    cudaMemcpy(&state, d.neuron_state, sizeof(state), cudaMemcpyDeviceToHost);
    assert(state.log_ampa_scale != 0);
    body.alive = 0; body.age = 105; body.behavior.offspring = 4;
    body.behavior.death_cause = 1;
    cudaMemcpy(d.animals, &body, sizeof(body), cudaMemcpyHostToDevice);
    observe_plasticity_cohort<<<(P+127)/128, 128>>>(d, 505);
    body.id = 11; body.alive = 1;
    cudaMemcpy(d.animals, &body, sizeof(body), cudaMemcpyHostToDevice);
    observe_plasticity_cohort<<<(P+127)/128, 128>>>(d, 506);
    cudaMemcpy(&record, d.plasticity_cohort, sizeof(record), cudaMemcpyDeviceToHost);
    assert(record.death_step == 505 && record.observed_step == 505);
    assert(record.final_offspring == 4 && record.death_cause == 1 && !record.alive);
    std::cout << "Plasticity freeze, intact release, homeostasis, death recording and slot reuse passed\n";
}
