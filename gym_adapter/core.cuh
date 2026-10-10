#pragma once
#include "physiology.cuh"
// A scalar lane owns one complete brain. Traversal and accumulation order match
// the CPU backend, including recurrent/parallel edges and stochastic release.
NS_HD inline bool finite_voltage(float v) {
#ifdef __CUDA_ARCH__
    return isfinite(v);
#else
    return std::isfinite(v);
#endif
}
struct BrainView {
    sim::NeuronGene *genes;
    sim::SynapseGene *connections;
    sim::NeuronState *neurons;
    sim::SynapseState *synapses;
    int *motors, *sensory;
    unsigned char (*history)[9];
    float (*excitation)[sim::DENDRITES];
    float *inhibition;
    unsigned char *modulation;
    int count,edges,outputs,channels;
    int64_t tick;
};
NS_HD inline void brain_reset(BrainView &b,uint32_t seed) {
    b.tick=0;
    for(int i=0;i<b.count;++i){b.neurons[i]=sim::initial_neuron_state(b.genes[i],seed+uint32_t(i)*7919u);for(int k=0;k<9;++k)b.history[i][k]=0;}
    for(int i=0;i<b.edges;++i)b.synapses[i]=sim::initial_synapse_state(b.connections[i],seed+uint32_t(i)*104729u);
}
NS_HD inline bool brain_step(BrainView &b,const float *input,int ticks,float input_gain,float *output) {
    using namespace sim;
    for(int j=0;j<b.outputs;++j)output[j]=0;
    for(int t=0;t<ticks;++t,++b.tick) {
      PHASE_START;
      for(int i=0;i<b.count;++i){for(int k=0;k<DENDRITES;++k)b.excitation[i][k]=0;b.inhibition[i]=0;b.modulation[i]=0;}
      PHASE_END(0);
      for(int j=0;j<b.edges;++j) {
        const auto &g=b.connections[j];auto &s=b.synapses[j];s.arrived=0;
        if(!g.active || !b.genes[g.pre].active || !b.genes[g.post].active)continue;
        s.arrived=b.tick>=g.delay && b.history[g.pre][(b.tick-g.delay)%9];
        if(!s.arrived)continue;
        const auto &source=b.genes[g.pre];
        float released=!source.inhibitory_transmitter || source.modulator?synapse_release(g,s,b.tick):s.weight_ns;
        if(source.modulator){if(released>0)b.modulation[g.post]|=1u<<(source.modulator-1);}
        else if(source.inhibitory_transmitter)b.inhibition[g.post]+=released;
        else b.excitation[g.post][g.branch]+=released*expf(b.neurons[g.post].log_ampa_scale);
      }
      PHASE_END(1);
      for(int j=0;j<b.count;++j) {
        const auto &g=b.genes[j];auto &s=b.neurons[j];
        if(g.active) {
          float current=0; // Only observation transduction supplies applied current.
          if(b.sensory[j]>=0)current=g.type==5?receptor_current(g,s,input[b.sensory[j]]):input_gain*input[b.sensory[j]];
          neuron_advance(g,s,current,b.excitation[j],b.inhibition[j]);
          if(!finite_voltage(s.voltage_mv))return false;
        } else s.spike=0;
        b.history[j][b.tick%9]=s.spike;
      }
      PHASE_END(2);
      for(int j=0;j<b.edges;++j) {
        const auto &g=b.connections[j];auto &s=b.synapses[j];
        if(!g.active || !b.genes[g.pre].active || !b.genes[g.post].active)continue;
        const auto &a=b.genes[g.pre],&z=b.genes[g.post];
        bool pair=!a.inhibitory_transmitter && excitatory_type(a.type) && !a.modulator && !z.inhibitory_transmitter && excitatory_type(z.type);
        synapse_plasticity_step(g,s,s.arrived,b.neurons[g.post].spike,pair,b.neurons[g.post].recent_depol_mv,true,b.modulation[g.post]);
      }
      PHASE_END(3);
      for(int j=0;j<b.outputs;++j)output[j]+=b.neurons[b.motors[j]].spike;
    }
    PHASE_START;
    for(int j=0;j<b.count;++j)if(b.genes[j].active)
      homeostatic_advance(b.genes[j],b.neurons[j],float(ticks),b.neurons[j].total_spikes*1000.0f/float(ticks>b.tick?ticks:b.tick));
    PHASE_END(4);
    return true;
}
