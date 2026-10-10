// Gymnasium bridge: host execution of the same physiological equations as CUDA.
#include "physiology.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef NS_PROFILE
#include <chrono>
static double phase_seconds[5]{};
static unsigned long long phase_calls=0;
using Clock=std::chrono::steady_clock;
#define PHASE_START auto phase_start=Clock::now()
#define PHASE_END(i) phase_seconds[i]+=std::chrono::duration<double>(Clock::now()-phase_start).count();phase_start=Clock::now()
#else
#define PHASE_START
#define PHASE_END(i)
#endif
#include "core.cuh"
using namespace sim;
struct Brain {
    std::vector<NeuronGene> genes;
    std::vector<SynapseGene> connections;
    std::vector<NeuronState> neurons;
    std::vector<SynapseState> synapses;
    std::vector<int> motors;
    std::vector<std::array<unsigned char, 9>> history;
    std::vector<std::array<float, DENDRITES>> excitation;
    std::vector<float> inhibition;
    std::vector<unsigned char> modulation;
    std::vector<int> sensory;
    std::vector<uint32_t> original_neurons, original_synapses;
    int channels;
    BrainView view() {return {genes.data(),connections.data(),neurons.data(),synapses.data(),motors.data(),sensory.data(),reinterpret_cast<unsigned char (*)[9]>(history.data()),reinterpret_cast<float (*)[DENDRITES]>(excitation.data()),inhibition.data(),modulation.data(),int(genes.size()),int(connections.size()),int(motors.size()),channels,tick};}
    int64_t tick=0;
};
static thread_local std::string error;
extern "C" {
#ifdef NS_PROFILE
void ns_profile(double *out) {for(int i=0;i<5;++i)out[i]=phase_seconds[i];out[5]=double(phase_calls);}
void ns_profile_clear() {for(auto &x:phase_seconds)x=0;phase_calls=0;}
#endif
const char *ns_error() { return error.c_str(); }
void *ns_create(int count, int edges, int channels, int outputs,
                const float *ng, const float *sg, const int *motors) {
 try {
    if(count<1 || edges<0 || channels<1 || channels>65535 || outputs<1)
        throw std::runtime_error("Invalid brain dimensions");
    auto b=new Brain; b->channels=channels;
    try {
      for(int i=0;i<count;++i) {
        const float *p=ng+i*10;
        if(p[0]<0 || p[0]>6 || p[1]<-1 || p[1]>=channels || p[2]<0 || p[2]>1 || p[3]<0 || p[3]>1 || p[4]<0 || p[4]>3)
            throw std::runtime_error("Invalid neuron gene");
        NeuronGene g=initial_profile(static_cast<unsigned char>(p[0]));
        b->sensory.push_back(int(p[1]));
        g.sensory_channel=p[1]<0?-1:0;g.active=p[2];
        g.inhibitory_transmitter=p[3];g.modulator=p[4];
        if(p[5]<=0 || p[6]<=0 || p[7]<-80 || p[7]>-20 || p[8]<0 || p[9]<0)
            throw std::runtime_error("Invalid membrane parameters");
        g.capacitance_pf*=p[5];g.leak_ns*=p[6];g.threshold_mv=p[7];
        g.receptor_baseline_hz=0;g.receptor_gain_hz=p[8];g.receptor_event_ns=p[9];
        b->genes.push_back(g);
      }
      for(int i=0;i<edges;++i) {
        const float *p=sg+i*11;
        if(p[0]<0 || p[0]>=count || p[1]<0 || p[1]>=count || p[2]<0 || p[3]<0 || p[3]>1 || p[4]<1 || p[4]>8 || p[5]<0 || p[5]>=DENDRITES || p[6]<=0 || p[6]>1 || p[7]<=0 || p[8]<0 || p[9]<0 || p[9]>5 || p[10]<0 || p[10]>7)
            throw std::runtime_error("Invalid synapse gene");
        SynapseGene g{};g.pre=p[0];g.post=p[1];g.initial_weight_ns=p[2];
        g.active=p[3];g.delay=p[4];g.branch=p[5];g.utilization=p[6];
        g.recovery_ms=p[7];g.facilitation_ms=p[8];g.plasticity=p[9];
        g.mod_receptors=p[10];g.release_sites=14;b->connections.push_back(g);
      }
      for(int i=0;i<outputs;++i) {if(motors[i]<0 || motors[i]>=count)throw std::runtime_error("Invalid motor index");b->motors.push_back(motors[i]);}
      b->neurons.resize(count);b->synapses.resize(edges);b->history.resize(count);
      b->excitation.resize(count);b->inhibition.resize(count);b->modulation.resize(count);
    } catch(...) {delete b;throw;}
    return b;
 } catch(const std::exception &e) {error=e.what();return nullptr;}
}
void ns_original_indices(void *p,const uint32_t *neurons,const uint32_t *synapses) {
    auto &b=*static_cast<Brain *>(p);
    b.original_neurons.assign(neurons,neurons+b.genes.size());
    if(!b.connections.empty())b.original_synapses.assign(synapses,synapses+b.connections.size());
}
void ns_destroy(void *p) {delete static_cast<Brain *>(p);}
void ns_reset(void *p, uint32_t seed) {
    auto &b=*static_cast<Brain *>(p);b.tick=0;
    auto view=b.view();brain_reset(view,seed);b.tick=view.tick;
    if(!b.original_neurons.empty())for(size_t i=0;i<b.genes.size();++i)b.neurons[i]=initial_neuron_state(b.genes[i],seed+b.original_neurons[i]*7919u);
    if(!b.original_synapses.empty())for(size_t i=0;i<b.connections.size();++i)b.synapses[i]=initial_synapse_state(b.connections[i],seed+b.original_synapses[i]*104729u);
}
int ns_step(void *p, const float *input, int ticks, float input_gain, float *output) {
 try {
    auto &b=*static_cast<Brain *>(p);
    if(ticks<1 || ticks>10000 || !std::isfinite(input_gain) || input_gain<0)throw std::runtime_error("Invalid neural integration interval");
    for(int c=0;c<b.channels;++c)if(!std::isfinite(input[c]) || input[c]<0 || input[c]>1)throw std::runtime_error("Invalid encoded observation");
    auto view=b.view();
    if(!brain_step(view,input,ticks,input_gain,output))throw std::runtime_error("Nonfinite neural state");
    b.tick=view.tick;
#ifdef NS_PROFILE
    ++phase_calls;
#endif
    return 0;
 } catch(const std::exception &e) {error=e.what();return -1;}
}
}
// Exact symbolic partial view; environment dynamics remain in Gymnasium.
extern "C" void ns_grid_observation(const unsigned char *world,const unsigned char *transparent,
 int width,int height,int topx,int topy,int size,int direction,int through,int reserved,
 const unsigned char *carrying,unsigned char *out) {
 std::vector<unsigned char> cells(size*size*3),clear(size*size),mask(size*size,through?1:0);
 for(int x=0;x<size;++x)for(int y=0;y<size;++y){
  int sx=x,sy=y;for(int k=0;k<direction+1;++k){int old=sx;sx=size-1-sy;sy=old;}
  sx+=topx;sy+=topy;int i=x*size+y;
  if(sx>=0&&sx<width&&sy>=0&&sy<height){int j=sy*width+sx;for(int c=0;c<3;++c)cells[i*3+c]=world[j*3+c];clear[i]=transparent[j];}
  else{cells[i*3]=2;cells[i*3+1]=5;cells[i*3+2]=0;clear[i]=0;}
 }
 if(!through){
  mask[(size/2)*size+size-1]=1;
  for(int y=size-1;y>=0;--y){
   for(int x=0;x<size-1;++x)if(mask[x*size+y]&&clear[x*size+y]){mask[(x+1)*size+y]=1;if(y>0){mask[(x+1)*size+y-1]=1;mask[x*size+y-1]=1;}}
   for(int x=size-1;x>=1;--x)if(mask[x*size+y]&&clear[x*size+y]){mask[(x-1)*size+y]=1;if(y>0){mask[(x-1)*size+y-1]=1;mask[x*size+y-1]=1;}}
  }
 }
 int agent=(size/2)*size+size-1;
 for(int c=0;c<3;++c)cells[agent*3+c]=carrying[c];
 for(int i=0;i<size*size;++i)for(int c=0;c<3;++c)out[i*3+c]=mask[i]?cells[i*3+c]:0;
}
