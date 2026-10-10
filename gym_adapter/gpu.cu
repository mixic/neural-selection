// CUDA batch backend. Scalar lanes preserve each brain's edge accumulation order.
#include <cuda_runtime.h>
#include <cstdlib>
#include "native.cpp"
static void check(cudaError_t code) {if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct GpuView {BrainView brain; float *released; int *incoming_start,*incoming_edges; uint32_t *original_neurons,*original_synapses;};
struct GpuBatch {
    GpuView *brains=nullptr;
    float *input=nullptr,*output=nullptr;
    unsigned char *mask=nullptr;
    uint32_t *seeds=nullptr;
    int *status=nullptr;
    int count,channels,outputs,threads=32;
    std::vector<void *> allocations;
    ~GpuBatch(){for(auto p:allocations)cudaFree(p);}
    template<class T>T *allocate(size_t count) {T *p=nullptr;if(count){check(cudaMalloc((void**)&p,sizeof(T)*count));allocations.push_back(p);}return p;}
};
__global__ void reset_batch(GpuView *views,int count,const unsigned char *mask,const uint32_t *seeds) {
    int i=blockIdx.x;if(i>=count || !mask[i])return;auto &v=views[i];auto &b=v.brain;
    for(int j=threadIdx.x;j<b.count;j+=blockDim.x){b.neurons[j]=initial_neuron_state(b.genes[j],seeds[i]+v.original_neurons[j]*7919u);for(int k=0;k<9;++k)b.history[j][k]=0;}
    for(int j=threadIdx.x;j<b.edges;j+=blockDim.x)b.synapses[j]=initial_synapse_state(b.connections[j],seeds[i]+v.original_synapses[j]*104729u);
    if(threadIdx.x==0)b.tick=0;
}
__global__ void step_batch(GpuView *views,int count,const unsigned char *mask,const float *inputs,float *outputs,int ticks,float gain,int *status) {
    int i=blockIdx.x;if(i>=count)return;
    auto &v=views[i];BrainView b=v.brain;float *output=outputs+i*b.outputs;const float *input=inputs+i*b.channels;
    if(threadIdx.x==0)status[i]=0;
    for(int j=threadIdx.x;j<b.outputs;j+=blockDim.x)output[j]=0;
    if(!mask[i])return;
    for(int t=0;t<ticks;++t,++b.tick) {
      for(int j=threadIdx.x;j<b.edges;j+=blockDim.x) {
        const auto &g=b.connections[j];auto &s=b.synapses[j];s.arrived=0;v.released[j]=0;
        if(!g.active || !b.genes[g.pre].active || !b.genes[g.post].active)continue;
        s.arrived=b.tick>=g.delay && b.history[g.pre][(b.tick-g.delay)%9];
        if(s.arrived){const auto &source=b.genes[g.pre];v.released[j]=!source.inhibitory_transmitter || source.modulator?synapse_release(g,s,b.tick):s.weight_ns;}
      }
      __syncthreads();
      for(int j=threadIdx.x;j<b.count;j+=blockDim.x) {
        float excitation[DENDRITES]{};float inhibition=0;unsigned char modulation=0;
        // Ascending original edge order preserves floating-point accumulation.
        for(int k=v.incoming_start[j];k<v.incoming_start[j+1];++k) {
          int edge=v.incoming_edges[k];const auto &g=b.connections[edge];float released=v.released[edge];
          const auto &source=b.genes[g.pre];
          if(source.modulator){if(released>0)modulation|=1u<<(source.modulator-1);}
          else if(source.inhibitory_transmitter)inhibition+=released;
          else if(b.synapses[edge].arrived)excitation[g.branch]+=released*expf(b.neurons[j].log_ampa_scale);
        }
        b.modulation[j]=modulation;const auto &g=b.genes[j];auto &s=b.neurons[j];
        if(g.active){float current=0;if(b.sensory[j]>=0)current=g.type==5?receptor_current(g,s,input[b.sensory[j]]):gain*input[b.sensory[j]];neuron_advance(g,s,current,excitation,inhibition);if(!isfinite(s.voltage_mv))atomicExch(status+i,-1);}else s.spike=0;
      }
      __syncthreads();
      for(int j=threadIdx.x;j<b.count;j+=blockDim.x)b.history[j][b.tick%9]=b.neurons[j].spike;
      for(int j=threadIdx.x;j<b.edges;j+=blockDim.x) {
        const auto &g=b.connections[j];auto &s=b.synapses[j];
        if(!g.active || !b.genes[g.pre].active || !b.genes[g.post].active)continue;
        const auto &a=b.genes[g.pre],&z=b.genes[g.post];
        bool pair=!a.inhibitory_transmitter && excitatory_type(a.type) && !a.modulator && !z.inhibitory_transmitter && excitatory_type(z.type);
        synapse_plasticity_step(g,s,s.arrived,b.neurons[g.post].spike,pair,b.neurons[g.post].recent_depol_mv,true,b.modulation[g.post]);
      }
      for(int j=threadIdx.x;j<b.outputs;j+=blockDim.x)output[j]+=b.neurons[b.motors[j]].spike;
      __syncthreads();
    }
    for(int j=threadIdx.x;j<b.count;j+=blockDim.x)if(b.genes[j].active)homeostatic_advance(b.genes[j],b.neurons[j],float(ticks),b.neurons[j].total_spikes*1000.0f/float(ticks>b.tick?ticks:b.tick));
    if(threadIdx.x==0)v.brain.tick=b.tick;
}
extern "C" {
void *ns_gpu_create(void **handles,int count) {
 try {
    if(count<1)throw std::runtime_error("Empty GPU batch");
    auto batch=new GpuBatch;
    if(const char *threads=std::getenv("GYM_CUDA_THREADS"))batch->threads=std::atoi(threads);
    if(batch->threads!=32&&batch->threads!=64&&batch->threads!=128&&batch->threads!=256){delete batch;throw std::runtime_error("Invalid CUDA block size");}
    try {
      auto &first=*static_cast<Brain *>(handles[0]);batch->count=count;batch->channels=first.channels;batch->outputs=first.motors.size();
      size_t neurons=0,edges=0;
      for(int i=0;i<count;++i){auto &b=*static_cast<Brain *>(handles[i]);if(b.channels!=batch->channels || int(b.motors.size())!=batch->outputs)throw std::runtime_error("Batch ports differ");neurons+=b.genes.size();edges+=b.connections.size();}
      auto original_neurons=batch->allocate<uint32_t>(neurons);auto original_synapses=batch->allocate<uint32_t>(edges);
      auto genes=batch->allocate<NeuronGene>(neurons);auto connections=batch->allocate<SynapseGene>(edges);
      auto states=batch->allocate<NeuronState>(neurons);auto synapses=batch->allocate<SynapseState>(edges);
      auto motors=batch->allocate<int>(count*batch->outputs);auto sensory=batch->allocate<int>(neurons);
      auto history=batch->allocate<unsigned char>(neurons*9);auto excitation=batch->allocate<float>(neurons*DENDRITES);
      auto inhibition=batch->allocate<float>(neurons);auto modulation=batch->allocate<unsigned char>(neurons);
      auto released=batch->allocate<float>(edges);auto starts=batch->allocate<int>(neurons+count);auto incoming=batch->allocate<int>(edges);
      std::vector<GpuView> views;size_t n=0,e=0;
      for(int i=0;i<count;++i) {
        auto &b=*static_cast<Brain *>(handles[i]);int nc=b.genes.size(),ec=b.connections.size();
        std::vector<uint32_t> ni(nc),ei(ec);
        for(int j=0;j<nc;++j)ni[j]=b.original_neurons.empty()?j:b.original_neurons[j];
        for(int j=0;j<ec;++j)ei[j]=b.original_synapses.empty()?j:b.original_synapses[j];
        check(cudaMemcpy(original_neurons+n,ni.data(),nc*sizeof(uint32_t),cudaMemcpyHostToDevice));
        if(ec)check(cudaMemcpy(original_synapses+e,ei.data(),ec*sizeof(uint32_t),cudaMemcpyHostToDevice));
        check(cudaMemcpy(genes+n,b.genes.data(),nc*sizeof(NeuronGene),cudaMemcpyHostToDevice));
        if(ec)check(cudaMemcpy(connections+e,b.connections.data(),ec*sizeof(SynapseGene),cudaMemcpyHostToDevice));
        check(cudaMemcpy(motors+i*batch->outputs,b.motors.data(),batch->outputs*sizeof(int),cudaMemcpyHostToDevice));
        check(cudaMemcpy(sensory+n,b.sensory.data(),nc*sizeof(int),cudaMemcpyHostToDevice));
        std::vector<int> offsets(nc+1),edge_indices;
        for(int post=0;post<nc;++post){offsets[post]=edge_indices.size();for(int edge=0;edge<ec;++edge)if(b.connections[edge].post==post)edge_indices.push_back(edge);}
        offsets[nc]=edge_indices.size();
        check(cudaMemcpy(starts+n+i,offsets.data(),offsets.size()*sizeof(int),cudaMemcpyHostToDevice));
        if(ec)check(cudaMemcpy(incoming+e,edge_indices.data(),ec*sizeof(int),cudaMemcpyHostToDevice));
        views.push_back({{genes+n,ec?connections+e:nullptr,states+n,ec?synapses+e:nullptr,motors+i*batch->outputs,sensory+n,reinterpret_cast<unsigned char (*)[9]>(history+n*9),reinterpret_cast<float (*)[DENDRITES]>(excitation+n*DENDRITES),inhibition+n,modulation+n,nc,ec,batch->outputs,batch->channels,0},ec?released+e:nullptr,starts+n+i,ec?incoming+e:nullptr,original_neurons+n,ec?original_synapses+e:nullptr});n+=nc;e+=ec;
      }
      batch->brains=batch->allocate<GpuView>(count);check(cudaMemcpy(batch->brains,views.data(),sizeof(GpuView)*count,cudaMemcpyHostToDevice));
      batch->input=batch->allocate<float>(count*batch->channels);batch->output=batch->allocate<float>(count*batch->outputs);
      batch->mask=batch->allocate<unsigned char>(count);batch->seeds=batch->allocate<uint32_t>(count);batch->status=batch->allocate<int>(count);
    } catch(...) {delete batch;throw;}
    return batch;
 } catch(const std::exception &e){error=e.what();return nullptr;}
}
void ns_gpu_destroy(void *p){delete static_cast<GpuBatch *>(p);}
int ns_gpu_reset(void *p,const unsigned char *mask,const uint32_t *seeds) {
 try {auto &b=*static_cast<GpuBatch *>(p);check(cudaMemcpy(b.mask,mask,b.count,cudaMemcpyHostToDevice));check(cudaMemcpy(b.seeds,seeds,b.count*sizeof(uint32_t),cudaMemcpyHostToDevice));reset_batch<<<b.count,b.threads>>>(b.brains,b.count,b.mask,b.seeds);check(cudaGetLastError());check(cudaDeviceSynchronize());return 0;}catch(const std::exception &e){error=e.what();return -1;}
}
int ns_gpu_step(void *p,const float *input,const unsigned char *mask,int ticks,float gain,float *output,int *status) {
 try {
    auto &b=*static_cast<GpuBatch *>(p);
    if(ticks<1 || ticks>10000 || !std::isfinite(gain) || gain<0)throw std::runtime_error("Invalid neural integration interval");
    for(int i=0;i<b.count;++i)if(mask[i])for(int j=0;j<b.channels;++j){float v=input[i*b.channels+j];if(!std::isfinite(v)||v<0||v>1)throw std::runtime_error("Invalid encoded observation");}
    check(cudaMemcpy(b.input,input,b.count*b.channels*sizeof(float),cudaMemcpyHostToDevice));check(cudaMemcpy(b.mask,mask,b.count,cudaMemcpyHostToDevice));
    step_batch<<<b.count,b.threads>>>(b.brains,b.count,b.mask,b.input,b.output,ticks,gain,b.status);
    check(cudaGetLastError());check(cudaMemcpy(output,b.output,b.count*b.outputs*sizeof(float),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,b.status,b.count*sizeof(int),cudaMemcpyDeviceToHost));return 0;
 }catch(const std::exception &e){error=e.what();return -1;}
}
}
