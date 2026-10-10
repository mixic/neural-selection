from __future__ import annotations
import ctypes as ct
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / 'build' / 'libgym_brain.so'
NEURON_FIELDS = 10
SYNAPSE_FIELDS = 11


def build_native():
    sources = [ROOT / 'gym_adapter/native.cpp', *ROOT.glob('src/*.cuh'), ROOT/'gym_adapter/core.cuh']
    if not LIBRARY.exists() or any(p.stat().st_mtime > LIBRARY.stat().st_mtime for p in sources):
        LIBRARY.parent.mkdir(exist_ok=True)
        subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-I', str(ROOT/'src'), str(sources[0]), '-o', str(LIBRARY)], check=True)
    return LIBRARY


_library = None

def native():
    global _library
    if _library is None:
        lib = ct.CDLL(str(build_native()))
        ptr = ct.POINTER(ct.c_float)
        lib.ns_create.argtypes = [ct.c_int]*4 + [ptr, ptr, ct.POINTER(ct.c_int)]
        lib.ns_create.restype = ct.c_void_p
        lib.ns_destroy.argtypes = [ct.c_void_p]
        lib.ns_reset.argtypes = [ct.c_void_p, ct.c_uint32]
        lib.ns_step.argtypes = [ct.c_void_p, ptr, ct.c_int, ct.c_float, ptr]
        lib.ns_step.restype = ct.c_int
        lib.ns_error.restype = ct.c_char_p
        _library = lib
    return _library


@dataclass
class Genome:
    # Neurons: profile, sensory channel, active, inhibitory, modulator,
    # capacitance multiplier, leak multiplier, threshold, receptor rate, event nS.
    # Synapses: pre, post, weight nS, active, delay, branch, U, recovery,
    # facilitation, plasticity phenotype, modulator receptor bit mask.
    neurons: list[list[float]]
    synapses: list[list[float]]
    motors: list[int]
    channels: int

    def clone(self):
        return Genome([n[:] for n in self.neurons], [s[:] for s in self.synapses], self.motors[:], self.channels)

    def save(self, path):
        Path(path).write_text(json.dumps({'format':1, **asdict(self)}, indent=2)+'\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.pop('format', None) != 1:
            raise ValueError('Unsupported genome format')
        return cls(**data)


def neuron(profile=4, channel=-1, inhibitory=0):
    threshold = {0:-52, 1:-42, 2:-42, 3:-42, 4:-40, 5:-40, 6:-51.8}[profile]
    return [profile, channel, 1, inhibitory, 0, 1., 1., threshold, 43., 2.]


def synapse(rng, pre, post):
    return [pre, post, float(rng.uniform(1,18)), 1, int(rng.integers(1,5)), int(rng.integers(4)), .8, 60., 50., 0, 0]


def random_genome(channels, outputs, rng, hidden=8, connections=64):
    # Ports are fixed. No observation-dependent decision rules or reward inputs.
    ns = [neuron(channel=i) for i in range(channels)]
    ms = list(range(channels, channels+outputs))
    ns += [neuron() for _ in ms]
    ns += [neuron(profile=int(rng.choice([0,1,4])), inhibitory=int(rng.random()<.2)) for _ in range(hidden)]
    edges=[]
    for _ in range(connections):
        edges.append(synapse(rng, int(rng.integers(len(ns))), int(rng.integers(channels,len(ns)))))
    return Genome(ns, edges, ms, channels)


def mutate(parent, rng, rate=.12):
    child=parent.clone()
    fixed=child.channels+len(child.motors)
    for i,n in enumerate(child.neurons):
        if rng.random()<rate:
            field=int(rng.choice([5,6,7]))
            if field==7:n[field]=float(np.clip(n[field]+rng.normal(0,1.5),-65,-30))
            else:n[field]=float(np.clip(n[field]*math.exp(rng.normal(0,.15)),.25,4))
        if i>=fixed:
            if rng.random()<rate*.15:n[2]=1-n[2]
            if rng.random()<rate*.1:n[3]=1-n[3]
            if rng.random()<rate*.04:n[4]=int(rng.integers(4))
            if rng.random()<rate*.08:
                profile=int(rng.choice([0,1,4]))
                n[0]=profile;n[7]={0:-52,1:-42,4:-40}[profile]
    for s in child.synapses:
        if rng.random()<rate:s[2]=float(np.clip(s[2]*math.exp(rng.normal(0,.4)),.05,80))
        if rng.random()<rate*.3:s[1]=int(rng.integers(child.channels,len(child.neurons)))
        if rng.random()<rate*.3:s[0]=int(rng.integers(len(child.neurons)))
        if rng.random()<rate*.1:s[3]=1-s[3]
        if rng.random()<rate*.1:s[5]=int(rng.integers(4))
        if rng.random()<rate*.1:s[4]=int(rng.integers(1,9))
        if rng.random()<rate*.08:s[9]=int(rng.integers(6));s[10]=int(rng.integers(8))
    if rng.random()<.3:
        child.synapses.append(synapse(rng,int(rng.integers(len(child.neurons))),int(rng.integers(child.channels,len(child.neurons)))))
    if child.synapses and rng.random()<.2:child.synapses.pop(int(rng.integers(len(child.synapses))))
    if rng.random()<.12:
        index=len(child.neurons);child.neurons.append(neuron(profile=int(rng.choice([0,1,4])),inhibitory=int(rng.random()<.2)))
        child.synapses.extend([synapse(rng,int(rng.integers(index)),index),synapse(rng,index,int(rng.integers(child.channels,index)))])
    if len(child.neurons)>fixed and rng.random()<.1:
        index=int(rng.integers(fixed,len(child.neurons)));child.neurons.pop(index)
        child.synapses=[s for s in child.synapses if s[0]!=index and s[1]!=index]
        for s in child.synapses:
            s[0]-=int(s[0]>index);s[1]-=int(s[1]>index)
    return child


class SpaceAdapter:
    def __init__(self, observation_space, action_space, observation_scale=None):
        if not observation_space.is_np_flattenable:
            raise TypeError('Observation space must be flattenable (Box, Discrete, Dict, Tuple, etc.)')
        self.observation_space=observation_space
        self.action_space=action_space
        self.dim=spaces.utils.flatdim(observation_space)
        if self.dim*2>65535:raise ValueError('At most 32767 flattened observations are supported')
        box=spaces.utils.flatten_space(observation_space)
        low,high=box.low.reshape(-1).astype(float),box.high.reshape(-1).astype(float)
        bounded=np.isfinite(low)&np.isfinite(high)
        self.center=np.where(bounded,(np.where(bounded,low,0)+np.where(bounded,high,0))/2,0)
        self.scale=np.where(bounded,np.maximum((np.where(bounded,high,1)-np.where(bounded,low,-1))/2,1e-6),1.)
        if observation_scale is not None:
            scale=np.asarray(observation_scale,dtype=float)
            if scale.shape!=(self.dim,) or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('One positive finite scale is required per flattened observation')
            self.scale=scale
            self.center=np.zeros(self.dim)
        if isinstance(action_space,spaces.Discrete):self.outputs=action_space.n
        elif isinstance(action_space,spaces.Box):
            if not np.isfinite(action_space.low).all() or not np.isfinite(action_space.high).all():raise ValueError('Continuous action bounds must be finite')
            self.outputs=int(np.prod(action_space.shape))*2
        else:raise TypeError('Supported actions: Discrete and finite Box')
        self.channels=self.dim*2
        self._two_actions=isinstance(action_space,spaces.Discrete) and action_space.n==2

    def encode(self, observation):
        x=np.asarray(spaces.utils.flatten(self.observation_space,observation),dtype=float)
        if not np.isfinite(x).all():raise ValueError('Nonfinite observation')
        signed=np.tanh((x-self.center)/self.scale)
        # Opponent code: neutral observations stimulate both sensory channels
        # equally. This is explicit sensory transduction, not a somatic bias.
        return np.asarray(np.column_stack(((1+signed)/2,(1-signed)/2)).ravel(),dtype=np.float32)

    def encode_many(self, observations):
        if isinstance(self.observation_space,spaces.Box):
            # Match Gymnasium flatten's dtype conversion before double arithmetic.
            x=np.asarray(observations,dtype=self.observation_space.dtype).reshape(len(observations),self.dim).astype(float)
            if not np.isfinite(x).all():raise ValueError('Nonfinite observation')
            signed=np.tanh((x-self.center)/self.scale)
            out=np.empty((len(observations),self.channels),np.float32)
            out[:,0::2]=(1+signed)/2;out[:,1::2]=(1-signed)/2
            return out
        return np.asarray([self.encode(obs) for obs in observations],np.float32)

    def decode(self, activity, rng):
        if self._two_actions:
            if activity[0]>activity[1]:return self.action_space.start
            if activity[1]>activity[0]:return self.action_space.start+1
            # Preserve the original random tie choice and RNG sequence.
            return int(rng.choice(2))+self.action_space.start
        if isinstance(self.action_space,spaces.Discrete):
            tied=np.flatnonzero(activity==np.max(activity))
            return int(rng.choice(tied))+self.action_space.start
        pairs=activity.reshape(-1,2)
        value=(pairs[:,0]-pairs[:,1])/(pairs.sum(axis=1)+1.)
        low=self.action_space.low.reshape(-1);high=self.action_space.high.reshape(-1)
        return np.asarray(low+(value+1)/2*(high-low),dtype=self.action_space.dtype).reshape(self.action_space.shape)


class SpikingPolicy:
    def __init__(self, genome, adapter, ticks=16, input_gain=230., smoothing=.4):
        if genome.channels!=adapter.channels or len(genome.motors)!=adapter.outputs:raise ValueError('Genome ports do not match the environment')
        if not 0<smoothing<=1:raise ValueError('Smoothing must be in (0, 1]')
        self.adapter=adapter;self.ticks=ticks;self.input_gain=input_gain;self.smoothing=smoothing
        ns=np.ascontiguousarray(genome.neurons,dtype=np.float32).reshape(-1,NEURON_FIELDS)
        ss=np.ascontiguousarray(genome.synapses,dtype=np.float32).reshape(-1,SYNAPSE_FIELDS)
        if not np.isfinite(ns).all() or not np.isfinite(ss).all():raise ValueError('Nonfinite gene')
        ms=np.ascontiguousarray(genome.motors,dtype=np.int32)
        self.lib=native();self.ptr=self.lib.ns_create(len(ns),len(ss),genome.channels,len(ms),ns.ctypes.data_as(ct.POINTER(ct.c_float)),ss.ctypes.data_as(ct.POINTER(ct.c_float)),ms.ctypes.data_as(ct.POINTER(ct.c_int)))
        if not self.ptr:raise ValueError(self.lib.ns_error().decode())
        if os.environ.get('GYM_FULL_GRAPH')!='1':
            from .compile_graph import compile_genome
            genome=compile_genome(genome)
            ns=np.ascontiguousarray(genome.neurons,dtype=np.float32).reshape(-1,NEURON_FIELDS)
            ss=np.ascontiguousarray(genome.synapses,dtype=np.float32).reshape(-1,SYNAPSE_FIELDS)
            ms=np.ascontiguousarray(genome.motors,dtype=np.int32)
            compact=self.lib.ns_create(len(ns),len(ss),genome.channels,len(ms),ns.ctypes.data_as(ct.POINTER(ct.c_float)),ss.ctypes.data_as(ct.POINTER(ct.c_float)),ms.ctypes.data_as(ct.POINTER(ct.c_int)))
            if not compact:
                self.close();raise ValueError(self.lib.ns_error().decode())
            self.lib.ns_destroy(self.ptr);self.ptr=compact
        if hasattr(genome,'_original_neurons'):
            ids=np.asarray(genome._original_neurons,np.uint32);edges=np.asarray(genome._original_synapses,np.uint32)
            self.lib.ns_original_indices.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32),ct.POINTER(ct.c_uint32)]
            self.lib.ns_original_indices(self.ptr,ids.ctypes.data_as(ct.POINTER(ct.c_uint32)),edges.ctypes.data_as(ct.POINTER(ct.c_uint32)))
        self.activity=np.zeros(adapter.outputs,dtype=np.float32)
        self.last_spikes=np.zeros(adapter.outputs,dtype=np.float32)
        self._output_pointer=self.last_spikes.ctypes.data_as(ct.POINTER(ct.c_float))
        self.reset(0)

    def reset(self, seed=0):
        if not self.ptr:raise RuntimeError('Policy is closed')
        self.lib.ns_reset(self.ptr,int(seed)&0xffffffff)
        self.activity.fill(0);self.last_spikes.fill(0);self.rng=np.random.default_rng(seed)

    def act(self, observation):
        if not self.ptr:raise RuntimeError('Policy is closed')
        x=self.adapter.encode(observation)
        if self.lib.ns_step(self.ptr,x.ctypes.data_as(ct.POINTER(ct.c_float)),self.ticks,self.input_gain,self._output_pointer):raise RuntimeError(self.lib.ns_error().decode())
        self.activity*=1-self.smoothing;self.activity+=self.smoothing*self.last_spikes
        return self.adapter.decode(self.activity,self.rng)

    def close(self):
        if getattr(self,'ptr',None):self.lib.ns_destroy(self.ptr);self.ptr=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):self.close()
