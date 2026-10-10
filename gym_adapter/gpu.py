"""Batched CUDA inference; Gymnasium environments still run on the host."""
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .adapter import ROOT, SpaceAdapter, SpikingPolicy


def build_gpu():
    target=ROOT/'build/libgym_brain_gpu.so'
    sources=[ROOT/'gym_adapter/gpu.cu',ROOT/'gym_adapter/native.cpp',ROOT/'gym_adapter/core.cuh',*ROOT.glob('src/*.cuh')]
    if target.exists() and all(p.stat().st_mtime<=target.stat().st_mtime for p in sources):return target
    compiler=os.environ.get('NVCC') or shutil.which('nvcc')
    local=ROOT/'build/cuda-toolchain/extracted'
    if not compiler and (local/'usr/lib/nvidia-cuda-toolkit/bin/nvcc').exists():compiler=str(local/'usr/lib/nvidia-cuda-toolkit/bin/nvcc')
    if not compiler:raise RuntimeError('CUDA backend requires nvcc; set NVCC or install the CUDA toolkit')
    command=[compiler,'-O3','-std=c++17','--fmad=false','--prec-div=true','--prec-sqrt=true','-arch='+os.environ.get('GYM_CUDA_ARCH','native'),'-shared','-Xcompiler','-fPIC','-I',str(ROOT/'src'),str(sources[0]),'-o',str(target),'--cudart','shared','--cudadevrt','none','-L',str(ROOT/'build')]
    env=os.environ.copy()
    if str(local) in compiler:
        import nvidia.cuda_runtime,nvidia.cuda_nvcc
        runtime=Path(nvidia.cuda_runtime.__file__).parent;nvcc=Path(nvidia.cuda_nvcc.__file__).parent
        command+=['-ccbin',str(local/'usr/bin/g++-12'),'--libdevice-directory',str(nvcc/'nvvm/libdevice'),'-I',str(runtime/'include'),'-I',str(nvcc/'include'),'-L/usr/lib/x86_64-linux-gnu']
        env['PATH']=str(local/'usr/bin')+':'+str(local/'usr/lib/nvidia-cuda-toolkit/bin')+':'+env['PATH']
    subprocess.run(command,env=env,check=True)
    return target


class GpuPolicyBatch:
    def __init__(self,genomes,adapter,ticks=16,input_gain=230.,smoothing=.4):
        if not genomes:raise ValueError('Empty batch')
        if not 1<=ticks<=10000 or not 0<smoothing<=1:raise ValueError('Invalid ticks or smoothing')
        self.adapter=adapter;self.ticks=ticks;self.input_gain=input_gain;self.smoothing=smoothing;self.count=len(genomes);self.ptr=None
        lib=ct.CDLL(str(build_gpu()));f=ct.POINTER(ct.c_float);u=ct.POINTER(ct.c_ubyte);i=ct.POINTER(ct.c_int)
        lib.ns_gpu_create.argtypes=[ct.POINTER(ct.c_void_p),ct.c_int];lib.ns_gpu_create.restype=ct.c_void_p
        lib.ns_gpu_destroy.argtypes=[ct.c_void_p];lib.ns_gpu_reset.argtypes=[ct.c_void_p,u,ct.POINTER(ct.c_uint32)];lib.ns_gpu_reset.restype=ct.c_int
        lib.ns_gpu_step.argtypes=[ct.c_void_p,f,u,ct.c_int,ct.c_float,f,i];lib.ns_gpu_step.restype=ct.c_int;lib.ns_error.restype=ct.c_char_p
        self.lib=lib;policies=[]
        try:
            for g in genomes:policies.append(SpikingPolicy(g,adapter,ticks,input_gain,smoothing))
            handles=(ct.c_void_p*self.count)(*[p.ptr for p in policies]);self.ptr=lib.ns_gpu_create(handles,self.count)
            if not self.ptr:raise RuntimeError(lib.ns_error().decode())
        finally:
            for p in policies:p.close()
        self.input=np.zeros((self.count,adapter.channels),np.float32);self.last_spikes=np.zeros((self.count,adapter.outputs),np.float32);self.activity=np.zeros_like(self.last_spikes)
        self.status=np.zeros(self.count,np.int32);self.rng=[None]*self.count
        self.reset(np.arange(self.count,dtype=np.uint32))

    def reset(self,seeds,mask=None):
        if not self.ptr:raise RuntimeError('Policy is closed')
        seeds=np.ascontiguousarray(seeds,dtype=np.uint32);mask=np.ones(self.count,np.uint8) if mask is None else np.ascontiguousarray(mask,dtype=np.uint8)
        if seeds.shape!=(self.count,) or mask.shape!=(self.count,):raise ValueError('Reset dimensions differ')
        if self.lib.ns_gpu_reset(self.ptr,mask.ctypes.data_as(ct.POINTER(ct.c_ubyte)),seeds.ctypes.data_as(ct.POINTER(ct.c_uint32))):raise RuntimeError(self.lib.ns_error().decode())
        for i in np.flatnonzero(mask):self.activity[i].fill(0);self.last_spikes[i].fill(0);self.rng[i]=np.random.default_rng(int(seeds[i]))

    def step_encoded(self,encoded,mask=None):
        if not self.ptr:raise RuntimeError('Policy is closed')
        encoded=np.ascontiguousarray(encoded,dtype=np.float32);mask=np.ones(self.count,np.uint8) if mask is None else np.ascontiguousarray(mask,dtype=np.uint8)
        if encoded.shape!=self.input.shape or mask.shape!=(self.count,):raise ValueError('Batch dimensions differ')
        rc=self.lib.ns_gpu_step(self.ptr,encoded.ctypes.data_as(ct.POINTER(ct.c_float)),mask.ctypes.data_as(ct.POINTER(ct.c_ubyte)),self.ticks,self.input_gain,self.last_spikes.ctypes.data_as(ct.POINTER(ct.c_float)),self.status.ctypes.data_as(ct.POINTER(ct.c_int)))
        if rc:raise RuntimeError(self.lib.ns_error().decode())
        if self.status.any():raise RuntimeError('Nonfinite neural state in CUDA batch')
        return self.last_spikes

    def act(self,observations,mask=None):
        mask=np.ones(self.count,np.uint8) if mask is None else np.ascontiguousarray(mask,dtype=np.uint8)
        selected=np.flatnonzero(mask)
        if len(selected):self.input[selected]=self.adapter.encode_many([observations[i] for i in selected])
        self.step_encoded(self.input,mask);actions=[None]*self.count
        activity=self.activity[selected].copy()
        activity*=1-self.smoothing;activity+=self.smoothing*self.last_spikes[selected]
        self.activity[selected]=activity
        from gymnasium import spaces
        if isinstance(self.adapter.action_space,spaces.Discrete) and len(selected):
            maxima=activity.max(axis=1);winners=activity.argmax(axis=1)
            ties=(activity==maxima[:,None]).sum(axis=1)
            for row,i in enumerate(selected):
                if ties[row]==1:actions[i]=int(winners[row])+self.adapter.action_space.start
                else:actions[i]=self.adapter.decode(activity[row],self.rng[i])
        else:
            for row,i in enumerate(selected):actions[i]=self.adapter.decode(activity[row],self.rng[i])
        return actions

    def close(self):
        if getattr(self,'ptr',None):self.lib.ns_gpu_destroy(self.ptr);self.ptr=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):self.close()
