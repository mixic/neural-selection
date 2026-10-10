"""Experimental exact CUDA ObstructedMaze batch, with CPU reference resets.

Only the maze transitions/views move to CUDA. Neural kernels are included verbatim
from gpu.cu. NumPy action tie-breaking and its RNG stream remain on the host.
"""
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .adapter import ROOT, SpaceAdapter
from .environments import make_env

WIDTH=HEIGHT=16
VIEW=7
OBJECTS=512
AGENT_FIELDS=9
SUPPORTED='MiniGrid-ObstructedMaze-Full-v1'


def build_maze_gpu():
    target=ROOT/'build/libgym_maze_gpu.so'
    sources=[ROOT/'gym_adapter/maze_gpu.cu',ROOT/'gym_adapter/gpu.cu',ROOT/'gym_adapter/native.cpp',ROOT/'gym_adapter/core.cuh',*ROOT.glob('src/*.cuh')]
    if target.exists() and all(p.stat().st_mtime<=target.stat().st_mtime for p in sources):return target
    compiler=os.environ.get('NVCC') or shutil.which('nvcc');local=ROOT/'build/cuda-toolchain/extracted'
    if not compiler and (local/'usr/lib/nvidia-cuda-toolkit/bin/nvcc').exists():compiler=str(local/'usr/lib/nvidia-cuda-toolkit/bin/nvcc')
    if not compiler:raise RuntimeError('CUDA backend requires nvcc')
    command=[compiler,'-O3','-std=c++17','--fmad=false','--prec-div=true','--prec-sqrt=true','-arch='+os.environ.get('GYM_CUDA_ARCH','native'),'-shared','-Xcompiler','-fPIC','-I',str(ROOT/'src'),str(sources[0]),'-o',str(target),'--cudart','shared','--cudadevrt','none','-L',str(ROOT/'build')]
    env=os.environ.copy()
    if str(local) in compiler:
        import nvidia.cuda_runtime,nvidia.cuda_nvcc
        runtime=Path(nvidia.cuda_runtime.__file__).parent;nvcc=Path(nvidia.cuda_nvcc.__file__).parent
        command+=['-ccbin',str(local/'usr/bin/g++-12'),'--libdevice-directory',str(nvcc/'nvvm/libdevice'),'-I',str(runtime/'include'),'-I',str(nvcc/'include'),'-L/usr/lib/x86_64-linux-gnu']
        env['PATH']=str(local/'usr/bin')+':'+str(local/'usr/lib/nvidia-cuda-toolkit/bin')+':'+env['PATH']
    subprocess.run(command,env=env,check=True)
    return target


def export_world(env):
    """Serialize object identities, including objects nested in boxes."""
    base=env.unwrapped
    if (base.width,base.height,base.agent_view_size)!=(WIDTH,HEIGHT,VIEW):raise ValueError('Unsupported maze dimensions')
    grid=np.full(WIDTH*HEIGHT,-1,np.int32);objects=np.zeros((OBJECTS,4),np.int32);objects[:,3]=-1
    refs=[];ids={}
    def identify(obj):
        if obj is None:return -1
        if obj in ids:return ids[obj]
        idx=len(refs)
        if idx>=OBJECTS:raise ValueError('Too many objects')
        ids[obj]=idx;refs.append(obj)
        objects[idx,:3]=obj.encode()
        objects[idx,3]=identify(getattr(obj,'contains',None))
        return idx
    for i,obj in enumerate(base.grid.grid):grid[i]=identify(obj)
    carrying=identify(base.carrying);target=identify(base.obj)
    agents=np.asarray([*base.agent_pos,base.agent_dir,carrying,target,base.step_count,base.max_steps,int(base.see_through_walls),0],np.int32)
    return grid,objects,agents,refs


def reset_world(job):
    config,seed=job;env=make_env(config)
    try:env.reset(seed=int(seed));return export_world(env)[:3]
    finally:env.close()


class GpuMazeBatch:
    def __init__(self,count,adapter):
        self.count=count;self.ptr=None;self.adapter=adapter
        if adapter.observation_space.shape!=(VIEW,VIEW,3) or adapter.channels!=VIEW*VIEW*6 or adapter.outputs!=7:raise ValueError('Unsupported ports')
        # Lookup tables preserve NumPy's exact float32 transduction, avoiding
        # differences between CPU and CUDA transcendental implementations.
        images=[np.full((VIEW,VIEW,3),value,np.uint8) for value in range(11)]
        lookup=np.ascontiguousarray(adapter.encode_many(images),np.float32)
        lib=ct.CDLL(str(build_maze_gpu()));p=ct.c_void_p;integer=ct.POINTER(ct.c_int);byte=ct.POINTER(ct.c_ubyte);flt=ct.POINTER(ct.c_float);dbl=ct.POINTER(ct.c_double)
        lib.ns_maze_create.argtypes=[ct.c_int,flt];lib.ns_maze_create.restype=p;lib.ns_maze_destroy.argtypes=[p]
        lib.ns_maze_reset.argtypes=[p,integer,integer,integer];lib.ns_maze_reset.restype=ct.c_int
        lib.ns_maze_step.argtypes=[p,integer];lib.ns_maze_step.restype=ct.c_int
        lib.ns_maze_read.argtypes=[p,byte,dbl,byte,integer,integer,integer];lib.ns_maze_read.restype=ct.c_int
        lib.ns_maze_brain_step.argtypes=[p,p,byte,ct.c_int,ct.c_float,flt,integer];lib.ns_maze_brain_step.restype=ct.c_int;lib.ns_error.restype=ct.c_char_p
        self.lib=lib;self.ptr=lib.ns_maze_create(count,lookup.ctypes.data_as(flt))
        if not self.ptr:raise RuntimeError(lib.ns_error().decode())
        self.rewards=np.zeros(count,np.float64);self.dones=np.zeros(count,np.uint8)
    def check(self,rc):
        if rc:raise RuntimeError(self.lib.ns_error().decode())
    def reset(self,worlds):
        if len(worlds)!=self.count:raise ValueError('World batch differs')
        arrays=[np.ascontiguousarray([w[k] for w in worlds],np.int32) for k in range(3)]
        expected=[(self.count,WIDTH*HEIGHT),(self.count,OBJECTS,4),(self.count,AGENT_FIELDS)]
        if any(a.shape!=shape for a,shape in zip(arrays,expected)):raise ValueError('Invalid serialized world dimensions')
        self.check(self.lib.ns_maze_reset(self.ptr,*[x.ctypes.data_as(ct.POINTER(ct.c_int)) for x in arrays]))
        self.rewards.fill(0);self.dones.fill(0)
    def step(self,actions,read=True):
        array=np.ascontiguousarray([-1 if a is None else a for a in actions],np.int32)
        if array.shape!=(self.count,) or (array < -1).any() or (array>6).any():raise ValueError('Invalid actions')
        self.check(self.lib.ns_maze_step(self.ptr,array.ctypes.data_as(ct.POINTER(ct.c_int))))
        if read:self.read_results()
    def read_results(self):
        self.check(self.lib.ns_maze_read(self.ptr,None,self.rewards.ctypes.data_as(ct.POINTER(ct.c_double)),self.dones.ctypes.data_as(ct.POINTER(ct.c_ubyte)),None,None,None))
        return self.rewards,self.dones
    def snapshot(self):
        obs=np.empty((self.count,VIEW,VIEW,3),np.uint8);grid=np.empty((self.count,WIDTH*HEIGHT),np.int32);objects=np.empty((self.count,OBJECTS,4),np.int32);agents=np.empty((self.count,AGENT_FIELDS),np.int32)
        self.check(self.lib.ns_maze_read(self.ptr,obs.ctypes.data_as(ct.POINTER(ct.c_ubyte)),None,None,*[x.ctypes.data_as(ct.POINTER(ct.c_int)) for x in [grid,objects,agents]]))
        return obs,grid,objects,agents
    def act(self,policy,mask):
        mask=np.ascontiguousarray(mask,np.uint8)
        self.check(self.lib.ns_maze_brain_step(self.ptr,policy.ptr,mask.ctypes.data_as(ct.POINTER(ct.c_ubyte)),policy.ticks,policy.input_gain,policy.last_spikes.ctypes.data_as(ct.POINTER(ct.c_float)),policy.status.ctypes.data_as(ct.POINTER(ct.c_int))))
        if policy.status.any():raise RuntimeError('Nonfinite neural state')
        selected=np.flatnonzero(mask);activity=policy.activity[selected].copy();activity*=1-policy.smoothing;activity+=policy.smoothing*policy.last_spikes[selected];policy.activity[selected]=activity
        actions=[None]*self.count
        if len(selected):
            maxima=activity.max(axis=1);winners=activity.argmax(axis=1);ties=(activity==maxima[:,None]).sum(axis=1)
            for row,i in enumerate(selected):actions[i]=int(winners[row])+policy.adapter.action_space.start if ties[row]==1 else policy.adapter.decode(activity[row],policy.rng[i])
        return actions
    def close(self):
        if self.ptr:self.lib.ns_maze_destroy(self.ptr);self.ptr=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        if getattr(self,'ptr',None):self.close()
