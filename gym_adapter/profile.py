"""Reproducible phase timings; the instrumented library is never used for training."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import time
import gymnasium as gym
from .environments import make_env
import numpy as np
from . import adapter as a


def load_instrumented():
    target=a.ROOT/'build/libgym_brain_profile.so'
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-DNS_PROFILE','-I',str(a.ROOT/'src'),str(a.ROOT/'gym_adapter/native.cpp'),'-o',str(target)],check=True)
    original=a.native();lib=ct.CDLL(str(target))
    for name in ('ns_create','ns_destroy','ns_reset','ns_step','ns_error'):
        fn=getattr(lib,name);reference=getattr(original,name)
        fn.argtypes=reference.argtypes;fn.restype=reference.restype
    lib.ns_profile.argtypes=[ct.POINTER(ct.c_double)]
    return lib


def measure(genome, config, episodes, instrumented=False):
    env=make_env(config);space=a.SpaceAdapter(env.observation_space,env.action_space,config['observation_scale'])
    phases={k:0. for k in ('encode','native','decode','environment','other')};returns=[];steps=0
    start=time.perf_counter()
    with a.SpikingPolicy(genome,space,config['ticks'],smoothing=config['smoothing']) as p:
        if instrumented:p.lib.ns_profile_clear()
        for seed in range(2001000,2001000+episodes):
            obs,_=env.reset(seed=seed);p.reset(seed);total=0
            for _ in range(config['max_steps']):
                t=time.perf_counter();x=space.encode(obs);phases['encode']+=time.perf_counter()-t
                t=time.perf_counter();rc=p.lib.ns_step(p.ptr,x.ctypes.data_as(ct.POINTER(ct.c_float)),p.ticks,p.input_gain,p.last_spikes.ctypes.data_as(ct.POINTER(ct.c_float)));phases['native']+=time.perf_counter()-t
                if rc:raise RuntimeError(p.lib.ns_error().decode())
                t=time.perf_counter();p.activity*=1-p.smoothing;p.activity+=p.smoothing*p.last_spikes;action=space.decode(p.activity,p.rng);phases['decode']+=time.perf_counter()-t
                t=time.perf_counter();obs,reward,done,truncated,_=env.step(action);phases['environment']+=time.perf_counter()-t
                steps+=1;total+=reward
                if done or truncated:break
            returns.append(total)
        native_phases=None
        if instrumented:
            out=(ct.c_double*6)();p.lib.ns_profile(out)
            native_phases=dict(zip(['input_clear','synapse_delivery','neuron_integration','plasticity','homeostasis','calls'],out))
    elapsed=time.perf_counter()-start;env.close();phases['other']=elapsed-sum(phases.values())
    return {'seconds':elapsed,'steps':steps,'steps_per_second':steps/elapsed,'phase_seconds':phases,'phase_percent':{k:100*v/elapsed for k,v in phases.items()},'native_phases':native_phases,'returns':returns}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--episodes',type=int,default=20);parser.add_argument('--output',default='runs/gym-profile.json');args=parser.parse_args()
    config=json.loads((a.ROOT/'examples/gym-cartpole/results.json').read_text())['config']
    cases={'champion':a.Genome.load(a.ROOT/'examples/gym-cartpole/champion.json')}
    for hidden,edges in ((64,256),(256,1024)):
        cases[f'hidden{hidden}']=a.random_genome(8,2,np.random.default_rng(21),hidden,edges)
    normal=a.native();instrumented=load_instrumented();report={}
    for name,genome in cases.items():
        a._library=normal;baseline=measure(genome,config,args.episodes)
        a._library=instrumented;detail=measure(genome,config,args.episodes,True)
        assert baseline['returns']==detail['returns'], 'Instrumentation altered episode returns'
        report[name]={'neurons':len(genome.neurons),'synapses':len(genome.synapses),'baseline':baseline,'instrumented':detail}
        print(name,json.dumps(report[name]),flush=True)
    a._library=normal;path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
