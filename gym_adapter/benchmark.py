"""Compare CPU and CUDA inference and complete population evaluations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes as ct
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from .adapter import Genome,SpaceAdapter,SpikingPolicy,random_genome,ROOT
from .environments import make_env
from .gpu import GpuPolicyBatch
from .evolve import evaluate
from .batch_evaluate import evaluate_batch
from .env_pool import EnvPool


def inference(genomes,adapter,repeats):
    observations=np.full((len(genomes),adapter.channels),.5,np.float32)
    pointers=[row.ctypes.data_as(ct.POINTER(ct.c_float)) for row in observations]
    policies=[SpikingPolicy(g,adapter) for g in genomes]
    start=time.perf_counter()
    for _ in range(repeats):
        for p,x in zip(policies,pointers):
            if p.lib.ns_step(p.ptr,x,p.ticks,p.input_gain,p._output_pointer):raise RuntimeError(p.lib.ns_error().decode())
    cpu=time.perf_counter()-start
    for p in policies:p.close()
    start=time.perf_counter()
    with GpuPolicyBatch(genomes,adapter) as batch:
        setup=time.perf_counter()-start
        for _ in range(10):batch.step_encoded(observations)
        start=time.perf_counter()
        for _ in range(repeats):batch.step_encoded(observations)
        cuda=time.perf_counter()-start
    return {'cpu_seconds':cpu,'cuda_seconds':cuda,'cuda_setup_seconds':setup,'speedup_vs_single_cpu':cpu/cuda,'brain_actions':len(genomes)*repeats}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--env',default='CartPole-v1');p.add_argument('--population',type=int,default=128);p.add_argument('--repeats',type=int,default=100);p.add_argument('--workers',type=int,default=8);p.add_argument('--output',default='runs/gym-benchmark.json');args=p.parse_args()
    config=json.loads((ROOT/'examples/gym-cartpole/results.json').read_text())['config'];
    if args.env!='CartPole-v1':config.update(env=args.env,observation_scale=None,max_steps=100)
    env=make_env(config);adapter=SpaceAdapter(env.observation_space,env.action_space,config['observation_scale']);env.close()
    cases={'champion':[Genome.load(ROOT/'examples/gym-cartpole/champion.json') for _ in range(args.population)]} if args.env=='CartPole-v1' else {'founders':[random_genome(adapter.channels,adapter.outputs,np.random.default_rng(i+11),8,96) for i in range(args.population)]}
    for h in ((64,256) if args.env=='CartPole-v1' else ()):cases[f'hidden{h}']=[random_genome(8,2,np.random.default_rng(i+11),h,h*4) for i in range(args.population)]
    report={}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        list(pool.map(evaluate,[(next(iter(cases.values()))[0],config,[2001000])]*args.workers))
        for name,genomes in cases.items():
            result={'inference':inference(genomes,adapter,args.repeats)}
            start=time.perf_counter();cpu=list(pool.map(evaluate,[(g,config,[2001000,2001001]) for g in genomes]));cpu_seconds=time.perf_counter()-start
            with EnvPool(config,len(genomes),args.workers) as environments:
                environments.reset(2000999) # Warm startup, as for CPU workers.
                start=time.perf_counter();cuda=evaluate_batch(genomes,config,[2001000,2001001],environments);cuda_seconds=time.perf_counter()-start
            result['evaluation']={'cpu_workers':args.workers,'cpu_seconds':cpu_seconds,'cuda_seconds':cuda_seconds,'speedup':cpu_seconds/cuda_seconds,'identical_returns':all(c['returns']==g['returns'] for c,g in zip(cpu,cuda)),'cpu_steps':sum(r['steps'] for r in cpu),'cuda_steps':sum(r['steps'] for r in cuda)}
            report[name]=result;print(name,json.dumps(result),flush=True)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
