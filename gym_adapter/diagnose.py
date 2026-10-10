"""Frozen-genome sensory ablations, separate from training and final testing."""
import argparse
import json
from pathlib import Path
import numpy as np
from .adapter import Genome,SpaceAdapter,SpikingPolicy
from .environments import make_env


def measure(genome,config,seeds,condition):
    env=make_env(config);adapter=SpaceAdapter(env.observation_space,env.action_space,config['observation_scale']);encode=adapter.encode;returns=[];steps=[]
    try:
        with SpikingPolicy(genome,adapter,config['ticks'],smoothing=config['smoothing']) as policy:
            for seed in seeds:
                obs,_=env.reset(seed=seed);policy.reset(seed)
                if condition=='neutral':adapter.encode=lambda observation:np.full(adapter.channels,.5,np.float32)
                elif condition=='frozen':
                    first=encode(obs);adapter.encode=lambda observation:first.copy()
                else:adapter.encode=encode
                total=0
                for step in range(config['max_steps']):
                    obs,reward,terminated,truncated,_=env.step(policy.act(obs));total+=float(reward)
                    if terminated or truncated:break
                returns.append(total);steps.append(step+1)
    finally:env.close()
    result={'returns':returns,'mean':float(np.mean(returns)),'episode_lengths':steps}
    if config['env'].startswith('MiniGrid-'):result['success_rate']=float(np.mean(np.array(returns)>0))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment');p.add_argument('--episodes',type=int,default=100);p.add_argument('--seed',type=int,default=6000000);args=p.parse_args()
    path=Path(args.experiment);config=json.loads((path/'results.json').read_text())['config'];genome=Genome.load(path/'champion.json');seeds=list(range(args.seed,args.seed+args.episodes))
    result={'config':config,'seeds':seeds,'purpose':'Post-training sensory-dependence diagnostic; these results never select genomes','conditions':{}}
    for condition in ('normal','neutral','frozen'):
        result['conditions'][condition]=measure(genome,config,seeds,condition);print(condition,result['conditions'][condition]['mean'],result['conditions'][condition].get('success_rate'),flush=True)
    (path/'sensory-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
