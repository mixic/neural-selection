"""Evaluate a population in lockstep with persistent CUDA brain state."""
import numpy as np
from .adapter import SpaceAdapter
from .environments import make_env
from .gpu import GpuPolicyBatch


def evaluate_batch(genomes,config,seeds,env_pool=None):
    envs=[]
    try:
        for _ in (genomes if env_pool is None else genomes[:1]):envs.append(make_env(config))
        adapter=SpaceAdapter(envs[0].observation_space,envs[0].action_space,config['observation_scale'])
        results=[{'returns':[],'motor_spikes':0.,'steps':0} for _ in genomes]
        with GpuPolicyBatch(genomes,adapter,ticks=config['ticks'],smoothing=config['smoothing']) as policy:
            seed_matrix=np.asarray(seeds,dtype=np.uint32)
            if seed_matrix.ndim==1:seed_matrix=np.broadcast_to(seed_matrix,(len(genomes),len(seed_matrix)))
            if seed_matrix.ndim!=2 or seed_matrix.shape[0]!=len(genomes):raise ValueError('Episode seed batch differs')
            for episode_seeds in seed_matrix.T:
                observations=env_pool.reset(episode_seeds) if env_pool else [env.reset(seed=int(seed))[0] for env,seed in zip(envs,episode_seeds)]
                policy.reset(episode_seeds);active=np.ones(len(genomes),np.uint8);totals=np.zeros(len(genomes))
                for _ in range(config['max_steps']):
                    actions=policy.act(observations,active)
                    if env_pool:
                        observations,rewards,dones=env_pool.step(actions)
                    for i in np.flatnonzero(active):
                        if env_pool:reward=rewards[i];terminated=dones[i];truncated=False
                        else:observations[i],reward,terminated,truncated,_=envs[i].step(actions[i])
                        totals[i]+=float(reward)
                        results[i]['motor_spikes']+=float(policy.last_spikes[i].sum());results[i]['steps']+=1
                        if terminated or truncated:active[i]=0
                    if not active.any():break
                for result,total in zip(results,totals):result['returns'].append(float(total))
        for result in results:result['mean']=float(np.mean(result['returns']))
        return results
    finally:
        for env in envs:env.close()
