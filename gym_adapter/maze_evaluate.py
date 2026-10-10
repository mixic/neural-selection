"""GPU maze evaluation with unchanged genomes, episode resets and action RNG."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from .adapter import SpaceAdapter
from .environments import make_env
from .gpu import GpuPolicyBatch
from .maze_gpu import GpuMazeBatch,reset_world,SUPPORTED


class MazeResetPool:
    def __init__(self,config,count,workers=8):
        if config['env']!=SUPPORTED:raise ValueError('Unsupported GPU environment')
        self.config=config;self.count=count
        self.executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'))
    def reset_worlds(self,seeds):
        if len(seeds)!=self.count:raise ValueError('Episode batch differs')
        return list(self.executor.map(reset_world,[(self.config,int(s)) for s in seeds],chunksize=32))
    def close(self):self.executor.shutdown()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def evaluate_maze_episodes(genomes,config,seeds,env_pool,timings=None):
    """Equivalent to evaluate_episodes; host resets, CUDA transitions and views."""
    if config['env']!=SUPPORTED:raise ValueError('Unsupported GPU environment')
    timings={} if timings is None else timings
    episodes=len(seeds[0]);expanded=[g for g in genomes for _ in range(episodes)]
    episode_seeds=np.asarray(seeds,np.uint32).reshape(-1);count=len(expanded)
    if episode_seeds.shape!=(count,):raise ValueError('Episode seeds differ')
    env=make_env(config)
    try:adapter=SpaceAdapter(env.observation_space,env.action_space,config['observation_scale'])
    finally:env.close()
    start=time.perf_counter();worlds=env_pool.reset_worlds(episode_seeds);timings['cpu_reset']=time.perf_counter()-start
    start=time.perf_counter()
    with GpuPolicyBatch(expanded,adapter,ticks=config['ticks'],smoothing=config['smoothing']) as policy,GpuMazeBatch(count,adapter) as maze:
        timings['policy_create']=time.perf_counter()-start
        start=time.perf_counter();maze.reset(worlds);policy.reset(episode_seeds);timings['reset_upload']=time.perf_counter()-start
        active=np.ones(count,np.uint8);totals=np.zeros(count);spikes=np.zeros(count);steps=np.zeros(count,np.int64)
        brain_time=environment_time=accounting_time=0.
        start=time.perf_counter()
        for _ in range(config['max_steps']):
            t=time.perf_counter();actions=maze.act(policy,active);brain_time+=time.perf_counter()-t
            t=time.perf_counter();maze.step(actions);rewards,dones=maze.rewards,maze.dones;environment_time+=time.perf_counter()-t
            t=time.perf_counter();selected=active.astype(bool);totals[selected]+=rewards[selected];spikes[selected]+=policy.last_spikes[selected].sum(axis=1);steps[selected]+=1
            active[dones.astype(bool)]=0;accounting_time+=time.perf_counter()-t
            if not active.any():break
        timings['rollout']=time.perf_counter()-start;timings['brain_and_action_decode']=brain_time;timings['environment_and_results']=environment_time;timings['accounting']=accounting_time
    results=[]
    for i in range(len(genomes)):
        section=slice(i*episodes,(i+1)*episodes);returns=totals[section].tolist()
        results.append({'mean':float(np.mean(returns)),'returns':returns,'motor_spikes':float(sum(spikes[section])),'steps':int(sum(steps[section]))})
    return results
