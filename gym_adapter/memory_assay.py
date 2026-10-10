"""Counterfactual cue and neural-state tests; never used for breeding."""
from __future__ import annotations
import numpy as np
from .adapter import SpaceAdapter,SpikingPolicy
from .environments import make_env


def cue_location(base,obs):
    """Locate the start-room cue in MiniGrid's standard rotated partial view."""
    topx,topy,_,_=base.get_view_exts()
    x,y=1-topx,base.height//2-1-topy
    size=base.agent_view_size
    if not (0<=x<size and 0<=y<size):return None
    for _ in range(base.agent_dir+1):x,y=y,size-1-x
    if obs[x,y,0] not in (5,6):return None # green key/ball, actually visible
    return x,y


def set_cue(base,kind):
    from minigrid.core.world_object import Key,Ball
    base.grid.set(1,base.height//2-1,(Key if kind=='key' else Ball)('green'))
    top,bottom=base.success_pos,base.failure_pos
    if top[1]>bottom[1]:top,bottom=bottom,top
    target=base.grid.get(top[0],top[1]-1)
    if target.type==kind:base.success_pos,base.failure_pos=top,bottom
    else:base.success_pos,base.failure_pos=bottom,top


def trial(genome,config,seed,kind,reset_neural=False,policy_factory=SpikingPolicy):
    env=make_env(config)
    adapter=SpaceAdapter(env.observation_space,env.action_space,config['observation_scale'])
    try:
        obs,_=env.reset(seed=int(seed));base=env.unwrapped;set_cue(base,kind)
        obs=base.gen_obs()['image'];seen=False;reset_done=False;total=0.
        with policy_factory(genome,adapter,ticks=config['ticks'],smoothing=config['smoothing']) as policy:
            policy.reset(int(seed))
            for step in range(config['max_steps']):
                location=cue_location(base,obs)
                current=obs.copy()
                if seen:
                    if location is not None:current[location]=np.array([1,0,0],np.uint8)
                    if reset_neural and not reset_done:
                        # Keep decoder activity and action RNG intact. Ablate only
                        # neural state immediately after the first cue-bearing act.
                        policy.lib.ns_reset(policy.ptr,int(seed)&0xffffffff)
                        reset_done=True
                action=policy.act(current)
                if location is not None:seen=True
                obs,reward,terminated,truncated,_=env.step(action);total+=float(reward)
                if terminated or truncated:break
        return {'return':total,'success':total>0,'steps':step+1,'cue_seen':seen,'neural_reset':reset_done}
    finally:env.close()


def measure(genome,config,seeds):
    if config['env']!='MiniGrid-MemoryS7-v0':raise ValueError('Assay currently covers MemoryS7 only')
    normal=[];reset=[];pairs=[]
    for seed in seeds:
        n=[trial(genome,config,seed,kind) for kind in ['key','ball']]
        a=[trial(genome,config,seed,kind,True) for kind in ['key','ball']]
        normal.extend(n);reset.extend(a);pairs.append(all(r['success'] for r in n))
    success=float(np.mean([r['success'] for r in normal]));ablated=float(np.mean([r['success'] for r in reset]))
    return {'purpose':'Each layout/seed tested with both cue identities. Cue hidden after its first observed action. Control resets neural state once, preserving decoder activity and action RNG.',
            'seeds':[int(s) for s in seeds],'cue_hidden_success':success,'paired_success':float(np.mean(pairs)),
            'neural_reset_success':ablated,'success_drop':success-ablated,
            'cue_seen_rate':float(np.mean([r['cue_seen'] for r in normal])),
            'normal_trials':normal,'reset_trials':reset}
