"""CUDA maze parity against real MiniGrid, including mutable object identities."""
import numpy as np
import pytest
from gym_adapter.adapter import SpaceAdapter,ROOT
from gym_adapter.environments import make_env
from gym_adapter.maze_gpu import GpuMazeBatch,export_world,SUPPORTED

pytestmark=pytest.mark.skipif(not (ROOT/'build/libgym_brain_gpu.so').exists(),reason='Optional CUDA backend has not been built')
CONFIG={'env':SUPPORTED,'observation_scale':None}


def test_mutable_transitions_and_visibility():
    from minigrid.core.world_object import Door,Key,Ball,Box,Wall,Floor,Goal,Lava
    # Exercise every action from all orientations, with correct/wrong keys,
    # carried boxes (including their contents), and target/non-target balls.
    cases=[]
    for direction in range(4):
        for kind in ['empty','wall','floor','goal','lava','open','closed','locked','key','target','ball','box','empty_box']:
            for carrying in ['none','correct_key','wrong_key','ball','box','target']:
                for action in range(7):cases.append((direction,kind,carrying,action))
    envs=[];worlds=[];refs=[];adapter=None
    # Keep batches modest so the assertions can inspect every object.
    for first in range(0,len(cases),128):
        chunk=cases[first:first+128];envs=[];worlds=[];refs=[]
        try:
            for direction,kind,carrying,action in chunk:
                env=make_env(CONFIG);env.reset(seed=123);envs.append(env);b=env.unwrapped
                b.agent_pos=(7,7);b.agent_dir=direction
                b.grid.set(7,7,None)
                for x,y in [(6,7),(8,7),(7,6),(7,8)]:b.grid.set(x,y,None)
                b.carrying={'none':None,'correct_key':Key('yellow'),'wrong_key':Key('red'),'ball':Ball('red'),'box':Box('purple',Key('green')),'target':b.obj}[carrying]
                cell={'empty':None,'wall':Wall(),'floor':Floor(),'goal':Goal(),'lava':Lava(),'open':Door('yellow',is_open=True),'closed':Door('yellow'),'locked':Door('yellow',is_locked=True),'key':Key('yellow'),'target':b.obj,'ball':Ball(b.obj.color),'box':Box('purple',Key('yellow')),'empty_box':Box('purple',None)}[kind]
                b.grid.set(*b.front_pos,cell)
                world=export_world(env);worlds.append(world[:3]);refs.append(world[3])
            adapter=SpaceAdapter(envs[0].observation_space,envs[0].action_space)
            with GpuMazeBatch(len(chunk),adapter) as batch:
                batch.reset(worlds);batch.step([c[3] for c in chunk])
                snapshots=batch.snapshot();rewards,dones=batch.read_results()
                for i,(env,case) in enumerate(zip(envs,chunk)):
                    obs,reward,terminated,truncated,_=env.step(case[3]);b=env.unwrapped;ids={obj:j for j,obj in enumerate(refs[i])}
                    np.testing.assert_array_equal(snapshots[0][i],obs)
                    np.testing.assert_array_equal(snapshots[1][i],[ids.get(obj,-1) for obj in b.grid.grid])
                    for j,obj in enumerate(refs[i]):np.testing.assert_array_equal(snapshots[2][i,j,:3],obj.encode())
                    np.testing.assert_array_equal(snapshots[3][i,:4],[*b.agent_pos,b.agent_dir,ids.get(b.carrying,-1)])
                    assert rewards[i]==reward;assert bool(dones[i])==(terminated or truncated)
        finally:
            for env in envs:env.close()


def test_seeded_trajectories_and_horizon():
    count=16;envs=[make_env(CONFIG) for _ in range(count)]
    try:
        for i,env in enumerate(envs):env.reset(seed=10000+i);env.unwrapped.max_steps=400
        worlds=[export_world(env) for env in envs];adapter=SpaceAdapter(envs[0].observation_space,envs[0].action_space)
        rng=np.random.default_rng(61)
        with GpuMazeBatch(count,adapter) as batch:
            batch.reset([w[:3] for w in worlds]);active=np.ones(count,bool)
            for tick in range(400):
                actions=rng.integers(0,7,count);batch.step([int(a) if active[i] else None for i,a in enumerate(actions)])
                obs,grids,objects,agents=batch.snapshot();rewards,dones=batch.read_results()
                for i,env in enumerate(envs):
                    if not active[i]:continue
                    ref,reward,terminated,truncated,_=env.step(int(actions[i]));ids={obj:j for j,obj in enumerate(worlds[i][3])}
                    np.testing.assert_array_equal(obs[i],ref)
                    np.testing.assert_array_equal(grids[i],[ids.get(obj,-1) for obj in env.unwrapped.grid.grid])
                    for j,obj in enumerate(worlds[i][3]):np.testing.assert_array_equal(objects[i,j,:3],obj.encode())
                    assert rewards[i]==reward;assert bool(dones[i])==(terminated or truncated)
                    active[i]=not(terminated or truncated)
            assert not active.any()
    finally:
        for env in envs:env.close()


def test_brain_driven_trajectory_parity():
    from gym_adapter.gpu import GpuPolicyBatch
    from gym_adapter.adapter import random_genome
    count=32;envs=[make_env(CONFIG) for _ in range(count)]
    try:
        seeds=np.arange(40100,40100+count,dtype=np.uint32);observations=[e.reset(seed=int(s))[0] for e,s in zip(envs,seeds)]
        adapter=SpaceAdapter(envs[0].observation_space,envs[0].action_space);rng=np.random.default_rng(16)
        genomes=[random_genome(adapter.channels,adapter.outputs,rng,hidden=16,connections=160) for _ in range(count)]
        with GpuPolicyBatch(genomes,adapter) as reference,GpuPolicyBatch(genomes,adapter) as gpu,GpuMazeBatch(count,adapter) as maze:
            reference.reset(seeds);gpu.reset(seeds);maze.reset([export_world(e)[:3] for e in envs]);mask=np.ones(count,np.uint8)
            for tick in range(256):
                expected=reference.act(observations,mask);actual=maze.act(gpu,mask)
                assert actual==expected;np.testing.assert_array_equal(gpu.last_spikes,reference.last_spikes);np.testing.assert_array_equal(gpu.activity,reference.activity)
                maze.step(actual);rewards,dones=maze.read_results()
                for i,e in enumerate(envs):
                    if not mask[i]:continue
                    observations[i],reward,terminated,truncated,_=e.step(expected[i]);assert rewards[i]==reward;assert bool(dones[i])==(terminated or truncated)
                    if terminated or truncated:mask[i]=0
                np.testing.assert_array_equal(maze.snapshot()[0],observations)
    finally:
        for env in envs:env.close()
