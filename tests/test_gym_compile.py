import os
import numpy as np
from gymnasium import spaces
from gym_adapter.adapter import SpaceAdapter,SpikingPolicy,random_genome
from gym_adapter.compile_graph import compile_genome


def test_compiled_graph_preserves_motor_traces_and_original_rng(monkeypatch):
    a=SpaceAdapter(spaces.Box(-1,1,(4,),dtype=np.float32),spaces.Discrete(3))
    rng=np.random.default_rng(53)
    for trial in range(3):
        g=random_genome(8,3,rng,hidden=45,connections=180)
        # Mix recurrent edges, disabled genes, inhibitory and modulatory sources,
        # and all plasticity phenotypes. Leave a disconnected component.
        for i,n in enumerate(g.neurons[11:],11):
            n[4]=i%4;n[2]=int(i%7!=0)
        for i,e in enumerate(g.synapses):
            e[9]=i%6;e[10]=i%8;e[3]=int(i%5!=0)
            if e[1]>=35:e[0]=int(rng.integers(35,len(g.neurons)))
            else:e[0]=int(rng.integers(35))
        assert len(compile_genome(g).neurons)<len(g.neurons)
        monkeypatch.setenv('GYM_FULL_GRAPH','1');full=SpikingPolicy(g,a)
        monkeypatch.delenv('GYM_FULL_GRAPH');compiled=SpikingPolicy(g,a)
        try:
            for seed in [12,2**31+47]:
                full.reset(seed);compiled.reset(seed)
                for step in range(160):
                    obs=rng.uniform(-1,1,4).astype(np.float32)
                    assert full.act(obs)==compiled.act(obs)
                    assert np.array_equal(full.last_spikes,compiled.last_spikes)
        finally:full.close();compiled.close()


def test_episode_batch_preserves_returns_spikes_and_seed_order():
    from gym_adapter.env_pool import EnvPool
    from gym_adapter.batch_evaluate import evaluate_batch
    from gym_adapter.episode_batch import evaluate_episodes
    config={'env':'CartPole-v1','ticks':16,'smoothing':.4,'observation_scale':None,'max_steps':60}
    rng=np.random.default_rng(71)
    genomes=[random_genome(8,2,rng,hidden=8,connections=32) for _ in range(4)]
    seeds=[[2**31+11+i*3+j for j in range(3)] for i in range(4)]
    with EnvPool(config,4,2) as pool:expected=evaluate_batch(genomes,config,seeds,pool)
    with EnvPool(config,12,2) as pool:actual=evaluate_episodes(genomes,config,seeds,pool)
    assert actual==expected


def test_compiled_minigrid_observations_match_reference(monkeypatch):
    from gym_adapter.environments import make_env
    for name in ['MiniGrid-Empty-Random-6x6-v0','MiniGrid-MemoryS7-v0']:
        monkeypatch.setenv('GYM_REFERENCE_OBSERVATION','1');reference=make_env({'env':name})
        monkeypatch.delenv('GYM_REFERENCE_OBSERVATION');fast=make_env({'env':name})
        rng=np.random.default_rng(12)
        try:
            for seed in range(30):
                x,_=reference.reset(seed=seed);y,_=fast.reset(seed=seed);assert np.array_equal(x,y)
                for _ in range(150):
                    action=int(rng.integers(7));a=reference.step(action);b=fast.step(action)
                    assert np.array_equal(a[0],b[0]);assert a[1:4]==b[1:4]
                    if a[2] or a[3]:break
        finally:reference.close();fast.close()
