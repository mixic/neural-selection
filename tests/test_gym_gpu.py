import numpy as np
import pytest
from gymnasium import spaces
from gym_adapter.adapter import SpaceAdapter,SpikingPolicy,Genome,neuron,random_genome
from gym_adapter.environments import make_env
from pathlib import Path


def test_minigrid_uses_full_partial_view_and_unsigned_bounds():
    pytest.importorskip('minigrid')
    env=make_env({'env':'MiniGrid-Empty-5x5-v0'})
    try:
        a=SpaceAdapter(env.observation_space,env.action_space);obs,_=env.reset(seed=0)
        assert obs.shape==(7,7,3) and a.channels==294
        assert env.observation_space.contains(obs)
        encoded=a.encode(obs);assert np.isfinite(encoded).all() and (encoded>=0).all() and (encoded<=1).all()
        # Exercise channels beyond the ecological simulator's signed-byte field.
        g=Genome([neuron(channel=i) for i in range(a.channels)]+[neuron() for _ in range(a.outputs)],[],list(range(a.channels,a.channels+a.outputs)),a.channels)
        with SpikingPolicy(g,a) as p:p.act(obs)
    finally:env.close()


@pytest.mark.skipif(not Path('build/libgym_brain_gpu.so').exists(),reason='Optional CUDA backend has not been built')
def test_cuda_batch_matches_spikes_masks_and_episode_reset():
    from gym_adapter.gpu import GpuPolicyBatch
    a=SpaceAdapter(spaces.Box(-1,1,(4,),dtype=np.float32),spaces.Discrete(2,start=3))
    rng=np.random.default_rng(3);gs=[random_genome(8,2,rng,hidden=h,connections=32+h*2) for h in (0,2,8,64)]
    for g in gs:
        for j,s in enumerate(g.synapses):s[9]=j%6;s[10]=j%8
    ps=[SpikingPolicy(g,a) for g in gs]
    try:
        with GpuPolicyBatch(gs,a) as batch:
            seeds=np.arange(4,dtype=np.uint32)+20;batch.reset(seeds)
            for p,seed in zip(ps,seeds):p.reset(int(seed))
            for step in range(80):
                mask=np.array([1,step%2,1,step%3==0],np.uint8);obs=rng.uniform(-1,1,(4,4)).astype(np.float32)
                actions=batch.act(obs,mask)
                for i,p in enumerate(ps):
                    if mask[i]:
                        assert p.act(obs[i])==actions[i]
                        assert np.array_equal(p.last_spikes,batch.last_spikes[i])
                    else:assert actions[i] is None and not batch.last_spikes[i].any()
            batch.reset(seeds);p=ps[0];p.reset(int(seeds[0]))
            for _ in range(30):
                obs=np.zeros((4,4),np.float32);assert batch.act(obs)[0]==p.act(obs[0]);assert np.array_equal(batch.last_spikes[0],p.last_spikes)
    finally:
        for p in ps:p.close()


@pytest.mark.skipif(not Path('build/libgym_brain_gpu.so').exists(),reason='Optional CUDA backend has not been built')
def test_cuda_population_uses_individual_episode_seeds():
    from gym_adapter.batch_evaluate import evaluate_batch
    from gym_adapter.env_pool import EnvPool
    from gym_adapter.evolve import evaluate,fresh_training_seeds
    config={'env':'CartPole-v1','ticks':16,'smoothing':.4,'observation_scale':None,'max_steps':40}
    rng=np.random.default_rng(7);genomes=[random_genome(8,2,rng,hidden=0,connections=8) for _ in range(4)]
    flat=fresh_training_seeds(14,0,12);seeds=[flat[i*3:(i+1)*3] for i in range(4)]
    expected=[evaluate((g,config,s)) for g,s in zip(genomes,seeds)]
    with EnvPool(config,4,2) as pool:actual=evaluate_batch(genomes,config,seeds,pool)
    assert [r['returns'] for r in actual]==[r['returns'] for r in expected]
