import json
import numpy as np
import pytest
from gymnasium import spaces
from gym_adapter.adapter import Genome, SpaceAdapter, SpikingPolicy, neuron, random_genome, mutate


def test_generic_spaces_and_bounded_actions():
    obs=spaces.Dict({'x':spaces.Box(-1,1,(2,),dtype=np.float32),'kind':spaces.Discrete(3)})
    actions=spaces.Box(np.array([-2.,0.],dtype=np.float32),np.array([2.,10.],dtype=np.float32))
    adapter=SpaceAdapter(obs,actions)
    encoded=adapter.encode({'x':np.array([-.5,.5],dtype=np.float32),'kind':2})
    assert encoded.shape==(10,)
    assert np.allclose(encoded.reshape(-1,2).sum(1),1)
    assert np.isfinite(encoded).all()
    assert actions.contains(adapter.decode(np.array([100.,0.,0.,100.],dtype=np.float32),np.random.default_rng(0)))
    with pytest.raises(ValueError):SpaceAdapter(spaces.Box(-np.inf,np.inf,(32768,)),spaces.Discrete(2))
    with pytest.raises(ValueError):SpaceAdapter(spaces.Box(-1,1,(2,)),actions,[1,-1])


def test_reset_clears_episode_state_and_replays_actions():
    rng=np.random.default_rng(2)
    adapter=SpaceAdapter(spaces.Box(-1,1,(2,)),spaces.Discrete(2,start=3))
    g=random_genome(adapter.channels,adapter.outputs,rng)
    obs=np.array([.4,-.2],dtype=np.float32)
    with SpikingPolicy(g,adapter) as p:
        p.reset(19);first=[(p.act(obs),p.last_spikes.copy()) for _ in range(40)]
        p.reset(19);second=[(p.act(obs),p.last_spikes.copy()) for _ in range(40)]
        assert [a for a,_ in first]==[a for a,_ in second]
        assert all(np.array_equal(a,b) for (_,a),(_,b) in zip(first,second))
        assert all(adapter.action_space.contains(a) for a,_ in first)


def test_unconnected_motor_does_not_receive_bias():
    adapter=SpaceAdapter(spaces.Box(-1,1,(1,)),spaces.Discrete(2))
    g=Genome([neuron(channel=0),neuron(channel=1),neuron(),neuron()],[],[2,3],2)
    with SpikingPolicy(g,adapter) as p:
        for _ in range(100):
            p.act(np.array([1.],dtype=np.float32))
            assert not p.last_spikes.any()


def test_add_remove_mutations_preserve_ports_and_valid_graph(tmp_path):
    rng=np.random.default_rng(6);g=random_genome(4,2,rng,hidden=4)
    baseline=g.clone();observed_sizes=set();observed_edges=set()
    for _ in range(200):
        g=mutate(g,rng);observed_sizes.add(len(g.neurons));observed_edges.add(len(g.synapses))
        assert g.motors==baseline.motors
        assert len(g.neurons)>=6
        assert all(0<=s[0]<len(g.neurons) and 0<=s[1]<len(g.neurons) for s in g.synapses)
    assert len(observed_sizes)>1 and len(observed_edges)>1
    assert not any('state' in name for name in vars(g))
    path=tmp_path/'genome.json';g.save(path);assert Genome.load(path)==g


def test_recurrence_self_connections_and_invalid_genes():
    adapter=SpaceAdapter(spaces.Box(-1,1,(1,)),spaces.Discrete(1))
    g=Genome([neuron(channel=0),neuron(channel=1),neuron()],[[2,2,3.,1,2,0,.8,60.,0.,1,0]],[2],2)
    with SpikingPolicy(g,adapter) as p:p.act(np.zeros(1,dtype=np.float32))
    g.synapses[0][1]=99
    with pytest.raises(ValueError):SpikingPolicy(g,adapter)


def test_connections_carry_sensory_activity_and_leave_genome_unchanged():
    adapter=SpaceAdapter(spaces.Box(-1,1,(1,)),spaces.Discrete(1))
    g=Genome([neuron(channel=0),neuron(channel=1),neuron()],[[0,2,8.,1,1,0,.8,60.,0.,1,0]],[2],2)
    before=g.clone()
    with SpikingPolicy(g,adapter) as p:
        spikes=0
        for _ in range(40):
            p.act(np.ones(1,dtype=np.float32));spikes+=p.last_spikes.sum()
        assert spikes>0
        assert g==before
    p.close()
    with pytest.raises(RuntimeError):p.act(np.ones(1,dtype=np.float32))


@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_evolution_resume_matches_continuous_search(tmp_path,backend):
    if backend=='cuda' and not __import__('pathlib').Path('build/libgym_brain_gpu.so').exists():pytest.skip('Optional CUDA backend has not been built')
    import subprocess,sys
    common=['--backend',backend,'--population','4','--episodes','1','--validation-episodes','1','--test-episodes','1','--hidden','0','--connections','8','--workers','1','--max-steps','40']
    continuous=tmp_path/'continuous';resumed=tmp_path/'resumed'
    def run(out,generations,resume=None):
        command=[sys.executable,'-m','gym_adapter.evolve',*common,'--output',str(out),'--generations',str(generations)]
        if resume:command+=['--resume',str(resume)]
        subprocess.run(command,check=True,capture_output=True,text=True)
    run(continuous,3);run(resumed,2);run(resumed,3,resumed/'evolution.json')
    assert json.loads((continuous/'evolution.json').read_text())==json.loads((resumed/'evolution.json').read_text())
    assert (continuous/'champion.json').read_bytes()==(resumed/'champion.json').read_bytes()
