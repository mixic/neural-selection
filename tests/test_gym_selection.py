import json
from types import SimpleNamespace
import numpy as np
from gym_adapter import evolve


def test_training_seeds_are_unique_disjoint_and_resume_exactly():
    first=evolve.fresh_training_seeds(21,0,10000)
    resumed=evolve.fresh_training_seeds(21,10000,10000)
    assert first+resumed==evolve.fresh_training_seeds(21,0,20000)
    assert len(set(first+resumed))==20000
    assert all(2**31<=seed<2**32 for seed in first+resumed)


def test_validation_cannot_change_breeding_champion_rng_or_stop(monkeypatch,tmp_path):
    class InlinePool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,fn,jobs):return map(fn,jobs)
    monkeypatch.setattr(evolve,'ProcessPoolExecutor',InlinePool)
    checkpoints=[]
    for sign in (-1,1):
        training_seeds=[];validation_calls=[]
        def evaluate(job):
            genome,config,seeds=job
            if all(seed>=2**31 for seed in seeds):
                training_seeds.extend(seeds)
                mean=float(sum(s[2] for s in genome.synapses))
            else:
                validation_calls.append(seeds)
                mean=sign*1e9*len(validation_calls)
            return {'mean':mean,'returns':[mean]*len(seeds),'motor_spikes':0.,'steps':1}
        monkeypatch.setattr(evolve,'evaluate',evaluate)
        output=tmp_path/str(sign)
        args=SimpleNamespace(population=8,generations=6,episodes=3,validation_episodes=4,test_episodes=2,workers=1,max_steps=20,hidden=2,connections=8,mutation_rate=.5,ticks=16,smoothing=.4,output=str(output),resume=None,backend='cpu',env='CartPole-v1',obs_scale=None,seed=13,test_seed=8000000,skip_final_evaluation=True,target_return=10000)
        evolve.train(args)
        checkpoints.append(json.loads((output/'evolution.json').read_text()))
        assert len(training_seeds)==len(set(training_seeds))==8*3*6
        assert len(validation_calls)==6
        assert validation_calls==[validation_calls[0]]*6
    left,right=checkpoints
    assert left['champion_score']!=right['champion_score']
    assert left['next_generation']==right['next_generation']==6
    for key in ('population','champion','initial','rng','training_seed_cursor'):
        assert left[key]==right[key],key
