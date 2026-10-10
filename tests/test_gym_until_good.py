import json
from pathlib import Path
from unittest.mock import patch

from gym_adapter import until_good as goals
from gym_adapter.memory_assay import cue_location, set_cue
from gym_adapter.environments import make_env

ROOT=Path(__file__).resolve().parents[1]


def test_goal_requires_confirmation_and_never_changes_training(tmp_path):
    source=ROOT/'runs/gym-suite-corrected/acrobot/evolution.json'
    directory=tmp_path/'training';directory.mkdir()
    checkpoint=directory/'evolution.json';checkpoint.write_bytes(source.read_bytes())
    original=checkpoint.read_bytes()
    state={'audit_seed_cursor':0,'experiments':{'acrobot':{}}}
    seen=[]
    def evaluate(job):
        seen.extend(job[2]);return {'mean':-90 if len(seen)==100 else -101,'returns':[-90]*100}
    with patch.object(goals,'evaluate',evaluate):goals.audit('acrobot',directory,tmp_path,state,100)
    assert state['experiments']['acrobot']['status']=='queued'
    assert len(seen)==len(set(seen))==200
    assert checkpoint.read_bytes()==original
    # After a failed confirmation, use completely fresh sets for both rounds.
    with patch.object(goals,'evaluate',return_value={'mean':-90,'returns':[-90]*100}):
        goals.audit('acrobot',directory,tmp_path,state,100)
    report=json.loads((tmp_path/'acrobot/good.json').read_text())
    assert report['accepted']
    assert min(report['first']['seeds'])>max(seen)
    assert set(report['first']['seeds']).isdisjoint(report['confirmation']['seeds'])
    assert checkpoint.read_bytes()==original


def test_memory_needs_cue_and_neural_state_evidence():
    goal=goals.GOALS['minigrid-memory']
    assert not goals.passes({'mean':.95,'success_rate':1},goal)
    result={'mean':.95,'success_rate':1,'memory_assay':{'cue_hidden_success':1,'paired_success':1,'neural_reset_success':.5,'success_drop':.5}}
    assert goals.passes(result,goal)
    result['memory_assay']['neural_reset_success']=.9
    assert not goals.passes(result,goal)


def test_counterfactual_cue_changes_goal_and_only_cue_observation():
    import numpy as np
    config=json.loads((ROOT/'runs/gym-suite-corrected/minigrid-memory/evolution.json').read_text())['config']
    env=make_env(config)
    try:
        for seed in range(12):
            env.reset(seed=seed);base=env.unwrapped
            for direction in range(4):
                base.agent_dir=direction
                set_cue(base,'key');key=base.gen_obs()['image'];success=base.success_pos
                location=cue_location(base,key)
                set_cue(base,'ball');ball=base.gen_obs()['image']
                assert base.failure_pos==success
                assert location==cue_location(base,ball)
                changed=np.argwhere(np.any(key!=ball,axis=2))
                if location is None:assert len(changed)==0
                else:
                    assert changed.tolist()==[list(location)]
                    assert key[location][0]==5 and ball[location][0]==6
    finally:env.close()
