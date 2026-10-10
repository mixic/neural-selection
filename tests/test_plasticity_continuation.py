"""Censored outcomes use post-assignment reproduction and trial-level intervals."""
import csv
from tools.plasticity_continuation import summarize_trial, aggregate


def test_post_assignment_outcomes_and_censoring(tmp_path):
    path=tmp_path/'cohort.csv'
    fields=['frozen','alive','start_step','observed_step','initial_offspring',
            'final_offspring','death_cause']
    with path.open('w') as f:
        out=csv.DictWriter(f,fieldnames=fields);out.writeheader()
        for row in [(1,0,100,105,20,22,1),(1,1,100,110,100,101,0),
                    (0,1,100,110,50,54,0),(0,1,100,110,80,80,0)]:
            out.writerow(dict(zip(fields,row)))
    result=summarize_trial(path)
    assert result['differences_frozen_minus_control']=={
        'survival_fraction':-.5,'mean_remaining_life_steps':-2.5,
        'mean_additional_offspring':-.5}
    assert result['groups']['frozen']['deaths_by_cause']['1']==1
    assert aggregate([result,result])['survival_fraction']['bootstrap_95_percent_interval'] is None
    assert aggregate([result]*3)['survival_fraction']['bootstrap_95_percent_interval']==[-.5,-.5]
