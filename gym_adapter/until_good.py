"""Continue training to preregistered goals, with fresh stopping-only audits."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import subprocess
from .adapter import ROOT,Genome
from .evolve import evaluate,atomic_json

GOALS={
 'pendulum':{'mean_return':-200.},
 'acrobot':{'mean_return':-100.},
 'mountaincar':{'mean_return':-110.},
 'minigrid-memory':{'mean_return':.9,'success_rate':.95,'cue_hidden_success':.95,'paired_success':.95,'neural_reset_success_max':.60,'success_drop':.35},
 'doorkey':{'mean_return':.9,'success_rate':.95},
}


def passes(result,goal):
    if result['mean']<goal['mean_return']:return False
    if 'success_rate' in goal and result.get('success_rate',0)<goal['success_rate']:return False
    if 'cue_hidden_success' in goal:
        assay=result.get('memory_assay',{})
        for key in ['cue_hidden_success','paired_success','success_drop']:
            if assay.get(key,0)<goal[key]:return False
        if assay.get('neural_reset_success',1)>goal['neural_reset_success_max']:return False
    return True


def reserve_seeds(output,state,count):
    cursor=state['audit_seed_cursor'];start=60_000_000+cursor
    if start+count>=2**31:raise RuntimeError('Stopping audit seed range exhausted')
    state['audit_seed_cursor']+=count
    atomic_json(output/'status.json',state) # Reserve before evaluation; never reused on resume.
    return list(range(start,start+count))


def assessment(genome,config,seeds,goal):
    result=dict(evaluate((genome,config,seeds)));result['seeds']=seeds
    if config['env'].startswith('MiniGrid-'):result['success_rate']=sum(r>0 for r in result['returns'])/len(seeds)
    # Expensive state/cue assay only after ordinary task performance passes.
    basic={k:v for k,v in goal.items() if k in ['mean_return','success_rate']}
    if 'cue_hidden_success' in goal and passes(result,basic):
        from .memory_assay import measure
        result['memory_assay']=measure(genome,config,seeds)
    result['passed']=passes(result,goal)
    return result


def audit(name,directory,output,state,episodes):
    saved=json.loads((directory/'evolution.json').read_text())
    if saved['selection_protocol']!='fresh-training-only-v2':raise RuntimeError('Invalid selection protocol')
    generation=saved['next_generation'];genome=Genome(**saved['champion']);target=output/name;target.mkdir(exist_ok=True)
    snapshot=target/f'candidate-{generation}.json';genome.save(snapshot)
    first=assessment(genome,saved['config'],reserve_seeds(output,state,episodes),GOALS[name])
    report={'generation':generation,'genome':str(snapshot),'goal':GOALS[name],
            'purpose':'Fresh stopping-only evaluations. Never supplied to breeding, mutation, or training winner selection. Fixed validation remains report-only.',
            'first':first}
    if first['passed']:
        report['confirmation']=assessment(genome,saved['config'],reserve_seeds(output,state,episodes),GOALS[name])
    report['accepted']=first['passed'] and report.get('confirmation',{}).get('passed',False)
    atomic_json(target/f'audit-{generation}.json',report)
    entry=state['experiments'][name];entry.update(generation=generation,last_audit_generation=generation,
            audit=str(target/f'audit-{generation}.json'),audit_mean=first['mean'],audit_success=first.get('success_rate'))
    if report['accepted']:
        genome.save(target/'good-genome.json');atomic_json(target/'good.json',report)
        entry.update(status='good',accepted=str(target/'good.json'))
    else:entry['status']='queued'
    atomic_json(output/'status.json',state);print('AUDIT',name,generation,first['mean'],first.get('success_rate'),'accepted',report['accepted'],flush=True)


def command_for(directory,target,workers):
    saved=json.loads((directory/'evolution.json').read_text());s=saved['settings'];c=saved['config']
    command=[sys.executable,'-m','gym_adapter.evolve','--output',str(directory),'--resume',str(directory/'evolution.json'),
             '--generations',str(target),'--workers',str(workers),'--skip-final-evaluation']
    options={'env':c['env'],'ticks':c['ticks'],'smoothing':c['smoothing'],'max_steps':c['max_steps'],**s}
    for key,value in options.items():command+=['--'+key.replace('_','-'),str(value)]
    if c['observation_scale'] is not None:command+=['--obs-scale',','.join(map(str,c['observation_scale']))]
    return command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='runs/gym-until-good');p.add_argument('--chunk',type=int,default=250)
    p.add_argument('--audit-interval',type=int,default=1000);p.add_argument('--audit-episodes',type=int,default=100)
    p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if min(args.chunk,args.audit_interval,args.audit_episodes,args.workers)<1:raise ValueError('Invalid interval/count')
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    directories={name:ROOT/'runs/gym-suite-corrected'/name for name in GOALS if name!='doorkey'}
    directories['doorkey']=ROOT/'runs/gym-doorkey'
    plan={'generation_cap':None,'chunk':args.chunk,'audit_interval':args.audit_interval,'audit_episodes_per_round':args.audit_episodes,
          'confirmation':'Same training-selected genome must pass two disjoint fresh audit sets. Audits only decide stopping, never breeding.',
          'navigation':'Already good and excluded','goals':GOALS,'directories':{k:str(v) for k,v in directories.items()}}
    plan_path=output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Supervisor plan differs; restore existing settings')
    atomic_json(plan_path,plan)
    status=output/'status.json'
    if status.exists():state=json.loads(status.read_text())
    else:state={'audit_seed_cursor':0,'experiments':{name:{'status':'queued','last_audit_generation':None} for name in directories}}
    atomic_json(status,state)
    while True:
        remaining=False
        for name,directory in directories.items():
            entry=state['experiments'][name]
            if entry['status'] in ['good','failed']:continue
            remaining=True
            try:
                saved=json.loads((directory/'evolution.json').read_text());generation=saved['next_generation']
                entry['generation']=generation
                if entry['last_audit_generation'] is None or generation-entry['last_audit_generation']>=args.audit_interval:
                    entry['status']='auditing';atomic_json(status,state)
                    audit(name,directory,output,state,args.audit_episodes)
                    if entry['status']=='good':continue
                target=generation+args.chunk;command=command_for(directory,target,args.workers)
                entry.update(status='running',chunk_target=target,training_log=str(directory/'training.log'));atomic_json(status,state)
                atomic_json(output/name/'last-command.json',command);print('TRAIN',name,generation,'->',target,flush=True)
                with (directory/'training.log').open('a') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                entry.update(status='queued',generation=json.loads((directory/'evolution.json').read_text())['next_generation']);atomic_json(status,state)
            except Exception as error:
                entry.update(status='failed',error=str(error));atomic_json(status,state);print('FAILED',name,str(error),flush=True)
        if not remaining:break
    print('GOAL SEARCH FINISHED',json.dumps(state['experiments']),flush=True)

if __name__=='__main__':main()
