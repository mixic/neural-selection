"""Run a bounded exploratory environment suite with resumable experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from .adapter import ROOT

SPECS=[
    {'name':'minigrid-empty-random','env':'MiniGrid-Empty-Random-6x6-v0','backend':'cuda','population':64,'generations':120,'episodes':4,'validation_episodes':12,'test_episodes':100,'max_steps':144,'hidden':8,'connections':96,'seed':21},
    {'name':'pendulum','env':'Pendulum-v1','backend':'cpu','population':64,'generations':160,'episodes':3,'validation_episodes':8,'test_episodes':50,'max_steps':200,'hidden':8,'connections':48,'seed':22},
    {'name':'acrobot','env':'Acrobot-v1','backend':'cpu','population':64,'generations':140,'episodes':3,'validation_episodes':8,'test_episodes':50,'max_steps':500,'hidden':8,'connections':64,'seed':23},
    {'name':'mountaincar','env':'MountainCar-v0','backend':'cpu','population':64,'generations':100,'episodes':3,'validation_episodes':8,'test_episodes':50,'max_steps':200,'hidden':8,'connections':48,'seed':24},
    {'name':'minigrid-memory','env':'MiniGrid-MemoryS7-v0','backend':'cuda','population':64,'generations':120,'episodes':4,'validation_episodes':12,'test_episodes':100,'max_steps':196,'hidden':16,'connections':160,'seed':25},
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='runs/gym-suite');p.add_argument('--workers',type=int,default=8)
    p.add_argument('--test-seed',type=int,default=4000000,help='Final test seed range base')
    p.add_argument('--generations',type=int,help='Total generation target for every environment')
    p.add_argument('--generation-chunk',type=int,default=0,help='Rotate environments after this many generations; 0 runs each to its target')
    args=p.parse_args()
    if args.generation_chunk<0 or (args.generations is not None and args.generations<1):raise ValueError('Invalid generation target/chunk')
    specs=[dict(spec) for spec in SPECS]
    if args.generations is not None:
        for spec in specs:spec['generations']=args.generations
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'plan.json').write_text(json.dumps(specs,indent=2)+'\n')
    report={};started=time.time();failed=set()
    def save_status():
        from .evolve import atomic_json
        atomic_json(output/'status.json',{'started':started,'updated':time.time(),'experiments':report})
    while True:
        pending=False
        for spec in specs:
            name=spec['name'];directory=output/name;directory.mkdir(exist_ok=True)
            if name in failed:continue
            checkpoint=directory/'evolution.json';completed=0
            if checkpoint.exists():completed=json.loads(checkpoint.read_text())['next_generation']
            results=directory/'results.json'
            if results.exists() and json.loads(results.read_text())['completed_generations']>=spec['generations']:
                report[name]={'status':'complete','generation':completed,'target':spec['generations'],'results':str(results)};continue
            pending=True
            target=min(spec['generations'],completed+args.generation_chunk) if args.generation_chunk else spec['generations']
            command=[sys.executable,'-m','gym_adapter.evolve','--output',str(directory),'--workers',str(args.workers),'--test-seed',str(args.test_seed)]
            for key,value in spec.items():
                if key!='name':command+=['--'+key.replace('_','-'),str(target if key=='generations' else value)]
            if checkpoint.exists():command+=['--resume',str(checkpoint)]
            if target<spec['generations']:command+=['--skip-final-evaluation']
            (directory/'command.json').write_text(json.dumps(command,indent=2)+'\n')
            with (directory/'commands.jsonl').open('a') as commands:commands.write(json.dumps(command)+'\n')
            report[name]={'status':'running','generation':completed,'chunk_target':target,'target':spec['generations'],'log':str(directory/'training.log')};save_status()
            print(f'START {name} {completed} -> {target}',flush=True)
            with (directory/'training.log').open('a') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            completed=json.loads(checkpoint.read_text())['next_generation'] if checkpoint.exists() else 0
            report[name]={'status':'complete' if result.returncode==0 and completed>=spec['generations'] else ('queued' if result.returncode==0 else 'failed'),'returncode':result.returncode,'generation':completed,'target':spec['generations'],'results':str(results),'log':str(directory/'training.log')}
            if result.returncode:failed.add(name)
            save_status();print('DONE '+name+' '+json.dumps(report[name]),flush=True)
        if not pending:break
    save_status();print('SUITE FINISHED',flush=True)

if __name__=='__main__':main()
