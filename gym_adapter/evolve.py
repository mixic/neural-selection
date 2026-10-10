"""Evolution of spiking genomes against Gymnasium episode return."""
from __future__ import annotations
import argparse
import csv
from contextlib import nullcontext
import hashlib
from dataclasses import asdict
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import gymnasium as gym
from .environments import make_env
import numpy as np
from .adapter import Genome, SpaceAdapter, SpikingPolicy, build_native, random_genome, mutate


def evaluate(job):
    genome, config, seeds = job
    env = make_env(config)
    adapter = SpaceAdapter(env.observation_space,env.action_space,config['observation_scale'])
    returns=[];motor_spikes=0;steps=0
    try:
        with SpikingPolicy(genome,adapter,ticks=config['ticks'],smoothing=config['smoothing']) as policy:
            for seed in seeds:
                obs,_=env.reset(seed=int(seed));policy.reset(int(seed))
                total=0.
                for _ in range(config['max_steps']):
                    action=policy.act(obs)
                    motor_spikes+=float(policy.last_spikes.sum());steps+=1
                    obs,reward,terminated,truncated,_=env.step(action)
                    total+=float(reward)
                    if terminated or truncated:break
                returns.append(total)
    finally:env.close()
    return {'mean':float(np.mean(returns)),'returns':returns,'motor_spikes':motor_spikes,'steps':steps}


def random_baseline(config,seeds):
    env=make_env(config);returns=[]
    try:
        for seed in seeds:
            obs,_=env.reset(seed=int(seed));env.action_space.seed(int(seed));total=0.
            for _ in range(config['max_steps']):
                _,reward,terminated,truncated,_=env.step(env.action_space.sample());total+=float(reward)
                if terminated or truncated:break
            returns.append(total)
    finally:env.close()
    return returns


SELECTION_PROTOCOL='fresh-training-only-v2'


def fresh_training_seeds(run_seed, cursor, count):
    # Permute a counter bijectively in the high half of uint32. Every training
    # episode gets a unique seed, including across resumes, with no growing set.
    # Fixed validation/test seeds occupy the low half instead.
    if cursor<0 or cursor+count>2**31:raise ValueError('Training seed sequence exhausted')
    seeds=[]
    mask=2**31-1
    for counter in range(cursor,cursor+count):
        x=(counter+int(run_seed)*104729)&mask
        x^=x>>16;x=(x*0x7feb352d)&mask
        x^=x>>15;x=(x*0x846ca68b)&mask
        x^=x>>16
        seeds.append(x+2**31)
    return seeds


def breed(population, training_results, rng, mutation_rate):
    # This function receives no validation results or validation-selected archive.
    rank=np.argsort([-r['mean'] for r in training_results])
    offspring=[population[int(i)].clone() for i in rank[:max(2,len(population)//8)]]
    while len(offspring)<len(population):
        contestants=rng.choice(len(population),size=3,replace=False)
        parent=population[int(max(contestants,key=lambda i:training_results[int(i)]['mean']))]
        offspring.append(mutate(parent,rng,mutation_rate))
    return offspring,rank


def genome_dict(genome):
    # JSON serializes synchronously before the population is mutated again.
    # Avoid recursive dataclass copies of hundreds of thousands of scalar genes.
    return {'neurons':genome.neurons,'synapses':genome.synapses,
            'motors':genome.motors,'channels':genome.channels}


def atomic_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,separators=(',',':'))+'\n');temp.replace(path)


def train(args):
    if args.population<4 or args.generations<1 or args.episodes<1 or args.validation_episodes<1 or args.test_episodes<1 or args.workers<1 or args.max_steps<1 or args.hidden<0 or args.connections<0 or not 0<args.mutation_rate<=1 or not 1<=args.ticks<=10000 or not 0<args.smoothing<=1:raise ValueError('Invalid population, generations, episodes, workers or horizon')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'evolution.json').exists() and not args.resume:raise ValueError('Output already contains an experiment; use --resume or choose a new directory')
    build_native()
    if args.backend=='cuda':
        from .gpu import build_gpu
        build_gpu()
    config={'env':args.env,'ticks':args.ticks,'smoothing':args.smoothing,'observation_scale':args.obs_scale,'max_steps':args.max_steps}
    source_root=Path(__file__).resolve().parents[1]
    source_files=[source_root/'gym_adapter/native.cpp',source_root/'src/physiology.cuh',source_root/'src/lp_cell.cuh',source_root/'src/prinz_cell.cuh',source_root/'gym_adapter/adapter.py',source_root/'gym_adapter/core.cuh',source_root/'gym_adapter/environments.py',source_root/'gym_adapter/evolve.py',source_root/'gym_adapter/compile_graph.py',source_root/'gym_adapter/fast_observation.py']
    if args.backend=='cuda':source_files += [source_root/'gym_adapter/gpu.cu',source_root/'gym_adapter/gpu.py',source_root/'gym_adapter/batch_evaluate.py',source_root/'gym_adapter/env_pool.py',source_root/'gym_adapter/episode_batch.py']
    source_hash=hashlib.sha256(b''.join(p.read_bytes() for p in source_files)).hexdigest()
    settings={k:getattr(args,k) for k in ('seed','population','episodes','validation_episodes','test_episodes','hidden','connections','mutation_rate','test_seed','backend')}
    if not 0<=args.test_seed+args.seed*1000<2**31-args.test_episodes or not 0<=500000+args.seed*100<2**31-args.validation_episodes:raise ValueError('Validation and test seed ranges must be in the low half of uint32')
    rng=np.random.default_rng(args.seed)
    env=make_env(config);adapter=SpaceAdapter(env.observation_space,env.action_space,args.obs_scale);env.close()
    if args.resume:
        saved=json.loads(Path(args.resume).read_text())
        if saved.get('selection_protocol')!=SELECTION_PROTOCOL:raise ValueError('Checkpoint used the invalid validation-guided selection protocol; start a fresh population')
        if saved['config']!=config or saved.get('settings')!=settings:raise ValueError('Resume settings differ from checkpoint')
        if saved.get('source_hash')!=source_hash:raise ValueError('Neural model differs from checkpoint')
        population=[Genome(**g) for g in saved['population']]
        if len(population)!=args.population:raise ValueError('Resume population differs')
        rng.bit_generator.state=saved['rng']
        training_seed_cursor=saved['training_seed_cursor']
        if saved['next_generation']>args.generations:raise ValueError('Set --generations at or above the saved next generation')
        start=saved['next_generation'];champion=Genome(**saved['champion']) if saved['champion'] else None
        champion_score=saved['champion_score'];initial=Genome(**saved['initial']);initial_validation=saved['initial_validation']
    else:
        population=[random_genome(adapter.channels,adapter.outputs,rng,args.hidden,args.connections) for _ in range(args.population)]
        training_seed_cursor=0
        start=0;champion=None;champion_score=-float('inf');initial=None;initial_validation=None
    validation_seeds=list(range(500_000+args.seed*100,500_000+args.seed*100+args.validation_episodes))
    begin=time.monotonic();history=out/'history.csv'
    from .env_pool import EnvPool
    env_context=EnvPool(config,args.population*args.episodes,args.workers) if args.backend=='cuda' else nullcontext(None)
    with env_context as env_pool, ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool,history.open('a' if args.resume else 'w',newline='') as log:
        writer=csv.DictWriter(log,fieldnames=['generation','best_training','mean_training','best_validation','champion_validation','neurons','synapses','motor_spikes_per_step','seconds'])
        if not args.resume:writer.writeheader()
        for generation in range(start,args.generations):
            flat_seeds=fresh_training_seeds(args.seed,training_seed_cursor,args.population*args.episodes)
            training_seed_cursor+=len(flat_seeds)
            seeds=[flat_seeds[i*args.episodes:(i+1)*args.episodes] for i in range(args.population)]
            if args.backend=='cuda':
                from .episode_batch import evaluate_episodes
                results=evaluate_episodes(population,config,seeds,env_pool)
            else:results=list(pool.map(evaluate,[(g,config,s) for g,s in zip(population,seeds)]))
            next_population,rank=breed(population,results,rng,args.mutation_rate)
            best=population[int(rank[0])]
            # Report the current training winner; validation cannot retain a genome.
            champion=best.clone();champion.save(out/'champion.json')
            score=evaluate((best,config,validation_seeds))['mean']
            if initial is None:initial=best.clone();initial_validation=score;initial.save(out/'initial.json')
            champion_score=score # Diagnostic only: never used by breeding or stopping.
            result=results[int(rank[0])]
            row={'generation':generation,'best_training':result['mean'],'mean_training':float(np.mean([r['mean'] for r in results])),'best_validation':score,'champion_validation':champion_score,'neurons':len(best.neurons),'synapses':len(best.synapses),'motor_spikes_per_step':result['motor_spikes']/max(1,result['steps']),'seconds':time.monotonic()-begin}
            writer.writerow(row);log.flush();print(json.dumps(row),flush=True)
            population=next_population
            atomic_json(out/'evolution.json',{'format':2,'selection_protocol':SELECTION_PROTOCOL,'training_seed_cursor':training_seed_cursor,'config':config,'settings':settings,'source_hash':source_hash,'next_generation':generation+1,'population':[genome_dict(g) for g in population],'rng':rng.bit_generator.state,'champion':genome_dict(champion),'champion_score':champion_score,'initial':genome_dict(initial),'initial_validation':initial_validation})
            if args.target_return is not None and result['mean']>=args.target_return:break
    if args.skip_final_evaluation:
        print('CHECKPOINT '+str(out/'evolution.json'),flush=True)
        return
    generation=args.generations-1 if start==args.generations else generation
    holdout=list(range(args.test_seed+args.seed*1000,args.test_seed+args.seed*1000+args.test_episodes))
    first=evaluate((initial,config,holdout));final=evaluate((champion,config,holdout));random=random_baseline(config,holdout)
    summary={'selection_protocol':SELECTION_PROTOCOL,'champion_selection':'winner of last generation by fresh training return; validation is reporting only','config':config,'settings':settings,'source_hash':source_hash,'seed':args.seed,'population':args.population,'completed_generations':generation+1,'validation_champion':champion_score,'initial_holdout':first,'champion_holdout':final,'random_holdout':{'mean':float(np.mean(random)),'returns':random},'gymnasium_version':gym.__version__,'numpy_version':np.__version__,'backend':args.backend+'; shared src/physiology.cuh; no hidden/motor current bias','heldout_seeds':holdout}
    if args.env.startswith('MiniGrid-'):
        summary['success_rate']=float(np.mean(np.array(final['returns'])>0))
        summary['initial_success_rate']=float(np.mean(np.array(first['returns'])>0))
        summary['random_success_rate']=float(np.mean(np.array(random)>0))
    summary['final_evaluation_backend']='cpu'
    atomic_json(out/'results.json',summary)
    print('FINAL '+json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env',default='CartPole-v1');p.add_argument('--population',type=int,default=64)
    p.add_argument('--generations',type=int,default=80);p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--validation-episodes',type=int,default=8);p.add_argument('--test-episodes',type=int,default=50)
    p.add_argument('--max-steps',type=int,default=500);p.add_argument('--hidden',type=int,default=8)
    p.add_argument('--connections',type=int,default=64);p.add_argument('--ticks',type=int,default=16)
    p.add_argument('--smoothing',type=float,default=.4);p.add_argument('--mutation-rate',type=float,default=.15)
    p.add_argument('--obs-scale',type=lambda x:[float(v) for v in x.split(',')]);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--workers',type=int,default=min(8,os.cpu_count() or 1));p.add_argument('--output',default='runs/gym-cartpole')
    p.add_argument('--test-seed',type=int,default=1_000_000,help='Independent final test seed range base')
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--skip-final-evaluation',action='store_true',help='Save the population without final test evaluation (for chunked training)')
    p.add_argument('--resume');p.add_argument('--target-return',type=float,help='Optional stopping threshold on fresh TRAINING return only')
    args=p.parse_args();train(args)

if __name__=='__main__':main()
