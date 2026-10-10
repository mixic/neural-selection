"""Replay a saved genetic policy; optionally write a GIF of its Gymnasium rollout."""
import argparse
import json
from pathlib import Path
import gymnasium as gym
from .environments import make_env
from .adapter import Genome, SpaceAdapter, SpikingPolicy


def rollout(genome_path, config, seed=1, gif=None):
    env=make_env(config,render_mode='rgb_array' if gif else None)
    frames=[];total=0.;steps=0
    try:
        adapter=SpaceAdapter(env.observation_space,env.action_space,config['observation_scale'])
        with SpikingPolicy(Genome.load(genome_path),adapter,ticks=config['ticks'],smoothing=config['smoothing']) as policy:
            obs,_=env.reset(seed=seed);policy.reset(seed)
            if gif:frames.append(env.render())
            for _ in range(config['max_steps']):
                obs,reward,terminated,truncated,_=env.step(policy.act(obs));total+=float(reward);steps+=1
                if gif:frames.append(env.render())
                if terminated or truncated:break
        if gif:
            from PIL import Image
            output=Path(gif);output.parent.mkdir(parents=True,exist_ok=True)
            images=[Image.fromarray(frame) for frame in frames[::2]]
            images[0].save(output,save_all=True,append_images=images[1:],duration=round(2000/env.metadata.get('render_fps',50)),loop=0)
    finally:env.close()
    return {'seed':seed,'return':total,'steps':steps}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('genome');p.add_argument('--results',required=True,help='results.json from the evolution run')
    p.add_argument('--seed',type=int,default=1);p.add_argument('--gif')
    args=p.parse_args();config=json.loads(Path(args.results).read_text())['config']
    print(json.dumps(rollout(args.genome,config,args.seed,args.gif)))

if __name__=='__main__':main()
