"""Run the existing evolutionary trainer with the parity-checked GPU maze backend.

The existing selection, seeds, breeding, checkpoints, neural source hash and CPU
validation remain in evolve.py. This entry point replaces only episode evaluation
and the worker pool used to create the reference MiniGrid starting worlds.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
from .adapter import ROOT
from .evolve import atomic_json
from .maze_gpu import SUPPORTED
from .maze_evaluate import MazeResetPool,evaluate_maze_episodes


def main():
    arguments=sys.argv[1:]
    def option(name,default=None):return arguments[arguments.index(name)+1] if name in arguments else default
    if option('--backend')!='cuda' or option('--env')!=SUPPORTED:raise ValueError('This entry point only supports CUDA ObstructedMaze-Full-v1')
    output=Path(option('--output'));output.mkdir(parents=True,exist_ok=True)
    files=['gym_adapter/maze_gpu.cu','gym_adapter/maze_gpu.py','gym_adapter/maze_evaluate.py','gym_adapter/evolve_maze_gpu.py']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    manifest=output/'maze-backend.json'
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        if existing['source_hashes']!=hashes:raise ValueError('GPU maze backend changed since checkpoint; review and migrate explicitly')
    else:
        resume=option('--resume');saved=json.loads(Path(resume).read_text()) if resume else None
        atomic_json(manifest,{'backend':'exact-cuda-maze-v1','environment':SUPPORTED,'source_hashes':hashes,
            'started_from_generation':saved['next_generation'] if saved else 0,
            'parent_checkpoint_sha256':hashlib.sha256(Path(resume).read_bytes()).hexdigest() if resume else None,
            'neural_source_hash':saved['source_hash'] if saved else None,
            'semantics':'Same neural kernels, initial worlds from seeded reference MiniGrid, GPU transitions and partial sensory views, exact NumPy sensory lookup, unchanged host action RNG. CPU validation and stopping audits unchanged.'})
    from . import env_pool,episode_batch,evolve
    env_pool.EnvPool=MazeResetPool
    def measured(genomes,config,seeds,pool):
        timings={};start=time.perf_counter();results=evaluate_maze_episodes(genomes,config,seeds,pool,timings)
        row={'seconds':time.perf_counter()-start,'episodes':sum(map(len,seeds)),'world_steps':sum(r['steps'] for r in results),'stages':timings}
        with (output/'gpu-performance.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('GPU_MAZE '+json.dumps(row),flush=True)
        return results
    episode_batch.evaluate_episodes=measured
    print('Using parity-checked GPU maze transitions and observations; unchanged evolution and neural dynamics.',flush=True)
    evolve.main()


if __name__=='__main__':main()
