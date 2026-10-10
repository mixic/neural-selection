"""Run independent episodes concurrently, preserving per-episode neural resets."""
from .batch_evaluate import evaluate_batch


def evaluate_episodes(genomes, config, seeds, env_pool):
    episodes=len(seeds[0])
    expanded=[g for g in genomes for _ in range(episodes)]
    expanded_seeds=[[seed] for row in seeds for seed in row]
    raw=evaluate_batch(expanded,config,expanded_seeds,env_pool)
    results=[]
    for i in range(len(genomes)):
        group=raw[i*episodes:(i+1)*episodes]
        returns=[r['returns'][0] for r in group]
        # Match the reference arithmetic, including NumPy mean order.
        import numpy as np
        results.append({'mean':float(np.mean(returns)), 'returns':returns,
                        'motor_spikes':sum(r['motor_spikes'] for r in group),
                        'steps':sum(r['steps'] for r in group)})
    return results
