"""Environment construction shared by evolution, replay and benchmarks."""
import gymnasium as gym
import numpy as np
import os


def make_env(config, **kwargs):
    name=config['env']
    if name.startswith('MiniGrid-'):
        import minigrid  # Registers Farama's environments.
        from minigrid.wrappers import ImgObsWrapper
        # Standard partial symbolic view. Fixed-task missions are excluded;
        # no global map, agent position, goal vector or reward enters the brain.
        env=ImgObsWrapper(gym.make(name,**kwargs))
        if os.environ.get('GYM_REFERENCE_OBSERVATION')!='1':
            from .fast_observation import SUPPORTED,install
            if name in SUPPORTED:install(env)
        high=np.broadcast_to(np.array([10,5,2],np.uint8),env.observation_space.shape).copy()
        env.observation_space=gym.spaces.Box(np.zeros_like(high),high,dtype=np.uint8)
        return env
    return gym.make(name,**kwargs)
