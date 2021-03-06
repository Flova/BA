import time
import math
import numpy as np

from active_soccer_vision.envs import SoccerWorldEnv
from stable_baselines3.common.env_checker import check_env

env = SoccerWorldEnv("default.yaml")
check_env(env)
env.reset()

done = False
for i in range(9999999):
    obs = env.step(np.array([math.sin(i/4 -math.pi/2), math.sin(i/8 -math.pi/2)]))
    print(obs)
    if obs[2]: break
    env.render()
