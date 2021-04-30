import time
import math

from active_soccer_vision.envs import SoccerWorldEnv
from stable_baselines3.common.env_checker import check_env

env = SoccerWorldEnv()
check_env(env)
env.reset()

for i in range(8000):
    print(env.step([0.0, 0.5]))
    env.render()
    time.sleep(env.sim.time_delta)