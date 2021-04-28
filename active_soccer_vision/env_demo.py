import time
import math

from active_soccer_vision.envs import SoccerWorldEnv
from stable_baselines3.common.env_checker import check_env

env = SoccerWorldEnv()
check_env(env)

for i in range(8000):
    print(env.step([-1.0,0.02]))
    env.render()
    time.sleep(env.time_delta)