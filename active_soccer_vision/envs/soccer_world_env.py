import gym
import numpy as np
from gym import spaces

class SoccerWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(100, 100, 4), dtype=np.uint8)

    def step(self, action):
        observation = np.zeros((100,100,4), dtype=np.uint8)
        reward = 0
        done = False
        info = {"Lol": "no"}
        return observation, reward, done, info

    def reset(self):
        observation = np.zeros((100,100,4), dtype=np.uint8)
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        print("So much CGI")
        pass
    
    def close (self):
        pass