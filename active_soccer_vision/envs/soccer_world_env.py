import cv2
import gym
import random
import numpy as np
from gym import spaces

from active_soccer_vision.sim.sim import SoccerWorldSim

class SoccerWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        
        action_ones = np.ones((2,))
        self.action_space = spaces.Box(-action_ones, action_ones, dtype=np.float32)

        self.observation_space = spaces.Box(0, 1, (7,), dtype=np.float)

        self.sim = SoccerWorldSim(

        )

        self.counter = 0

    def step(self, action):

        observation = self.sim.step(action)

        reward = self.sim.get_reward()

        #self.render()
        done = self.counter > 2000
        self.counter += 1
        info = {}
        return observation, reward, done, info

    def reset(self):
        self.sim = SoccerWorldSim()
        return np.zeros((7,), dtype=np.float32)

    def render(self, mode='human'):
        viz = self.sim.render()

        # SHow the image
        cv2.imshow("Dist", viz)
        cv2.waitKey(1)

        return viz
    
    def close (self):
        pass