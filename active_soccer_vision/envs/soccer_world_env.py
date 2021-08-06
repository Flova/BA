import time
import cv2
import gym
import math
import random
import numpy as np
from gym import spaces

from active_soccer_vision.sim.sim import SoccerWorldSim

class SoccerWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(np.array([-1,-1]), np.array([1,1]), dtype=np.float32)

        self.observation_space = spaces.Box(0, 1, (8,), dtype=np.float)

        self._sim_length = 2000

        self.sim = SoccerWorldSim(

        )

        self.counter = 0

    def step(self, action):

        observation = self.sim.step(action)

        reward = self._get_reward()

        #self.render()
        done = self.counter > self._sim_length
        self.counter += 1
        info = {}
        return observation, reward, done, info

    def _get_reward(self):
        # Calculate reward
        reward = 1
        # Reward for visibility
        if self.sim.camera.check_if_point_is_visible(self.sim.ball_position):
            reward += 2
        # Reward for looking around
        reward -= (self.sim.camera.get_pan(normalize=True) - (math.sin(self.sim._sim_step * math.pi * 0.2 * self.sim.time_delta) + 1) * 0.5 ) ** 2
        return reward

    def reset(self):
        self.counter = 0
        self.sim = SoccerWorldSim()
        return np.zeros((8,), dtype=np.float32)

    def render(self, mode='human'):
        if self.counter != 0: 
            viz = self.sim.render()

            time.sleep(self.sim.time_delta)

            # SHow the image
            cv2.imshow("Dist", viz)
            cv2.waitKey(1)

            return viz
    
    def close (self):
        pass