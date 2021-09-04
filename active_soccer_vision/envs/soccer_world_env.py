import os
import yaml
import time
import cv2
import gym
import math
import random
import numpy as np
from gym import spaces

from active_soccer_vision.sim.sim import SoccerWorldSim

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class SoccerWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_file):
        super().__init__()

        with open(os.path.join(__location__, '../config', config_file)) as f:
            self.config = yaml.safe_load(f)

        self.action_space = spaces.Box(np.array([-1,-1]), np.array([1,1]), dtype=np.float32)

        if not self.config['rl']['observation']['maps']['observation_maps']:
            self.observation_space = spaces.Box(0, 1, (self.config['rl']['observation']['vec']['num'],), dtype=np.float32)
        else:
            self.observation_space = spaces.Dict({
                "vec": spaces.Box(0, 1, (self.config['rl']['observation']['vec']['num'],), dtype=np.float32),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.config['sim']['field_size'][0],
                    self.config['sim']['field_size'][1], 1), dtype=np.uint8)
            })

        self._sim_length = self.config['sim']['length']

        self.sim = SoccerWorldSim(self.config)

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
        if self.sim.camera.check_if_point_is_visible(self.sim.ball.get_2d_position()):
            reward += 2
        # Reward for looking around
        reward -= (self.sim.camera.get_pan(normalize=True) - (math.sin(self.sim._sim_step * math.pi * 0.2 * self.sim.time_delta) + 1) * 0.5 ) ** 2
        return reward

    def reset(self):
        self.counter = 0
        self.sim = SoccerWorldSim(self.config)

        if not self.config['rl']['observation']['maps']['observation_maps']:
            return np.zeros((self.config['rl']['observation']['vec']['num'],), dtype=np.float32)
        else:
            return {
                "vec": np.zeros((self.config['rl']['observation']['vec']['num'],), dtype=np.float32),
                "map": np.zeros((
                    self.config['sim']['field_size'][0],
                    self.config['sim']['field_size'][1], 1), dtype=np.uint8)
            }

    def render(self, mode='human'):
        if self.counter != 0:
            viz = self.sim.render()

            time.sleep(self.sim.time_delta)

            # SHow the image
            cv2.imshow("Top Down Viz", viz)
            cv2.waitKey(1)

            return viz

    def close (self):
        pass
