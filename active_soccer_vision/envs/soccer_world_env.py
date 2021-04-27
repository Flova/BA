import cv2
import gym
import random
import numpy as np
from gym import spaces
from scipy.stats import multivariate_normal

from active_soccer_vision.sim.ball import ball_position_gen

class SoccerWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)

        self.resolution = 100
        self.field_size = (9, 6)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.resolution * self.field_size[1], self.resolution * self.field_size[0], 3), dtype=np.uint8)

        self.seed = 42
        self.time_delta = 1/10

        self.ball_position_generator = ball_position_gen( 
            time_delta=self.time_delta, 
            ball_init_position=(
                random.uniform(0, self.field_size[0]), 
                random.uniform(1, self.field_size[1])))

    def step(self, action):
        observation = np.zeros((self.resolution * self.field_size[1], self.resolution * self.field_size[0], 3), dtype=np.uint8)

        self.prop_dist = observation.copy()

        ball_position, ball_ground_truth = self.ball_position_generator.__next__()

        cv2.circle(self.prop_dist, tuple([int(e * self.resolution) for e in ball_position]), 10, (0,0,255), -1)

        reward = 0
        done = False
        info = {"Lol": "no"}
        return observation, reward, done, info

    def reset(self):
        observation = np.zeros((self.resolution * self.field_size[1], self.resolution * self.field_size[0], 3), dtype=np.uint8)
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        cv2.imshow("Dist", self.prop_dist)
        cv2.waitKey(1)
    
    def close (self):
        pass