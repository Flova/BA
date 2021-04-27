import math
import transforms3d
import cv2
import gym
import random
import numpy as np
from gym import spaces
from scipy.stats import multivariate_normal

from active_soccer_vision.sim.ball import ball_position_gen
from active_soccer_vision.sim.robot import robot_position_gen
from active_soccer_vision.sim.camera import Camera

class SoccerWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(4)

        self.resolution = 100
        self.field_size = (9, 6)

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.resolution * self.field_size[1], self.resolution * self.field_size[0], 3), dtype=np.uint8)

        self.seed = 42
        self.time_delta = 1/10

        self.ball_position_generator = ball_position_gen( 
            time_delta=self.time_delta, 
            ball_init_position=(
                random.uniform(0, self.field_size[0]), 
                random.uniform(1, self.field_size[1])))

        self.robot_position_generator = robot_position_gen(
            time_delta=self.time_delta, 
            robot_init_position=(
                random.uniform(0, self.field_size[0]), 
                random.uniform(1, self.field_size[1])))
        self.robot_pose = None
        
        self.camera = Camera(fov=math.radians(45), width=1920, height=1080)

    def step(self, action):
        observation = np.zeros((self.resolution * self.field_size[1], self.resolution * self.field_size[0], 3), dtype=np.uint8)

        self.prop_dist = observation.copy()

        ball_position, ball_ground_truth = self.ball_position_generator.__next__()
        self.robot_pose, _ = self.robot_position_generator.__next__()

        if self.camera.check_if_point_is_visible(ball_position):
            cv2.circle(self.prop_dist, tuple([int(e * self.resolution) for e in ball_position]), 10, (0,255,0), -1)
        else:
            cv2.circle(self.prop_dist, tuple([int(e * self.resolution) for e in ball_position]), 10, (0,0,255), -1)

        """
        for i in [x / 10.0 for x in range(0, 100)]:
            for u in [x / 10.0 for x in range(0, 90)]:
                ball_position = np.array([i, u], dtype=np.float)
                print(ball_position)
                if self.camera.check_if_point_is_visible(ball_position):
                    cv2.circle(self.prop_dist, tuple([int(e * self.resolution) for e in ball_position]), 5, (0,255,0), -1)
                else:
                    print("drawing circle")
                    cv2.circle(self.prop_dist, tuple([int(e * self.resolution) for e in ball_position]), 5, (0,0,255), -1)
        

        T, R, Z, A = transforms3d.affines.decompose(self.camera.camera_frame)

        R = transforms3d.euler.euler2mat(0, math.radians(30), math.radians(self.p))

        self.camera.camera_frame = transforms3d.affines.compose(T, R, Z, A)

        self.p += 10

        if self.p > 360:
            self.p = 0

        print(self.p)

        """

        reward = 0
        done = False
        info = {"Lol": "no"}
        return observation, reward, done, info

    def reset(self):
        observation = np.zeros((self.resolution * self.field_size[1], self.resolution * self.field_size[0], 3), dtype=np.uint8)
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        yaw = 0.0
        robot_in_image = (self.robot_pose * self.resolution).astype(np.int)

        fov = self.camera.fov
        robot_in_image_heading_vector = robot_in_image + (np.array([math.cos(yaw), math.sin(yaw)]) * self.resolution).astype(np.int)
        robot_in_image_heading_min_vector = robot_in_image + (np.array([math.cos(yaw - fov/2), math.sin(yaw - fov/2)]) * self.resolution).astype(np.int)
        robot_in_image_heading_max_vector = robot_in_image + (np.array([math.cos(yaw + fov/2), math.sin(yaw + fov/2)]) * self.resolution).astype(np.int)
        cv2.line(self.prop_dist, tuple(robot_in_image), tuple(robot_in_image_heading_min_vector), (255,255,255), 2)
        cv2.line(self.prop_dist, tuple(robot_in_image), tuple(robot_in_image_heading_max_vector), (255,255,255), 2)

        corners = (self.camera.get_projected_image_corners() * self.resolution).astype(np.int32)
        corners = corners.reshape((-1,1,2))
        cv2.polylines(self.prop_dist,[corners],True,(0,255,255), 5)

        cv2.imshow("Dist", self.prop_dist)
        cv2.waitKey(1)
    
    def close (self):
        pass