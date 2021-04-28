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

        self.action_space = spaces.Box(-1, 1, (2,))

        self.resolution = 100
        self.field_size = (9, 6)

        #self.observation_space = spaces.Box(low=0, high=255,
        #                                    shape=(self.resolution * self.field_size[1], self.resolution * self.field_size[0], 3), dtype=np.uint8)
        self.action_space = spaces.Box(-1, 1, (12,))

        self.seed = 42
        self.time_delta = 1/10

        self.ball_position_generator = ball_position_gen( 
            time_delta=self.time_delta, 
            ball_init_position=(
                random.uniform(0, self.field_size[0]), 
                random.uniform(1, self.field_size[1])))
        self.ball_position = None

        self.robot_position_generator = robot_position_gen(
            time_delta=self.time_delta, 
            robot_init_position=(
                random.uniform(0, self.field_size[0]), 
                random.uniform(1, self.field_size[1])))
        self.robot_pose = None
        
        self.camera = Camera(fov=math.radians(45), width=1920, height=1080)

    def step(self, action):
        # Init observation
        observation = np.zeros((12,))

        # Generate ball and robot pose
        self.ball_position, _ = self.ball_position_generator.__next__()
        self.robot_pose, _ = self.robot_position_generator.__next__()


        # Apply action to the camera
        _,p,y = transforms3d.euler.mat2euler(
            transforms3d.affines.decompose(self.camera.camera_frame)[1])
        R = transforms3d.euler.euler2mat(math.pi/2, (action[1] + p)%math.tau, (action[0] + y)%math.tau)
        self.camera.camera_frame = transforms3d.affines.compose(np.array([self.robot_pose[0], self.robot_pose[1], 1.0]), R, np.ones(3))

        reward = 0
        done = False
        info = {"Lol": "no"}
        return observation, reward, done, info

    def reset(self):
        observation = np.zeros((12,))
        return observation

    def render(self, mode='human'):
        # Create canvas
        canvas = np.zeros((self.resolution * self.field_size[1], self.resolution * self.field_size[0], 3), dtype=np.uint8)

        # Draw camera wit fov indicator
        _,_,yaw = transforms3d.euler.mat2euler(
            transforms3d.affines.decompose(self.camera.camera_frame)[1])
        robot_in_image = (self.robot_pose * self.resolution).astype(np.int)
        fov = self.camera.fov
        robot_in_image_heading_vector = robot_in_image + (np.array([math.cos(yaw), math.sin(yaw)]) * self.resolution).astype(np.int)
        robot_in_image_heading_min_vector = robot_in_image + (np.array([math.cos(yaw - fov/2), math.sin(yaw - fov/2)]) * self.resolution).astype(np.int)
        robot_in_image_heading_max_vector = robot_in_image + (np.array([math.cos(yaw + fov/2), math.sin(yaw + fov/2)]) * self.resolution).astype(np.int)
        cv2.line(canvas, tuple(robot_in_image), tuple(robot_in_image_heading_min_vector), (255,255,255), 2)
        cv2.line(canvas, tuple(robot_in_image), tuple(robot_in_image_heading_max_vector), (255,255,255), 2)

        # Draw approximated visible field area
        corners = (self.camera.get_projected_image_corners() * self.resolution).astype(np.int32)
        cv2.polylines(canvas,[corners.reshape((-1,1,2))],True,(0,255,255), 5)

        render_ball_grid = False
        if render_ball_grid:
            # Simulate and check ball grid for testing purses
            for i in [x / 5.0 for x in range(0, 50)]:
                for u in [x / 5.0 for x in range(0, 45)]:
                    ball_position = np.array([i, u], dtype=np.float)
                    if self.camera.check_if_point_is_visible(ball_position):
                        cv2.circle(canvas, tuple([int(e * self.resolution) for e in ball_position]), 5, (0,255,0), -1)
                    else:
                        cv2.circle(canvas, tuple([int(e * self.resolution) for e in ball_position]), 5, (0,0,255), -1)
        
        # Check if the ball is vivible
        if self.camera.check_if_point_is_visible(self.ball_position):
            cv2.circle(canvas, tuple([int(e * self.resolution) for e in self.ball_position]), 10, (0,255,0), -1)
        else:
            cv2.circle(canvas, tuple([int(e * self.resolution) for e in self.ball_position]), 10, (0,0,255), -1)
            pass

        # SHow the image
        cv2.imshow("Dist", canvas)
        cv2.waitKey(1)
    
    def close (self):
        pass