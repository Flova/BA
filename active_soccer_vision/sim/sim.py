import time
import math
import transforms3d
import cv2
import random
import numpy as np
from gym import spaces
from scipy.stats import multivariate_normal

from active_soccer_vision.sim.ball import ball_position_gen
from active_soccer_vision.sim.robot import robot_position_gen, robot_orientation_gen, Robot
from active_soccer_vision.sim.camera import Camera

class SoccerWorldSim:

    def __init__(self):
        super().__init__()

        self.resolution = 100
        self.field_size = (9, 6)

        self.time_delta = 1/10

        self.ball_position_generator = ball_position_gen( 
            time_delta=self.time_delta, 
            ball_init_position=(
                random.uniform(0, self.field_size[0]), 
                random.uniform(1, self.field_size[1])))
        self.ball_position = None
        self._last_observed_ball_position = np.array([self.field_size[0]/2, self.field_size[1]/2])
        self._last_observed_ball_position_conf = 0.0

        robot_position_generator = robot_position_gen(
            time_delta=self.time_delta, 
            robot_init_position=(
                random.uniform(0, self.field_size[0]), 
                random.uniform(1, self.field_size[1])))

        robot_orientation_generator = robot_orientation_gen(
            time_delta=self.time_delta,
        )
        
        self.robot = Robot(robot_position_generator, robot_orientation_generator)

        self.camera = Camera(fov=math.radians(45), width=1920, height=1080)

        self._last_pan = 0.0
        self._last_tilt = 0.0  
        self._sim_step = 0

        self.action_mode = "Pattern"  # Velocity, Position

    def step(self, action):

        # Generate ball and robot pose
        self.ball_position, _ = self.ball_position_generator.__next__()

        self.robot.step()

        self.camera.set_parent_frame(self.robot.get_base_footprint())

        if self.action_mode == "Pattern":
            self.camera.set_pan(
                min(1, max(0, (math.sin(self._sim_step * math.pi * 0.05) * ((action[0] + 1) / 2)  + action[1] + 1) / 2)), normalized=True)
            self.camera.set_tilt(0.3, normalized=True)
        elif self.action_mode == "Position":
            self.camera.set_pan((action[0] + 1)/2, normalized=True)
            self.camera.set_tilt((action[1] + 1)/2, normalized=True)
        else:
            print("Unknown action mode")

        # Drop ball confidence
        self._last_observed_ball_position_conf = max(self._last_observed_ball_position_conf - 0.1 * self.time_delta, 0.0)

        # Calculate reward and ball observation
        if self.camera.check_if_point_is_visible(self.ball_position):
            self._last_observed_ball_position = self.ball_position
            self._last_observed_ball_position_conf = 1.0

        # Build observation
        observation = np.array([
            #self.robot.get_2d_position()[0]/self.field_size[0], # Base footprint position x
            #self.robot.get_2d_position()[1]/self.field_size[1], # Base footprint position y
            #(math.sin(self.robot.get_heading()) + 1)/2,  # Base footprint heading part 1
            #(math.cos(self.robot.get_heading()) + 1)/2,  # Base footprint heading part 2
            self.camera.get_2d_position()[0]/self.field_size[0], # Camera position x
            self.camera.get_2d_position()[1]/self.field_size[1], # Camera position y
            self.camera.get_pan(normalize=True),  # Current Camera Pan
            self.camera.get_tilt(normalize=True),  # Current Camera Tilt
            (action[0] + 1)/2,
            (action[1] + 1)/2,
            self._last_observed_ball_position[0]/self.field_size[0],   # Observed ball x
            self._last_observed_ball_position[1]/self.field_size[1],   # Observed ball y
            self._last_observed_ball_position_conf,   # Observed ball confidence
        ], dtype=np.float32)

        self._last_pan = self.camera.get_pan(normalize=True),  # Current Camera Pan
        self._last_tilt = self.camera.get_tilt(normalize=True),  # Current Camera Tilt

        self._sim_step += 1

        return observation

    def render(self, mode='human'):
        # Create canvas
        canvas = np.zeros((self.resolution * self.field_size[1], self.resolution * self.field_size[0], 3), dtype=np.uint8)

        # Draw camera wit fov indicator
        yaw = self.camera.get_heading()
        camera_on_canvas = (self.camera.get_2d_position() * self.resolution).astype(np.int)  
        fov = self.camera.fov
        camera_in_image_heading_vector = camera_on_canvas + (np.array([math.cos(yaw), math.sin(yaw)]) * self.resolution).astype(np.int)
        camera_in_image_heading_min_vector = camera_on_canvas + (np.array([math.cos(yaw - fov/2), math.sin(yaw - fov/2)]) * self.resolution).astype(np.int)
        camera_in_image_heading_max_vector = camera_on_canvas + (np.array([math.cos(yaw + fov/2), math.sin(yaw + fov/2)]) * self.resolution).astype(np.int)
        cv2.line(canvas, tuple(camera_on_canvas), tuple(camera_in_image_heading_min_vector), (255,255,255), 2)
        cv2.line(canvas, tuple(camera_on_canvas), tuple(camera_in_image_heading_max_vector), (255,255,255), 2)

        # Draw robot pose
        robot_on_canvas = (self.robot.get_2d_position() * self.resolution).astype(np.int)  # Todo use different one for camere position
        robot_in_image_heading_vector = robot_on_canvas + (np.array([math.cos(self.robot.get_heading()), math.sin(self.robot.get_heading())]) * self.resolution).astype(np.int)
        cv2.arrowedLine(canvas, tuple(robot_on_canvas), tuple(robot_in_image_heading_vector), (255,100,200), 2)

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
        
        # Check if the ball is visable
        if self.camera.check_if_point_is_visible(self.ball_position):
            cv2.circle(canvas, tuple([int(e * self.resolution) for e in self.ball_position]), 10, (0,255,0), -1)
        else:
            cv2.circle(canvas, tuple([int(e * self.resolution) for e in self.ball_position]), 10, (0,0,255), -1)

        return canvas