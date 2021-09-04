import time
import math
import transforms3d
import cv2
import random
import numpy as np
from gym import spaces
from scipy.stats import multivariate_normal

from active_soccer_vision.sim.ball import ball_position_gen, Ball
from active_soccer_vision.sim.robot import robot_position_gen, robot_orientation_gen, Robot
from active_soccer_vision.sim.camera import Camera

class SoccerWorldSim:

    def __init__(self):
        super().__init__()

        self.resolution = 100
        self.field_size = (9, 6)

        self.time_delta = 1/10

        self.num_robots = 4

        ball_position_generator = ball_position_gen(
            time_delta=self.time_delta,
            ball_init_position=(
                random.uniform(0, self.field_size[0]),
                random.uniform(1, self.field_size[1])))

        self.ball = Ball(ball_position_generator, self.time_delta)

        self.robots = []
        for i in range(self.num_robots):
            robot_position_generator = robot_position_gen(
                time_delta=self.time_delta,
                robot_init_position=(
                    random.uniform(0, self.field_size[0]),
                    random.uniform(1, self.field_size[1])))

            robot_orientation_generator = robot_orientation_gen(
                time_delta=self.time_delta,
            )

            self.robots.append(Robot(robot_position_generator, robot_orientation_generator, self.time_delta))

        self.my_robot = self.robots[0]
        self.other_robots = self.robots[1:]

        self.camera = Camera(fov=math.radians(45), width=1920, height=1080)

        self._last_pan = 0.5
        self._last_tilt = 0.5
        self._sim_step = 0

        self.action_mode = "Velocity"  # Pattern, Velocity, Position

        self.observation_config = {
            "base_position": True,
            "base_heading": False,
            "camera_position": False,
            "neck_joint_position": True,
            "neck_joint_position_history": True,
            "sin_phase": True,
            "action_history": False,
            "estimated_ball_state": True,
            "estimated_robot_states": True,
        }

    def step(self, action):

        # Scalse actions to 0-1
        action = (action + 1) / 2

        # Generate ball and robot pose
        self.ball.step()
        [bot.step() for bot in self.robots]

        # Set new parent frame in camera
        self.camera.set_parent_frame(self.my_robot.get_base_footprint())

        if self.action_mode == "Pattern":
            self.camera.set_pan(
                min(1,
                    max(0,
                        (math.sin(self._sim_step * math.pi * 0.5 * self.time_delta) + 1) * 0.5 * action[0] + (action[1] - 0.5))),
                normalized=True)
            self.camera.set_tilt(0.3, normalized=True)
        elif self.action_mode == "Position":
            self.camera.set_pan(action[0], normalized=True)
            self.camera.set_tilt(action[1], normalized=True)
        elif self.action_mode == "Velocity":
            self.camera.set_pan(self.camera.get_pan(normalize=True) + (action[0] - 0.5) * self.time_delta, normalized=True)
            self.camera.set_tilt(0.3) # self.camera.get_tilt(normalize=True) + (action[0] - 0.5) * self.time_delta, normalized=True)
        else:
            print("Unknown action mode")

        # Check if we are ably to observe the Ball
        if self.camera.check_if_point_is_visible(self.ball.get_2d_position()):
            self.ball.observe()

        # Check if we are ably to observe any robots
        for robot in self.other_robots:
            if self.camera.check_if_point_is_visible(robot.get_2d_position()):
                robot.observe()

        # Build observation
        observation = []

        # Base position
        if self.observation_config["base_position"]:
            observation += [
                self.my_robot.get_2d_position()[0]/self.field_size[0], # Base footprint position x
                self.my_robot.get_2d_position()[1]/self.field_size[1], # Base footprint position y
            ]

        # Base heading
        if self.observation_config["base_heading"]:
            observation += [
                (math.sin(self.my_robot.get_heading()) + 1)/2,  # Base footprint heading part 1
                (math.cos(self.my_robot.get_heading()) + 1)/2,  # Base footprint heading part 2
            ]

        # Camera position
        if self.observation_config["camera_position"]:
            observation += [
                self.camera.get_2d_position()[0]/self.field_size[0], # Camera position x
                self.camera.get_2d_position()[1]/self.field_size[1], # Camera position y
            ]

        # Neck state
        if self.observation_config["neck_joint_position"]:
            observation += [
                self.camera.get_pan(normalize=True),  # Current Camera Pan
                #self.camera.get_tilt(normalize=True),  # Current Camera Tilt
            ]
        if self.observation_config["neck_joint_position_history"]:
            observation += [
                self._last_pan,
                #self._last_tilt,
            ]

        # Phase
        if self.observation_config["sin_phase"]:
            observation += [
                (math.sin(self._sim_step * math.pi * 0.2 * self.time_delta) + 1) * 0.5,
            ]

        # Action history
        if self.observation_config["action_history"]:
            observation += [
                (action[0] + 1)/2,
                (action[1] + 1)/2,
            ]

        # Ball world model
        if self.observation_config["estimated_ball_state"]:
            observation += [
                self.ball.get_last_observed_2d_position()[0][0]/self.field_size[0],   # Observed ball x
                self.ball.get_last_observed_2d_position()[0][1]/self.field_size[1],   # Observed ball y
                self.ball.get_last_observed_2d_position()[1],   # Observed ball confidence
            ]

        # Robots world model
        if self.observation_config["estimated_robot_states"]:
            for robot in self.other_robots:
                observation += [
                    robot.get_last_observed_2d_position()[0][0]/self.field_size[0], # Observed x
                    robot.get_last_observed_2d_position()[0][1]/self.field_size[1], # Observed x
                    robot.get_last_observed_2d_position()[1],  # Confidence
                ]

        self._last_pan = self.camera.get_pan(normalize=True)  # Current Camera Pan
        self._last_tilt = self.camera.get_tilt(normalize=True)  # Current Camera Tilt

        self._sim_step += 1

        return np.array(observation, dtype=np.float32)

    def render(self, mode='human'):
        # Create canvas
        canvas = np.zeros((self.resolution * self.field_size[1], self.resolution * self.field_size[0], 3), dtype=np.uint8)

        # Draw camera wit fov indicator
        yaw = self.camera.get_heading()
        camera_on_canvas = (self.camera.get_2d_position() * self.resolution).astype(np.int)
        fov = self.camera.fov
        length = 0.5 #m
        camera_in_image_heading_min_vector = camera_on_canvas + (np.array([math.cos(yaw - fov/2), math.sin(yaw - fov/2)]) * length * self.resolution).astype(np.int)
        camera_in_image_heading_max_vector = camera_on_canvas + (np.array([math.cos(yaw + fov/2), math.sin(yaw + fov/2)]) * length * self.resolution).astype(np.int)
        cv2.line(canvas, tuple(camera_on_canvas), tuple(camera_in_image_heading_min_vector), (255,255,255), 2)
        cv2.line(canvas, tuple(camera_on_canvas), tuple(camera_in_image_heading_max_vector), (255,255,255), 2)

        # Draw robot poses
        def draw_robot(robot, length=0.5):
            robot_on_canvas = (robot.get_2d_position() * self.resolution).astype(np.int)  # Todo use different one for camere position
            robot_in_image_heading_vector = robot_on_canvas + (np.array([math.cos(robot.get_heading()), math.sin(robot.get_heading())]) * length * self.resolution).astype(np.int)
            if self.camera.check_if_point_is_visible(robot.get_2d_position()):
                color = (100, 255, 100)
            else:
                color = (100, 100, 255)
            if robot == self.my_robot:
                color = (255, 100, 100)
            cv2.arrowedLine(canvas, tuple(robot_on_canvas), tuple(robot_in_image_heading_vector), color, 2)

        # Draw other robots
        [draw_robot(robot) for robot in self.robots]

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
        if self.camera.check_if_point_is_visible(self.ball.get_2d_position()):
            cv2.circle(canvas, tuple([int(e * self.resolution) for e in self.ball.get_2d_position()]), 10, (0,255,0), -1)
        else:
            cv2.circle(canvas, tuple([int(e * self.resolution) for e in self.ball.get_2d_position()]), 10, (0,0,255), -1)

        return canvas
