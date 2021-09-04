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

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.render_resolution = self.config['sim']['render_resolution']
        self.field_size = self.config['sim']['field_size']

        self.time_delta = self.config['sim']['time_delta']

        self.num_robots = self.config['misc']['num_robots']

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

        self.camera = Camera(fov=math.radians(45), width=1920, height=1080, robot=self.my_robot)

        self._last_pan = 0.5
        self._last_tilt = 0.5
        self._sim_step = 0

    def step(self, action):
        # Scalse actions to 0-1
        action = (action + 1) / 2

        # Generate ball and robot pose
        self.ball.step()
        [bot.step() for bot in self.robots]

        if self.config['rl']['action']['mode'] == "Pattern":
            self.camera.set_pan(
                min(1,
                    max(0,
                        (math.sin(self._sim_step * math.pi * 0.5 * self.time_delta) + 1) * 0.5 * action[0] + (action[1] - 0.5))),
                normalized=True)
            self.camera.set_tilt(0.3, normalized=True)
        elif  self.config['rl']['action']['mode'] == "Position":
            self.camera.set_pan(action[0], normalized=True)
            self.camera.set_tilt(action[1], normalized=True)
        elif  self.config['rl']['action']['mode'] == "Velocity":
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
        observation_vector = []

        observation_vector_config = self.config['rl']['observation']['vec']

        # Base position
        if observation_vector_config["base_position"]:
            observation_vector += [
                self.my_robot.get_2d_position()[0]/self.field_size[0], # Base footprint position x
                self.my_robot.get_2d_position()[1]/self.field_size[1], # Base footprint position y
            ]

        # Base heading
        if observation_vector_config["base_heading"]:
            observation_vector += [
                (math.sin(self.my_robot.get_heading()) + 1)/2,  # Base footprint heading part 1
                (math.cos(self.my_robot.get_heading()) + 1)/2,  # Base footprint heading part 2
            ]

        # Camera position
        if observation_vector_config["camera_position"]:
            observation_vector += [
                self.camera.get_2d_position()[0]/self.field_size[0], # Camera position x
                self.camera.get_2d_position()[1]/self.field_size[1], # Camera position y
            ]

        # Neck state
        if observation_vector_config["neck_joint_position"]:
            observation_vector += [
                self.camera.get_pan(normalize=True),  # Current Camera Pan
                #self.camera.get_tilt(normalize=True),  # Current Camera Tilt
            ]
        if observation_vector_config["neck_joint_position_history"]:
            observation_vector += [
                self._last_pan,
                #self._last_tilt,
            ]

        # Phase
        if observation_vector_config["sin_phase"]:
            observation_vector += [
                (math.sin(self._sim_step * math.pi * 0.2 * self.time_delta) + 1) * 0.5,
            ]

        # Action history
        if observation_vector_config["action_history"]:
            observation_vector += [
                (action[0] + 1)/2,
                (action[1] + 1)/2,
            ]

        # Ball world model
        if observation_vector_config["estimated_ball_state"]:
            observation_vector += [
                self.ball.get_last_observed_2d_position()[0][0]/self.field_size[0],   # Observed ball x
                self.ball.get_last_observed_2d_position()[0][1]/self.field_size[1],   # Observed ball y
                self.ball.get_last_observed_2d_position()[1],   # Observed ball confidence
            ]

        # Robots world model
        if observation_vector_config["estimated_robot_states"]:
            for robot in self.other_robots:
                observation_vector += [
                    robot.get_last_observed_2d_position()[0][0]/self.field_size[0], # Observed x
                    robot.get_last_observed_2d_position()[0][1]/self.field_size[1], # Observed x
                    robot.get_last_observed_2d_position()[1],  # Confidence
                ]

        # Render observation maps if necessary
        observation_maps = None
        observation_map_config = self.config['rl']['observation']['maps']
        if observation_map_config["observation_maps"]:
            observation_maps = np.zeros((self.field_size[0], self.field_size[1], 1), dtype=np.uint8)
            # Robots world model for the map
            if observation_map_config["estimated_robot_states_map"]:
                for robot in self.other_robots:
                    # Draw robot on map if the cell is not occupied by a
                    idx =(  int(min(self.field_size[0] - 1, robot.get_last_observed_2d_position()[0][0])),
                            int(min(self.field_size[1] - 1, robot.get_last_observed_2d_position()[0][1])))
                    if robot.get_last_observed_2d_position()[1] * 254 + 1 > observation_maps[idx]:
                        observation_maps[idx] = robot.get_last_observed_2d_position()[1] * 254 + 1

        #cv2.imshow("map", cv2.resize(cv2.flip(cv2.rotate(observation_maps, cv2.ROTATE_90_CLOCKWISE), 1), (909, 600)))

        self._last_pan = self.camera.get_pan(normalize=True)  # Current Camera Pan
        self._last_tilt = self.camera.get_tilt(normalize=True)  # Current Camera Tilt

        self._sim_step += 1

        # Check if we have observation maps
        if observation_maps is None:
            return np.array(observation_vector, dtype=np.float32)
        else:
            return {
                "vec": np.array(observation_vector, dtype=np.float32),
                "map": observation_maps,
            }

    def render(self, mode='human'):
        # Create canvas
        canvas = np.zeros((self.render_resolution * self.field_size[1], self.render_resolution * self.field_size[0], 3), dtype=np.uint8)

        # Draw camera wit fov indicator
        yaw = self.camera.get_heading()
        camera_on_canvas = (self.camera.get_2d_position() * self.render_resolution).astype(np.int)
        fov = self.camera.fov
        length = 0.5 #m
        camera_in_image_heading_min_vector = camera_on_canvas + (np.array([math.cos(yaw - fov/2), math.sin(yaw - fov/2)]) * length * self.render_resolution).astype(np.int)
        camera_in_image_heading_max_vector = camera_on_canvas + (np.array([math.cos(yaw + fov/2), math.sin(yaw + fov/2)]) * length * self.render_resolution).astype(np.int)
        cv2.line(canvas, tuple(camera_on_canvas), tuple(camera_in_image_heading_min_vector), (255,255,255), 2)
        cv2.line(canvas, tuple(camera_on_canvas), tuple(camera_in_image_heading_max_vector), (255,255,255), 2)

        # Draw robot poses
        def draw_robot(robot, length=0.5):
            robot_on_canvas = (robot.get_2d_position() * self.render_resolution).astype(np.int)  # Todo use different one for camere position
            robot_in_image_heading_vector = robot_on_canvas + (np.array([math.cos(robot.get_heading()), math.sin(robot.get_heading())]) * length * self.render_resolution).astype(np.int)
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
        corners = (self.camera.get_projected_image_corners() * self.render_resolution).astype(np.int32)
        cv2.polylines(canvas,[corners.reshape((-1,1,2))],True,(0,255,255), 5)

        render_ball_grid = False
        if render_ball_grid:
            # Simulate and check ball grid for testing purses
            for i in [x / 5.0 for x in range(0, 50)]:
                for u in [x / 5.0 for x in range(0, 45)]:
                    ball_position = np.array([i, u], dtype=np.float)
                    if self.camera.check_if_point_is_visible(ball_position):
                        cv2.circle(canvas, tuple([int(e * self.render_resolution) for e in ball_position]), 5, (0,255,0), -1)
                    else:
                        cv2.circle(canvas, tuple([int(e * self.render_resolution) for e in ball_position]), 5, (0,0,255), -1)

        # Check if the ball is visable
        if self.camera.check_if_point_is_visible(self.ball.get_2d_position()):
            cv2.circle(canvas, tuple([int(e * self.render_resolution) for e in self.ball.get_2d_position()]), 10, (0,255,0), -1)
        else:
            cv2.circle(canvas, tuple([int(e * self.render_resolution) for e in self.ball.get_2d_position()]), 10, (0,0,255), -1)

        return canvas
