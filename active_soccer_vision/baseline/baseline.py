from random import sample
import time
import math
from cv2 import norm
import numpy as np

from itertools import repeat

from active_soccer_vision.sim.camera import Camera
from active_soccer_vision.sim.robot import Robot

from active_soccer_vision.envs import SoccerWorldEnv


class EntropyVision:
    def __init__(self) -> None:
        self.field_size = np.array([9, 6])
        self.height = 0.7
        self.num_robots = 3
        self.exploration = False

        self.last_action = np.array([0.5, 0.5])

        self.samples = 20
        self.d = 0.1

    def predict(self, observation):
        
        # Create summy robot pose generators
        dummy_robot_orientation_gen = repeat(tuple([[0,0,math.atan2(
            2 * observation[2] - 1,
            2 * observation[3] - 1)]]*2))
        dummy_robot_position_gen = repeat(tuple(
            [[observation[0] * self.field_size[0],
             observation[1] * self.field_size[1], 0]]* 2))

        # Create robot object
        dummy_robot = Robot(
            orientation_generator=dummy_robot_orientation_gen,
            position_generator=dummy_robot_position_gen,
            height=self.height,
            time_delta=0.1)

        # Calculate linked operations
        dummy_robot.step()

        # Denormalize ball position
        ball_position = observation[4:6] * self.field_size

        # Create camera for our robot
        camera = Camera(fov=math.radians(70), width=1920, height=1080, robot=dummy_robot)

        # Sample action space
        samples = []
        for i in range(self.samples):
            # Sample action with normal distribution between 0 and 1 for both axis
            samples_joint_positions = np.random.normal(self.last_action, np.array([self.d] * self.last_action.shape[0]))
            samples_joint_positions = np.clip(samples_joint_positions, 0, 1)

            # Set normalized actions in camera
            camera.set_pan(samples_joint_positions[0], normalized=True)
            camera.set_tilt(samples_joint_positions[1], normalized=True)

            # Check which objects would be visible from that perspective
            # Check point on map
            score_center_point = int(camera.check_if_point_is_visible(np.array([9/2, 6/2])))

            # Check ball visibility
            if observation[6] > 0.9:
                score_ball = int(camera.check_if_point_is_visible(ball_position))
            else:
                score_ball = 0

            # Check robot visibilities
            def check_robot_visible(id):
                # Check for confidence and visibility
                return observation[7 + id * 3 + 2] > 0.9 and \
                    camera.check_if_point_is_visible(
                        np.array([
                            observation[7 + id * 3],
                            observation[7 + id * 3 + 1]]) * self.field_size)
            # Check how many are visible
            robot_visible_frac = len(list(filter(check_robot_visible, range(self.num_robots)))) / self.num_robots

            # Calculate joint movement penalty
            score_movement_penalty = np.linalg.norm(samples_joint_positions - self.last_action)

            # Weighted sum
            score = score_ball + robot_visible_frac - score_movement_penalty

            # Append sample
            samples.append((samples_joint_positions, score, (score_center_point, score_ball, score_movement_penalty)))

        # Select best action from sampled ones
        joint_positions, max_score, d = max(samples, key=lambda x: x[1])

        # Add gaussian for random exploration
        if self.exploration:
            joint_positions = np.random.normal(joint_positions, np.array([0.1] * joint_positions.shape[0]))
            joint_positions = np.clip(joint_positions, 0, 1)

        # Save state
        self.last_action = joint_positions

        # Normalize
        normalized_action = 2 * joint_positions - 1
        return normalized_action


class EntropyVisionRunner:
    def __init__(self, policy, render=True) -> None:
        self.env = SoccerWorldEnv("entropy_recorded.yaml")
        self.obs = self.env.reset()
        self.policy = policy

    def run(self) -> None:
        while True:
            next_action = self.policy.predict(self.obs)
            res = self.env.step(next_action)
            if res[2]: break
            self.obs = res[0]
            self.env.render()


if __name__ == "__main__":
    active_vision_policy = EntropyVision()
    runner = EntropyVisionRunner(active_vision_policy)
    runner.run()
