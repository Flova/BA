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

        self.last_action = np.array([0.5, 0.5])

        self.samples = 20
        self.d = 0.1

    def predict(self, observation):
        dummy_robot_orientation_gen = repeat(tuple([[0,0,math.atan2(
            2 * observation[2] - 1,
            2 * observation[3] - 1)]]*2))

        dummy_robot_position_gen = repeat(tuple(
            [[observation[0] * self.field_size[0],
             observation[1] * self.field_size[1], 0]]* 2))

        dummy_robot = Robot(
            orientation_generator=dummy_robot_orientation_gen,
            position_generator=dummy_robot_position_gen,
            height=self.height,
            time_delta=0.1)

        dummy_robot.step()

        ball_position = observation[4:6] * self.field_size

        print(np.array([
            observation[7 + 2 * 3],
            observation[7 + 2 * 3 + 1]]))

        camera = Camera(fov=math.radians(70), width=1920, height=1080, robot=dummy_robot)

        samples = []

        for i in range(self.samples):
            samples_joint_positions = np.random.normal(self.last_action, np.array([self.d] * self.last_action.shape[0]))

            samples_joint_positions = np.clip(samples_joint_positions, 0, 1)

            camera.set_pan(samples_joint_positions[0], normalized=True)
            camera.set_tilt(samples_joint_positions[1], normalized=True)

            score_center_point = int(camera.check_if_point_is_visible(np.array([9/2, 6/2])))
            score_movement_penalty = np.linalg.norm(samples_joint_positions - self.last_action)
            if observation[6] > 0.9:
                score_ball = int(camera.check_if_point_is_visible(ball_position))
            else:
                score_ball = 0

            def check_robot_visible(id):
                return observation[7 + id * 3 + 2] > 0.9 and \
                    camera.check_if_point_is_visible(
                        np.array([
                            observation[7 + id * 3],
                            observation[7 + id * 3 + 1]]) * self.field_size)

            robot_visible_frac = len(list(filter(check_robot_visible, range(self.num_robots)))) / self.num_robots

            score = score_ball + robot_visible_frac - score_movement_penalty

            samples.append((samples_joint_positions, score, (score_center_point, score_ball, score_movement_penalty)))

        joint_positions, max_score, d = max(samples, key=lambda x: x[1])

        self.last_action = joint_positions

        print(max_score, joint_positions)

        return 2 * joint_positions - 1


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
