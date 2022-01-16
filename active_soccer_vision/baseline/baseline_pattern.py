from random import sample
import time
import math
from cv2 import norm
import numpy as np

from itertools import repeat

from active_soccer_vision.sim.camera import Camera
from active_soccer_vision.sim.robot import Robot

from active_soccer_vision.envs import SoccerWorldEnv


class PatternVision:
    def __init__(self) -> None:

        self.max_speed = 0.03

        self.pattern = self.generate_pattern(
            lineCount = 3,
            maxHorizontalAngleLeft=0,
            maxHorizontalAngleRight=1,
            maxVerticalAngleUp=1,
            maxVerticalAngleDown=0,
            reduce_last_scanline=0,
            interpolation_steps=0)

        self.step = 0
        self.index = 0

    def _lineAngle(self, line, line_count, min_angle, max_angle):
        """
        Converts a scanline number to an tilt angle
        """
        delta = abs(min_angle - max_angle)
        steps = delta / (line_count - 1)
        value = steps * line + min_angle
        return value

    def _calculateHorizontalAngle(self, is_right, angle_right, angle_left):
        """
        The right/left position to an pan angle
        """
        if is_right:
            return angle_right
        else:
            return angle_left

    def _interpolatedSteps(self, steps, tilt, min_pan, max_pan):
        """
        Splits a scanline in a number of dedicated steps
        """
        if steps == 0:
           return []
        steps += 1
        delta = abs(min_pan - max_pan)
        step_size = delta / float(steps)
        output_points = list()
        for i in range(1, steps):
            value = int(i * step_size + min_pan)
            point = (value, tilt)
            output_points.append(point)
        return output_points

    def generate_pattern(self, lineCount, maxHorizontalAngleLeft, maxHorizontalAngleRight, maxVerticalAngleUp, maxVerticalAngleDown, reduce_last_scanline=1, interpolation_steps=0):
        """
        :param lineCount: Number of scanlines
        :param maxHorizontalAngleLeft: maximum look left angle
        :param maxHorizontalAngleRight: maximum look right angle
        :param maxVerticalAngleUp: maximum upwards angle
        :param maxVerticalAngleDown: maximum downwards angle
        :param interpolation_steps: number of interpolation steps for each line
        :return: List of angles (Pan, Tilt)
        """
        keyframes = []
        # Init first state
        downDirection = False
        rightSide = False
        rightDirection = True
        line = lineCount - 1
        # Calculate number of keyframes
        iterations = max((2 * lineCount - 2) * 2, 2)

        for i in range(iterations):
            # Create keyframe
            currentPoint = (self._calculateHorizontalAngle(rightSide, maxHorizontalAngleRight, maxHorizontalAngleLeft),
                            self._lineAngle(line, lineCount, maxVerticalAngleDown, maxVerticalAngleUp))
            # Add keyframe
            keyframes.append(currentPoint)

            # Interpolate to next keyframe if we are moving horizontally
            if rightSide != rightDirection:
                interpolatedKeyframes = self._interpolatedSteps(interpolation_steps, currentPoint[1], maxHorizontalAngleRight, maxHorizontalAngleLeft)
                if rightDirection:
                    interpolatedKeyframes.reverse()
                keyframes.extend(interpolatedKeyframes)

            # Next state
            # Switch side
            if rightSide != rightDirection:
                rightSide = rightDirection
            # Or go up/down
            elif rightSide == rightDirection:
                rightDirection = not rightDirection
                if line in [0, lineCount - 1]:
                    downDirection = not downDirection
                if downDirection:
                    line -= 1
                else:
                    line += 1

        # Reduce the with of the last scanline if wanted.
        for index, keyframe in enumerate(keyframes):
            if keyframe[1] == maxVerticalAngleDown:
                keyframes[index] = (keyframe[0] * reduce_last_scanline, maxVerticalAngleDown)

        return keyframes

    def predict(self, observation):

        x_diff = self.pattern[self.index][0] - self.pattern[(self.index + 1) % len(self.pattern)][0]
        y_diff = self.pattern[self.index][1] - self.pattern[(self.index + 1) % len(self.pattern)][1]

        x = self.pattern[self.index][0] - (x_diff / (max(map(abs, [x_diff, y_diff, self.max_speed])) / self.max_speed)) * self.step
        y = self.pattern[self.index][1] - (y_diff / (max(map(abs, [x_diff, y_diff, self.max_speed])) / self.max_speed)) * self.step

        print(x,y)
        print(self.pattern[(self.index + 1) % len(self.pattern)][1])

        if abs(x - self.pattern[(self.index + 1) % len(self.pattern)][0]) < 0.1 and  abs(y - self.pattern[(self.index + 1) % len(self.pattern)][1]) < 0.1:
            self.step = 0
            self.index = (self.index + 1) % len(self.pattern)

        self.step += 1


        joint_positions = np.array([x,y])
        #print(self.pattern, joint_positions)


        # Normalize
        normalized_action = 2 * joint_positions - 1
        return normalized_action


class EntropyVisionRunner:
    def __init__(self, policy, render=True) -> None:
        self.env = SoccerWorldEnv("pattern_recorded.yaml")
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
    active_vision_policy = PatternVision()
    runner = EntropyVisionRunner(active_vision_policy)
    runner.run()
