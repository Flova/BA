import random
import itertools
import numpy as np

class robot_position_gen(object):
    def __init__(self,
                 robot_init_position=(0.0, 0.0),
                 init_velocity=(-0.1, -0.1),
                 walk_speed_factor=1,
                 time_delta = 0.1,
                 ball_noise = 0.1,
                 velocity_to_robot_noise = 0.2,
                 robot_position_interval = (9.0, 6.0),
                 back_velocity = 1):
      
        self._robot_position = np.array(robot_init_position)
        self._velocity = np.array(init_velocity)
        self._time_delta = time_delta
        self._ball_noise = ball_noise
        self._walk_speed_factor = walk_speed_factor
        self._velocity_to_robot_noise = velocity_to_robot_noise
        self._robot_position_interval = robot_position_interval
        self._back_velocity = back_velocity
 
    def __iter__(self):
        return self

    def __next__(self):
        if random.randrange(0, 100) / 100 < 0.1: self.walk()
        if random.randrange(0, 100) / 100 < 0.3: self._velocity = np.array([0.01, 0.01])
        self._apply_velocity()
        self.push_in_field()
        return self._robot_with_noise() , self._robot_position

    def push_in_field(self):
        if not (0 <= self._robot_position[0] <= self._robot_position_interval[0] and  0 <= self._robot_position[1] <= self._robot_position_interval[1]):
            self._velocity[0] = self.push_in_field_one_axis(self._robot_position[0], self._robot_position_interval[0])
            self._velocity[1]= self.push_in_field_one_axis(self._robot_position[1], self._robot_position_interval[1])
            self._apply_velocity()

    def push_in_field_one_axis(self, position, limit):
        if position > limit:
            return -self._back_velocity
        elif position < 0:
            return self._back_velocity
        else:
            return 0.0

    def _apply_velocity(self):
        self._robot_position += self._velocity * self._time_delta

    def walk(self):
        self._velocity = np.random.randn(2) * self._walk_speed_factor

    def _robot_with_noise(self):
        return \
            np.clip(
                self._robot_position + \
                np.random.randn(2) * self._ball_noise * \
                max(1, np.linalg.norm(self._velocity) * self._velocity_to_robot_noise),
                np.array([0.0, 0.0]), np.array(self._robot_position_interval))
