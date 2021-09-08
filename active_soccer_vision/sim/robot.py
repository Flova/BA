import random
import itertools
import numpy as np
import transforms3d

class robot_position_gen(object):
    def __init__(self,
                 time_delta = 0.1,
                 robot_position_interval = (9.0, 6.0),
                 robot_init_position=(0.0, 0.0),
                 init_velocity=(-0.1, -0.1),
                 walk_speed_factor=0.1,
                 noise = 0.05,
                 walk_prop = 0.1,
                 stop_prop = 0.01,
                 back_velocity = 1):

        self._robot_position = np.array(robot_init_position)
        self._velocity = np.array(init_velocity)
        self._time_delta = time_delta
        self._robot_noise = noise
        self._walk_speed_factor = walk_speed_factor
        self._robot_position_interval = robot_position_interval
        self._back_velocity = back_velocity
        self._walk_prop = walk_prop
        self._stop_prop = stop_prop

    def __iter__(self):
        return self

    def __next__(self):
        if random.randrange(0, 100) / 100 < self._walk_prop: self.walk()
        if random.randrange(0, 100) / 100 < self._stop_prop:
            self._velocity = np.array([0.0, 0.0])
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
                np.random.randn(2) * self._robot_noise,
                np.array([0.0, 0.0]), np.array(self._robot_position_interval))


class robot_orientation_gen(object):
    def __init__(self,
                 time_delta = 0.1,
                 robot_init_orientation=(0.0, 0.0, 0.0),
                 init_velocity=(-0.0, -0.0, 0.0),
                 turn_speed_factor=0.2,
                 turn_prop = 0.1,
                 stop_prop = 0.01,
                 noise = 0.0000001):

        self._robot_orientation = np.array(robot_init_orientation)
        self._velocity = np.array(init_velocity)
        self._time_delta = time_delta
        self._noise = noise
        self._turn_speed_factor = turn_speed_factor
        self._turn_prop = turn_prop
        self._stop_prop = stop_prop

    def __iter__(self):
        return self

    def __next__(self):
        if random.randrange(0, 100) / 100 < self._turn_prop: self.turn()
        if random.randrange(0, 100) / 100 < self._stop_prop:
            self._velocity = np.array([0.0, 0.0, 0.0])
        self._apply_velocity()
        return self._robot_orientation_with_noise(), self._robot_orientation

    def _apply_velocity(self):
        self._robot_orientation += self._velocity * self._time_delta

    def turn(self):
        self._velocity[2] = np.random.randn(1) * self._turn_speed_factor

    def _robot_orientation_with_noise(self):
        pos = self._robot_orientation + np.random.randn(3) * self._noise
        return pos

class Robot:
    def __init__(self, position_generator, orientation_generator, time_delta):
        self.position_generator = position_generator
        self.orientation_generator = orientation_generator
        self.time_delta = time_delta
        self.position = None
        self.orientation = None
        self.base_footprint = None
        self.last_observed_position = np.array([0, 0])
        self.last_observed_heading = 0
        self.conf = 0

    def step(self):
        self.orientation, _ = self.orientation_generator.__next__()
        self.position, _ = self.position_generator.__next__()
        self.conf = max(self.conf - 0.1 * self.time_delta, 0.0)

    def get_base_footprint(self):
        return transforms3d.affines.compose(
                [self.position[0], self.position[1], 0.0],
                transforms3d.euler.euler2mat(
                    self.orientation[0],
                    self.orientation[1],
                    self.orientation[2]
                ),
                np.ones(3)
            )

    def get_heading(self):
        return self.orientation[2]

    def get_2d_position(self):
        return self.position

    def observe(self):
        self.last_observed_position = self.get_2d_position()
        self.last_observed_heading = self.get_heading()
        self.conf = 1

    def get_last_observed_2d_position(self):
        return self.last_observed_position, self.conf

    def get_last_observed_heading(self):
        return self.last_observed_heading, self.conf
