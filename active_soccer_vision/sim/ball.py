import random
import itertools
import numpy as np

class ball_position_gen(object):
    def __init__(self,
                 ball_init_position=(0.0, 0.0),
                 init_velocity=(0.0, 0.0),
                 time_delta = 0.1,
                 friction_factor = 0.8,
                 kick_intensity = 5.0,
                 kick_prop = 0.005,
                 ball_noise = 0.1,
                 velocity_to_ball_noise = 0.2,
                 ball_position_interval = (9.0, 6.0),
                 back_velocity = 1):
      
        self._ball_position = np.array(ball_init_position)
        self._velocity = np.array(init_velocity)
        self._time_delta = time_delta
        self._friction_factor = friction_factor
        self._kick_intensity = kick_intensity
        self._kick_prop = kick_prop
        self._ball_noise = ball_noise
        self._velocity_to_ball_noise = velocity_to_ball_noise
        self._ball_position_interval = ball_position_interval
        self._back_velocity = back_velocity
 
    def __iter__(self):
        return self

    def __next__(self):
        # Kick with a prop
        if random.randrange(0, 100) / 100 < self._kick_prop: self.kick()
        self._apply_velocity()
        self._apply_friction()
        self.push_in_field()
        return self._ball_with_noise() , self._ball_position

    def push_in_field(self):
        if not (0 <= self._ball_position[0] <= self._ball_position_interval[0] and  0 <= self._ball_position[1] <= self._ball_position_interval[1]):
            self._velocity[0] = self.push_in_field_one_axis(self._ball_position[0], self._ball_position_interval[0])
            self._velocity[1]= self.push_in_field_one_axis(self._ball_position[1], self._ball_position_interval[1])
            self._apply_velocity()

    def push_in_field_one_axis(self, position, limit):
        if position > limit:
            return -self._back_velocity
        elif position < 0:
            return self._back_velocity
        else:
            return 0.0

    def kick(self):
        if np.linalg.norm(self._velocity) > 0.01: return
        self._velocity = np.random.randn(2) * self._kick_intensity

    def _apply_velocity(self):
        self._ball_position += self._velocity * self._time_delta

    def _apply_friction(self):
        self._velocity *= self._friction_factor 

    def _ball_with_noise(self):
        return \
            np.clip(
                self._ball_position + \
                np.random.randn(2) * self._ball_noise * \
                max(1, np.linalg.norm(self._velocity) * self._velocity_to_ball_noise),
                np.array([0.0, 0.0]), np.array(self._ball_position_interval))

"""
positions = np.array(
    list(
        itertools.islice(
            ball_position_gen(), 
            3000)))
"""
