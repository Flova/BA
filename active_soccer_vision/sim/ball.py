import random
import itertools
import numpy as np

class ball_position_gen(object):
    def __init__(self,
                 time_delta = 0.1,
                 ball_position_interval = (9.0, 6.0),
                 ball_init_position=(0.0, 0.0),
                 init_velocity=(0.0, 0.0),
                 friction_factor = 0.8,
                 kick_intensity = 5.0,
                 kick_prop = 0.005,
                 ball_noise = 0.1,
                 back_velocity = 1):

        self._ball_position = np.array(ball_init_position)
        self._velocity = np.array(init_velocity)
        self._time_delta = time_delta
        self._friction_factor = friction_factor
        self._kick_intensity = kick_intensity
        self._kick_prop = kick_prop
        self._ball_noise = ball_noise
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
                np.random.randn(2) * self._ball_noise,
                np.array([0.0, 0.0]), np.array(self._ball_position_interval))

class ball_position_player:
    def __init__(self,
                 game_log,
                 time_delta = 0.1,
                 start=0.0,
                 stop=None,
                 ball_position_interval = (9.0, 6.0),
                 ball_noise = 0.1):

        self._time_delta = time_delta
        self._ball_noise = ball_noise
        self._ball_position_interval = np.array(ball_position_interval)
        self._game_log = game_log
        ball_id = self._game_log.x3d.get_ball_id()
        self._ball_movement = self._game_log.game_data.get_interpolated_translations(
            id=ball_id, start=start, stop=stop, step_size=time_delta)[:, 0:2]
        self._step = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Get current frame
        ball_position = self._ball_movement[self._step]

        # Transform coordinates
        ball_position += self._ball_position_interval / 2

        # Apply noise
        ball_with_noise = np.clip(
                ball_position + np.random.randn(2) * self._ball_noise,
                np.array([0.0, 0.0]), self._ball_position_interval)

        # Step
        self._step += 1

        return ball_with_noise , ball_position


class Ball:
    def __init__(self, position_generator, time_delta):
        self.position_generator = position_generator
        self.time_delta = time_delta
        self.position = np.array([0, 0])
        self.last_observed_position = np.array([0, 0])
        self.conf = 0

    def step(self):
        self.position, _ = self.position_generator.__next__()
        self.conf = max(self.conf - 0.1 * self.time_delta, 0.0)

    def get_2d_position(self):
        return self.position

    def observe(self):
        self.last_observed_position = self.get_2d_position()
        self.conf = 1.0

    def get_last_observed_2d_position(self):
        return self.last_observed_position, self.conf
