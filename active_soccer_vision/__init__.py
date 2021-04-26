from gym.envs.registration import register

register(
    id='soccer_world-v0',
    entry_point='active_soccer_vision.envs:SoccerWorldEnv',
)