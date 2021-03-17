from gym.envs.registration import register

register(
    id='matris-v0',
    entry_point='gym_matris.envs:MatrisEnv',
)