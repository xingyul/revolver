from gym.envs.registration import register

register(
    id='GeneralizedAnt-v0',
    entry_point='envs.mujoco.ant:GeneralizedAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='GeneralizedHumanoid-v0',
    entry_point='envs.mujoco.humanoid:GeneralizedHumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='GeneralHalfCheetah-v0',
    entry_point='envs.mujoco.half_cheetah:GeneralizedHalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='GeneralizedHopper-v0',
    entry_point='envs.mujoco.hopper:GeneralizedHopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
