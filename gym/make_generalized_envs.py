

from make_generalized_envs_ant import *
from make_generalized_envs_humanoid import *


generalized_envs = {
        'Ant-v2': generalized_ant_original,
        # 'Ant-v2-leg-length-mass': generalized_ant_leg_length_mass,
        'Ant-v2-leg-emerge': generalized_ant_leg_emerge,

        'Humanoid-v2': generalized_humanoid_original,
        'Humanoid-v2-leg-length-mass': generalized_humanoid_leg_length_mass,
        }


if __name__ == '__main__':
    import envs.mujoco

    interp_param = 0.

    generalized_env = 'Ant-v2-leg-length-mass'
    generalized_env = 'Ant-v2-leg-emerge'
    action_dim = 8

    env = generalized_envs[generalized_env](interp_param=interp_param)

    viz = True
    if viz:
        while True:
            env.step(np.zeros([action_dim]))
            env.render()


