import logging
import gym
import numpy as np
from mjrl.utils import tensor_utils
logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time as timer
logging.disable(logging.CRITICAL)


# Single core rollout to sample trajectories
# =======================================================
def do_rollout(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        env_kwargs=None,
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environment (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :param env_kwargs:  dictionary with parameters, will be passed to env generator
    :return:
    """

    # get the correct env behavior
    if type(env) == str:
        env = gym.make(env)
    elif isinstance(env, gym.wrappers.time_limit.TimeLimit):
        pass
    elif isinstance(env, list):
        for i in range(len(env)):
            if type(env[i]) == str:
                env[i] = gym.make(env[i])
            if isinstance(env[i], gym.wrappers.time_limit.TimeLimit):
                pass
    else:
        print("Unsupported environment format")
        raise AttributeError

    paths = []
    for ep in range(num_traj):
        if isinstance(env, list):
            e = env[ep]
        else:
            e = env

        # seeding
        if base_seed is not None:
            seed = base_seed + ep
            o = e.reset(seed=seed)
            np.random.seed(seed)
        else:
            o = e.reset()

        observations=[]
        actions=[]
        rewards=[]
        env_infos = []

        done = False
        t = 0
        while t < horizon and done != True:
            a, agent_info = policy.get_action(o)
            if eval_mode:
                a = agent_info['evaluation']
            next_o, r, done, env_info = e.step(a)
            # below is important to ensure correct env_infos for the timestep
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            env_infos.append(env_info)
            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done
        )
        paths.append(path)

    if isinstance(env, list):
        for e in env:
            del(e)
    del(env)
    return paths


def sample_paths(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        max_process_time=300,
        max_timeouts=4,
        suppress_print=False,
        env_kwargs=None,
        ):

    assert type(num_cpu) == int
    assert num_cpu == 1

    if num_cpu == 1:
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs)
        # dont invoke multiprocessing if not necessary
        return do_rollout(**input_dict)


