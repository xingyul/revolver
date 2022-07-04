import numpy as np
import torch
import gym
import argparse
import glob
import os
# import matplotlib.pyplot as plt
import cv2
import utils
import datetime
import time
import mujoco_py

from torch.utils.tensorboard import SummaryWriter

import envs.mujoco

import make_generalized_envs


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, generalized_env_name, interp_param, seed, log_string, \
        render=False, eval_episodes=10, tmp_file_dir='/tmp/'):
    eval_env = make_generalized_envs.generalized_envs[generalized_env_name]( \
            interp_param, tmp_file_dir=tmp_file_dir)

    if render:
        camera_name = 'track'
        mode = 'rgb_array'
        camera_id = eval_env.env.model.camera_name2id(camera_name)
        viewer = eval_env.env._get_viewer(mode)

    rewards = []
    for e in range(eval_episodes):
        episode_reward = 0
        state, done = eval_env.reset(seed=2**32 - seed - eval_episodes*2 + e), False
        while not done:
            if render:
                # viewer.render()
                viewer.render(500, 500, camera_id=camera_id)
                image = viewer.read_pixels(500, 500, depth=False)
                cv2.imshow('viz', image[::-1, :, ::-1])
                cv2.waitKey(20)

            action = policy.select_action(np.array(state))

            state, reward, done, _ = eval_env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)

    log_string("---------------------------------------")
    log_string("Evaluation over {} episodes: {:.3f}".format(eval_episodes, np.mean(rewards)))
    log_string("---------------------------------------")
    return rewards


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="GRAC")                 # Policy name (GRAC)
    parser.add_argument("--generalized_env", default="Ant-v2-leg-length")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--save_freq", default=5e5, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=None)               # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument('--n_repeat', default=20, type=int)
    parser.add_argument('--use_expl_noise', action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--eval', default=0, type=int)
    parser.add_argument('--eval_interp_param', default=0., type=float)
    parser.add_argument("--actor_lr", type=float, default=0.0003)   # Actor learning rate
    parser.add_argument("--critic_lr", type=float, default=0.0003)  # Critic learning rate
    parser.add_argument("--comment", default="")
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--which_cuda", default='0')
    parser.add_argument('--command_file', default=None, help='Command file name [default: None]')
    parser.add_argument("--log_dir", default='log')

    args = parser.parse_args()

    if args.which_cuda == 'None':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.which_cuda))

    file_name = args.log_dir
    file_name += "_{}".format(args.comment) if args.comment != "" else ""
    folder_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

    result_folder = os.path.join(args.log_dir, folder_name)

    if not os.path.exists('{}/models/'.format(result_folder)):
        os.system('mkdir -p {}/models/'.format(result_folder))

    tmp_file_dir = os.path.join(os.path.abspath(result_folder), 'tmp')
    if not os.path.exists(tmp_file_dir):
        os.system('mkdir -p {}'.format(tmp_file_dir))

    LOG_FOUT = open(os.path.join(result_folder, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(args)+'\n')
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)

    os.system('cp {} {}'.format(__file__, result_folder))
    os.system('cp {} {}'.format(args.policy + '.py', result_folder))
    os.system('cp {} {}'.format(args.command_file, result_folder))
    os.system('cp {} {}'.format('make_generalized_envs*.py', result_folder))

    log_string('pid: %s'%(str(os.getpid())))


    log_string("---------------------------------------")
    log_string("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.generalized_env, args.seed))
    log_string("---------------------------------------")



    env = make_generalized_envs.generalized_envs[args.generalized_env]( \
            interp_param=0., tmp_file_dir=tmp_file_dir)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.save_model is False:
        args.save_model = True

    # Initialize policy
    if 'TD3' in args.policy:
        kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "batch_size": args.batch_size,
                "discount": args.discount,
                "tau": args.tau,
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
                "device": device,
        }
        TD3 = __import__(args.policy)
        policy = TD3.TD3(**kwargs)
    elif 'SAC' in args.policy:
        kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "batch_size": args.batch_size,
                "discount": args.discount,
                "tau": args.tau,
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
                "device": device,
        }
        SAC = __import__(args.policy)
        policy = SAC.SAC(**kwargs)


    print(args.load_model)
    models_to_load = glob.glob(os.path.join(args.load_model + '*', '*', 'models', 'interp_1.0_model_actor'))
    if len(models_to_load) == 0:
        # models_to_load = glob.glob(os.path.join(args.load_model + '*', '*', 'models', 'iter_3000000_model_actor'))
        models_to_load = glob.glob(os.path.join(args.load_model + '*', '*', 'models', 'iter_{}_model_actor'.format(args.max_timesteps)))
    models_to_load = [m.split('_actor')[0] for m in models_to_load]
    models_to_load.sort()
    print(str(len(models_to_load)) + ' models in total')

    eval_rewards = []
    for model in models_to_load:
        print('')
        print(model)
        policy.load(model, load_optim=False)

        # Evaluate well-trained policy
        if args.eval_interp_param >= 0:
            evaluations = eval_policy(policy, args.generalized_env, args.eval_interp_param, args.seed, log_string, render=args.render, tmp_file_dir=tmp_file_dir)
            print(evaluations)
            eval_rewards += evaluations
        else:
            for interp in np.linspace(0, 1., 10+1):
                log_string('Interp: {}'.format(interp))
                evaluations = [eval_policy(policy, args.generalized_env, float(interp), args.seed, log_string, render=False, tmp_file_dir=tmp_file_dir)]

    mean = np.mean(eval_rewards)
    std = np.std(eval_rewards)
    print('{:.2f}'.format(mean) + u' \u00B1 ' + '{:.2f}'.format(std))

    eval_rewards = np.reshape(eval_rewards, [len(models_to_load), -1])
    eval_rewards = np.mean(eval_rewards, axis=-1)
    mean = np.mean(eval_rewards)
    std = np.std(eval_rewards)
    print('{:.2f}'.format(mean) + u' \u00B1 ' + '{:.2f}'.format(std))
