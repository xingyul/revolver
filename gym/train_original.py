import numpy as np
import torch
import gym
import argparse
import os
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

    avg_reward = 0.
    for e in range(eval_episodes):
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
            avg_reward += reward

    avg_reward /= eval_episodes

    log_string("---------------------------------------")
    log_string("Evaluation over {} episodes: {:.3f}".format(eval_episodes, avg_reward))
    log_string("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="GRAC")                 # Policy name (GRAC)
    parser.add_argument("--generalized_env", default="Ant-v2-leg-length")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--save_freq", default=5e5, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=None)               # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--actor_lr", type=float, default=0.0003)   # Actor learning rate
    parser.add_argument("--critic_lr", type=float, default=0.0003)  # Critic learning rate
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument('--n_repeat', default=20, type=int)
    parser.add_argument('--use_expl_noise', action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--eval', default=0, type=int)
    parser.add_argument('--interp', default=0., type=float)
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

    if args.expl_noise == 'None':
        args.expl_noise = None
    else:
        args.expl_noise = float(args.expl_noise)

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


    env = make_generalized_envs.generalized_envs[args.generalized_env](interp_param=args.interp, tmp_file_dir=tmp_file_dir)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
                "device": device,
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
        }
        TD3 = __import__(args.policy)
        policy = TD3.TD3(**kwargs)
    elif 'DDPG' in args.policy:
        kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "batch_size": args.batch_size,
                "discount": args.discount,
                "tau": args.tau,
                "device": device,
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
        }
        DDPG = __import__(args.policy)
        policy = DDPG.DDPG(**kwargs)
    elif 'SAC' in args.policy:
        kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "batch_size": args.batch_size,
                "discount": args.discount,
                "tau": args.tau,
                "device": device,
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
        }
        SAC = __import__(args.policy)
        policy = SAC.SAC(**kwargs)


    if args.load_model != "":
        policy_file = 'model' if args.load_model == "default" else args.load_model
        policy.load(policy_file)
        log_string('model loaded')

    replay_buffer = utils.ReplayBufferTorch(state_dim, action_dim, device=device)

    # Evaluate untrained policy
    log_string('eval for interp {}'.format(args.interp))
    evaluations = [eval_policy(policy, args.generalized_env, args.interp, args.seed, log_string, render=False, tmp_file_dir=tmp_file_dir)]
    if args.eval:
        exit()

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    state, done = env.reset(seed=args.seed + episode_num), False

    # writer = utils.WriterLoggerWrapper(result_folder, comment=file_name, max_timesteps=args.max_timesteps)
    writer = SummaryWriter(log_dir=result_folder, comment=file_name)

    #record all parameters value
    with open("{}/parameters.txt".format(result_folder), 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if (t < args.start_timesteps) and (args.load_model == ""):
            action = np.random.uniform(-max_action, max_action, action_dim)
        else:
            action = policy.select_action(np.array(state), sample_noise=args.expl_noise)


        # Perform action
        try:
            next_state, reward, done, _ = env.step(action)
            writer.add_scalar('test/reward', reward, t+1)
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward
        except:
            log_string('NaN in Sampling')
            env = make_generalized_envs.generalized_envs[args.generalized_env](interp_param=args.interp, tmp_file_dir=tmp_file_dir)

            done = True


        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            train_tuple = replay_buffer.sample(args.batch_size)
            policy.train(train_tuple)

        if done:
            if t < args.start_timesteps:
                log_string('Warp up process')
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            log_string("Total T: {} Episode Num: {} Episode T: {} Reward: {:.3f}".format(t+1, episode_num+1, episode_timesteps, episode_reward))
            # Reset environment
            state, done = env.reset(seed=args.seed + episode_num + 1), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluation = eval_policy(policy, args.generalized_env, args.interp, args.seed, log_string, tmp_file_dir=tmp_file_dir)
            evaluations.append(evaluation)
            writer.add_scalar('test/avg_return', evaluation, t+1)
            np.save("{}/evaluations".format(result_folder), evaluations)

        if (t + 1) % args.save_freq == 0:
            policy.save("./{}/models/iter_{}_model".format(result_folder, t + 1))

        # if (t + 1) % args.eval_freq == 0:
        #     if args.save_model:
        #         policy.save("./{}/models/iter_{}_model".format(result_folder, t + 1))
        #         # replay_buffer.save(result_folder)

        # save to txt
        # if (t + 1) % 50000 == 0:
            # writer.logger.save_to_txt()
