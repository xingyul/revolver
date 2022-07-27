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
    parser.add_argument("--save_freq", default=5e4, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default='None')               # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--r_shaping", default=1.)            # Local reward shaping factor
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument('--n_repeat', default=20, type=int)
    parser.add_argument('--use_expl_noise', action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--eval', default=0, type=int)
    parser.add_argument('--eval_interp_param', default=-1., type=float)
    parser.add_argument('--render', default=0, type=int)
    parser.add_argument('--interp_start', default=0., type=float)
    parser.add_argument('--interp_end', default=1., type=float)
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--num_robots', default=10, type=int)
    parser.add_argument('--num_interp', default=10, type=int)
    parser.add_argument('--robot_sample_range', default=0.05, type=float)
    parser.add_argument('--train_sample_range', default=0.05, type=float)
    parser.add_argument("--comment", default="")
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

    if args.r_shaping == 'None':
        args.r_shaping = None
    else:
        args.r_shaping = float(args.r_shaping)

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


    if args.load_model != "":
        policy_file = 'model' if args.load_model == "default" else args.load_model
        policy.load(policy_file, load_optim=False)
        log_string('model loaded')


    # Evaluate well-trained policy
    if args.eval:
        if args.eval_interp_param >= 0:
            evaluations = eval_policy(policy, args.generalized_env, args.eval_interp_param, args.seed, log_string, render=args.render, tmp_file_dir=tmp_file_dir)
        else:
            for interp in np.linspace(0, 1., 10+1):
                log_string('Interp: {}'.format(interp))
                evaluations = [eval_policy(policy, args.generalized_env, float(interp), args.seed, log_string, render=False, tmp_file_dir=tmp_file_dir)]
        exit()

    # writer = utils.WriterLoggerWrapper(result_folder, comment=file_name, max_timesteps=args.max_timesteps)
    writer = SummaryWriter(log_dir=result_folder, comment=file_name)

    # record all parameters value
    with open("{}/parameters.txt".format(result_folder), 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))

    robot_interp_params = np.linspace(args.interp_start, args.interp_end, args.num_robots+1)
    envs = []

    for idx, interp_param in enumerate(robot_interp_params):
        env = make_generalized_envs.generalized_envs[args.generalized_env]( \
                interp_param=float(interp_param), \
                tmp_file_dir=tmp_file_dir, \
                r_shaping=args.r_shaping*interp_param)
        envs.append(env)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # replay_buffer = utils.ReplayBufferTorch(state_dim, action_dim, \
    #         max_size=int(1e6), device=device)
    replay_buffer = utils.ReplayBufferTorchInterp(state_dim, action_dim, \
            max_size=int(args.max_timesteps * args.train_sample_range), device=device)

    interp_params = np.linspace(args.interp_start, args.interp_end, args.num_interp+1)
    step_timesteps = int(args.max_timesteps / (args.num_interp+1))

    total_t = 0
    episode_num = 0
    for idx, interp_param in enumerate(interp_params):

        interp_param_l = interp_param
        interp_param_h = min(interp_param + args.robot_sample_range, 1.)

        replay_buffer.clean([max(interp_param_h - args.train_sample_range, 0), interp_param_h])

        interp_param_sample = np.random.uniform(low=interp_param_l, \
                high=interp_param_h)
        interp_idx = int(round(interp_param_sample * args.num_robots))
        env = envs[interp_idx]

        state, done = env.reset(seed=args.seed + episode_num), False
        episode_reward = 0
        episode_timesteps = 0

        for t in range(step_timesteps):

            episode_timesteps += 1
            total_t += 1

            # Select action randomly or according to policy
            action = policy.select_action(np.array(state), sample_noise=args.expl_noise)

            # Perform action
            next_state, reward, done, _ = env.step(action)

            writer.add_scalar('test/reward', reward, t+1)
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            # replay_buffer.add(state, action, next_state, reward, done_bool)
            replay_buffer.add(state, action, next_state, reward, done_bool, interp_param_sample)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if total_t >= args.start_timesteps:
                train_tuple = replay_buffer.sample(args.batch_size)
                policy.train(train_tuple)

            if done:
                if total_t < args.start_timesteps:
                    log_string('Warm up process')

                log_string("Total T: {} Interp: {} Sampled Interp: {} Episode Num: {} Episode T: {} Reward: {:.3f}".format(total_t, float(interp_param) + args.interp_start, float(interp_param_sample), episode_num+1, episode_timesteps, episode_reward / np.exp(args.r_shaping * interp_param_sample)))

                interp_param_l = interp_param
                interp_param_h = min(interp_param + args.robot_sample_range, 1.)
                interp_param_sample = np.random.uniform(low=interp_param_l, \
                        high=interp_param_h)
                interp_idx = int(round(interp_param_sample * args.num_robots))
                env = envs[interp_idx]

                # Reset environment
                state, done = env.reset(seed=args.seed + episode_num), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (total_t + 1) % args.eval_freq == 0:
                log_string('eval for interp {}'.format(interp_param))
                evaluation_current = eval_policy(policy, args.generalized_env, float(interp_param), args.seed, log_string, tmp_file_dir=tmp_file_dir)
                # evaluations_current.append(evaluation_current)
                # np.save("{}/evaluations_current".format(result_folder), evaluations_current)

                log_string('eval for interp {}'.format(1.))
                evaluation_final = eval_policy(policy, args.generalized_env, 1., args.seed, log_string, tmp_file_dir=tmp_file_dir)
                # evaluations_final.append(evaluation_final)
                # np.save("{}/evaluations_final".format(result_folder), evaluations_final)

                # policy.save("./{}/models/iter_{}_model".format(result_folder, t + 1))

        policy.save("./{}/models/interp_{}_model".format(result_folder, interp_param))

