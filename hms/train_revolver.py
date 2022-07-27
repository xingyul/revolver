
from mjrl.algos.npg_cg import NPG
import mjrl.samplers.core as trajectory_sampler
import os
import json
import gym
import numpy as np
from datetime import datetime
import time as timer
import copy
import torch
import pickle
import random
from tabulate import tabulate
import argparse
import gc

import hand_manipulation_suite
import make_generalized_envs

# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Natural policy gradient from mjrl on mujoco environments')

parser.add_argument('--command_file', type=str, default='command_train_revolver.sh', help='command file name')
parser.add_argument('--generalized_env', type=str, default='hammer-v0-finger-shrink', help='env name')
parser.add_argument('--gpu', type=str, default='-1', help='gpu id')
parser.add_argument('--algorithm', type=str, default='NPG', help='algo name')
parser.add_argument('--policy', type=str, required=True, help='path to policy file')
parser.add_argument('--baseline', type=str, default='None', help='path to baseline file')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--interp_start', type=float, default=0., help='interp start')
parser.add_argument('--interp_end', type=float, default=1., help='interp end')
parser.add_argument('--interp_progression', type=float, default=0.02, help='interp progression every step')
parser.add_argument('--r_shaping', type=float, default=0., help='local reward shaping factor')
parser.add_argument('--num_robots', type=int, default=1000, help='number of robots')
parser.add_argument('--random_interp_range', type=float, default=0.05, help='range of random')
parser.add_argument('--sample_mode', type=str, default='trajectories', help='sample mode')
parser.add_argument('--num_traj', type=int, default=16, help='num of trajectories to sample')
parser.add_argument('--num_cpu', type=int, default=1, help='num of cpu')
parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
parser.add_argument('--step_size', type=float, default=0.0001, help='training step size')
parser.add_argument('--gamma', type=float, default=0.995, help='gamma')
parser.add_argument('--gae', type=float, default=0.97, help='gae')
parser.add_argument('--log_dir', type=str, required=True, help='location to store results')
args = parser.parse_args()


if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if not os.path.exists(os.path.join(args.log_dir, 'iterations')):
    os.mkdir(os.path.join(args.log_dir, 'iterations'))

os.system('cp {} {}'.format(args.command_file, args.log_dir))
os.system('cp {} {}'.format('make_generalized_envs.py', args.log_dir))
os.system('cp {} {}'.format('hand_manipulation_suite/generalized_*.py', args.log_dir))
os.system('cp {} {}'.format(__file__, args.log_dir))


LOG_FOUT = open(os.path.join(args.log_dir, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
log_string('pid: %s'%(str(os.getpid())))


# ===============================================================================
# Train Loop
# ===============================================================================

policy = pickle.load(open(args.policy, 'rb'))
baseline = pickle.load(open(args.baseline, 'rb'))

if (args.gpu != 'None') and (args.gpu != '-1'):
    device = torch.device('cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')

policy.to(device)
baseline.to(device)


tmp_file_dir = os.path.join(os.path.abspath(args.log_dir), 'tmp')
if not os.path.exists(tmp_file_dir):
    os.system('mkdir -p {}'.format(tmp_file_dir))

env = make_generalized_envs.generalized_envs[args.generalized_env]( \
        interp_param=1., \
        dense_reward=False, \
        r_shaping=0.)

np.random.seed(args.seed)
policy.seed(args.seed)


# Construct the algorithm
if args.algorithm == 'NPG':
    # Other hyperparameters (like number of CG steps) can be specified in config for pass through
    # or default hyperparameters will be used
    agent = NPG(env, policy, baseline, \
            normalized_step_size=args.step_size,
            seed=args.seed, save_logs=True, device=device, **dict())


robot_interp_params = np.linspace(args.interp_start, args.interp_end, args.num_robots+1)
envs = {}

warmup_interp_params = robot_interp_params[\
    (robot_interp_params <= (args.interp_start + args.random_interp_range)) & \
    (robot_interp_params > args.interp_start)]


# for idx, interp_param in enumerate(robot_interp_params):
for idx, interp_param in enumerate(warmup_interp_params):
    env = make_generalized_envs.generalized_envs[args.generalized_env]( \
            interp_param=float(interp_param), \
            dense_reward=False, \
            r_shaping=float(interp_param)*args.r_shaping)
    env.reset()
    envs[interp_param] = env


progressing_interps = np.arange(args.interp_start, args.interp_end + 1e-6, args.interp_progression)

for idx, interp_param in enumerate(progressing_interps):

    interp_param_l = interp_param
    interp_param_h = min(interp_param + args.random_interp_range, args.interp_end)

    interps_to_sample_from = robot_interp_params[ \
            (robot_interp_params <= interp_param_h) & \
            (robot_interp_params >= interp_param_l)]

    #### deal with memory free
    current_interps = np.array(list(envs.keys()))
    interps_to_throw = current_interps[current_interps < interp_param_l]
    for i in interps_to_throw:
        del envs[i]
        # gc.collect()
    interps_to_add = robot_interp_params[ \
            (robot_interp_params <= interp_param_h) & \
            (robot_interp_params >= interp_param_l)]
    for i in interps_to_add:
        if i not in envs:
            env = make_generalized_envs.generalized_envs[args.generalized_env]( \
                    interp_param=float(i), \
                    dense_reward=False, \
                    r_shaping=float(i)*args.r_shaping)
            env.reset()
            envs[i] = env

    iteration = 0
    consecutive_pass = 0
    while True:

        if interps_to_sample_from.shape[0] >= args.num_traj:
            interp_param_sampled = np.random.choice(interps_to_sample_from, \
                    size=args.num_traj, replace=False)
        else:
            interp_param_sampled = np.concatenate([interps_to_sample_from, \
                    np.random.choice(interps_to_sample_from, \
                    size=args.num_traj - interps_to_sample_from.shape[0], \
                    replace=True)])
        env = [envs[i] for i in interp_param_sampled]

        N = args.num_traj if args.sample_mode == 'trajectories' else num_samples
        train_args = dict(N=N, env=env, sample_mode=args.sample_mode, gamma=args.gamma, \
                gae_lambda=args.gae, num_cpu=args.num_cpu, train_baseline_only=False)
        stats = agent.train_step(**train_args)

        scaled_reward = stats[0]
        original_reward = np.array(scaled_reward) / \
                np.exp(np.array([e.env.r_shaping for e in env]))
        max_original_reward = np.max(original_reward)

        train_log = agent.logger.get_current_log()
        stoc_pol_max = float(train_log['stoc_pol_max'])
        success_rate_train = float(train_log['success_rate'])

        iteration += 1

        log_string("")
        log_string("...........................................................")
        log_string(str(datetime.now()))
        log_string("Interp: {} ".format(interp_param))
        log_string("Iteration : {} ".format(iteration))
        log_string("Interp Sampled: {} ".format(interp_param_sampled.tolist()))
        log_string('max original reward: {} '.format(max_original_reward))

        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1, train_log.items()))
            log_string(tabulate(print_data))

        if interp_param < 1.0:
            num_traj = 20
        else:
            num_traj = 50

        if (iteration % 5) == 0:
            env = [envs[i] for i in sorted(envs.keys(), reverse=False)][0]

            input_dict = dict(num_traj=num_traj, env=env, \
                    policy=agent.policy, \
                    horizon=1e6, base_seed=2**30, \
                    num_cpu=args.num_cpu, eval_mode=True)
            paths = trajectory_sampler.sample_paths(**input_dict)
            success_rate_eval = env.env.evaluate_success(paths) / 100.
            log_string('Eval Success Rate: {}'.format(success_rate_eval))
            if interp_param < 1.0:
                if success_rate_eval >= 0.5:
                    break
            else:
                if success_rate_eval >= 0.9:
                    break

    if ((idx + 1) % args.save_freq == 0):
        policy_file = 'policy_{}.pickle'.format('{:.4f}'.format(interp_param))
        baseline_file = 'baseline_{}.pickle'.format('{:.4f}'.format(interp_param))
        pickle.dump(agent.policy, \
                open(os.path.join(args.log_dir, 'iterations', policy_file), 'wb'))
        pickle.dump(agent.baseline, \
                open(os.path.join(args.log_dir, 'iterations', baseline_file), 'wb'))


