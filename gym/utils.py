import numpy as np
import math
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device('cuda')):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

class ReplayBufferTorch(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device('cuda')):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = torch.zeros((max_size, state_dim), device=device)
		self.action = torch.zeros((max_size, action_dim), device=device)
		self.next_state = torch.zeros((max_size, state_dim), device=device)
		self.reward = torch.zeros((max_size, 1), device=device)
		self.not_done = torch.zeros((max_size, 1), device=device)

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = torch.tensor(state, device=self.device)
		self.action[self.ptr] = torch.tensor(action, device=self.device)
		self.next_state[self.ptr] = torch.tensor(next_state, device=self.device)
		self.reward[self.ptr] = torch.tensor(reward, device=self.device)
		self.not_done[self.ptr] = torch.tensor(1. - done, device=self.device)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind]
		)
	# def save(self, log_dir):
	# 	import pickle
	# 	with open('{}/buffer_{}.pkl'.format(log_dir, self.ptr), 'wb+') as f:
	# 		pickle.dump({
	# 			'state': self.state,
	# 			'action': self.action,
	# 			'next_state': self.next_state,
	# 			'reward': self.reward,
	# 			'not_done': self.not_done
	# 		}, f)

class ReplayBufferTorchInterp(object):
        def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device('cuda')):
                self.max_size = max_size
                self.ptr = 0
                self.size = 0

                self.state = torch.zeros((max_size, state_dim), device=device)
                self.action = torch.zeros((max_size, action_dim), device=device)
                self.next_state = torch.zeros((max_size, state_dim), device=device)
                self.reward = torch.zeros((max_size, 1), device=device)
                self.not_done = torch.zeros((max_size, 1), device=device)
                self.interp_param = torch.ones((max_size, ), device=device) * -1

                self.state_dim = state_dim
                self.action_dim = action_dim
                self.device = device

        def add(self, state, action, next_state, reward, done, interp):
                self.state[self.ptr] = torch.tensor(state, device=self.device)
                self.action[self.ptr] = torch.tensor(action, device=self.device)
                self.next_state[self.ptr] = torch.tensor(next_state, device=self.device)
                self.reward[self.ptr] = torch.tensor(reward, device=self.device)
                self.not_done[self.ptr] = torch.tensor(1. - done, device=self.device)
                self.interp_param[self.ptr] = torch.tensor(interp, device=self.device)

                self.ptr = (self.ptr + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)


        def sample(self, batch_size):
                ind = np.random.randint(0, self.size, size=batch_size)
                return (
                        self.state[ind],
                        self.action[ind],
                        self.next_state[ind],
                        self.reward[ind],
                        self.not_done[ind]
                )


        def clean(self, interp_range):
                print('clean replay buffer')
                print('replay buffer size before cleaning: {}'.format(self.size))

                if self.size == 0:
                    return

                interp_param_flag = torch.logical_and(self.interp_param < interp_range[1], \
                        self.interp_param >= interp_range[0])

                self.state = self.state[interp_param_flag]
                self.action = self.action[interp_param_flag]
                self.next_state = self.next_state[interp_param_flag]
                self.reward = self.reward[interp_param_flag]
                self.not_done = self.not_done[interp_param_flag]
                self.interp_param = self.interp_param[interp_param_flag]

                self.size = torch.sum(interp_param_flag)
                self.ptr = self.size % self.max_size
                print('replay buffer size after cleaning: {}'.format(self.size))

                if self.size < self.max_size:
                    pad = torch.zeros((self.max_size-self.size, self.state_dim), \
                            device=self.device)
                    self.state = torch.cat([self.state, pad], axis=0)

                    pad = torch.zeros((self.max_size-self.size, self.action_dim), \
                            device=self.device)
                    self.action = torch.cat([self.action, pad], axis=0)

                    pad = torch.zeros((self.max_size-self.size, self.state_dim), \
                            device=self.device)
                    self.next_state = torch.cat([self.next_state, pad], axis=0)

                    pad = torch.zeros((self.max_size-self.size, 1), device=self.device)
                    self.reward = torch.cat([self.reward, pad], axis=0)

                    pad = torch.zeros((self.max_size-self.size, 1), device=self.device)
                    self.not_done = torch.cat([self.not_done, pad], axis=0)

                    pad = torch.ones((self.max_size-self.size), device=self.device) * -1
                    self.interp_param = torch.cat([self.interp_param, pad], axis=0)


