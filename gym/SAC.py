import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import pickle

import utils


# Implementation of Soft Actor-Critic (SAC)
# Paper: https://arxiv.org/abs/1801.01290


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
                # mean
		self.l3 = nn.Linear(256, action_dim)
                # std
		self.l4 = nn.Linear(256, action_dim)


	def forward(self, state):
                epsilon = 1e-6
                a = F.relu(self.l1(state))
                a = F.relu(self.l2(a))
                mean = self.l3(a)
                logstd = self.l4(a)

                logstd = torch.clamp(logstd, -30, 30) # prevent NaN
                std = logstd.exp()
                std = torch.clamp(std, -1e2, 1e2) # prevent NaN

                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()
                x_t = torch.clamp(x_t, -1e2, 1e2) # prevent NaN
                y_t = torch.tanh(x_t)
                action = y_t

                log_prob = normal.log_prob(x_t)
                prob = F.relu(1 - y_t.pow(2)) + epsilon
                log_prob -= torch.log(prob)
                log_prob = log_prob.sum(1, keepdim=True)
                mean = torch.tanh(mean)

                return action, log_prob, mean


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


class SAC(object):
        def __init__(
                self,
                state_dim,
                action_dim,
                max_action,
                batch_size=256,
                discount=0.99,
                tau=0.005,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=1,
                actor_lr=3e-4,
                critic_lr=3e-4,
                temp_lr=3e-4,
                alpha=0.2,
                target_entropy=None,
                device=torch.device('cuda'),
	):

                self.device = device
                self.actor_lr = actor_lr
                self.critic_lr = critic_lr
                self.temp_lr = temp_lr
                self.discount = discount
                self.tau = tau
                self.alpha = alpha
                self.policy_noise = policy_noise
                self.noise_clip = noise_clip
                self.policy_freq = policy_freq
                self.target_entropy = target_entropy if target_entropy else -action_dim

                self.total_it = 0

                # actor
                self.actor = Actor(state_dim, action_dim).to(self.device)
                self.actor_target = copy.deepcopy(self.actor)
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

                # double critic
                self.critic = Critic(state_dim, action_dim).to(self.device)
                self.critic_target = copy.deepcopy(self.critic)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
                self.critic_target.eval()
                for p in self.critic_target.parameters():
                    p.requires_grad = False


        def select_action(self, state, sample_noise=None):
                with torch.no_grad():
                    action, _, mean = self.actor(torch.Tensor(state).view(1,-1).to(self.device))
                if sample_noise is None:
                    return mean.squeeze().cpu().numpy()
                else:
                    return np.atleast_1d(action.squeeze().cpu().numpy())


        def train(self, train_tuple, state_filter=None):
            q1_loss, q2_loss, pi_loss, a_loss = 0, 0, 0, 0

            state, action, next_state, reward, not_done = train_tuple
            done = 1 - not_done

            with torch.no_grad():
                next_action, logprobs, _ = self.actor(next_state)
                q_t1, q_t2 = self.critic_target(next_state, next_action)
                q_target = torch.min(q_t1, q_t2) - self.alpha * logprobs
                next_q = reward + (1.0 - done) * self.discount * q_target

            q1, q2 = self.critic(state, action)
            q1_loss = F.mse_loss(q1, next_q)
            q2_loss = F.mse_loss(q2, next_q)
            q_loss = q1_loss + q2_loss

            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()

            pi, logprobs, _ = self.actor(state)
            q_1, q_2 = self.critic(state, pi)
            q_val = torch.min(q_1, q_2)
            policy_loss = (self.alpha * logprobs - q_val).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = torch.tensor(0.).to(self.device)

            if self.total_it % self.policy_freq == 0:
                with torch.no_grad():
                    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


        def save(self, filename):
                torch.save(self.critic.state_dict(), filename + "_critic")
                torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

                torch.save(self.actor.state_dict(), filename + "_actor")
                torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


        def load(self, filename, load_optim=True):
                self.critic.load_state_dict(torch.load(filename + "_critic"))
                if load_optim:
                    self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
                self.critic_target = copy.deepcopy(self.critic)

                self.actor.load_state_dict(torch.load(filename + "_actor"))
                if load_optim:
                    self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
                self.actor_target = copy.deepcopy(self.actor)

