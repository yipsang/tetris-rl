from torch import random
import torch
from torch.nn.modules import linear
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import logging
import numpy as np
import random

from value_network import ValueNetwork, ConvValueNetwork


class ADQNAgent:
    def __init__(
        self,
        input_shape,
        optimizer,
        shared_model,
        shared_model_target,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        freeze_step=5,
        layers_size=[128, 256],
        is_noisy=False,
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.freeze_step = freeze_step
        self.is_noisy = is_noisy

        self.model = ValueNetwork(input_shape[0], layers_size=layers_size, is_noisy=is_noisy)
        self.shared_model = shared_model
        self.shared_model_target = shared_model_target
        self.optimizer = optimizer

    def act(self, next_states, eps=0.1):
        state = torch.from_numpy(next_states).float()
        if self.is_noisy or random.random() > eps:
            self.model.eval()
            with torch.no_grad():
                state_values = self.model(state)
                action = np.argmax(state_values.cpu().data.numpy())
            self.model.train()
        else:
            action = random.choice(np.arange(len(next_states)))
        return action

    def train(self, transitions, T, end_reward):
        """
        transitions: (Tuple[torch.Tensor]): tuple of (s, s', r, done) tuples
        """
        # self.optimizer.zero_grad()
        states, next_states, rewards, dones = transitions

        all_rewards = torch.zeros(len(rewards), 1)
        rewards_sum = end_reward
        for i in reversed(range(len(rewards))):
            R = self.gamma * rewards_sum + rewards[i]
            rewards_sum = R

            all_rewards[i] = R

        q_targets = self.model(states)

        print("Rewards Sum: {}".format(rewards_sum))

        loss = F.mse_loss(all_rewards, q_targets)
        loss.backward()

        clip_grad_norm_(self.model.parameters(), 1)

        self.ensure_shared_grads()
        self.optimizer.step()

        if self.is_noisy:
            self.shared_model.reset_noise()
            self.shared_model_target.reset_noise()

        if T % self.freeze_step == 0:
            self._update_frozen_dqn()

        return loss.detach().cpu().numpy()

    def _update_frozen_dqn(self):
        self.shared_model_target.load_state_dict(self.shared_model.state_dict())

    def sync_model_params(self):
        self.model.load_state_dict(self.shared_model_target.state_dict()) 

    def sync_model_params_local(self):
        self.model.load_state_dict(self.shared_model.state_dict())  
    
    def sync_model_params_target(self):
        self.shared_model_target.load_state_dict(self.shared_model.state_dict())  

    def ensure_shared_grads(self):
        for param, shared_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
