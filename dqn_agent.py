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
from state_only_replay_buffer import StateOnlyReplayBuffer


class TetrisDQNAgent:
    def __init__(
        self,
        input_shape,
        buffer_size,
        batch_size,
        gamma,
        lr,
        freeze_step,
        gpu=False,
        layers_size=[128, 256],
        use_conv=False,
        conv_layers_config=[(2, 1, 0, 32), (2, 1, 1, 64), (1, 1, 0, 64)],
        conv_linear_size=128,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.freeze_step = freeze_step
        self.gpu = gpu

        self.n_train_steps = 0

        self.device = "cpu"
        if gpu:
            self.device = "cuda"

        if use_conv:
            self.value_net_local = ConvValueNetwork(
                input_shape[:2],
                input_shape[-1],
                layers_config=conv_layers_config,
                linear_layer_size=conv_linear_size,
            ).to(self.device)
            self.value_net_target = ConvValueNetwork(
                input_shape[:2],
                input_shape[-1],
                layers_config=conv_layers_config,
                linear_layer_size=conv_linear_size,
            ).to(self.device)
        else:
            self.value_net_local = ValueNetwork(
                input_shape[0], layers_size=layers_size
            ).to(self.device)
            self.value_net_target = ValueNetwork(
                input_shape[0], layers_size=layers_size
            ).to(self.device)

        self.optimizer = optim.Adam(self.value_net_local.parameters(), lr=self.lr)

        self.replay_buffer = StateOnlyReplayBuffer(buffer_size, gpu=gpu)

        logging.info(self.value_net_local)

    def act(self, next_states, eps=0.1):
        state = torch.from_numpy(next_states).float().to(self.device)
        if random.random() > eps:
            self.value_net_local.eval()
            with torch.no_grad():
                state_values = self.value_net_local(state)
                action = np.argmax(state_values.cpu().data.numpy())
            self.value_net_local.train()
        else:
            action = random.choice(np.arange(len(next_states)))
        return action

    def train_step(self, state, next_state, reward, done):
        self.replay_buffer.add(state, next_state, reward, done)

        # if there are enough transition in replay buffer, then train the agent
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.n_train_steps += 1
        return self._train(self.replay_buffer.sample(self.batch_size))

    def _train(self, transitions):
        """
        transitions: (Tuple[torch.Tensor]): tuple of (s, s', r, done) tuples
        """
        self.optimizer.zero_grad()
        states, next_states, rewards, dones = transitions
        q_targets_next = self.value_net_target(next_states)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        q_expcteds = self.value_net_local(states)
        loss = F.mse_loss(q_expcteds, q_targets)
        loss.backward()
        clip_grad_norm_(self.value_net_local.parameters(), 1)
        self.optimizer.step()

        if self.n_train_steps % self.freeze_step == 0:
            self._update_frozen_dqn()

        return loss.detach().cpu().numpy()

    def _update_frozen_dqn(self):
        self.value_net_target.load_state_dict(self.value_net_local.state_dict())
