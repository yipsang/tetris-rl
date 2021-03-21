from torch import random
from replay_buffer import ReplayBuffer
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import logging
import numpy as np
import random

from dqn import DQN
from replay_buffer import ReplayBuffer


class TrajectoryPlanner:
    def __init__(
        self,
        state_shape,
        action_size,
        buffer_size,
        batch_size,
        gamma,
        lr,
        freeze_step,
        her=True,
        gpu=False,
    ):
        self.state_shape = state_shape
        self.action_size = action_size
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

        self.dqn_local = DQN(
            action_size, input_shape=state_shape[:2], in_channels=state_shape[2] + 1
        ).to(self.device)
        self.dqn_target = DQN(
            action_size, input_shape=state_shape[:2], in_channels=state_shape[2] + 1
        ).to(self.device)

        self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer(buffer_size, gpu=gpu, her=her)

        logging.info(self.dqn_local)

    def act(self, state, goal, eps=0.1):
        _state = np.dstack((state, goal))
        state = torch.from_numpy(_state).float().unsqueeze(0).to(self.device)
        self.dqn_local.eval()
        with torch.no_grad():
            action_values = self.dqn_local(state)
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        return action

    def train_step(self, state, next_state, goal, action, reward, done):
        # There is a chance that the goal is None when the tetris episode ends, because
        # there is no valid move anymore
        if goal is not None:
            self.replay_buffer.add(state, next_state, goal, action, reward, done)
        else:
            # print("goal is None")
            #     print(state[:, :, 0], "\n", state[:, :, 1])
            # print("current episode: ", len(self.replay_buffer.current_episode))
            self.replay_buffer.current_episode = []

        # if there are enough transition in replay buffer, then train the agent
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.n_train_steps += 1
        return self._train(self.replay_buffer.sample(self.batch_size))

    def _train(self, transitions):
        """
        transitions: (Tuple[torch.Tensor]): tuple of (s, s', a, r, done) tuples
        """
        self.optimizer.zero_grad()
        states, next_states, goals, actions, rewards, dones = transitions
        q_targets_next = (
            self.dqn_target(torch.cat((next_states, goals.unsqueeze(3)), dim=3))
            .detach()
            .max(1)[0]
            .unsqueeze(1)
        )
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        q_expcteds = self.dqn_local(
            torch.cat((states, goals.unsqueeze(3)), dim=3)
        ).gather(1, actions)
        loss = F.mse_loss(q_expcteds, q_targets)
        loss.backward()
        clip_grad_norm_(self.dqn_local.parameters(), 1)
        self.optimizer.step()

        if self.n_train_steps % self.freeze_step == 0:
            self._update_frozen_dqn()

        return loss.detach().cpu().numpy()

    def _update_frozen_dqn(self):
        self.dqn_target.load_state_dict(self.dqn_local.state_dict())
