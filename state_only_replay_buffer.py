import torch
import numpy as np
import random
from collections import deque, namedtuple
import logging

Transition = namedtuple(
    "Transition",
    field_names=["state", "next_state", "reward", "done"],
)


class StateOnlyReplayBuffer:
    def __init__(self, buffer_size, gpu=False):
        self.memory = deque()
        self.buffer_size = buffer_size
        self.gpu = gpu
        self.memory = deque(maxlen=buffer_size)
        self.current_episode = []

    def add(self, state, next_state, reward, done):
        self.current_episode.append(Transition(state, next_state, reward, done))
        if done:
            logging.info(
                "an episode ends! episode size: {}".format(len(self.current_episode))
            )
            for i, transition in enumerate(self.current_episode):
                self.memory.append(transition)
                continue

            self.current_episode = []

    def sample(self, batch_size):
        device = "cpu"
        if self.gpu:
            device = "cuda"

        transitions = random.sample(self.memory, batch_size)
        states = (
            torch.from_numpy(np.stack([t.state for t in transitions]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(np.stack([t.next_state for t in transitions]))
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.vstack([t.reward for t in transitions]))
            .long()
            .to(device)
        )
        dones = (
            torch.from_numpy(np.vstack([t.done for t in transitions])).int().to(device)
        )

        return (states, next_states, rewards, dones)

    def __len__(self):
        return len(self.memory)
