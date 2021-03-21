import torch
import numpy as np
import random
from collections import deque, namedtuple
import logging

Transition = namedtuple(
    "Transition",
    field_names=["state", "next_state", "goal", "action", "reward", "done"],
)


class ReplayBuffer:
    def __init__(self, buffer_size, gpu=False, her=False, replay_k=4):
        self.memory = deque()
        self.buffer_size = buffer_size
        self.gpu = gpu
        self.memory = deque(maxlen=buffer_size)
        self.current_episode = []
        self.her = her
        self.her_prob = replay_k / (1 + replay_k)

    def add(self, state, next_state, goal, action, reward, done):
        self.current_episode.append(
            Transition(state, next_state, goal, action, reward, done)
        )
        if done:
            logging.info(
                "an episode ends! episode size: {}".format(len(self.current_episode))
            )
            for i, transition in enumerate(self.current_episode):
                if not self.her:
                    self.memory.append(transition)
                    continue

                updated_transition = transition
                if random.random() < self.her_prob:
                    new_goal = self.sample_goal(i)
                    updated_transition = updated_transition._replace(goal=new_goal)
                    if np.array_equal(
                        updated_transition.next_state[:, :, 0]
                        + updated_transition.next_state[:, :, 1],
                        new_goal,
                    ):
                        updated_transition = updated_transition._replace(reward=1)
                    else:
                        updated_transition = updated_transition._replace(reward=0)
                self.memory.append(updated_transition)

            self.current_episode = []

    def sample_goal(self, idx):
        transition = random.choice(self.current_episode[idx:])
        # transiiton = self.current_episode[-1]
        return transition.state[:, :, 0] + transition.state[:, :, 1]

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
        goals = (
            torch.from_numpy(np.stack([t.goal for t in transitions])).float().to(device)
        )
        actions = (
            torch.from_numpy(np.vstack([t.action for t in transitions]))
            .long()
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

        return (states, next_states, goals, actions, rewards, dones)

    def __len__(self):
        return len(self.memory)
