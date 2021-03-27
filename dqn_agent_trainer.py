import os
from numpy.lib.npyio import save
import torch
import gym
from gym_matris import *
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dqn_agent import TetrisDQNAgent
from wrappers import PositionAction


class TetrisDQNAgentTrainer:
    """
    episodes: number of episodes to train, the episode here means each falling piece round in tetris
    """

    def __init__(
        self,
        layers_size=[128, 256],
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        freeze_step=5,
        gpu=False,
        episodes=10000,
        render=False,
        random_action_episodes=100,
        episolon_decay_episodes=2000,
        start_eps=1,
        end_eps=0.01,
        max_episode_step=1000,
        save_every=100,
    ):
        self.episodes = episodes
        env = gym.make("matris-v0", render=render, timestep=0.05)
        env = PositionAction(env)
        self.env = env
        self.render = render
        self.random_action_episodes = random_action_episodes
        self.episolon_decay_episodes = episolon_decay_episodes
        self.eps = start_eps
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.max_episode_step = max_episode_step
        self.writer = SummaryWriter()
        self.save_every = save_every
        self.tetris_dqn_agent = TetrisDQNAgent(
            env.observation_space.shape[0],
            buffer_size,
            batch_size,
            gamma,
            lr,
            freeze_step,
            gpu,
            layers_size,
        )

    def start(self):
        logging.info("start training...")
        save_dir = "trained_models/dqn_{}/".format(
            datetime.now().strftime("%d_%m_%Y_%H_%M")
        )
        self.losses = []
        self.rewards = []

        state, actions_and_next_states, reward, done, info = self.env.reset()
        frame_count = 0
        eps_interval = self.start_eps - self.end_eps
        for i in range(self.episodes):
            logging.info("training at episode {} =================".format(i))
            episode_losses = []
            episode_rewards = []
            step_count = 0
            while not done and step_count < self.max_episode_step:
                step_count += 1
                frame_count += 1
                actions, next_states = zip(*actions_and_next_states)
                action_idx = self.tetris_dqn_agent.act(
                    np.array(next_states), eps=self.eps
                )
                action = actions[action_idx]
                actions_and_next_states, reward, done, info = self.env.step(action)
                if not done and self.render:
                    self.env.render()
                if done:
                    next_state = actions_and_next_states
                else:
                    next_state = next_states[action_idx]
                loss = self.tetris_dqn_agent.train_step(
                    state,
                    next_state,
                    reward,
                    done or step_count >= self.max_episode_step,
                )
                state = next_state
                if loss is not None:
                    episode_losses.append(loss)
                episode_rewards.append(reward)

            logging.info("an episode ends")
            (
                state,
                actions_and_next_states,
                reward,
                done,
                info,
            ) = self.env.reset()

            if i < self.random_action_episodes:
                self.eps = self.start_eps
            else:
                self.eps -= eps_interval / self.episolon_decay_episodes
                self.eps = max(self.end_eps, self.eps)

            self.losses += episode_losses
            self.rewards.append(episode_rewards[-1])
            self.log_stat(i, episode_losses, episode_rewards)
            episode_losses = []
            episode_rewards = []

            if i % self.save_every == 0:
                logging.info("Saving model...")
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                torch.save(
                    self.tetris_dqn_agent.value_net_local.state_dict(),
                    save_dir + "dqn_{}_{}.model".format(i, self.episodes),
                )

        torch.save(
            self.tetris_dqn_agent.value_net_local.state_dict(),
            save_dir + "dqn_final.model",
        )
        return (self.losses, self.rewards)

    def log_stat(self, i, episode_losses, episode_rewards):
        num_steps = len(episode_rewards)
        mean_loss = sum(episode_losses) / max(len(episode_losses), 1)
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        last_episode_rewards = sum(episode_rewards)

        self.writer.add_scalar("Number of steps/train", num_steps, i)
        self.writer.add_scalar("Mean loss/train", mean_loss, i)
        self.writer.add_scalar("Mean reward/train", mean_reward, i)
        self.writer.add_scalar("Last episode rewards/train", last_episode_rewards, i)

        logging.info("number of steps: {}".format(num_steps))
        logging.info("mean loss: {}".format(mean_loss))
        logging.info("mean reward: {}".format(mean_reward))
        logging.info("last episode rewards: {}".format(last_episode_rewards))
