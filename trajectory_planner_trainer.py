import torch
import gym
from gym_matris import *
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from trajectory_planner import TrajectoryPlanner
from wrappers import GoalConditionedReward


class TrajectorPlannerTrainer:
    """
    episodes: number of episodes to train, the episode here means each falling piece round in tetris
    """

    def __init__(
        self,
        state_shape=(20, 10, 2),
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        freeze_step=5,
        gpu=False,
        episodes=10000,
        render=False,
    ):
        self.episodes = episodes
        env = gym.make("matris-v0", render=render, timestep=10)
        env = GoalConditionedReward(env)
        self.env = env
        self.render = render
        self.writer = SummaryWriter()
        self.trajector_planner = TrajectoryPlanner(
            state_shape,
            env.action_space.n,
            buffer_size,
            batch_size,
            gamma,
            lr,
            freeze_step,
            gpu,
        )

    def start(self):
        logging.info("start training...")
        self.losses = []
        self.rewards = []

        state, reward, done, info = self.env.reset()
        for i in range(self.episodes):
            logging.info("training at episode {} =================".format(i))
            # although it's called episode losses, the q network is actually trained randomly
            # on the transitions in replay buffer
            episode_losses = []
            episode_rewards = []
            while True:
                action = self.trajector_planner.act(state, info["goal"])
                next_state, reward, done, info = self.env.step(action)
                if not done and self.render:
                    self.env.render()
                # else:
                #     print(state[:, :, 0])
                loss = self.trajector_planner.train_step(
                    state, next_state, info["goal"], action, reward, info["round_done"]
                )
                state = next_state
                if loss is not None:
                    episode_losses.append(loss)
                episode_rewards.append(reward)

                if info["round_done"] or done:
                    logging.info("an episode ends")
                    break

            if done:
                logging.info("a tetris episode ends")
                state, reward, done, info = self.env.reset()

            self.log_stat(i, episode_losses, episode_rewards)

            self.losses += episode_losses
            self.rewards += episode_rewards
            episode_losses = []
            episode_rewards = []

        torch.save(
            self.trajector_planner.dqn_frozen.state_dict(),
            "trajector_planner_{}.model".format(
                datetime.now().strftime("%d_%m_%Y_%H_%M")
            ),
        )
        return (self.losses, self.rewards)

    def log_stat(self, i, episode_losses, episode_rewards):
        num_steps = len(episode_rewards)
        mean_loss = sum(episode_losses) / max(len(episode_losses), 1)
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        last_reward = episode_rewards[-1]
        overall_mean_reward = sum(self.rewards) / len(self.rewards)

        self.writer.add_scalar("Number of steps/train", num_steps, i)
        self.writer.add_scalar("Mean loss/train", mean_loss, i)
        self.writer.add_scalar("Mean reward/train", mean_reward, i)
        self.writer.add_scalar("Last step reward/train", last_reward, i)

        logging.info("number of steps: {}".format(num_steps))
        logging.info("mean loss: {}".format(mean_loss))
        logging.info("mean reward: {}".format(mean_reward))
        logging.info("last reward: {}".format(last_reward))
        logging.info("overal mean reward {}".format(overall_mean_reward))