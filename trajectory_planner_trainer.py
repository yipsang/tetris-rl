import torch
import gym
from gym_matris import *
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from trajectory_planner import TrajectoryPlanner
from wrappers import GoalConditionedReward, FallingPieceFrameStack


class TrajectorPlannerTrainer:
    """
    episodes: number of episodes to train, the episode here means each falling piece round in tetris
    """

    def __init__(
        self,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        freeze_step=5,
        gpu=False,
        episodes=10000,
        render=False,
        her=True,
        random_action_frames=1000,
        episolon_decay_frames=10000,
        start_eps=1,
        end_eps=0.1,
        n_stack_frames=4,
        double_dqn=True,
    ):
        self.episodes = episodes
        env = gym.make("matris-v0", render=render, timestep=10)
        env = GoalConditionedReward(env)
        env = FallingPieceFrameStack(env, n_stack_frames)
        self.env = env
        self.render = render
        self.random_action_frames = random_action_frames
        self.episolon_decay_frames = episolon_decay_frames
        self.eps = start_eps
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.writer = SummaryWriter()
        self.trajector_planner = TrajectoryPlanner(
            env.observation_space.shape,
            env.action_space.n,
            buffer_size,
            batch_size,
            gamma,
            lr,
            freeze_step,
            her,
            gpu,
            double_dqn,
        )

    def start(self):
        logging.info("start training...")
        self.losses = []
        self.rewards = []

        state, reward, done, info = self.env.reset()
        frame_count = 0
        eps_interval = self.start_eps - self.end_eps
        for i in range(self.episodes):
            logging.info("training at episode {} =================".format(i))
            # although it's called episode losses, the q network is actually trained randomly
            # on the transitions in replay buffer
            episode_losses = []
            episode_rewards = []
            while True:
                frame_count += 1
                if frame_count < self.random_action_frames:
                    self.eps = self.start_eps
                else:
                    self.eps -= eps_interval / self.episolon_decay_frames
                    self.eps = max(self.end_eps, self.eps)
                action = self.trajector_planner.act(state, info["goal"], eps=self.eps)
                next_state, reward, done, info = self.env.step(action)
                if not done and self.render:
                    self.env.render()
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

            self.losses += episode_losses
            self.rewards.append(episode_rewards[-1])
            self.log_stat(i, episode_losses, episode_rewards)
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
        last_100_mean_reward = sum(self.rewards[-100:]) / len(self.rewards[-100:])

        self.writer.add_scalar("Number of steps/train", num_steps, i)
        self.writer.add_scalar("Mean loss/train", mean_loss, i)
        self.writer.add_scalar("Mean reward/train", mean_reward, i)
        self.writer.add_scalar("Last step reward/train", last_reward, i)

        logging.info("number of steps: {}".format(num_steps))
        logging.info("mean loss: {}".format(mean_loss))
        logging.info("mean reward: {}".format(mean_reward))
        logging.info("last reward: {}".format(last_reward))
        logging.info("Last 100 episodes mean reward {}".format(last_100_mean_reward))
