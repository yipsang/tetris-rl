import torch
import torch.nn.functional as F
import random
import numpy as np

from wrappers import PositionAction
import gym
from gym_matris import *
from value_network import ValueNetwork
from adqn_agent import ADQNAgent

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, optimizer):
    env = gym.make("matris-v0", render=False, timestep=0.05)
    env = PositionAction(env, handcrafted_features=True)

    agent = ADQNAgent(env.observation_space.shape, optimizer, shared_model)

    state, actions_and_next_states, reward, done, info = env.reset()
    # state = torch.from_numpy(state)

    agent.sync_model_params()

    episode_steps = 100
    render = False
    random_action_episodes = 100
    eps = 1
    start_eps = 1
    end_eps = 0.01
    eps_interval = start_eps - end_eps
    epsilon_decay_episodes = 2000

    done = False
    episode_num = 0
    while True:
        episode_num += 1
        
        prev_states = []
        prev_next_states = []
        prev_rewards = []
        prev_dones = []

        episode_length = 0
        while not done and episode_length < episode_steps:
            episode_length += 1

            with lock:
                counter.value += 1
                T = counter.value

            actions, next_states = zip(*actions_and_next_states)

            # next_states_n = torch.from_numpy(next_states).float()
            action_idx = agent.act(np.array(next_states), eps=eps)

            action = actions[action_idx]
            actions_and_next_states, reward, done, info = env.step(action)

            if not done and render:
                env.render()
            
            if done:
                next_state = actions_and_next_states
            else:
                next_state = next_states[action_idx]

            prev_states.append(state)
            prev_next_states.append(next_state)
            prev_rewards.append(reward)
            prev_dones.append(done)

            state = next_state
        
        # episode has ended
        print("EPISODE COMPLETE - Length: {}".format(episode_length))

        if episode_num < random_action_episodes:
            eps = start_eps
        else:
            eps -= eps_interval / epsilon_decay_episodes
            eps = max(end_eps, eps)
        
        f_states = (torch.from_numpy(np.stack([x for x in prev_states])).float())
        f_next_states = (torch.from_numpy(np.stack([x for x in prev_next_states])).float())
        f_rewards = (torch.from_numpy(np.vstack([x for x in prev_rewards])).long())
        f_dones = (torch.from_numpy(np.vstack([x for x in prev_dones])).int())

        to_train = (f_states, f_next_states, f_rewards, f_dones)
        loss = agent.train(to_train, episode_num)

        state, actions_and_next_states, reward, done, info = env.reset()
