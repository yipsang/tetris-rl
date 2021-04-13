import torch
import torch.nn.functional as F

from wrappers import PositionAction
import gym
from gym_matris import *
from actor_critic_value_network import ActorCriticValueNetwork

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, optimizer):
    env = gym.make("matris-v0", render=False, timestep=0.05)
    env = PositionAction(env, handcrafted_features=True)

    model = ActorCriticValueNetwork(env.observation_space.shape[0], env.action_space.n)

    model.train()

    state, actions_and_next_states, reward, done, info = env.reset()
    state = torch.from_numpy(state)

    max_episode_length = 100
    gamma = 0.8
    num_steps = 100
    gae_lambda = 0.8
    entropy_coef = 0.1
    value_loss_coef = 0.1
    max_grad_norm = 10000
    render = False

    done = False
    while True:
        model.load_state_dict(shared_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        entropies = []

        episode_length = 0
        while not done and episode_length < max_episode_length:
            episode_length += 1

            with lock:
                counter.value += 1
            
            actions, next_states = zip(*actions_and_next_states)

            value, logit = model(state)
            # action_idx = np.argmax(state_values.cpu().data.numpy())
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)

            mults = log_prob * prob
            entropy = -(log_prob * prob).sum(dim=-1, keepdim=True)
            entropies.append(entropy)

            action_idx = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob[action_idx]

            action = actions[action_idx]
            actions_and_next_states, reward, done, info = env.step(action)

            done = done or episode_length >= max_episode_length
            # reward = max(min(reward, 1), -1)

            if done:
                next_state = actions_and_next_states
            else:
                next_state = next_states[action_idx]
            state = next_state

            if not done and render:
                env.render()
            
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
        
        # episode has ended
        state, actions_and_next_states, reward, done, info = env.reset()
        state = torch.from_numpy(state)
        
        R = torch.zeros(1, 1)

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + gamma * values[i + 1] - values[i]
            gae = gae * gamma * gae_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - entropy_coef * entropies[i]

        optimizer.zero_grad()

        to_backward = policy_loss + value_loss_coef * value_loss
        to_backward.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        print("NEW EPISODE COMPLTED - Policy Loss: {}, Value Loss: {}".format(policy_loss, value_loss))




            

