import time
from collections import deque

import torch
import torch.nn.functional as F

from actor_critic_value_network import ActorCriticValueNetwork


def test(rank, args, shared_model, counter):
    env = gym.make("matris-v0", render=False, timestep=0.05)
    env = PositionAction(env, handcrafted_features=True)

    model = ActorCriticValueNetwork(env.observation_space.shape[0], env.action_space.n)

    model.eval()

    max_episode_length = 100

    state, actions_and_next_states, reward, done, info = env.reset()
    state = torch.from_numpy(state)

    reward_sum = 0

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0

    while True:
        episode_length += 1
        # Sync with the shared model

        if done:
            model.load_state_dict(shared_model.state_dict())

        with torch.no_grad():
            value, logit = model(state)
        
        prob = F.softmax(logit, dim=-1)
        action_idx = prob.max(1, keepdim=True)[1].numpy()

        action = actions[action_idx]
        actions_and_next_states, reward, done, info = self.env.step(action)

        done = done or episode_length >= max_episode_length
        reward_sum += reward

        if done:
            next_state = actions_and_next_states
        else:
            next_state = next_states[action_idx]
        state = next_state

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state, actions_and_next_states, reward, done, info = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)