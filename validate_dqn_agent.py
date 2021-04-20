import sys
import torch
import gym
from gym_matris import *
import numpy as np

from value_network import ConvValueNetwork, ValueNetwork
from wrappers import PositionAction

MODEL_PATH = "trained_models/dqn_27_03_2021_23_21/dqn_8500_12000.model"

GPU_ENABLED = False
RAW_IMAGE = True 
IS_NOISY = True

TRIAL_EPISODES = 100
MAX_STEP = 500


def main():
    env = gym.make("matris-v0", render=False, timestep=0.05)
    env = PositionAction(
        env, handcrafted_features=not RAW_IMAGE, with_next_tetromino=True
    )

    device = "cpu"
    if GPU_ENABLED:
        device = "cuda"

    print("loading model...")
    if not RAW_IMAGE:
        value_net = ValueNetwork(
            env.observation_space.shape[0], layers_size=[128, 256], is_noisy=IS_NOISY
        ).to(device)
        value_net.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device(device))
        )
    else:
        input_shape = env.observation_space.shape
        value_net = ConvValueNetwork(
            input_shape[:2],
            input_shape[-1],
            layers_config=[(2, 1, 0, 32), (2, 2, 1, 32), (1, 1, 0, 32)],
            linear_layer_size=256,
            is_noisy=True,
        ).to(device)
        value_net.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device(device))
        )
    value_net.eval()

    steps = []
    rewards = []
    for i in range(TRIAL_EPISODES):
        sys.stdout.write("\033[K")
        print(
            "running tests: {}%".format(round((i + 1) / TRIAL_EPISODES, 2) * 100),
            end="\r",
        )
        step_count = 0
        total_reward = 0
        state, actions_and_next_states, reward, done, info = env.reset()
        while step_count < MAX_STEP and not done:
            step_count += 1
            actions, next_states = zip(*actions_and_next_states)
            next_states = torch.from_numpy(np.array(next_states)).float().to(device)
            state_values = value_net(next_states)

            action_idx = np.argmax(state_values.cpu().data.numpy())
            action = actions[action_idx]
            state = next_states[action_idx]

            actions_and_next_states, reward, done, info = env.step(action)
            total_reward += reward

        steps.append(step_count)
        rewards.append(total_reward)

    print("")
    steps = np.array(steps)
    rewards = np.array(rewards)
    print("Avg reward:", np.mean(rewards), "| std:", np.std(rewards))
    print("Avg steps:", np.mean(steps), "| std:", np.std(steps))

    env.close()


if __name__ == "__main__":
    main()