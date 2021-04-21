import torch
import gym
from gym_matris import *
import numpy as np
import cv2

from value_network import ConvValueNetwork, ValueNetwork
from wrappers import PositionAction

MODEL_PATH = "trained_models/dqn_27_03_2021_01_21/dqn_6500_10000.model"
GPU_ENABLED = False
RECORD = True
VIDEO_PATH = "dqn.mp4"
RAW_IMAGE = False
IS_NOISY = True

MAX_STEP = 10000


def main():
    env = gym.make("matris-v0", render=True, timestep=0.05)
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

    state, actions_and_next_states, reward, done, info = env.reset()

    step_count = 0
    video_stream = None
    while step_count <= MAX_STEP:
        step_count += 1
        actions, next_states = zip(*actions_and_next_states)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        state_values = value_net(next_states)
        action_idx = np.argmax(state_values.cpu().data.numpy())
        action = actions[action_idx]
        actions_and_next_states, reward, done, info = env.step(action)
        if not done:
            frame = env.render()
            if RECORD:
                if not video_stream:
                    width, height, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_stream = cv2.VideoWriter(
                        VIDEO_PATH, fourcc, 15, (width, height)
                    )

                video_stream.write(frame.transpose(1, 0, 2))
        else:
            break

    print(step_count)
    env.close()


if __name__ == "__main__":
    main()