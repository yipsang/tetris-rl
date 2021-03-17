import gym
from gym_matris import *

import cv2
from PIL import Image
import numpy as np

MAX_STEP = 1000

block_size = 30
fps = 60
output = "tetris.mp4"


def convert_board_state_to_img(board, width, height):
    imgs = []
    for i in range(board.shape[-1]):
        plane = Image.fromarray(board[:, :, i] * 255, "L")
        plane = plane.resize(
            (width * block_size, height * block_size), resample=Image.NEAREST
        )
        imgs.append(plane)
    imgs.append(
        Image.fromarray(np.zeros((height * block_size, width * block_size)), "L")
    )
    img = Image.merge("RGB", imgs)
    img = np.array(img)
    return img


if __name__ == "__main__":
    env = gym.make("matris-v0", timestep=10)

    video_stream = None
    for _ in range(MAX_STEP):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if done:
            break

        if not video_stream:
            height, width, _ = state.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_stream = cv2.VideoWriter(
                output, fourcc, fps, (width * block_size, height * block_size)
            )

        video_stream.write(convert_board_state_to_img(state, width, height))

        env.render()

    env.close()
