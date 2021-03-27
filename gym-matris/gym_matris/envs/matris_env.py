import os
import gym
from gym import spaces

import pygame
from pygame import Surface

import numpy as np

from Matris.matris import (
    Game,
    GameOver,
    Matris,
    WIDTH,
    HEIGHT,
    MATRIX_HEIGHT,
    MATRIX_WIDTH,
    BLOCKSIZE,
    BORDERWIDTH,
    VISIBLE_MATRIX_HEIGHT,
    BORDERCOLOR,
    MATRIS_OFFSET,
)

HIDDEN_HEIGHT = 2

ACTION_EVENTS = [pygame.K_UP, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, None]


class GymGame(Game):
    def __init__(self):
        super(GymGame, self).__init__()

    def init_screen(self, screen):
        pygame.init()
        self.screen = screen
        self.matris = Matris(screen)

        matris_border = Surface(
            (
                MATRIX_WIDTH * BLOCKSIZE + BORDERWIDTH * 2,
                VISIBLE_MATRIX_HEIGHT * BLOCKSIZE + BORDERWIDTH * 2,
            )
        )
        matris_border.fill(BORDERCOLOR)
        screen.blit(matris_border, (MATRIS_OFFSET, MATRIS_OFFSET))


class MatrisEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False, timestep=1000):
        super(MatrisEnv, self).__init__()
        if not render:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        self.game = GymGame()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.game.init_screen(self.screen)
        self.clock = pygame.time.Clock()
        self.timestep = timestep
        self.prev_lines_cleared = 0
        self.action_space = spaces.Discrete(len(ACTION_EVENTS))
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(VISIBLE_MATRIX_HEIGHT, MATRIX_WIDTH, 2),
            dtype=np.uint8,
        )
        self.empty_matrix = self._create_empty_matrix()

    @property
    def matris(self):
        return self.game.matris

    @property
    def info(self):
        return {"lines_cleared": self.matris.lines, "matris": self.matris}

    def matrix_to_nparray(self, matrix):
        plane = np.zeros((VISIBLE_MATRIX_HEIGHT, MATRIX_WIDTH), dtype=np.uint8)
        for pos, value in matrix.items():
            real_pos = (pos[0] - HIDDEN_HEIGHT, pos[1])
            if value is not None and real_pos[0] >= 0:
                plane[real_pos] = 1
        return plane

    def _create_empty_matrix(self):
        matrix = dict()
        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                matrix[(y, x)] = None
        return matrix

    def _convert_board_to_nparray(self, matris):
        # popupate the matrix without the current falling piece
        fixed_board = self.matrix_to_nparray(matris.matrix)

        # populate the falling piece plane
        # blend with an empty matrix, so that only the current piece remains
        cur_piece_matrix = matris.blend(matrix=self.empty_matrix)
        if not cur_piece_matrix:
            cur_piece_board = np.zeros(
                (VISIBLE_MATRIX_HEIGHT, MATRIX_WIDTH), dtype=np.uint8
            )
        else:
            cur_piece_board = self.matrix_to_nparray(cur_piece_matrix)

        return np.stack((fixed_board, cur_piece_board), axis=2)

    def reset(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.game.init_screen(self.screen)

        return self._convert_board_to_nparray(self.matris), 0, False, self.info

    def step(self, action):
        action_event = ACTION_EVENTS[action]
        if action_event:
            # simulate the key press action
            events = [
                pygame.event.Event(
                    pygame.KEYDOWN, key=action_event, mod=pygame.KMOD_NONE
                ),
                pygame.event.Event(
                    pygame.KEYUP, key=action_event, mod=pygame.KMOD_NONE
                ),
            ]
            for e in events:
                pygame.event.post(e)
        done = False
        try:
            self.matris.update(self.timestep)
        except GameOver:
            done = True

        line_cleared = self.matris.lines - self.prev_lines_cleared
        reward = 1 + line_cleared ** 2 * 2
        self.prev_lines_cleared = self.matris.lines
        if done:
            reward -= 2
        return (
            self._convert_board_to_nparray(self.matris),
            reward,
            done,
            self.info,
        )

    def render(self, mode="human"):
        self.game.redraw()

    def close(self):
        return
