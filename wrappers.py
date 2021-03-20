import random
import gym
import numpy as np


class GoalConditionedReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.prev_num_tetrominoes = None
        self.current_goal = None

    def _random_sample_goal(self, matris):
        posY, posX = matris.tetromino_position
        valid_matrices = []
        for i in range(4):
            shape = matris.rotated(i)
            maxPosX = posX
            while matris.blend(shape=shape, position=(posY, maxPosX)):
                maxPosX += 1
            maxPosX -= 1
            testPosX = maxPosX
            while matris.blend(shape=shape, position=(posY, testPosX)):
                testPosY = posY + 1
                while matris.blend(shape=shape, position=(testPosY, testPosX)):
                    testPosY += 1
                # the last position y is an invalid one, roll 1 step back
                testPosY -= 1
                valid_matrices.append(
                    matris.blend(shape=shape, position=(testPosY, testPosX))
                )
                testPosX -= 1

        if len(valid_matrices) == 0:
            return None
        goal_matrix = random.choice(valid_matrices)
        return self.matrix_to_nparray(goal_matrix)

    def reset(self):
        state, _, done, info = self.env.reset()
        self.prev_num_tetrominoes = info["matris"].num_tetrominoes
        goal = self._random_sample_goal(info["matris"])
        self.current_goal = goal
        info["goal"] = goal
        if np.array_equal(state[:, :, 0], goal):
            reward = 1
        else:
            reward = 0
        # a round ends when there is new falling piece
        info["round_done"] = False
        return state, reward, done, info

    def step(self, action):
        state, _, done, info = self.env.step(action)
        if np.array_equal(state[:, :, 0], self.current_goal):
            reward = 1
        else:
            reward = 0
        info["round_done"] = False
        info["goal"] = self.current_goal
        if self.prev_num_tetrominoes != info["matris"].num_tetrominoes:
            # if it's none, then it's the 1st step, no need to set round_done to True
            if self.prev_num_tetrominoes is not None:
                info["round_done"] = True
            self.prev_num_tetrominoes = info["matris"].num_tetrominoes
            goal = self._random_sample_goal(info["matris"])
            self.current_goal = goal
            info["goal"] = goal
        return state, reward, done, info
