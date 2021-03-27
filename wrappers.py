import random
from Matris.matris import MATRIX_WIDTH
import gym
from gym import spaces
import numpy as np
from collections import deque

COLOR_TO_ROTATION = {
    "blue": 2,
    "yellow": 1,
    "pink": 4,
    "green": 2,
    "red": 2,
    "cyan": 4,
    "orange": 4,
}

COLOR_TO_ID = {
    "blue": 0,
    "yellow": 1,
    "pink": 2,
    "green": 3,
    "red": 4,
    "cyan": 5,
    "orange": 6,
}

COLOR_TO_OHE = {}
for color, id_ in COLOR_TO_ID.items():
    next_tetromino_id = COLOR_TO_ID[color]
    ohe = [0] * len(COLOR_TO_ID.items())
    ohe[next_tetromino_id] = 1
    COLOR_TO_OHE[color] = ohe


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


class FallingPieceFrameStack(gym.Wrapper):
    def __init__(self, env, num_stack):
        super(FallingPieceFrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        self.observation_space = spaces.Box(
            low=0,
            high=0,
            shape=(
                self.observation_space.shape[0],
                self.observation_space.shape[1],
                1 + num_stack,
            ),
            dtype=self.observation_space.dtype,
        )

    def _get_falling_piece_frame(self, observation):
        return observation[:, :, 1]

    def _get_observation(self, observation):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        stacks = [observation[:, :, 0]] + list(self.frames)
        return np.stack(stacks, axis=2)

    def _fill_frames(self, observation):
        for _ in range(self.num_stack):
            self.frames.append(self._get_falling_piece_frame(observation))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if info["round_done"]:
            self._fill_frames(observation)
        else:
            self.frames.append(self._get_falling_piece_frame(observation))
        return self._get_observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        self._fill_frames(observation)
        return self._get_observation(observation), reward, done, info


class PositionAction(gym.Wrapper):
    def __init__(self, env, handcrafted_features=True, with_next_tetromino=True):
        super().__init__(env)
        self.handcrafted_features = handcrafted_features
        self.with_next_tetromino = with_next_tetromino
        state_shape = (20, 10, 1)
        if handcrafted_features:
            state_size = MATRIX_WIDTH * 2 + 1
            if with_next_tetromino:
                state_size += len(COLOR_TO_OHE.items())
            state_shape = (state_size,)
        self.observation_space = spaces.Box(low=0, high=255, shape=state_shape)

    def _get_column_heights_and_holes(self, observation):
        board = observation
        heights = []
        holes = []
        for col in board.T:
            solid_col = np.trim_zeros(col, "f")
            heights.append(solid_col.shape[0])
            holes.append(np.count_nonzero(solid_col == 0))
        return heights, holes

    def _get_complete_lines(self, observation):
        board = observation
        n_complete_lines = 0
        for row in board:
            if np.sum(row) == MATRIX_WIDTH:
                n_complete_lines += 1
        return n_complete_lines

    def _get_n_holes(self, observation):
        board = observation
        n_holes = 0
        for col in board.T:
            solid_col = np.trim_zeros(col, "f")
            n_holes += np.count_nonzero(solid_col == 0)
        return n_holes

    def _get_handcrafted_observation(self, observation):
        next_tetromino = self.env.matris.next_tetromino
        heights, holes = self._get_column_heights_and_holes(observation)
        handcrafted_observation = (
            heights + holes + [self._get_complete_lines(observation)]
        )
        if self.with_next_tetromino:
            handcrafted_observation += COLOR_TO_OHE[next_tetromino.color]
        return np.array(handcrafted_observation)

    def _get_all_next_states(self, matris):
        posY, posX = matris.tetromino_position
        valid_observations = []
        valid_actions = []
        for i in range(COLOR_TO_ROTATION[matris.current_tetromino.color]):
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
                valid_observations.append(
                    self.matrix_to_nparray(
                        matris.blend(shape=shape, position=(testPosY, testPosX))
                    )
                )
                valid_actions.append((i, testPosY, testPosX))
                testPosX -= 1

        if len(valid_observations) == 0:
            return []
        return list(zip(valid_actions, valid_observations))

    def _get_observation(self, observation):
        if self.handcrafted_features:
            return self._get_handcrafted_observation(observation)
        return np.expand_dims(observation, axis=2)

    def _skip_frames(self, frames):
        for _ in range(frames):
            self.env.step(4)

    def reset(self):
        obs, reward, done, info = self.env.reset()

        all_next_states = self._get_all_next_states(self.matris)

        if self.handcrafted_features:
            all_next_states_ = []
            for action, observation in all_next_states:
                all_next_states_.append(
                    (action, self._get_handcrafted_observation(observation))
                )
            all_next_states = all_next_states_
        else:
            all_next_states_ = []
            for action, observation in all_next_states:
                all_next_states_.append((action, np.expand_dims(observation, axis=2)))
            all_next_states = all_next_states_

        return self._get_observation(obs[:, :, 0]), all_next_states, reward, done, info

    def step(self, action):
        """
        action Tuple(int, int, int): 1st entry is the number of rotation needed.
        The 2nd and 3rd entries are valid target position (y, x) for the falling piece
        """
        rotation, y, x = action
        cur_position = self.matris.tetromino_position
        for _ in range(rotation):
            obs, reward, done, info = self.env.step(0)
            # if it's done only return the final state
            if done:
                return (
                    self._get_observation(obs[:, :, 0]),
                    reward,
                    done,
                    info,
                )
        # dy = y - cur_position[0]
        dx = x - cur_position[1]
        for _ in range(abs(dx)):
            if dx > 0:
                obs, reward, done, info = self.env.step(2)
                if done:
                    return (
                        self._get_observation(obs[:, :, 0]),
                        reward,
                        done,
                        info,
                    )
            else:
                obs, reward, done, info = self.env.step(1)
                if done:
                    return (
                        self._get_observation(obs[:, :, 0]),
                        reward,
                        done,
                        info,
                    )
        obs, reward, done, info = self.env.step(3)
        if done:
            return (
                self._get_observation(obs[:, :, 0]),
                reward,
                done,
                info,
            )

        all_next_states = self._get_all_next_states(self.matris)

        if len(all_next_states) == 0:
            raise Exception("no possible next state")

        if self.handcrafted_features:
            all_next_states_ = []
            for action, observation in all_next_states:
                all_next_states_.append(
                    (action, self._get_handcrafted_observation(observation))
                )
            all_next_states = all_next_states_
        else:
            all_next_states_ = []
            for action, observation in all_next_states:
                all_next_states_.append((action, np.expand_dims(observation, axis=2)))
            all_next_states = all_next_states_

        return all_next_states, reward, done, info
