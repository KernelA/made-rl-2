from typing import Optional
import random
from collections import namedtuple
from abc import ABC, abstractmethod
import logging
import json
import sys

import numpy as np
from tictac_rl.contstants import EMPTY_STATE

from tictac_rl.env.tictac import DRAW

from ..utils import QTableDict
from ..env import TicTacToe, CIRCLE_PLAYER, CROSS_PLAYER, simulate, CallbackInfo
from ..policies import BasePolicy, EpsilonGreedyPolicy

from ..stat import StreamignMean
from ..utils import GameStat, compute_game_stat

TDLearningRes = namedtuple(
    "TDLearningRes", ["mean_reward", "test_cross_win", "test_circle_win", "test_draw", "test_episode_num"])


class TDLearning(ABC):
    def __init__(self, env: TicTacToe,
                 cross_policy: BasePolicy,
                 circle_policy: BasePolicy,
                 generator: random.Random = None,
                 **kwargs):
        self._env = env
        self._cross_policy = cross_policy
        self._circle_policy = circle_policy
        self._is_learning = kwargs["is_learning"]
        self._logger = logging.getLogger("simulation")

        assert self._circle_policy is not self._cross_policy

        self._q_player = CROSS_PLAYER

        # if "cross" in self._path_to_state_file and isinstance(self._circle_policy, EpsilonGreedyPolicy):
        #     raise ValueError("State file does not correspond Q-player")

        # if "circle" in self._path_to_state_file and isinstance(self._cross_policy, EpsilonGreedyPolicy):
        #     raise ValueError("State file does not correspond Q-player")

        if isinstance(self._circle_policy, EpsilonGreedyPolicy):
            self._q_player = CIRCLE_PLAYER
            self._q_table = self._circle_policy.q_function
        else:
            self._q_table = self._cross_policy.q_function

        if isinstance(self._cross_policy, EpsilonGreedyPolicy) and\
                isinstance(self._circle_policy, EpsilonGreedyPolicy) and self._is_learning:
            raise ValueError(
                "Both player are based on QTable please specify other policy for one the him")

        if generator is None:
            generator = random.Random()

        self._generator = generator

    def _transform_reward(self, reward: Optional[int]) -> float:
        """Map -1 0 1 to 0 0.5 1
        """
        if reward is None:
            reward = 0

        return 0.5 * (self._q_player * reward + 1)

    def _inverse_transform_reward(self, reward: np.ndarray) -> np.ndarray:
        return np.rint((self._q_player * reward) * 2 - 1)

    def _init_qtable(self, q_table: QTableDict):
        for state_str in self._env.observation_space_values(self._q_player):
            for action in self._env.from_state_str(state_str).getEmptySpaces():
                q_table.set_value(state_str, self._env.int_from_action(action), 0.5)

    @abstractmethod
    def _update_q_function(self, info: CallbackInfo):
        pass

    def _generate_episode(self) -> int:
        total_rewards = simulate(self._env, self._cross_policy,
                                 self._circle_policy, self._update_q_function)

        return self._transform_reward(total_rewards)

    def _evaluate(self, num_episodes: int) -> np.ndarray:
        self.eval()
        game_stat = np.zeros(num_episodes, dtype=np.int8)

        for i in range(num_episodes):
            game_stat[i] = self._generate_episode()
            # If reward is negative
            if game_stat[i] not in (CIRCLE_PLAYER, CROSS_PLAYER, DRAW):
                if self._q_player == CIRCLE_PLAYER:
                    game_stat[i] = CROSS_PLAYER
                else:
                    game_stat[i] = CIRCLE_PLAYER

        return self._inverse_transform_reward(game_stat)

    def train(self):
        self._is_learning = True

    def eval(self):
        self._is_learning = False

    def simulate(self, num_episodes: int, num_policy_exp: Optional[int] = None) -> TDLearningRes:
        self._init_qtable(self._q_table)

        assert len(self._q_table) > 0, "Empty qtable"

        test_cross_win = []
        test_circle_win = []
        test_draw = []
        test_episode_num = []

        mean_train_reward = StreamignMean()

        print_every = num_episodes // 20
        compute_every = 100

        for episode in range(num_episodes):
            self.train()
            reward = self._generate_episode()
            mean_train_reward.add_value(reward)

            if num_policy_exp != -1 and num_policy_exp is not None and episode % compute_every == 0:
                game_stats = self._evaluate(num_policy_exp)
                stat = compute_game_stat(game_stats)
                test_cross_win.append(stat.cross_win_fraction)
                test_circle_win.append(stat.circle_win_fraction)
                test_draw.append(stat.draw_fraction)
                test_episode_num.append(episode)

            if (episode + 1) % print_every == 0:
                self._logger.info(f"Progress: {episode / num_episodes:.2%}")

        if num_policy_exp == -1:
            game_stats = self._evaluate(num_policy_exp)
            stat = compute_game_stat(game_stats)
            test_cross_win.append(stat.cross_win_fraction)
            test_circle_win.append(stat.circle_win_fraction)
            test_draw.append(stat.draw_fraction)
            test_episode_num.append(1)

        return TDLearningRes(mean_train_reward.mean(), test_cross_win, test_circle_win, test_draw, test_episode_num)


class QLearningSimulation(TDLearning):
    def __init__(self, env: TicTacToe,
                 cross_policy: BasePolicy,
                 circle_policy: BasePolicy,
                 generator: random.Random = None, **kwargs):
        super().__init__(env=env, cross_policy=cross_policy,
                         circle_policy=circle_policy, generator=generator, **kwargs)
        self._alpha = kwargs["alpha"]
        self._gamma = kwargs["gamma"]
        self._is_q_player_make_step = False
        kwargs.pop("alpha")
        kwargs.pop("gamma")

    def train(self):
        super().train()
        self._is_q_player_make_step = False

    def _update_q_function(self, callback_info: CallbackInfo):
        if callback_info.action_player == self._q_player and not self._is_q_player_make_step:
            self._is_q_player_make_step = True
            return

        if self._is_learning and callback_info.action_player != self._q_player and self._is_q_player_make_step:
            old_state = callback_info.old_env_state
            action = self._env.int_from_action(callback_info.old_action)
            new_state = callback_info.new_state
            reward = self._transform_reward(callback_info.reward)

            if callback_info.is_end:
                self._q_table.set_value(old_state, action, reward)
            else:
                old_value = self._q_table.get_value(old_state, action)
                greedy_reward = self._gamma * max(self._q_table.get_actions(new_state).values())
                self._q_table.set_value(old_state, action, old_value + self._alpha *
                                        (greedy_reward - old_value))


# class Sarsa(TDLearning):
#     def __init__(self, env: TicTacToe, policy, q_table: QTableDict, action_space, **kwargs):
#         super().__init__(env, policy, q_table, action_space=action_space, **kwargs)
#         self._alpha = kwargs["alpha"]
#         self._gamma = kwargs["gamma"]

#     def _update_q_function(self, old_state, action: int, new_state, reward: float, new_action: int):
#         if self._is_learning:
#             old_value = self._q_table.get_value(old_state, action)
#             greedy_reward = self._gamma * self._q_table.get_value(new_state, new_action)
#             self._q_table.set_value(old_state, action, old_value + self._alpha *
#                                     (reward + greedy_reward - old_value))

#     def _generate_episode(self) -> Tuple[float, int]:
#         state = self._env.reset()

#         total_rewards = 0
#         total_steps = 0

#         done = False

#         action = self._policy.action(state)

#         while not done:
#             old_state = state
#             state, reward, done, _ = self._env.step(action)

#             old_action = action
#             action = self._policy.action(state)

#             total_rewards += reward

#             total_steps += 1
#             self._update_q_function(old_state, old_action, state, reward, action)

#         return total_rewards, total_steps
