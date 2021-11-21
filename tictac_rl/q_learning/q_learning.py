from os import read
from typing import Sequence, Tuple, Optional
import random
from collections import namedtuple
from abc import ABC, abstractmethod
import logging
import json

import numpy as np

from ..utils import QTableDict
from ..env import TicTacToe, CIRCLE_PLAYER, CROSS_PLAYER, simulate, CallbackInfo
from ..policies import BasePolicy, EpsilonGreedyPolicy

from ..stat import StreamignMean

TDLearningRes = namedtuple(
    "TDLearningRes", ["mean_reward", "mean_step", "test_mean_rewards", "test_episodes"])


class TDLearning(ABC):
    def __init__(self, env: TicTacToe,
                 cross_policy: BasePolicy,
                 circle_policy: BasePolicy,
                 path_to_state_file: str,
                 generator: random.Random = None,
                 **kwargs):
        self._env = env
        self._cross_policy = cross_policy
        self._circle_policy = circle_policy
        self._is_learning = kwargs["is_learning"]
        self._logger = logging.getLogger("simulation")

        self._q_player = CROSS_PLAYER

        self._path_to_state_file = path_to_state_file

        if "cross" in self._path_to_state_file and isinstance(self._circle_policy, EpsilonGreedyPolicy):
            raise ValueError("State file does not correspond Q-player")

        if "circle" in self._path_to_state_file and isinstance(self._cross_policy, EpsilonGreedyPolicy):
            raise ValueError("State file does not correspond Q-player")

        if isinstance(self._circle_policy, EpsilonGreedyPolicy):
            self._q_player = CIRCLE_PLAYER
            self._q_table = self._circle_policy.q_function
        else:
            self._q_table = self._cross_policy.q_function

        if isinstance(self._circle_policy, EpsilonGreedyPolicy) and\
                isinstance(self._circle_policy, EpsilonGreedyPolicy) and self._is_learning:
            raise ValueError(
                "Both player are based on QTable please specify other policy for one the him")

        if generator is None:
            generator = random.Random()

        self._generator = generator

    def _init_qtable(self, q_table: QTableDict):
        with open(self._path_to_state_file, "r", encoding="utf-8") as file:
            all_states_for_player = json.load(file)

        for state in all_states_for_player:
            new_env = self._env.from_state_str(state)

            for action in all_states_for_player[state]:
                state, reward, is_end = new_env.step(action)
                if is_end:
                    break
                if reward is not None:
                    reward *= self._q_player
                else:
                    reward = 0

                q_table.set_value(state, action, reward)

    @abstractmethod
    def _update_q_function(self, info: CallbackInfo):
        pass

    def _generate_episode(self) -> int:
        total_rewards = simulate(self._env, self._circle_policy,
                                 self._circle_policy, self._update_q_function)

        return total_rewards

    def _evaluate(self, num_episodes: int):
        self._is_learning = False
        exp_rewards = StreamignMean()

        for _ in range(num_episodes):
            reward = self._generate_episode()
            exp_rewards.add_value(reward)

        return exp_rewards.mean()

    def simulate(self, num_episodes: int, num_policy_exp: Optional[int] = None) -> TDLearningRes:
        self._init_qtable(self._q_table)

        test_rewards = []
        test_episode = []

        mean_train_reward = StreamignMean()
        mean_step = StreamignMean()

        print_every = num_episodes // 10
        compute_every = 4

        for episode in range(num_episodes):
            self._is_learning = True
            reward = self._generate_episode()
            mean_train_reward.add_value(reward)

            if num_policy_exp != -1 and num_policy_exp is not None and episode % compute_every == 0:
                test_rewards.append(self._evaluate(num_policy_exp))
                test_episode.append(episode)

            if (episode + 1) % print_every == 0:
                self._logger.info(f"Progress: {episode / num_episodes:.2%}")

        if num_policy_exp == -1:
            test_rewards.append(self._evaluate(num_episodes))
            test_episode.append(1)

        return TDLearningRes(mean_train_reward.mean(), mean_step.mean(), np.array(test_rewards), np.array(test_episode))


class QLearningSimulation(TDLearning):
    def __init__(self, env: TicTacToe,
                 cross_policy: BasePolicy,
                 circle_policy: BasePolicy,
                 path_to_state_file: str,
                 generator: random.Random = None, **kwargs):
        super().__init__(env=env, cross_policy=cross_policy, circle_policy=circle_policy,
                         path_to_state_file=path_to_state_file, generator=generator, **kwargs)
        self._alpha = kwargs["alpha"]
        self._gamma = kwargs["gamma"]
        self._is_q_player_make_step = False
        kwargs.pop("alpha")
        kwargs.pop("gamma")

    def _update_q_function(self, callback_info: CallbackInfo):
        if callback_info.action_player == self._is_q_player_make_step:
            self._is_q_player_make_step = True

        if self._is_learning and self._is_q_player_make_step and callback_info.action_player != self._q_player:
            old_state = callback_info.old_env_state
            action = self._env.int_from_action(callback_info.action)
            new_state = callback_info.new_state
            reward = callback_info.reward
            if reward is None:
                reward = 0
            reward *= self._q_player

            old_value = self._q_table.get_value(old_state, action)
            greedy_reward = self._gamma * max(self._q_table.get_actions(new_state).values())
            self._q_table.set_value(old_state, action, old_value + self._alpha *
                                    (reward + greedy_reward - old_value))


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
