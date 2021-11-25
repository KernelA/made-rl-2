from typing import Optional, Union
import random
from collections import namedtuple
from abc import ABC, abstractmethod
import logging

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from tictac_rl.nn.transforms import board_state2batch, board_state_str2batch

from ..utils import QTableDict
from ..env import TicTacToe, CIRCLE_PLAYER, CROSS_PLAYER, DRAW, simulate, CallbackInfo, BAD_REWARD
from ..policies import BasePolicy, EpsilonGreedyPolicy, NetworkPolicy
from ..nn.transition_memory import ReplayMemory, Transition
from ..stat import StreamignMean
from ..utils import GameStat, compute_game_stat

TDLearningRes = namedtuple(
    "TDLearningRes", ["mean_reward", "test_cross_win", "test_circle_win", "test_draw", "test_episode_num"])


class TDLearning(ABC):
    def __init__(self, env: TicTacToe,
                 cross_policy: BasePolicy,
                 circle_policy: BasePolicy,
                 q_player: int,
                 generator: random.Random = None,
                 **kwargs):
        self._env = env
        self._cross_policy = cross_policy
        self._circle_policy = circle_policy
        self._is_learning = kwargs["is_learning"]
        self._logger = logging.getLogger("simulation")

        assert self._circle_policy is not self._cross_policy

        self.q_player = q_player

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

        if reward == BAD_REWARD:
            return BAD_REWARD

        return 0.5 * (self.q_player * reward + 1)

    def _inverse_transform_reward(self, reward: Union[np.ndarray, float]) -> Union[int, np.ndarray]:
        if isinstance(reward, float):
            if reward == BAD_REWARD:
                return BAD_REWARD
            else:
                return np.rint((self.q_player * reward) * 2 - 1)

        mask = reward == BAD_REWARD
        new_reward = np.rint((self.q_player * reward) * 2 - 1)
        new_reward[mask] = BAD_REWARD

        return new_reward

    @abstractmethod
    def _update_q_function(self, info: CallbackInfo):
        pass

    def _generate_episode(self) -> int:
        total_rewards = simulate(self._env, self._cross_policy,
                                 self._circle_policy, self._update_q_function)

        return total_rewards

    def _evaluate(self, num_episodes: int) -> np.ndarray:
        self.eval()
        game_stat = np.zeros(num_episodes, dtype=np.int8)

        for i in range(num_episodes):
            game_stat[i] = self._generate_episode()
            # If reward is negative
            if game_stat[i] == BAD_REWARD:
                if self.q_player == CIRCLE_PLAYER:
                    game_stat[i] = CROSS_PLAYER
                else:
                    game_stat[i] = CIRCLE_PLAYER

        return game_stat

    def train(self):
        self._is_learning = True

    def eval(self):
        self._is_learning = False

    @abstractmethod
    def _pre_init_sim(self):
        pass

    def simulate(self, num_episodes: int, num_policy_exp: Optional[int] = None) -> TDLearningRes:
        self._pre_init_sim()

        test_cross_win = []
        test_circle_win = []
        test_draw = []
        test_episode_num = []

        mean_train_reward = StreamignMean()

        print_every = num_episodes // 20
        compute_every = 100

        for episode in range(num_episodes):
            self.train()
            reward = self._transform_reward(self._generate_episode())
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

        q_player = CROSS_PLAYER

        if isinstance(circle_policy, EpsilonGreedyPolicy):
            q_player = CIRCLE_PLAYER
            self._q_table = circle_policy.q_function
        else:
            self._q_table = cross_policy.q_function

        self._alpha = kwargs["alpha"]
        self._gamma = kwargs["gamma"]
        self._is_q_player_make_step = False
        kwargs.pop("alpha")
        kwargs.pop("gamma")

        super().__init__(env=env, cross_policy=cross_policy, q_player=q_player,
                         circle_policy=circle_policy, generator=generator, **kwargs)


    def _pre_init_sim(self):
        self._init_qtable(self._q_table)

    def train(self):
        super().train()
        self._is_q_player_make_step = False

    def _init_qtable(self, q_table: QTableDict):
        for state_str in self._env.observation_space_values(self.q_player):
            for action in self._env.from_state_str(state_str).getEmptySpaces():
                q_table.set_value(state_str, self._env.int_from_action(action), 0.5)

    def _update_q_function(self, callback_info: CallbackInfo):
        if callback_info.action_player == self.q_player and not self._is_q_player_make_step:
            self._is_q_player_make_step = True
            return

        if self._is_learning and callback_info.action_player != self.q_player and self._is_q_player_make_step:
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


class QNeuralNetworkSimulation(TDLearning):
    def __init__(self, env: TicTacToe,
        cross_policy: BasePolicy,
        circle_policy: BasePolicy,
        optimizer,
        memory_capacity: int,
        batch_size: int,
        device,
        writer: SummaryWriter,
        scheduler = None,
        generator: random.Random = None,
        **kwargs):

        self._device = device
        q_player = CROSS_PLAYER
        self._q_network: NetworkPolicy = self._cross_policy

        if isinstance(circle_policy, NetworkPolicy):
            q_player = CIRCLE_PLAYER
            self._q_network = self._circle_policy

        self._writer = writer

        self._memory = ReplayMemory(capacity=memory_capacity, generator=generator)
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._gamma = kwargs["gamma"]
        self._is_q_player_make_step = False
        super().__init__(env, cross_policy, circle_policy, q_player, generator=generator, **kwargs)

    def _pre_init_sim(self):
        pass

    def train(self):
        super().train()
        self._is_q_player_make_step = False

    def _train_model(self):
        if len(self._memory) < self._batch_size:
            return

        transitions = self._memory.sample(self._batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(batch.is_end, device=self._device, dtype=torch.bool)

        non_final_next_states = board_state_str2batch(batch.next_state, self._env.n_rows, self._env.n_cols)[non_final_mask].to(self._device)
        state_batch = torch.tensor(batch.state, device=self._device)
        action_batch = torch.tensor(batch.action, device=self._device)
        reward_batch = torch.tensor(batch.reward, device=self._device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self._q_network._model(state_batch).gather(1, action_batch)

        next_stat_values = torch.zeros(self._batch_size, device=self._device)
        next_stat_values[non_final_mask] = self._q_network._model(non_final_next_states).max(dim=-1)[0].detach()

        expected_q_function = self._gamma * next_stat_values + reward_batch

        self._optimizer.zero_grad()

        loss = F.smooth_l1_loss(state_action_values, expected_q_function)

        loss.backward()
        torch.nn.utils.clip_grad(self._model.parameters())
        self._optimizer.step()

    def _update_q_function(self, callback_info: CallbackInfo):
        if callback_info.action_player == self.q_player and not self._is_q_player_make_step:
            self._is_q_player_make_step = True
            return

        if self._is_learning and callback_info.action_player != self.q_player and self._is_q_player_make_step:
            old_state = callback_info.old_env_state
            action = self._env.int_from_action(callback_info.old_action)
            new_state = callback_info.new_state
            reward = self._transform_reward(callback_info.reward)

            self._memory.push(old_state, action, new_state, reward, callback_info.is_end)
            self._train_model()

