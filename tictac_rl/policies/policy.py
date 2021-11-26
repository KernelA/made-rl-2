import random
from abc import ABC, abstractmethod
from typing import Optional
import numpy

import torch

from ..env import TicTacToe, ActionType
from ..utils import QTableDict
from ..base_tree import GameTreeBase
from ..nn import QNetwork, board_state2batch
from ..contstants import EMPTY_STATE


class BasePolicy(ABC):
    @abstractmethod
    def action(self, env: TicTacToe, env_hash: Optional[str]) -> ActionType:
        pass

    @abstractmethod
    def reset(self):
        pass


class RandomPolicy(BasePolicy):
    def __init__(self, generator=None):
        if generator is None:
            generator = random.Random()
        self._generator = generator

    def action(self, env: TicTacToe, env_hash: str) -> ActionType:
        return self._generator.choice(env.getEmptySpaces())

    def reset(self):
        pass


class TreePolicy(BasePolicy):
    def __init__(self, tree: GameTreeBase):
        self.tree = tree
        self.reset()

    def action(self, env: TicTacToe, env_hash: str) -> ActionType:
        if env_hash != EMPTY_STATE:
            self._prev_state = self.tree.transit_to_state(self._prev_state, env_hash, env)

        is_max_player = True
        if env._start_player != env.curTurn:
            is_max_player = False

        action, self._prev_state = self.tree.best_move(self._prev_state, env, is_max=is_max_player)

        return action

    def reset(self):
        self._prev_state = self.tree.root


class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, q_function: QTableDict, epsilon: float, seed: int):
        self.q_function = q_function
        self._epsiolon = epsilon
        self._generator = random.Random(seed)
        self.seed = seed

    def action(self, env: TicTacToe, env_hash: str) -> ActionType:
        if self._generator.random() < self._epsiolon:
            return random.choice(env.getEmptySpaces())

        max_value = max(self.q_function.get_actions(env_hash).values())
        best_action = self._generator.choice([action for action, value in self.q_function.get_actions(env_hash).items() if value == max_value])

        return env.action_from_int(best_action)

    def reset(self):
        pass


class NetworkPolicy(BasePolicy):
    def __init__(self, model: QNetwork, epsilon: float, seed: int):
        super().__init__()
        self._model = model
        self._epislon = epsilon
        self._generator = random.Random(seed)
        self.seed = seed

    @torch.no_grad()
    def action(self, env: TicTacToe, env_hash: str) -> ActionType:
        if self._generator.random() < self._epislon:
            return random.choice(env.getEmptySpaces())

        self._model.eval()
        int_actions = list(map(env.int_from_action, env.getEmptySpaces()))
        best_action_index = self._model(board_state2batch(env.board))[:, int_actions].argmax(dim=-1).cpu().item()

        return env.action_from_int(int_actions[best_action_index])

    def reset(self):
        pass
