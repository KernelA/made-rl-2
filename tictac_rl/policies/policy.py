import random
from abc import ABC, abstractmethod
from typing import Optional

from tictac_rl.min_max_tree import MinMaxTree

from ..env import TicTacToe, ActionType
from ..utils import QTableDict
from ..base_tree import GameTreeBase


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

    def action(self, env: TicTacToe, env_hash: Optional[str]) -> ActionType:
        return self._generator.choice(env.getEmptySpaces())

    def reset(self):
        pass


class TreePolicy(BasePolicy):
    def __init__(self, tree: GameTreeBase):
        self.tree = tree
        self.reset()

    def action(self, env: TicTacToe, env_hash: Optional[str]) -> ActionType:
        if env_hash is not None:
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

    def action(self, env: TicTacToe, env_hash: Optional[str]) -> ActionType:
        if self._generator.random() < self._epsiolon and env_hash is None:
            return random.choice(env.getEmptySpaces())
        else:
            action = max(self.q_function.get_actions(env_hash).items(), key=lambda x: x[1])[0]
            return env.action_from_int(action)
