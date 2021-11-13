from abc import ABC, abstractclassmethod, abstractmethod
from typing import Tuple

from anytree import NodeMixin

from .env import TicTacToe

StepType = Tuple[int]


class GameTreeBase(ABC):
    @abstractmethod
    def build_from_env(self, env: TicTacToe):
        pass

    @abstractmethod
    def best_move(self, prev_state: NodeMixin, env: TicTacToe, is_max: bool) -> Tuple[StepType, NodeMixin]:
        pass

    @abstractmethod
    def find_game_state(self, prev_state: NodeMixin, env_state: str) -> NodeMixin:
        pass
