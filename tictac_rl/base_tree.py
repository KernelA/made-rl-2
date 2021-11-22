from abc import ABC, abstractmethod
from typing import Tuple

from anytree import NodeMixin

from .env import TicTacToe, ActionType
from .utils import load_from_dump, dump


class GameTreeBase(ABC):
    @abstractmethod
    def build_from_env(self, env: TicTacToe):
        pass

    @abstractmethod
    def best_move(self, prev_state: NodeMixin, env: TicTacToe, is_max: bool) -> Tuple[ActionType, NodeMixin]:
        pass

    def transit_to_state(self, prev_node, env_state: str, env: TicTacToe):
        pass

    @abstractmethod
    def set_random_proba(self, eps: float):
        pass

    def dump(self, path_to_file: str) -> None:
        dump(self, path_to_file)

    @ staticmethod
    def load_from_dump(path_to_file: str):
        return load_from_dump(path_to_file)

    @abstractmethod
    def find_game_state(self, prev_state: NodeMixin, env_state: str) -> NodeMixin:
        pass
