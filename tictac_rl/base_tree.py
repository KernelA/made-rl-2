import pickle
from abc import ABC, abstractmethod
from typing import Tuple

from anytree import NodeMixin

from .contstants import PICKLE_PROTOCOL
from .env import TicTacToe, ActionType


def load_from_dump(path_to_file: str):
    with open(path_to_file, "rb") as dump_file:
        return pickle.load(dump_file)


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
        with open(path_to_file, "wb") as dump_file:
            pickle.dump(self, dump_file, protocol=PICKLE_PROTOCOL)

    @ staticmethod
    def load_from_dump(path_to_file: str):
        return load_from_dump(path_to_file)

    @abstractmethod
    def find_game_state(self, prev_state: NodeMixin, env_state: str) -> NodeMixin:
        pass
