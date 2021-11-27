from collections import UserDict, namedtuple
from typing import Dict
import random
import sys
import pickle

import numpy as np

from .contstants import PICKLE_PROTOCOL
from .env import CROSS_PLAYER, CIRCLE_PLAYER

GameStat = namedtuple("GameStat", ["cross_win_fraction", "circle_win_fraction", "draw_fraction"])


def get_seed() -> int:
    return random.randrange(sys.maxsize)


def compute_game_stat(game_stat: np.ndarray) -> GameStat:
    assert game_stat.ndim == 1
    cross_win = np.count_nonzero(game_stat == CROSS_PLAYER) / game_stat.shape[0]
    circle_win = np.count_nonzero(game_stat == CIRCLE_PLAYER) / game_stat.shape[0]
    draw = max(1 - cross_win - circle_win, 0)

    return GameStat(cross_win, circle_win, draw)


def load_from_dump(path_to_file: str):
    with open(path_to_file, "rb") as dump_file:
        return pickle.load(dump_file)


def dump(obj, path_to_file: str):
    with open(path_to_file, "wb") as dump_file:
        pickle.dump(obj, dump_file, protocol=PICKLE_PROTOCOL)


class QTableDict(UserDict):
    """Represnet discrete Q(s, a) tabular function
    """

    def __init__(self):
        super().__init__(dict())

    def set_value(self, state: str, action: int, value: float):
        if state not in self.data:
            self.data[state] = {action: value}
        else:
            self.data[state][action] = value

    def get_actions(self, state) -> Dict[int, float]:
        return self.data[state]

    def get_value(self, state, action) -> float:
        return self.data[state][action]
