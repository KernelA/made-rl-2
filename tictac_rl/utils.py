from .contstants import PICKLE_PROTOCOL
from collections import UserDict
from typing import Dict
import random
import sys
import pickle


def get_seed() -> int:
    return random.randrange(sys.maxsize)


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
