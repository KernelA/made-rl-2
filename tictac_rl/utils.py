from collections import UserDict
from typing import Dict
import random
import sys


def get_seed() -> int:
    return random.randrange(sys.maxsize)


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
