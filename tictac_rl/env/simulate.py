from typing import List, Tuple

from .tictac import TicTacToe
from ..policies import BasePolicy
from ..env import CROSS_PLAYER


def simulate(env: TicTacToe, cross_policy: BasePolicy, circle_policy: BasePolicy, record_history: bool = False) -> int:
    is_end = False
    env.reset()
    cross_policy.reset()
    circle_policy.reset()

    state_str = None

    while not is_end:
        if env.curTurn == CROSS_PLAYER:
            step = cross_policy.action(env, state_str)
        else:
            step = circle_policy.action(env, state_str)

        (state_str, _, _), reward, is_end = env.step(step)

        if is_end:
            return reward
