from typing import Tuple, Callable
from collections import namedtuple

from .tictac import TicTacToe
from ..policies import BasePolicy
from ..env import CROSS_PLAYER, ActionType
from ..contstants import EMPTY_STATE

CallbackInfo = namedtuple(
    "CallbackInfo", ["old_env_state", "action", "new_state", "reward", "action_player", "is_end"])


def simulate(env: TicTacToe, cross_policy: BasePolicy, circle_policy: BasePolicy,
             callback: Callable[[CallbackInfo], None] = None) -> int:
    is_end = False
    env.reset()
    cross_policy.reset()
    circle_policy.reset()

    state_str = EMPTY_STATE

    while not is_end:
        if env.curTurn == CROSS_PLAYER:
            step = cross_policy.action(env, state_str)
        else:
            step = circle_policy.action(env, state_str)

        old_state = state_str
        old_turn = env.curTurn

        (state_str, *_), reward, is_end = env.step(step)

        if callback is not None:
            callback(CallbackInfo(old_state, step, state_str, reward, old_turn, is_end))

        if is_end:
            return reward
