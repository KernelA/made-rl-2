from typing import Tuple, Callable
from collections import namedtuple

from .tictac import CIRCLE_PLAYER, TicTacToe
from ..policies import BasePolicy
from ..env import CROSS_PLAYER, ActionType
from ..contstants import EMPTY_STATE

CallbackInfo = namedtuple(
    "CallbackInfo", ["old_env_state", "old_action", "new_state", "reward", "action_player", "is_end"])


def simulate(env: TicTacToe, cross_policy: BasePolicy, circle_policy: BasePolicy,
             callback: Callable[[CallbackInfo], None] = None) -> int:
    is_end = False
    env.reset()
    cross_policy.reset()
    circle_policy.reset()

    state_str = EMPTY_STATE
    cross_step = None
    old_cross_state = state_str
    circle_step = None
    old_circle_state = None

    while not is_end:
        if env.curTurn == CROSS_PLAYER:
            step = cross_policy.action(env, state_str)
            cross_step = step
            old_cross_state = state_str
        else:
            step = circle_policy.action(env, state_str)
            circle_step = step
            old_circle_state = state_str

        old_turn = env.curTurn

        (state_str, *_), reward, is_end = env.step(step)

        if callback is not None:
            try:
                callback(CallbackInfo(old_cross_state if old_turn == CIRCLE_PLAYER else old_circle_state, circle_step if old_turn == CROSS_PLAYER else cross_step, state_str, reward, old_turn, is_end))
            except TypeError:
                pass
        if is_end:
            return reward
