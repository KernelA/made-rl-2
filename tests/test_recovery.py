import random
import itertools

import pytest

from tictac_rl import TicTacToe
from tictac_rl.env.tictac import CIRCLE_PLAYER, CROSS_PLAYER
from tictac_rl.contstants import EMPTY_STATE


@pytest.mark.parametrize("start_player", [CIRCLE_PLAYER, CROSS_PLAYER])
def test_recovery(start_player):
    gen = random.Random()
    gen.seed(112)
    env = TicTacToe(3, 3, 3, start_player=start_player)

    for _, _ in itertools.product(range(env.n_rows), range(env.n_cols)):
        env.reset()

        is_end = False

        free_space = env.getEmptySpaces()

        while not is_end:
            action = gen.choice(free_space)
            state, _, is_end = env.step(action)
            new_env = env.from_state_str(state[0])
            assert env._getHash() == new_env._getHash()
            assert env.curTurn == new_env.curTurn
            assert (env.board == new_env.board).all()
            free_space = state[1]


@pytest.mark.parametrize("start_player", [CIRCLE_PLAYER, CROSS_PLAYER])
def test_empty_recovery(start_player):
    gen = random.Random()
    gen.seed(112)
    env = TicTacToe(3, 3, 3, start_player=start_player)

    new_env = env.from_state_str(EMPTY_STATE)

    assert new_env.curTurn == start_player
    assert new_env._start_player == start_player
    assert (new_env.board == 0).all()
