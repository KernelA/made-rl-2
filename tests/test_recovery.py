import random
import itertools

import pytest

from tictac_rl import TicTacToe
from tictac_rl.env.tictac import StartPlayer


@pytest.mark.parametrize("start_player", [StartPlayer.circle, StartPlayer.cross])
def test_recovery(start_player):
    gen = random.Random()
    gen.seed(112)
    env = TicTacToe(3, 3, 3, start_player=start_player)

    for start_x, start_y in itertools.product(range(env.n_rows), range(env.n_cols)):
        env.reset()

        is_end = False

        free_space = env.getEmptySpaces()

        while not is_end:
            action = gen.choice(free_space)
            state, _, is_end = env.step(action)
            new_env = env.from_state_str(state[0])
            assert env._getHash() == new_env._getHash()
            assert env.curTurn == new_env.curTurn
            free_space = env.getEmptySpaces()
