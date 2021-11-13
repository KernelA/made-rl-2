import logging
import itertools

import pytest

from tictac_rl import MinMaxTree, TicTacToe
from tictac_rl.env.tictac import CIRCLE_PLAYER, CROSS_PLAYER

from .conftest import MINMAX_CROSS, MINMAX_CIRCLE

LOGGER = logging.getLogger("tictoc")


def simulate_game(tree: MinMaxTree, start_player: int):
    env = TicTacToe(3, 3, 3, start_player)

    for start_x, start_y in itertools.product(range(env.n_rows), range(env.n_cols)):
        LOGGER.info("start")
        env.reset()
        is_max = True
        hashes = []
        (hash_state, *_), *_ = env.step((start_x, start_y))
        hashes.append(hash_state)

        while True:
            is_max = not is_max
            move = tree.best_move(hashes, is_max)
            (hash_state, *_), reward, is_end = env.step(move)
            hashes.append(hash_state)
            if is_end:
                if reward != 0:
                    for hash_str in hashes:
                        LOGGER.info(hash_str)
                        LOGGER.info(env.from_state_str(hash_str))
                    LOGGER.handlers[0].flush()
                assert reward == 0
                break


@pytest.mark.parametrize(["start", "file"],
                         [[CIRCLE_PLAYER, MINMAX_CIRCLE]])
def test_minmax(start, file):
    tree = MinMaxTree.load_from_dump(file)
    simulate_game(tree, start)
