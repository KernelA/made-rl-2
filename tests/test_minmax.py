import logging
import itertools

import pytest

from tictac_rl import MinMaxTree, TicTacToe
from tictac_rl.env.tictac import CROSS_PLAYER

from .conftest import MINMAX_CROSS


def simulate_game(tree: MinMaxTree, start_player: int):
    env = TicTacToe(3, 3, 3, start_player)

    for start_x, start_y in itertools.product(range(env.n_rows), range(env.n_cols)):
        env.reset()
        is_max = True
        start_node = tree.root
        (hash_state, *_), reward, is_end = env.step((start_x, start_y))
        hashes = [hash_state]
        start_node = tree.find_game_state(tree.root, hash_state)

        while True:
            is_max = not is_max
            move, start_node = tree.best_move(start_node, env, is_max)
            (hash_state, *_), reward, is_end = env.step(move)
            hashes.append(hash_state)

            if is_end:
                assert reward == 0
                break


@ pytest.mark.parametrize(["start", "file"],
                          [[CROSS_PLAYER, MINMAX_CROSS]])
def test_minmax(start, file):
    tree = MinMaxTree.load_from_dump(file)
    simulate_game(tree, start)
