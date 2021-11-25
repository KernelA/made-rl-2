import pytest

import numpy as np

from tictac_rl.utils import compute_game_stat
from tictac_rl.env import CROSS_PLAYER, CIRCLE_PLAYER, DRAW


def test_all_cross():
    stat = compute_game_stat(np.repeat(CROSS_PLAYER, 10))
    assert stat.cross_win_fraction == 1
    assert stat.circle_win_fraction == 0
    assert stat.draw_fraction == 0

def test_all_circle():
    stat = compute_game_stat(np.repeat(CIRCLE_PLAYER, 10))
    assert stat.cross_win_fraction == 0
    assert stat.circle_win_fraction == 1
    assert stat.draw_fraction == 0


def test_all_draw():
    stat = compute_game_stat(np.repeat(DRAW, 10))
    assert stat.cross_win_fraction == 0
    assert stat.circle_win_fraction == 0
    assert stat.draw_fraction == 1

def test_random_draw():
    stat = compute_game_stat(np.array((CROSS_PLAYER, CROSS_PLAYER, CIRCLE_PLAYER, DRAW)))
    assert stat.cross_win_fraction == 0.5
    assert stat.circle_win_fraction == 0.25
    assert stat.draw_fraction == 0.25
