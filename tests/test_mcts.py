import pytest
import copy

from tictac_rl import TreePolicy, MCTS, simulate
from tictac_rl.env import DRAW, CIRCLE_PLAYER, CROSS_PLAYER


@pytest.mark.parametrize("eps", [0, 0.5, 1])
@pytest.mark.parametrize("max_depth", [None, 1, 2])
def test_mcts(eps, max_depth, env_3x3):
    env_3x3.reset()
    tree = MCTS(env_3x3, eps=eps, depth_limit=max_depth)
    cross_policy = TreePolicy(tree)
    circle_policy = TreePolicy(copy.deepcopy(tree))

    end = max_depth

    if max_depth is None:
        end = 1
    end += 1

    for _ in range(end):
        assert simulate(env_3x3, cross_policy, circle_policy) in (DRAW, CIRCLE_PLAYER, CROSS_PLAYER)
