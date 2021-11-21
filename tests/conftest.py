import pytest

from tictac_rl import TicTacToe, RandomPolicy, TreePolicy, MinMaxTree, MCTS

MINMAX_CROSS = "./trees/3/cross/3_3_3_start_1.pickle"


@pytest.fixture(scope="session")
def env_3x3():
    return TicTacToe(3, 3, 3)


@pytest.fixture(scope="function")
def random_policy():
    return RandomPolicy()


@pytest.fixture(scope="function")
def mctc_policy_circle(env_3x3):
    env_3x3.reset()
    tree = MCTS(env_3x3)
    return TreePolicy(tree)


@pytest.fixture(scope="function")
def mctc_policy_cross(env_3x3):
    env_3x3.reset()
    tree = MCTS(env_3x3)
    return TreePolicy(tree)


@pytest.fixture(scope="session")
def minmax_circle_policy():
    tree = MinMaxTree.load_from_dump(MINMAX_CROSS)
    return TreePolicy(tree)


@pytest.fixture(scope="session")
def minmax_cross_policy():
    tree = MinMaxTree.load_from_dump(MINMAX_CROSS)
    return TreePolicy(tree)
