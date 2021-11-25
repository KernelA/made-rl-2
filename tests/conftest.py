import pytest

from tictac_rl import TicTacToe, RandomPolicy, TreePolicy, MinMaxTree, MCTS
from tictac_rl.env.tictac import CROSS_PLAYER, CIRCLE_PLAYER
from tictac_rl.policies import EpsilonGreedyPolicy
from tictac_rl.utils import QTableDict

MINMAX_CROSS = "./trees/minmax/3/cross/3_3_3_start_1.pickle"


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

@pytest.fixture(scope="session")
def q_table_circle_policy(env_3x3: TicTacToe):
    env_3x3.reset()
    table = QTableDict()
    for state in env_3x3.observation_space_values(CIRCLE_PLAYER):
        new_env = env_3x3.from_state_str(state)
        for action in new_env.getEmptySpaces():
            table.set_value(state, new_env.int_from_action(action), 0.5)

    return EpsilonGreedyPolicy(table, 0.1, 23)

@pytest.fixture(scope="session")
def q_table_cross_policy(env_3x3: TicTacToe):
    env_3x3.reset()
    table = QTableDict()
    for state in env_3x3.observation_space_values(CROSS_PLAYER):
        new_env = env_3x3.from_state_str(state)
        for action in new_env.getEmptySpaces():
            table.set_value(state, new_env.int_from_action(action), 0.5)

    return EpsilonGreedyPolicy(table, 0.1, 23)
