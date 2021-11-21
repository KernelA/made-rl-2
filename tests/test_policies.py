import pytest

from tictac_rl import simulate
from tictac_rl.env import DRAW, CIRCLE_PLAYER, CROSS_PLAYER


@pytest.mark.parametrize("circle_policy", [pytest.lazy_fixture("random_policy"),  pytest.lazy_fixture("minmax_cross_policy"), pytest.lazy_fixture("mctc_policy_cross")])
@pytest.mark.parametrize("cross_policy", [pytest.lazy_fixture("random_policy"), pytest.lazy_fixture("minmax_circle_policy"), pytest.lazy_fixture("mctc_policy_circle")])
def test_simulate(circle_policy, cross_policy, env_3x3):
    for _ in range(2):
        assert simulate(env_3x3, cross_policy, circle_policy) in (DRAW, CIRCLE_PLAYER, CROSS_PLAYER)
