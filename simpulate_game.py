import json
import pathlib
import logging

import hydra
import tqdm

import log_set
from tictac_rl import TreePolicy, simulate
from tictac_rl.env import CIRCLE_PLAYER, CROSS_PLAYER, DRAW


@hydra.main(config_path="configs", config_name="simulate")
def main(config):
    logger = logging.getLogger()

    env = hydra.utils.instantiate(config.env)

    cross_policy = hydra.utils.instantiate(config.cross_policy)
    circle_policy = hydra.utils.instantiate(config.circle_policy)

    if isinstance(cross_policy, TreePolicy):
        cross_policy.tree.set_random_proba(config.random_action_proba)

    if isinstance(circle_policy, TreePolicy):
        circle_policy.tree.set_random_proba(config.random_action_proba)

    rewards = []

    for _ in tqdm.trange(config.num_sim, miniters=100):
        rewards.append(simulate(env, cross_policy, circle_policy))

    metric_file = pathlib.Path(config.metric_file)
    metric_file.parent.mkdir(parents=True, exist_ok=True)

    game_stat = {"cross_win": rewards.count(CROSS_PLAYER) / len(rewards), "circle_win": rewards.count(
        CIRCLE_PLAYER) / len(rewards), "draw": rewards.count(DRAW) / len(rewards)}

    logger.info("Save metric to %s", metric_file)

    with open(metric_file, "w", encoding="utf-8") as file:
        json.dump(game_stat, file)


if __name__ == "__main__":
    main()
