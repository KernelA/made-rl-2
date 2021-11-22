import json
import pathlib
import logging

import hydra
import tqdm
from omegaconf import OmegaConf

import log_set
from tictac_rl import TreePolicy, simulate
from tictac_rl.env import CIRCLE_PLAYER, CROSS_PLAYER, DRAW
from tictac_rl.utils import load_from_dump


@hydra.main(config_path="configs", config_name="simulate")
def main(config):
    logger = logging.getLogger()

    if config.train_config is not None:
        logger.warning("Use training config. Skip all other parameters")
        config = OmegaConf.load(config.train_config)
        exp_dir = pathlib.Path(config.exp_dir)
        policy_dir = exp_dir / "policies"
        cross_policy = load_from_dump(str(policy_dir / "cross.pickle"))
        circle_policy = load_from_dump(str(policy_dir / "circle.pickle"))
    else:
        cross_policy = hydra.utils.instantiate(config.cross_policy)
        circle_policy = hydra.utils.instantiate(config.circle_policy)

        if isinstance(cross_policy, TreePolicy):
            cross_policy.tree.set_random_proba(config.random_action_proba)

        if isinstance(circle_policy, TreePolicy):
            circle_policy.tree.set_random_proba(config.random_action_proba)

    env = hydra.utils.instantiate(config.env)

    rewards = []

    for _ in tqdm.trange(config.num_sim, miniters=500):
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
