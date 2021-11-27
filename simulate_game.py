import json
import pathlib
import logging

import hydra
import tqdm
import numpy as np
from omegaconf import OmegaConf

import log_set
from tictac_rl.min_max_tree import MinMaxTree
from tictac_rl import TreePolicy, simulate
from tictac_rl.env import CROSS_PLAYER, CIRCLE_PLAYER, DRAW
from tictac_rl.utils import load_from_dump, compute_game_stat


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
        if "cross_policy" in config:
            cross_policy = hydra.utils.instantiate(config.cross_policy)
        elif "cross_player" in config:
            cross_policy = hydra.utils.instantiate(config.cross_player.object)
        else:
             raise ValueError("Cannot find config for cross policy")

        if "circle_policy" in config:
            circle_policy = hydra.utils.instantiate(config.circle_policy)
        elif "circle_player" in config:
            circle_policy = hydra.utils.instantiate(config.circle_player.object)
        else:
            raise ValueError("Cannot find config for circle policy")

        if isinstance(cross_policy, TreePolicy) and config.random_action_proba is not None and isinstance(cross_policy.tree, MinMaxTree):
            cross_policy.tree.set_random_proba(config.random_action_proba)

        if isinstance(circle_policy, TreePolicy) and config.random_action_proba is not None and isinstance(circle_policy.tree, MinMaxTree):
            circle_policy.tree.set_random_proba(config.random_action_proba)

    env = hydra.utils.instantiate(config.env)

    game_stat = np.zeros(config.num_sim, dtype=np.int8)

    for i in tqdm.trange(config.num_sim, miniters=500):
        game_stat[i] = simulate(env, cross_policy, circle_policy)

    metric_file = pathlib.Path(config.metric_file)
    metric_file.parent.mkdir(parents=True, exist_ok=True)

    stat = compute_game_stat(game_stat)

    game_stat = {"cross_fraction_win": stat.cross_win_fraction,
                "circle_fraction_win": stat.circle_win_fraction,
                "draw_fraction": stat.draw_fraction}

    logger.info("Save metric to %s", metric_file)

    with open(metric_file, "w", encoding="utf-8") as file:
        json.dump(game_stat, file)


if __name__ == "__main__":
    main()
