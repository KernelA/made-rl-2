import logging
import pathlib

import hydra
import numpy as np
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from tictac_rl.env.simulate import simulate
import log_set
from tictac_rl import TreePolicy
from tictac_rl.env.tictac import CIRCLE_PLAYER, CROSS_PLAYER, DRAW
from tictac_rl.q_learning import QLearningSimulation
from tictac_rl.policies.policy import NetworkPolicy
from tictac_rl.q_learning.q_learning import TDLearning
from tictac_rl.utils import dump, compute_game_stat


@hydra.main(config_path="configs", config_name="table_q_learning")
def main(config):
    logger = logging.getLogger()

    env = hydra.utils.instantiate(config.env)

    cross_policy = hydra.utils.instantiate(config.cross_policy)
    circle_policy = hydra.utils.instantiate(config.circle_policy)

    if isinstance(cross_policy, TreePolicy):
        cross_policy.tree.set_random_proba(config.tree_random_action_proba)

    if isinstance(circle_policy, TreePolicy):
        circle_policy.tree.set_random_proba(config.tree_random_action_proba)


    logger.info("Exp dir %s", config.exp_dir)
    exp_dir = pathlib.Path(config.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    log_dir = exp_dir / "logs"
    config_path = exp_dir / "config.yaml"

    with open(config_path, "w", encoding="utf-8") as config_file:
        config_file.write(OmegaConf.to_yaml(config))

    policy_dump = exp_dir / "policies"
    policy_dump.mkdir(exist_ok=True, parents=True)

    logger.info("Save logs to %s", log_dir)

    add_args = dict()

    if "optimizer" in config:
        if isinstance(cross_policy, NetworkPolicy):
            optimizer = hydra.utils.instantiate(config.optimizer, cross_policy._model.parameters())
            add_args["optimizer"] = optimizer
        elif isinstance(circle_policy, NetworkPolicy):
            optimizer = hydra.utils.instantiate(config.optimizer, circle_policy._model.parameters())
            add_args["optimizer"] = optimizer


    with SummaryWriter(log_dir) as writer:
        q_learner: TDLearning = hydra.utils.instantiate(
            config.q_learner,
            env=env,
            cross_policy=cross_policy,
            circle_policy=circle_policy,
            writer=writer,
            **add_args)

        try:
            td_res = q_learner.simulate(config.num_train_iterations, config.num_test_iterations)
        except TypeError:
            logger.exception("Unexpected exception")
            return

        train_mean_reward = td_res.mean_reward

        dump(cross_policy, str(policy_dump / "cross.pickle"))
        dump(circle_policy, str(policy_dump / "circle.pickle"))

        logger.info("Validate on %d", config.num_valid_episodes)

        game_stat = np.zeros(config.num_valid_episodes, dtype=np.int8)

        for i in range(config.num_valid_episodes):
            game_stat[i] = simulate(env, cross_policy, circle_policy)
            # This state is busy. Fail
            if game_stat[i] not in (CIRCLE_PLAYER, CROSS_PLAYER, DRAW):
                if q_learner.q_player == CIRCLE_PLAYER:
                    game_stat[i] = CROSS_PLAYER
                else:
                    game_stat[i] = CIRCLE_PLAYER

        stat = compute_game_stat(game_stat)

        writer.add_hparams({"gamma": config.q_learner.gamma,
                            "alpha (learning rate)": config.q_learner.alpha},
                        {"valid_cross_win": stat.cross_win_fraction,
                            "valid_circle_win": stat.circle_win_fraction,
                            "valid_draw": stat.draw_fraction,
                            "train_mean_reward": train_mean_reward})


if __name__ == "__main__":
    main()
