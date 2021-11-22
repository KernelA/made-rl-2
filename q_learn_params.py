import logging
import pathlib

import hydra
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

import log_set
from tictac_rl import TreePolicy, EpsilonGreedyPolicy
from tictac_rl.q_learning import QLearningSimulation
from tictac_rl.utils import dump


# TODO: debug simulation
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

    q_learner = QLearningSimulation(env, cross_policy, circle_policy,
                                    config.path_to_state,
                                    is_learning=True,
                                    gamma=config.gamma,
                                    alpha=config.learning_rate)

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

    with SummaryWriter(log_dir) as writer:
        td_res = q_learner.simulate(config.num_train_iterations, config.num_test_iterations)

        writer.add_hparams({"gamma": config.gamma,
                            "alpha (learning rate)": config.learning_rate},
                           {"mean_reward": td_res.mean_reward})

        for step, value in zip(td_res.test_episodes, td_res.test_mean_rewards):
            writer.add_scalar("Test/Mean reward", value, global_step=step)

        dump(cross_policy, str(policy_dump / "cross.pickle"))
        dump(circle_policy, str(policy_dump / "circle.pickle"))

        logger.info("Validate on %d", config.num_valid_episodes)

        td_res = q_learner.simulate(config.num_valid_episodes, -1)
        value = td_res.test_mean_rewards[0]
        step = td_res.test_episodes[0]

        logger.info("valid mean reward %f", value)

        writer.add_scalar("Valid/Mean reward", value, global_step=step)


if __name__ == "__main__":
    main()
