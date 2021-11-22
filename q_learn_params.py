import logging

import hydra
from torch.utils.tensorboard import SummaryWriter

import log_set
from tictac_rl import TreePolicy
from tictac_rl.q_learning import QLearningSimulation


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

    generator = hydra.utils.instantiate(config.generator)

    q_learner = QLearningSimulation(env, cross_policy, circle_policy, config.path_to_state,
                                    is_learning=True,
                                    generator=generator, gamma=config.gamma, alpha=config.learning_rate)
    with SummaryWriter(config.log_dir):
        td_res = q_learner.simulate(config.num_train_iterations, config.num_test_iterations)


if __name__ == "__main__":
    main()
