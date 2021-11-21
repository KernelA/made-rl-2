import log_set
import tqdm
import os
import logging

import hydra

from tictac_rl import TicTacToe, simulate, GameTreeBase, TreePolicy


@hydra.main(config_path="configs", config_name="mcts_precompute")
def main(config):
    logger = logging.getLogger()

    os.makedirs(config.out_dir, exist_ok=True)

    env: TicTacToe = hydra.utils.instantiate(config.env)

    logger.info("Precompute MCTS tree for %s %d simulations", repr(env), config.num_simulation)

    cross_policy = hydra.utils.instantiate(config.cross_policy)
    circle_policy = hydra.utils.instantiate(config.circle_policy)

    for _ in tqdm.trange(config.num_simulation):
        simulate(env, cross_policy, circle_policy)

    path_to_dump = os.path.join(config.out_dir,
                                f"{env.n_rows}_{env.n_cols}_{env.n_win}_start_{env._start_player}.pickle")

    if isinstance(cross_policy, TreePolicy):
        path_to_dump = os.path.join(
            config.out_dir, f"{env.n_rows}_{env.n_cols}_{env.n_win}_cross_start_{env._start_player}.pickle")
        cross_policy._tree.dump(path_to_dump)

    del cross_policy

    if isinstance(circle_policy, TreePolicy):
        path_to_dump = os.path.join(
            config.out_dir, f"{env.n_rows}_{env.n_cols}_{env.n_win}_circle_start_{env._start_player}.pickle")
        circle_policy._tree.dump(path_to_dump)


if __name__ == "__main__":
    main()
