import os
import logging

import hydra
from hydra import utils
import log_set
from tictac_rl import TicTacToe, MinMaxTree
from tictac_rl.env.tictac import CROSS_PLAYER


@hydra.main(config_name="minmax_precompute", config_path="configs")
def main(config):
    os.makedirs(config.out_dir, exist_ok=True)
    logger = logging.getLogger()

    env: TicTacToe = utils.instantiate(config.env)

    logger.info("Build tree for %s", repr(env))
    tree = MinMaxTree()
    tree.build_from_env(env)

    dump_name = os.path.join(
        config.out_dir, f"{env.n_rows}_{env.n_cols}_{env.n_win}_start_{env._start_player}.pickle")
    logger.info("Save dump to %s", dump_name)
    tree.dump(dump_name)


if __name__ == '__main__':
    main()
