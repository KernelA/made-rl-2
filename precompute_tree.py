import argparse
import os
import logging
import sys

import log_set
from tictac_rl import TicTacToe, MinMaxTree
from tictac_rl.env.tictac import StartPlayer

sys.setrecursionlimit(sys.getrecursionlimit() * 20)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    logger = logging.getLogger()

    start_player = StartPlayer.cross

    if args.start_player == "circle":
        start_player = StartPlayer.circle

    env = TicTacToe(args.n, args.n, args.n, start_player)

    logger.info("Build tree")
    tree = MinMaxTree.build_from_env(env)

    dump_name = os.path.join(args.out_dir, f"{args.n}_{args.n}_start_{args.start_player}.pickle")
    logger.info("Save dump to %s", dump_name)
    tree.dump(dump_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, required=True, help="A size of grid n x n")
    parser.add_argument("--start_player", type=str,
                        choices=["cross", "circle"],
                        required=True, help="First turn")
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    main(args)
