import pathlib
import json
from collections import defaultdict

import anytree
import hydra
import tqdm

from tictac_rl import TicTacToe, CIRCLE_PLAYER, CROSS_PLAYER


@hydra.main(config_path="configs", config_name="generate_states")
def main(config):
    tree = hydra.utils.instantiate(config.tree)
    env: TicTacToe = hydra.utils.instantiate(config.env)

    assert config.start_player in (CIRCLE_PLAYER, CROSS_PLAYER)

    path_to_file = pathlib.Path(config.state_file)
    path_to_file.parent.mkdir(exist_ok=True, parents=True)

    possible_states = defaultdict(set)

    is_save = config.start_player == env._start_player

    def add_states(node):
        for child in node.children:
            possible_states[node.name].add(env.int_from_action(child.step))

    for group in tqdm.tqdm(anytree.LevelOrderGroupIter(tree.root), total=tree.root.height):
        for node in group:
            if node.is_terminal:
                possible_states[node.name] = set()

            if is_save:
                add_states(node)

        is_save = not is_save

    for state in tuple(possible_states.keys()):
        possible_states[state] = list(possible_states[state])

    with open(path_to_file, "w", encoding="utf-8") as file:
        json.dump(possible_states, file)


if __name__ == "__main__":
    main()
