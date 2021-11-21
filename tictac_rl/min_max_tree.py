from typing import Tuple, Optional, Iterable
from operator import gt, lt
import pickle
import random

from anytree import Node
from anytree.node.nodemixin import NodeMixin
from tqdm.std import trange

from .env import TicTacToe, ActionType
from .contstants import PICKLE_PROTOCOL
from .base_tree import GameTreeBase


def nodename2hash_key(node: NodeMixin):
    return node.separator.join([""] + [str(node_path.name) for node_path in node.path])


def nodepath(root_node: NodeMixin, game_history: Iterable[str]):
    return f"{root_node.separator}{root_node.name}{root_node.separator}" + \
        f"{root_node.separator}".join(game_history)


def compute_total_nodes(total_cells: int, start_level: int, end_level: int):
    assert start_level > 0
    assert start_level <= end_level
    node_at_prev_levels = 1
    total_nodes = node_at_prev_levels

    for level in range(start_level, end_level + 1):
        node_at_level = (total_cells - level + 1) * node_at_prev_levels
        total_nodes += node_at_level
        node_at_prev_levels = node_at_level

    return total_nodes


def r_build_minmax_tree(tic_tac_env: TicTacToe, parent_node: Node, progress):
    total_cells = tic_tac_env.n_cols * tic_tac_env.n_rows
    positions = tic_tac_env.getEmptySpaces()

    for pos in positions:
        progress.update()
        new_env = tic_tac_env.clone()
        node_name, reward, is_end = new_env.step(pos)
        node = Node(node_name[0], parent=parent_node, reward=reward *
                    int(new_env._start_player), step=tuple(pos))

        if not is_end:
            r_build_minmax_tree(new_env, node, progress)
        elif node.depth != total_cells:
            progress.update(compute_total_nodes(total_cells, node.depth + 1, total_cells))


class MinMaxTree(GameTreeBase):
    def __init__(self, generator: Optional[random.Random] = None, eps: float = 0):
        if generator is None:
            generator = random.Random()

        self.root = Node("empty", reward=0)
        self._generator = generator
        self._eps = eps

    def build_from_env(self, tic_tac_env: TicTacToe):
        tic_tac_env.reset()
        total_cells = tic_tac_env.n_cols * tic_tac_env.n_rows

        total_nodes = compute_total_nodes(total_cells, 1, total_cells)

        progress = trange(total_nodes, miniters=10_000)
        r_build_minmax_tree(tic_tac_env, self.root, progress)

    def dump(self, path_to_file: str) -> None:
        with open(path_to_file, "wb") as dump_file:
            pickle.dump(self, dump_file, protocol=PICKLE_PROTOCOL)

    @ staticmethod
    def load_from_dump(path_to_file: str) -> "MinMaxTree":
        with open(path_to_file, "rb") as dump_file:
            return pickle.load(dump_file)

    def _minmax_tt(self, node: Node, alpha: Optional[float], beta: Optional[float], is_max: bool) -> Tuple[int, Node]:
        if node.is_leaf:
            return node.reward, node

        # Random action
        if self._generator.random() < self._eps:
            best_node = self._generator.choice(node.children)
            return 0, best_node

        max_func = lt

        if is_max:
            max_func = gt

        children_iter = iter(node.children)

        first_node = next(children_iter)
        best_node = first_node
        best_score, _ = self._minmax_tt(first_node, alpha, beta, not is_max)

        if is_max:
            alpha = best_score
        else:
            beta = best_score

        for child_node in children_iter:
            score, _ = self._minmax_tt(child_node, alpha, beta, not is_max)

            if max_func(score, best_score):
                best_score = score
                best_node = child_node
            elif score == best_score:
                if self._generator.random() < 0.5:
                    best_score = score
                    best_node = child_node

            if is_max:
                if beta is not None and best_score > beta:
                    break
                alpha = max(alpha, best_score)
            else:
                if alpha is not None and best_score < alpha:
                    break
                beta = min(beta, best_score)

        return best_score, best_node

    def find_game_state(self, prev_state: Node, env_state: str):
        for node in prev_state.children:
            if node.name == env_state:
                return node
        raise RuntimeError(f"Cannot find state from root node: {prev_state.name}")

    def transit_to_state(self, prev_node, env_state: str, env: TicTacToe) -> Node:
        return self.find_game_state(prev_node, env_state)

    def best_move(self, current_state: Node, env: TicTacToe, is_max: bool) -> Tuple[ActionType, Node]:
        _, best_node = self._minmax_tt(current_state, None, None, is_max)
        return best_node.step, best_node
