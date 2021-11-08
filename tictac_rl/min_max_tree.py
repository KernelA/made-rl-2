from typing import Tuple, Optional, Iterable
from operator import gt, lt
import pickle
import random

from anytree import Node

from .env import TicTacToe
from .contstants import PICKLE_PROTOCOL

StepType = Tuple[int]


def r_build_minmax_tree(tic_tac_env: TicTacToe, parent_node: Node, hash_table: dict):
    positions = tic_tac_env.getEmptySpaces()

    for pos in positions:
        new_env = tic_tac_env.clone()
        node_name, reward, is_end = new_env.step(pos)
        node = Node(node_name[0], parent=parent_node, reward=reward, step=tuple(pos))
        path = node.separator.join([""] + [str(node_path.name) for node_path in node.path])
        hash_table[path] = node

        if not is_end:
            r_build_minmax_tree(new_env, node, hash_table)


class MinMaxTree:
    def __init__(self, generator: Optional[random.Random] = None):
        if generator is None:
            generator = random.Random()

        self.root = Node("empty", reward=0)
        self.node_hash_table = dict()
        self._generator = generator

    @staticmethod
    def build_from_env(tic_tac_env: TicTacToe, generator: Optional[random.Random] = None) -> "MinMaxTree":
        tree = MinMaxTree(generator)
        r_build_minmax_tree(tic_tac_env, tree.root, tree.node_hash_table)

        return tree

    def dump(self, path_to_file: str) -> None:
        with open(path_to_file, "wb") as dump_file:
            pickle.dump(self, dump_file, protocol=PICKLE_PROTOCOL)

    @staticmethod
    def load_from_dump(path_to_file: str) -> "MinMaxTree":
        with open(path_to_file, "rb") as dump_file:
            return pickle.load(dump_file)

    def _minmax_tt(self, node: Node, alpha: Optional[float], beta: Optional[float], is_max: bool) -> Tuple[int, StepType]:
        if node.is_leaf:
            return node.reward, node.step

        max_func = lt

        if is_max:
            max_func = gt

        children_iter = iter(node.children)

        first_node = next(children_iter)
        best_step = first_node.step
        best_score, _ = self._minmax_tt(first_node, alpha, beta, not is_max)

        if is_max:
            alpha = best_score
        else:
            beta = best_score

        for child_node in children_iter:
            score, _ = self._minmax_tt(child_node, alpha, beta, not is_max)

            if max_func(score, best_score):
                best_score = score
                best_step = child_node.step
            elif score == best_score:
                if self._generator.random() < 0.5:
                    best_score = score
                    best_step = child_node.step

            if is_max:
                if beta is not None and best_score > beta:
                    break

                alpha = max(alpha, best_score)
            else:
                if alpha is not None and best_score < alpha:
                    break
                beta = min(beta, best_score)

        return best_score, best_step

    def get_node_by_step_history(self, node_history: Iterable[str]) -> Node:
        node_path = f"{self.root.separator}{self.root.name}{self.root.separator}" + \
            f"{self.root.separator}".join(node_history)
        return self.node_hash_table[node_path]

    def best_move(self, node_history: Iterable[str], is_max: bool) -> StepType:
        node = self.get_node_by_step_history(node_history)
        return self._minmax_tt(node, None, None, is_max)[1]
