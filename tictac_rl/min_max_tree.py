from typing import Tuple, Optional
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
        hash_table[node.name] = node

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

    def _minmax_tt(self, node: Node, is_max: bool) -> Tuple[int, StepType]:
        if node.is_leaf:
            return node.reward, node.step

        max_func = lt

        if is_max:
            max_func = gt

        scores = (self._minmax_tt(child_node, not is_max) for child_node in node.children)

        iter_score = iter(scores)
        best_score, best_step = next(scores)

        for score, step in iter_score:
            if max_func(score, best_score):
                best_score = score
                best_step = step
            elif score == best_score:
                if self._generator.random() < 0.5:
                    best_score = score
                    best_step = step

        return best_score, best_step

    def best_move(self, hash_table_state: str, is_max: bool) -> StepType:
        node = self.node_hash_table.get(hash_table_state)
        if node is None:
            raise ValueError(f"Cannot find hash state for '{hash_table_state}'")

        return self._minmax_tt(node, is_max)[1]
