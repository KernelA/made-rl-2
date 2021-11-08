from typing import Tuple
from operator import ge, le
import pickle

from anytree import Node

from .env import TicTacToe
from .contstants import PICKLE_PROTOCOL

StepType = Tuple[int]


class MinMaxTree:
    def __init__(self):
        self.root = Node("empty", reward=0)
        self.node_hash_table = dict()

    @staticmethod
    def build_from_env(tic_tac_env: TicTacToe) -> "MinMaxTree":
        tree = MinMaxTree()
        r_build_minmax_tree(tic_tac_env, tree.root, tree.node_hash_table)

        return tree

    def dump(self, path_to_file: str) -> None:
        with open(path_to_file, "wb") as dump_file:
            pickle.dump(self, dump_file, protocol=PICKLE_PROTOCOL)

    @staticmethod
    def load_from_dump(path_to_file: str) -> "MinMaxTree":
        with open(path_to_file, "rb") as dump_file:
            return pickle.load(dump_file)


def build_minmax_tree(tic_tac_env: TicTacToe) -> Tuple[Node, dict]:
    root = Node("root", reward=0)
    node_hash_table = dict()
    r_build_minmax_tree(tic_tac_env, root, node_hash_table)
    return root, node_hash_table


def r_build_minmax_tree(tic_tac_env: TicTacToe, parent_node: Node, hash_table: dict):
    positions = tic_tac_env.getEmptySpaces()

    for pos in positions:
        new_env = tic_tac_env.clone()
        node_name, reward, is_end = new_env.step(pos)
        node = Node(node_name[0], parent=parent_node, reward=reward, step=tuple(pos))
        hash_table[node.name] = node

        if not is_end:
            r_build_minmax_tree(new_env, node, hash_table)


def minmax_tt(node: Node, is_max: bool) -> Tuple[int, StepType]:
    if node.is_leaf:
        return node.reward, node.step

    max_func = le

    if is_max:
        max_func = ge

    scores = (minmax_tt(child_node, not is_max) for child_node in node.children)

    iter_score = iter(scores)
    best_score, best_step = next(scores)

    for score, step in iter_score:
        if max_func(score, best_score):
            best_score = score
            best_step = step

    return best_score, best_step


def best_move(node: Node, is_max: bool) -> StepType:
    return minmax_tt(node, is_max)[1]
