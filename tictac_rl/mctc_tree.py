from logging import root
import math
from typing import Optional, Sequence, Tuple
import random

from anytree import NodeMixin


from .env import TicTacToe, CROSS_PLAYER, CIRCLE_PLAYER, DRAW, ActionType
from .base_tree import GameTreeBase


class MCTSNode(NodeMixin):
    CONSTANT = math.sqrt(2)

    def __init__(self, *,
                 name: str,
                 step: Tuple[int],
                 is_terminal: bool,
                 reward: Optional[int],
                 generator: random.Random = None,
                 parent: Optional["MCTSNode"] = None,
                 children=None) -> None:
        super().__init__()
        self.name = name
        self.parent = parent
        self.step = step
        self.num_visits = 0
        self._num_wins = 0
        self.is_terminal = is_terminal
        self.reward = reward
        self._child_table = dict()

        if generator is None:
            generator = random.Random()

        self._generator = generator

        if children:
            self.children = children

    def __str__(self):
        str_repr = f"{self.name} {self._num_wins}/{self.num_visits}"
        if self.is_terminal:
            str_repr += f"Winner: {self.reward}"
        return str_repr

    def _post_attach_children(self, children):
        super()._post_attach_children(children)
        for node in self.children:
            self._child_table[node.name] = node

    def _post_attach(self, parent):
        parent._post_attach_children(parent.children)

    def get_child_node(self, name: str) -> Optional["MCTSNode"]:
        return self._child_table.get(name, None)

    def get_ucb1(self, start_node_name: str) -> float:
        if self.num_visits == 0:
            return math.inf

        parent_node = self.parent
        total_simulations = 0

        while parent_node.name != start_node_name:
            total_simulations += parent_node.num_visits
            parent_node = parent_node.parent

        total_simulations += parent_node.num_visits

        return self._num_wins / self.num_visits + self.CONSTANT * math.sqrt(math.log(total_simulations) / self.num_visits)

    def get_value(self):
        if self.num_visits == 0:
            return 0
        return self._num_wins / self.num_visits


class MCTS(GameTreeBase):
    def __init__(self, env: TicTacToe, generator: Optional[random.Random] = None, eps: float = 0, depth_limit: Optional[int] = None):
        if generator is None:
            generator = random.Random()
        self._generator = generator
        self.root = MCTSNode(name="empty", step=None, is_terminal=False,
                             reward=None, generator=self._generator)
        self._eps = eps
        self._depth_limit = depth_limit
        self._add_start_nodes(env.clone())

    def _add_start_nodes(self, env: TicTacToe):
        for pos in env.getEmptySpaces():
            cur_env = env.clone()
            (board_state, *_), reward, is_end = cur_env.step(pos)
            new_state_node = MCTSNode(name=board_state, step=pos,
                                      is_terminal=is_end, reward=reward, parent=self.root)

    def build_from_env(self, env: TicTacToe):
        raise NotImplementedError()

    def find_game_state(self, prev_state: NodeMixin, env_state: str):
        for node in prev_state.children:
            if node.name == env_state:
                return node
        raise RuntimeError(f"Cannot find state from root node: {prev_state.name}")

    def transit_to_state(self, prev_node: "MCTSNode", env_state: str, env: TicTacToe) -> "MCTSNode":
        new_node = prev_node.get_child_node(env_state)

        if new_node is None:
            new_env = env.from_state_str(prev_node.name)
            for pos in new_env.getEmptySpaces():
                cur_env = new_env.clone()
                (board_state, *_), reward, is_end = cur_env.step(pos)
                new_state_node = MCTSNode(name=board_state, step=pos,
                                          is_terminal=is_end, reward=reward, parent=prev_node)
                if board_state == env_state:
                    new_node_start = new_state_node
        else:
            new_node_start = new_node

        assert new_node_start is not None

        return new_node_start

    def _find_best_child(self, children, root_name: str, func):
        if self._generator.random() < self._eps:
            return self._generator.choice(children)

        ucb_1 = tuple(func(child, root_name) for child in children)
        max_value = max(ucb_1)

        best_children = [children[i] for i in range(len(ucb_1)) if ucb_1[i] == max_value]
        return self._generator.choice(best_children)

    def _selection(self, env: TicTacToe, root_node: "MCTSNode", is_max: bool) -> Tuple[ActionType, MCTSNode]:
        root_node = root_node
        l_node = root_node

        if l_node is self.root:
            is_max = False

        while not l_node.is_leaf:
            is_max = not is_max
            l_node = self._find_best_child(
                l_node.children, root_node.name, lambda loc_node, root_name: loc_node.get_ucb1(root_name))

        if not l_node.is_terminal:
            self._expansion(env, root_node, l_node, is_max)
        else:
            self._backpropogation(env._start_player, l_node.reward, l_node, root_node, is_max)

        best_move = self._find_best_child(
            root_node.children, root_node.name, lambda loc_node, root_name: loc_node.get_value())

        return best_move.step, best_move

    def best_move(self, game_state_node: MCTSNode, env: TicTacToe, is_max: bool) -> Tuple[ActionType, MCTSNode]:
        return self._selection(env, game_state_node, True)

    def _expansion(self, env: TicTacToe, root_node: "MCTSNode", l_node: "MCTSNode", is_max: bool):
        if l_node is root_node:
            new_env = env.clone()
        else:
            new_env = env.from_state_str(l_node.name)

        for pos in new_env.getEmptySpaces():
            curr_env = new_env.clone()
            state, reward, is_end = curr_env.step(pos)
            node = MCTSNode(name=state[0], step=pos, reward=reward,
                            is_terminal=is_end, generator=self._generator, parent=l_node)

        c_node = self._choose_node(l_node.children)

        if c_node.is_terminal:
            winner = c_node.reward
        else:
            winner = self._simulation(env.from_state_str(c_node.name))
        self._backpropogation(env.curTurn, winner, c_node, root_node, not is_max)

    def _choose_node(self, nodes: Sequence["MCTSNode"]) -> "MCTSNode":
        return self._generator.choice(nodes)

    def _make_move(self, env: TicTacToe, free_space) -> Tuple[int]:
        return self._generator.choice(free_space)

    def _simulation(self, env: TicTacToe) -> int:
        free_space = env.getEmptySpaces()
        is_end = False

        while not is_end:
            pos = self._make_move(env, free_space)
            state, reward, is_end = env.step(pos)
            free_space = state[1]

        return reward

    def _backpropogation(self, start_player: int,
                         winner: int,
                         c_node: "MCTSNode",
                         root_node: "MCTSNode",
                         is_max: bool):
        c_node.num_visits += 1

        win_count = 1 if (start_player == winner or winner == DRAW) and is_max else 0
        c_node._num_wins += win_count

        is_max = not is_max
        parent_node: MCTSNode = c_node.parent

        while parent_node.name != root_node.name:
            parent_node.num_visits += 1

            if is_max:
                parent_node._num_wins += win_count

            is_max = not is_max
            parent_node = parent_node.parent

        parent_node.num_visits += 1

        if is_max:
            parent_node._num_wins += win_count
