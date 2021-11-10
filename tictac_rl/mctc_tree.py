import math
from typing import Optional, Sequence, Tuple
import random

from anytree import NodeMixin, node


from .env import TicTacToe, StartPlayer, Winner
from .min_max_tree import StepType, nodename2hash_key, nodepath


class MCTSNode(NodeMixin):
    CONSTANT = math.sqrt(2)

    def __init__(self, name, step: Tuple[int], generator: random.Random = None, parent: Optional["MCTSNode"] = None, children=None) -> None:
        super().__init__()
        self.name = name
        self.parent = parent
        self.step = step
        self.num_visits = 0
        self._num_wins = 0

        if generator is None:
            generator = random.Random()

        self._generator = generator

        if children:
            self.children = children

    def get_ucb1(self, start_node_name: str) -> float:
        if self.num_visits == 0:
            return math.inf

        parent_node = self.parent
        total_simulations = 0

        while parent_node is not None and parent_node.name != start_node_name:
            total_simulations += parent_node.num_visits
            parent_node = parent_node.parent

        return self._num_wins / self.num_visits + self.CONSTANT * math.sqrt(math.log(total_simulations) / self.num_visits)


class MCTS:
    def __init__(self, generator: Optional[random.Random] = None):
        if generator is None:
            generator = random.Random()
        self.generator = generator
        self.root = MCTSNode("empty", None, self.generator)
        self._hash_table = dict()

    def _selection(self, env: TicTacToe, game_history: Sequence[str]) -> StepType:
        node_hash = nodepath(self.root, game_history)

        root_node = self._hash_table.get(node_hash, None)
        l_node = root_node

        if root_node is not None:
            while not l_node.is_leaf:
                l_node = max(l_node.children, key=lambda x: x.get_ucb1(root_node.name))
        else:
            root_node = self.root
            l_node = root_node

        self._expansion(env, root_node, l_node)

        best_move = max(l_node.children, key=lambda x: x.get_ucb1(root_node.name))

        return best_move.step

    def _expansion(self, env: TicTacToe, root_node: "MCTSNode", l_node: "MCTSNode"):
        new_env = env.from_state_str(l_node.name)
        reward = new_env.isTerminal()

        if reward is not None:
            self._backpropogation(env.curTurn, reward, l_node, root_node)

        for pos in new_env.getEmptySpaces():
            curr_env = new_env.clone()
            state, reward, is_end = curr_env.step(pos)
            node = MCTSNode(state[0], pos, self._generator, parent=l_node)
            self._hash_table[nodename2hash_key(node)] = node

        c_node = self._generator.choice(l_node.children)
        winner = self._simulation(c_node, env)
        self._backpropogation(env.curTurn, winner, c_node, root_node)

    def _simulation(self, c_node: "MCTSNode", env: TicTacToe) -> Winner:
        new_env = env.from_state_str(c_node)
        free_space = new_env.getEmptySpaces()
        is_end = False

        while not is_end:
            pos = self._generator.choice(free_space)
            state, reward, is_end = new_env.step(pos)
            free_space = state[1]

        return reward

    def _backpropogation(self, start_player: StartPlayer, winner: StartPlayer, c_node: "MCTSNode", root_node: "MCTSNode"):
        c_node.num_visits += 1
        is_win = False

        if start_player == winner or winner == Winner.draw:
            c_node._num_wins += 1
            is_win = True

        parent_node: MCTSNode = c_node.parent

        while parent_node.name != root_node.name:
            is_win = not is_win
            parent_node.num_visits += 1

            if is_win:
                parent_node._num_wins += 1

        parent_node
        if is_win:
            parent_node._num_wins += 1
