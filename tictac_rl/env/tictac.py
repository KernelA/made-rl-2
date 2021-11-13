import enum
from typing import Optional

from numba import jit
import numpy as np


CIRCLE_PLAYER = -1
CROSS_PLAYER = 1
DRAW = 0


@jit(nopython=True, parallel=True)
def is_terminal(board: np.ndarray, current_turn: int, n_win: int) -> Optional[int]:
    # проверим, не закончилась ли игра
    cur_marks, cur_p = np.where(board == current_turn), current_turn

    n_rows = board.shape[0]
    n_cols = board.shape[1]

    for i, j in zip(cur_marks[0], cur_marks[1]):
        win = False
        if i <= n_rows - n_win:
            if np.all(board[i: i + n_win, j] == cur_p):
                win = True
        if not win:
            if j <= n_cols - n_win:
                if np.all(board[i, j: j + n_win] == cur_p):
                    win = True
        if not win:
            if i <= n_rows - n_win and j <= n_cols - n_win:
                if np.all(np.array([board[i+k, j+k] == cur_p for k in range(n_win)])):
                    win = True
        if not win:
            if i <= n_rows - n_win and j >= n_win - 1:
                if np.all(np.array([board[i+k, j-k] == cur_p for k in range(n_win)])):
                    win = True
        if win:
            return current_turn

    if len(get_empty_space(board)) == 0:
        return DRAW

    return None


@jit(nopython=True)
def get_empty_space(board: np.ndarray) -> np.ndarray:
    res = np.where(board == 0)
    return np.vstack(res).astype(np.uint16).T


class TicTacToe:
    def __init__(self, n_rows: int, n_cols: int, n_win: int, start_player: int = CROSS_PLAYER):
        assert start_player in (
            CROSS_PLAYER, CIRCLE_PLAYER), "A player mus be in -1 (circle) or 1 (cross)"
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_win = n_win
        self._start_player = start_player
        self.reset()

    def __deepcopy__(self) -> "TicTacToe":
        new_env = TicTacToe(self.n_rows, self.n_cols, self.n_win, self._start_player)
        new_env.board = self.board.copy()
        new_env.curTurn = self.curTurn
        new_env.emptySpaces = None
        new_env.boardHash = None

        return new_env

    def clone(self) -> "TicTacToe":
        return self.__deepcopy__()

    def getEmptySpaces(self) -> np.ndarray:
        if self.emptySpaces is None:
            self.emptySpaces = get_empty_space(self.board)
        return self.emptySpaces

    def _makeMove(self, player, i: int, j: int) -> None:
        self.board[i, j] = player
        self.emptySpaces = None
        self.boardHash = None

    def _change_turn(self):
        self.curTurn = -self.curTurn

    def _getHash(self):
        if self.boardHash is None:
            self.boardHash = ''.join(
                f"{x + 1}" for x in self.board.reshape(self.n_rows * self.n_cols))
        return self.boardHash

    def isTerminal(self):
        # проверим, не закончилась ли игра
        return is_terminal(self.board, self.curTurn, self.n_win)

    def _str_repr(self, board):
        board_repr = ""
        for i in range(self.n_rows):
            board_repr += "\n" + '----' * (self.n_cols) + '-'
            out = '| '
            for j in range(self.n_cols):
                if board[i, j] == 1:
                    token = 'x'
                elif board[i, j] == -1:
                    token = 'o'
                else:
                    token = ' '
                out += token + ' | '
            board_repr += f"\n{out}"
        board_repr += f"\n{'----'*(self.n_cols)}" + '-'
        return board_repr

    def getState(self):
        return (self._getHash(), self.getEmptySpaces(), self.curTurn)

    def action_from_int(self, action_int):
        return (int(action_int / self.n_cols), int(action_int % self.n_cols))

    def int_from_action(self, action):
        return action[0] * self.n_cols + action[1]

    def step(self, action):
        if self.board[action[0], action[1]] != 0:
            return self.getState(), -10, True
        self._makeMove(self.curTurn, action[0], action[1])
        reward = self.isTerminal()
        self._change_turn()
        return self.getState(), 0 if reward is None else reward, reward is not None

    def from_state_str(self, state_str: str) -> "TicTacToe":
        new_env = self.clone()
        new_env.board = np.array(tuple(map(int, state_str)), dtype=self.board.dtype) - 1

        counts = new_env.board.size - np.count_nonzero(new_env.board == np.int8(0))
        new_env.curTurn = int(self._start_player)

        if counts % 2 == 1:
            new_env._change_turn()

        new_env.board = new_env.board.reshape(self.n_rows, self.n_cols)

        return new_env

    def __str__(self) -> str:
        return self._str_repr(self.board)

    def reset(self):
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=np.int8)
        if np.iinfo(self.board.dtype).max < self.n_win or np.iinfo(self.board.dtype).min > -self.n_win:
            raise ValueError("Incorrect dtype for board. Please specify other dtype")
        self.boardHash = None
        self.gameOver = False
        self.emptySpaces = None
        self.curTurn = int(self._start_player)
