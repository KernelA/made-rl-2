import copy
import enum
from typing import Optional

from numba import jit
import numpy as np


class Action(enum.IntEnum):
    circle = -1
    cross = 1


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
        return 0

    return None


@jit(nopython=True)
def get_empty_space(board: np.ndarray) -> np.ndarray:
    res = np.where(board == 0)
    return np.vstack(res).T


class TicTacToe:
    def __init__(self, n_rows: int, n_cols: int, n_win: int, clone: Optional["TicTacToe"] = None):
        if clone is not None:
            self.n_rows, self.n_cols, self.n_win = clone.n_rows, clone.n_cols, clone.n_win
            self.board = clone.board.copy()
            self.curTurn = clone.curTurn
            self.emptySpaces = None
            self.boardHash = None
        else:
            self.n_rows = n_rows
            self.n_cols = n_cols
            self.n_win = n_win

            self.reset()

    def clone(self) -> "TicTacToe":
        new_env = TicTacToe(self.n_rows, self.n_cols, self.n_win, self)
        return new_env

    def getEmptySpaces(self) -> np.ndarray:
        if self.emptySpaces is None:
            self.emptySpaces = get_empty_space(self.board)
        return self.emptySpaces

    def _makeMove(self, player, i: int, j: int) -> None:
        self.board[i, j] = player
        self.emptySpaces = None
        self.boardHash = None

    def _getHash(self):
        if self.boardHash is None:
            self.boardHash = ''.join([f"{x + 1}"
                                     for x in self.board.reshape(self.n_rows * self.n_cols)])
        return self.boardHash

    def _isTerminal(self):
        # проверим, не закончилась ли игра
        return is_terminal(self.board, self.curTurn, self.n_win)
        # cur_marks, cur_p = np.where(self.board == self.curTurn), self.curTurn

        # for i, j in zip(cur_marks[0], cur_marks[1]):
        #     win = False
        #     if i <= self.n_rows - self.n_win:
        #         if np.all(self.board[i:i+self.n_win, j] == cur_p):
        #             win = True
        #     if not win:
        #         if j <= self.n_cols - self.n_win:
        #             if np.all(self.board[i, j:j+self.n_win] == cur_p):
        #                 win = True
        #     if not win:
        #         if i <= self.n_rows - self.n_win and j <= self.n_cols - self.n_win:
        #             if np.all(np.array([self.board[i+k, j+k] == cur_p for k in range(self.n_win)])):
        #                 win = True
        #     if not win:
        #         if i <= self.n_rows - self.n_win and j >= self.n_win-1:
        #             if np.all(np.array([self.board[i+k, j-k] == cur_p for k in range(self.n_win)])):
        #                 win = True
        #     if win:
        #         self.gameOver = True
        #         return self.curTurn

        # if len(self.getEmptySpaces()) == 0:
        #     self.gameOver = True
        #     return 0

        # self.gameOver = False
        # return None

    def printBoard(self):
        for i in range(self.n_rows):
            print('----'*(self.n_cols)+'-')
            out = '| '
            for j in range(self.n_cols):
                if self.board[i, j] == 1:
                    token = 'x'
                elif self.board[i, j] == -1:
                    token = 'o'
                else:
                    token = ' '
                out += token + ' | '
            print(out)
        print('----'*(self.n_cols)+'-')

    def _getState(self):
        return (self._getHash(), self.getEmptySpaces(), self.curTurn)

    def action_from_int(self, action_int):
        return (int(action_int / self.n_cols), int(action_int % self.n_cols))

    def int_from_action(self, action):
        return action[0] * self.n_cols + action[1]

    def step(self, action):
        if self.board[action[0], action[1]] != 0:
            return self._getState(), -10, True
        self._makeMove(self.curTurn, action[0], action[1])
        reward = self._isTerminal()
        self.curTurn = -self.curTurn
        return self._getState(), 0 if reward is None else reward, reward is not None

    def reset(self):
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=np.int8)
        if np.iinfo(self.board.dtype).max < self.n_win or np.iinfo(self.board.dtype).min > -self.n_win:
            raise ValueError("Incorrect dtype for board. Please specify other dtype")
        self.boardHash = None
        self.gameOver = False
        self.emptySpaces = None
        self.curTurn = 1
