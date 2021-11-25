from typing import Sequence
import numpy as np
import torch

from ..env import TicTacToe

def board_state2batch(board_state: np.ndarray) -> torch.Tensor:
    assert board_state.ndim == 2
    return torch.from_numpy(board_state)[None, None, ...].to(torch.get_default_dtype())

def board_state_str2batch(state_str: Sequence[str], n_rows: int, n_cols: int) -> torch.Tensor:
    return torch.cat([board_state2batch(TicTacToe.env_state_str2board(state).reshape((n_rows, n_cols))) for state in state_str], dim=0)

