import torch

from tictac_rl.env import TicTacToe
from tictac_rl.nn import board_state2batch, board_state_str2batch

def test_transform(env_3x3: TicTacToe):
    tensor = board_state2batch(env_3x3.board)
    assert tensor.shape == (1, 1, env_3x3.n_rows, env_3x3.n_cols)
    assert tensor.dtype == torch.get_default_dtype()

def test_transform_batch():
    states = ["100000000", "100000002"]

    batch = board_state_str2batch(states, 3, 3)
    assert batch.ndim == 4
    assert batch.shape[0] == len(states)
    assert batch.shape[1] == 1

