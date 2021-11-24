import torch

from tictac_rl import board_state2batch, TicTacToe

def test_transform(env_3x3: TicTacToe):
    tensor = board_state2batch(env_3x3.board)
    assert tensor.shape == (1, 1, env_3x3.n_rows, env_3x3.n_cols)
    assert tensor.dtype == torch.get_default_dtype()
