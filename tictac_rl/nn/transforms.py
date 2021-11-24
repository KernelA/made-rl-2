import numpy as np
import torch

def board_state2batch(board_state: np.ndarray):
    assert board_state.ndim == 2
    return torch.from_numpy(board_state)[None, None, ...].to(torch.get_default_dtype())
