import torch
from tictac_rl import QNetwork, TicTacToe, board_state2batch

@torch.no_grad()
def test_q_netqork(env_3x3: TicTacToe):
    net = QNetwork(env_3x3.n_rows, env_3x3.n_cols)
    net.eval()

    batch = board_state2batch(env_3x3.board)

    q_values = net(batch)
    assert q_values.shape[0] == batch.shape[0]
    assert q_values.shape[1] == env_3x3.n_cols * env_3x3.n_cols
