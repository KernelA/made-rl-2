import math

from torch import nn
from torch import functional as F

def conv_output_size(input_size: int, padding: int, kernel_size: int, stride: int):
    return math.floor((input_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)

class QNetwork(nn.Module):
    def __init__(self, n_row: int, n_cols: int):
        super().__init__()
        self._feature_extractor = []

        input_size = n_row

        out_channels = 8
        in_channels = 1
        last_out_channels = out_channels

        while input_size > 1:
            self._feature_extractor.append(nn.Conv2d(in_channels, out_channels, kernel_size=n_row))
            input_size = conv_output_size(n_row, padding=0, kernel_size=n_row, stride=1)
            in_channels = out_channels
            last_out_channels = out_channels
            out_channels *= 2

        self._feature_extractor.append(nn.Flatten())

        linear_output = last_out_channels * 2

        self._feature_extractor = nn.Sequential(*self._feature_extractor)
        self._action_predictor = nn.Sequential(
            nn.Linear(last_out_channels, out_features=linear_output),
            nn.GELU(),
            nn.Linear(linear_output, n_row * n_cols),
        )

    def forward(self, batch_states):
        """Return Q values of all actions

            batch_states [B x 1 x n_row x n_col]
        """
        features = self._feature_extractor(batch_states)

        return self._action_predictor(features)

    def best_action(self, batch_states):
        return self.forward(batch_states).argmax(dim=-1)

