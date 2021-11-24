from torch import nn
from torch import functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, n_row: int, n_cols: int):
        self._feature_extartcor = []

