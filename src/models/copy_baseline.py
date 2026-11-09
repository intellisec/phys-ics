import torch
from torch import nn


class CopyLastBaseline1d(nn.Module):
    """
    Copy the last time step for all features.

    Input:  (batch, window, num_features)
    Output: (batch, outsize) with outsize == num_features
    """

    def __init__(self, num_features: int, outsize: int, window: int, **kwargs):
        super().__init__()
        assert outsize == num_features, (
            f"CopyLastBaseline1d expects outsize == num_features, "
            f"got outsize={outsize}, num_features={num_features}"
        )
        self.num_features = num_features
        self.outsize = outsize
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, num_features)
        last_step = x[:, -1, :]  # (batch, num_features)
        return last_step  # (batch, outsize)
