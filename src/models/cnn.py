import torch
from torch import nn


class TemporalCNN1d(nn.Module):
    """
    Scans over the window dimension (treated as length) and
    uses AdaptiveAvgPool1d to end with length-1. Each feature stays
    a separate channel until the final MLP fuses them.

    Input:  (batch, window, num_features)
    Output: (batch, outsize)
    """

    def __init__(
        self,
        num_features: int,
        outsize: int,  # output dim (= num targets)
        window: int = 5,
        channels=(64, 128, 256),  # conv widths
        fc_size: int = 256,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        layers, in_ch = [], num_features  # each feature is its own channel
        for ch in channels:
            layers += [
                nn.Conv1d(
                    in_ch, ch, kernel_size=2, stride=1, padding=0
                ),  # slide over time
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ]
            in_ch = ch

        # Collapse the (now-short) time axis to length-1
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.temporal_conv = nn.Sequential(*layers)

        # After adaptive pool the tensor is (batch, channels[-1], 1)
        flat_dim = channels[-1]

        self.fc = nn.Sequential(
            nn.Flatten(),  # (batch, flat_dim)
            nn.Linear(flat_dim, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc_size, outsize),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, num_features) -> (batch, num_features, window)
        x = x.permute(0, 2, 1)
        x = self.temporal_conv(x)  # (batch, channels[-1], 1)
        return self.fc(x)  # (batch, outsize)
