import torch
import torch.nn as nn


class TimeSeriesLSTMForecaster(nn.Module):
    """
    Two-stage temporal LSTM forecaster.

    Stage 1: encodes the raw sequence (window long) into a hidden representation.
    Stage 2: processes that representation again to capture higher-order
             temporal patterns.

    Input  x : (batch, window, num_features)
    Output y : (batch, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes=(256, 512),  # (stage1_hidden, stage2_hidden)
        dropout: float = 0.2,
        layernorm: bool = False,
    ):
        super().__init__()

        h1, h2 = hidden_sizes

        # first temporal stage
        self.stage1 = nn.LSTM(
            input_size=in_dim,
            hidden_size=h1,
            num_layers=1,
            batch_first=True,
        )

        # second temporal stage
        self.stage2 = nn.LSTM(
            input_size=h1,
            hidden_size=h2,
            num_layers=1,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(h2) if layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(h2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B, window, num_features)."""
        # stage-1 LSTM over the raw sequence
        out1, _ = self.stage1(x)  # (B, window, h1)

        # stage-2 LSTM over the encoded sequence
        out2, (h2, _) = self.stage2(out1)  # h2: (1, B, h2)

        # final hidden state -> prediction
        feat = self.norm(h2[-1])  # (B, h2)
        feat = self.dropout(feat)
        return self.fc(feat)
