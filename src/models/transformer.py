import torch
import math
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (same formula as the original Transformer)."""

    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  indices
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding and apply dropout. x: (seq_len, batch, d_model)."""
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TimeSeriesTransformerForecaster(nn.Module):
    """
    Single-stage temporal Transformer forecaster.

    Input  x : (batch, window, num_features)
    Output y : (batch, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_last_token: bool = True,  # else mean pooling over time
        max_len: int = 2048,
    ):
        super().__init__()

        # project feature vectors to the model dimension
        self.input_proj = nn.Linear(in_dim, d_model)

        # positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        # transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # (seq_len, batch, d_model)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # prediction head
        self.use_last_token = use_last_token
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, window, num_features)."""
        x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # -> (seq_len, batch, d_model)
        x = self.pos_enc(x)  # add positional info
        x = self.encoder(x)  # transformer encoding

        # aggregation
        if self.use_last_token:
            feat = x[-1]  # last time step
        else:
            feat = x.mean(dim=0)  # mean over time

        feat = self.norm(feat)
        feat = self.dropout(feat)
        return self.fc(feat)
