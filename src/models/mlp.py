from torch import nn


class MLPForecaster(nn.Module):
    """
    Sequence-to-one forecaster that simply flattens the input window.

    x: (B, S, F)  ->  y: (B, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        window: int,  # window length S
        hidden: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.seq_len = window
        self.out_dim = out_dim

        flat_dim = in_dim * window  # total features after flattening

        self.proj = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # x: (B, S, F) with S == self.seq_len
        feats = x.flatten(start_dim=1)  # (B, S·F)
        return self.proj(feats)
