from __future__ import annotations
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    Multi-step direct forecaster: feeds the last LSTM output through MLP -> horizon outputs.
    Input:  [B, lookback, n_features]
    Output: [B, horizon] (normalized target space)
    """
    def __init__(self, n_features: int, horizon: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * mult),
            nn.Linear(hidden_size * mult, hidden_size * mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * mult, horizon),
        )

    def forward(self, x):
        out, _ = self.lstm(x)          # out: [B, L, H*mult]
        last = out[:, -1, :]           # [B, H*mult]
        y = self.head(last)            # [B, horizon]
        return y
