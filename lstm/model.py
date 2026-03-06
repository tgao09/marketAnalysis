from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class LSTMConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False


class StockLSTM(nn.Module):
    """
    Mask-aware LSTM forecaster that consumes padded sequences and lengths.

    The model relies on `pack_padded_sequence` so that padded timesteps never
    contribute to the gradient, satisfying the masking requirement.
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        final_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.regressor = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Dropout(config.dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, 1),
        )

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        sequences: (batch, seq_len, input_dim)
        lengths: actual (pre-pad) lengths per batch item
        """
        lengths = lengths.clamp(min=1)
        packed = pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)

        if self.config.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]

        prediction = self.regressor(last_hidden).squeeze(-1)
        return prediction

    def get_config_dict(self) -> Dict:
        return {
            "input_dim": self.config.input_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "bidirectional": self.config.bidirectional,
        }
