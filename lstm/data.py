import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_ticker_universe(tickers_file: str) -> List[str]:
    """Read the comma separated ticker universe file."""
    path = Path(tickers_file)
    if not path.exists():
        raise FileNotFoundError(f"Ticker list not found: {tickers_file}")

    with path.open("r") as handle:
        contents = handle.read().strip()

    tickers = [token.strip().upper() for token in contents.split(",") if token.strip()]
    logger.info("Loaded %d tickers from %s", len(tickers), tickers_file)
    return tickers


def add_forward_return_target(df: pd.DataFrame, return_col: str = "weekly_return",
                              target_col: str = "target_return") -> pd.DataFrame:
    """Create a one-week ahead return target per ticker."""
    df = df.copy()
    df[target_col] = (
        df.sort_values(["ticker", "Date"])
        .groupby("ticker")[return_col]
        .shift(-1)
    )
    return df


def sanitize_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """
    Fill missing/inf values with industry standard forward/backward fill per ticker.

    Missing values at both ends are set to zero so that padding is neutral during scaling.
    """
    df = df.copy()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = (
        df.sort_values(["ticker", "Date"])
        .groupby("ticker")[feature_cols]
        .transform(lambda g: g.fillna(method="ffill").fillna(method="bfill"))
    )
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def determine_time_split(df: pd.DataFrame, train_ratio: float = 0.8,
                         cutoff_date: Optional[datetime] = None) -> datetime:
    """Determine the chronological cutoff between train/validation segments."""
    if cutoff_date is not None:
        return cutoff_date

    unique_dates = np.array(sorted(df["Date"].unique()))
    cutoff_index = int(len(unique_dates) * train_ratio)
    cutoff_index = np.clip(cutoff_index, 1, len(unique_dates) - 1)
    cutoff_date = pd.Timestamp(unique_dates[cutoff_index])
    logger.info("Time-based split cutoff resolved to %s", cutoff_date.date())
    return cutoff_date


@dataclass
class SequenceSample:
    sequence: np.ndarray
    mask: np.ndarray
    length: int
    target: float
    ticker: str
    asof_date: pd.Timestamp


class MaskedStockDataset(Dataset):
    """
    Converts panel data into left-padded sequences with explicit masks.

    Each sample represents up to `sequence_length` chronological observations with the
    target being the forward (next week) return.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: Sequence[str], target_col: str,
                 sequence_length: int = 32) -> None:
        self.sequence_length = sequence_length
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.num_features = len(feature_cols)
        self.samples: List[SequenceSample] = []
        self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> None:
        df = df.sort_values(["ticker", "Date"])

        for ticker, group in df.groupby("ticker"):
            features = group[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
            targets = group[self.target_col].to_numpy(dtype=np.float32, copy=True)
            dates = group["Date"].to_numpy(copy=True)

            if len(group) == 0:
                continue

            for current_idx in range(len(group)):
                target_value = targets[current_idx]
                if np.isnan(target_value):
                    continue  # cannot train on the last observation per ticker

                seq_start = max(0, current_idx - self.sequence_length + 1)
                seq = features[seq_start:current_idx + 1]
                actual_len = seq.shape[0]

                if actual_len == 0:
                    continue

                padded = np.zeros((self.sequence_length, self.num_features), dtype=np.float32)
                mask = np.zeros(self.sequence_length, dtype=np.float32)
                padded[-actual_len:] = seq
                mask[-actual_len:] = 1.0

                self.samples.append(
                    SequenceSample(
                        sequence=padded,
                        mask=mask,
                        length=actual_len,
                        target=float(target_value),
                        ticker=str(ticker),
                        asof_date=pd.Timestamp(dates[current_idx]),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SequenceSample:
        return self.samples[idx]


def masked_collate_fn(batch: Sequence[SequenceSample]) -> Dict[str, np.ndarray]:
    sequences = np.stack([sample.sequence for sample in batch], axis=0)
    masks = np.stack([sample.mask for sample in batch], axis=0)
    lengths = np.array([sample.length for sample in batch], dtype=np.int64)
    targets = np.array([sample.target for sample in batch], dtype=np.float32)
    tickers = [sample.ticker for sample in batch]
    asof_dates = [sample.asof_date for sample in batch]

    return {
        "sequences": sequences,
        "masks": masks,
        "lengths": lengths,
        "targets": targets,
        "tickers": tickers,
        "asof_dates": asof_dates,
    }


def dump_metadata(metadata: Dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)


def load_metadata(metadata_path: str) -> Dict:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with path.open("r") as handle:
        return json.load(handle)
