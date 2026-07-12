"""Causal price/volume features and fold-local normalization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


RETURN_HORIZONS = (1, 5, 10, 20, 60)
VOL_HORIZONS = (5, 20, 60)


def _canonical_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy OHLCV columns into canonical lower-case names."""
    lookup = {str(column).lower(): column for column in frame.columns}
    missing = [name for name in ("open", "high", "low", "close", "volume") if name not in lookup]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    out = frame[[lookup[name] for name in ("open", "high", "low", "close", "volume")]].copy(deep=True)
    out.columns = ["open", "high", "low", "close", "volume"]
    return out.astype(float)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    relative = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative)
    return rsi.where(loss.ne(0.0), 100.0).where(gain.ne(0.0), 0.0)


def build_asset_features(frame: pd.DataFrame, spy: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build features at time t using data no later than t.

    Output is intentionally not shifted: decisions may observe bar-t close, while
    execution layer must enforce fills no earlier than bar t+1.
    """
    bars = _canonical_ohlcv(frame)
    close, volume = bars["close"], bars["volume"]
    one_day = close.pct_change(fill_method=None)
    features = pd.DataFrame(index=bars.index)
    for horizon in RETURN_HORIZONS:
        features[f"return_{horizon}"] = close.pct_change(horizon, fill_method=None)
        features[f"momentum_{horizon}"] = close / close.shift(horizon) - 1.0
    features["rsi_14"] = _rsi(close) / 100.0
    mean20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    features["bollinger_z_20"] = (close - mean20) / std20.replace(0.0, np.nan)
    features["bollinger_width_20"] = 4.0 * std20 / mean20.replace(0.0, np.nan)
    for horizon in VOL_HORIZONS:
        features[f"volatility_{horizon}"] = one_day.rolling(horizon, min_periods=horizon).std(ddof=0)
    features["volume_change_1"] = volume.pct_change(fill_method=None)
    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std(ddof=0)
    features["volume_z_20"] = (volume - volume_mean) / volume_std.replace(0.0, np.nan)

    if spy is not None:
        spy_close = _canonical_ohlcv(spy)["close"].reindex(bars.index)
        spy_return = spy_close.pct_change(fill_method=None)
        features["relative_return_1"] = one_day - spy_return
        for horizon in (5, 20, 60):
            features[f"relative_momentum_{horizon}"] = (
                close / close.shift(horizon) - spy_close / spy_close.shift(horizon)
            )
        features["market_corr_20"] = one_day.rolling(20, min_periods=20).corr(spy_return)
        market_var = spy_return.rolling(20, min_periods=20).var(ddof=0)
        covariance = one_day.rolling(20, min_periods=20).cov(spy_return, ddof=0)
        features["market_beta_20"] = covariance / market_var.replace(0.0, np.nan)

    return features.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class TrainingFoldScaler:
    """Immutable normalization statistics fitted on training rows only."""

    means: Mapping[str, float]
    stds: Mapping[str, float]
    feature_names: tuple[str, ...]
    fingerprint: str

    @classmethod
    def fit(cls, training: pd.DataFrame | Mapping[str, pd.DataFrame]) -> "TrainingFoldScaler":
        frames = list(training.values()) if isinstance(training, Mapping) else [training]
        if not frames:
            raise ValueError("training features cannot be empty")
        columns = tuple(frames[0].columns)
        if not columns or any(tuple(frame.columns) != columns for frame in frames):
            raise ValueError("all assets must have identical non-empty feature columns")
        combined = pd.concat([frame.loc[:, columns] for frame in frames], axis=0)
        means = combined.mean(axis=0, skipna=True)
        stds = combined.std(axis=0, skipna=True, ddof=0).replace(0.0, 1.0).fillna(1.0)
        means = means.fillna(0.0)
        payload = {
            "features": list(columns),
            "means": [float(means[name]) for name in columns],
            "stds": [float(stds[name]) for name in columns],
            "rows": int(len(combined)),
        }
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return cls(
            MappingProxyType({name: float(means[name]) for name in columns}),
            MappingProxyType({name: float(stds[name]) for name in columns}),
            columns,
            fingerprint,
        )

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if tuple(features.columns) != self.feature_names:
            raise ValueError("feature schema differs from fitted training schema")
        means = pd.Series(dict(self.means), index=self.feature_names)
        stds = pd.Series(dict(self.stds), index=self.feature_names)
        return (features.copy(deep=True) - means) / stds

