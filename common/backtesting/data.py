"""Single-asset yfinance source and normalized OHLCV data contract."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yfinance as yf

from ..yfinance_utils import configure_yfinance_cache


configure_yfinance_cache()


HistoryFetcher = Callable[..., pd.DataFrame]


def canonical_column_name(name: object) -> str:
    """Normalize yfinance-style column names without changing values."""

    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, canonical, single-asset bars safe for target creation."""

    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame.")
    if bars.empty:
        raise ValueError("bars is empty.")
    if isinstance(bars.columns, pd.MultiIndex):
        raise ValueError("MultiIndex columns are unsupported; run one symbol per backtest.")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must use a DatetimeIndex.")
    if bars.index.hasnans:
        raise ValueError("bars index contains missing timestamps.")
    if bars.index.has_duplicates:
        raise ValueError("bars index must be unique.")

    normalized = bars.copy(deep=True).sort_index()
    columns = [canonical_column_name(column) for column in normalized.columns]
    if not all(columns) or len(columns) != len(set(columns)):
        raise ValueError("bars columns become empty or duplicate after normalization.")
    normalized.columns = columns

    if "close" not in normalized.columns:
        raise ValueError("bars must contain a close column.")
    close = pd.to_numeric(normalized["close"], errors="coerce")
    close_values = close.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(close_values).all() or (close_values <= 0).any():
        raise ValueError("close must contain only finite positive values.")
    normalized["close"] = close_values
    return normalized


def fingerprint_bars(bars: pd.DataFrame) -> str:
    """Stable data fingerprint retained with a result for reproducibility."""

    digest = hashlib.sha256()
    digest.update("|".join(map(str, bars.columns)).encode())
    digest.update("|".join(map(str, bars.dtypes)).encode())
    row_hashes = pd.util.hash_pandas_object(bars, index=True).to_numpy(dtype=np.uint64)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MarketData:
    """Normalized bars plus source metadata used by ``BacktestEngine``."""

    bars: pd.DataFrame
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class YFinanceSource:
    """Explicit adjusted-price yfinance adapter for one symbol."""

    symbol: str
    period: str | None = "5y"
    interval: str = "1d"
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    auto_adjust: bool = True
    history_fetcher: HistoryFetcher | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if not isinstance(self.interval, str) or not self.interval.strip():
            raise ValueError("interval must be a non-empty string.")
        if not self.auto_adjust:
            raise ValueError("auto_adjust must be True so close has one adjusted-price meaning.")
        if self.start is None and self.end is None and not self.period:
            raise ValueError("period is required when start and end are omitted.")

    def load(self) -> MarketData:
        kwargs: dict[str, Any] = {"interval": self.interval, "auto_adjust": True}
        if self.start is None and self.end is None:
            kwargs["period"] = self.period
        else:
            if self.start is not None:
                kwargs["start"] = self.start
            if self.end is not None:
                kwargs["end"] = self.end

        try:
            if self.history_fetcher is None:
                raw = yf.Ticker(self.symbol).history(**kwargs)
            else:
                raw = self.history_fetcher(self.symbol, **kwargs)
            bars = normalize_bars(raw)
        except Exception as exc:
            request = ", ".join(f"{key}={value}" for key, value in kwargs.items())
            raise RuntimeError(
                f"Unable to fetch usable yfinance bars for {self.symbol} ({request}). "
                "Check network/DNS access, Yahoo availability, and requested date range."
            ) from exc
        metadata = {
            "source": "yfinance",
            "symbol": self.symbol,
            "period": self.period if self.start is None and self.end is None else None,
            "interval": self.interval,
            "start": None if self.start is None else str(self.start),
            "end": None if self.end is None else str(self.end),
            "auto_adjust": True,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "yfinance_version": getattr(yf, "__version__", "unknown"),
            "data_hash": fingerprint_bars(bars),
        }
        return MarketData(bars=bars, metadata=metadata)


@dataclass(frozen=True)
class YFinancePanelSource:
    """One primary ticker plus auxiliary adjusted-close series."""

    primary_symbol: str
    auxiliaries: Mapping[str, str]
    period: str | None = "5y"
    interval: str = "1d"
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    history_fetcher: HistoryFetcher | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.auxiliaries:
            raise ValueError("auxiliaries must contain at least one named symbol.")
        reserved = {"close", "open", "high", "low", "volume"}
        names = [canonical_column_name(name) for name in self.auxiliaries]
        if len(names) != len(set(names)) or any(name in reserved or not name for name in names):
            raise ValueError("auxiliary names must be unique and cannot replace primary OHLCV columns.")

    def load(self) -> MarketData:
        source_args = {
            "period": self.period,
            "interval": self.interval,
            "start": self.start,
            "end": self.end,
            "history_fetcher": self.history_fetcher,
        }
        primary = YFinanceSource(self.primary_symbol, **source_args).load()
        bars = primary.bars.copy(deep=True)
        symbols = {"primary": primary.metadata["symbol"]}
        hashes = {"primary": primary.metadata["data_hash"]}
        for name, symbol in self.auxiliaries.items():
            try:
                auxiliary = YFinanceSource(symbol, **source_args).load()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Unable to fetch auxiliary {symbol} ({name}) for {self.primary_symbol}."
                ) from exc
            column = canonical_column_name(name)
            if self.interval.endswith("d"):
                primary_dates = bars.index.tz_localize(None).normalize() if bars.index.tz is not None else bars.index.normalize()
                auxiliary_dates = (
                    auxiliary.bars.index.tz_localize(None).normalize()
                    if auxiliary.bars.index.tz is not None
                    else auxiliary.bars.index.normalize()
                )
                auxiliary_close = pd.Series(
                    auxiliary.bars["close"].to_numpy(),
                    index=auxiliary_dates,
                )
                auxiliary_close = auxiliary_close.loc[~auxiliary_close.index.duplicated(keep="last")]
                bars[column] = auxiliary_close.reindex(primary_dates).ffill().to_numpy()
            else:
                bars[column] = auxiliary.bars["close"].reindex(bars.index).ffill()
            symbols[column] = auxiliary.metadata["symbol"]
            hashes[column] = auxiliary.metadata["data_hash"]
        metadata = dict(primary.metadata)
        metadata.update(
            {
                "source": "yfinance_panel",
                "symbols": symbols,
                "component_data_hashes": hashes,
                "data_hash": fingerprint_bars(bars),
            }
        )
        return MarketData(bars=bars, metadata=metadata)
