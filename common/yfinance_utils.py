"""Boilerplate yfinance helpers for stocks and sector ETF proxies."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

DEFAULT_SECTOR_ETF_MAP: Dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

_SECTOR_ALIASES: Dict[str, str] = {
    "communication services": "Communication Services",
    "consumer discretionary": "Consumer Discretionary",
    "consumer cyclical": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "consumer defensive": "Consumer Staples",
    "energy": "Energy",
    "financials": "Financials",
    "financial services": "Financials",
    "health care": "Health Care",
    "healthcare": "Health Care",
    "industrials": "Industrials",
    "materials": "Materials",
    "basic materials": "Materials",
    "real estate": "Real Estate",
    "technology": "Technology",
    "information technology": "Technology",
    "utilities": "Utilities",
}

_DEFAULT_NORMALIZED_SECTORS: Dict[str, str] = {
    re.sub(r"[^a-z0-9]+", " ", sector.lower()).strip(): sector
    for sector in DEFAULT_SECTOR_ETF_MAP
}
_SECTOR_ALIASES_NORMALIZED: Dict[str, str] = {
    re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip(): sector
    for alias, sector in _SECTOR_ALIASES.items()
}


def _normalize_sector_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def canonicalize_sector_name(sector: Optional[str]) -> Optional[str]:
    if not isinstance(sector, str):
        return None
    normalized = _normalize_sector_key(sector)
    if not normalized:
        return None
    if normalized in _DEFAULT_NORMALIZED_SECTORS:
        return _DEFAULT_NORMALIZED_SECTORS[normalized]
    return _SECTOR_ALIASES_NORMALIZED.get(normalized)


def _validate_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Symbol must be a non-empty string.")
    return symbol.strip().upper()


def _resolve_sector_etf(sector: str, sector_map: Optional[Dict[str, str]]) -> str:
    if not isinstance(sector, str) or not sector.strip():
        raise ValueError("Sector must be a non-empty string.")
    mapping = sector_map if sector_map is not None else DEFAULT_SECTOR_ETF_MAP
    raw_key = sector.strip()
    sector_key = canonicalize_sector_name(raw_key)
    if sector_key in mapping:
        return mapping[sector_key]
    if raw_key in mapping:
        return mapping[raw_key]
    normalized_lookup = {
        _normalize_sector_key(candidate): candidate
        for candidate in mapping
    }
    normalized_raw = _normalize_sector_key(raw_key)
    if normalized_raw in normalized_lookup:
        return mapping[normalized_lookup[normalized_raw]]
    if sector_key is not None:
        normalized_sector_key = _normalize_sector_key(sector_key)
        if normalized_sector_key in normalized_lookup:
            return mapping[normalized_lookup[normalized_sector_key]]
    if sector_key is None:
        available = ", ".join(sorted(mapping.keys()))
        raise KeyError(f"Unknown sector: {raw_key}. Available: {available}")
    available = ", ".join(sorted(mapping.keys()))
    raise KeyError(f"Unknown sector: {raw_key}. Canonicalized: {sector_key}. Available: {available}")


def get_ticker(symbol: str) -> yf.Ticker:
    symbol = _validate_symbol(symbol)
    return yf.Ticker(symbol)


def get_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    ticker = get_ticker(symbol)
    use_period = period
    if start is not None or end is not None:
        use_period = None
    history = ticker.history(
        period=use_period,
        interval=interval,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
    )
    if history is None or history.empty:
        raise ValueError(f"No history returned for symbol: {symbol}")
    return history


def get_info(symbol: str) -> Dict[str, Any]:
    ticker = get_ticker(symbol)
    info = ticker.info
    if not info:
        raise ValueError(f"No info returned for symbol: {symbol}")
    return info


def get_sector_etf_metrics(
    sector: str,
    sector_map: Optional[Dict[str, str]] = None,
    period: str = "1y",
    interval: str = "1d",
) -> Dict[str, Any]:
    etf_symbol = _resolve_sector_etf(sector, sector_map)
    history = get_history(etf_symbol, period=period, interval=interval)

    close = history["Close"].dropna()
    if close.empty:
        raise ValueError(f"No close prices for sector ETF: {etf_symbol}")

    daily_returns = close.pct_change().dropna()
    total_return = (close.iloc[-1] / close.iloc[0]) - 1.0
    volatility = float(daily_returns.std()) if not daily_returns.empty else None
    avg_volume = float(history["Volume"].dropna().mean()) if "Volume" in history else None

    return {
        "sector": sector,
        "etf_symbol": etf_symbol,
        "last_close": float(close.iloc[-1]),
        "total_return": float(total_return),
        "volatility": volatility,
        "avg_volume": avg_volume,
        "period": period,
        "interval": interval,
    }


def get_sector_holdings(
    sector: str,
    sector_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    etf_symbol = _resolve_sector_etf(sector, sector_map)
    ticker = get_ticker(etf_symbol)

    holdings = None
    if hasattr(ticker, "holdings"):
        holdings = ticker.holdings
    elif hasattr(ticker, "get_holdings"):
        holdings = ticker.get_holdings()
    elif hasattr(ticker, "fund_holdings"):
        holdings = ticker.fund_holdings

    if holdings is None:
        raise ValueError(
            f"Holdings not available for sector ETF: {etf_symbol}. "
            "Try another ETF or provide your own sector_map."
        )

    if isinstance(holdings, pd.DataFrame):
        if holdings.empty:
            raise ValueError(f"Empty holdings for sector ETF: {etf_symbol}")
        return holdings

    if isinstance(holdings, dict):
        frame = pd.DataFrame(holdings)
        if frame.empty:
            raise ValueError(f"Empty holdings for sector ETF: {etf_symbol}")
        return frame

    raise ValueError(f"Unexpected holdings format for sector ETF: {etf_symbol}")


@dataclass(frozen=True)
class TickerClient:
    symbol: str

    def history(self, **kwargs: Any) -> pd.DataFrame:
        return get_history(self.symbol, **kwargs)

    def info(self) -> Dict[str, Any]:
        return get_info(self.symbol)


@dataclass(frozen=True)
class SectorClient:
    sector: str
    sector_map: Optional[Dict[str, str]] = None

    def metrics(self, **kwargs: Any) -> Dict[str, Any]:
        return get_sector_etf_metrics(self.sector, self.sector_map, **kwargs)

    def holdings(self) -> pd.DataFrame:
        return get_sector_holdings(self.sector, self.sector_map)
