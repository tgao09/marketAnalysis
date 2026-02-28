"""Boilerplate yfinance helpers for stocks and sector ETF proxies."""

from __future__ import annotations

import re
from typing import Any, Dict

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


def canonicalize_sector_name(sector: str) -> str | None:
    normalized = _normalize_sector_key(sector)
    if normalized in _DEFAULT_NORMALIZED_SECTORS:
        return _DEFAULT_NORMALIZED_SECTORS[normalized]
    return _SECTOR_ALIASES_NORMALIZED.get(normalized)


def _validate_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _resolve_sector_etf(sector: str, sector_map: Dict[str, str] | None) -> str:
    mapping = sector_map if sector_map else DEFAULT_SECTOR_ETF_MAP
    sector_key = canonicalize_sector_name(sector)
    if sector_key in mapping:
        return mapping[sector_key]
    raise KeyError("Sector not found")


def get_ticker(symbol: str) -> yf.Ticker:
    symbol = _validate_symbol(symbol)
    return yf.Ticker(symbol)


def get_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    ticker = get_ticker(symbol)
    use_period = None if start or end else period
    return ticker.history(
        period=use_period,
        interval=interval,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
    )


def get_info(symbol: str) -> Dict[str, Any]:
    ticker = get_ticker(symbol)
    return ticker.info


def get_sector_etf_metrics(
    sector: str,
    sector_map: Dict[str, str] | None = None,
    period: str = "1y",
    interval: str = "1d",
) -> Dict[str, Any]:
    etf_symbol = _resolve_sector_etf(sector, sector_map)
    history = get_history(etf_symbol, period=period, interval=interval)

    close = history["Close"].dropna()
    if close.empty:
        raise ValueError(f"No close prices for sector ETF: {etf_symbol}")

    daily_returns = close.pct_change().dropna()
    total_return: float = (close.iloc[-1] / close.iloc[0]) - 1.0
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
    sector_map: Dict[str, str] | None = None,
) -> pd.DataFrame:
    etf_symbol = _resolve_sector_etf(sector, sector_map)
    ticker = get_ticker(etf_symbol)

    holdings = getattr(ticker, "holdings", None)
    if holdings is None and hasattr(ticker, "get_holdings"):
        holdings = ticker.get_holdings()
    if holdings is None and hasattr(ticker, "fund_holdings"):
        holdings = ticker.fund_holdings
    return pd.DataFrame(holdings)
