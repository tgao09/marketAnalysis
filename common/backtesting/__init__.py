"""Universal yfinance-backed, leakage-safe walk-forward backtesting."""

from .config import BacktestConfig, LogReturnTarget, WalkForwardConfig
from .data import MarketData, YFinanceSource, canonical_column_name, fingerprint_bars, normalize_bars
from .engine import BacktestEngine, BacktestResult, FoldContext, FoldModel, FoldSummary

__all__ = [
    "BacktestConfig",
    "LogReturnTarget",
    "WalkForwardConfig",
    "MarketData",
    "YFinanceSource",
    "canonical_column_name",
    "fingerprint_bars",
    "normalize_bars",
    "BacktestEngine",
    "BacktestResult",
    "FoldContext",
    "FoldModel",
    "FoldSummary",
]
