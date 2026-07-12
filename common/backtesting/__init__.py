"""Universal yfinance-backed, leakage-safe walk-forward backtesting."""

from .config import BacktestConfig, CalendarWalkForwardConfig, LogReturnTarget, WalkForwardConfig
from .data import MarketData, YFinancePanelSource, YFinanceSource, canonical_column_name, fingerprint_bars, normalize_bars
from .engine import BacktestEngine, BacktestResult, FoldContext, FoldModel, FoldSummary
from .portfolio import (
    Bar, CancelOrder, ClosePosition, Hold, Order, OrderRequest, OrderSide,
    OrderStatus, OrderType, PortfolioObservation, PortfolioPolicy,
    PortfolioReplay, PortfolioSnapshot, Position, ReplaceOrder, ReplayResult,
    ScalePosition, TargetPosition,
)
from .portfolio_metrics import portfolio_metrics
from .portfolio_engine import (
    PortfolioFoldContext, PortfolioFoldSummary, PortfolioPolicyFactory,
    PortfolioSelectionFold, PortfolioWalkForward, PortfolioWalkForwardConfig, PortfolioWalkForwardResult,
)

__all__ = [
    "BacktestConfig",
    "CalendarWalkForwardConfig",
    "LogReturnTarget",
    "WalkForwardConfig",
    "MarketData",
    "YFinanceSource",
    "YFinancePanelSource",
    "canonical_column_name",
    "fingerprint_bars",
    "normalize_bars",
    "BacktestEngine",
    "BacktestResult",
    "FoldContext",
    "FoldModel",
    "FoldSummary",
    "Bar", "CancelOrder", "ClosePosition", "Hold", "Order", "OrderRequest",
    "OrderSide", "OrderStatus", "OrderType", "PortfolioObservation",
    "PortfolioPolicy", "PortfolioReplay", "PortfolioSnapshot", "Position",
    "ReplaceOrder", "ReplayResult", "ScalePosition", "TargetPosition",
    "portfolio_metrics",
    "PortfolioFoldContext", "PortfolioFoldSummary", "PortfolioPolicyFactory",
    "PortfolioSelectionFold", "PortfolioWalkForward", "PortfolioWalkForwardConfig", "PortfolioWalkForwardResult",
]
