"""Machine-readable metrics derived from causal portfolio replay results."""

from __future__ import annotations

from collections import defaultdict, deque
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from .portfolio import OrderSide, OrderStatus, ReplayResult


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    median_days = float(np.median(np.diff(index.asi8)) / 86_400_000_000_000)
    return 252.0 * 6.5 if median_days < 0.2 else 252.0


def _completed_trades(result: ReplayResult) -> list[dict[str, Any]]:
    """FIFO-match fills into completed long/short lots."""
    replay_index = pd.DatetimeIndex([timestamp for timestamp, _ in result.snapshots])
    ordinal = {timestamp: i for i, timestamp in enumerate(replay_index)}
    median_days = (float(np.median(np.diff(replay_index.asi8))) / 86_400_000_000_000
                   if len(replay_index) > 1 else 1.0)
    bars_per_trading_day = 6.5 if median_days < 0.2 else 1.0
    lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    trades: list[dict[str, Any]] = []
    fills = sorted(
        (o for o in result.orders if o.status is OrderStatus.FILLED),
        key=lambda o: (o.filled_at, o.order_id),
    )
    for order in fills:
        signed = order.filled_quantity * (1 if order.request.side is OrderSide.BUY else -1)
        remaining = signed
        queue = lots[order.request.symbol]
        while queue and remaining * queue[0]["quantity"] < 0:
            lot = queue[0]
            closed = min(abs(remaining), abs(lot["quantity"]))
            direction = 1 if lot["quantity"] > 0 else -1
            pnl = closed * (float(order.fill_price) - lot["price"]) * direction
            duration = order.filled_at - lot["opened_at"]
            holding_bars = ordinal[order.filled_at] - ordinal[lot["opened_at"]]
            trades.append({
                "symbol": order.request.symbol,
                "side": "long" if direction > 0 else "short",
                "quantity": closed,
                "pnl": pnl,
                "holding_hours": duration.total_seconds() / 3600,
                "holding_days": holding_bars / bars_per_trading_day,
            })
            lot["quantity"] -= direction * closed
            remaining += direction * closed
            if abs(lot["quantity"]) < 1e-12:
                queue.popleft()
        if abs(remaining) >= 1e-12:
            queue.append({"quantity": remaining, "price": float(order.fill_price), "opened_at": order.filled_at})
    return trades


def portfolio_metrics(result: ReplayResult) -> dict[str, Any]:
    """Return robust portfolio/trade metrics; unavailable values are ``None``."""
    if not result.snapshots:
        raise ValueError("replay result has no snapshots")
    index = pd.DatetimeIndex([timestamp for timestamp, _ in result.snapshots])
    equity = pd.Series([snapshot.equity for _, snapshot in result.snapshots], index=index, dtype=float)
    returns = equity.pct_change().dropna()
    periods = _periods_per_year(index)
    years = max((index[-1] - index[0]).total_seconds() / (365.25 * 86400), 1 / periods)
    cumulative = equity.iloc[-1] / equity.iloc[0] - 1
    annualized = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if equity.iloc[0] > 0 else None
    volatility = float(returns.std(ddof=1) * sqrt(periods)) if len(returns) > 1 else None
    mean = float(returns.mean()) if len(returns) else None
    std = float(returns.std(ddof=1)) if len(returns) > 1 else None
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else None
    sharpe = mean / std * sqrt(periods) if std and std > 0 else None
    sortino = mean / downside_std * sqrt(periods) if downside_std and downside_std > 0 else None
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(-drawdown.min())
    calmar = annualized / max_drawdown if annualized is not None and max_drawdown > 0 else None
    q05 = float(returns.quantile(0.05)) if len(returns) else None
    cvar05 = float(returns[returns <= q05].mean()) if q05 is not None else None

    snapshots = [snapshot for _, snapshot in result.snapshots]
    gross_ratios = [s.gross_exposure / s.equity if s.equity > 0 else np.nan for s in snapshots]
    net_ratios = [s.net_exposure / s.equity if s.equity > 0 else np.nan for s in snapshots]
    cash_ratios = [s.cash / s.equity if s.equity > 0 else np.nan for s in snapshots]
    filled = [o for o in result.orders if o.status is OrderStatus.FILLED]
    fill_notional = sum(o.filled_quantity * float(o.fill_price) for o in filled)
    trades = _completed_trades(result)
    pnls = np.asarray([t["pnl"] for t in trades], dtype=float)
    holding_days = np.asarray([t["holding_days"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) else (None if not len(wins) else float("inf"))

    return {
        "starting_equity": float(equity.iloc[0]),
        "ending_equity": float(equity.iloc[-1]),
        "cumulative_return": float(cumulative),
        "annualized_return": None if annualized is None else float(annualized),
        "sharpe": None if sharpe is None else float(sharpe),
        "sortino": None if sortino is None else float(sortino),
        "maximum_drawdown": max_drawdown,
        "calmar": None if calmar is None else float(calmar),
        "volatility": volatility,
        "value_at_risk_05": q05,
        "conditional_value_at_risk_05": cvar05,
        "turnover": float(fill_notional / equity.mean()),
        "average_gross_exposure": float(np.nanmean(gross_ratios)),
        "maximum_gross_exposure": float(np.nanmax(gross_ratios)),
        "average_net_exposure": float(np.nanmean(net_ratios)),
        "percentage_time_in_cash": float(np.mean(np.asarray(gross_ratios) < 1e-12)),
        "average_cash_ratio": float(np.nanmean(cash_ratios)),
        "completed_trades": len(trades),
        "win_rate": float(np.mean(pnls > 0)) if len(pnls) else None,
        "profit_factor": profit_factor,
        "average_trade_pnl": float(pnls.mean()) if len(pnls) else None,
        "median_trade_pnl": float(np.median(pnls)) if len(pnls) else None,
        "average_holding_days": float(holding_days.mean()) if len(holding_days) else None,
        "median_holding_days": float(np.median(holding_days)) if len(holding_days) else None,
        "holding_days_distribution": [float(x) for x in holding_days],
        "long_contribution": float(sum(t["pnl"] for t in trades if t["side"] == "long")),
        "short_contribution": float(sum(t["pnl"] for t in trades if t["side"] == "short")),
        "orders_submitted": len(result.orders),
        "orders_filled": len(filled),
        "orders_rejected": sum(o.status.value == "rejected" for o in result.orders),
    }
