"""Causal multi-asset portfolio replay and accounting primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import pandas as pd


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type is OrderType.LIMIT and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("positive limit_price required for limit order")


@dataclass(frozen=True)
class CancelOrder:
    order_id: int


@dataclass(frozen=True)
class Hold:
    """Explicit no-op action."""


@dataclass(frozen=True)
class ReplaceOrder:
    order_id: int
    replacement: OrderRequest


@dataclass(frozen=True)
class TargetPosition:
    symbol: str
    target_quantity: float | None = None
    target_notional: float | None = None
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if (self.target_quantity is None) == (self.target_notional is None):
            raise ValueError("set exactly one target_quantity or target_notional")


@dataclass(frozen=True)
class ScalePosition:
    symbol: str
    delta_quantity: float | None = None
    delta_notional: float | None = None
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if (self.delta_quantity is None) == (self.delta_notional is None):
            raise ValueError("set exactly one delta_quantity or delta_notional")


@dataclass(frozen=True)
class ClosePosition:
    symbol: str
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None


@dataclass(frozen=True)
class Order:
    order_id: int
    request: OrderRequest
    submitted_at: pd.Timestamp
    eligible_after: pd.Timestamp
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    fill_price: float | None = None
    filled_at: pd.Timestamp | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Position:
    quantity: float = 0.0
    average_cost: float = 0.0
    realized_pnl: float = 0.0
    age_bars: int = 0


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    gross_exposure: float
    net_exposure: float
    recent_turnover: float
    drawdown: float
    positions: Mapping[str, Position]
    pending_orders: tuple[Order, ...]


@dataclass(frozen=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class PortfolioObservation:
    timestamp: pd.Timestamp
    bars: Mapping[str, Bar]
    portfolio: PortfolioSnapshot


PortfolioAction = OrderRequest | CancelOrder | Hold | ReplaceOrder | TargetPosition | ScalePosition | ClosePosition


class PortfolioPolicy(Protocol):
    def act(self, observation: PortfolioObservation) -> Sequence[PortfolioAction]: ...


@dataclass(frozen=True)
class ReplayResult:
    snapshots: tuple[tuple[pd.Timestamp, PortfolioSnapshot], ...]
    orders: tuple[Order, ...]


class PortfolioReplay:
    """Event-at-a-time replay. Decisions on bar t can execute only after t."""

    REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

    def __init__(self, starting_cash: float = 10_000.0, leverage_policy: str = "clip") -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if leverage_policy not in {"clip", "reject"}:
            raise ValueError("leverage_policy must be clip or reject")
        self.starting_cash = float(starting_cash)
        self.leverage_policy = leverage_policy

    def run(self, bars: Mapping[str, pd.DataFrame], policy: PortfolioPolicy) -> ReplayResult:
        clean = self._validate_bars(bars)
        timestamps = sorted(set().union(*(frame.index for frame in clean.values())))
        cash = self.starting_cash
        positions: dict[str, Position] = {}
        orders: list[Order] = []
        marks: dict[str, float] = {}
        history: list[tuple[pd.Timestamp, PortfolioSnapshot]] = []
        peak_equity = self.starting_cash
        fill_notionals: list[float] = []

        for timestamp in timestamps:
            current = {symbol: self._bar(frame.loc[timestamp]) for symbol, frame in clean.items() if timestamp in frame.index}
            if not current:
                continue
            positions = {
                symbol: replace(position, age_bars=position.age_bars + 1)
                if abs(position.quantity) >= 1e-12 else position
                for symbol, position in positions.items()
            }
            open_marks = {**marks, **{s: b.open for s, b in current.items()}}
            cash, positions, orders, fill_notional = self._fill_eligible(
                timestamp, current, open_marks, cash, positions, orders
            )
            fill_notionals.append(fill_notional)
            marks.update({symbol: bar.close for symbol, bar in current.items()})
            equity = cash + sum(
                position.quantity * marks.get(symbol, position.average_cost)
                for symbol, position in positions.items()
            )
            peak_equity = max(peak_equity, equity)
            recent_turnover = sum(fill_notionals[-20:]) / equity if equity > 0 else 0.0
            snapshot = self._snapshot(
                cash, positions, marks, orders, peak_equity, recent_turnover
            )
            immutable_bars = MappingProxyType(dict(current))
            observation = PortfolioObservation(timestamp, immutable_bars, snapshot)
            actions = tuple(policy.act(observation))
            for action in actions:
                if isinstance(action, Hold):
                    continue
                if isinstance(action, CancelOrder):
                    orders = [replace(o, status=OrderStatus.CANCELLED, reason="cancelled") if o.order_id == action.order_id and o.status is OrderStatus.PENDING else o for o in orders]
                    continue
                if isinstance(action, ReplaceOrder):
                    matched = any(o.order_id == action.order_id and o.status is OrderStatus.PENDING for o in orders)
                    if not matched:
                        raise ValueError(f"pending order not found: {action.order_id}")
                    orders = [replace(o, status=OrderStatus.CANCELLED, reason="replaced") if o.order_id == action.order_id and o.status is OrderStatus.PENDING else o for o in orders]
                    request = action.replacement
                else:
                    request = self._resolve_action(action, positions, marks)
                if request is None:
                    continue
                if not isinstance(request, OrderRequest):
                    raise TypeError(f"unsupported action: {type(action)!r}")
                if request.symbol not in clean:
                    raise ValueError(f"unknown symbol: {request.symbol}")
                orders.append(Order(len(orders) + 1, request, timestamp, timestamp))
            snapshot = self._snapshot(
                cash, positions, marks, orders, peak_equity, recent_turnover
            )
            self._assert_invariants(snapshot)
            history.append((timestamp, snapshot))
        return ReplayResult(tuple(history), tuple(orders))

    @staticmethod
    def _resolve_action(
        action: PortfolioAction,
        positions: Mapping[str, Position],
        marks: Mapping[str, float],
    ) -> OrderRequest | None:
        if isinstance(action, OrderRequest):
            return action
        if not isinstance(action, (TargetPosition, ScalePosition, ClosePosition)):
            raise TypeError(f"unsupported action: {type(action)!r}")
        if action.symbol not in marks:
            raise ValueError(f"no causal mark for symbol: {action.symbol}")
        current = positions.get(action.symbol, Position()).quantity
        if isinstance(action, TargetPosition):
            target = action.target_quantity
            if target is None:
                target = float(action.target_notional) / marks[action.symbol]
            delta = target - current
        elif isinstance(action, ScalePosition):
            delta = action.delta_quantity
            if delta is None:
                delta = float(action.delta_notional) / marks[action.symbol]
        else:
            delta = -current
        if abs(delta) < 1e-12:
            return None
        return OrderRequest(
            symbol=action.symbol,
            side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
            quantity=abs(delta),
            order_type=action.order_type,
            limit_price=action.limit_price,
        )

    def _fill_eligible(
        self,
        timestamp: pd.Timestamp,
        bars: Mapping[str, Bar],
        marks: Mapping[str, float],
        cash: float,
        positions: dict[str, Position],
        orders: list[Order],
    ) -> tuple[float, dict[str, Position], list[Order], float]:
        updated = list(orders)
        fill_notional = 0.0
        for index, order in enumerate(updated):
            if order.status is not OrderStatus.PENDING or timestamp <= order.eligible_after:
                continue
            bar = bars.get(order.request.symbol)
            if bar is None:
                continue
            fill_price = self._fill_price(order.request, bar)
            if fill_price is None:
                continue
            signed_requested = order.request.quantity * (1.0 if order.request.side is OrderSide.BUY else -1.0)
            signed_fill = self._constrain_quantity(
                order.request.symbol, signed_requested, fill_price, cash, positions, marks
            )
            if abs(signed_fill) < 1e-12:
                updated[index] = replace(order, status=OrderStatus.REJECTED, reason="gross exposure limit")
                continue
            cash -= signed_fill * fill_price
            fill_notional += abs(signed_fill * fill_price)
            positions = dict(positions)
            positions[order.request.symbol] = self._apply_trade(
                positions.get(order.request.symbol, Position()), signed_fill, fill_price
            )
            reason = "clipped to gross exposure limit" if abs(signed_fill) + 1e-12 < abs(signed_requested) else None
            updated[index] = replace(
                order,
                status=OrderStatus.FILLED,
                filled_quantity=abs(signed_fill),
                fill_price=fill_price,
                filled_at=timestamp,
                reason=reason,
            )
        return cash, positions, updated, fill_notional

    def _constrain_quantity(
        self,
        symbol: str,
        requested: float,
        price: float,
        cash: float,
        positions: Mapping[str, Position],
        marks: Mapping[str, float],
    ) -> float:
        snapshot = self._snapshot(cash, positions, marks, ())
        current = positions.get(symbol, Position()).quantity
        other_gross = sum(
            abs(position.quantity * marks.get(name, position.average_cost))
            for name, position in positions.items()
            if name != symbol
        )
        max_abs_position = max(0.0, snapshot.equity - other_gross) / price
        target = current + requested
        clipped_target = min(max(target, -max_abs_position), max_abs_position)
        constrained = clipped_target - current
        if abs(constrained - requested) > 1e-10 and self.leverage_policy == "reject":
            return 0.0
        return constrained

    @staticmethod
    def _apply_trade(position: Position, quantity: float, price: float) -> Position:
        old = position.quantity
        new = old + quantity
        realized = position.realized_pnl
        if old == 0 or old * quantity > 0:
            average = (abs(old) * position.average_cost + abs(quantity) * price) / abs(new)
            age_bars = 0 if old == 0 else position.age_bars
        else:
            closed = min(abs(old), abs(quantity))
            realized += closed * (price - position.average_cost) * (1.0 if old > 0 else -1.0)
            if abs(new) < 1e-12:
                new, average = 0.0, 0.0
                age_bars = 0
            elif old * new < 0:
                average = price
                age_bars = 0
            else:
                average = position.average_cost
                age_bars = position.age_bars
        return Position(new, average, realized, age_bars)

    @staticmethod
    def _fill_price(request: OrderRequest, bar: Bar) -> float | None:
        if request.order_type is OrderType.MARKET:
            return bar.open
        limit = float(request.limit_price)
        if request.side is OrderSide.BUY:
            if bar.open <= limit:
                return bar.open
            return limit if bar.low <= limit else None
        if bar.open >= limit:
            return bar.open
        return limit if bar.high >= limit else None

    @staticmethod
    def _snapshot(
        cash: float,
        positions: Mapping[str, Position],
        marks: Mapping[str, float],
        orders: Sequence[Order],
        peak_equity: float | None = None,
        recent_turnover: float = 0.0,
    ) -> PortfolioSnapshot:
        market_values = {symbol: position.quantity * marks.get(symbol, position.average_cost) for symbol, position in positions.items()}
        unrealized = sum(
            position.quantity * (marks.get(symbol, position.average_cost) - position.average_cost)
            for symbol, position in positions.items()
        )
        realized = sum(position.realized_pnl for position in positions.values())
        equity = cash + sum(market_values.values())
        peak = equity if peak_equity is None else peak_equity
        drawdown = equity / peak - 1.0 if peak > 0 else 0.0
        immutable_positions = MappingProxyType(dict(positions))
        return PortfolioSnapshot(
            cash=cash,
            equity=equity,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            gross_exposure=sum(abs(value) for value in market_values.values()),
            net_exposure=sum(market_values.values()),
            recent_turnover=recent_turnover,
            drawdown=drawdown,
            positions=immutable_positions,
            pending_orders=tuple(order for order in orders if order.status is OrderStatus.PENDING),
        )

    @classmethod
    def _validate_bars(cls, bars: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        if not bars:
            raise ValueError("bars cannot be empty")
        clean: dict[str, pd.DataFrame] = {}
        for symbol, frame in bars.items():
            missing = set(cls.REQUIRED_COLUMNS) - set(frame.columns)
            if missing:
                raise ValueError(f"{symbol} missing columns: {sorted(missing)}")
            if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
                raise ValueError(f"{symbol} index must be unique and increasing")
            selected = frame.loc[:, cls.REQUIRED_COLUMNS].astype(float).copy(deep=True)
            if selected.isna().any().any() or (selected.loc[:, ["Open", "High", "Low", "Close"]] <= 0).any().any():
                raise ValueError(f"{symbol} contains invalid bars")
            scale = selected[["Open", "High", "Low", "Close"]].abs().max(axis=1).clip(lower=1.0)
            tolerance = scale * 1e-12
            if (selected["Low"] > selected[["Open", "Close", "High"]].min(axis=1) + tolerance).any() or (selected["High"] < selected[["Open", "Close", "Low"]].max(axis=1) - tolerance).any():
                raise ValueError(f"{symbol} contains inconsistent OHLC")
            clean[symbol] = selected
        return clean

    @staticmethod
    def _bar(row: pd.Series) -> Bar:
        return Bar(float(row.Open), float(row.High), float(row.Low), float(row.Close), float(row.Volume))

    @staticmethod
    def _assert_invariants(snapshot: PortfolioSnapshot) -> None:
        tolerance = max(1e-8, abs(snapshot.equity) * 1e-10)
        assert snapshot.equity >= -tolerance, "negative equity"
        assert snapshot.gross_exposure <= snapshot.equity + tolerance, "gross exposure exceeds equity"
        assert abs(snapshot.net_exposure) <= snapshot.gross_exposure + tolerance
        assert abs(snapshot.equity - snapshot.cash - snapshot.net_exposure) <= tolerance
