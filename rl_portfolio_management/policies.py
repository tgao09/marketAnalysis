"""Deterministic sanity policies for portfolio replay correctness."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from common.backtesting import ClosePosition, Hold, PortfolioObservation, TargetPosition


class AlwaysCashPolicy:
    def act(self, observation: PortfolioObservation):
        actions = [ClosePosition(symbol) for symbol, position in observation.portfolio.positions.items() if abs(position.quantity) > 1e-12]
        return tuple(actions) if actions else (Hold(),)


@dataclass
class EqualWeightLongPolicy:
    rebalance_every: int = 5
    _steps: int = 0

    def act(self, observation: PortfolioObservation):
        self._steps += 1
        if (self._steps - 1) % self.rebalance_every:
            return (Hold(),)
        symbols = sorted(observation.bars)
        target = observation.portfolio.equity / len(symbols)
        return tuple(TargetPosition(symbol, target_notional=target) for symbol in symbols)


@dataclass
class MomentumPolicy:
    lookback: int = 20
    rebalance_every: int = 5
    long_fraction: float = 0.5
    _steps: int = 0
    _closes: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def act(self, observation: PortfolioObservation):
        self._steps += 1
        for symbol, bar in observation.bars.items():
            history = self._closes[symbol]
            history.append(bar.close)
            while len(history) > self.lookback + 1:
                history.popleft()
        if self._steps <= self.lookback or (self._steps - self.lookback - 1) % self.rebalance_every:
            return (Hold(),)
        scores = {
            symbol: history[-1] / history[0] - 1
            for symbol, history in self._closes.items() if len(history) == self.lookback + 1
        }
        ranked = sorted(scores, key=lambda symbol: (scores[symbol], symbol), reverse=True)
        count = max(1, int(round(len(ranked) * self.long_fraction)))
        selected = set(ranked[:count])
        target = observation.portfolio.equity / count
        return tuple(TargetPosition(symbol, target_notional=target if symbol in selected else 0.0) for symbol in ranked)


@dataclass
class RandomValidPolicy:
    seed: int = 42
    rebalance_every: int = 5
    gross_budget: float = 0.5
    _steps: int = 0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def act(self, observation: PortfolioObservation):
        self._steps += 1
        if (self._steps - 1) % self.rebalance_every:
            return (Hold(),)
        symbols = sorted(observation.bars)
        raw = self._rng.normal(size=len(symbols))
        gross = float(np.abs(raw).sum())
        weights = self.gross_budget * raw / gross if gross else raw
        return tuple(
            TargetPosition(symbol, target_notional=float(weight * observation.portfolio.equity))
            for symbol, weight in zip(symbols, weights)
        )
