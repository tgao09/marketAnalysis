"""Gymnasium environment for a shared cross-asset portfolio policy.

Actions observed after close t become fixed target quantities using that close,
then fill at open t+1. This matches PortfolioReplay target-notional semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from common.backtesting.portfolio import PortfolioReplay, Position


@dataclass(frozen=True)
class RewardConfig:
    drawdown: float = 0.05
    turnover: float = 0.001
    exposure_instability: float = 0.001
    holding_time: float = 0.001
    holding_target_min: int = 3
    holding_target_max: int = 5


class CrossAssetPortfolioEnv(gym.Env):
    """Continuous signed target weights over one aligned feature/bar panel."""

    metadata = {"render_modes": []}

    def __init__(self, bars: Mapping[str, pd.DataFrame], features: Mapping[str, pd.DataFrame],
                 *, starting_equity: float = 10_000.0, gross_budget: float = 0.9,
                 reward: RewardConfig | None = None, action_cadence: int = 1) -> None:
        super().__init__()
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if not 0 < gross_budget <= 1:
            raise ValueError("gross_budget must be in (0, 1]")
        if isinstance(action_cadence, bool) or not isinstance(action_cadence, int) or action_cadence <= 0:
            raise ValueError("action_cadence must be a positive integer")
        self.symbols = tuple(sorted(bars))
        if not self.symbols or set(features) != set(self.symbols):
            raise ValueError("bars and features must contain identical symbols")
        clean = PortfolioReplay._validate_bars(bars)
        index = clean[self.symbols[0]].index
        columns = tuple(features[self.symbols[0]].columns)
        if len(index) < 2 or not columns:
            raise ValueError("at least two bars and one feature are required")
        for symbol in self.symbols:
            frame = features[symbol]
            if not clean[symbol].index.equals(index):
                raise ValueError("bar panel must be fully aligned")
            if not frame.index.equals(index) or tuple(frame.columns) != columns:
                raise ValueError("feature panel index/schema mismatch")
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError("features must be finite; trim warm-up rows")
        self.bars = {s: clean[s].copy(deep=True) for s in self.symbols}
        self.features = {s: features[s].astype(float).copy(deep=True) for s in self.symbols}
        self.index, self.feature_names = index.copy(), columns
        self.starting_equity, self.gross_budget = float(starting_equity), float(gross_budget)
        self.action_cadence = action_cadence
        self.reward_config = reward or RewardConfig()
        size = len(self.symbols) * (len(columns) + 3) + 7
        self.action_space = spaces.Box(-1.0, 1.0, (len(self.symbols),), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self._reset_state()

    def _reset_state(self) -> None:
        self._i, self._cash, self._peak = 0, self.starting_equity, self.starting_equity
        self._positions = {s: Position() for s in self.symbols}
        self._recent_turnover = 0.0
        self._turnovers: list[float] = []
        self._previous_weights = np.zeros(len(self.symbols))
        self._completed_holding_times: list[int] = []
        self._invalid = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._observation(), self._info()

    def step(self, action):
        if self._i >= len(self.index) - 1:
            raise RuntimeError("step called after episode termination")
        old_equity = self._equity(self._i, "Close")
        if self._i % self.action_cadence == 0:
            weights = self._project_weights(action)
            targets = {s: weights[j] * old_equity / float(self.bars[s].iloc[self._i].Close)
                       for j, s in enumerate(self.symbols)}
        else:
            # Off-cadence actions are ignored. Holding quantities avoids orders,
            # fills, and turnover while marks and reward still advance.
            targets = {s: self._positions[s].quantity for s in self.symbols}
        old_drawdown = old_equity / self._peak - 1.0
        for s, p in tuple(self._positions.items()):
            if abs(p.quantity) > 1e-12:
                self._positions[s] = Position(p.quantity, p.average_cost, p.realized_pnl, p.age_bars + 1)
        self._i += 1
        fill_notional, completed = 0.0, []
        open_marks = {s: float(self.bars[s].iloc[self._i].Open) for s in self.symbols}
        for s in self.symbols:
            p, delta = self._positions[s], targets[s] - self._positions[s].quantity
            if abs(delta) < 1e-12:
                continue
            price = open_marks[s]
            delta = PortfolioReplay(starting_cash=self.starting_equity)._constrain_quantity(
                s, delta, price, self._cash, self._positions, open_marks
            )
            if abs(delta) < 1e-12:
                continue
            if p.quantity * delta < 0 and abs(delta) >= abs(p.quantity) - 1e-12:
                completed.append(p.age_bars)
            self._cash -= delta * price
            fill_notional += abs(delta * price)
            self._positions[s] = PortfolioReplay._apply_trade(p, delta, price)
        equity = self._equity(self._i, "Close")
        gross_notional = sum(abs(p.quantity * float(self.bars[s].iloc[self._i].Close))
                             for s, p in self._positions.items())
        self._invalid = equity <= 0 or gross_notional > equity + max(1e-8, abs(equity) * 1e-10)
        self._peak = max(self._peak, equity)
        drawdown = equity / self._peak - 1.0
        turnover = fill_notional / max(equity, 1e-12)
        self._turnovers.append(turnover)
        self._recent_turnover = float(sum(self._turnovers[-20:]))
        actual = self._weights(equity)
        instability = float(np.abs(actual - self._previous_weights).sum())
        self._previous_weights = actual
        self._completed_holding_times.extend(completed)
        cfg = self.reward_config
        hold_penalty = sum(max(cfg.holding_target_min-age, 0, age-cfg.holding_target_max) for age in completed)
        reward = (np.log(max(equity, 1e-12) / max(old_equity, 1e-12))
                  - cfg.drawdown * max(0.0, abs(drawdown)-abs(old_drawdown))
                  - cfg.turnover * turnover - cfg.exposure_instability * instability
                  - cfg.holding_time * hold_penalty)
        if self._invalid:
            reward -= 10.0
        return self._observation(), float(reward), self._invalid, self._i == len(self.index)-1, self._info()

    def _project_weights(self, action) -> np.ndarray:
        values = np.asarray(action, dtype=float)
        if values.shape != self.action_space.shape or not np.isfinite(values).all():
            raise ValueError(f"action must be finite with shape {self.action_space.shape}")
        values = np.clip(values, -1, 1)
        gross = float(np.abs(values).sum())
        return values * (self.gross_budget / gross) if gross > self.gross_budget else values

    def _equity(self, i: int, field: str) -> float:
        return self._cash + sum(p.quantity * float(self.bars[s].iloc[i][field]) for s, p in self._positions.items())

    def _weights(self, equity=None) -> np.ndarray:
        equity = self._equity(self._i, "Close") if equity is None else equity
        return np.asarray([self._positions[s].quantity * float(self.bars[s].iloc[self._i].Close)
                           / max(equity, 1e-12) for s in self.symbols])

    def _observation(self) -> np.ndarray:
        equity, state = self._equity(self._i, "Close"), []
        weights = self._weights(equity)
        for j, s in enumerate(self.symbols):
            state.extend(self.features[s].iloc[self._i].to_numpy(float))
            p, close = self._positions[s], float(self.bars[s].iloc[self._i].Close)
            distance = 0.0 if not p.quantity else np.sign(p.quantity) * (close / p.average_cost - 1)
            state.extend((weights[j], distance, p.age_bars / 20.0))
        state.extend((self._cash/max(equity, 1e-12), equity/self.starting_equity,
                      float(np.abs(weights).sum()), float(weights.sum()), equity/self._peak-1,
                      self._recent_turnover, 0.0))
        return np.asarray(state, dtype=np.float32)

    def _info(self) -> dict:
        equity, weights = self._equity(self._i, "Close"), self._weights()
        return {"timestamp": self.index[self._i], "equity": equity, "cash": self._cash,
                "weights": weights.copy(), "gross_exposure": float(np.abs(weights).sum()),
                "net_exposure": float(weights.sum()), "drawdown": equity/self._peak-1,
                "recent_turnover": self._recent_turnover,
                "invalid": self._invalid,
                "completed_holding_times": tuple(self._completed_holding_times)}
