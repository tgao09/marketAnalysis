"""Leakage-safe walk-forward orchestration for stateful portfolio policies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

import pandas as pd

from .data import fingerprint_bars
from .portfolio import PortfolioPolicy, PortfolioReplay, ReplayResult
from .portfolio_metrics import portfolio_metrics


Panel = Mapping[str, pd.DataFrame]


@dataclass(frozen=True)
class PortfolioWalkForwardConfig:
    """Rolling calendar folds; gaps are measured in aligned panel bars."""

    train_years: int = 2
    validation_months: int = 6
    test_months: int = 6
    step_months: int = 6
    purge_bars: int = 0
    embargo_bars: int = 0
    starting_equity: float = 10_000.0
    max_folds: int | None = None

    def __post_init__(self) -> None:
        for name in ("train_years", "validation_months", "test_months", "step_months"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("purge_bars", "embargo_bars"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if self.max_folds is not None and (not isinstance(self.max_folds, int) or self.max_folds <= 0):
            raise ValueError("max_folds must be a positive integer")


@dataclass(frozen=True)
class PortfolioFoldContext:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    planned_test_start: pd.Timestamp
    planned_test_end: pd.Timestamp
    train_hashes: Mapping[str, str]
    validation_hashes: Mapping[str, str]


@dataclass(frozen=True)
class PortfolioFoldSummary:
    context: PortfolioFoldContext
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_hashes: Mapping[str, str]
    metrics: Mapping[str, Any]
    replay: ReplayResult
    test_accessed: bool


@dataclass(frozen=True)
class PortfolioWalkForwardResult:
    folds: tuple[PortfolioFoldSummary, ...]
    panel_hash: str


@dataclass(frozen=True)
class PortfolioSelectionFold:
    """Train/validation-only view. Deliberately contains no test metadata."""

    fold: int
    train: Panel
    validation: Panel
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_hashes: Mapping[str, str]
    validation_hashes: Mapping[str, str]


@dataclass(frozen=True)
class _FoldPlan:
    fold: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


PortfolioPolicyFactory = Callable[[Panel, Panel, PortfolioFoldContext], PortfolioPolicy]


class PortfolioWalkForward:
    """Fit/select on train+validation, then replay frozen policy on untouched test."""

    def __init__(self, config: PortfolioWalkForwardConfig | None = None) -> None:
        self.config = config or PortfolioWalkForwardConfig()

    def selection_folds(self, bars: Panel) -> Iterator[PortfolioSelectionFold]:
        """Yield copied train/validation folds without materializing test slices/hashes."""
        panel = self._aligned_panel(bars)
        index = next(iter(panel.values())).index
        found = False
        for plan in self._fold_plans(index):
            found = True
            train = self._slice(panel, plan.train_start, plan.train_end)
            validation = self._slice(panel, plan.validation_start, plan.validation_end)
            yield PortfolioSelectionFold(
                fold=plan.fold,
                train=self._copy(train), validation=self._copy(validation),
                train_start=index[plan.train_start], train_end=index[plan.train_end - 1],
                validation_start=index[plan.validation_start],
                validation_end=index[plan.validation_end - 1],
                train_hashes=self._hashes(train), validation_hashes=self._hashes(validation),
            )
        if not found:
            raise ValueError("no complete 2y/6m/6m portfolio folds")

    def run(self, bars: Panel, policy_factory: PortfolioPolicyFactory) -> PortfolioWalkForwardResult:
        panel = self._aligned_panel(bars)
        index = next(iter(panel.values())).index
        summaries: list[PortfolioFoldSummary] = []
        for plan in self._fold_plans(index):
            train = self._slice(panel, plan.train_start, plan.train_end)
            validation = self._slice(panel, plan.validation_start, plan.validation_end)
            context = PortfolioFoldContext(
                fold=plan.fold,
                train_start=next(iter(train.values())).index[0],
                train_end=next(iter(train.values())).index[-1],
                validation_start=next(iter(validation.values())).index[0],
                validation_end=next(iter(validation.values())).index[-1],
                planned_test_start=index[plan.test_start],
                planned_test_end=index[plan.test_end - 1],
                train_hashes=self._hashes(train),
                validation_hashes=self._hashes(validation),
            )
            policy = policy_factory(self._copy(train), self._copy(validation), context)

            # Once selection is frozen, provide causal history through the bar
            # immediately before test. Factory never sees this warmup slice.
            prepare = getattr(policy, "prepare_replay", None)
            if callable(prepare):
                prepare(self._copy(self._slice(panel, 0, plan.test_start)))

            # Test materialization happens only after selection returns.
            test = self._slice(panel, plan.test_start, plan.test_end)
            replay = PortfolioReplay(starting_cash=self.config.starting_equity).run(test, policy)
            summaries.append(PortfolioFoldSummary(
                context=context,
                test_start=index[plan.test_start],
                test_end=index[plan.test_end - 1],
                test_hashes=self._hashes(test),
                metrics=portfolio_metrics(replay),
                replay=replay,
                test_accessed=True,
            ))
        if not summaries:
            raise ValueError("no complete 2y/6m/6m portfolio folds")
        return PortfolioWalkForwardResult(tuple(summaries), self._panel_hash(panel))

    def _fold_plans(self, index: pd.DatetimeIndex) -> Iterator[_FoldPlan]:
        anchor, fold = index[0], 0
        while True:
            train_stop = anchor + pd.DateOffset(years=self.config.train_years)
            validation_stop = train_stop + pd.DateOffset(months=self.config.validation_months)
            test_stop = validation_stop + pd.DateOffset(months=self.config.test_months)
            if test_stop > index[-1]:
                return
            train_start = 0 if fold == 0 else int(index.searchsorted(anchor))
            train_end = int(index.searchsorted(train_stop, side="left")) - self.config.purge_bars
            validation_start = int(index.searchsorted(train_stop, side="left")) + self.config.embargo_bars
            validation_end = int(index.searchsorted(validation_stop, side="left")) - self.config.purge_bars
            test_start = int(index.searchsorted(validation_stop, side="left")) + self.config.embargo_bars
            test_end = int(index.searchsorted(test_stop, side="left"))
            if min(train_end - train_start, validation_end - validation_start,
                   test_end - test_start) <= 0:
                raise ValueError("purge/embargo removes an entire fold slice")
            yield _FoldPlan(fold, train_start, train_end, validation_start,
                            validation_end, test_start, test_end)
            fold += 1
            if self.config.max_folds is not None and fold >= self.config.max_folds:
                return
            anchor += pd.DateOffset(months=self.config.step_months)
            if index.searchsorted(anchor) >= len(index):
                return

    @staticmethod
    def _aligned_panel(bars: Panel) -> dict[str, pd.DataFrame]:
        clean = PortfolioReplay._validate_bars(bars)
        common = next(iter(clean.values())).index
        for frame in tuple(clean.values())[1:]:
            common = common.intersection(frame.index, sort=False)
        common = common.sort_values()
        if common.empty:
            raise ValueError("multi-asset panel has no common timestamps")
        return {symbol: frame.loc[common].copy(deep=True) for symbol, frame in sorted(clean.items())}

    @staticmethod
    def _slice(panel: Panel, start: int, stop: int) -> dict[str, pd.DataFrame]:
        return {symbol: frame.iloc[start:stop].copy(deep=True) for symbol, frame in panel.items()}

    @staticmethod
    def _copy(panel: Panel) -> dict[str, pd.DataFrame]:
        return {symbol: frame.copy(deep=True) for symbol, frame in panel.items()}

    @staticmethod
    def _hashes(panel: Panel) -> dict[str, str]:
        return {symbol: fingerprint_bars(frame) for symbol, frame in panel.items()}

    @classmethod
    def _panel_hash(cls, panel: Panel) -> str:
        digest = hashlib.sha256()
        for symbol, value in cls._hashes(panel).items():
            digest.update(symbol.encode())
            digest.update(value.encode())
        return digest.hexdigest()
