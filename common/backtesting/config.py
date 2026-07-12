"""Configuration for leakage-safe walk-forward forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
import re


def _parse_window(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"\s*\d+\s*[dwmyDWMY]\s*", value):
        raise ValueError(f"{name} must be a positive window like '21d', '1m', or '2y'.")
    if int(re.search(r"\d+", value).group()) <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_int(value: int, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 0 or (value == 0 and not allow_zero):
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}.")


@dataclass(frozen=True)
class LogReturnTarget:
    """Forward log return over observed bars, never calendar days."""

    horizon_bars: int = 1
    price_column: str = "close"

    def __post_init__(self) -> None:
        _validate_int(self.horizon_bars, "horizon_bars")
        if not isinstance(self.price_column, str) or not self.price_column.strip():
            raise ValueError("price_column must be a non-empty string.")


@dataclass(frozen=True)
class WalkForwardConfig:
    """Row-based split settings.

    Target-horizon purging is always automatic. ``extra_purge_bars`` adds a
    further pre-test gap. ``embargo_rows`` excludes rows immediately after
    earlier OOS windows from later training folds.
    """

    min_train_rows: int = 252
    test_rows: int = 21
    step_rows: int | None = None
    max_train_rows: int | None = None
    extra_purge_bars: int = 0
    embargo_rows: int = 0
    reuse_prior_oos: bool = True

    def __post_init__(self) -> None:
        _validate_int(self.min_train_rows, "min_train_rows")
        _validate_int(self.test_rows, "test_rows")
        _validate_int(self.extra_purge_bars, "extra_purge_bars", allow_zero=True)
        _validate_int(self.embargo_rows, "embargo_rows", allow_zero=True)
        if self.step_rows is not None:
            _validate_int(self.step_rows, "step_rows")
            if self.step_rows < self.test_rows:
                raise ValueError("step_rows must be at least test_rows; overlapping OOS windows are unsupported.")
        if self.max_train_rows is not None:
            _validate_int(self.max_train_rows, "max_train_rows")
            if self.max_train_rows < self.min_train_rows:
                raise ValueError("max_train_rows cannot be smaller than min_train_rows.")
        if not isinstance(self.reuse_prior_oos, bool):
            raise TypeError("reuse_prior_oos must be a bool.")

    @property
    def effective_step_rows(self) -> int:
        return self.test_rows if self.step_rows is None else int(self.step_rows)


@dataclass(frozen=True)
class CalendarWalkForwardConfig:
    """Calendar-window walk-forward settings for parity with legacy research."""

    train_window: str = "2y"
    test_window: str = "1m"
    test_rows: int | None = None
    step_window: str | None = None
    min_train_rows: int = 252
    pre_test_gap_rows: int = 0
    extra_purge_bars: int = 0
    embargo_rows: int = 0
    reuse_prior_oos: bool = True

    def __post_init__(self) -> None:
        _parse_window(self.train_window, "train_window")
        _parse_window(self.test_window, "test_window")
        if self.test_rows is not None:
            _validate_int(self.test_rows, "test_rows")
        if self.step_window is not None:
            _parse_window(self.step_window, "step_window")
        _validate_int(self.min_train_rows, "min_train_rows")
        _validate_int(self.pre_test_gap_rows, "pre_test_gap_rows", allow_zero=True)
        _validate_int(self.extra_purge_bars, "extra_purge_bars", allow_zero=True)
        _validate_int(self.embargo_rows, "embargo_rows", allow_zero=True)
        if not isinstance(self.reuse_prior_oos, bool):
            raise TypeError("reuse_prior_oos must be a bool.")

    @property
    def effective_step_window(self) -> str:
        return self.test_window if self.step_window is None else self.step_window


@dataclass(frozen=True)
class BacktestConfig:
    """Universal backtester settings for one single-asset forecast target."""

    target: LogReturnTarget = field(default_factory=LogReturnTarget)
    walk_forward: WalkForwardConfig | CalendarWalkForwardConfig = field(default_factory=WalkForwardConfig)
    target_column: str = "target_log_return"
    prediction_column: str = "prediction"

    def __post_init__(self) -> None:
        for name in ("target_column", "prediction_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if self.target_column == self.prediction_column:
            raise ValueError("target_column and prediction_column must differ.")
