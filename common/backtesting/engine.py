"""Leakage-safe universal walk-forward backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Protocol

import numpy as np
import pandas as pd

from ..walk_forward import parse_window
from .config import BacktestConfig, CalendarWalkForwardConfig
from .data import MarketData, YFinanceSource, canonical_column_name, fingerprint_bars, normalize_bars


@dataclass(frozen=True)
class FoldContext:
    """Immutable fold metadata plus raw pre-test history for causal warmup."""

    fold: int
    target_column: str
    prediction_column: str
    horizon_bars: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    warmup: pd.DataFrame


class FoldModel(Protocol):
    """Model boundary. Test frames never expose targets or realized outcomes."""

    def fit(self, train: pd.DataFrame, context: FoldContext) -> object:
        """Fit from raw training bars plus the configured target column."""

    def predict(self, test: pd.DataFrame, context: FoldContext) -> pd.DataFrame | pd.Series:
        """Return test frame with prediction column, or indexed prediction series."""


ModelFactory = Callable[[FoldContext], FoldModel]


@dataclass(frozen=True)
class FoldSummary:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    test_rows: int
    purged_rows: int
    embargoed_rows: int
    prior_oos_excluded_rows: int


@dataclass(frozen=True)
class BacktestResult:
    config: BacktestConfig
    predictions: pd.DataFrame
    folds: tuple[FoldSummary, ...]
    metrics: Mapping[str, float | int | None]
    metadata: Mapping[str, Any]


class BacktestEngine:
    """Run forecast-only expanding or rolling walk-forward folds.

    The engine owns target creation, target-horizon purge, embargo, and output
    validation. Models own only causal feature construction and prediction.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run_yfinance(self, source: YFinanceSource, model_factory: ModelFactory) -> BacktestResult:
        snapshot = source.load()
        return self.run(snapshot.bars, model_factory, data_metadata=snapshot.metadata)

    def run(
        self,
        bars: pd.DataFrame | MarketData,
        model_factory: ModelFactory,
        *,
        data_metadata: Mapping[str, Any] | None = None,
    ) -> BacktestResult:
        if isinstance(bars, MarketData):
            if data_metadata is not None:
                raise ValueError("Pass metadata either in MarketData or data_metadata, not both.")
            data_metadata = bars.metadata
            bars = bars.bars

        raw = normalize_bars(bars)
        self._validate_reserved_columns(raw)
        target, target_end = self._build_target(raw)
        config = self.config
        splitter = config.walk_forward
        horizon = config.target.horizon_bars
        frames: list[pd.DataFrame] = []
        summaries: list[FoldSummary] = []
        prior_tests: list[tuple[int, int]] = []
        model_instances: list[object] = []
        model_ids: set[int] = set()
        last_test_end = 0

        for test_start, test_end, candidate_positions in self._fold_plans(raw.index):
            train_positions, embargoed_rows, prior_oos_excluded_rows, purged_rows = self._training_positions(
                test_start,
                prior_tests,
                candidate_positions,
            )
            if len(train_positions) >= splitter.min_train_rows:
                fold = len(summaries) + 1
                train = raw.iloc[train_positions].copy(deep=True)
                train[config.target_column] = target.iloc[train_positions].to_numpy(dtype=float)
                test = raw.iloc[test_start:test_end].copy(deep=True)
                context = FoldContext(
                    fold=fold,
                    target_column=config.target_column,
                    prediction_column=config.prediction_column,
                    horizon_bars=horizon,
                    train_start=train.index[0],
                    train_end=train.index[-1],
                    test_start=test.index[0],
                    test_end=test.index[-1],
                    warmup=raw.iloc[:test_start].copy(deep=True),
                )
                model = model_factory(context)
                if model is None or not hasattr(model, "fit") or not hasattr(model, "predict"):
                    raise TypeError("model_factory must return an object with fit() and predict().")
                if id(model) in model_ids:
                    raise ValueError("model_factory must return a fresh model for every fold.")
                model_ids.add(id(model))
                model_instances.append(model)

                fitted = model.fit(train.copy(deep=True), context)
                predictor = model if fitted is None else fitted
                if not hasattr(predictor, "predict"):
                    raise TypeError("fit() returned an object without predict().")
                if id(predictor) not in model_ids:
                    model_ids.add(id(predictor))
                    model_instances.append(predictor)
                elif predictor is not model:
                    raise ValueError("fit() must not return a predictor reused from an earlier fold.")
                output = predictor.predict(test.copy(deep=True), context)
                prediction, adapter_output = self._extract_prediction(output, test.index, test.columns)

                actual = target.iloc[test_start:test_end].to_numpy(dtype=float)
                scored = test.copy(deep=True)
                scored[config.target_column] = actual
                scored[config.prediction_column] = prediction.to_numpy(dtype=float)
                for column in adapter_output.columns:
                    scored[column] = adapter_output[column].to_numpy()
                scored["target_end"] = target_end.iloc[test_start:test_end].to_numpy()
                scored["fold"] = fold
                scored["error"] = scored[config.prediction_column] - scored[config.target_column]
                scored["direction_correct"] = (
                    np.sign(scored[config.prediction_column]) == np.sign(scored[config.target_column])
                )
                frames.append(scored)
                summaries.append(
                    FoldSummary(
                        fold=fold,
                        train_start=train.index[0],
                        train_end=train.index[-1],
                        test_start=test.index[0],
                        test_end=test.index[-1],
                        train_rows=len(train),
                        test_rows=len(test),
                        purged_rows=purged_rows,
                        embargoed_rows=embargoed_rows,
                        prior_oos_excluded_rows=prior_oos_excluded_rows,
                    )
                )
                prior_tests.append((test_start, test_end))
                last_test_end = test_end

        if not frames:
            raise ValueError("No complete walk-forward folds. Supply more bars or reduce split settings.")

        predictions = pd.concat(frames, axis=0)
        metadata = dict(data_metadata or {})
        metadata.update(
            {
                "bar_count": len(raw),
                "data_hash": fingerprint_bars(raw),
                "target_type": "log_return",
                "price_column": canonical_column_name(config.target.price_column),
                "horizon_bars": horizon,
                "timezone": None if raw.index.tz is None else str(raw.index.tz),
                "completed_folds": len(summaries),
                "unscored_tail_rows": len(raw) - last_test_end,
            }
        )
        return BacktestResult(
            config=config,
            predictions=predictions,
            folds=tuple(summaries),
            metrics=self._metrics(predictions),
            metadata=metadata,
        )

    def _validate_reserved_columns(self, raw: pd.DataFrame) -> None:
        collisions = {self.config.target_column, self.config.prediction_column}.intersection(raw.columns)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"Raw bars already contain reserved result columns: {names}.")

    def _build_target(self, raw: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        price_column = canonical_column_name(self.config.target.price_column)
        if price_column not in raw.columns:
            raise ValueError(f"Target price column is missing: {price_column}.")
        price = pd.to_numeric(raw[price_column], errors="raise").astype(float)
        horizon = self.config.target.horizon_bars
        target = np.log(price.shift(-horizon) / price)
        target_end = pd.Series(raw.index, index=raw.index).shift(-horizon)
        return target, target_end

    def _fold_plans(self, index: pd.DatetimeIndex) -> Iterator[tuple[int, int, np.ndarray]]:
        splitter = self.config.walk_forward
        horizon = self.config.target.horizon_bars
        if not isinstance(splitter, CalendarWalkForwardConfig):
            test_start = splitter.min_train_rows + horizon + splitter.extra_purge_bars
            while test_start + splitter.test_rows + horizon <= len(index):
                test_end = test_start + splitter.test_rows
                yield test_start, test_end, np.arange(test_start, dtype=int)
                test_start += splitter.effective_step_rows
            return

        train_offset = parse_window(splitter.train_window)
        test_offset = parse_window(splitter.test_window)
        step_offset = parse_window(splitter.effective_step_window)
        train_end = index.min() + train_offset
        previous_test_end = 0
        while True:
            train_start = train_end - train_offset
            candidates = np.flatnonzero((index > train_start) & (index <= train_end))
            if candidates.size:
                test_start = int(candidates[-1]) + splitter.pre_test_gap_rows + 1
                if test_start >= len(index):
                    break
                if splitter.test_rows is None:
                    test_end_date = index[test_start] + test_offset
                    if test_end_date > index[-1]:
                        break
                    test_positions = np.flatnonzero((index >= index[test_start]) & (index <= test_end_date))
                else:
                    test_positions = np.arange(test_start, test_start + splitter.test_rows, dtype=int)
                if not test_positions.size or int(test_positions[-1]) >= len(index) or int(test_positions[-1]) + horizon >= len(index):
                    break
                test_end = int(test_positions[-1]) + 1
                if test_start < previous_test_end:
                    train_end = train_end + step_offset
                    continue
                yield test_start, test_end, candidates
                previous_test_end = test_end
            train_end = train_end + step_offset
            if train_end >= index[-1]:
                break

    def _training_positions(
        self,
        test_start: int,
        prior_tests: list[tuple[int, int]],
        candidate_positions: np.ndarray,
    ) -> tuple[np.ndarray, int, int, int]:
        splitter = self.config.walk_forward
        boundary = test_start - self.config.target.horizon_bars - splitter.extra_purge_bars
        if boundary <= 0:
            return np.array([], dtype=int), 0, 0, len(candidate_positions)

        candidates = np.asarray(candidate_positions, dtype=int)
        candidates = candidates[candidates < test_start]
        before_purge = len(candidates)
        candidates = candidates[candidates < boundary]
        embargo_mask = np.zeros(boundary, dtype=bool)
        prior_oos_mask = np.zeros(boundary, dtype=bool)
        for prior_start, prior_end in prior_tests:
            if not splitter.reuse_prior_oos:
                prior_oos_mask[prior_start:min(prior_end, boundary)] = True
            embargo_start = min(prior_end, boundary)
            embargo_end = min(prior_end + splitter.embargo_rows, boundary)
            embargo_mask[embargo_start:embargo_end] = True
        eligible = candidates[~(embargo_mask[candidates] | prior_oos_mask[candidates])]
        max_train_rows = getattr(splitter, "max_train_rows", None)
        if max_train_rows is not None:
            eligible = eligible[-max_train_rows:]
        return (
            eligible,
            int(embargo_mask[candidates].sum()),
            int(prior_oos_mask[candidates].sum()),
            before_purge - len(candidates),
        )

    def _extract_prediction(
        self,
        output: object,
        expected_index: pd.DatetimeIndex,
        raw_columns: pd.Index,
    ) -> tuple[pd.Series, pd.DataFrame]:
        adapter_output = pd.DataFrame(index=expected_index)
        if isinstance(output, pd.DataFrame):
            if not output.index.equals(expected_index):
                raise ValueError("Returned prediction frame index must exactly match the test index.")
            if self.config.prediction_column not in output.columns:
                raise ValueError(f"Returned prediction frame lacks {self.config.prediction_column!r}.")
            prediction = output[self.config.prediction_column]
            reserved = {
                self.config.target_column,
                "target_end",
                "fold",
                "error",
                "direction_correct",
            }
            extras = [
                column
                for column in output.columns
                if column not in raw_columns and column != self.config.prediction_column
            ]
            collisions = reserved.intersection(extras)
            if collisions:
                raise ValueError(f"Adapter output uses reserved result columns: {', '.join(sorted(collisions))}.")
            adapter_output = output.loc[:, extras].copy()
        elif isinstance(output, pd.Series):
            if not output.index.equals(expected_index):
                raise ValueError("Returned prediction series index must exactly match the test index.")
            prediction = output
        else:
            raise TypeError("predict() must return a pandas DataFrame or Series.")

        prediction = pd.to_numeric(prediction, errors="coerce")
        values = prediction.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError("Predictions must be finite numeric values for every test row.")
        return pd.Series(values, index=expected_index, name=self.config.prediction_column), adapter_output

    def _metrics(self, predictions: pd.DataFrame) -> Mapping[str, float | int | None]:
        actual = predictions[self.config.target_column].to_numpy(dtype=float)
        predicted = predictions[self.config.prediction_column].to_numpy(dtype=float)
        correlation: float | None
        if len(actual) < 2 or np.isclose(actual.std(), 0.0) or np.isclose(predicted.std(), 0.0):
            correlation = None
        else:
            correlation = float(np.corrcoef(actual, predicted)[0, 1])
        return {
            "count": int(len(actual)),
            "mae": float(np.mean(np.abs(predicted - actual))),
            "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "correlation": correlation,
            "directional_hit_rate": float(np.mean(np.sign(predicted) == np.sign(actual))),
        }
