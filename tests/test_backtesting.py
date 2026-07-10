"""Focused verification for universal walk-forward backtesting."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from common.backtesting import (
    BacktestConfig,
    BacktestEngine,
    LogReturnTarget,
    WalkForwardConfig,
    YFinanceSource,
    normalize_bars,
)


def make_bars(rows: int = 24) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=rows)
    close = np.exp(np.arange(rows, dtype=float) * 0.01) * 100.0
    return pd.DataFrame(
        {"Open": close * 0.99, "High": close * 1.01, "Low": close * 0.98, "Close": close, "Volume": 1_000},
        index=index,
    )


class CapturingModel:
    def __init__(self, capture: dict[str, object]) -> None:
        self.capture = capture

    def fit(self, train: pd.DataFrame, context: object) -> "CapturingModel":
        self.capture.setdefault("model_ids", []).append(id(self))
        self.capture.setdefault("train_columns", []).append(tuple(train.columns))
        self.capture.setdefault("train_indices", {})[context.fold] = train.index.copy()
        train.loc[:, "close"] = -1.0
        return self

    def predict(self, test: pd.DataFrame, context: object) -> pd.DataFrame:
        self.capture.setdefault("test_columns", []).append(tuple(test.columns))
        self.capture.setdefault("warmup_lengths", []).append(len(context.warmup))
        test.loc[:, "close"] = -1.0
        test[context.prediction_column] = 0.01
        return test


class NaNModel:
    def fit(self, train: pd.DataFrame, context: object) -> "NaNModel":
        return self

    def predict(self, test: pd.DataFrame, context: object) -> pd.Series:
        return pd.Series(np.nan, index=test.index)


class BacktestEngineTests(unittest.TestCase):
    def test_horizon_purge_hides_test_target_and_scores_alignment(self) -> None:
        bars = make_bars()
        capture: dict[str, object] = {}
        config = BacktestConfig(
            target=LogReturnTarget(horizon_bars=2),
            walk_forward=WalkForwardConfig(
                min_train_rows=4,
                test_rows=3,
                step_rows=3,
                extra_purge_bars=1,
            ),
            target_column="future_log_2",
            prediction_column="forecast",
        )
        result = BacktestEngine(config).run(bars, lambda _: CapturingModel(capture))

        self.assertEqual(result.metrics["count"], 15)
        self.assertEqual(len(set(capture["model_ids"])), len(capture["model_ids"]))
        self.assertTrue(all("future_log_2" in columns for columns in capture["train_columns"]))
        self.assertTrue(all("forecast" not in columns for columns in capture["train_columns"]))
        self.assertTrue(all("future_log_2" not in columns for columns in capture["test_columns"]))
        self.assertTrue(all("forecast" not in columns for columns in capture["test_columns"]))

        first_test_position = bars.index.get_loc(result.folds[0].test_start)
        first_train = capture["train_indices"][1]
        last_train_position = bars.index.get_loc(first_train[-1])
        self.assertLess(last_train_position + 2 + 1, first_test_position)
        self.assertEqual(capture["warmup_lengths"][0], first_test_position)

        first_row = result.predictions.iloc[0]
        self.assertEqual(first_row.name, bars.index[first_test_position])
        self.assertEqual(first_row["target_end"], bars.index[first_test_position + 2])
        expected = np.log(bars["Close"].iloc[first_test_position + 2] / bars["Close"].iloc[first_test_position])
        self.assertAlmostEqual(first_row["future_log_2"], expected)
        self.assertEqual(first_row["close"], bars["Close"].iloc[first_test_position])

    def test_embargo_excludes_rows_from_later_training_fold(self) -> None:
        bars = make_bars(18)
        capture: dict[str, object] = {}
        config = BacktestConfig(
            target=LogReturnTarget(horizon_bars=1),
            walk_forward=WalkForwardConfig(
                min_train_rows=5,
                test_rows=2,
                step_rows=2,
                embargo_rows=2,
            ),
        )
        result = BacktestEngine(config).run(bars, lambda _: CapturingModel(capture))

        self.assertGreaterEqual(len(result.folds), 3)
        self.assertGreater(result.folds[2].embargoed_rows, 0)
        self.assertNotIn(bars.index[8], capture["train_indices"][3])

    def test_rejects_nonfinite_predictions(self) -> None:
        config = BacktestConfig(
            walk_forward=WalkForwardConfig(min_train_rows=4, test_rows=2),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            BacktestEngine(config).run(make_bars(12), lambda _: NaNModel())


class DataContractTests(unittest.TestCase):
    def test_yfinance_source_normalizes_and_records_metadata_without_network(self) -> None:
        raw = make_bars(10)
        source = YFinanceSource(
            "spy",
            period="1mo",
            history_fetcher=lambda symbol, **kwargs: raw,
        )
        snapshot = source.load()

        self.assertEqual(snapshot.metadata["symbol"], "SPY")
        self.assertEqual(snapshot.metadata["interval"], "1d")
        self.assertIn("data_hash", snapshot.metadata)
        self.assertIn("close", snapshot.bars.columns)

    def test_normalizer_rejects_multiasset_and_duplicate_timestamp_input(self) -> None:
        bars = make_bars(8)
        multiasset = bars.copy()
        multiasset.columns = pd.MultiIndex.from_product([multiasset.columns, ["SPY"]])
        with self.assertRaisesRegex(ValueError, "MultiIndex"):
            normalize_bars(multiasset)

        duplicate = pd.concat([bars, bars.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_bars(duplicate)


if __name__ == "__main__":
    unittest.main()
