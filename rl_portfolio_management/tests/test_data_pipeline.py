import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rl_portfolio_management.data_pipeline import SnapshotConfig, align_frames, canonicalize, persist_snapshot


class DataPipelineTests(unittest.TestCase):
    def test_daily_dates_localize_to_exchange_date_without_prior_day_shift(self):
        raw = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
            index=pd.DatetimeIndex(["2024-01-03"]),
        )
        got = canonicalize(raw, "AAA", SnapshotConfig("2024-01-01", "2024-02-01"))
        self.assertEqual(got.index[0].date().isoformat(), "2024-01-03")
        self.assertEqual(str(got.index.tz), "America/New_York")

    def setUp(self):
        self.config = SnapshotConfig("2020-01-01", "2020-01-10", symbols=("AAPL",), benchmark="SPY")

    def test_canonicalize_orders_deduplicates_and_does_not_fill(self):
        index = pd.to_datetime(["2020-01-03 21:00Z", "2020-01-02 21:00Z", "2020-01-03 21:00Z"])
        raw = pd.DataFrame({"Open": [3, 1, 4], "High": [3, 1, 4], "Low": [3, 1, 4], "Close": [3, 1, 4], "Volume": [30, 10, 40]}, index=index)
        got = canonicalize(raw, "AAPL", self.config)
        self.assertTrue(got.index.is_monotonic_increasing)
        self.assertEqual(len(got), 2)
        self.assertEqual(got.iloc[-1]["close"], 4)
        self.assertTrue(pd.isna(got.iloc[0]["dividends"]) is False)

    def test_alignment_uses_intersection_without_fill(self):
        a = pd.DataFrame({"close": [1, 2]}, index=pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"]))
        b = pd.DataFrame({"close": [3, 4]}, index=pd.to_datetime(["2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z"]))
        got = align_frames({"A": a, "B": b})
        self.assertEqual(list(got["A"].index), [pd.Timestamp("2020-01-02T00:00:00Z")])
        self.assertEqual(got["B"].iloc[0]["close"], 3)

    def test_manifest_is_versioned_and_fingerprinted(self):
        frame = pd.DataFrame({c: [1.0] for c in ("open", "high", "low", "close", "volume", "dividends", "stock_splits")}, index=pd.DatetimeIndex(["2020-01-02"], tz="America/New_York", name="timestamp"))
        with tempfile.TemporaryDirectory() as tmp:
            path = persist_snapshot({"AAPL": frame, "SPY": frame}, self.config, tmp)
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["content_sha256"]), 64)
            self.assertIn("survivorship", manifest["survivorship_disclosure"])
            self.assertEqual(manifest["symbols"]["AAPL"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
