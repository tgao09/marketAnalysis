import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from telemetry_dashboard.artifacts import collect_artifact_snapshot
from telemetry_dashboard.registry import workflow_by_id


class ArtifactSnapshotTests(unittest.TestCase):
    def test_training_snapshot_reads_metrics(self):
        spec = workflow_by_id("training_gbm_return")
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "AAPL" / "regular"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "summary": {"folds": 3, "mae_mean": 0.12, "mse_mean": 0.34},
                        "folds": [
                            {"fold": 1, "mae": 0.1, "mse": 0.2},
                            {"fold": 2, "mae": 0.12, "mse": 0.3},
                        ],
                    }
                )
            )
            snapshot = collect_artifact_snapshot(spec, [artifact_dir])
            self.assertIn("AAPL folds", snapshot["cards"])
            self.assertIn("mae", snapshot["series"])

    def test_backtest_snapshot_reads_summary_and_trades(self):
        spec = workflow_by_id("backtesting_gbm_return")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_path = tmp / "gbm_return_summary.json"
            trades_path = tmp / "gbm_return_trades.csv"
            summary_path.write_text(json.dumps({"avg_return_pct": 0.08, "avg_pnl": 120.0, "total_trades": 2}))
            pd.DataFrame({"pnl": [100.0, -20.0]}).to_csv(trades_path, index=False)
            snapshot = collect_artifact_snapshot(spec, [summary_path, trades_path])
            self.assertEqual(snapshot["cards"]["avg_return_pct"], 0.08)
            self.assertIn("cumulative_pnl", snapshot["series"])

    def test_optimization_snapshot_reads_final_report(self):
        spec = workflow_by_id("optimization_gp_return")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "final_report.json").write_text(
                json.dumps(
                    {
                        "winner": {
                            "trial_number": 5,
                            "holdout": {"aggregate": {"basket_mean_avg_return_pct": 0.11}},
                        },
                        "top_10_trials": [
                            {"trial_number": 4, "objective_value": 0.09},
                            {"trial_number": 5, "objective_value": 0.11},
                        ],
                    }
                )
            )
            snapshot = collect_artifact_snapshot(spec, [run_dir])
            self.assertEqual(snapshot["cards"]["winner_trial"], 5)
            self.assertIn("objective", snapshot["series"])


if __name__ == "__main__":
    unittest.main()
