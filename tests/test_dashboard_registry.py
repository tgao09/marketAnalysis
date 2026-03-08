import unittest

from telemetry_dashboard.registry import workflow_by_id


class RegistryCommandTests(unittest.TestCase):
    def test_training_gbm_command_includes_expected_flags(self):
        spec = workflow_by_id("training_gbm_return")
        command = spec.command(
            {
                "tickers": "AAPL,NVDA",
                "train_window": "18m",
                "feature_set": "F1",
                "lgbm_param_preset": "baseline",
                "include_time_index": True,
            }
        )
        self.assertIn("--tickers", command)
        self.assertIn("AAPL,NVDA", command)
        self.assertIn("--feature-set", command)
        self.assertIn("F1", command)
        self.assertIn("--include-time-index", command)

    def test_gp_vol_training_command_is_non_interactive(self):
        spec = workflow_by_id("training_gp_vol")
        command = spec.command(
            {
                "train_window": "2y",
                "test_window": "1m",
                "step_window": "1m",
                "train_iters": 50,
                "kernel_mode": "custom",
                "kernel_equation": "1+4",
                "kernel_lengthscale": 1.5,
                "kernel_period_length": 5.0,
                "kernel_outputscale": 2.0,
                "drop_time_index": True,
            }
        )
        self.assertIn("--kernel-mode", command)
        self.assertIn("custom", command)
        self.assertIn("--kernel-equation", command)
        self.assertIn("--kernel-lengthscale", command)
        self.assertIn("--drop-time-index", command)

    def test_prediction_gp_command_uses_pca_flag(self):
        spec = workflow_by_id("prediction_gp_return")
        command = spec.command({"tickers": "AAPL", "pca": True})
        self.assertIn("--tickers", command)
        self.assertIn("AAPL", command)
        self.assertIn("--pca", command)

    def test_optimization_gbm_command_includes_skip_flags(self):
        spec = workflow_by_id("optimization_gbm_return")
        command = spec.command(
            {
                "tickers": "AAPL,NVDA",
                "train_window": "2y",
                "test_window": "1m",
                "step_window": "1m",
                "n_trials": 10,
                "holdout_top_n": 3,
                "notional": 10000,
                "drawdown_worsen_limit": 0.1,
                "include_time_index": False,
                "skip_retrain": True,
            }
        )
        self.assertIn("--n-trials", command)
        self.assertIn("10", command)
        self.assertIn("--skip-retrain", command)


if __name__ == "__main__":
    unittest.main()
