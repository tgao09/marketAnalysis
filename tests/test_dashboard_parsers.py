import unittest

from telemetry_dashboard.parsers import LineParser


class LineParserTests(unittest.TestCase):
    def test_training_lines_emit_loss_and_fold_metrics(self):
        parser = LineParser("training_gp_return")
        parser.parse("Fold 2 | Train: 2024-01-01 -> 2024-06-01 | Test: 2024-06-02 -> 2024-07-01")
        events = parser.parse("MAE(log): 0.123456 | MAE(simple): 1.23% | MSE: 0.456789 | Dir: 55.00% | Coverage95: 80.00%")
        metric_names = {event["name"] for event in events if event["kind"] == "series"}
        self.assertIn("mae", metric_names)
        self.assertIn("mse", metric_names)
        self.assertIn("directional", metric_names)
        self.assertIn("coverage_95", metric_names)

        loss_events = parser.parse("Iter 50/200 - Loss: 1.2345")
        self.assertTrue(any(event["kind"] == "series" and event["name"] == "loss" for event in loss_events))

    def test_optimization_lines_track_best_objective(self):
        parser = LineParser("optimization_gp_return")
        parser.parse("[I 2026-03-08 12:00:00] Trial 0 finished with value: 0.12 and parameters: {}")
        events = parser.parse("[I 2026-03-08 12:00:01] Trial 1 finished with value: 0.18 and parameters: {}")
        objectives = [event for event in events if event["kind"] == "series" and event["name"] == "best_objective"]
        self.assertEqual(len(objectives), 1)
        self.assertEqual(objectives[0]["y"], 0.18)

    def test_prediction_lines_capture_probabilities_and_interval(self):
        parser = LineParser("prediction_hmm_regime")
        parser.parse("State: Risk Off (id=3)")
        events = parser.parse("Probabilities: p_state_0=0.1000, p_state_1=0.2000, p_state_2=0.3000, p_state_3=0.4000")
        self.assertTrue(any(event["kind"] == "probabilities" for event in events))

        gp_parser = LineParser("prediction_gp_return")
        gp_parser.parse("AAPL 5-day forward log-return forecast")
        prediction_events = gp_parser.parse("Mean simple return: 1.25%")
        self.assertTrue(any(event["kind"] == "prediction" and event["label"] == "AAPL" for event in prediction_events))
        interval_events = gp_parser.parse("95% interval (simple return): [-3.50%, 5.25%]")
        self.assertTrue(any(event["kind"] == "interval" and event["label"] == "AAPL" for event in interval_events))

    def test_artifact_lines_are_captured(self):
        parser = LineParser("backtesting_gbm_return")
        events = parser.parse("Summary saved to: C:\\tmp\\gbm_return_summary.json")
        artifacts = [event for event in events if event["kind"] == "artifact"]
        self.assertEqual(len(artifacts), 1)
        self.assertTrue(str(artifacts[0]["path"]).endswith("gbm_return_summary.json"))

    def test_backtest_lines_emit_live_summary_scalars(self):
        parser = LineParser("backtesting_gbm_return")
        events = parser.parse("2025-01-03 | Train: 2024-01-01 -> 2024-12-31 | Pred: +1.25% | PnL: +12.50")
        scalar_values = {
            event["name"]: event["value"]
            for event in events
            if event["kind"] == "scalar"
        }
        self.assertEqual(scalar_values["total_trades"], 1)
        self.assertEqual(scalar_values["cumulative_pnl"], 12.5)
        self.assertEqual(scalar_values["avg_pnl"], 12.5)
        self.assertEqual(scalar_values["win_rate"], 1.0)

        events = parser.parse("2025-01-10 | Train: 2024-01-08 -> 2025-01-07 | Pred: -0.50% | PnL: -2.50")
        scalar_values = {
            event["name"]: event["value"]
            for event in events
            if event["kind"] == "scalar"
        }
        self.assertEqual(scalar_values["total_trades"], 2)
        self.assertEqual(scalar_values["cumulative_pnl"], 10.0)
        self.assertEqual(scalar_values["avg_pnl"], 5.0)
        self.assertEqual(scalar_values["win_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
