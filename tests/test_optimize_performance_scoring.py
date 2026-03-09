import importlib
import unittest

from gbm_return.optimize_performance import assess_candidate as assess_gbm_candidate
from gbm_return.optimize_performance import aggregate_basket_summary as aggregate_gbm_basket_summary
from gbm_return.optimize_performance import compute_objective_score as compute_gbm_objective_score
import gp_return.backtest_walk_forward as gp_backtest

if not hasattr(gp_backtest, "build_backtest_pca_transformer"):
    gp_backtest.build_backtest_pca_transformer = lambda *args, **kwargs: None

gp_optimize = importlib.import_module("gp_return.optimize_performance")
aggregate_gp_basket_summary = gp_optimize.aggregate_basket_summary
compute_gp_objective_score = gp_optimize.compute_objective_score


class OptimizePerformanceScoringTests(unittest.TestCase):
    def test_gp_objective_uses_avg_pnl_and_max_drawdown(self):
        aggregate = aggregate_gp_basket_summary(
            {
                "AAPL": {
                    "avg_return_pct": 0.12,
                    "win_rate": 0.5,
                    "avg_pnl": 120.0,
                    "max_drawdown": -250.0,
                    "total_trades": 10,
                }
            }
        )

        self.assertAlmostEqual(aggregate["basket_worst_max_drawdown"], -250.0)
        self.assertAlmostEqual(aggregate["basket_objective_score"], -130.0)

    def test_gbm_objective_uses_avg_pnl_and_max_drawdown(self):
        aggregate = aggregate_gbm_basket_summary(
            {
                "AAPL": {
                    "avg_return_pct": 0.20,
                    "win_rate": 0.5,
                    "avg_pnl": 200.0,
                    "max_drawdown": -150.0,
                    "return_tstat": 2.0,
                    "trade_rate": 0.4,
                    "total_trades": 12,
                }
            }
        )

        self.assertAlmostEqual(aggregate["basket_worst_max_drawdown"], -150.0)
        self.assertAlmostEqual(aggregate["basket_objective_score"], 50.0)

    def test_explicit_objective_score_matches_formula(self):
        aggregate = {
            "basket_mean_avg_pnl": 180.0,
            "basket_worst_max_drawdown": -70.0,
        }

        self.assertAlmostEqual(compute_gp_objective_score(aggregate), 110.0)
        self.assertAlmostEqual(compute_gbm_objective_score(aggregate), 110.0)

    def test_gbm_assessment_reports_guardrail_failure_without_changing_objective(self):
        result = {
            "tickers": {"AAPL": {"max_drawdown": -150.0}},
            "aggregate": {
                "basket_total_trades": 120,
                "basket_objective_score": 50.0,
            },
        }
        baseline = {"AAPL": {"max_drawdown": -100.0}}

        assessment = assess_gbm_candidate(
            result=result,
            baseline_ticker_summaries=baseline,
            drawdown_worsen_limit=0.10,
            min_basket_trades=80,
        )

        self.assertFalse(assessment["guardrail_pass"])
        self.assertFalse(assessment["hard_reject"])
        self.assertAlmostEqual(assessment["basket_objective_score"], 50.0)

    def test_gbm_assessment_still_hard_rejects_trade_count_failures(self):
        result = {
            "tickers": {"AAPL": {"max_drawdown": -50.0}},
            "aggregate": {
                "basket_total_trades": 20,
                "basket_objective_score": 25.0,
            },
        }
        baseline = {"AAPL": {"max_drawdown": -100.0}}

        assessment = assess_gbm_candidate(
            result=result,
            baseline_ticker_summaries=baseline,
            drawdown_worsen_limit=0.10,
            min_basket_trades=80,
        )

        self.assertTrue(assessment["guardrail_pass"])
        self.assertTrue(assessment["hard_reject"])
        self.assertAlmostEqual(assessment["basket_objective_score"], 25.0)


if __name__ == "__main__":
    unittest.main()
