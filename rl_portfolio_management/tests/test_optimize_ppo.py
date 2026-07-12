import unittest

from rl_portfolio_management.optimize_ppo import robust_validation_score


class RobustValidationScoreTests(unittest.TestCase):
    def test_invalid_metrics_receive_finite_penalty(self):
        self.assertEqual(robust_validation_score([]), -1000.0)
        self.assertEqual(robust_validation_score([{"calmar": None, "sharpe": 1,
                                                   "maximum_drawdown": .1}]), -1000.0)

    def test_exceptional_fold_is_bounded_and_dispersion_penalized(self):
        stable = [{"calmar": 1, "sharpe": 1, "maximum_drawdown": .1}] * 2
        spike = [stable[0], {"calmar": 1_000_000, "sharpe": 1_000_000,
                             "maximum_drawdown": .01}]
        self.assertGreater(robust_validation_score(stable), robust_validation_score(spike))
        self.assertLess(robust_validation_score(spike), 10)

    def test_higher_risk_adjusted_result_scores_better(self):
        weak = [{"calmar": .2, "sharpe": .3, "maximum_drawdown": .3}]
        strong = [{"calmar": 1.2, "sharpe": 1.0, "maximum_drawdown": .1}]
        self.assertGreater(robust_validation_score(strong), robust_validation_score(weak))


if __name__ == "__main__":
    unittest.main()
