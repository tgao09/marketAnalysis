import unittest

import pandas as pd

from common.backtesting import PortfolioWalkForward, PortfolioWalkForwardConfig
from rl_portfolio_management.final_evaluate import config_from_params, exclude_development_fold


def panel():
    index = pd.bdate_range("2018-01-01", "2023-12-31", tz="America/New_York")
    frame = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                          "Close": 100.0, "Volume": 1_000.0}, index=index)
    return {"A": frame, "B": frame.copy()}


class FinalEvaluateTests(unittest.TestCase):
    def test_config_construction_freezes_params_and_seed(self):
        config = config_from_params({
            "net_width": 64, "learning_rate": 1e-4, "action_cadence": 5,
            "reward_drawdown": 0.12, "reward_turnover": 0.004,
        }, timesteps=321, seed=29)
        self.assertEqual(config.net_arch, (64, 64))
        self.assertEqual((config.timesteps, config.seed, config.action_cadence), (321, 29, 5))
        self.assertAlmostEqual(config.reward.drawdown, 0.12)
        self.assertAlmostEqual(config.reward.turnover, 0.004)

    def test_exclusion_starts_at_original_fold_one_train_start(self):
        bars = panel()
        config = PortfolioWalkForwardConfig(purge_bars=60, embargo_bars=5)
        original = tuple(PortfolioWalkForward(config).selection_folds(bars))
        sliced, audit = exclude_development_fold(bars, config)
        self.assertEqual(next(iter(sliced.values())).index[0], original[1].train_start)
        self.assertEqual(audit["original_development_fold"], 0)
        self.assertEqual(audit["remaining_panel_start"], original[1].train_start.isoformat())


if __name__ == "__main__":
    unittest.main()
