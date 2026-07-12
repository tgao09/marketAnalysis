import unittest

import pandas as pd

from common.backtesting import PortfolioReplay, portfolio_metrics
from rl_portfolio_management.policies import AlwaysCashPolicy, EqualWeightLongPolicy, MomentumPolicy, RandomValidPolicy


def trending_frame(multiplier=1.0, periods=35):
    index = pd.date_range("2024-01-02", periods=periods, freq="B", tz="UTC")
    close = pd.Series([100 + multiplier * i for i in range(periods)], index=index)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1_000})


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.bars = {"UP": trending_frame(1), "FLAT": trending_frame(0), "DOWN": trending_frame(-0.5)}

    def test_always_cash(self):
        result = PortfolioReplay().run(self.bars, AlwaysCashPolicy())
        self.assertEqual(result.snapshots[-1][1].equity, 10_000)
        self.assertEqual(len(result.orders), 0)

    def test_equal_weight_respects_no_leverage(self):
        result = PortfolioReplay().run(self.bars, EqualWeightLongPolicy())
        self.assertTrue(all(snapshot.gross_exposure <= snapshot.equity + 1e-8 for _, snapshot in result.snapshots))
        self.assertGreater(len(result.orders), 0)

    def test_momentum_is_causal_and_selects_winner(self):
        result = PortfolioReplay().run(self.bars, MomentumPolicy(lookback=5, rebalance_every=5, long_fraction=1/3))
        self.assertGreater(result.snapshots[-1][1].positions["UP"].quantity, 0)
        self.assertGreater(portfolio_metrics(result)["ending_equity"], 10_000)

    def test_random_is_seeded_and_valid(self):
        first = PortfolioReplay().run(self.bars, RandomValidPolicy(seed=7))
        second = PortfolioReplay().run(self.bars, RandomValidPolicy(seed=7))
        self.assertEqual(first.orders, second.orders)
        self.assertTrue(all(snapshot.gross_exposure <= snapshot.equity + 1e-8 for _, snapshot in first.snapshots))


if __name__ == "__main__":
    unittest.main()
