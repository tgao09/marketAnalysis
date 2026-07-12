from __future__ import annotations

import pandas as pd
import unittest

from common.backtesting import Bar, PortfolioObservation, PortfolioSnapshot
from rl_portfolio_management.replay_forecast_baselines import ForecastTargetPolicy


def _observation(date: str) -> PortfolioObservation:
    timestamp = pd.Timestamp(date, tz="America/New_York")
    snapshot = PortfolioSnapshot(
        cash=10_000, equity=10_000, unrealized_pnl=0, realized_pnl=0,
        gross_exposure=0, net_exposure=0, recent_turnover=0, drawdown=0,
        positions={}, pending_orders=(),
    )
    return PortfolioObservation(timestamp, {"A": Bar(1, 1, 1, 1, 1), "B": Bar(1, 1, 1, 1, 1)}, snapshot)


class ForecastReplayTests(unittest.TestCase):
  def test_forecasts_are_date_exact_without_future_or_stale_access(self):
    forecasts = pd.DataFrame({
        "symbol": ["A", "B"],
        "date": [pd.Timestamp("2025-01-03", tz="America/New_York"),
                 pd.Timestamp("2025-01-06", tz="America/New_York")],
        "pred_mean_log": [0.02, -0.03],
    })
    policy = ForecastTargetPolicy(forecasts, symbols=("A", "B"))

    before = policy.weights_at(pd.Timestamp("2025-01-02", tz="America/New_York"))
    first = policy.weights_at(pd.Timestamp("2025-01-03", tz="America/New_York"))
    stale = policy.weights_at(pd.Timestamp("2025-01-04", tz="America/New_York"))

    self.assertEqual(before, {"A": 0.0, "B": 0.0})
    self.assertEqual(first, {"A": 0.5, "B": 0.0})
    self.assertEqual(stale, {"A": 0.0, "B": 0.0})


  def test_target_gross_is_budget_and_absent_symbol_is_zero(self):
    forecasts = pd.DataFrame({
        "symbol": ["A", "B"],
        "date": [pd.Timestamp("2025-01-03", tz="America/New_York")] * 2,
        "pred_mean_log": [0.01, -0.03],
    })
    policy = ForecastTargetPolicy(forecasts, symbols=("A", "B", "C"), gross_budget=0.5)
    weights = policy.weights_at(pd.Timestamp("2025-01-03", tz="America/New_York"))

    self.assertAlmostEqual(sum(abs(value) for value in weights.values()), 0.5)
    self.assertAlmostEqual(weights["A"], 0.125)
    self.assertAlmostEqual(weights["B"], -0.375)
    self.assertEqual(weights["C"], 0.0)
    actions = policy.act(_observation("2025-01-03"))
    self.assertAlmostEqual(sum(abs(action.target_notional) for action in actions), 5_000)

  def test_sparse_cross_section_uses_equal_signed_weights(self):
    date = pd.Timestamp("2025-01-03", tz="America/New_York")
    forecasts = pd.DataFrame({"symbol": ["A", "B"], "date": [date, date],
                              "pred_mean_log": [0.001, -0.04]})
    policy = ForecastTargetPolicy(forecasts, symbols=tuple("ABCDEFG"), gross_budget=0.5)
    weights = policy.weights_at(date)
    self.assertAlmostEqual(weights["A"], 0.25)
    self.assertAlmostEqual(weights["B"], -0.25)
    self.assertTrue(all(weights[symbol] == 0 for symbol in "CDEFG"))


if __name__ == "__main__":
    unittest.main()
