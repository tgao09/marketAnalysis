import unittest

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from common.backtesting import PortfolioReplay, TargetPosition
from rl_portfolio_management.rl_env import CrossAssetPortfolioEnv


def panel():
    index = pd.date_range("2024-01-02", periods=7, freq="B")
    bars = {"AAA": pd.DataFrame({
        "Open": [100, 110, 108, 115, 112, 118, 120],
        "High": [102, 113, 111, 118, 116, 121, 123],
        "Low": [98, 107, 105, 111, 109, 115, 118],
        "Close": [101, 109, 110, 113, 115, 119, 121],
        "Volume": [1000] * 7,
    }, index=index)}
    features = {"AAA": pd.DataFrame({"signal": np.linspace(-1, 1, 7)}, index=index)}
    return bars, features


class ScriptedPolicy:
    def __init__(self, weights):
        self.weights = iter(weights)

    def act(self, observation):
        try:
            weight = next(self.weights)
        except StopIteration:
            return ()
        return (TargetPosition("AAA", target_notional=weight * observation.portfolio.equity),)


class CrossAssetPortfolioEnvTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        bars, features = panel()
        check_env(CrossAssetPortfolioEnv(bars, features), skip_render_check=True)

    def test_scripted_actions_match_portfolio_replay(self):
        bars, features = panel()
        actions = [float(x) for x in np.asarray([0.5, -0.4, 0.0, 0.7, 0.2, 0.0], dtype=np.float32)]
        replay = PortfolioReplay(starting_cash=10_000).run(bars, ScriptedPolicy(actions))
        env = CrossAssetPortfolioEnv(bars, features, starting_equity=10_000)
        env.reset(seed=9)
        infos = []
        for action in actions:
            _, _, _, _, info = env.step(np.asarray([action], dtype=np.float32))
            infos.append(info)
        replay_by_time = dict(replay.snapshots)
        for info in infos:
            snapshot = replay_by_time[info["timestamp"]]
            self.assertAlmostEqual(info["cash"], snapshot.cash, places=8)
            self.assertAlmostEqual(info["equity"], snapshot.equity, places=8)
            self.assertAlmostEqual(info["gross_exposure"] * info["equity"], snapshot.gross_exposure, places=8)

    def test_action_is_projected_and_fills_next_open(self):
        bars, features = panel()
        env = CrossAssetPortfolioEnv(bars, features, gross_budget=0.9)
        _, info0 = env.reset()
        _, _, _, _, info1 = env.step(np.asarray([1.0], dtype=np.float32))
        target_qty = 0.9 * info0["equity"] / bars["AAA"].iloc[0].Close
        expected_cash = 10_000 - target_qty * bars["AAA"].iloc[1].Open
        self.assertAlmostEqual(info1["cash"], expected_cash)
        self.assertLessEqual(info1["gross_exposure"], 1.0)

    def test_seeded_reset_is_deterministic(self):
        bars, features = panel()
        env = CrossAssetPortfolioEnv(bars, features)
        first, first_info = env.reset(seed=11)
        env.step(np.asarray([0.5], dtype=np.float32))
        second, second_info = env.reset(seed=11)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_info["equity"], second_info["equity"])

    def test_off_cadence_actions_create_no_fills_or_turnover(self):
        bars, features = panel()
        env = CrossAssetPortfolioEnv(bars, features, action_cadence=3)
        env.reset()
        env.step(np.asarray([0.5], dtype=np.float32))
        cash_after_decision = env._cash
        quantity_after_decision = env._positions["AAA"].quantity
        turnover_count = len(env._turnovers)
        env.step(np.asarray([-1.0], dtype=np.float32))
        self.assertEqual(env._cash, cash_after_decision)
        self.assertEqual(env._positions["AAA"].quantity, quantity_after_decision)
        self.assertEqual(len(env._turnovers), turnover_count + 1)
        self.assertEqual(env._turnovers[-1], 0.0)
        env.step(np.asarray([1.0], dtype=np.float32))
        self.assertEqual(env._cash, cash_after_decision)
        self.assertEqual(env._positions["AAA"].quantity, quantity_after_decision)
        self.assertEqual(env._turnovers[-1], 0.0)

    def test_action_cadence_must_be_positive_integer(self):
        bars, features = panel()
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CrossAssetPortfolioEnv(bars, features, action_cadence=value)


if __name__ == "__main__":
    unittest.main()
