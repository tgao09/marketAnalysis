import unittest

import numpy as np
import pandas as pd

from common.backtesting import Hold, PortfolioReplay
from rl_portfolio_management.features import TrainingFoldScaler, build_asset_features
from rl_portfolio_management.ppo import SB3PortfolioPolicy, prepare_replay_features
from rl_portfolio_management.rl_env import CrossAssetPortfolioEnv


def bars(periods=140):
    index = pd.date_range("2020-01-02", periods=periods, freq="B")
    result = {}
    for j, symbol in enumerate(("AAA", "BBB")):
        close = 100 + j * 20 + np.linspace(0, 15, periods) + np.sin(np.arange(periods) / 7)
        result[symbol] = pd.DataFrame({"Open": close - .2, "High": close + 1,
            "Low": close - 1, "Close": close, "Volume": 1_000 + np.arange(periods)}, index=index)
    return result


class DummyModel:
    def __init__(self):
        self.observations = []

    def predict(self, observation, deterministic=True):
        self.observations.append(observation.copy())
        return np.zeros(2, dtype=np.float32), None


class CapturePolicy:
    def __init__(self, adapter):
        self.adapter, self.first = adapter, None

    def act(self, observation):
        if self.first is None:
            self.first = self.adapter.observation_vector(observation)
        return ()


class PPOAdapterTests(unittest.TestCase):
    def test_adapter_and_env_observations_match(self):
        panel = bars()
        features = {s: build_asset_features(f).iloc[60:] for s, f in panel.items()}
        scaler = TrainingFoldScaler.fit(features)
        scaled = {s: scaler.transform(f) for s, f in features.items()}
        trimmed = {s: f.iloc[60:] for s, f in panel.items()}
        env = CrossAssetPortfolioEnv(trimmed, scaled)
        env_observation, _ = env.reset()
        capture = CapturePolicy(SB3PortfolioPolicy(DummyModel(), scaled))
        PortfolioReplay().run(trimmed, capture)
        np.testing.assert_allclose(capture.first, env_observation, rtol=0, atol=1e-7)

    def test_future_bar_change_cannot_change_earlier_features(self):
        panel = bars()
        context = {s: f.iloc[:100] for s, f in panel.items()}
        replay = {s: f.iloc[100:] for s, f in panel.items()}
        train_features = {s: build_asset_features(f).iloc[60:] for s, f in context.items()}
        scaler = TrainingFoldScaler.fit(train_features)
        first = prepare_replay_features(context, replay, scaler)
        changed = {s: f.copy() for s, f in replay.items()}
        changed["AAA"].iloc[-1, changed["AAA"].columns.get_loc("Close")] *= 10
        second = prepare_replay_features(context, changed, scaler)
        cutoff = replay["AAA"].index[-2]
        pd.testing.assert_frame_equal(first["AAA"].loc[:cutoff], second["AAA"].loc[:cutoff])

    def test_adapter_holds_off_cadence_and_keeps_observations_aligned(self):
        panel = bars()
        features = {s: build_asset_features(f).iloc[60:] for s, f in panel.items()}
        scaler = TrainingFoldScaler.fit(features)
        scaled = {s: scaler.transform(f) for s, f in features.items()}
        trimmed = {s: f.iloc[60:] for s, f in panel.items()}
        model = DummyModel()
        adapter = SB3PortfolioPolicy(model, scaled, action_cadence=3)
        PortfolioReplay().run(trimmed, adapter)
        self.assertEqual(len(model.observations), (len(trimmed["AAA"]) + 2) // 3)

        env = CrossAssetPortfolioEnv(trimmed, scaled, action_cadence=3)
        observation, _ = env.reset()
        np.testing.assert_allclose(model.observations[0], observation, rtol=0, atol=1e-7)
        for _ in range(3):
            observation, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(model.observations[1], observation, rtol=0, atol=1e-7)

        recorded = []
        second = SB3PortfolioPolicy(DummyModel(), scaled, action_cadence=3)
        class Recorder:
            def act(self, observation):
                action = second.act(observation)
                recorded.append(action)
                return action
        PortfolioReplay().run(trimmed, Recorder())
        self.assertIsInstance(recorded[1][0], Hold)
        self.assertIsInstance(recorded[2][0], Hold)
        second._action_counter = 2
        adapter.prepare_replay(trimmed)
        self.assertEqual(adapter._action_counter, 0)


if __name__ == "__main__":
    unittest.main()
