import unittest

import numpy as np
import pandas as pd

from rl_portfolio_management.features import TrainingFoldScaler, build_asset_features


def bars(seed: int = 7, periods: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-02", periods=periods)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, periods)))
    return pd.DataFrame(
        {
            "OPEN": close * (1 + rng.normal(0, 0.002, periods)),
            "High": close * 1.01,
            "low": close * 0.99,
            "Close": close,
            "VOLUME": rng.integers(1_000_000, 5_000_000, periods),
        },
        index=index,
    )


class FeatureTests(unittest.TestCase):
    def test_future_mutation_cannot_change_earlier_features(self):
        asset, spy = bars(1), bars(2)
        original = build_asset_features(asset, spy)
        cutoff = asset.index[119]
        mutated_asset, mutated_spy = asset.copy(), spy.copy()
        mutated_asset.loc[mutated_asset.index > cutoff, "Close"] *= 25
        mutated_asset.loc[mutated_asset.index > cutoff, "VOLUME"] *= 50
        mutated_spy.loc[mutated_spy.index > cutoff, "Close"] *= 0.05
        changed = build_asset_features(mutated_asset, mutated_spy)
        pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])

    def test_expected_features_and_case_insensitive_ohlcv(self):
        result = build_asset_features(bars(3), bars(4))
        expected = {
            "return_1", "return_60", "rsi_14", "bollinger_z_20",
            "bollinger_width_20", "volatility_60", "volume_change_1",
            "volume_z_20", "relative_return_1", "relative_momentum_60",
            "market_corr_20", "market_beta_20",
        }
        self.assertTrue(expected.issubset(result.columns))
        self.assertTrue(np.isfinite(result.iloc[-1]).all())

    def test_scaler_uses_training_only_and_is_immutable(self):
        columns = ["a", "b"]
        train = {
            "AAA": pd.DataFrame([[1.0, 10.0], [3.0, 14.0]], columns=columns),
            "BBB": pd.DataFrame([[5.0, 18.0]], columns=columns),
        }
        scaler = TrainingFoldScaler.fit(train)
        validation = pd.DataFrame([[1_000_000.0, -1_000_000.0]], columns=columns)
        transformed = scaler.transform(validation)
        self.assertEqual(dict(scaler.means), {"a": 3.0, "b": 14.0})
        self.assertGreater(transformed.iloc[0, 0], 100_000)
        with self.assertRaises(TypeError):
            scaler.means["a"] = 0.0

    def test_scaler_fingerprint_is_deterministic_and_data_sensitive(self):
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [4.0, 8.0]})
        first = TrainingFoldScaler.fit(frame)
        second = TrainingFoldScaler.fit(frame.copy())
        changed = frame.copy()
        changed.loc[1, "x"] = 3.0
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, TrainingFoldScaler.fit(changed).fingerprint)


if __name__ == "__main__":
    unittest.main()
