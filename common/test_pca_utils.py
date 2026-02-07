import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pca_utils import (
    PCATransformer,
    load_pca_json,
    save_pca_json,
    select_k_by_variance,
)


class TestPCAUtils(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        index = pd.date_range("2023-01-02", periods=32, freq="B")
        self.df = pd.DataFrame(
            {
                "f1": rng.normal(0, 1, size=len(index)),
                "f2": rng.normal(0, 1, size=len(index)),
                "f3": rng.normal(0, 1, size=len(index)),
                "f4": rng.normal(0, 1, size=len(index)),
            },
            index=index,
        )
        self.feature_cols = ["f1", "f2", "f3", "f4"]
        self.train_df = self.df.iloc[:24].copy()
        self.test_df = self.df.iloc[24:].copy()

    def test_fit_transform_preserves_index_and_row_count(self):
        tr = PCATransformer()
        tr.fit(self.train_df, self.feature_cols)
        transformed = tr.transform(self.test_df)

        self.assertEqual(len(transformed), len(self.test_df))
        self.assertTrue(transformed.index.equals(self.test_df.index))

    def test_replace_mode_outputs_only_pc_columns(self):
        tr = PCATransformer(mode="replace")
        out = tr.fit_transform(self.train_df, self.feature_cols)

        self.assertTrue(all(col.startswith("pc_") for col in out.columns))
        self.assertTrue(all(col not in out.columns for col in self.feature_cols))

    def test_append_mode_outputs_raw_plus_pc_columns(self):
        tr = PCATransformer(mode="append")
        out = tr.fit_transform(self.train_df, self.feature_cols)

        k = tr.k_selected_
        self.assertIsNotNone(k)
        expected_cols = self.feature_cols + [f"pc_{i}" for i in range(1, int(k) + 1)]
        self.assertEqual(out.columns.tolist(), expected_cols)

    def test_select_k_by_variance_respects_threshold_and_max(self):
        explained = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)
        self.assertEqual(select_k_by_variance(explained, threshold=0.80, max_pcs=12), 3)
        self.assertEqual(select_k_by_variance(explained, threshold=0.95, max_pcs=2), 2)
        self.assertEqual(select_k_by_variance(explained, threshold=0.20, max_pcs=12), 1)

    def test_serialization_roundtrip_preserves_transform_results(self):
        tr = PCATransformer(mode="replace")
        tr.fit(self.train_df, self.feature_cols)
        expected = tr.transform(self.test_df)

        payload = tr.to_dict()
        restored = PCATransformer.from_dict(payload)
        from_dict_out = restored.transform(self.test_df)
        assert_frame_equal(expected, from_dict_out, rtol=1e-10, atol=1e-10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pca.json"
            save_pca_json(path, tr)
            loaded = load_pca_json(path)
            from_json_out = loaded.transform(self.test_df)
            assert_frame_equal(expected, from_json_out, rtol=1e-10, atol=1e-10)

    def test_missing_feature_column_raises_clear_error(self):
        tr = PCATransformer()
        tr.fit(self.train_df, self.feature_cols)
        broken = self.test_df.drop(columns=["f4"])
        with self.assertRaisesRegex(ValueError, "Missing feature columns"):
            tr.transform(broken)

    def test_zero_variance_feature_handled(self):
        train = self.train_df.copy()
        test = self.test_df.copy()
        train["constant"] = 1.0
        test["constant"] = 1.0

        tr = PCATransformer()
        feature_cols = self.feature_cols + ["constant"]
        tr.fit(train, feature_cols)

        self.assertEqual(float(tr.scaler_std_["constant"]), 1.0)
        out = tr.transform(test)
        self.assertFalse(out.isna().any().any())

    def test_inf_values_are_imputed(self):
        train = self.train_df.copy()
        test = self.test_df.copy()
        train.loc[train.index[0], "f1"] = np.inf
        train.loc[train.index[1], "f2"] = -np.inf
        test.loc[test.index[0], "f3"] = np.inf

        tr = PCATransformer()
        tr.fit(train, self.feature_cols)
        out = tr.transform(test)
        self.assertFalse(out.isna().any().any())

    def test_unfitted_transform_raises_clear_error(self):
        tr = PCATransformer()
        with self.assertRaisesRegex(ValueError, "not fitted"):
            tr.transform(self.test_df)


if __name__ == "__main__":
    unittest.main()
