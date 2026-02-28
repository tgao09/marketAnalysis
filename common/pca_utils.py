"""Reusable PCA transformer utilities for strategy pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def select_k_by_variance(
    explained_var_ratio: Sequence[float], threshold: float = 0.80, max_pcs: int = 12
) -> int:
    """Select the minimum number of components that reaches a variance threshold."""
    explained = np.asarray(explained_var_ratio, dtype=float)
    cumulative = np.cumsum(explained)
    k = int(np.searchsorted(cumulative, threshold) + 1)
    return max(1, min(k, max_pcs, explained.size))


class PCATransformer:
    """Leakage-safe fold transformer with imputation, scaling, and PCA projection."""

    def __init__(
        self,
        threshold: float = 0.80,
        max_pcs: int = 12,
        impute_strategy: str = "median",
        mode: str = "replace",
        pc_prefix: str = "pc_",
    ) -> None:
        self.threshold = float(threshold)
        self.max_pcs = int(max_pcs)
        self.impute_strategy = impute_strategy
        self.mode = mode
        self.pc_prefix = pc_prefix

        self.feature_columns_: list[str] | None = None
        self.imputer_values_: pd.Series | None = None
        self.scaler_mean_: pd.Series | None = None
        self.scaler_std_: pd.Series | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.k_selected_: int | None = None
        self.n_features_: int | None = None

    def fit(self, train_df: pd.DataFrame, feature_cols: Sequence[str]) -> PCATransformer:
        """Fit fold-local preprocessing and PCA on training rows only."""
        columns = [str(col) for col in feature_cols]
        prepared = self._prepare_input_frame(train_df, columns)

        if self.impute_strategy == "median":
            imputer_values = prepared.median()
        else:
            imputer_values = prepared.mean()

        filled = prepared.fillna(imputer_values)
        scaler_mean = filled.mean()
        scaler_std = filled.std().replace(0.0, 1.0).fillna(1.0)
        scaled = (filled - scaler_mean) / scaler_std

        scaled = scaled.replace([np.inf, -np.inf], np.nan)

        pca = PCA()
        pca.fit(scaled.to_numpy(dtype=float))

        k_selected = select_k_by_variance(
            explained_var_ratio=pca.explained_variance_ratio_,
            threshold=self.threshold,
            max_pcs=self.max_pcs,
        )

        self.feature_columns_ = columns
        self.imputer_values_ = imputer_values.reindex(columns)
        self.scaler_mean_ = scaler_mean.reindex(columns)
        self.scaler_std_ = scaler_std.reindex(columns)
        self.components_ = np.asarray(pca.components_, dtype=float)
        self.explained_variance_ratio_ = np.asarray(pca.explained_variance_ratio_, dtype=float)
        self.k_selected_ = int(k_selected)
        self.n_features_ = int(len(columns))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform rows using previously-fitted preprocessing and PCA state."""
        prepared = self._prepare_input_frame(frame, self.feature_columns_)
        filled = prepared.fillna(self.imputer_values_)
        scaled = (filled - self.scaler_mean_) / self.scaler_std_
        scaled = scaled.replace([np.inf, -np.inf], np.nan)

        pcs = scaled.to_numpy(dtype=float) @ self.components_[: self.k_selected_].T
        pc_cols = [f"{self.pc_prefix}{i}" for i in range(1, self.k_selected_ + 1)]
        pc_df = pd.DataFrame(pcs, index=frame.index, columns=pc_cols)

        if self.mode == "replace":
            return pc_df
        return frame[self.feature_columns_].copy().join(pc_df)

    def fit_transform(self, train_df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
        """Fit on a training frame and return transformed training rows."""
        return self.fit(train_df, feature_cols).transform(train_df)

    def transform_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit on train and transform both train/test with fold-local state."""
        self.fit(train_df, feature_cols)
        return self.transform(train_df), self.transform(test_df)

    def to_dict(self) -> dict:
        """Serialize fitted transformer state to a JSON-safe dictionary."""
        return {
            "feature_columns": list(self.feature_columns_),
            "imputer": {
                "strategy": self.impute_strategy,
                "values": {k: float(v) for k, v in self.imputer_values_.to_dict().items()},
            },
            "scaler": {
                "mean": {k: float(v) for k, v in self.scaler_mean_.to_dict().items()},
                "std": {k: float(v) for k, v in self.scaler_std_.to_dict().items()},
            },
            "pca": {
                "components": self.components_.tolist(),
                "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
                "k_selected": int(self.k_selected_),
                "n_features": int(self.n_features_),
            },
            "config": {
                "threshold": float(self.threshold),
                "max_pcs": int(self.max_pcs),
                "impute_strategy": self.impute_strategy,
                "mode": self.mode,
                "pc_prefix": self.pc_prefix,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> PCATransformer:
        """Deserialize a fitted transformer from payload produced by to_dict()."""
        feature_columns = payload["feature_columns"]
        imputer = payload["imputer"]
        scaler = payload["scaler"]
        pca = payload["pca"]
        config = payload["config"]

        transformer = cls(
            threshold=float(config.get("threshold", 0.80)),
            max_pcs=int(config.get("max_pcs", 12)),
            impute_strategy=str(config.get("impute_strategy", imputer["strategy"])),
            mode=str(config.get("mode", "replace")),
            pc_prefix=str(config.get("pc_prefix", "pc_")),
        )

        mean_map = scaler["mean"]
        std_map = scaler["std"]
        imputer_map = imputer["values"]
        components = np.asarray(pca["components"], dtype=float)
        explained = np.asarray(pca["explained_variance_ratio"], dtype=float)
        k_selected = int(pca["k_selected"])
        n_features = int(pca["n_features"])

        mean_series = pd.Series(mean_map, dtype=float).reindex(feature_columns)
        std_series = pd.Series(std_map, dtype=float).reindex(feature_columns)
        imputer_series = pd.Series(imputer_map, dtype=float).reindex(feature_columns)

        transformer.feature_columns_ = list(feature_columns)
        transformer.imputer_values_ = imputer_series
        transformer.scaler_mean_ = mean_series
        transformer.scaler_std_ = std_series.replace(0.0, 1.0).fillna(1.0)
        transformer.components_ = components
        transformer.explained_variance_ratio_ = explained
        transformer.k_selected_ = k_selected
        transformer.n_features_ = n_features
        return transformer

    @staticmethod
    def _prepare_input_frame(frame: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
        out = frame.loc[:, list(feature_cols)].copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        return out


def save_pca_json(path: str | Path, transformer: PCATransformer) -> None:
    """Write a fitted transformer payload to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(transformer.to_dict(), indent=2), encoding="utf-8")


def load_pca_json(path: str | Path) -> PCATransformer:
    """Load a fitted transformer from JSON."""
    in_path = Path(path)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    return PCATransformer.from_dict(payload)


__all__ = [
    "PCATransformer",
    "select_k_by_variance",
    "save_pca_json",
    "load_pca_json",
]
