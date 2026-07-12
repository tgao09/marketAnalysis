"""GP return adapter for shared leakage-safe walk-forward backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gpytorch
import numpy as np
import pandas as pd
import torch

from common.backtesting import BacktestConfig, BacktestEngine, BacktestResult, FoldContext, MarketData, YFinancePanelSource
from common.pca_utils import PCATransformer
from gp_return.train import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_LINEAR,
    DEFAULT_MATERN_NU,
    DEFAULT_RQ,
    DEFAULT_TRAIN_ITERS,
    DEFAULT_WEIGHT_DECAY,
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    REGIME_SCORE_WINDOW,
    WINDOW_RET,
    build_features,
    build_pca_transformer,
    normalize_features,
    resolve_sector_etf,
    train_gp,
)


@dataclass(frozen=True)
class GPAdapterConfig:
    device: torch.device
    train_iters: int = DEFAULT_TRAIN_ITERS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    matern_nu: float = DEFAULT_MATERN_NU
    use_rq: bool = DEFAULT_RQ
    use_linear: bool = DEFAULT_LINEAR
    pca_enabled: bool = False
    regime_config: Mapping[str, Any] | None = None


class GPReturnFoldModel:
    def __init__(self, config: GPAdapterConfig) -> None:
        self.config = config
        self.model = None
        self.likelihood = None
        self.feature_columns: list[str] = []
        self.scaler: Mapping[str, Any] | None = None
        self.pca: PCATransformer | None = None

    def fit(self, train: pd.DataFrame, context: FoldContext) -> "GPReturnFoldModel":
        features = self._features(context.warmup)
        frame = features.reindex(train.index).copy()
        frame[context.target_column] = train[context.target_column]
        frame = frame.dropna()
        if len(frame) < 2:
            raise ValueError(f"Fold {context.fold}: insufficient causal GP training rows.")
        base_columns = [column for column in frame.columns if column != context.target_column]
        if self.config.pca_enabled:
            self.pca = build_pca_transformer()
            train_x_df = self.pca.fit_transform(frame, base_columns)
        else:
            train_x_df, _, self.scaler = normalize_features(
                frame,
                frame,
                feature_cols=base_columns,
            )
        self.feature_columns = list(train_x_df.columns)
        train_x = torch.tensor(train_x_df.values, dtype=torch.float32, device=self.config.device)
        train_y = torch.tensor(frame[context.target_column].values, dtype=torch.float32, device=self.config.device)
        self.model, self.likelihood = train_gp(
            train_x,
            train_y,
            train_iters=self.config.train_iters,
            device=self.config.device,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            matern_nu=self.config.matern_nu,
            use_rq=self.config.use_rq,
            use_linear=self.config.use_linear,
        )
        return self

    def predict(self, test: pd.DataFrame, context: FoldContext) -> pd.DataFrame:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("fit() must complete before predict().")
        panel = pd.concat([context.warmup, test], axis=0)
        panel = panel.loc[~panel.index.duplicated(keep="last")]
        features = self._features(panel).reindex(test.index)
        if features.isna().any().any():
            raise ValueError(f"Fold {context.fold}: test rows lack causal GP features.")
        if self.pca is not None:
            test_x_df = self.pca.transform(features)
        else:
            mean = pd.Series(self.scaler["mean"])
            std = pd.Series(self.scaler["std"]).replace(0.0, 1.0)
            test_x_df = (features[mean.index] - mean) / std
        test_x = torch.tensor(test_x_df[self.feature_columns].values, dtype=torch.float32, device=self.config.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(test_x))
        result = test.copy()
        result[context.prediction_column] = posterior.mean.detach().cpu().numpy()
        result["pred_std_log"] = posterior.stddev.detach().cpu().numpy()
        return result

    def _features(self, panel: pd.DataFrame) -> pd.DataFrame:
        required = ["close", "volume", "sector_close", "gld_close", "spy_close", "vix_close"]
        missing = [column for column in required if column not in panel.columns]
        if missing:
            raise ValueError(f"GP market panel is missing: {', '.join(missing)}.")
        regime = dict(self.config.regime_config or {"enabled": False, "score_window": REGIME_SCORE_WINDOW, "score_clip": REGIME_SCORE_CLIP, "weights": REGIME_SCORE_WEIGHTS})
        return build_features(
            panel["close"], panel["volume"], panel["sector_close"], panel["gld_close"], panel["spy_close"], panel["vix_close"], regime
        )


def load_gp_market_data(ticker: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> tuple[MarketData, str, str | None]:
    sector_etf, sector_name, sector_error = resolve_sector_etf(ticker)
    if sector_error:
        print(f"{ticker}: sector fallback to {sector_etf} ({sector_error})")
    source = YFinancePanelSource(
        primary_symbol=ticker,
        auxiliaries={"sector_close": sector_etf, "gld_close": "GLD", "spy_close": "SPY", "vix_close": "^VIX"},
        start=start,
        end=end,
        interval="1d",
    )
    return source.load(), sector_etf, sector_name


def run_gp_backtest(market_data: MarketData, backtest_config: BacktestConfig, adapter_config: GPAdapterConfig) -> BacktestResult:
    target = backtest_config.target
    if target.horizon_bars != WINDOW_RET or target.price_column != "close":
        raise ValueError(f"GP adapter requires {WINDOW_RET}-bar log returns from close.")
    return BacktestEngine(backtest_config).run(market_data, lambda _: GPReturnFoldModel(adapter_config))
