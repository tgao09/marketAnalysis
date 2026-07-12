"""GBM adapter for shared leakage-safe walk-forward backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from common.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    FoldContext,
    LogReturnTarget,
    MarketData,
    YFinancePanelSource,
)

from gbm_return.configuration import FEATURE_SET_F0
from gbm_return.train import (
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    REGIME_SCORE_WINDOW,
    WINDOW_RET,
    build_features,
    prepare_lgbm_training_data,
    resolve_sector_etf,
    set_time_index,
    train_lgbm,
)


_PANEL_COLUMNS = ("close", "sector_close", "gld_close", "spy_close", "vix_close")


@dataclass(frozen=True)
class GBMAdapterConfig:
    lgbm_params: Mapping[str, Any]
    training_policy: Mapping[str, Any] | None = None
    drop_time_index: bool = True
    feature_set: str = FEATURE_SET_F0
    feature_set_file: str | None = None
    regime_score_enabled: bool = True
    regime_score_window: int = REGIME_SCORE_WINDOW
    regime_score_clip: float = REGIME_SCORE_CLIP
    regime_score_weights: Mapping[str, float] | None = None


class GBMReturnFoldModel:
    """Fresh fold-local LightGBM model; target and splits stay in engine."""

    def __init__(self, config: GBMAdapterConfig) -> None:
        self.config = config
        self.model = None
        self.feature_columns: list[str] = []
        self.train_start: pd.Timestamp | None = None

    def fit(self, train: pd.DataFrame, context: FoldContext) -> "GBMReturnFoldModel":
        self.train_start = context.train_start
        features = self._build_features(context.warmup)
        frame = features.reindex(train.index).copy()
        frame[context.target_column] = train[context.target_column]
        frame = frame.dropna()
        if frame.empty:
            raise ValueError(f"Fold {context.fold}: no GBM training rows remain after causal feature build.")
        self.feature_columns = [column for column in frame.columns if column != context.target_column]
        train_x, train_y, sample_weight, _ = prepare_lgbm_training_data(
            frame,
            feature_columns=self.feature_columns,
            target_column=context.target_column,
            training_policy=dict(self.config.training_policy or {}),
        )
        self.model = train_lgbm(train_x, train_y, dict(self.config.lgbm_params), sample_weight=sample_weight)
        return self

    def predict(self, test: pd.DataFrame, context: FoldContext) -> pd.DataFrame:
        if self.model is None or self.train_start is None:
            raise RuntimeError("fit() must complete before predict().")
        panel = pd.concat([context.warmup, test], axis=0)
        panel = panel.loc[~panel.index.duplicated(keep="last")]
        features = self._build_features(panel).reindex(test.index)
        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Fold {context.fold}: GBM prediction features missing: {', '.join(missing)}.")
        test_x = features.loc[:, self.feature_columns]
        if test_x.isna().any().any():
            missing_rows = int(test_x.isna().any(axis=1).sum())
            raise ValueError(f"Fold {context.fold}: {missing_rows} test rows lack causal GBM features.")
        result = test.copy()
        result[context.prediction_column] = self.model.predict(test_x)
        return result

    def _build_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in _PANEL_COLUMNS if column not in panel.columns]
        if missing:
            raise ValueError(f"GBM market panel is missing: {', '.join(missing)}.")
        features = build_features(
            panel["close"],
            panel["sector_close"],
            panel["gld_close"],
            panel["spy_close"],
            panel["vix_close"],
            drop_time_index=self.config.drop_time_index,
            feature_set=self.config.feature_set,
            feature_set_file=self.config.feature_set_file,
            regime_score_enabled=self.config.regime_score_enabled,
            regime_score_window=self.config.regime_score_window,
            regime_score_clip=self.config.regime_score_clip,
            regime_score_weights=dict(self.config.regime_score_weights or REGIME_SCORE_WEIGHTS),
        )
        if not self.config.drop_time_index:
            if self.train_start is None:
                raise RuntimeError("train_start is unavailable before fit().")
            features = set_time_index(features, self.train_start)
        return features


def load_gbm_market_data(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[MarketData, str, str | None]:
    """Fetch one reproducible GBM raw panel. ``end`` is yfinance-exclusive."""

    sector_etf, sector_name, sector_error = resolve_sector_etf(ticker)
    if sector_error:
        print(f"{ticker}: sector fallback to {sector_etf} ({sector_error})")
    source = YFinancePanelSource(
        primary_symbol=ticker,
        auxiliaries={
            "sector_close": sector_etf,
            "gld_close": "GLD",
            "spy_close": "SPY",
            "vix_close": "^VIX",
        },
        start=start,
        end=end,
        interval="1d",
    )
    return source.load(), sector_etf, sector_name


def run_gbm_backtest(
    market_data: MarketData,
    backtest_config: BacktestConfig,
    adapter_config: GBMAdapterConfig,
) -> BacktestResult:
    """Run all GBM OOS evaluation through common.backtesting."""

    target = backtest_config.target
    if target.horizon_bars != WINDOW_RET or target.price_column != "close":
        raise ValueError(f"GBM adapter requires {WINDOW_RET}-bar log returns from close.")
    engine = BacktestEngine(backtest_config)
    return engine.run(market_data, lambda _: GBMReturnFoldModel(adapter_config))


def gbm_feature_columns(panel: pd.DataFrame, config: GBMAdapterConfig) -> list[str]:
    """Resolve configured GBM columns for reporting without fitting a model."""

    probe = GBMReturnFoldModel(config)
    probe.train_start = panel.index.min()
    return list(probe._build_features(panel).columns)
