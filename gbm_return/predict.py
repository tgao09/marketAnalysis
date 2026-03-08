import json
import math
import pickle
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from gbm_return.train import (
    ARTIFACT_DIR_DEFAULT,
    DATA_YEARS,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    REGIME_SCORE_WINDOW,
    TICKER_GOLD,
    TICKER_SPY,
    TICKER_VIX,
    build_features,
    build_target,
    compute_regime_score,
    compute_start_date,
    extract_field,
    fetch_history_cached,
    resolve_direction_mode,
    resolve_sector_etf,
    resolve_artifact_variant,
    set_time_index,
)


def prompt_tickers():
    raw = input("Tickers to predict (comma/space separated): ").strip()
    if not raw:
        return []
    tokens = [token.strip().upper() for token in re.split(r"[,\s]+", raw) if token.strip()]
    seen = set()
    tickers = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            tickers.append(token)
    return tickers


def load_artifacts(artifact_dir: Path):
    model_path = artifact_dir / "model_state.pt"
    config_path = artifact_dir / "config.json"

    with model_path.open("rb") as fh:
        model_blob = pickle.load(fh)
    config = json.loads(config_path.read_text())

    feature_cols = model_blob["feature_columns"]
    raw_feature_cols = model_blob.get("raw_feature_columns") or feature_cols
    return model_blob["model_str"], config, feature_cols, raw_feature_cols


def resolve_regime_config(config):
    regime = (config or {}).get("regime_score") or {}
    return {
        "enabled": regime.get("enabled", True),
        "score_window": regime.get("score_window", REGIME_SCORE_WINDOW),
        "score_clip": regime.get("score_clip", REGIME_SCORE_CLIP),
        "weights": regime.get("weights", REGIME_SCORE_WEIGHTS),
    }


def rebuild_features(config):
    regime_config = resolve_regime_config(config)
    end_date = pd.Timestamp.today().normalize()
    start_date = compute_start_date(
        end_date,
        config["data_years"],
        config["train_window"],
        config["test_window"],
        regime_config["score_window"],
    )

    ticker = config["ticker"]
    sector_etf = config.get("sector_etf") or resolve_sector_etf(ticker)[0]

    history_cache = {}
    stock_history = fetch_history_cached(ticker, start_date, end_date, history_cache)
    sector_history = fetch_history_cached(sector_etf, start_date, end_date, history_cache)
    gld_history = fetch_history_cached(TICKER_GOLD, start_date, end_date, history_cache)
    spy_history = fetch_history_cached(TICKER_SPY, start_date, end_date, history_cache)
    vix_history = fetch_history_cached(TICKER_VIX, start_date, end_date, history_cache)

    price_stock = extract_field(stock_history, "Close", ticker)
    volume_stock = extract_field(stock_history, "Volume", ticker)
    price_sector = extract_field(sector_history, "Close", sector_etf)
    price_gld = extract_field(gld_history, "Close", TICKER_GOLD)
    price_spy = extract_field(spy_history, "Close", TICKER_SPY)
    price_vix = extract_field(vix_history, "Close", TICKER_VIX)

    features = build_features(price_stock, volume_stock, price_sector, price_gld, price_spy, price_vix)
    target = build_target(price_stock)

    if regime_config.get("enabled", True):
        regime_score = compute_regime_score(
            price_vix.reindex(price_stock.index).ffill(),
            price_spy.reindex(price_stock.index).ffill(),
            regime_config["score_window"],
            regime_config["score_clip"],
            regime_config["weights"],
        )
    else:
        regime_score = pd.Series(0.0, index=price_stock.index, name="regime_score")
    features["regime_score"] = regime_score

    dataset = features.join([target]).dropna()
    final_start = pd.Timestamp(config["final_train_window"]["start"])
    if pd.isna(final_start):
        final_start = dataset.index.min()
    features = set_time_index(features, final_start)
    return features


def get_latest_feature_row(features, feature_cols):
    usable = features[feature_cols].dropna()
    latest = usable.iloc[-1]
    return latest.name, latest[feature_cols]


def predict_next_window(artifact_dir: Path):
    model_str, config, model_feature_cols, raw_feature_cols = load_artifacts(artifact_dir)
    features = rebuild_features(config)
    asof_date, latest_features = get_latest_feature_row(features, raw_feature_cols)

    booster = lgb.Booster(model_str=model_str)
    x = np.asarray([latest_features.values], dtype=float)
    mean_log = float(booster.predict(x)[0])
    mean_simple = math.exp(mean_log) - 1.0
    direction_mode = resolve_direction_mode(config.get("direction_mode"))
    if direction_mode == "long_only":
        action = "long"
    elif direction_mode == "short_only":
        action = "short"
    else:
        action = "long" if mean_log >= 0.0 else "short"
    return {
        "asof_date": asof_date,
        "mean_log": mean_log,
        "mean_simple": mean_simple,
        "feature_count": len(model_feature_cols),
        "action": action,
    }


def main():
    tickers = prompt_tickers()
    if not tickers:
        print("No ticker provided. Exiting.")
        return

    for idx, ticker in enumerate(tickers):
        if idx:
            print("")
        artifact_dir = ARTIFACT_DIR_DEFAULT / ticker / resolve_artifact_variant()
        result = predict_next_window(artifact_dir)

        asof = result["asof_date"]
        print(f"{ticker} 5-day forward log-return forecast")
        print(f"As of: {asof.date()} (last trading day with full features)")
        print(f"Mean log return: {result['mean_log']:.6f}")
        print(f"Mean simple return: {result['mean_simple']:.2%}")
        print(f"Action: {result['action']}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
