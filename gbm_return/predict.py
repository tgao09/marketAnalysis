import argparse
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
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    REGIME_SCORE_WINDOW,
    build_features,
    compute_start_date,
    resolve_direction_mode,
    resolve_artifact_variant,
    set_time_index,
)
from gbm_return.backtester import load_gbm_market_data


def parse_args():
    parser = argparse.ArgumentParser(description="Predict GBM return model.")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR_DEFAULT))
    parser.add_argument("--end", default=None, help="Inclusive historical end date (YYYY-MM-DD).")
    return parser.parse_args()


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
    return model_blob["model_str"], config, feature_cols


def resolve_regime_config(config):
    regime = (config or {}).get("regime_score") or {}
    return {
        "enabled": regime.get("enabled", True),
        "score_window": regime.get("score_window", REGIME_SCORE_WINDOW),
        "score_clip": regime.get("score_clip", REGIME_SCORE_CLIP),
        "weights": regime.get("weights", REGIME_SCORE_WEIGHTS),
    }


def rebuild_features(config, end_date: pd.Timestamp | None = None):
    regime_config = resolve_regime_config(config)
    end_date = (
        pd.Timestamp.today().normalize()
        if end_date is None and not config.get("end_date")
        else pd.Timestamp(end_date or config["end_date"]).normalize()
    )
    start_date = compute_start_date(
        end_date,
        config["data_years"],
        config["train_window"],
        config["test_window"],
        regime_config["score_window"],
    )

    market_data, _, _ = load_gbm_market_data(
        config["ticker"],
        start_date,
        end_date + pd.Timedelta(days=1),
    )
    panel = market_data.bars

    features = build_features(
        panel["close"],
        panel["sector_close"],
        panel["gld_close"],
        panel["spy_close"],
        panel["vix_close"],
        drop_time_index=bool(config.get("drop_time_index", True)),
        feature_set=str(config.get("feature_set", "f0")),
        feature_set_file=config.get("feature_set_file"),
        regime_score_enabled=bool(regime_config.get("enabled", True)),
        regime_score_window=int(regime_config["score_window"]),
        regime_score_clip=float(regime_config["score_clip"]),
        regime_score_weights=regime_config.get("weights"),
    )
    if not bool(config.get("drop_time_index", True)):
        final_start_raw = (config.get("final_train_window") or {}).get("start")
        final_start = pd.Timestamp(final_start_raw) if final_start_raw else pd.NaT
        if pd.isna(final_start):
            final_start = features.index.min()
        features = set_time_index(features, final_start)
    return features


def get_latest_feature_row(features, feature_cols):
    usable = features[feature_cols].dropna()
    latest = usable.iloc[-1]
    return latest.name, latest[feature_cols]


def predict_next_window(artifact_dir: Path, end_date: pd.Timestamp | None = None):
    model_str, config, model_feature_cols = load_artifacts(artifact_dir)
    features = rebuild_features(config, end_date=end_date)
    asof_date, latest_features = get_latest_feature_row(features, model_feature_cols)

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
    args = parse_args()
    tickers = prompt_tickers()
    if not tickers:
        print("No ticker provided. Exiting.")
        return

    for idx, ticker in enumerate(tickers):
        if idx:
            print("")
        artifact_dir = Path(args.artifact_dir) / ticker / resolve_artifact_variant()
        end_date = pd.Timestamp(args.end).normalize() if args.end else None
        result = predict_next_window(artifact_dir, end_date=end_date)

        asof = result["asof_date"]
        print(f"{ticker} 5-day forward log-return forecast")
        print(f"As of: {asof.date()} (last trading day with full features)")
        print(f"Mean log return: {result['mean_log']:.6f}")
        print(f"Mean simple return: {result['mean_simple']:.2%}")
        print(f"Action: {result['action']}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
