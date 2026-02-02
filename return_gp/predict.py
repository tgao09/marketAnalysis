import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
import numpy as np

import pandas as pd
import torch
import gpytorch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import walk_forward_splits
from return_gp.train import (
    ARTIFACT_DIR_DEFAULT,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
    DEFAULT_STEP_WINDOW,
    DATA_YEARS,
    TICKER_GOLD,
    TICKER_SPY,
    TICKER_VIX,
    ReturnGPModel,
    build_features,
    build_target,
    compute_start_date,
    extract_field,
    fetch_history_cached,
    normalize_features,
    resolve_sector_etf,
    set_time_index,
)


def prompt_ticker():
    raw = input("Ticker to predict: ").strip()
    if not raw:
        return None
    return raw.upper()


def load_artifacts(artifact_dir: Path):
    model_path = artifact_dir / "model_state.pt"
    scaler_path = artifact_dir / "scaler.json"
    config_path = artifact_dir / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler artifact: {scaler_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config artifact: {config_path}")

    model_blob = torch.load(model_path, map_location="cpu")
    scaler = json.loads(scaler_path.read_text())
    config = json.loads(config_path.read_text())

    feature_cols = model_blob.get("feature_columns")
    if not feature_cols:
        raise ValueError("Artifact is missing feature columns.")

    return model_blob, scaler, config, feature_cols


def rebuild_training_data(config, feature_cols, history_cache):
    end_date = pd.Timestamp.today().normalize()
    start_date = compute_start_date(
        end_date,
        config.get("data_years", DATA_YEARS),
        config.get("train_window", DEFAULT_TRAIN_WINDOW),
        config.get("test_window", DEFAULT_TEST_WINDOW),
    )

    ticker = config.get("ticker")
    sector_etf = config.get("sector_etf")
    if not sector_etf:
        sector_etf, _, _ = resolve_sector_etf(ticker)

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
    target, noise = build_target(price_stock)

    dataset = features.join([target, noise]).dropna()
    if dataset.empty:
        raise ValueError("No rows left after feature/target alignment.")

    splits = list(
        walk_forward_splits(
            dataset,
            train_window=config.get("train_window", DEFAULT_TRAIN_WINDOW),
            test_window=config.get("test_window", DEFAULT_TEST_WINDOW),
            step=config.get("step_window", DEFAULT_STEP_WINDOW),
            min_train_rows=60,
        )
    )
    if not splits:
        raise ValueError("No walk-forward splits available to rebuild training data.")

    last_split = splits[-1]
    fold_start = last_split.train_start
    train_df = set_time_index(last_split.train.copy(), fold_start)
    features = set_time_index(features, fold_start)

    train_x_df, _, _ = normalize_features(train_df, train_df, feature_cols)
    train_x = torch.tensor(train_x_df.values, dtype=torch.float32)
    train_y = torch.tensor(train_df["target"].values, dtype=torch.float32)
    train_noise = torch.tensor(train_df["noise"].values, dtype=torch.float32).clamp_min(1e-8)

    return train_x, train_y, train_noise, features, noise


def get_latest_feature_row(features, noise, feature_cols):
    usable = features[feature_cols].join(noise).dropna()
    if usable.empty:
        raise ValueError("No usable feature rows found for prediction.")
    latest = usable.iloc[-1]
    return latest.name, latest[feature_cols], latest["noise"]


def predict_next_window(artifact_dir: Path):
    model_blob, scaler, config, feature_cols = load_artifacts(artifact_dir)

    history_cache = {}
    train_x, train_y, train_noise, features, noise = rebuild_training_data(
        config,
        feature_cols,
        history_cache,
    )

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train_noise,
        learn_additional_noise=True,
    )
    model = ReturnGPModel(train_x, train_y, likelihood)
    model.load_state_dict(model_blob["model_state_dict"])
    likelihood.load_state_dict(model_blob["likelihood_state_dict"])

    asof_date, latest_features, latest_noise = get_latest_feature_row(
        features,
        noise,
        feature_cols,
    )

    mean = pd.Series(scaler["mean"])
    std = pd.Series(scaler["std"]).replace(0.0, 1.0)
    latest_scaled = (latest_features - mean) / std

    x = torch.tensor(latest_scaled.values, dtype=torch.float32).unsqueeze(0)
    test_noise = torch.tensor([latest_noise], dtype=torch.float32).clamp_min(1e-8)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(x), noise=test_noise)
        mean_log = preds.mean.item()
        std_log = preds.variance.sqrt().item()

    mean_simple = math.exp(mean_log) - 1.0
    lower_simple = math.exp(mean_log - (1.96 * std_log)) - 1.0
    upper_simple = math.exp(mean_log + (1.96 * std_log)) - 1.0

    return {
        "asof_date": asof_date,
        "mean_log": mean_log,
        "std_log": std_log,
        "mean_simple": mean_simple,
        "lower_simple": lower_simple,
        "upper_simple": upper_simple,
    }


def main():
    ticker = prompt_ticker()
    if not ticker:
        print("No ticker provided. Exiting.")
        return

    artifact_dir = ARTIFACT_DIR_DEFAULT / ticker
    result = predict_next_window(artifact_dir)

    asof = result["asof_date"]
    now_utc = datetime.now(UTC).isoformat()

    print(f"{ticker} 5-day forward log-return forecast")
    print(f"As of: {asof.date()} (last trading day with full features)")
    print(f"Generated: {now_utc}")
    print(f"Mean log return: {result['mean_log']:.6f}")
    print(f"Log-return std: {result['std_log']:.6f}")
    print(
        "95% interval (simple return): "
        f"[{result['lower_simple']:.2%}, {result['upper_simple']:.2%}]"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
