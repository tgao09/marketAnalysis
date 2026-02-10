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

from common import parse_window, walk_forward_splits
from gp_vol.train import (
    NOISE_WINDOW,
    WINDOW_VOL,
    VolGPModel,
    build_features,
    build_target,
    extract_field,
    fetch_data,
    normalize_features,
)


ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "artifacts"


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


def rebuild_training_data(config, feature_cols):
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=config.get("data_years", 2))
    train_offset = parse_window(config.get("train_window", "2y"))
    test_offset = parse_window(config.get("test_window", "1m"))
    buffer_days = NOISE_WINDOW + (2 * WINDOW_VOL) + 5

    min_start = end_date - train_offset
    min_start = min_start - test_offset
    min_start = min_start - pd.DateOffset(days=buffer_days)
    if min_start < start_date:
        start_date = min_start

    tickers = [
        config.get("ticker_target", "XLK"),
        config.get("ticker_gold", "GLD"),
        config.get("ticker_spy", "SPY"),
        config.get("ticker_vix", "^VIX"),
    ]

    data = fetch_data(tickers, start_date, end_date)
    price_xlk = extract_field(data, "Close", tickers[0])
    volume_xlk = extract_field(data, "Volume", tickers[0])
    price_gld = extract_field(data, "Close", tickers[1])
    price_spy = extract_field(data, "Close", tickers[2])
    price_vix = extract_field(data, "Close", tickers[3])

    features = build_features(price_xlk, volume_xlk, price_gld, price_spy, price_vix)
    target, noise = build_target(price_xlk)
    dataset = features.join([target, noise]).dropna()
    if dataset.empty:
        raise ValueError("No rows left after feature/target alignment.")

    splits = list(
        walk_forward_splits(
        dataset,
        train_window=config.get("train_window", "2y"),
        test_window=config.get("test_window", "1m"),
        step=config.get("step_window", "1m"),
        min_train_rows=60,
        )
    )
    if not splits:
        raise ValueError("No walk-forward splits available to rebuild training data.")

    last_split = splits[-1]
    train_df = last_split.train
    train_x_df, _, _ = normalize_features(train_df, train_df, feature_cols)
    train_x = torch.tensor(train_x_df.values, dtype=torch.float32)
    train_y = torch.tensor(train_df["target"].values, dtype=torch.float32)
    train_noise = (
        torch.tensor(train_df["noise"].values, dtype=torch.float32).clamp_min(1e-8)
    )

    return train_x, train_y, train_noise, features


def get_latest_feature_row(features, feature_cols):
    usable = features[feature_cols].dropna()
    if usable.empty:
        raise ValueError("No usable feature rows found for prediction.")
    latest_row = usable.iloc[-1]
    return latest_row.name, latest_row


def predict_next_week(artifact_dir: Path):
    model_blob, scaler, config, feature_cols = load_artifacts(artifact_dir)

    train_x, train_y, train_noise, features = rebuild_training_data(
        config, feature_cols
    )

    kernel_config = config.get("kernel", {})
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train_noise,
        learn_additional_noise=False,
    )
    model = VolGPModel(train_x, train_y, likelihood, kernel_config)
    model.load_state_dict(model_blob["model_state_dict"])
    likelihood.load_state_dict(model_blob["likelihood_state_dict"])

    asof_date, latest_features = get_latest_feature_row(features, feature_cols)

    mean = pd.Series(scaler["mean"])
    std = pd.Series(scaler["std"]).replace(0.0, 1.0)
    latest_scaled = (latest_features - mean) / std

    x = torch.tensor(latest_scaled.values, dtype=torch.float32).unsqueeze(0)
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        obs_noise = torch.full(
            (x.size(0),),
            float(train_noise.median().item()),
            dtype=torch.float32,
        )
        preds = likelihood(model(x), noise=obs_noise)
        mean_log = preds.mean.item()
        std_log = preds.variance.sqrt().item()

    mean_vol = math.exp(mean_log)
    lower_vol = math.exp(mean_log - (1.96 * std_log))
    upper_vol = math.exp(mean_log + (1.96 * std_log))

    return {
        "asof_date": asof_date,
        "mean_vol": mean_vol,
        "lower_vol": lower_vol,
        "upper_vol": upper_vol,
    }


def main():
    artifact_dir = ARTIFACT_DIR_DEFAULT
    result = predict_next_week(artifact_dir)

    asof = result["asof_date"]
    now_utc = datetime.now(UTC).isoformat()

    print("XLK 5-day forward realized volatility forecast")
    print(f"As of: {asof.date()} (last trading day with full features)")
    print(f"Generated: {now_utc}")
    print(f"Annualized vol (mean): {result['mean_vol']:.4f}")
    print(f"95% interval: [{result['lower_vol']:.4f}, {result['upper_vol']:.4f}]")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
