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

from common import parse_window
from gp_vol.train import (
    NOISE_WINDOW,
    WINDOW_VOL,
    VolGPModel,
    build_features,
    extract_field,
    fetch_data,
    set_time_index,
)


ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "artifacts"


def load_artifacts(artifact_dir: Path):
    model_path = artifact_dir / "model_state.pt"
    scaler_path = artifact_dir / "scaler.json"
    config_path = artifact_dir / "config.json"

    model_blob = torch.load(model_path, map_location="cpu")
    scaler = json.loads(scaler_path.read_text())
    config = json.loads(config_path.read_text())

    train_x = model_blob["train_inputs"].to(dtype=torch.float32)
    train_y = model_blob["train_targets"].to(dtype=torch.float32)
    train_noise = model_blob["train_noise"].to(dtype=torch.float32)
    return model_blob, scaler, config, model_blob["feature_columns"], train_x, train_y, train_noise


def rebuild_latest_features(config, feature_cols):
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=config["data_years"])
    train_offset = parse_window(config["train_window"])
    test_offset = parse_window(config["test_window"])
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
    final_start = pd.Timestamp(config["final_train_window"]["start"])
    features = set_time_index(features, final_start)
    return features


def get_latest_feature_row(features, feature_cols):
    usable = features[feature_cols].dropna()
    latest_row = usable.iloc[-1]
    return latest_row.name, latest_row


def predict_next_week(artifact_dir: Path):
    model_blob, scaler, config, feature_cols, train_x, train_y, train_noise = load_artifacts(artifact_dir)

    features = rebuild_latest_features(config, feature_cols)

    kernel_config = config["kernel"]
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train_noise,
        learn_additional_noise=True,
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
