import argparse
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

from common import load_pca_json
from gp_return.train import (
    ARTIFACT_DIR_DEFAULT,
    REGIME_SCORE_WINDOW,
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    ReturnGPModel,
    build_features,
    compute_start_date,
    resolve_artifact_variant,
    resolve_device,
)
from gp_return.backtester import load_gp_market_data


def parse_args():
    parser = argparse.ArgumentParser(description="Predict next 5-day return with trained GP artifacts.")
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Use PCA artifact variant (ticker/pca). Default uses regular variant (ticker/regular).",
    )
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR_DEFAULT))
    parser.add_argument("--end", default=None, help="Inclusive historical end date (YYYY-MM-DD).")
    return parser.parse_args()


def prompt_tickers():
    raw = input("Tickers to predict (comma separated): ").strip()
    if not raw:
        return []
    tickers = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return tickers


def load_artifacts(artifact_dir: Path, device: torch.device, pca_enabled: bool):
    model_path = artifact_dir / "model_state.pt"
    scaler_path = artifact_dir / "scaler.json"
    pca_path = artifact_dir / "pca.json"
    config_path = artifact_dir / "config.json"

    model_blob = torch.load(model_path, map_location=device)
    config = json.loads(config_path.read_text())
    model_feature_cols = list(model_blob.get("feature_columns") or [])

    scaler = None
    pca_transformer = None
    if pca_enabled:
        pca_transformer = load_pca_json(pca_path)
    else:
        scaler = json.loads(scaler_path.read_text())

    train_x = model_blob["train_inputs"].to(device=device, dtype=torch.float32)
    train_y = model_blob["train_targets"].to(device=device, dtype=torch.float32)
    model_init_kwargs = dict(model_blob.get("model_init_kwargs") or {})
    return (
        model_blob,
        scaler,
        pca_transformer,
        config,
        model_feature_cols,
        train_x,
        train_y,
        model_init_kwargs,
    )


def resolve_regime_config(config):
    regime = (config or {}).get("regime_score") or (config or {}).get("regime_noise") or {}
    return {
        "enabled": regime.get("enabled", True),
        "score_window": regime.get("score_window", REGIME_SCORE_WINDOW),
        "score_clip": regime.get("score_clip", REGIME_SCORE_CLIP),
        "weights": regime.get("weights", REGIME_SCORE_WEIGHTS),
    }


def scale_with_saved_scaler(frame, feature_cols, scaler):
    mean = pd.Series(scaler["mean"], dtype=float)
    std = pd.Series(scaler["std"], dtype=float).replace(0.0, 1.0)
    mean = mean.reindex(feature_cols)
    std = std.reindex(feature_cols)

    scaled = (frame[feature_cols] - mean) / std
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    return scaled, mean, std


def rebuild_latest_features(
    config,
    history_cache,
    end_date: pd.Timestamp | None = None,
):
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

    market_data, _, _ = load_gp_market_data(config["ticker"], start_date, end_date + pd.Timedelta(days=1))
    panel = market_data.bars

    features = build_features(
        panel["close"],
        panel["volume"],
        panel["sector_close"],
        panel["gld_close"],
        panel["spy_close"],
        panel["vix_close"],
        regime_config,
    )
    return features


def get_latest_feature_row(features, feature_cols):
    usable = features[feature_cols].dropna()
    latest = usable.iloc[-1]
    return latest.name, latest[feature_cols]


def predict_next_window(
    artifact_dir: Path,
    device: torch.device,
    pca_enabled: bool,
    end_date: pd.Timestamp | None = None,
):
    (
        model_blob,
        scaler,
        pca_transformer,
        config,
        model_feature_cols,
        train_x,
        train_y,
        model_init_kwargs,
    ) = load_artifacts(
        artifact_dir, device, pca_enabled
    )

    history_cache = {}
    features = rebuild_latest_features(
        config,
        history_cache,
        end_date=end_date,
    )
    if pca_enabled:
        feature_cols = list(getattr(pca_transformer, "feature_columns_", None) or features.columns)
    else:
        feature_cols = model_feature_cols or list(features.columns)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ReturnGPModel(train_x, train_y, likelihood, **model_init_kwargs).to(device)
    model.load_state_dict(model_blob["model_state_dict"])
    likelihood.load_state_dict(model_blob["likelihood_state_dict"])

    asof_date, latest_features = get_latest_feature_row(features, feature_cols)

    if pca_enabled:
        latest_frame = pd.DataFrame([latest_features], index=[asof_date])
        latest_transformed = pca_transformer.transform(latest_frame)
        x = torch.tensor(latest_transformed.values, dtype=torch.float32, device=device)
    else:
        _, mean, std = scale_with_saved_scaler(features.loc[[asof_date]], feature_cols, scaler)
        latest_scaled = (latest_features - mean.reindex(feature_cols)) / std.reindex(feature_cols)
        latest_scaled = latest_scaled.replace([np.inf, -np.inf], np.nan)
        x = torch.tensor(latest_scaled.values, dtype=torch.float32, device=device).unsqueeze(0)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(x))
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
    args = parse_args()
    tickers = prompt_tickers()
    if not tickers:
        print("No ticker provided. Exiting.")
        return

    device = resolve_device()
    print(f"Using device: {device.type}")
    now_utc = datetime.now(UTC).isoformat()
    artifact_variant = resolve_artifact_variant(args.pca)

    for idx, ticker in enumerate(tickers):
        if idx:
            print("")
        artifact_dir = Path(args.artifact_dir) / ticker / artifact_variant
        end_date = pd.Timestamp(args.end).normalize() if args.end else None
        result = predict_next_window(artifact_dir, device, args.pca, end_date=end_date)

        asof = result["asof_date"]

        print(f"{ticker} 5-day forward log-return forecast")
        print(f"As of: {asof.date()} (last trading day with full features)")
        print(f"Generated: {now_utc}")
        print(f"Mean log return: {result['mean_log']:.6f}")
        print(f"Log-return std: {result['std_log']:.6f}")
        print(f"Mean simple return: {result['mean_simple']:.2%}")
        print(
            "95% interval (simple return): "
            f"[{result['lower_simple']:.2%}, {result['upper_simple']:.2%}]"
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
