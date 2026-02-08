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
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
    DATA_YEARS,
    REGIME_SCORE_WINDOW,
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    TICKER_GOLD,
    TICKER_SPY,
    TICKER_VIX,
    WINDOW_RET,
    ReturnGPModel,
    build_features,
    build_target,
    compute_regime_score,
    compute_start_date,
    extract_field,
    fetch_history_cached,
    latest_train_window,
    resolve_artifact_variant,
    resolve_sector_etf,
    resolve_device,
    set_time_index,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict next 5-day return with trained GP artifacts.")
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Use PCA artifact variant (ticker/pca). Default uses regular variant (ticker/regular).",
    )
    return parser.parse_args()


def prompt_tickers():
    raw = input("Tickers to predict (comma separated): ").strip()
    if not raw:
        return None
    tickers = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not tickers:
        return None
    return tickers


def load_artifacts(artifact_dir: Path, device: torch.device, pca_enabled: bool):
    model_path = artifact_dir / "model_state.pt"
    scaler_path = artifact_dir / "scaler.json"
    pca_path = artifact_dir / "pca.json"
    config_path = artifact_dir / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config artifact: {config_path}")

    model_blob = torch.load(model_path, map_location=device)
    config = json.loads(config_path.read_text())

    feature_cols = model_blob.get("feature_columns")
    if not feature_cols:
        raise ValueError("Artifact is missing feature columns.")
    raw_feature_cols = model_blob.get("raw_feature_columns") or feature_cols

    model_pca_enabled = bool((config.get("pca") or {}).get("enabled", False))
    expected_variant = resolve_artifact_variant(pca_enabled)
    artifact_variant = config.get("artifact_variant")
    if artifact_variant != expected_variant:
        raise ValueError(
            f"Artifact variant mismatch in {artifact_dir}. "
            f"Expected '{expected_variant}' but found '{artifact_variant}'."
        )
    if model_pca_enabled != pca_enabled:
        raise ValueError(
            f"PCA mode mismatch for {artifact_dir}. "
            f"CLI --pca={pca_enabled} but artifact config pca.enabled={model_pca_enabled}."
        )

    scaler = None
    pca_transformer = None
    if pca_enabled:
        if not pca_path.exists():
            raise FileNotFoundError(f"Missing PCA artifact: {pca_path}")
        pca_transformer = load_pca_json(pca_path)
    else:
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler artifact: {scaler_path}")
        scaler = json.loads(scaler_path.read_text())

    return model_blob, scaler, pca_transformer, config, feature_cols, raw_feature_cols


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

    missing = [col for col in feature_cols if pd.isna(mean[col]) or pd.isna(std[col])]
    if missing:
        raise ValueError(f"Scaler is missing feature stats for: {', '.join(missing)}")

    scaled = (frame[feature_cols] - mean) / std
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    if scaled.isna().any().any():
        bad_cols = [col for col in feature_cols if scaled[col].isna().any()]
        raise ValueError(f"Scaled features contain NaNs for columns: {', '.join(bad_cols)}")

    return scaled, mean, std


def rebuild_training_data(
    config,
    model_feature_cols,
    raw_feature_cols,
    scaler,
    pca_transformer,
    pca_enabled: bool,
    history_cache,
    device: torch.device,
):
    regime_config = resolve_regime_config(config)
    end_date = pd.Timestamp.today().normalize()
    start_date = compute_start_date(
        end_date,
        config.get("data_years", DATA_YEARS),
        config.get("train_window", DEFAULT_TRAIN_WINDOW),
        config.get("test_window", DEFAULT_TEST_WINDOW),
        regime_config["score_window"],
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
    target = build_target(price_stock)

    # Keep regime inputs causal to avoid leaking future values.
    price_spy_regime = price_spy.reindex(price_stock.index).ffill()
    price_vix_regime = price_vix.reindex(price_stock.index).ffill()

    if regime_config.get("enabled", True):
        regime_score = compute_regime_score(
            price_vix_regime,
            price_spy_regime,
            regime_config["score_window"],
            regime_config["score_clip"],
            regime_config["weights"],
        )
    else:
        regime_score = pd.Series(0.0, index=price_stock.index, name="regime_score")

    features["regime_score"] = regime_score

    dataset = features.join([target])
    dataset = dataset.dropna()
    if dataset.empty:
        raise ValueError("No rows left after feature/target alignment.")

    final_train_raw = latest_train_window(
        dataset,
        train_window=config.get("train_window", DEFAULT_TRAIN_WINDOW),
        min_train_rows=60,
    )
    fold_start = final_train_raw.index.min()
    train_df = set_time_index(final_train_raw.copy(), fold_start)
    features = set_time_index(features, fold_start)

    if pca_enabled:
        train_x_df = pca_transformer.transform(train_df)
        if train_x_df.columns.tolist() != list(model_feature_cols):
            raise ValueError(
                "PCA transformed training columns do not match model feature columns. "
                f"model={model_feature_cols}, transformed={train_x_df.columns.tolist()}"
            )
        mean = None
        std = None
    else:
        train_x_df, mean, std = scale_with_saved_scaler(train_df, model_feature_cols, scaler)
    train_x = torch.tensor(train_x_df.values, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_df["target"].values, dtype=torch.float32, device=device)

    return train_x, train_y, features, mean, std


def get_latest_feature_row(features, feature_cols):
    usable = features[feature_cols].dropna()
    if usable.empty:
        raise ValueError("No usable feature rows found for prediction.")
    latest = usable.iloc[-1]
    return latest.name, latest[feature_cols]


def predict_next_window(artifact_dir: Path, device: torch.device, pca_enabled: bool):
    model_blob, scaler, pca_transformer, config, model_feature_cols, raw_feature_cols = load_artifacts(
        artifact_dir, device, pca_enabled
    )

    history_cache = {}
    train_x, train_y, features, mean, std = rebuild_training_data(
        config,
        model_feature_cols,
        raw_feature_cols,
        scaler,
        pca_transformer,
        pca_enabled,
        history_cache,
        device,
    )

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ReturnGPModel(train_x, train_y, likelihood).to(device)
    model.load_state_dict(model_blob["model_state_dict"])
    likelihood.load_state_dict(model_blob["likelihood_state_dict"])

    asof_date, latest_features = get_latest_feature_row(features, raw_feature_cols)

    if pca_enabled:
        latest_frame = pd.DataFrame([latest_features], index=[asof_date])
        latest_transformed = pca_transformer.transform(latest_frame)
        if latest_transformed.columns.tolist() != list(model_feature_cols):
            raise ValueError(
                "PCA transformed inference columns do not match model feature columns. "
                f"model={model_feature_cols}, transformed={latest_transformed.columns.tolist()}"
            )
        x = torch.tensor(latest_transformed.values, dtype=torch.float32, device=device)
    else:
        latest_scaled = (latest_features - mean.reindex(model_feature_cols)) / std.reindex(model_feature_cols)
        latest_scaled = latest_scaled.replace([np.inf, -np.inf], np.nan)
        if latest_scaled.isna().any():
            bad_cols = [col for col in model_feature_cols if pd.isna(latest_scaled[col])]
            raise ValueError(f"Latest feature row produced NaNs after scaling for: {', '.join(bad_cols)}")
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
        artifact_dir = ARTIFACT_DIR_DEFAULT / ticker / artifact_variant
        try:
            result = predict_next_window(artifact_dir, device, args.pca)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{ticker}: {exc}")
            continue

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
