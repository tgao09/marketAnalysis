import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from gp_return.train import resolve_device
from hmm_regime.train import build_market_dataset
from hmm_gp_return.train import (
    ARTIFACT_DIR_DEFAULT,
    CONFIG_FILENAME,
    WINDOW_RET,
    add_meta_features,
    build_strategy_dataset,
    compute_gp_prediction_for_date,
    compute_hmm_state_for_date,
    compute_strategy_start,
    latest_strategy_features,
    load_model_blob,
    predict_meta_residual,
    resolve_artifact_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next 5-day return with HMM + GP ensemble artifacts.")
    parser.add_argument("--ticker", default=None, help="Ticker symbol.")
    parser.add_argument("--date", default=None, help="As-of date YYYY-MM-DD (default: today).")
    parser.add_argument("--output-dir", default=ARTIFACT_DIR_DEFAULT)
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path for the latest row.")
    return parser.parse_args()


def prompt_ticker() -> str | None:
    raw = input("Ticker to predict: ").strip()
    if not raw:
        return None
    return raw.upper()


def load_config(artifact_dir: Path) -> dict:
    return json.loads((artifact_dir / CONFIG_FILENAME).read_text())


def build_latest_base_row(
    ticker: str,
    asof_date: pd.Timestamp,
    base_train_window: str,
    meta_train_window: str,
    device,
) -> dict[str, object]:
    start_date = compute_strategy_start(
        end_date=asof_date,
        base_train_window=base_train_window,
        meta_train_window=meta_train_window,
    )
    history_cache: dict[str, pd.DataFrame] = {}
    strategy_dataset = build_strategy_dataset(ticker, start_date, asof_date, history_cache=history_cache)
    latest_strategy = latest_strategy_features(ticker, start_date, asof_date, history_cache=history_cache)
    market_dataset = build_market_dataset(start_date, asof_date)

    dataset = strategy_dataset["dataset"]
    usable_feature_frame = latest_strategy["features"].dropna()
    if usable_feature_frame.empty:
        raise ValueError(f"{ticker}: no usable feature rows available for prediction.")

    test_date = usable_feature_frame.index.max()
    test_features = usable_feature_frame.loc[test_date]
    latest_close = latest_strategy["close"].reindex(usable_feature_frame.index).loc[test_date]
    if pd.isna(latest_close):
        raise ValueError(f"{ticker}: missing close price for {test_date.date()}.")

    gp_row = compute_gp_prediction_for_date(
        dataset=dataset,
        test_date=test_date,
        base_train_window=base_train_window,
        device=device,
        all_feature_index=usable_feature_frame.index,
        test_features=test_features,
    )
    hmm_row = compute_hmm_state_for_date(
        market_dataset=market_dataset,
        test_date=test_date,
    )

    row: dict[str, object] = {
        "symbol": ticker.upper(),
        "date": test_date,
        "entry_close": float(latest_close),
    }
    row.update(gp_row)
    row.update(hmm_row)
    return row


def predict_next_window(
    artifact_dir: Path,
    ticker: str,
    asof_date: pd.Timestamp,
    device,
) -> dict[str, object]:
    config = load_config(artifact_dir)
    model_blob = load_model_blob(artifact_dir)

    base_row = build_latest_base_row(
        ticker=ticker,
        asof_date=asof_date,
        base_train_window=config["base_train_window"],
        meta_train_window=config["meta_train_window"],
        device=device,
    )
    base_frame = pd.DataFrame([base_row])
    meta_residual_pred = float(
        predict_meta_residual(
            scaler=model_blob["scaler"],
            model=model_blob["model"],
            rows=add_meta_features(base_frame),
        )[0]
    )
    ensemble_pred_mean_log = float(base_row["gp_pred_mean_log"]) + meta_residual_pred
    ensemble_pred_mean_simple = math.exp(ensemble_pred_mean_log) - 1.0
    action = "long" if ensemble_pred_mean_log >= 0.0 else "short"

    result = dict(base_row)
    result.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "artifact_dir": str(artifact_dir),
            "meta_residual_pred": meta_residual_pred,
            "ensemble_pred_mean_log": ensemble_pred_mean_log,
            "ensemble_pred_mean_simple": ensemble_pred_mean_simple,
            "action": action,
            "horizon_trading_days": WINDOW_RET,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    ticker = args.ticker or prompt_ticker()
    if not ticker:
        print("No ticker provided. Exiting.")
        return

    ticker = ticker.upper()
    asof_date = pd.Timestamp(args.date).normalize() if args.date else pd.Timestamp.today().normalize()
    artifact_dir = resolve_artifact_dir(args.output_dir, ticker)
    device = resolve_device()

    result = predict_next_window(
        artifact_dir=artifact_dir,
        ticker=ticker,
        asof_date=asof_date,
        device=device,
    )

    output_path = Path(args.output_csv) if args.output_csv else artifact_dir / "latest_prediction.csv"
    pd.DataFrame([result]).to_csv(output_path, index=False)

    print(f"Using device: {device.type}")
    print(f"{ticker} {WINDOW_RET}-day forward ensemble log-return forecast")
    print(f"As of: {pd.Timestamp(result['date']).date()} (last trading day with full features)")
    print(f"Generated: {result['generated_at']}")
    print(f"Entry close: {float(result['entry_close']):.2f}")
    print(f"GP mean log return: {float(result['gp_pred_mean_log']):.6f}")
    print(f"GP log-return std: {float(result['gp_pred_std_log']):.6f}")
    print(f"Meta residual adjustment: {float(result['meta_residual_pred']):.6f}")
    print(f"Ensemble mean log return: {float(result['ensemble_pred_mean_log']):.6f}")
    print(f"Ensemble mean simple return: {float(result['ensemble_pred_mean_simple']):.2%}")
    print(f"Action: {result['action']}")
    print(
        "HMM state: "
        f"{result['state_label']} "
        f"(shift={float(result['shift_prob']):.4f}, "
        f"p0={float(result['p_state_0']):.4f}, "
        f"p1={float(result['p_state_1']):.4f}, "
        f"p2={float(result['p_state_2']):.4f}, "
        f"p3={float(result['p_state_3']):.4f})"
    )
    print(f"Latest prediction row saved to: {output_path}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
