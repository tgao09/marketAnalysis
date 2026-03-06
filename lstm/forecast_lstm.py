import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import load_metadata, load_ticker_universe, sanitize_features
from model import LSTMConfig, StockLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate next-week forecasts with the trained LSTM.")
    parser.add_argument("--data-file", default="dataset/stock_dataset_with_lags.csv",
                        help="Historical feature store (requires at least the latest week).")
    parser.add_argument("--tickers-file", default="dataset/training_stocks.txt",
                        help="Universe of tickers. Only stocks included here will be predicted.")
    parser.add_argument("--metadata-file", required=True,
                        help="JSON metadata emitted by train_lstm.py (points to model + scaler).")
    parser.add_argument("--results-dir", default="lstm/results/forecasts",
                        help="Directory for forecast CSV files.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_artifacts(metadata: Dict, device: str) -> Dict:
    model_config = LSTMConfig(**metadata["model_config"])
    model = StockLSTM(model_config)
    state_dict = torch.load(metadata["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    scaler = joblib.load(metadata["scaler_path"])
    if not hasattr(scaler, "transform"):
        raise TypeError(
            f"Loaded scaler object does not support transform(): got {type(scaler).__name__}. "
            "Ensure the scaler_path points to the StandardScaler artifact produced during training."
        )
    return {"model": model, "scaler": scaler}


def build_inference_batch(df: pd.DataFrame, feature_cols: List[str],
                          sequence_length: int) -> Dict[str, np.ndarray]:
    sequences = []
    lengths = []
    tickers = []
    latest_dates = []

    for ticker, group in df.groupby("ticker"):
        ordered = group.sort_values("Date")
        feature_matrix = ordered[feature_cols].to_numpy(dtype=np.float32, copy=True)
        if feature_matrix.size == 0:
            continue

        effective_len = min(sequence_length, feature_matrix.shape[0])
        padded = np.zeros((sequence_length, feature_matrix.shape[1]), dtype=np.float32)
        padded[-effective_len:] = feature_matrix[-effective_len:]

        sequences.append(padded)
        lengths.append(effective_len)
        tickers.append(str(ticker))
        latest_dates.append(ordered["Date"].iloc[-1])

    if not sequences:
        raise RuntimeError("No sequences were created for forecasting.")

    return {
        "sequences": np.stack(sequences, axis=0),
        "lengths": np.array(lengths, dtype=np.int64),
        "tickers": tickers,
        "latest_dates": latest_dates,
    }


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    metadata = load_metadata(args.metadata_file)
    artifacts = load_artifacts(metadata, args.device)
    model: StockLSTM = artifacts["model"]
    scaler = artifacts["scaler"]

    tickers = set(load_ticker_universe(args.tickers_file))
    df = pd.read_csv(args.data_file)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["Date"])
    df = df[df["ticker"].isin(tickers)]
    df = df.sort_values(["ticker", "Date"])

    feature_cols = metadata["feature_columns"]
    missing_cols = sorted(set(feature_cols) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Dataset missing required feature columns: {missing_cols}")

    df = sanitize_features(df, feature_cols)
    df[feature_cols] = scaler.transform(df[feature_cols]).astype(np.float32)

    latest_data = df.groupby("ticker").tail(1)
    logger.info("Latest data spans %d tickers covering up to %s",
                latest_data["ticker"].nunique(),
                latest_data["Date"].max())

    inference_batch = build_inference_batch(df, feature_cols, metadata["sequence_length"])
    sequences = torch.from_numpy(inference_batch["sequences"]).to(args.device)
    lengths = torch.from_numpy(inference_batch["lengths"]).to(args.device)

    with torch.no_grad():
        preds = model(sequences, lengths).cpu().numpy()

    run_timestamp = datetime.now(timezone.utc)

    records = []
    for idx, ticker in enumerate(inference_batch["tickers"]):
        latest_date = inference_batch["latest_dates"][idx]
        target_week = (latest_date + timedelta(weeks=1)).date()
        records.append({
            "ticker": ticker,
            "forecast_horizon_weeks": 1,
            "predicted_return": preds[idx],
            "latest_data_date": latest_date,
            "sequence_length_used": int(inference_batch["lengths"][idx]),
            "generated_at_utc": run_timestamp,
            "forecast_start_date": latest_date.date(),
            "forecast_target_week": target_week,
        })

    forecasts_df = pd.DataFrame(records).sort_values("ticker")
    timestamp = run_timestamp.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.results_dir, f"lstm_forecasts_{timestamp}.csv")
    forecasts_df.to_csv(output_file, index=False)

    logger.info("Wrote %d forecasts to %s", len(forecasts_df), output_file)


if __name__ == "__main__":
    main()
