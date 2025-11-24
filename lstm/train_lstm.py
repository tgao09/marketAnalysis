import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import (
    MaskedStockDataset,
    add_forward_return_target,
    determine_time_split,
    dump_metadata,
    load_ticker_universe,
    masked_collate_fn,
    sanitize_features,
)
from model import LSTMConfig, StockLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a masked LSTM to predict next week's returns.")
    parser.add_argument("--data-file", default="dataset/stock_dataset_with_lags.csv",
                        help="Historical panel dataset with engineered features.")
    parser.add_argument("--tickers-file", default="dataset/training_stocks.txt",
                        help="Universe of tickers to include.")
    parser.add_argument("--sequence-length", type=int, default=32,
                        help="Number of weeks fed into the LSTM.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true",
                        help="Enable a bidirectional LSTM encoder.")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Portion of the timeline to keep for training if cutoff-date is not supplied.")
    parser.add_argument("--cutoff-date", type=str, default=None,
                        help="Explicit training cutoff date (YYYY-MM-DD). Overrides train-ratio.")
    parser.add_argument("--models-dir", default="lstm/models", help="Directory to write model artifacts.")
    parser.add_argument("--results-dir", default="lstm/results/training",
                        help="Directory to store training metrics.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 weight decay applied to the optimizer.")
    parser.add_argument("--early-stop-patience", type=int, default=8,
                        help="Stop training if validation loss does not improve for this many epochs (<=0 disables).")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum improvement in validation loss to reset patience.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_feature_list(df: pd.DataFrame, exclude_cols: Optional[set] = None) -> list:
    exclude_cols = exclude_cols or set()
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def create_dataloader(df: pd.DataFrame, feature_cols: list, sequence_length: int,
                      batch_size: int, shuffle: bool) -> DataLoader:
    dataset = MaskedStockDataset(df, feature_cols, target_col="target_return", sequence_length=sequence_length)
    if len(dataset) == 0:
        logger.warning("Dataset yielded zero samples (dates=%s).", df["Date"].tail(1).values if len(df) else "n/a")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=masked_collate_fn, drop_last=False)


def train_one_epoch(model: StockLSTM, loader: DataLoader, optimizer: Adam,
                    criterion: SmoothL1Loss, device: torch.device) -> float:
    model.train()
    running_loss = 0.0

    for batch in loader:
        sequences = torch.from_numpy(batch["sequences"]).to(device=device, dtype=torch.float32)
        lengths = torch.from_numpy(batch["lengths"]).to(device=device, dtype=torch.long)
        targets = torch.from_numpy(batch["targets"]).to(device=device, dtype=torch.float32)

        optimizer.zero_grad()
        predictions = model(sequences, lengths)
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * len(targets)

    if len(loader.dataset) == 0:
        return float("nan")
    return running_loss / len(loader.dataset)


def evaluate(model: StockLSTM, loader: DataLoader, criterion: SmoothL1Loss,
             device: torch.device) -> float:
    model.eval()
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            sequences = torch.from_numpy(batch["sequences"]).to(device=device, dtype=torch.float32)
            lengths = torch.from_numpy(batch["lengths"]).to(device=device, dtype=torch.long)
            targets = torch.from_numpy(batch["targets"]).to(device=device, dtype=torch.float32)

            predictions = model(sequences, lengths)
            loss = criterion(predictions, targets)

            running_loss += loss.item() * len(targets)
            total += len(targets)

    if total == 0:
        return float("nan")

    return running_loss / total


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    tickers = load_ticker_universe(args.tickers_file)
    df = pd.read_csv(args.data_file)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["Date"])
    df = df[df["ticker"].isin(tickers)].copy()
    df = add_forward_return_target(df, return_col="weekly_return", target_col="target_return")

    df = df.sort_values(["ticker", "Date"])
    df = df[df["target_return"].notna()]

    exclude_cols = {"ticker", "Date", "target_return"}
    feature_cols = build_feature_list(df, exclude_cols)
    df = sanitize_features(df, feature_cols)

    cutoff = None
    if args.cutoff_date:
        cutoff = pd.to_datetime(args.cutoff_date)

    cutoff = determine_time_split(df, train_ratio=args.train_ratio, cutoff_date=cutoff)
    train_df = df[df["Date"] <= cutoff].copy()
    val_df = df[df["Date"] > cutoff].copy()

    if len(val_df) == 0:
        logger.warning("Validation segment empty; using last 10%% of training period as proxy.")
        unique_dates = sorted(train_df["Date"].unique())
        proxy_idx = max(1, int(len(unique_dates) * 0.9))
        proxy_cutoff = unique_dates[proxy_idx]
        val_df = train_df[train_df["Date"] >= proxy_cutoff].copy()
        train_df = train_df[train_df["Date"] < proxy_cutoff].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols]).astype(np.float32)
    val_df[feature_cols] = scaler.transform(val_df[feature_cols]).astype(np.float32)

    train_loader = create_dataloader(train_df, feature_cols, args.sequence_length,
                                     batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_df, feature_cols, args.sequence_length,
                                   batch_size=args.batch_size, shuffle=False)

    model_config = LSTMConfig(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    model = StockLSTM(model_config).to(args.device)
    criterion = SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Older torch versions don't support the verbose flag on ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    history_rows = []
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss = evaluate(model, val_loader, criterion, args.device)
        scheduler.step(val_loss if not np.isnan(val_loss) else train_loss)

        logger.info("Epoch %03d | train=%.6f | val=%.6f | lr=%.6f",
                    epoch, train_loss, val_loss, optimizer.param_groups[0]["lr"])

        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        current_val = val_loss if not np.isnan(val_loss) else train_loss
        if current_val < (best_val_loss - args.min_delta):
            best_val_loss = current_val
            best_state = model.state_dict()
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if args.early_stop_patience > 0 and epochs_since_improvement >= args.early_stop_patience:
            logger.info("Early stopping at epoch %d (no val improvement for %d epochs).",
                        epoch, args.early_stop_patience)
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model.")

    logger.info("Best model from epoch %d with val loss %.6f", best_epoch, best_val_loss)

    timestamp_dt = datetime.now(timezone.utc)
    timestamp = timestamp_dt.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.models_dir, f"lstm_next_week_{timestamp}.pt")
    scaler_path = os.path.join(args.models_dir, f"feature_scaler_{timestamp}.pkl")
    metadata_path = os.path.join(args.models_dir, f"lstm_metadata_{timestamp}.json")

    torch.save(best_state, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "feature_columns": feature_cols,
        "sequence_length": args.sequence_length,
        "train_cutoff_date": str(pd.Timestamp(cutoff).date()),
        "tickers_file": args.tickers_file,
        "data_file": args.data_file,
        "training_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "model_config": model.get_config_dict(),
        "timestamp_utc": timestamp_dt.isoformat(),
        "best_epoch": best_epoch,
        "max_epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "training_hyperparams": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "early_stop_patience": args.early_stop_patience,
            "min_delta": args.min_delta,
        },
    }
    dump_metadata(metadata, metadata_path)

    history_df = pd.DataFrame(history_rows)
    history_file = os.path.join(args.results_dir, f"training_history_{timestamp}.csv")
    history_df.to_csv(history_file, index=False)

    logger.info("Artifacts written:\n model=%s\n scaler=%s\n metadata=%s\n history=%s",
                model_path, scaler_path, metadata_path, history_file)


if __name__ == "__main__":
    main()
