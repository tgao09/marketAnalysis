import argparse
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss, SmoothL1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

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
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (set >0 to speed up batch prep).")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true",
                        help="Pin host memory for faster CUDA transfer.")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false",
                        help="Disable pinned memory (override CUDA default).")
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
    parser.add_argument("--loss", choices=["smooth_l1", "mse", "l1"], default="smooth_l1",
                        help="Regression loss to optimize.")
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0,
                        help="Beta parameter for SmoothL1 loss when selected.")
    parser.add_argument("--clip-target-quantiles", nargs=2, type=float, default=[1.0, 99.0],
                        metavar=("LOW_Q", "HIGH_Q"),
                        help="Percentiles used to clip target returns (set with --disable-target-clipping to skip).")
    parser.add_argument("--disable-target-clipping", action="store_true",
                        help="Skip clipping of extreme target returns.")
    parser.add_argument("--balance-ticker-samples", dest="balance_ticker_samples", action="store_true",
                        help="Rebalance sampler so tickers with long histories do not dominate.")
    parser.add_argument("--no-balance-ticker-samples", dest="balance_ticker_samples", action="store_false",
                        help="Disable ticker-balanced sampling.")
    parser.add_argument("--use-amp", dest="use_amp", action="store_true",
                        help="Enable mixed precision (recommended on CUDA).")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false",
                        help="Disable mixed precision.")
    parser.add_argument("--early-stop-patience", type=int, default=8,
                        help="Stop training if validation loss does not improve for this many epochs (<=0 disables).")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum improvement in validation loss to reset patience.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(pin_memory=None, use_amp=None, balance_ticker_samples=True)
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
                      batch_size: int, shuffle: bool, num_workers: int = 0,
                      pin_memory: bool = False, sampler=None) -> Tuple[MaskedStockDataset, DataLoader]:
    dataset = MaskedStockDataset(df, feature_cols, target_col="target_return", sequence_length=sequence_length)
    if len(dataset) == 0:
        logger.warning("Dataset yielded zero samples (dates=%s).", df["Date"].tail(1).values if len(df) else "n/a")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=masked_collate_fn,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, loader


def build_loss_fn(loss_name: str, smooth_l1_beta: float):
    if loss_name == "smooth_l1":
        return SmoothL1Loss(beta=smooth_l1_beta)
    if loss_name == "mse":
        return MSELoss()
    if loss_name == "l1":
        return L1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {"hit_rate": float("nan"), "pearson": float("nan"), "r2": float("nan")}
    if preds.size == 0 or targets.size == 0:
        return metrics

    mask = np.isfinite(preds) & np.isfinite(targets)
    preds = preds[mask]
    targets = targets[mask]
    if preds.size == 0:
        return metrics

    non_zero_mask = targets != 0
    if non_zero_mask.any():
        hit_rate = np.mean(np.sign(preds[non_zero_mask]) == np.sign(targets[non_zero_mask]))
        metrics["hit_rate"] = float(hit_rate)

    if preds.std() > 0 and targets.std() > 0:
        metrics["pearson"] = float(np.corrcoef(preds, targets)[0, 1])

    variance = float(((targets - targets.mean()) ** 2).sum())
    if variance > 0:
        resid = preds - targets
        r2 = 1.0 - float((resid ** 2).sum()) / variance
        metrics["r2"] = r2

    return metrics


def train_one_epoch(model: StockLSTM, loader: DataLoader, optimizer: Adam,
                    criterion, device: torch.device, scaler: Optional[GradScaler],
                    use_amp: bool) -> float:
    model.train()
    running_loss = 0.0

    for batch in loader:
        sequences = torch.from_numpy(batch["sequences"]).to(device=device, dtype=torch.float32)
        lengths = torch.from_numpy(batch["lengths"]).to(device=device, dtype=torch.long)
        targets = torch.from_numpy(batch["targets"]).to(device=device, dtype=torch.float32)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            predictions = model(sequences, lengths)
            loss = criterion(predictions, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * len(targets)

    if len(loader.dataset) == 0:
        return float("nan")
    return running_loss / len(loader.dataset)


def evaluate(model: StockLSTM, loader: DataLoader, criterion,
             device: torch.device, use_amp: bool) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            sequences = torch.from_numpy(batch["sequences"]).to(device=device, dtype=torch.float32)
            lengths = torch.from_numpy(batch["lengths"]).to(device=device, dtype=torch.long)
            targets = torch.from_numpy(batch["targets"]).to(device=device, dtype=torch.float32)

            with autocast(enabled=use_amp):
                predictions = model(sequences, lengths)
                loss = criterion(predictions, targets)

            running_loss += loss.item() * len(targets)
            total += len(targets)
            all_preds.append(predictions.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    if total == 0:
        return {
            "loss": float("nan"),
            "hit_rate": float("nan"),
            "pearson": float("nan"),
            "r2": float("nan"),
            "count": 0,
        }

    preds_arr = np.concatenate(all_preds) if all_preds else np.array([])
    targets_arr = np.concatenate(all_targets) if all_targets else np.array([])
    metrics = compute_regression_metrics(preds_arr, targets_arr)
    metrics["loss"] = running_loss / total
    metrics["count"] = total
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    pin_memory = args.pin_memory if args.pin_memory is not None else args.device.startswith("cuda")
    use_amp = args.use_amp if args.use_amp is not None else args.device.startswith("cuda")

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

    target_clip_values = None
    if not args.disable_target_clipping and len(train_df) > 0:
        low_q, high_q = args.clip_target_quantiles
        if not (0 <= low_q < high_q <= 100):
            raise ValueError("--clip-target-quantiles must satisfy 0 <= low < high <= 100")
        low_val = float(np.nanpercentile(train_df["target_return"], low_q))
        high_val = float(np.nanpercentile(train_df["target_return"], high_q))
        train_df["target_return"] = train_df["target_return"].clip(low_val, high_val)
        val_df["target_return"] = val_df["target_return"].clip(low_val, high_val)
        target_clip_values = (low_val, high_val)
        logger.info("Clipped target_return to [%.6f, %.6f] using quantiles %.2f/%.2f",
                    low_val, high_val, low_q, high_q)

    feature_scaler = StandardScaler()
    feature_scaler.fit(train_df[feature_cols])
    train_df[feature_cols] = feature_scaler.transform(train_df[feature_cols]).astype(np.float32)
    val_df[feature_cols] = feature_scaler.transform(val_df[feature_cols]).astype(np.float32)

    train_dataset, train_loader = create_dataloader(
        train_df,
        feature_cols,
        args.sequence_length,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    sampler = None
    if args.balance_ticker_samples and len(train_dataset) > 0:
        ticker_counts = Counter(sample.ticker for sample in train_dataset.samples)
        weights = torch.as_tensor(
            [1.0 / ticker_counts[sample.ticker] for sample in train_dataset.samples],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=masked_collate_fn,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        logger.info("Enabled ticker-balanced sampling across %d tickers.", len(ticker_counts))

    _, val_loader = create_dataloader(
        val_df,
        feature_cols,
        args.sequence_length,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model_config = LSTMConfig(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    model = StockLSTM(model_config).to(args.device)
    criterion = build_loss_fn(args.loss, args.smooth_l1_beta)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Older torch versions don't support the verbose flag on ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    amp_scaler = GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    history_rows = []
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device, amp_scaler, use_amp)
        val_metrics = evaluate(model, val_loader, criterion, args.device, use_amp)
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss if not np.isnan(val_loss) else train_loss)

        logger.info(
            "Epoch %03d | train=%.6f | val=%.6f | hit=%.3f | r2=%.3f | corr=%.3f | lr=%.6f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["hit_rate"],
            val_metrics["r2"],
            val_metrics["pearson"],
            optimizer.param_groups[0]["lr"],
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_hit_rate": val_metrics["hit_rate"],
            "val_pearson": val_metrics["pearson"],
            "val_r2": val_metrics["r2"],
        })

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
    joblib.dump(feature_scaler, scaler_path)

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
            "loss": args.loss,
            "smooth_l1_beta": args.smooth_l1_beta,
            "clip_target_quantiles": None if args.disable_target_clipping else args.clip_target_quantiles,
            "clip_target_values": target_clip_values,
            "balance_ticker_samples": args.balance_ticker_samples,
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
            "use_amp": use_amp,
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
