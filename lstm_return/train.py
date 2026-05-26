import argparse
import copy
import json
import pickle
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import parse_window, walk_forward_splits
from gbm_return.train import (
    build_features,
    extract_field,
    fetch_history_cached,
    resolve_sector_etf,
    select_feature_columns,
    set_time_index,
    validate_alignment_and_nan,
)
from gp_return.train import (
    DATA_YEARS,
    DEFAULT_STEP_WINDOW,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    REGIME_SCORE_WINDOW,
    TICKER_GOLD,
    TICKER_SPY,
    TICKER_VIX,
    build_target,
    compute_regime_score,
    compute_start_date as compute_base_start_date,
    resolve_device,
)
from hmm_regime.train import (
    DEFAULT_MIN_TRAIN_ROWS as HMM_DEFAULT_MIN_TRAIN_ROWS,
    apply_scaler as apply_hmm_scaler,
    build_market_dataset,
    build_state_output,
    compute_dataset_start as compute_hmm_start_date,
    compute_filtered_state_probs,
    compute_shift_probability,
    fit_hmm_bundle,
    select_training_features,
)


ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_VARIANT_REGULAR = "regular"
WINDOW_RET = 5
DEFAULT_SEQ_LEN = 60
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.0
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_RANDOM_STATE = 42
DEFAULT_HMM_TRAIN_WINDOW = "3y"
DEFAULT_HMM_N_ITER = 250
DEFAULT_HMM_N_INIT = 3
DEFAULT_MIN_TRAIN_SEQUENCES = 32
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_MIN_VAL_SEQUENCES = 16
DEFAULT_EARLY_STOPPING_PATIENCE = 8
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_HUBER_BETA = 0.02
HMM_FEATURE_COLUMNS = [f"p_state_{idx}" for idx in range(4)] + ["shift_prob"]
BASE_FEATURE_GROUPS = {
    "returns": {"ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d"},
    "volatility": {"vol_5d", "vol_10d", "vol_20d", "vol_60d", "skew_20d", "vol_ratio_5_20"},
    "trend": {"stock_ma20_gap", "stock_ma60_gap", "drawdown_60d"},
    "cross_asset": {
        "sector_ret_5d",
        "sector_vol_5d",
        "rel_strength_sector_20d",
        "gld_ret_5d",
        "gld_vol_5d",
        "spy_ret_5d",
        "spy_vol_20d",
        "spy_ma20_gap",
        "vix_chg_1d",
    },
    "calendar": {"q_phase_sin", "q_phase_cos", "time_index"},
    "regime": {"regime_score"},
}


class ReturnLSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        return self.head(hidden[-1]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM return model with HMM regime features.")
    parser.add_argument("--train-window", default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument(
        "--drop-time-index",
        action="store_true",
        help="Exclude time_index from training features (default).",
    )
    parser.add_argument(
        "--include-time-index",
        action="store_true",
        help="Include time_index in training features.",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR_DEFAULT))
    parser.add_argument("--hmm-train-window", default=DEFAULT_HMM_TRAIN_WINDOW)
    parser.add_argument("--hmm-n-iter", type=int, default=DEFAULT_HMM_N_ITER)
    parser.add_argument("--hmm-n-init", type=int, default=DEFAULT_HMM_N_INIT)
    parser.set_defaults(drop_time_index=True)
    return parser.parse_args()


def prompt_tickers() -> list[str]:
    raw = input("Enter tickers (comma-separated): ").strip()
    tokens = [token.strip().upper() for token in re.split(r"[,\s]+", raw) if token.strip()]
    seen: set[str] = set()
    tickers: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            tickers.append(token)
    return tickers


def compute_dataset_start(end_date: pd.Timestamp, train_window: str, hmm_train_window: str) -> pd.Timestamp:
    base_start = compute_base_start_date(
        end_date=pd.Timestamp(end_date).normalize(),
        data_years=DATA_YEARS,
        train_window=train_window,
        test_window=DEFAULT_TEST_WINDOW,
        regime_score_window=REGIME_SCORE_WINDOW,
    )
    hmm_start = compute_hmm_start_date(
        end_date=pd.Timestamp(end_date).normalize(),
        train_window=hmm_train_window,
        test_years=1,
    )
    return min(base_start, hmm_start)


def build_model_dataset(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    history_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    if history_cache is None:
        history_cache = {}
    sector_etf, sector_name, sector_error = resolve_sector_etf(ticker)
    if sector_error:
        print(f"{ticker}: sector fallback to {sector_etf} ({sector_error})")

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
    regime_score = compute_regime_score(
        price_vix.reindex(price_stock.index).ffill(),
        price_spy.reindex(price_stock.index).ffill(),
        REGIME_SCORE_WINDOW,
        REGIME_SCORE_CLIP,
        REGIME_SCORE_WEIGHTS,
    )

    validate_alignment_and_nan(
        ticker,
        features,
        target,
        price_stock,
        price_sector,
        price_gld,
        price_spy,
        price_vix,
    )

    dataset = features.join([target])
    dataset["regime_score"] = regime_score
    dataset = dataset.dropna()
    if dataset.empty:
        raise ValueError(f"{ticker}: No rows left after feature/target alignment.")
    if dataset.index.has_duplicates:
        dataset = dataset.loc[~dataset.index.duplicated(keep="last")]

    close_stock = price_stock.reindex(dataset.index)
    return {
        "dataset": dataset,
        "close": close_stock,
        "sector_etf": sector_etf,
        "sector_name": sector_name,
    }


def resolve_base_feature_columns(
    dataset: pd.DataFrame,
    drop_time_index: bool,
    feature_group_flags: dict[str, bool] | None = None,
) -> list[str]:
    feature_cols = select_feature_columns(
        dataset=dataset,
        drop_time_index=drop_time_index,
    )
    if not feature_group_flags:
        return feature_cols

    resolved_flags = {group: bool(feature_group_flags.get(group, True)) for group in BASE_FEATURE_GROUPS}
    selected: list[str] = []
    for col in feature_cols:
        include = True
        for group, group_cols in BASE_FEATURE_GROUPS.items():
            if col in group_cols:
                include = resolved_flags[group]
                break
        if include:
            selected.append(col)
    if not selected:
        raise ValueError("Feature group selection removed all base feature columns.")
    return selected


def fit_feature_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std(ddof=0).replace(0.0, 1.0)
    return {"mean": mean.to_dict(), "std": std.to_dict()}


def apply_feature_scaler(
    frame: pd.DataFrame,
    scaler: dict[str, dict[str, float]],
    feature_cols: list[str],
) -> pd.DataFrame:
    mean = pd.Series(scaler["mean"], dtype=float).reindex(feature_cols)
    std = pd.Series(scaler["std"], dtype=float).replace(0.0, 1.0).reindex(feature_cols)
    scaled = (frame[feature_cols] - mean) / std
    return scaled.replace([np.inf, -np.inf], np.nan)


def fit_hmm_window_bundle(
    market_dataset: pd.DataFrame,
    asof_date: pd.Timestamp,
    train_window: str,
    n_iter: int,
    n_init: int,
    random_state: int,
) -> dict[str, Any]:
    train_features = select_training_features(
        dataset=market_dataset,
        asof_date=pd.Timestamp(asof_date).normalize(),
        train_window=train_window,
        min_train_rows=HMM_DEFAULT_MIN_TRAIN_ROWS,
    )
    return fit_hmm_bundle(
        train_features=train_features,
        train_targets=market_dataset.loc[train_features.index],
        n_iter=n_iter,
        random_state=random_state,
        n_init=n_init,
    )


def compute_hmm_feature_frame(
    market_dataset: pd.DataFrame,
    hmm_bundle: dict[str, Any],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    align_index: pd.Index,
) -> pd.DataFrame:
    active_feature_columns = list(hmm_bundle["feature_columns"])
    filter_start = min(pd.Timestamp(start_date), pd.Timestamp(hmm_bundle["train_start"]))
    market_features = market_dataset.loc[
        (market_dataset.index >= filter_start)
        & (market_dataset.index <= pd.Timestamp(end_date)),
        active_feature_columns,
    ].dropna()
    if market_features.empty:
        raise ValueError("No market rows available for HMM feature computation.")

    scaled = apply_hmm_scaler(
        market_features,
        hmm_bundle["scaler"],
        feature_columns=active_feature_columns,
    )
    state_probs = compute_filtered_state_probs(hmm_bundle["model"], scaled.values)
    transition_matrix = np.asarray(hmm_bundle["transition_matrix"], dtype=float)
    shift_probability = compute_shift_probability(state_probs, transition_matrix)
    states = build_state_output(
        index=market_features.index,
        state_probs=state_probs,
        shift_probability=shift_probability,
        asof_date=market_features.index.max(),
        stress_state_id=int(hmm_bundle["stress_state_id"]),
    )
    regime = states.set_index("date")[HMM_FEATURE_COLUMNS]
    return regime.reindex(pd.DatetimeIndex(align_index)).ffill()


def build_feature_frame_with_hmm(
    dataset: pd.DataFrame,
    market_dataset: pd.DataFrame,
    hmm_bundle: dict[str, Any],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    time_index_start: pd.Timestamp,
) -> pd.DataFrame:
    frame = dataset.loc[
        (dataset.index >= pd.Timestamp(start_date)) & (dataset.index <= pd.Timestamp(end_date))
    ].copy()
    regime = compute_hmm_feature_frame(
        market_dataset=market_dataset,
        hmm_bundle=hmm_bundle,
        start_date=start_date,
        end_date=end_date,
        align_index=frame.index,
    )
    frame = frame.join(regime, how="left")
    frame = set_time_index(frame, pd.Timestamp(time_index_start))
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("No rows left after joining HMM regime features.")
    return frame


def build_sequence_samples(
    frame: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    eligible_dates: pd.Index,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    values = frame[feature_cols].to_numpy(dtype=np.float32)
    targets = frame["target"].to_numpy(dtype=np.float32)
    dates = pd.DatetimeIndex(frame.index)
    eligible_lookup = set(pd.DatetimeIndex(eligible_dates))

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    out_dates: list[pd.Timestamp] = []

    for end_pos in range(int(seq_len) - 1, len(frame)):
        current_date = dates[end_pos]
        if current_date not in eligible_lookup:
            continue
        window = values[end_pos - int(seq_len) + 1 : end_pos + 1]
        target_value = targets[end_pos]
        if not np.isfinite(window).all() or not np.isfinite(target_value):
            continue
        x_rows.append(window)
        y_rows.append(float(target_value))
        out_dates.append(current_date)

    if not x_rows:
        return (
            np.empty((0, int(seq_len), len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DatetimeIndex([]),
        )

    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        pd.DatetimeIndex(out_dates),
    )


def train_validation_split(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    n_rows = len(train_x)
    if n_rows < (DEFAULT_MIN_TRAIN_SEQUENCES + DEFAULT_MIN_VAL_SEQUENCES):
        return train_x, train_y, None, None

    val_count = max(DEFAULT_MIN_VAL_SEQUENCES, int(round(n_rows * float(val_fraction))))
    train_count = n_rows - val_count
    if train_count < DEFAULT_MIN_TRAIN_SEQUENCES:
        return train_x, train_y, None, None
    return (
        train_x[:train_count],
        train_y[:train_count],
        train_x[train_count:],
        train_y[train_count:],
    )


def build_dataloader(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
        torch.tensor(train_w, dtype=torch.float32),
    )
    effective_batch_size = max(1, min(int(batch_size), len(dataset)))
    return DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)


def fit_lstm_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    model_kwargs: dict[str, Any],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[ReturnLSTMModel, dict[str, float | int | None]]:
    if len(train_x) < DEFAULT_MIN_TRAIN_SEQUENCES:
        raise ValueError(
            f"Not enough train sequences for LSTM fit ({len(train_x)} < {DEFAULT_MIN_TRAIN_SEQUENCES})."
        )

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = ReturnLSTMModel(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    train_x_fit, train_y_fit, val_x, val_y = train_validation_split(train_x, train_y)
    train_weights = np.linspace(0.7, 1.0, len(train_x_fit), dtype=np.float32)
    train_loader = build_dataloader(train_x_fit, train_y_fit, train_weights, batch_size=batch_size)
    val_inputs = None
    val_targets = None
    if val_x is not None and val_y is not None:
        val_inputs = torch.tensor(val_x, dtype=torch.float32, device=device)
        val_targets = torch.tensor(val_y, dtype=torch.float32, device=device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    last_train_loss = float("nan")
    stale_epochs = 0

    for _ in range(int(epochs)):
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = F.smooth_l1_loss(preds, batch_y, reduction="none", beta=DEFAULT_HUBER_BETA)
            loss = (loss * batch_w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP_NORM)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        last_train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        if val_inputs is None or val_targets is None:
            best_state = copy.deepcopy(model.state_dict())
            continue

        model.eval()
        with torch.no_grad():
            val_preds = model(val_inputs)
            val_loss = float(F.smooth_l1_loss(val_preds, val_targets, beta=DEFAULT_HUBER_BETA).item())
        scheduler.step(val_loss)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= DEFAULT_EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, {
        "epochs": int(epochs),
        "train_loss": last_train_loss,
        "best_val_loss": None if not np.isfinite(best_val_loss) else best_val_loss,
        "train_sequences": int(len(train_x)),
        "fit_sequences": int(len(train_x_fit)),
        "val_sequences": 0 if val_x is None else int(len(val_x)),
    }


def predict_sequences(model: ReturnLSTMModel, seq_x: np.ndarray, device: torch.device) -> np.ndarray:
    if len(seq_x) == 0:
        return np.empty((0,), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(seq_x, dtype=torch.float32, device=device)
        preds = model(inputs).detach().cpu().numpy().astype(np.float32)
    return preds


def evaluate_predictions(pred: np.ndarray, actual: np.ndarray) -> dict[str, float | None]:
    pred_series = pd.Series(np.asarray(pred, dtype=float))
    actual_series = pd.Series(np.asarray(actual, dtype=float))
    mae = float(np.mean(np.abs(pred_series - actual_series)))
    mse = float(np.mean((pred_series - actual_series) ** 2))
    pred_simple = np.exp(pred_series) - 1.0
    actual_simple = np.exp(actual_series) - 1.0
    mae_simple = float(np.mean(np.abs(pred_simple - actual_simple)))
    directional = float(np.mean(np.sign(pred_series) == np.sign(actual_series)))
    corr = float(pred_series.corr(actual_series)) if len(pred_series) > 1 else None
    return {
        "mae": mae,
        "mse": mse,
        "mae_simple": mae_simple,
        "directional": directional,
        "corr": corr,
    }


def summarize_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, float | int | None]:
    mae_values = [fold["mae"] for fold in fold_metrics]
    mse_values = [fold["mse"] for fold in fold_metrics]
    mae_simple_values = [fold["mae_simple"] for fold in fold_metrics]
    dir_values = [fold["directional"] for fold in fold_metrics]
    corr_values = [fold["corr"] for fold in fold_metrics if fold.get("corr") is not None]
    return {
        "folds": len(fold_metrics),
        "mae_mean": float(np.mean(mae_values)) if mae_values else None,
        "mae_median": float(np.median(mae_values)) if mae_values else None,
        "mse_mean": float(np.mean(mse_values)) if mse_values else None,
        "mse_median": float(np.median(mse_values)) if mse_values else None,
        "mae_simple_mean": float(np.mean(mae_simple_values)) if mae_simple_values else None,
        "mae_simple_median": float(np.median(mae_simple_values)) if mae_simple_values else None,
        "directional_mean": float(np.mean(dir_values)) if dir_values else None,
        "directional_median": float(np.median(dir_values)) if dir_values else None,
        "corr_mean": float(np.mean(corr_values)) if corr_values else None,
        "corr_median": float(np.median(corr_values)) if corr_values else None,
    }


def serialize_hmm_bundle(hmm_bundle: dict[str, Any]) -> dict[str, Any]:
    scaler = hmm_bundle["scaler"]
    return {
        "model": hmm_bundle["model"],
        "feature_columns": list(hmm_bundle["feature_columns"]),
        "transition_matrix": np.asarray(hmm_bundle["transition_matrix"], dtype=float).tolist(),
        "stress_state_id": int(hmm_bundle["stress_state_id"]),
        "seed": int(hmm_bundle["seed"]),
        "train_start": str(pd.Timestamp(hmm_bundle["train_start"]).date()),
        "train_end": str(pd.Timestamp(hmm_bundle["train_end"]).date()),
        "scaler": {
            "mean": pd.Series(scaler["mean"], dtype=float).to_dict(),
            "std": pd.Series(scaler["std"], dtype=float).to_dict(),
            "lower": pd.Series(scaler["lower"], dtype=float).to_dict(),
            "upper": pd.Series(scaler["upper"], dtype=float).to_dict(),
        },
    }


def save_artifacts(
    artifact_dir: Path,
    model: ReturnLSTMModel,
    model_kwargs: dict[str, Any],
    scaler: dict[str, dict[str, float]],
    hmm_bundle: dict[str, Any],
    feature_cols: list[str],
    base_feature_cols: list[str],
    fold_metrics: list[dict[str, Any]],
    summary_metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    model_blob = {
        "model_state_dict": cpu_state_dict,
        "model_kwargs": dict(model_kwargs),
        "feature_columns": list(feature_cols),
        "base_feature_columns": list(base_feature_cols),
        "regime_feature_columns": list(HMM_FEATURE_COLUMNS),
        "seq_len": int(config["seq_len"]),
        "scaler": scaler,
        "hmm_bundle": serialize_hmm_bundle(hmm_bundle),
    }
    with (artifact_dir / "model_state.pkl").open("wb") as fh:
        pickle.dump(model_blob, fh)

    metrics_out = {
        "summary": summary_metrics,
        "folds": fold_metrics,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    (artifact_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nArtifacts saved to: {artifact_dir}")


def load_artifacts(artifact_dir: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], ReturnLSTMModel]:
    with (artifact_dir / "model_state.pkl").open("rb") as fh:
        model_blob = pickle.load(fh)
    config = json.loads((artifact_dir / "config.json").read_text())
    model = ReturnLSTMModel(**model_blob["model_kwargs"]).to(device)
    model.load_state_dict(model_blob["model_state_dict"])
    model.eval()
    return model_blob, config, model


def build_latest_sequence(
    ticker: str,
    config: dict[str, Any],
    model_blob: dict[str, Any],
    end_date: pd.Timestamp,
    history_cache: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.Timestamp, np.ndarray]:
    history_cache = history_cache or {}
    hmm_bundle = model_blob["hmm_bundle"]
    final_train_start = pd.Timestamp(config["final_train_window"]["start"])
    hmm_train_start = pd.Timestamp(hmm_bundle["train_start"])
    start_date = min(
        compute_dataset_start(
            end_date=end_date,
            train_window=config["train_window"],
            hmm_train_window=config["hmm_train_window"],
        ),
        final_train_start,
        hmm_train_start,
    )

    data = build_model_dataset(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        history_cache=history_cache,
    )
    market_dataset = build_market_dataset(start_date, end_date)
    frame = build_feature_frame_with_hmm(
        dataset=data["dataset"],
        market_dataset=market_dataset,
        hmm_bundle=hmm_bundle,
        start_date=min(final_train_start, hmm_train_start),
        end_date=end_date,
        time_index_start=final_train_start,
    )
    feature_cols = list(model_blob["feature_columns"])
    scaler = model_blob["scaler"]
    scaled = frame.copy()
    scaled[feature_cols] = apply_feature_scaler(frame, scaler, feature_cols)
    usable = scaled.dropna(subset=feature_cols)
    if len(usable) < int(model_blob["seq_len"]):
        raise ValueError(
            f"{ticker}: not enough rows for latest sequence ({len(usable)} < {int(model_blob['seq_len'])})."
        )

    latest = usable.iloc[-int(model_blob["seq_len"]) :]
    return pd.Timestamp(latest.index[-1]), latest[feature_cols].to_numpy(dtype=np.float32)


def prepare_fold_data(
    dataset: pd.DataFrame,
    market_dataset: pd.DataFrame,
    split_train_start: pd.Timestamp,
    split_train_end: pd.Timestamp,
    split_test_end: pd.Timestamp,
    split_train_dates: pd.Index,
    split_test_dates: pd.Index,
    base_feature_cols: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray, pd.DatetimeIndex, dict[str, Any], dict[str, Any]]:
    hmm_bundle = fit_hmm_window_bundle(
        market_dataset=market_dataset,
        asof_date=split_train_end,
        train_window=config["hmm_train_window"],
        n_iter=config["hmm_n_iter"],
        n_init=config["hmm_n_init"],
        random_state=config["random_state"],
    )
    frame = build_feature_frame_with_hmm(
        dataset=dataset,
        market_dataset=market_dataset,
        hmm_bundle=hmm_bundle,
        start_date=split_train_start,
        end_date=split_test_end,
        time_index_start=split_train_start,
    )
    feature_cols = list(base_feature_cols) + list(HMM_FEATURE_COLUMNS)
    train_frame = frame.loc[frame.index <= pd.Timestamp(split_train_end)]
    scaler = fit_feature_scaler(train_frame, feature_cols)
    scaled = frame.copy()
    scaled[feature_cols] = apply_feature_scaler(frame, scaler, feature_cols)

    train_x, train_y, train_dates = build_sequence_samples(
        frame=scaled,
        feature_cols=feature_cols,
        seq_len=config["seq_len"],
        eligible_dates=split_train_dates,
    )
    test_x, test_y, test_dates = build_sequence_samples(
        frame=scaled,
        feature_cols=feature_cols,
        seq_len=config["seq_len"],
        eligible_dates=split_test_dates,
    )
    return train_x, train_y, train_dates, test_x, test_y, test_dates, scaler, hmm_bundle


def train_for_ticker(
    ticker: str,
    config: dict[str, Any],
    history_cache: dict[str, pd.DataFrame],
    device: torch.device,
) -> dict[str, Any]:
    end_date = pd.Timestamp.today().normalize()
    start_date = compute_dataset_start(end_date, config["train_window"], config["hmm_train_window"])

    data = build_model_dataset(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        history_cache=history_cache,
    )
    market_dataset = build_market_dataset(start_date, end_date)
    dataset = data["dataset"]
    sector_etf = data["sector_etf"]
    sector_name = data["sector_name"]
    base_feature_cols = resolve_base_feature_columns(
        dataset=dataset,
        drop_time_index=bool(config["drop_time_index"]),
        feature_group_flags=config.get("feature_group_flags"),
    )

    splits = list(
        walk_forward_splits(
            dataset,
            train_window=config["train_window"],
            test_window=DEFAULT_TEST_WINDOW,
            embargo=WINDOW_RET,
            step=DEFAULT_STEP_WINDOW,
            min_train_rows=max(config["seq_len"] + DEFAULT_MIN_TRAIN_SEQUENCES, HMM_DEFAULT_MIN_TRAIN_ROWS),
        )
    )
    if not splits:
        raise ValueError(f"{ticker}: no walk-forward splits produced.")

    print(f"\nTraining {ticker} (sector ETF: {sector_etf})")
    fold_metrics: list[dict[str, Any]] = []
    model_kwargs = {
        "input_size": len(base_feature_cols) + len(HMM_FEATURE_COLUMNS),
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
    }

    for split in splits:
        (
            train_x,
            train_y,
            train_dates,
            test_x,
            test_y,
            test_dates,
            _,
            hmm_bundle,
        ) = prepare_fold_data(
            dataset=dataset,
            market_dataset=market_dataset,
            split_train_start=split.train_start,
            split_train_end=split.train_end,
            split_test_end=split.test_end,
            split_train_dates=split.train.index,
            split_test_dates=split.test.index,
            base_feature_cols=base_feature_cols,
            config=config,
        )
        if len(train_x) < DEFAULT_MIN_TRAIN_SEQUENCES:
            print(f"Fold {split.fold} skipped: insufficient train sequences ({len(train_x)}).")
            continue
        if len(test_x) == 0:
            print(f"Fold {split.fold} skipped: no test sequences after alignment.")
            continue

        model, fit_meta = fit_lstm_model(
            train_x=train_x,
            train_y=train_y,
            model_kwargs=model_kwargs,
            device=device,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            seed=config["random_state"],
        )
        preds = predict_sequences(model, test_x, device=device)
        metrics = evaluate_predictions(preds, test_y)
        print(
            f"Fold {split.fold} | Train: {split.train_start.date()} -> {split.train_end.date()} | "
            f"Test: {split.test_start.date()} -> {split.test_end.date()} | "
            f"Train seq: {len(train_x)} | Test seq: {len(test_x)} | "
            f"MAE(log): {metrics['mae']:.6f} | MSE: {metrics['mse']:.6f} | "
            f"Dir: {metrics['directional']:.2%}"
        )
        fold_metrics.append(
            {
                "fold": split.fold,
                "train_start": str(split.train_start.date()),
                "train_end": str(split.train_end.date()),
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
                "train_rows": int(len(split.train)),
                "test_rows": int(len(split.test)),
                "train_sequences": int(len(train_x)),
                "test_sequences": int(len(test_x)),
                "first_train_sequence_date": None if len(train_dates) == 0 else str(train_dates.min().date()),
                "first_test_sequence_date": None if len(test_dates) == 0 else str(test_dates.min().date()),
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "mae_simple": metrics["mae_simple"],
                "directional": metrics["directional"],
                "corr": metrics["corr"],
                "train_loss": fit_meta["train_loss"],
                "best_val_loss": fit_meta["best_val_loss"],
                "hmm_seed": int(hmm_bundle["seed"]),
                "hmm_train_start": str(pd.Timestamp(hmm_bundle["train_start"]).date()),
                "hmm_train_end": str(pd.Timestamp(hmm_bundle["train_end"]).date()),
            }
        )

    if not fold_metrics:
        raise ValueError(f"{ticker}: all folds were skipped.")

    summary_metrics = summarize_fold_metrics(fold_metrics)
    print(
        f"\nSummary | {ticker} | Folds: {summary_metrics['folds']} | "
        f"MAE(log) mean: {summary_metrics['mae_mean']:.6f} | "
        f"MAE(simple) mean: {summary_metrics['mae_simple_mean']:.4%} | "
        f"MSE mean: {summary_metrics['mse_mean']:.6f} | "
        f"Dir mean: {summary_metrics['directional_mean']:.2%}"
    )

    final_train_end = dataset.index.max()
    final_train_start = final_train_end - parse_window(config["train_window"])
    final_train_raw = dataset.loc[(dataset.index > final_train_start) & (dataset.index <= final_train_end)]
    if len(final_train_raw) < max(config["seq_len"] + DEFAULT_MIN_TRAIN_SEQUENCES, HMM_DEFAULT_MIN_TRAIN_ROWS):
        raise ValueError(
            f"{ticker}: not enough rows for final training window ({len(final_train_raw)} rows)."
        )

    (
        final_train_x,
        final_train_y,
        final_train_dates,
        _,
        _,
        _,
        final_scaler,
        final_hmm_bundle,
    ) = prepare_fold_data(
        dataset=dataset,
        market_dataset=market_dataset,
        split_train_start=final_train_raw.index.min(),
        split_train_end=final_train_raw.index.max(),
        split_test_end=final_train_raw.index.max(),
        split_train_dates=final_train_raw.index,
        split_test_dates=pd.DatetimeIndex([]),
        base_feature_cols=base_feature_cols,
        config=config,
    )
    if len(final_train_x) < DEFAULT_MIN_TRAIN_SEQUENCES:
        raise ValueError(f"{ticker}: not enough final train sequences ({len(final_train_x)}).")

    print(
        f"\nFinal model fit | Train: {final_train_raw.index.min().date()} -> "
        f"{final_train_raw.index.max().date()} | Sequences: {len(final_train_x)}"
    )
    final_model, final_fit_meta = fit_lstm_model(
        train_x=final_train_x,
        train_y=final_train_y,
        model_kwargs=model_kwargs,
        device=device,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        seed=config["random_state"],
    )

    config_out = dict(config)
    config_out.update(
        {
            "ticker": ticker,
            "sector": sector_name,
            "sector_etf": sector_etf,
            "artifact_variant": ARTIFACT_VARIANT_REGULAR,
            "base_feature_columns": base_feature_cols,
            "regime_feature_columns": HMM_FEATURE_COLUMNS,
            "feature_group_flags": config.get("feature_group_flags"),
            "final_train_window": {
                "start": str(final_train_raw.index.min().date()),
                "end": str(final_train_raw.index.max().date()),
                "rows": int(len(final_train_raw)),
                "sequences": int(len(final_train_dates)),
            },
            "final_fit": final_fit_meta,
            "generated_at": datetime.now(UTC).isoformat(),
        }
    )

    artifact_dir = Path(config["artifact_dir"]) / ticker / ARTIFACT_VARIANT_REGULAR
    save_artifacts(
        artifact_dir=artifact_dir,
        model=final_model,
        model_kwargs=model_kwargs,
        scaler=final_scaler,
        hmm_bundle=final_hmm_bundle,
        feature_cols=base_feature_cols + HMM_FEATURE_COLUMNS,
        base_feature_cols=base_feature_cols,
        fold_metrics=fold_metrics,
        summary_metrics=summary_metrics,
        config=config_out,
    )
    return summary_metrics


def main() -> None:
    args = parse_args()
    if args.include_time_index:
        args.drop_time_index = False
    parse_window(args.train_window)
    parse_window(args.hmm_train_window)
    if int(args.seq_len) <= 0:
        raise ValueError("--seq-len must be positive.")

    tickers = prompt_tickers()
    if not tickers:
        print("No ticker provided. Exiting.")
        return

    device = resolve_device()
    print(f"Using device: {device.type}")

    config = {
        "train_window": args.train_window,
        "artifact_dir": str(Path(args.artifact_dir)),
        "drop_time_index": args.drop_time_index,
        "seq_len": int(args.seq_len),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "random_state": int(args.random_state),
        "hmm_train_window": args.hmm_train_window,
        "hmm_n_iter": int(args.hmm_n_iter),
        "hmm_n_init": int(args.hmm_n_init),
    }

    np.random.seed(int(args.random_state))
    torch.manual_seed(int(args.random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.random_state))
    history_cache: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    for ticker in tickers:
        summaries[ticker] = train_for_ticker(ticker, config, history_cache, device)

    if summaries:
        print("\nFinished training:")
        for ticker, summary in summaries.items():
            print(
                f"  {ticker} | MAE(log) mean: {summary['mae_mean']:.6f} | "
                f"MAE(simple) mean: {summary['mae_simple_mean']:.4%} | "
                f"MSE mean: {summary['mse_mean']:.6f} | "
                f"Dir mean: {summary['directional_mean']:.2%}"
            )


if __name__ == "__main__":
    main()
