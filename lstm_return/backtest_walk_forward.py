import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import parse_window
from gp_return.train import resolve_device
from lstm_return.train import (
    ARTIFACT_DIR_DEFAULT,
    ARTIFACT_VARIANT_REGULAR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_HMM_N_INIT,
    DEFAULT_HMM_N_ITER,
    DEFAULT_HMM_TRAIN_WINDOW,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_SEQ_LEN,
    DEFAULT_TRAIN_WINDOW,
    DEFAULT_WEIGHT_DECAY,
    WINDOW_RET,
    apply_feature_scaler,
    build_feature_frame_with_hmm,
    build_model_dataset,
    compute_dataset_start,
    evaluate_predictions,
    fit_feature_scaler,
    fit_hmm_window_bundle,
    fit_lstm_model,
    predict_sequences,
)
from hmm_regime.train import build_market_dataset
from gbm_return.train import select_feature_columns


DEFAULT_TEST_YEARS = 1
TRADES_FILENAME = "lstm_return_trades.csv"
SUMMARY_FILENAME = "lstm_return_summary.json"
HMM_BACKTEST_FEATURE_COLUMNS = [f"p_state_{idx}" for idx in range(4)] + ["shift_prob"]
MIN_TRAIN_SEQUENCES = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for LSTM return model with HMM regime features."
    )
    parser.add_argument("--ticker", default=None, help="Ticker symbol.")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--notional",
        type=float,
        default=10000.0,
        help="Dollar notional to size each trade (default: 10000).",
    )
    parser.add_argument("--train-window", default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--include-time-index",
        action="store_true",
        help="Include time_index in features (default is to exclude).",
    )
    parser.add_argument("--hmm-train-window", default=DEFAULT_HMM_TRAIN_WINDOW)
    parser.add_argument("--hmm-n-iter", type=int, default=DEFAULT_HMM_N_ITER)
    parser.add_argument("--hmm-n-init", type=int, default=DEFAULT_HMM_N_INIT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR_DEFAULT))
    return parser.parse_args()


def prompt_ticker() -> str | None:
    raw = input("Ticker to backtest: ").strip()
    if not raw:
        return None
    return raw.upper()


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


def summarize_trades(trades: pd.DataFrame) -> dict[str, float | int | None]:
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": None,
            "avg_pnl": None,
            "median_pnl": None,
            "std_pnl": None,
            "max_drawdown": None,
            "avg_return_pct": None,
            "median_return_pct": None,
            "std_return_pct": None,
            "return_tstat": None,
            "profit_factor": None,
            "mae_log": None,
            "mse_log": None,
        }

    pnl = trades["pnl"]
    returns = trades["return_pct"]
    daily = trades.groupby("trade_date")["pnl"].sum().sort_index()
    equity = daily.cumsum()
    drawdown = equity - equity.cummax()
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    std_return = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    return_tstat = None
    if len(returns) > 1 and std_return > 0.0:
        return_tstat = float((returns.mean() / std_return) * math.sqrt(len(returns)))
    profit_factor = None
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss

    log_metrics = evaluate_predictions(
        pred=trades["predicted_log_return"].to_numpy(dtype=float),
        actual=trades["actual_log_return"].to_numpy(dtype=float),
    )
    return {
        "total_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "std_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "max_drawdown": max_drawdown,
        "avg_return_pct": float(returns.mean()),
        "median_return_pct": float(returns.median()),
        "std_return_pct": std_return,
        "return_tstat": return_tstat,
        "profit_factor": profit_factor,
        "mae_log": log_metrics["mae"],
        "mse_log": log_metrics["mse"],
    }


def prepare_backtest_data(
    ticker: str,
    end_date: pd.Timestamp,
    train_window: str,
    hmm_train_window: str,
    history_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    start_date = compute_dataset_start(end_date, train_window, hmm_train_window)
    test_start = end_date - pd.DateOffset(years=DEFAULT_TEST_YEARS)
    data = build_model_dataset(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        history_cache=history_cache,
    )
    market_dataset = build_market_dataset(start_date, end_date)
    dataset = data["dataset"]
    close_stock = data["close"]

    index_series = pd.Series(dataset.index, index=dataset.index)
    exit_date = index_series.shift(-WINDOW_RET)
    exit_close = close_stock.shift(-WINDOW_RET)
    candidates = pd.DataFrame(
        {"entry_close": close_stock, "exit_date": exit_date, "exit_close": exit_close}
    )
    candidates = candidates.loc[test_start:end_date]
    candidates = candidates.dropna(subset=["entry_close", "exit_date", "exit_close"])
    return {
        "ticker": ticker,
        "end_date": end_date,
        "test_start": test_start,
        "train_window": train_window,
        "dataset": dataset,
        "market_dataset": market_dataset,
        "close_stock": close_stock,
        "candidates": candidates,
        "test_dates": candidates.index,
        "dataset_index": dataset.index,
        "sector_etf": data["sector_etf"],
        "sector_name": data["sector_name"],
    }


def run_backtest_prepared(
    prepared: dict[str, Any],
    notional: float,
    include_time_index: bool,
    seq_len: int,
    model_config: dict[str, Any],
    hmm_config: dict[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    dataset = prepared["dataset"]
    market_dataset = prepared["market_dataset"]
    candidates = prepared["candidates"]
    test_dates = prepared["test_dates"]
    dataset_index = prepared["dataset_index"]
    base_feature_cols = select_feature_columns(
        dataset=dataset,
        drop_time_index=not include_time_index,
    )
    feature_cols = list(base_feature_cols) + list(HMM_BACKTEST_FEATURE_COLUMNS)
    model_kwargs = {
        "input_size": len(feature_cols),
        "hidden_size": model_config["hidden_size"],
        "num_layers": model_config["num_layers"],
        "dropout": model_config["dropout"],
    }

    trades: list[dict[str, Any]] = []
    skipped = 0

    for test_date in test_dates:
        test_pos = int(dataset_index.searchsorted(test_date, side="left"))
        train_end_pos = test_pos - WINDOW_RET - 1
        if train_end_pos < 0:
            continue

        train_end = dataset_index[train_end_pos]
        train_start = test_date - parse_window(prepared["train_window"]) - pd.offsets.BDay(WINDOW_RET)
        train_dates = dataset_index[(dataset_index > train_start) & (dataset_index <= train_end)]
        if len(train_dates) < max(seq_len + MIN_TRAIN_SEQUENCES, 252):
            skipped += 1
            continue

        hmm_bundle = fit_hmm_window_bundle(
            market_dataset=market_dataset,
            asof_date=train_end,
            train_window=hmm_config["train_window"],
            n_iter=hmm_config["n_iter"],
            n_init=hmm_config["n_init"],
            random_state=hmm_config["random_state"],
        )
        frame = build_feature_frame_with_hmm(
            dataset=dataset,
            market_dataset=market_dataset,
            hmm_bundle=hmm_bundle,
            start_date=train_dates.min(),
            end_date=test_date,
            time_index_start=train_dates.min(),
        )
        train_frame = frame.loc[frame.index <= train_end]
        if len(train_frame) < max(seq_len + MIN_TRAIN_SEQUENCES, 252):
            skipped += 1
            continue

        scaler = fit_feature_scaler(train_frame, feature_cols)
        scaled = frame.copy()
        scaled[feature_cols] = apply_feature_scaler(frame, scaler, feature_cols)
        train_x, train_y, _ = build_sequence_samples(
            frame=scaled,
            feature_cols=feature_cols,
            seq_len=seq_len,
            eligible_dates=train_dates,
        )
        test_x, _, test_seq_dates = build_sequence_samples(
            frame=scaled,
            feature_cols=feature_cols,
            seq_len=seq_len,
            eligible_dates=pd.DatetimeIndex([test_date]),
        )
        if len(train_x) < 32 or len(test_x) == 0:
            skipped += 1
            continue

        model, _ = fit_lstm_model(
            train_x=train_x,
            train_y=train_y,
            model_kwargs=model_kwargs,
            device=device,
            epochs=model_config["epochs"],
            batch_size=model_config["batch_size"],
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"],
            seed=hmm_config["random_state"],
        )
        pred = float(predict_sequences(model, test_x, device=device)[0])
        actual_log = float(dataset.loc[test_date, "target"])
        entry_close = float(candidates.loc[test_date, "entry_close"])
        exit_date = pd.Timestamp(candidates.loc[test_date, "exit_date"])
        exit_close = float(candidates.loc[test_date, "exit_close"])
        direction = "long" if pred >= 0.0 else "short"
        actual_simple = math.exp(actual_log) - 1.0
        predicted_simple = math.exp(pred) - 1.0
        pnl = notional * actual_simple if direction == "long" else notional * (-actual_simple)
        trades.append(
            {
                "trade_date": str(pd.Timestamp(test_date).date()),
                "sequence_asof": str(pd.Timestamp(test_seq_dates[-1]).date()),
                "exit_date": str(exit_date.date()),
                "direction": direction,
                "entry_close": entry_close,
                "exit_close": exit_close,
                "predicted_log_return": pred,
                "predicted_simple_return": predicted_simple,
                "actual_log_return": actual_log,
                "actual_simple_return": actual_simple,
                "pnl": pnl,
                "return_pct": pnl / notional,
            }
        )

    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades_df)
    summary.update(
        {
            "ticker": prepared["ticker"],
            "sector": prepared["sector_name"],
            "sector_etf": prepared["sector_etf"],
            "end_date": str(pd.Timestamp(prepared["end_date"]).date()),
            "test_start": str(pd.Timestamp(prepared["test_start"]).date()),
            "notional": float(notional),
            "train_window": prepared["train_window"],
            "seq_len": int(seq_len),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "skipped_dates": int(skipped),
        }
    )
    return trades_df, summary, feature_cols


def save_backtest_outputs(
    output_dir: Path,
    ticker: str,
    trades_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    artifact_dir = output_dir / ticker / ARTIFACT_VARIANT_REGULAR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(artifact_dir / TRADES_FILENAME, index=False)
    payload = dict(summary)
    payload["generated_at"] = datetime.now(UTC).isoformat()
    (artifact_dir / SUMMARY_FILENAME).write_text(json.dumps(payload, indent=2))
    print(f"Backtest outputs saved to: {artifact_dir}")


def main() -> None:
    args = parse_args()
    ticker = args.ticker or prompt_ticker()
    if not ticker:
        print("No ticker provided. Exiting.")
        return

    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    parse_window(args.train_window)
    parse_window(args.hmm_train_window)
    device = resolve_device()
    print(f"Using device: {device.type}")
    print(f"Building dataset for {ticker}...")

    prepared = prepare_backtest_data(
        ticker=ticker,
        end_date=end_date,
        train_window=args.train_window,
        hmm_train_window=args.hmm_train_window,
        history_cache={},
    )
    trades_df, summary, _ = run_backtest_prepared(
        prepared=prepared,
        notional=args.notional,
        include_time_index=args.include_time_index,
        seq_len=int(args.seq_len),
        model_config={
            "hidden_size": int(args.hidden_size),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        hmm_config={
            "train_window": args.hmm_train_window,
            "n_iter": int(args.hmm_n_iter),
            "n_init": int(args.hmm_n_init),
            "random_state": int(args.random_state),
        },
        device=device,
    )

    avg_return = summary["avg_return_pct"] if summary["avg_return_pct"] is not None else float("nan")
    win_rate = summary["win_rate"] if summary["win_rate"] is not None else float("nan")
    profit_factor = summary["profit_factor"] if summary["profit_factor"] is not None else float("nan")
    print(
        f"Trades: {summary['total_trades']} | "
        f"Avg return: {avg_return:.4%} | "
        f"Win rate: {win_rate:.2%} | "
        f"Profit factor: {profit_factor:.4f}"
    )
    save_backtest_outputs(Path(args.output_dir), ticker=ticker, trades_df=trades_df, summary=summary)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
