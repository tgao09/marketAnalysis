import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import parse_window
from return_gp.train import (
    ARTIFACT_DIR_DEFAULT,
    DATA_YEARS,
    DEFAULT_TRAIN_ITERS,
    DEFAULT_TRAIN_WINDOW,
    NOISE_WINDOW,
    WINDOW_RET,
    build_features,
    build_target,
    extract_field,
    fetch_history_cached,
    normalize_features,
    resolve_sector_etf,
    set_time_index,
    train_gp,
)


DEFAULT_TEST_YEARS = 1
MIN_TRAIN_ROWS = 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for return GP with 5-day trades."
    )
    parser.add_argument("--ticker", default=None, help="Ticker symbol.")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--train-iters", type=int, default=DEFAULT_TRAIN_ITERS)
    parser.add_argument("--train-window", default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument(
        "--include-time-index",
        action="store_true",
        help="Include time_index in features (default is to exclude).",
    )
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR_DEFAULT))
    return parser.parse_args()


def prompt_ticker():
    raw = input("Ticker to backtest: ").strip()
    if not raw:
        return None
    return raw.upper()


def compute_dataset_start(end_date: pd.Timestamp, train_window: str):
    train_offset = parse_window(train_window)
    base_start = end_date - pd.DateOffset(years=DATA_YEARS)
    buffer_days = NOISE_WINDOW + (2 * WINDOW_RET) + 5
    min_start = end_date - train_offset - pd.DateOffset(days=buffer_days)
    return min(base_start, min_start)


def build_dataset(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    history_cache: dict[str, pd.DataFrame] = {}

    sector_etf, sector_name, sector_error = resolve_sector_etf(ticker)
    if sector_error:
        print(f"{ticker}: sector fallback to {sector_etf} ({sector_error})")

    stock_history = fetch_history_cached(ticker, start_date, end_date, history_cache)
    sector_history = fetch_history_cached(sector_etf, start_date, end_date, history_cache)
    gld_history = fetch_history_cached("GLD", start_date, end_date, history_cache)
    spy_history = fetch_history_cached("SPY", start_date, end_date, history_cache)
    vix_history = fetch_history_cached("^VIX", start_date, end_date, history_cache)

    price_stock = extract_field(stock_history, "Close", ticker)
    open_stock = extract_field(stock_history, "Open", ticker)
    volume_stock = extract_field(stock_history, "Volume", ticker)
    price_sector = extract_field(sector_history, "Close", sector_etf)
    price_gld = extract_field(gld_history, "Close", "GLD")
    price_spy = extract_field(spy_history, "Close", "SPY")
    price_vix = extract_field(vix_history, "Close", "^VIX")

    features = build_features(
        price_stock,
        volume_stock,
        price_sector,
        price_gld,
        price_spy,
        price_vix,
    )
    target, noise = build_target(price_stock)

    dataset = features.join([target, noise]).dropna()
    if dataset.empty:
        raise ValueError(f"{ticker}: No rows left after feature/target alignment.")
    if dataset.index.has_duplicates:
        dataset = dataset.loc[~dataset.index.duplicated(keep="last")]

    open_stock = open_stock.reindex(dataset.index)
    close_stock = price_stock.reindex(dataset.index)

    return {
        "dataset": dataset,
        "open": open_stock,
        "close": close_stock,
        "sector_etf": sector_etf,
        "sector_name": sector_name,
    }


def summarize_trades(trades: pd.DataFrame):
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": None,
            "avg_pnl": None,
            "median_pnl": None,
            "std_pnl": None,
            "max_drawdown": None,
        }

    pnl = trades["pnl"]
    daily = trades.groupby("trade_date")["pnl"].sum().sort_index()
    equity = daily.cumsum()
    drawdown = equity - equity.cummax()
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None

    return {
        "total_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "std_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "max_drawdown": max_drawdown,
    }


def main():
    time_fail_count = 0
    
    args = parse_args()
    ticker = args.ticker or prompt_ticker()
    if not ticker:
        print("No ticker provided. Exiting.")
        return

    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    test_start = end_date - pd.DateOffset(years=DEFAULT_TEST_YEARS)
    dataset_start = compute_dataset_start(end_date, args.train_window)

    print(f"Building dataset for {ticker}...")
    data = build_dataset(ticker, dataset_start, end_date)
    dataset = data["dataset"]
    open_stock = data["open"]
    close_stock = data["close"]

    feature_cols = [col for col in dataset.columns if col not in ("target", "noise")]
    if not args.include_time_index:
        feature_cols = [col for col in feature_cols if col != "time_index"]

    index_series = pd.Series(dataset.index, index=dataset.index)
    exit_date = index_series.shift(-(WINDOW_RET + 1))
    exit_close = close_stock.shift(-(WINDOW_RET + 1))

    candidates = pd.DataFrame(
        {
            "entry_open": open_stock,
            "exit_date": exit_date,
            "exit_close": exit_close,
        }
    )
    candidates = candidates.loc[test_start:end_date]
    candidates = candidates.dropna(subset=["entry_open", "exit_date", "exit_close"])

    test_dates = candidates.index
    if test_dates.empty:
        raise ValueError("No eligible test dates found in the last year.")

    trades = []
    for test_date in test_dates:
        train_start = test_date - pd.DateOffset(years=2) - pd.offsets.BDay(WINDOW_RET)
        train_df = dataset.loc[(dataset.index > train_start) & (dataset.index < test_date - pd.offsets.BDay(WINDOW_RET))]
        if len(train_df) < MIN_TRAIN_ROWS:
            continue
        
        test_df = dataset.loc[[test_date]]
        
        fold_start = train_df.index.min()
        train_df = set_time_index(train_df.copy(), fold_start)
        test_df = set_time_index(test_df.copy(), fold_start)
        train_x_df, test_x_df, _ = normalize_features(train_df, test_df, feature_cols)

        train_x = torch.tensor(train_x_df.values, dtype=torch.float32)
        train_y = torch.tensor(train_df["target"].values, dtype=torch.float32)
        train_noise = torch.tensor(train_df["noise"].values, dtype=torch.float32).clamp_min(1e-8)

        model, likelihood = train_gp(
            train_x,
            train_y,
            args.train_iters,
        )

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            test_x = torch.tensor(test_x_df.values, dtype=torch.float32)
            test_noise = torch.tensor(test_df["noise"].values, dtype=torch.float32).clamp_min(1e-8)
            preds = likelihood(model(test_x), noise=test_noise)
            mean_log = float(preds.mean.item())
            std_log = float(preds.variance.sqrt().item())

        direction = "long" if mean_log > 0.0 else "short"
        try:
            entry_open = float(candidates.at[test_date + pd.offsets.BDay(1), "entry_open"])
            exit_dt = candidates.at[test_date, "exit_date"]
            exit_close = float(candidates.at[test_date, "exit_close"])
        except:
            time_fail_count+=1
            continue

        pnl = (exit_close - entry_open) if direction == "long" else (entry_open - exit_close)
        mean_simple = math.exp(mean_log) - 1.0
        actual_simple = (exit_close / entry_open) - 1.0

        trades.append(
            {
                "symbol": ticker,
                "trade_date": test_date,
                "exit_date": exit_dt,
                "direction": direction,
                "entry_open": entry_open,
                "exit_close": exit_close,
                "pred_mean_log": mean_log,
                "pred_std_log": std_log,
                "pred_mean_simple": mean_simple,
                "actual_simple_return": actual_simple,
                "pnl": pnl,
            }
        )

        print(
            f"{test_date.date()} | Train: {train_df.index.min().date()} -> {train_df.index.max().date()} | "
            f"Pred: {mean_simple:+.2%} | PnL: {pnl:+.2f}"
        )

    trades_df = pd.DataFrame(trades).sort_values("trade_date")
    summary = summarize_trades(trades_df)
    summary.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "ticker": ticker,
            "start_date": str(test_start.date()),
            "end_date": str(end_date.date()),
            "train_window": args.train_window,
            "test_years": DEFAULT_TEST_YEARS,
            "window_ret": WINDOW_RET,
            "time_fail_count": time_fail_count
        }
    )

    output_dir = Path(args.output_dir) / ticker
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "return_gp_trades.csv"
    summary_path = output_dir / "return_gp_summary.json"

    trades_df.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nTrades saved to: {trades_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Time failed count: {time_fail_count}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
