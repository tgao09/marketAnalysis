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

from common import PCATransformer, parse_window
from gp_return.train import (
    ARTIFACT_DIR_DEFAULT,
    DATA_YEARS,
    DEFAULT_TRAIN_ITERS,
    DEFAULT_TRAIN_WINDOW,
    FEATURE_LOOKBACK_MAX,
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    REGIME_SCORE_WINDOW,
    WINDOW_RET,
    build_features,
    build_target,
    compute_regime_score,
    extract_field,
    fetch_history_cached,
    normalize_features,
    resolve_artifact_variant,
    resolve_sector_etf,
    resolve_device,
    select_feature_columns,
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
    parser.add_argument(
        "--notional",
        type=float,
        default=10000.0,
        help="Dollar notional to size each trade (default: 10000).",
    )
    parser.add_argument("--train-iters", type=int, default=DEFAULT_TRAIN_ITERS)
    parser.add_argument("--train-window", default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument(
        "--include-time-index",
        action="store_true",
        help="Include time_index in features (default is to exclude).",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Enable fold-local PCA features for backtest and write outputs under ticker/pca.",
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
    buffer_days = max(FEATURE_LOOKBACK_MAX, REGIME_SCORE_WINDOW) + (2 * WINDOW_RET) + 5
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
    target = build_target(price_stock)

    price_spy_regime = price_spy.reindex(price_stock.index).ffill()
    price_vix_regime = price_vix.reindex(price_stock.index).ffill()
    regime_score = compute_regime_score(
        price_vix_regime,
        price_spy_regime,
        REGIME_SCORE_WINDOW,
        REGIME_SCORE_CLIP,
        REGIME_SCORE_WEIGHTS,
    )

    dataset = features.join([target])
    dataset["regime_score"] = regime_score
    dataset = dataset.dropna()
    if dataset.index.has_duplicates:
        dataset = dataset.loc[~dataset.index.duplicated(keep="last")]

    close_stock = price_stock.reindex(dataset.index)

    return {
        "dataset": dataset,
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


def build_backtest_pca_transformer() -> PCATransformer:
    return PCATransformer(
        threshold=0.80,
        max_pcs=12,
        impute_strategy="median",
        mode="replace",
        pc_prefix="pc_",
    )


def main():
    args = parse_args()
    device = resolve_device()
    print(f"Using device: {device.type}")
    ticker = args.ticker or prompt_ticker()
    if not ticker:
        print("No ticker provided. Exiting.")
        return

    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    test_start = end_date - pd.DateOffset(years=DEFAULT_TEST_YEARS)
    train_offset = parse_window(args.train_window)
    dataset_start = compute_dataset_start(end_date, args.train_window)

    print(f"Building dataset for {ticker}...")
    data = build_dataset(ticker, dataset_start, end_date)
    dataset = data["dataset"]
    close_stock = data["close"]

    feature_cols = select_feature_columns(
        dataset=dataset,
        drop_time_index=not args.include_time_index,
        pca_enabled=args.pca,
    )

    index_series = pd.Series(dataset.index, index=dataset.index)
    exit_date = index_series.shift(-WINDOW_RET)
    exit_close = close_stock.shift(-WINDOW_RET)

    candidates = pd.DataFrame(
        {
            "entry_close": close_stock,
            "exit_date": exit_date,
            "exit_close": exit_close,
        }
    )
    candidates = candidates.loc[test_start:end_date]
    candidates = candidates.dropna(subset=["entry_close", "exit_date", "exit_close"])

    test_dates = candidates.index
    dataset_index = dataset.index
    trades = []
    for test_date in test_dates:
        test_pos = int(dataset_index.searchsorted(test_date, side="left"))
        train_end_pos = test_pos - WINDOW_RET - 1
        if train_end_pos < 0:
            continue
        train_end = dataset_index[train_end_pos]
        train_start = test_date - train_offset - pd.offsets.BDay(WINDOW_RET)
        train_df = dataset.loc[(dataset.index > train_start) & (dataset.index <= train_end)]
        if len(train_df) < MIN_TRAIN_ROWS:
            continue
        
        test_df = dataset.loc[[test_date]]
        
        fold_start = train_df.index.min()
        train_df = set_time_index(train_df.copy(), fold_start)
        test_df = set_time_index(test_df.copy(), fold_start)
        fold_pca_k = None
        if args.pca:
            fold_pca = build_backtest_pca_transformer()
            train_x_df, test_x_df = fold_pca.transform_train_test(train_df, test_df, feature_cols)
            fold_pca_k = int(fold_pca.k_selected_)
        else:
            train_x_df, test_x_df, _ = normalize_features(train_df, test_df, feature_cols)

        train_x = torch.tensor(train_x_df.values, dtype=torch.float32, device=device)
        train_y = torch.tensor(train_df["target"].values, dtype=torch.float32, device=device)

        model, likelihood = train_gp(
            train_x,
            train_y,
            args.train_iters,
            device=device,
        )

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            test_x = torch.tensor(test_x_df.values, dtype=torch.float32, device=device)
            preds = likelihood(model(test_x))
            mean_log = float(preds.mean.item())
            std_log = float(preds.variance.sqrt().item())

        direction = "long" if mean_log > 0.0 else "short"
        entry_close = float(candidates.at[test_date, "entry_close"])
        exit_dt = candidates.at[test_date, "exit_date"]
        exit_close = float(candidates.at[test_date, "exit_close"])
        shares = args.notional / entry_close
        pnl_per_share = (exit_close - entry_close) if direction == "long" else (entry_close - exit_close)
        pnl = shares * pnl_per_share
        return_pct = pnl / args.notional
        mean_simple = math.exp(mean_log) - 1.0
        actual_simple = (exit_close / entry_close) - 1.0

        trades.append(
            {
                "symbol": ticker,
                "trade_date": test_date,
                "exit_date": exit_dt,
                "direction": direction,
                "entry_close": entry_close,
                "exit_close": exit_close,
                "notional": args.notional,
                "shares": shares,
                "pnl_per_share": pnl_per_share,
                "pred_mean_log": mean_log,
                "pred_std_log": std_log,
                "pred_mean_simple": mean_simple,
                "actual_simple_return": actual_simple,
                "pnl": pnl,
                "return_pct": return_pct,
                "pca_enabled": bool(args.pca),
                "pca_k": fold_pca_k,
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
            "notional": args.notional,
            "avg_return_pct": float(trades_df["return_pct"].mean()),
            "pca_enabled": bool(args.pca),
            "artifact_variant": resolve_artifact_variant(args.pca),
        }
    )

    output_dir = Path(args.output_dir) / ticker / resolve_artifact_variant(args.pca)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "gp_return_trades.csv"
    summary_path = output_dir / "gp_return_summary.json"

    trades_df.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nTrades saved to: {trades_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
