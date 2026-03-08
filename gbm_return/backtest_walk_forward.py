import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import parse_window
from gbm_return.configuration import (
    FEATURE_SET_CHOICES,
    FEATURE_SET_F0,
    resolve_lgbm_params,
)
from gbm_return.train import (
    ARTIFACT_DIR_DEFAULT,
    DATA_YEARS,
    DEFAULT_TRAIN_WINDOW,
    FEATURE_LOOKBACK_MAX,
    MIN_TRAIN_ROWS,
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    REGIME_SCORE_WINDOW,
    WINDOW_RET,
    build_features,
    build_target,
    compute_regime_score,
    extract_field,
    fetch_history_cached,
    prepare_lgbm_training_data,
    resolve_direction_mode,
    resolve_sector_etf,
    resolve_artifact_variant,
    select_feature_columns,
    set_time_index,
    train_lgbm,
)


DEFAULT_TEST_YEARS = 1
TRADES_FILENAME = "gbm_return_trades.csv"
SUMMARY_FILENAME = "gbm_return_summary.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for return GBM with 5-day trades."
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
    parser.add_argument(
        "--include-time-index",
        action="store_true",
        help="Include time_index in features (default is to exclude).",
    )
    parser.add_argument(
        "--feature-set",
        default=FEATURE_SET_F0,
        choices=FEATURE_SET_CHOICES,
        help="Feature set variant. F1/F2 are loaded from --feature-set-file.",
    )
    parser.add_argument(
        "--feature-set-file",
        default=str(ARTIFACT_DIR_DEFAULT / "feature_sets.json"),
        help="Path to feature_sets.json used for F1/F2 feature drops.",
    )
    parser.add_argument(
        "--lgbm-param-preset",
        default="baseline",
        help="Named LightGBM preset from gbm_return.configuration.",
    )
    parser.add_argument(
        "--lgbm-params-json",
        default=None,
        help="Optional JSON file with LightGBM params to merge on top of preset.",
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


def build_dataset(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    history_cache: dict[str, pd.DataFrame] | None = None,
):
    history_cache = history_cache or {}
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
    regime_score = compute_regime_score(
        price_vix.reindex(price_stock.index).ffill(),
        price_spy.reindex(price_stock.index).ffill(),
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
            "avg_return_pct": None,
            "median_return_pct": None,
            "std_return_pct": None,
            "return_tstat": None,
            "profit_factor": None,
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
    }


def prepare_backtest_data(
    ticker: str,
    end_date: pd.Timestamp,
    train_window: str,
    history_cache: dict[str, pd.DataFrame] | None = None,
):
    dataset_start = compute_dataset_start(end_date, train_window)
    test_start = end_date - pd.DateOffset(years=DEFAULT_TEST_YEARS)
    data = build_dataset(
        ticker=ticker,
        start_date=dataset_start,
        end_date=end_date,
        history_cache=history_cache,
    )
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
    test_dates = candidates.index
    return {
        "ticker": ticker,
        "end_date": end_date,
        "test_start": test_start,
        "train_window": train_window,
        "dataset": dataset,
        "close_stock": close_stock,
        "candidates": candidates,
        "test_dates": test_dates,
        "dataset_index": dataset.index,
        "sector_etf": data["sector_etf"],
        "sector_name": data["sector_name"],
    }


def run_backtest_prepared(
    prepared: dict[str, Any],
    notional: float,
    include_time_index: bool,
    feature_set: str,
    feature_set_file: str | None,
    lgbm_params: dict[str, Any],
    training_policy: dict[str, Any] | None = None,
    direction_mode: str | None = None,
    verbose: bool = True,
):
    dataset = prepared["dataset"]
    candidates = prepared["candidates"]
    test_dates = prepared["test_dates"]
    dataset_index = prepared["dataset_index"]
    train_offset = parse_window(prepared["train_window"])

    feature_cols = select_feature_columns(
        dataset=dataset,
        drop_time_index=not include_time_index,
        feature_set=feature_set,
        feature_set_file=feature_set_file,
    )
    resolved_direction_mode = resolve_direction_mode(direction_mode)

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
        train_x, train_y, sample_weight, _ = prepare_lgbm_training_data(
            train_df,
            feature_cols,
            training_policy,
        )
        model = train_lgbm(train_x, train_y, lgbm_params, sample_weight=sample_weight)
        mean_log = float(model.predict(test_df[feature_cols])[0])
        if resolved_direction_mode == "long_only":
            direction = "long"
        elif resolved_direction_mode == "short_only":
            direction = "short"
        else:
            direction = "long" if mean_log >= 0.0 else "short"

        entry_close = float(candidates.at[test_date, "entry_close"])
        exit_dt = candidates.at[test_date, "exit_date"]
        exit_close = float(candidates.at[test_date, "exit_close"])

        shares = notional / entry_close
        pnl_per_share = (exit_close - entry_close) if direction == "long" else (entry_close - exit_close)
        pnl = shares * pnl_per_share
        return_pct = pnl / notional
        mean_simple = math.exp(mean_log) - 1.0
        actual_simple = (exit_close / entry_close) - 1.0

        trades.append(
            {
                "symbol": prepared["ticker"],
                "trade_date": test_date,
                "exit_date": exit_dt,
                "direction": direction,
                "entry_close": entry_close,
                "exit_close": exit_close,
                "notional": notional,
                "shares": shares,
                "pnl_per_share": pnl_per_share,
                "pred_mean_log": mean_log,
                "pred_mean_simple": mean_simple,
                "actual_simple_return": actual_simple,
                "pnl": pnl,
                "return_pct": return_pct,
            }
        )

        if verbose:
            print(
                f"{test_date.date()} | Train: {train_df.index.min().date()} -> {train_df.index.max().date()} | "
                f"Pred: {mean_simple:+.2%} | PnL: {pnl:+.2f}"
            )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("trade_date")
    summary = summarize_trades(trades_df)
    summary.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "ticker": prepared["ticker"],
            "start_date": str(prepared["test_start"].date()),
            "end_date": str(prepared["end_date"].date()),
            "train_window": prepared["train_window"],
            "test_years": DEFAULT_TEST_YEARS,
            "window_ret": WINDOW_RET,
            "notional": notional,
            "candidate_trade_days": int(len(test_dates)),
            "trade_rate": float(len(trades_df) / len(test_dates)) if len(test_dates) else 0.0,
            "feature_set": feature_set,
            "feature_count": len(feature_cols),
            "lgbm_params": dict(lgbm_params),
            "direction_mode": resolved_direction_mode,
        }
    )
    return trades_df, summary, feature_cols


def run_backtest(
    ticker: str,
    end_date: pd.Timestamp,
    train_window: str,
    notional: float,
    include_time_index: bool,
    feature_set: str,
    feature_set_file: str | None,
    lgbm_params: dict[str, Any],
    training_policy: dict[str, Any] | None,
    direction_mode: str | None,
    output_dir: str | Path,
    lgbm_param_preset: str = "baseline",
    lgbm_params_json: str | None = None,
    write_outputs: bool = True,
    verbose: bool = True,
    prepared: dict[str, Any] | None = None,
):
    prepared = prepared or prepare_backtest_data(ticker=ticker, end_date=end_date, train_window=train_window)

    trades_df, summary, feature_cols = run_backtest_prepared(
        prepared=prepared,
        notional=notional,
        include_time_index=include_time_index,
        feature_set=feature_set,
        feature_set_file=feature_set_file,
        lgbm_params=lgbm_params,
        training_policy=training_policy,
        direction_mode=direction_mode,
        verbose=verbose,
    )
    summary.update(
        {
            "include_time_index": bool(include_time_index),
            "feature_set_file": feature_set_file,
            "lgbm_param_preset": lgbm_param_preset,
            "lgbm_params_json": lgbm_params_json,
            "training_policy": training_policy,
            "direction_mode": summary.get("direction_mode"),
        }
    )

    trades_path = None
    summary_path = None
    if write_outputs:
        ticker_dir = Path(output_dir) / ticker / resolve_artifact_variant()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        trades_path = ticker_dir / TRADES_FILENAME
        summary_path = ticker_dir / SUMMARY_FILENAME
        trades_df.to_csv(trades_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2))
        if verbose:
            print(f"\nTrades saved to: {trades_path}")
            print(f"Summary saved to: {summary_path}")

    return {
        "trades": trades_df,
        "summary": summary,
        "feature_columns": feature_cols,
        "trades_path": trades_path,
        "summary_path": summary_path,
    }


def main():
    args = parse_args()
    ticker = args.ticker or prompt_ticker()
    if not ticker:
        print("No ticker provided. Exiting.")
        return

    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    lgbm_params = resolve_lgbm_params(
        preset_name=args.lgbm_param_preset,
        params_json=args.lgbm_params_json,
    )
    print(f"Building dataset for {ticker}...")
    run_backtest(
        ticker=ticker,
        end_date=end_date,
        train_window=args.train_window,
        notional=args.notional,
        include_time_index=args.include_time_index,
        feature_set=args.feature_set,
        feature_set_file=args.feature_set_file,
        lgbm_params=lgbm_params,
        training_policy=None,
        direction_mode=None,
        output_dir=args.output_dir,
        lgbm_param_preset=args.lgbm_param_preset,
        lgbm_params_json=args.lgbm_params_json,
        write_outputs=True,
        verbose=True,
    )


if __name__ == "__main__":
    np.random.seed(42)
    main()
