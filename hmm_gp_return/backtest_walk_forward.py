import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from gp_return.train import DEFAULT_TRAIN_WINDOW, resolve_device
from hmm_gp_return.train import (
    ARTIFACT_DIR_DEFAULT,
    DEFAULT_META_TRAIN_WINDOW,
    DEFAULT_TEST_YEARS,
    MIN_META_TRAIN_ROWS,
    build_base_prediction_rows,
    resolve_artifact_dir,
    run_meta_backtest,
    save_backtest_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest for HMM + GP residual ensemble.")
    parser.add_argument("--ticker", default=None, help="Ticker symbol.")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--base-train-window", default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--meta-train-window", default=DEFAULT_META_TRAIN_WINDOW)
    parser.add_argument(
        "--notional",
        type=float,
        default=10000.0,
        help="Dollar notional to size each trade (default: 10000).",
    )
    parser.add_argument("--output-dir", default=ARTIFACT_DIR_DEFAULT)
    return parser.parse_args()


def prompt_ticker() -> str | None:
    raw = input("Ticker to backtest: ").strip()
    if not raw:
        return None
    return raw.upper()


def main() -> None:
    args = parse_args()
    ticker = args.ticker or prompt_ticker()
    if not ticker:
        print("No ticker provided. Exiting.")
        return

    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    eval_start = end_date - pd.DateOffset(years=DEFAULT_TEST_YEARS)
    device = resolve_device()
    print(f"Using device: {device.type}")

    base_rows = build_base_prediction_rows(
        ticker=ticker,
        end_date=end_date,
        base_train_window=args.base_train_window,
        meta_train_window=args.meta_train_window,
        device=device,
    )
    predictions, trades, summary = run_meta_backtest(
        base_rows=base_rows,
        meta_train_window=args.meta_train_window,
        eval_start=eval_start,
        eval_end=end_date,
        min_meta_train_rows=MIN_META_TRAIN_ROWS,
        notional=args.notional,
    )

    summary["ticker"] = ticker.upper()
    summary["base_train_window"] = args.base_train_window
    summary["artifact_variant"] = "regular"
    summary["notional"] = args.notional

    artifact_dir = resolve_artifact_dir(args.output_dir, ticker)
    save_backtest_outputs(
        artifact_dir=artifact_dir,
        predictions=predictions,
        trades=trades,
        summary=summary,
    )

    print(f"Predictions saved to: {artifact_dir / 'ensemble_return_predictions.csv'}")
    print(f"Trades saved to: {artifact_dir / 'ensemble_return_trades.csv'}")
    print(f"Summary saved to: {artifact_dir / 'ensemble_return_summary.json'}")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
