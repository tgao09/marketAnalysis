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

from common import get_history, parse_window, walk_forward_splits
from gp_vol.train import (
    ANNUALIZATION,
    DEFAULT_KERNEL_EQUATION,
    NOISE_WINDOW,
    WINDOW_VOL,
    build_features,
    build_target,
    extract_field,
    fetch_data,
    normalize_features,
    train_gp,
)


ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "artifacts"
XLK_TICKERS = ("XLK", "GLD", "SPY", "^VIX")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest proxy options straddles using GP vol forecast vs realized-vol proxy."
    )
    parser.add_argument("--symbols", default="XLK,PLTR,NVDA", help="Comma-separated tickers.")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument("--threshold", type=float, default=0.02, help="Absolute vol spread threshold.")
    parser.add_argument("--train-window", default="2y")
    parser.add_argument("--test-window", default="1m")
    parser.add_argument("--step-window", default="1m")
    parser.add_argument("--train-iters", type=int, default=200)
    parser.add_argument("--iv-window", type=int, default=20, help="Trailing days for IV proxy.")
    parser.add_argument("--exit-days", type=int, default=5, help="Holding period in trading days.")
    parser.add_argument("--fees", type=float, default=0.0, help="Per-contract fees.")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage as fraction of premium.")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR_DEFAULT))
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR_DEFAULT))
    return parser.parse_args()


def load_kernel_config(artifact_dir: Path):
    config_path = artifact_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        kernel = config.get("kernel")
        if kernel:
            return kernel
    return {
        "custom": False,
        "lengthscale": None,
        "period_length": None,
        "outputscale": None,
        "equation": DEFAULT_KERNEL_EQUATION,
    }


def build_gp_dataset(start_date: pd.Timestamp, end_date: pd.Timestamp):
    data = fetch_data(XLK_TICKERS, start_date, end_date)

    price_xlk = extract_field(data, "Close", XLK_TICKERS[0])
    volume_xlk = extract_field(data, "Volume", XLK_TICKERS[0])
    price_gld = extract_field(data, "Close", XLK_TICKERS[1])
    price_spy = extract_field(data, "Close", XLK_TICKERS[2])
    price_vix = extract_field(data, "Close", XLK_TICKERS[3])

    features = build_features(price_xlk, volume_xlk, price_gld, price_spy, price_vix)
    target, noise = build_target(price_xlk)

    dataset = features.join([target, noise]).dropna()
    if dataset.index.tz is not None:
        dataset.index = dataset.index.tz_localize(None)
    dataset.index = dataset.index.normalize()

    feature_cols = [col for col in dataset.columns if col not in ("target", "noise")]
    return dataset, feature_cols


def generate_forecast_series(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    kernel_config: dict,
    train_iters: int,
    train_window: str,
    test_window: str,
    step_window: str,
):
    forecasts = []
    # Target uses forward WINDOW_VOL days, so embargo must match to prevent leakage.
    horizon_embargo = WINDOW_VOL

    splits = walk_forward_splits(
        dataset,
        train_window=train_window,
        test_window=test_window,
        embargo=horizon_embargo,
        step=step_window,
        min_train_rows=60,
    )

    for split in splits:
        train_df = split.train
        test_df = split.test

        train_x_df, test_x_df, _ = normalize_features(train_df, test_df, feature_cols)

        train_x = torch.tensor(train_x_df.values, dtype=torch.float32)
        train_y = torch.tensor(train_df["target"].values, dtype=torch.float32)
        train_noise = torch.tensor(train_df["noise"].values, dtype=torch.float32).clamp_min(1e-8)

        model, likelihood = train_gp(
            train_x,
            train_y,
            train_noise,
            kernel_config,
            train_iters=train_iters,
        )

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            test_x = torch.tensor(test_x_df.values, dtype=torch.float32)
            preds = likelihood(model(test_x))
            mean_log = preds.mean.numpy()
            std_log = preds.variance.sqrt().numpy()

        mean_vol = np.exp(mean_log)
        lower_vol = np.exp(mean_log - (1.96 * std_log))
        upper_vol = np.exp(mean_log + (1.96 * std_log))

        forecasts.append(
            pd.DataFrame(
                {
                    "forecast_vol": mean_vol,
                    "forecast_vol_lower": lower_vol,
                    "forecast_vol_upper": upper_vol,
                },
                index=test_df.index,
            )
        )

        print(
            f"Fold {split.fold} | Train: {split.train_start.date()} -> {split.train_end.date()} | "
            f"Test: {split.test_start.date()} -> {split.test_end.date()}"
        )

    combined = pd.concat(forecasts).sort_index()
    if combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)
    combined.index = combined.index.normalize()
    return combined


def get_price_history(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    history = get_history(
        symbol,
        period=None,
        start=str(start_date.date()),
        end=str(end_date.date()),
        interval="1d",
        auto_adjust=True,
    )
    close = history["Close"].dropna()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index = close.index.normalize()
    return close


def build_trades_for_symbol(
    symbol: str,
    price: pd.Series,
    forecast_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    iv_window: int,
    exit_days: int,
    threshold: float,
    fees: float,
    slippage: float,
):
    if start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    if end_date.tz is not None:
        end_date = end_date.tz_localize(None)
    returns = price.pct_change()
    iv_proxy = returns.rolling(iv_window).std() * math.sqrt(ANNUALIZATION)

    index_series = pd.Series(price.index, index=price.index)
    exit_date = index_series.shift(-exit_days)
    future_price = price.shift(-exit_days)
    realized_move = (future_price - price).abs()
    premium = iv_proxy * price * math.sqrt(exit_days / ANNUALIZATION)

    frame = pd.DataFrame(
        {
            "price": price,
            "iv_proxy": iv_proxy,
            "exit_date": exit_date,
            "future_price": future_price,
            "realized_move": realized_move,
            "premium": premium,
        }
    )

    frame = frame.join(forecast_df[["forecast_vol"]], how="inner")
    frame = frame.loc[start_date:end_date]
    frame = frame.dropna(subset=["iv_proxy", "realized_move", "premium", "forecast_vol"])

    spread = frame["forecast_vol"] - frame["iv_proxy"]
    direction = np.where(spread > threshold, "long", np.where(spread < -threshold, "short", "flat"))
    frame["direction"] = direction
    frame = frame[frame["direction"] != "flat"]
    if frame.empty:
        return frame

    base_pnl = np.where(
        frame["direction"] == "long",
        (frame["realized_move"] - frame["premium"]) * 100.0,
        (frame["premium"] - frame["realized_move"]) * 100.0,
    )

    fees_total = fees * 2.0
    slippage_cost = slippage * frame["premium"] * 100.0
    pnl = base_pnl - fees_total - slippage_cost

    frame["pnl"] = pnl
    frame["symbol"] = symbol
    frame["trade_date"] = frame.index
    frame["spread"] = spread.loc[frame.index]

    return frame[
        [
            "symbol",
            "trade_date",
            "exit_date",
            "direction",
            "forecast_vol",
            "iv_proxy",
            "spread",
            "price",
            "future_price",
            "premium",
            "realized_move",
            "pnl",
        ]
    ]


def summarize_trades(trades: pd.DataFrame, exit_days: int):
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": None,
            "avg_pnl": None,
            "median_pnl": None,
            "std_pnl": None,
            "sharpe": None,
            "max_drawdown": None,
            "per_symbol": {},
        }

    pnl = trades["pnl"]
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = None
    if std_pnl > 0:
        sharpe = mean_pnl / std_pnl * math.sqrt(ANNUALIZATION / exit_days)

    daily = trades.groupby("trade_date")["pnl"].sum().sort_index()
    equity = daily.cumsum()
    drawdown = equity - equity.cummax()
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None

    per_symbol = {}
    for symbol, group in trades.groupby("symbol"):
        gpnl = group["pnl"]
        gmean = float(gpnl.mean())
        gstd = float(gpnl.std(ddof=1)) if len(gpnl) > 1 else 0.0
        gsharpe = None
        if gstd > 0:
            gsharpe = gmean / gstd * math.sqrt(ANNUALIZATION / exit_days)
        per_symbol[symbol] = {
            "trades": int(len(group)),
            "win_rate": float((gpnl > 0).mean()),
            "avg_pnl": gmean,
            "median_pnl": float(gpnl.median()),
            "std_pnl": gstd,
            "sharpe": gsharpe,
        }

    return {
        "total_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl": mean_pnl,
        "median_pnl": float(pnl.median()),
        "std_pnl": std_pnl,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "per_symbol": per_symbol,
    }


def main():
    args = parse_args()

    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]

    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    start_date = pd.Timestamp(args.start).normalize() if args.start else end_date - pd.DateOffset(years=2)

    train_offset = parse_window(args.train_window)
    buffer_days = NOISE_WINDOW + (2 * WINDOW_VOL) + 5
    dataset_start = start_date - train_offset - pd.DateOffset(days=buffer_days)

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_config = load_kernel_config(artifact_dir)

    print("Building GP dataset (XLK)...")
    dataset, feature_cols = build_gp_dataset(dataset_start, end_date)

    print("Generating walk-forward forecasts...")
    forecast_df = generate_forecast_series(
        dataset,
        feature_cols,
        kernel_config,
        train_iters=args.train_iters,
        train_window=args.train_window,
        test_window=args.test_window,
        step_window=args.step_window,
    )

    trades = []
    price_end = end_date + pd.DateOffset(days=args.exit_days + 5)
    price_start = start_date - pd.DateOffset(days=args.iv_window + 5)

    for symbol in symbols:
        print(f"Processing {symbol}...")
        price = get_price_history(symbol, price_start, price_end)
        trades_df = build_trades_for_symbol(
            symbol,
            price,
            forecast_df,
            start_date,
            end_date,
            iv_window=args.iv_window,
            exit_days=args.exit_days,
            threshold=args.threshold,
            fees=args.fees,
            slippage=args.slippage,
        )
        if not trades_df.empty:
            trades.append(trades_df)

    if trades:
        trades_df = pd.concat(trades).sort_values(["trade_date", "symbol"])
    else:
        trades_df = pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "exit_date",
                "direction",
                "forecast_vol",
                "iv_proxy",
                "spread",
                "price",
                "future_price",
                "premium",
                "realized_move",
                "pnl",
            ]
        )

    summary = summarize_trades(trades_df, args.exit_days)
    summary["generated_at"] = datetime.now(UTC).isoformat()
    summary["start_date"] = str(start_date.date())
    summary["end_date"] = str(end_date.date())
    summary["symbols"] = symbols
    summary["threshold"] = args.threshold
    summary["iv_window"] = args.iv_window
    summary["exit_days"] = args.exit_days
    summary["train_window"] = args.train_window
    summary["test_window"] = args.test_window
    summary["step_window"] = args.step_window

    trades_path = output_dir / "variance_proxy_trades.csv"
    summary_path = output_dir / "variance_proxy_summary.json"

    trades_df.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Trades saved to: {trades_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
