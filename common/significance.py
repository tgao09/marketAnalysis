"""Significance metrics for trade-level returns versus a benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .yfinance_utils import get_history
except ImportError:  # pragma: no cover - fallback for running as a script
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from common.yfinance_utils import get_history


REQUIRED_COLUMNS = {
    "trade_date",
    "exit_date",
    "actual_simple_return",
}

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class SignificanceConfig:
    benchmark_ticker: str = "SPY"
    return_column: str = "actual_simple_return"
    apply_direction: bool = True
    lag: Optional[int] = None
    json_output_name: str = "significance_report.json"

    @staticmethod
    def from_dict(values: Dict[str, Any]) -> "SignificanceConfig":
        config_fields = {field.name for field in SignificanceConfig.__dataclass_fields__.values()}
        filtered = {key: value for key, value in values.items() if key in config_fields}
        return SignificanceConfig(**filtered)


@dataclass
class SignificanceReport:
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "metrics": self.metrics,
        }


def load_trades(trades_csv_path: str, return_column: str) -> pd.DataFrame:
    path = Path(trades_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_csv_path}")

    trades = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(trades.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Trades CSV missing required columns: {missing_list}")
    if return_column not in trades.columns:
        raise ValueError(f"Return column not found: {return_column}")

    trades["trade_date"] = pd.to_datetime(trades["trade_date"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    if trades["trade_date"].isna().any():
        raise ValueError("trade_date contains invalid or missing values.")
    if trades["exit_date"].isna().any():
        raise ValueError("exit_date contains invalid or missing values.")

    trades[return_column] = pd.to_numeric(trades[return_column], errors="coerce")
    if trades[return_column].isna().any():
        raise ValueError(f"{return_column} contains invalid or missing values.")

    trades = trades.sort_values("exit_date").reset_index(drop=True)
    return trades


def run_significance(
    trades_csv_path: str,
    benchmark_ticker: Optional[str] = None,
    config: Optional[SignificanceConfig] = None,
) -> SignificanceReport:
    config = config or SignificanceConfig()
    benchmark = benchmark_ticker or config.benchmark_ticker
    trades = load_trades(trades_csv_path, config.return_column)

    trade_returns = trades[config.return_column].astype(float).copy()
    trade_returns = _apply_direction(trades, trade_returns, config.apply_direction)

    bench_returns, bench_index, bench_used, bench_dropped = benchmark_trade_returns(trades, benchmark)

    aligned = pd.DataFrame(
        {
            "trade_return": trade_returns,
            "benchmark_return": bench_returns,
        }
    )
    aligned = aligned.dropna()
    active_returns = aligned["trade_return"] - aligned["benchmark_return"]

    lag_used, median_holding_days = auto_lag(trades, bench_index, config.lag)
    info_ratio_raw = information_ratio(active_returns)
    info_ratio = annualize_information_ratio(info_ratio_raw, median_holding_days)
    t_stat = newey_west_tstat(active_returns, lag_used)

    metadata = {
        "trades_path": str(Path(trades_csv_path).resolve()),
        "benchmark_ticker": benchmark,
        "trade_count": int(len(trades)),
        "active_trades_used": int(len(active_returns)),
        "benchmark_trades_used": int(bench_used),
        "benchmark_trades_dropped": int(bench_dropped),
        "start_date": trades["exit_date"].min().date().isoformat() if len(trades) else None,
        "end_date": trades["exit_date"].max().date().isoformat() if len(trades) else None,
        "lag_used": lag_used,
        "median_holding_trading_days": median_holding_days,
    }

    metrics = {
        "information_ratio": info_ratio,
        "information_ratio_raw": info_ratio_raw,
        "t_stat": t_stat,
        "active_mean": float(active_returns.mean()) if len(active_returns) else float("nan"),
        "active_std": float(active_returns.std(ddof=1)) if len(active_returns) > 1 else float("nan"),
        "trade_mean": float(trade_returns.mean()) if len(trade_returns) else float("nan"),
        "trade_std": float(trade_returns.std(ddof=1)) if len(trade_returns) > 1 else float("nan"),
        "benchmark_mean": float(bench_returns.mean()) if len(bench_returns) else float("nan"),
        "benchmark_std": float(bench_returns.std(ddof=1)) if len(bench_returns) > 1 else float("nan"),
    }

    return SignificanceReport(metadata=metadata, metrics=metrics)


def benchmark_trade_returns(
    trades: pd.DataFrame,
    benchmark_ticker: str,
) -> Tuple[pd.Series, pd.DatetimeIndex, int, int]:
    start_date = trades["trade_date"].min().date().isoformat()
    end_date = trades["exit_date"].max().date().isoformat()
    history = get_history(
        benchmark_ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
    )
    close = history["Close"].dropna().sort_index()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    trade_dates = trades["trade_date"].dt.normalize()
    exit_dates = trades["exit_date"].dt.normalize()

    entry_prices = _asof_prices(close, trade_dates)
    exit_prices = _asof_prices(close, exit_dates)

    bench_returns = exit_prices / entry_prices - 1.0
    bench_used = int(bench_returns.notna().sum())
    bench_dropped = int(len(bench_returns) - bench_used)

    return bench_returns, close.index, bench_used, bench_dropped


def active_returns(trade_returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    aligned = pd.DataFrame(
        {"trade_return": trade_returns, "benchmark_return": benchmark_returns}
    ).dropna()
    return aligned["trade_return"] - aligned["benchmark_return"]


def information_ratio(series: pd.Series) -> float:
    if series is None or len(series) < 2:
        return float("nan")
    std = series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(series.mean() / std)


def annualize_information_ratio(
    info_ratio: float,
    median_holding_days: Optional[int],
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    if (
        info_ratio is None
        or (isinstance(info_ratio, float) and np.isnan(info_ratio))
        or median_holding_days is None
        or median_holding_days <= 0
    ):
        return float("nan")
    scale = np.sqrt(trading_days_per_year / float(median_holding_days))
    return float(info_ratio * scale)


def newey_west_tstat(series: pd.Series, lag: int) -> float:
    if series is None:
        return float("nan")
    values = series.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 2:
        return float("nan")

    mean = float(values.mean())
    demeaned = values - mean
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    if lag <= 0:
        var = gamma0
    else:
        weights = 1.0 - np.arange(1, lag + 1, dtype=float) / (lag + 1.0)
        gammas = []
        for l in range(1, lag + 1):
            cov = float(np.dot(demeaned[l:], demeaned[:-l]) / n)
            gammas.append(cov)
        var = gamma0 + 2.0 * float(np.dot(weights, np.array(gammas, dtype=float)))

    if var <= 0 or np.isnan(var):
        return float("nan")
    se = float(np.sqrt(var / n))
    if se == 0 or np.isnan(se):
        return float("nan")
    return float(mean / se)


def auto_lag(
    trades: pd.DataFrame,
    benchmark_index: pd.DatetimeIndex,
    override: Optional[int],
) -> Tuple[int, Optional[int]]:
    trade_dates = trades["trade_date"].dt.normalize()
    exit_dates = trades["exit_date"].dt.normalize()

    entry_idx = benchmark_index.get_indexer(pd.DatetimeIndex(trade_dates), method="pad")
    exit_idx = benchmark_index.get_indexer(pd.DatetimeIndex(exit_dates), method="pad")

    valid = (entry_idx >= 0) & (exit_idx >= 0)
    if not valid.any():
        return int(override) if override is not None else 0, None

    holding = exit_idx[valid] - entry_idx[valid]
    holding = holding[holding >= 0]
    if holding.size == 0:
        return int(override) if override is not None else 0, None

    median_days = int(np.median(holding))
    if override is not None:
        return int(override), median_days
    lag = max(0, median_days - 1)
    return lag, median_days


def _asof_prices(close: pd.Series, dates: pd.Series) -> pd.Series:
    index = close.index
    if not index.is_monotonic_increasing:
        close = close.sort_index()
        index = close.index
    date_index = pd.DatetimeIndex(dates)
    positions = index.get_indexer(date_index, method="pad")
    values = np.full(len(positions), np.nan, dtype=float)
    valid = positions >= 0
    if valid.any():
        values[valid] = close.iloc[positions[valid]].to_numpy(dtype=float)
    return pd.Series(values, index=dates.index)


def _apply_direction(
    trades: pd.DataFrame,
    returns: pd.Series,
    apply_direction: bool,
) -> pd.Series:
    if not apply_direction:
        return returns
    if "direction" not in trades.columns:
        raise ValueError("direction column is required when apply_direction is True.")
    direction = trades["direction"].astype(str).str.lower()
    sign = np.where(direction.str.startswith("short") | direction.eq("sell"), -1.0, 1.0)
    return returns * sign


def write_significance_json(
    report: SignificanceReport, trades_csv_path: str, output_name: str
) -> Path:
    trades_path = Path(trades_csv_path)
    output_path = trades_path.parent / output_name
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return output_path


def print_significance_report(report: SignificanceReport) -> None:
    metrics = report.metrics
    metadata = report.metadata

    def _fmt(value: Any, decimals: int = 4) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}"
        return str(value)

    print("Significance Summary")
    print(f"Trades: {metadata.get('trade_count')}")
    print(f"Active Trades Used: {metadata.get('active_trades_used')}")
    print(f"Date Range: {metadata.get('start_date')} -> {metadata.get('end_date')}")
    print(f"Information Ratio (Annualized): {_fmt(metrics.get('information_ratio'))}")
    print(f"Information Ratio (Per-Trade): {_fmt(metrics.get('information_ratio_raw'))}")
    print(f"HAC t-stat: {_fmt(metrics.get('t_stat'))}")
    print(f"Lag Used: {_fmt(metadata.get('lag_used'), 0)}")


def _parse_config(path: Optional[str]) -> Optional[SignificanceConfig]:
    if not path:
        return None
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Config file must contain a JSON object.")
    return SignificanceConfig.from_dict(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Significance metrics for a trades CSV.")
    parser.add_argument("trades_csv", help="Path to trades CSV.")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker (e.g., SPY).")
    parser.add_argument("--lag", type=int, default=None, help="Override Newey-West lag.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--output-name", default=None, help="Output JSON file name.")
    args = parser.parse_args()

    config = _parse_config(args.config) or SignificanceConfig()
    if args.lag is not None:
        config = SignificanceConfig(
            benchmark_ticker=config.benchmark_ticker,
            return_column=config.return_column,
            apply_direction=config.apply_direction,
            lag=args.lag,
            json_output_name=config.json_output_name,
        )

    report = run_significance(args.trades_csv, args.benchmark, config)
    output_name = args.output_name or config.json_output_name
    output_path = write_significance_json(report, args.trades_csv, output_name)
    print_significance_report(report)
    print(f"\nJSON report written to {output_path}")


if __name__ == "__main__":
    main()
