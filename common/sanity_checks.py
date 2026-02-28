"""Sanity checks for trade-level performance and regime shifts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .yfinance_utils import get_history
except ImportError:  # pragma: no cover - fallback for running as a script
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from common.yfinance_utils import get_history


@dataclass(frozen=True)
class SanityConfig:
    rolling_window_trades: int = 30
    rolling_consecutive_windows: int = 3
    rolling_mean_std_mult: float = 2.0
    rolling_win_rate_drop: float = 0.15
    payoff_ratio_threshold: float = 0.7
    loss_autocorr_threshold: float = 0.25
    loss_streak_threshold: int = 10
    cvar5_threshold: float = -0.025
    q01_threshold: float = -0.05
    max_drawdown_threshold: float = -0.20
    max_underwater_days_threshold: int = 90
    max_underwater_trades_threshold: int = 20
    benchmark_window_trades: int = 30
    benchmark_corr_shift_threshold: float = 0.30
    benchmark_beta_shift_threshold: float = 0.50
    json_output_name: str = "sanity_report.json"
    apply_direction: bool = False

    @staticmethod
    def from_dict(values: Dict[str, Any]) -> "SanityConfig":
        config_fields = {field.name for field in SanityConfig.__dataclass_fields__.values()}
        filtered = {key: value for key, value in values.items() if key in config_fields}
        return SanityConfig(**filtered)


@dataclass
class SanityCheckResult:
    check_id: str
    status: str
    severity: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    notes: str | None = None


@dataclass
class SanityReport:
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    checks: List[SanityCheckResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "metrics": self.metrics,
            "checks": [asdict(check) for check in self.checks],
        }


REQUIRED_COLUMNS = {
    "symbol",
    "trade_date",
    "exit_date",
    "direction",
    "entry_open",
    "exit_close",
    "actual_simple_return",
    "pnl",
}


def run_sanity_checks(
    trades_csv_path: str,
    benchmark_ticker: Optional[str] = None,
    config: Optional[SanityConfig] = None,
) -> SanityReport:
    config = config or SanityConfig()
    trades = _load_trades(trades_csv_path)

    checks: List[SanityCheckResult] = []
    metadata = _build_metadata(trades_csv_path, trades, benchmark_ticker)

    sorted_trades, order_note = _sort_trades(trades)
    if order_note:
        checks.append(
            SanityCheckResult(
                check_id="data.exit_date_order",
                status="warn",
                severity="warning",
                metrics={"exit_date_sorted": False},
                thresholds={},
                notes=order_note,
            )
        )

    duplicate_rows = _count_duplicates(sorted_trades)
    if duplicate_rows > 0:
        checks.append(
            SanityCheckResult(
                check_id="data.duplicates",
                status="warn",
                severity="warning",
                metrics={"duplicate_rows": int(duplicate_rows)},
                thresholds={},
                notes="Duplicate (symbol, trade_date, exit_date) rows found.",
            )
        )

    returns = _prepare_returns(sorted_trades, config)
    if returns.isna().any():
        raise ValueError("actual_simple_return contains NaN values after parsing.")

    direction_note = _direction_warning(sorted_trades, returns, config)
    if direction_note:
        checks.append(
            SanityCheckResult(
                check_id="data.direction_returns",
                status="warn",
                severity="warning",
                metrics={"apply_direction": config.apply_direction},
                thresholds={},
                notes=direction_note,
            )
        )

    equity = _compute_equity_curve(returns)
    drawdown = _compute_drawdown(equity)
    underwater_stats = _compute_underwater_stats(sorted_trades["exit_date"], drawdown)

    metrics = _build_summary_metrics(returns, equity, drawdown, underwater_stats)

    checks.extend(_check_extended_loss(underwater_stats, metrics["max_drawdown"], config))
    checks.extend(_check_rolling_performance(returns, config))
    checks.extend(_check_loss_clustering(returns, config))
    checks.extend(_check_tail_risk(returns, config))
    checks.extend(_check_hit_rate_payoff(returns, config))

    benchmark_report = None
    if benchmark_ticker:
        benchmark_report = _benchmark_checks(
            sorted_trades,
            returns,
            benchmark_ticker,
            config,
        )
        metrics.update(benchmark_report["metrics"])
        checks.extend(benchmark_report["checks"])
        metadata["benchmark_trades_used"] = benchmark_report["benchmark_trades_used"]
        metadata["benchmark_trades_dropped"] = benchmark_report["metrics"].get("benchmark_trades_dropped")

    return SanityReport(metadata=metadata, metrics=metrics, checks=checks)


def write_report_json(report: SanityReport, trades_csv_path: str, output_name: str) -> Path:
    trades_path = Path(trades_csv_path)
    output_path = trades_path.parent / output_name
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return output_path


def print_report(report: SanityReport) -> None:
    metrics = report.metrics
    def _fmt(value: Any, decimals: int = 4) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}"
        return str(value)

    print("Sanity Check Summary")
    print(f"Trades: {report.metadata.get('trade_count')}")
    print(f"Date Range: {report.metadata.get('start_date')} -> {report.metadata.get('end_date')}")
    print(f"Max Drawdown: {_fmt(metrics.get('max_drawdown'))}")
    print(f"Max Underwater Days: {_fmt(metrics.get('max_underwater_days'), 0)}")
    print(f"Max Underwater Trades: {_fmt(metrics.get('max_underwater_trades'), 0)}")
    print(f"Win Rate: {_fmt(metrics.get('win_rate'))}")
    print(f"CVaR5: {_fmt(metrics.get('cvar_5'))}")

    if "benchmark_corr" in metrics:
        print(f"Benchmark Corr: {_fmt(metrics.get('benchmark_corr'))}")
        print(f"Benchmark Beta: {_fmt(metrics.get('benchmark_beta'))}")

    warn_checks = [check for check in report.checks if check.status in {"warn", "critical"}]
    if warn_checks:
        print("\nWarnings")
        for check in warn_checks:
            print(f"- {check.check_id} ({check.severity}): {check.notes}")
    else:
        print("\nWarnings")
        print("- None")


def _load_trades(trades_csv_path: str) -> pd.DataFrame:
    path = Path(trades_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_csv_path}")

    trades = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(trades.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Trades CSV missing required columns: {missing_list}")

    trades["trade_date"] = pd.to_datetime(trades["trade_date"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    if trades["exit_date"].isna().any():
        raise ValueError("exit_date contains invalid or missing values.")

    trades["actual_simple_return"] = pd.to_numeric(trades["actual_simple_return"], errors="coerce")
    return trades


def _build_metadata(trades_csv_path: str, trades: pd.DataFrame, benchmark_ticker: Optional[str]) -> Dict[str, Any]:
    return {
        "trades_path": str(Path(trades_csv_path).resolve()),
        "benchmark_ticker": benchmark_ticker,
        "trade_count": int(len(trades)),
        "start_date": trades["exit_date"].min().date().isoformat() if len(trades) else None,
        "end_date": trades["exit_date"].max().date().isoformat() if len(trades) else None,
    }


def _sort_trades(trades: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    if trades["exit_date"].is_monotonic_increasing:
        return trades.reset_index(drop=True), None
    sorted_trades = trades.sort_values("exit_date").reset_index(drop=True)
    return sorted_trades, "exit_date was not sorted; trades were sorted for analysis."


def _count_duplicates(trades: pd.DataFrame) -> int:
    duplicates = trades.duplicated(subset=["symbol", "trade_date", "exit_date"]).sum()
    return int(duplicates)


def _prepare_returns(trades: pd.DataFrame, config: SanityConfig) -> pd.Series:
    returns = trades["actual_simple_return"].astype(float).copy()
    if config.apply_direction:
        direction = trades["direction"].astype(str).str.lower()
        sign = np.where(direction.str.startswith("short") | direction.eq("sell"), -1.0, 1.0)
        returns = returns * sign
    return returns


def _direction_warning(trades: pd.DataFrame, returns: pd.Series, config: SanityConfig) -> Optional[str]:
    if config.apply_direction:
        return None
    direction = trades["direction"].astype(str).str.lower()
    if direction.str.contains("short").any() and (returns >= 0).all():
        return (
            "Short trades detected but returns are non-negative. "
            "If actual_simple_return is unsigned, consider enabling apply_direction."
        )
    return None


def _compute_equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _compute_underwater_stats(exit_dates: pd.Series, drawdown: pd.Series) -> Dict[str, Any]:
    underwater = drawdown < 0
    if not underwater.any():
        return {
            "max_underwater_days": 0,
            "max_underwater_trades": 0,
            "percent_time_underwater": 0.0,
        }

    max_days = 0
    max_trades = 0
    total_underwater = int(underwater.sum())
    start_idx: Optional[int] = None

    for idx, is_under in enumerate(underwater):
        if is_under and start_idx is None:
            start_idx = idx
        if not is_under and start_idx is not None:
            end_idx = idx - 1
            days = (exit_dates.iloc[end_idx] - exit_dates.iloc[start_idx]).days
            trades = end_idx - start_idx + 1
            max_days = max(max_days, days)
            max_trades = max(max_trades, trades)
            start_idx = None

    if start_idx is not None:
        end_idx = len(underwater) - 1
        days = (exit_dates.iloc[end_idx] - exit_dates.iloc[start_idx]).days
        trades = end_idx - start_idx + 1
        max_days = max(max_days, days)
        max_trades = max(max_trades, trades)

    return {
        "max_underwater_days": int(max_days),
        "max_underwater_trades": int(max_trades),
        "percent_time_underwater": float(total_underwater / len(underwater)),
    }


def _build_summary_metrics(
    returns: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series,
    underwater_stats: Dict[str, Any],
) -> Dict[str, Any]:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = float((returns > 0).mean()) if len(returns) else 0.0
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0

    q01 = float(returns.quantile(0.01)) if len(returns) else 0.0
    q05 = float(returns.quantile(0.05)) if len(returns) else 0.0
    cvar5 = float(returns[returns <= q05].mean()) if len(returns) else 0.0

    return {
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "max_underwater_days": underwater_stats["max_underwater_days"],
        "max_underwater_trades": underwater_stats["max_underwater_trades"],
        "percent_time_underwater": underwater_stats["percent_time_underwater"],
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "q01": q01,
        "q05": q05,
        "cvar_5": cvar5,
    }


def _check_extended_loss(
    underwater_stats: Dict[str, Any],
    max_drawdown: float,
    config: SanityConfig,
) -> List[SanityCheckResult]:
    duration_flag = underwater_stats["max_underwater_days"] > config.max_underwater_days_threshold
    trades_flag = underwater_stats["max_underwater_trades"] > config.max_underwater_trades_threshold
    drawdown_flag = max_drawdown < config.max_drawdown_threshold
    critical = duration_flag and drawdown_flag

    if critical:
        status = "critical"
        severity = "critical"
    elif duration_flag or trades_flag or drawdown_flag:
        status = "warn"
        severity = "warning"
    else:
        status = "pass"
        severity = "info"

    return [
        SanityCheckResult(
            check_id="loss.extended_underwater",
            status=status,
            severity=severity,
            metrics={**underwater_stats, "max_drawdown": max_drawdown},
            thresholds={
                "max_drawdown_threshold": config.max_drawdown_threshold,
                "max_underwater_days_threshold": config.max_underwater_days_threshold,
                "max_underwater_trades_threshold": config.max_underwater_trades_threshold,
            },
            notes="Extended drawdown or underwater duration detected."
            if status in {"warn", "critical"}
            else "No extended loss regime detected.",
        )
    ]


def _check_rolling_performance(returns: pd.Series, config: SanityConfig) -> List[SanityCheckResult]:
    if len(returns) < config.rolling_window_trades:
        return [
            SanityCheckResult(
                check_id="performance.rolling_mean",
                status="skipped",
                severity="info",
                metrics={"available_trades": int(len(returns))},
                thresholds={"rolling_window_trades": config.rolling_window_trades},
                notes="Not enough trades for rolling window check.",
            )
        ]

    overall_mean = returns.mean()
    overall_std = returns.std(ddof=0)
    threshold = overall_mean - config.rolling_mean_std_mult * overall_std
    rolling_mean = returns.rolling(config.rolling_window_trades).mean()
    below = rolling_mean < threshold
    consecutive = _max_consecutive_true(below.fillna(False))

    status = "warn" if consecutive >= config.rolling_consecutive_windows else "pass"
    return [
        SanityCheckResult(
            check_id="performance.rolling_mean",
            status=status,
            severity="warning" if status == "warn" else "info",
            metrics={
                "rolling_mean_threshold": float(threshold),
                "max_consecutive_breaches": int(consecutive),
            },
            thresholds={
                "rolling_window_trades": config.rolling_window_trades,
                "rolling_consecutive_windows": config.rolling_consecutive_windows,
                "rolling_mean_std_mult": config.rolling_mean_std_mult,
            },
            notes="Rolling mean return fell below threshold for multiple windows."
            if status == "warn"
            else "Rolling mean return within expected range.",
        )
    ]


def _check_loss_clustering(returns: pd.Series, config: SanityConfig) -> List[SanityCheckResult]:
    if len(returns) < 2:
        return [
            SanityCheckResult(
                check_id="loss.clustering",
                status="skipped",
                severity="info",
                metrics={"available_trades": int(len(returns))},
                thresholds={},
                notes="Not enough trades for clustering check.",
            )
        ]

    losses = returns < 0
    autocorr = float(losses.astype(int).corr(losses.shift(1)))
    max_streak = _max_consecutive_true(losses)
    status = "warn" if autocorr > config.loss_autocorr_threshold or max_streak > config.loss_streak_threshold else "pass"
    return [
        SanityCheckResult(
            check_id="loss.clustering",
            status=status,
            severity="warning" if status == "warn" else "info",
            metrics={
                "loss_autocorr_lag1": autocorr,
                "max_loss_streak": int(max_streak),
            },
            thresholds={
                "loss_autocorr_threshold": config.loss_autocorr_threshold,
                "loss_streak_threshold": config.loss_streak_threshold,
            },
            notes="Losses appear clustered." if status == "warn" else "No loss clustering detected.",
        )
    ]


def _check_tail_risk(returns: pd.Series, config: SanityConfig) -> List[SanityCheckResult]:
    if len(returns) == 0:
        return [
            SanityCheckResult(
                check_id="risk.tail",
                status="skipped",
                severity="info",
                metrics={},
                thresholds={},
                notes="No trades to compute tail risk.",
            )
        ]

    q01 = float(returns.quantile(0.01))
    q05 = float(returns.quantile(0.05))
    cvar5 = float(returns[returns <= q05].mean())
    status = "warn" if (cvar5 < config.cvar5_threshold or q01 < config.q01_threshold) else "pass"

    return [
        SanityCheckResult(
            check_id="risk.tail",
            status=status,
            severity="warning" if status == "warn" else "info",
            metrics={"q01": q01, "q05": q05, "cvar5": cvar5},
            thresholds={"cvar5_threshold": config.cvar5_threshold, "q01_threshold": config.q01_threshold},
            notes="Tail losses exceed thresholds." if status == "warn" else "Tail risk within thresholds.",
        )
    ]


def _check_hit_rate_payoff(returns: pd.Series, config: SanityConfig) -> List[SanityCheckResult]:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = float((returns > 0).mean()) if len(returns) else 0.0
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(abs(losses.mean())) if not losses.empty else 0.0
    payoff_ratio = float(avg_win / avg_loss) if avg_loss > 0 else float("inf")

    rolling_win = returns.gt(0).rolling(config.rolling_window_trades).mean()
    win_threshold = win_rate - config.rolling_win_rate_drop
    consecutive = _max_consecutive_true((rolling_win < win_threshold).fillna(False))
    win_rate_flag = consecutive >= config.rolling_consecutive_windows
    payoff_flag = payoff_ratio < config.payoff_ratio_threshold

    status = "warn" if win_rate_flag or payoff_flag else "pass"
    notes = []
    if win_rate_flag:
        notes.append("Rolling win rate fell below threshold.")
    if payoff_flag:
        notes.append("Payoff ratio below threshold.")
    if not notes:
        notes.append("Hit rate and payoff ratio within thresholds.")

    return [
        SanityCheckResult(
            check_id="performance.hit_rate_payoff",
            status=status,
            severity="warning" if status == "warn" else "info",
            metrics={
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "payoff_ratio": payoff_ratio,
                "max_consecutive_win_rate_breaches": int(consecutive),
            },
            thresholds={
                "rolling_win_rate_drop": config.rolling_win_rate_drop,
                "payoff_ratio_threshold": config.payoff_ratio_threshold,
                "rolling_window_trades": config.rolling_window_trades,
                "rolling_consecutive_windows": config.rolling_consecutive_windows,
            },
            notes=" ".join(notes),
        )
    ]


def _benchmark_checks(
    trades: pd.DataFrame,
    returns: pd.Series,
    benchmark_ticker: str,
    config: SanityConfig,
) -> Dict[str, Any]:
    start_date = trades["exit_date"].min().date().isoformat()
    end_date = trades["exit_date"].max().date().isoformat()
    history = get_history(
        benchmark_ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
    )
    close = history["Close"].dropna()
    benchmark_returns = close.pct_change().dropna()
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index.date)

    aligned = pd.DataFrame(
        {
            "exit_date": trades["exit_date"].dt.normalize(),
            "trade_return": returns,
        }
    )
    aligned = aligned.merge(
        benchmark_returns.rename("benchmark_return"),
        left_on="exit_date",
        right_index=True,
        how="inner",
    )

    benchmark_trades_used = int(len(aligned))
    benchmark_trades_dropped = int(len(trades) - benchmark_trades_used)
    metrics: Dict[str, Any] = {
        "benchmark_corr": None,
        "benchmark_beta": None,
        "benchmark_trades_used": benchmark_trades_used,
        "benchmark_trades_dropped": benchmark_trades_dropped,
    }
    checks: List[SanityCheckResult] = []

    if benchmark_trades_used < config.benchmark_window_trades:
        checks.append(
            SanityCheckResult(
                check_id="benchmark.regime",
                status="skipped",
                severity="info",
                metrics={
                    "benchmark_trades_used": benchmark_trades_used,
                    "benchmark_trades_dropped": benchmark_trades_dropped,
                },
                thresholds={"benchmark_window_trades": config.benchmark_window_trades},
                notes="Not enough benchmark-aligned trades.",
            )
        )
        return {"metrics": metrics, "checks": checks, "benchmark_trades_used": benchmark_trades_used}

    trade_series = aligned["trade_return"]
    bench_series = aligned["benchmark_return"]
    overall_corr = float(trade_series.corr(bench_series))
    overall_beta = float(trade_series.cov(bench_series) / bench_series.var()) if bench_series.var() != 0 else 0.0

    rolling_corr = trade_series.rolling(config.benchmark_window_trades).corr(bench_series)
    rolling_cov = trade_series.rolling(config.benchmark_window_trades).cov(bench_series)
    rolling_var = bench_series.rolling(config.benchmark_window_trades).var()
    rolling_beta = rolling_cov / rolling_var

    corr_shift = (rolling_corr - overall_corr).abs().max()
    beta_shift = (rolling_beta - overall_beta).abs().max()
    beta_sign_change = False
    if abs(overall_beta) > 1e-6:
        beta_sign_change = (np.sign(rolling_beta.dropna()) != np.sign(overall_beta)).any()

    status = "warn" if corr_shift > config.benchmark_corr_shift_threshold or beta_shift > config.benchmark_beta_shift_threshold or beta_sign_change else "pass"

    metrics.update(
        {
            "benchmark_corr": overall_corr,
            "benchmark_beta": overall_beta,
            "benchmark_corr_shift_max": float(corr_shift),
            "benchmark_beta_shift_max": float(beta_shift),
            "benchmark_beta_sign_change": bool(beta_sign_change),
        }
    )
    checks.append(
        SanityCheckResult(
            check_id="benchmark.regime",
            status=status,
            severity="warning" if status == "warn" else "info",
            metrics={
                "benchmark_corr": overall_corr,
                "benchmark_beta": overall_beta,
                "corr_shift_max": float(corr_shift),
                "beta_shift_max": float(beta_shift),
                "beta_sign_change": bool(beta_sign_change),
                "benchmark_trades_used": benchmark_trades_used,
                "benchmark_trades_dropped": benchmark_trades_dropped,
            },
            thresholds={
                "benchmark_window_trades": config.benchmark_window_trades,
                "benchmark_corr_shift_threshold": config.benchmark_corr_shift_threshold,
                "benchmark_beta_shift_threshold": config.benchmark_beta_shift_threshold,
            },
            notes="Benchmark correlation or beta shifted beyond thresholds."
            if status == "warn"
            else "Benchmark correlation and beta within thresholds.",
        )
    )
    return {"metrics": metrics, "checks": checks, "benchmark_trades_used": benchmark_trades_used}


def _max_consecutive_true(series: Iterable[bool]) -> int:
    max_count = 0
    current = 0
    for value in series:
        if bool(value):
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


def _parse_config(path: Optional[str]) -> Optional[SanityConfig]:
    if not path:
        return None
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    values = json.loads(config_path.read_text(encoding="utf-8"))
    return SanityConfig.from_dict(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sanity checks on trades CSV.")
    parser.add_argument("trades_csv", help="Path to trades CSV.")
    parser.add_argument("--benchmark-ticker", default=None, help="Benchmark ticker (e.g., SPY).")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--output-name", default=None, help="Output JSON file name.")
    args = parser.parse_args()

    config = _parse_config(args.config)
    report = run_sanity_checks(args.trades_csv, args.benchmark_ticker, config)
    output_name = args.output_name or (config.json_output_name if config else SanityConfig().json_output_name)
    output_path = write_report_json(report, args.trades_csv, output_name)
    print_report(report)
    print(f"\nJSON report written to {output_path}")


if __name__ == "__main__":
    main()
