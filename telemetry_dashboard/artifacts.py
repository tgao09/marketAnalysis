from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from telemetry_dashboard.registry import REPO_ROOT, WorkflowSpec


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _history_item(label: str, path: Path, summary: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": path,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime),
        "summary": summary,
    }


def collect_artifact_snapshot(spec: WorkflowSpec, artifact_paths: list[Path]) -> dict[str, Any]:
    unique_paths = []
    seen = set()
    for raw_path in artifact_paths:
        path = Path(raw_path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)

    snapshot: dict[str, Any] = {"cards": {}, "series": {}, "rows": []}
    if spec.history_kind == "training_metrics":
        metrics_files = []
        for path in unique_paths:
            if path.is_dir():
                candidate = path / "metrics.json"
                if candidate.exists():
                    metrics_files.append(candidate)
            elif path.name == "metrics.json":
                metrics_files.append(path)
        for metrics_path in metrics_files:
            metrics = _read_json(metrics_path) or {}
            summary = metrics.get("summary") or {}
            ticker = metrics_path.parent.parent.name if metrics_path.parent.parent.exists() else metrics_path.parent.name
            snapshot["rows"].append({"label": ticker, "path": metrics_path})
            if summary:
                snapshot["cards"][f"{ticker} folds"] = summary.get("folds")
                if summary.get("mae_mean") is not None:
                    snapshot["cards"][f"{ticker} mae"] = summary.get("mae_mean")
                if summary.get("mse_mean") is not None:
                    snapshot["cards"][f"{ticker} mse"] = summary.get("mse_mean")
                if summary.get("directional_mean") is not None:
                    snapshot["cards"][f"{ticker} dir"] = summary.get("directional_mean")
                if summary.get("coverage_95_mean") is not None:
                    snapshot["cards"][f"{ticker} coverage"] = summary.get("coverage_95_mean")
            folds = metrics.get("folds") or []
            for name in ("mae", "mse", "mae_simple", "directional", "coverage_95"):
                points = [
                    (int(item["fold"]), float(item[name]))
                    for item in folds
                    if item.get(name) is not None
                ]
                if points:
                    snapshot["series"].setdefault(name, points)
    elif spec.history_kind in {"backtest_summary", "hmm_backtest"}:
        summary_path = None
        trades_path = None
        for path in unique_paths:
            if path.suffix == ".json" and "summary" in path.name:
                summary_path = path
            if path.suffix == ".csv" and "trades" in path.name:
                trades_path = path
        if summary_path is not None:
            summary = _read_json(summary_path) or {}
            for key in (
                "ticker",
                "symbols",
                "avg_return_pct",
                "avg_pnl",
                "win_rate",
                "max_drawdown",
                "total_trades",
                "acceptance_pass",
            ):
                if key in summary:
                    snapshot["cards"][key] = summary[key]
            snapshot["rows"].append({"label": "summary", "path": summary_path})
        if trades_path is not None and trades_path.exists():
            trades = pd.read_csv(trades_path)
            if "pnl" in trades.columns:
                cumulative_points = []
                trade_points = []
                cumulative = 0.0
                for idx, pnl in enumerate(trades["pnl"].tolist(), start=1):
                    pnl_value = float(pnl)
                    cumulative += pnl_value
                    trade_points.append((idx, pnl_value))
                    cumulative_points.append((idx, cumulative))
                snapshot["series"]["trade_pnl"] = trade_points
                snapshot["series"]["cumulative_pnl"] = cumulative_points
            snapshot["rows"].append({"label": "trades", "path": trades_path})
    elif spec.history_kind == "optimization_report":
        run_dir = None
        for path in unique_paths:
            if path.is_dir():
                run_dir = path
                break
            if path.name == "final_report.json":
                run_dir = path.parent
                break
        if run_dir is not None:
            final_report = _read_json(run_dir / "final_report.json") or {}
            winner = final_report.get("winner") or {}
            if winner.get("trial_number") is not None:
                snapshot["cards"]["winner_trial"] = winner.get("trial_number")
            holdout = winner.get("holdout") or {}
            aggregate = holdout.get("aggregate") or {}
            if aggregate.get("basket_mean_avg_return_pct") is not None:
                snapshot["cards"]["winner_holdout_avg_return_pct"] = aggregate.get("basket_mean_avg_return_pct")
            top_trials = final_report.get("top_10_trials") or final_report.get("tune_top_10") or []
            objective_points = []
            best_points = []
            best_value = None
            for idx, trial in enumerate(top_trials, start=1):
                trial_number = trial.get("trial_number", idx)
                value = trial.get("objective_value")
                if value is None:
                    value = trial.get("trial_value")
                if value is None:
                    continue
                trial_value = float(value)
                objective_points.append((int(trial_number), trial_value))
                best_value = trial_value if best_value is None else max(best_value, trial_value)
                best_points.append((int(trial_number), best_value))
            if objective_points:
                snapshot["series"]["objective"] = objective_points
                snapshot["series"]["best_objective"] = best_points
            snapshot["rows"].append({"label": "run", "path": run_dir})
    elif spec.history_kind == "hmm_training":
        for path in unique_paths:
            if path.is_dir():
                diagnostics = _read_json(path / "diagnostics.json") or {}
                if diagnostics:
                    snapshot["cards"]["train_start"] = diagnostics.get("train_start")
                    snapshot["cards"]["train_end"] = diagnostics.get("train_end")
                    occupancy = diagnostics.get("occupancy_raw") or []
                    for idx, value in enumerate(occupancy):
                        snapshot["cards"][f"state_{idx}_occupancy"] = value
                    snapshot["rows"].append({"label": "diagnostics", "path": path / "diagnostics.json"})
    return snapshot


def scan_recent_history(spec: WorkflowSpec, limit: int = 8) -> list[dict[str, Any]]:
    patterns: dict[str, str] = {
        "training_gbm_return": "gbm_return/artifacts/*/regular/metrics.json",
        "training_gp_return": "gp_return/artifacts/*/*/metrics.json",
        "training_gp_vol": "gp_vol/artifacts/metrics.json",
        "training_hmm_regime": "hmm_regime/artifacts/market/diagnostics.json",
        "optimization_gbm_return": "gbm_return/artifacts/optimization/*/final_report.json",
        "optimization_gp_return": "gp_return/artifacts/optuna_runs/*/final_report.json",
        "backtesting_gbm_return": "gbm_return/artifacts/*/regular/gbm_return_summary.json",
        "backtesting_gp_return": "gp_return/artifacts/*/*/gp_return_summary.json",
        "backtesting_gp_vol": "gp_vol/artifacts/variance_proxy_summary.json",
        "backtesting_hmm_regime": "hmm_regime/artifacts/market/walk_forward_summary.json",
    }
    pattern = patterns.get(spec.id)
    if pattern is None:
        return []

    items: list[dict[str, Any]] = []
    candidates = sorted(
        REPO_ROOT.glob(pattern),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[:limit]
    for path in candidates:
        if spec.history_kind == "training_metrics":
            payload = _read_json(path) or {}
            summary = payload.get("summary") or {}
            label = path.parent.parent.name if path.parent.parent.exists() else spec.label
            parts = []
            if summary.get("mae_mean") is not None:
                parts.append(f"MAE {summary['mae_mean']:.4f}")
            if summary.get("mse_mean") is not None:
                parts.append(f"MSE {summary['mse_mean']:.4f}")
            items.append(_history_item(label, path, " | ".join(parts) if parts else path.name))
        elif spec.history_kind == "optimization_report":
            payload = _read_json(path) or {}
            winner = payload.get("winner") or {}
            label = path.parent.name
            summary = f"Winner {winner.get('trial_number')}" if winner else "No winner"
            items.append(_history_item(label, path.parent, summary))
        elif spec.history_kind in {"backtest_summary", "hmm_backtest"}:
            payload = _read_json(path) or {}
            label = str(payload.get("ticker") or payload.get("symbols") or path.parent.name)
            if payload.get("avg_return_pct") is not None:
                summary = f"Avg return {payload['avg_return_pct']:.2%}"
            elif payload.get("acceptance_pass") is not None:
                summary = f"Acceptance {payload['acceptance_pass']}"
            else:
                summary = path.name
            items.append(_history_item(label, path, summary))
        elif spec.history_kind == "hmm_training":
            payload = _read_json(path) or {}
            summary = f"{payload.get('train_start')} -> {payload.get('train_end')}"
            items.append(_history_item(spec.label, path, summary))
    return items
