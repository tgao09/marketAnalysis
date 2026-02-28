import argparse
import itertools
import json
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import gbm_return.backtest_walk_forward as gbm_backtest
import gbm_return.feature_ablation as gbm_ablation
import gbm_return.train as gbm_train
from gbm_return.configuration import (
    FEATURE_SET_F0,
    FEATURE_SET_F1,
    FEATURE_SET_F2,
    resolve_lgbm_params,
    write_feature_set_file,
)


DEFAULT_TICKERS = ["AAPL", "NVDA", "AMZN", "KO"]
DEFAULT_BASELINE_END = "2026-02-10"
DEFAULT_TUNE_END = "2025-11-10"
DEFAULT_HOLDOUT_END = "2026-02-10"
DEFAULT_ABLATION_END = "2025-08-10"

COARSE_GRID = {
    "learning_rate": [0.02, 0.05, 0.08],
    "n_estimators": [300, 600, 1000],
    "num_leaves": [15, 31, 63],
    "min_data_in_leaf": [20, 50, 100],
    "feature_fraction": [0.7, 0.9, 1.0],
    "bagging_fraction": [0.7, 0.9, 1.0],
    "lambda_l1": [0.0, 0.5, 2.0],
    "lambda_l2": [0.0, 0.5, 2.0],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize gbm_return using feature ablation + hyperparameter sweep."
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ticker basket for tuning and validation.",
    )
    parser.add_argument("--train-window", default=gbm_train.DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--test-window", default=gbm_train.DEFAULT_TEST_WINDOW)
    parser.add_argument("--step-window", default=gbm_train.DEFAULT_STEP_WINDOW)
    parser.add_argument("--baseline-end", default=DEFAULT_BASELINE_END)
    parser.add_argument("--ablation-end", default=DEFAULT_ABLATION_END)
    parser.add_argument("--tune-end", default=DEFAULT_TUNE_END)
    parser.add_argument("--holdout-end", default=DEFAULT_HOLDOUT_END)
    parser.add_argument("--notional", type=float, default=10000.0)
    parser.add_argument("--drawdown-worsen-limit", type=float, default=0.10)
    parser.add_argument(
        "--include-time-index",
        action="store_true",
        help="Include time_index in feature columns.",
    )
    parser.add_argument("--output-dir", default=str(gbm_train.ARTIFACT_DIR_DEFAULT))
    parser.add_argument(
        "--feature-set-file",
        default=str(gbm_train.ARTIFACT_DIR_DEFAULT / "feature_sets.json"),
        help="Where to write/read F1/F2 feature drops.",
    )
    parser.add_argument(
        "--coarse-max-configs",
        type=int,
        default=None,
        help="Optional cap on coarse sweep candidates (useful for faster experimentation).",
    )
    parser.add_argument("--refine-top-n", type=int, default=10)
    parser.add_argument("--refine-max-configs", type=int, default=500)
    parser.add_argument("--holdout-top-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Skip final artifact retraining with the winning config.",
    )
    parser.add_argument(
        "--skip-tech-update",
        action="store_true",
        help="Skip appending final results to gbm_return/TECH.md.",
    )
    return parser.parse_args()


def parse_tickers(raw: str) -> list[str]:
    tickers = []
    seen = set()
    for token in raw.split(","):
        ticker = token.strip().upper()
        if not ticker:
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def json_ready(value: Any):
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2))


def aggregate_basket_summary(ticker_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def collect(field: str):
        values = [
            float(summary[field])
            for summary in ticker_summaries.values()
            if summary.get(field) is not None
        ]
        return values

    avg_return_values = collect("avg_return_pct")
    win_rate_values = collect("win_rate")
    avg_pnl_values = collect("avg_pnl")
    max_drawdowns = collect("max_drawdown")
    total_trades = sum(int(summary.get("total_trades", 0) or 0) for summary in ticker_summaries.values())

    return {
        "basket_mean_avg_return_pct": float(np.mean(avg_return_values)) if avg_return_values else None,
        "basket_mean_win_rate": float(np.mean(win_rate_values)) if win_rate_values else None,
        "basket_mean_avg_pnl": float(np.mean(avg_pnl_values)) if avg_pnl_values else None,
        "basket_worst_max_drawdown": float(np.min(max_drawdowns)) if max_drawdowns else None,
        "basket_total_trades": int(total_trades),
    }


def drawdown_guardrail_violations(
    candidate_summaries: dict[str, dict[str, Any]],
    baseline_summaries: dict[str, dict[str, Any]],
    worsen_limit: float,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for ticker, baseline in baseline_summaries.items():
        cand = candidate_summaries.get(ticker)
        if cand is None:
            violations.append({"ticker": ticker, "reason": "missing_candidate_summary"})
            continue
        base_dd = baseline.get("max_drawdown")
        cand_dd = cand.get("max_drawdown")
        if base_dd is None or cand_dd is None:
            continue
        base_dd = float(base_dd)
        cand_dd = float(cand_dd)
        if base_dd < 0:
            allowed_floor = base_dd * (1.0 + worsen_limit)
            if cand_dd < allowed_floor:
                violations.append(
                    {
                        "ticker": ticker,
                        "baseline_max_drawdown": base_dd,
                        "candidate_max_drawdown": cand_dd,
                        "allowed_floor": allowed_floor,
                    }
                )
        elif base_dd == 0:
            if cand_dd < 0:
                violations.append(
                    {
                        "ticker": ticker,
                        "baseline_max_drawdown": base_dd,
                        "candidate_max_drawdown": cand_dd,
                        "allowed_floor": 0.0,
                    }
                )
        else:
            allowed_floor = base_dd * (1.0 - worsen_limit)
            if cand_dd < allowed_floor:
                violations.append(
                    {
                        "ticker": ticker,
                        "baseline_max_drawdown": base_dd,
                        "candidate_max_drawdown": cand_dd,
                        "allowed_floor": allowed_floor,
                    }
                )
    return violations


def evaluate_candidate_on_end_date(
    tickers: list[str],
    end_date: pd.Timestamp,
    train_window: str,
    include_time_index: bool,
    feature_set: str,
    feature_set_file: str,
    lgbm_params: dict[str, Any],
    lgbm_param_preset: str,
    lgbm_params_json: str | None,
    notional: float,
    prepared_cache: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    ticker_summaries: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        cache_key = (ticker, str(end_date.date()), train_window)
        if cache_key not in prepared_cache:
            prepared_cache[cache_key] = gbm_backtest.prepare_backtest_data(
                ticker=ticker,
                end_date=end_date,
                train_window=train_window,
            )
        prepared = prepared_cache[cache_key]
        _, summary, _ = gbm_backtest.run_backtest_prepared(
            prepared=prepared,
            notional=notional,
            include_time_index=include_time_index,
            feature_set=feature_set,
            feature_set_file=feature_set_file,
            lgbm_params=lgbm_params,
            verbose=False,
        )
        summary["lgbm_param_preset"] = lgbm_param_preset
        summary["lgbm_params_json"] = lgbm_params_json
        ticker_summaries[ticker] = summary
    aggregate = aggregate_basket_summary(ticker_summaries)
    return {"aggregate": aggregate, "tickers": ticker_summaries}


def run_ablation_for_ticker(
    ticker: str,
    end_date: pd.Timestamp,
    train_window: str,
    test_window: str,
    step_window: str,
    include_time_index: bool,
    output_dir: Path,
    lgbm_params: dict[str, Any],
    lgbm_param_preset: str,
    lgbm_params_json: str | None,
    feature_set_file: str,
) -> dict[str, Any]:
    dataset_start = gbm_ablation.compute_dataset_start(end_date, train_window)
    data = gbm_ablation.build_dataset(ticker, dataset_start, end_date)
    dataset = data["dataset"]
    baseline_feature_cols = gbm_train.select_feature_columns(
        dataset=dataset,
        drop_time_index=not include_time_index,
        feature_set=FEATURE_SET_F0,
        feature_set_file=feature_set_file,
    )
    if not baseline_feature_cols:
        raise ValueError(f"{ticker}: No features available for ablation.")

    all_splits = list(
        gbm_ablation.walk_forward_splits(
            dataset,
            train_window=train_window,
            test_window=test_window,
            embargo=gbm_train.WINDOW_RET,
            step=step_window,
            min_train_rows=gbm_train.MIN_TRAIN_ROWS,
        )
    )
    if not all_splits:
        raise ValueError(f"{ticker}: No walk-forward splits produced for ablation.")

    eval_start, eval_end, selected_splits = gbm_ablation.select_last_6m_folds(
        all_splits,
        end_date,
        test_window,
        step_window,
    )
    baseline_summary, baseline_gain, baseline_split, baseline_shap = gbm_ablation.run_candidate(
        baseline_feature_cols,
        selected_splits,
        lgbm_params,
    )
    baseline_metrics = gbm_ablation.metric_triplet(baseline_summary)

    results = []
    for feature in baseline_feature_cols:
        candidate_feature_cols = [col for col in baseline_feature_cols if col != feature]
        if not candidate_feature_cols:
            results.append(
                {
                    "dropped_feature": feature,
                    "status": "failed",
                    "error": "Cannot drop the only feature.",
                }
            )
            continue
        try:
            candidate_summary, _, _, _ = gbm_ablation.run_candidate(
                candidate_feature_cols,
                selected_splits,
                lgbm_params,
            )
            results.append(
                {
                    "dropped_feature": feature,
                    "status": "ok",
                    "candidate_metrics": gbm_ablation.metric_triplet(candidate_summary),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "dropped_feature": feature,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    leaderboard, per_feature_deltas, failed_diagnostics = gbm_ablation.build_ranked_outputs(
        results,
        baseline_metrics,
    )
    args_ns = SimpleNamespace(
        train_window=train_window,
        test_window=test_window,
        step_window=step_window,
        include_time_index=include_time_index,
        feature_set=FEATURE_SET_F0,
        feature_set_file=feature_set_file,
        lgbm_param_preset=lgbm_param_preset,
        lgbm_params_json=lgbm_params_json,
    )
    ticker_output_dir = output_dir / ticker
    out_path = gbm_ablation.write_ablation_json(
        ticker_output_dir,
        ticker,
        eval_start,
        eval_end,
        args_ns,
        baseline_feature_cols,
        baseline_metrics,
        baseline_gain,
        baseline_split,
        baseline_shap,
        selected_splits,
        leaderboard,
        per_feature_deltas,
        failed_diagnostics,
    )
    return json.loads(out_path.read_text())


def rank_harmful_features(ablation_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    basket_size = len(ablation_payloads)
    for ticker, payload in ablation_payloads.items():
        rows = payload.get("per_feature_deltas", [])
        for row in rows:
            if row.get("status") != "ok":
                continue
            feature = row.get("dropped_feature")
            deltas = row.get("deltas") or {}
            improvements = row.get("improvements") or {}
            mae_delta = deltas.get("mae_simple_delta")
            dir_delta = deltas.get("directional_delta")
            mae_improvement = improvements.get("mae_simple_improvement")
            dir_improvement = improvements.get("directional_improvement")
            if mae_delta is None or dir_delta is None or mae_improvement is None:
                continue
            if not (mae_delta < 0 and dir_delta >= 0):
                continue
            if feature not in stats:
                stats[feature] = {
                    "feature": feature,
                    "support_count": 0,
                    "ticker_hits": [],
                    "mae_simple_improvement_sum": 0.0,
                    "directional_improvement_sum": 0.0,
                }
            stats[feature]["support_count"] += 1
            stats[feature]["ticker_hits"].append(ticker)
            stats[feature]["mae_simple_improvement_sum"] += float(mae_improvement)
            stats[feature]["directional_improvement_sum"] += float(dir_improvement or 0.0)

    ranked = []
    for feature, item in stats.items():
        support_count = int(item["support_count"])
        ranked.append(
            {
                "feature": feature,
                "support_count": support_count,
                "support_ratio": support_count / basket_size if basket_size else 0.0,
                "ticker_hits": item["ticker_hits"],
                "avg_mae_simple_improvement": item["mae_simple_improvement_sum"] / support_count,
                "avg_directional_improvement": item["directional_improvement_sum"] / support_count,
            }
        )
    ranked.sort(
        key=lambda x: (
            -x["support_count"],
            -x["avg_mae_simple_improvement"],
            -x["avg_directional_improvement"],
            x["feature"],
        )
    )
    return ranked


def choose_feature_sets(
    harmful_ranked: list[dict[str, Any]],
    basket_size: int,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    support_min = max(2, math.ceil(0.5 * basket_size))
    eligible = [item for item in harmful_ranked if item["support_count"] >= support_min]
    selected_pool = eligible if len(eligible) >= 5 else harmful_ranked
    f1 = [item["feature"] for item in selected_pool[:3]]
    f2 = [item["feature"] for item in selected_pool[:5]]
    return f1, f2, selected_pool


def generate_coarse_configs() -> list[dict[str, Any]]:
    keys = list(COARSE_GRID.keys())
    values = [COARSE_GRID[key] for key in keys]
    configs = []
    for combo in itertools.product(*values):
        config = {key: combo[idx] for idx, key in enumerate(keys)}
        configs.append(config)
    return configs


def maybe_cap_configs(
    configs: list[dict[str, Any]],
    max_configs: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_configs is None or max_configs <= 0 or max_configs >= len(configs):
        return configs
    rng = random.Random(seed)
    sample_idx = sorted(rng.sample(range(len(configs)), max_configs))
    return [configs[idx] for idx in sample_idx]


def candidate_key(feature_set: str, params: dict[str, Any]) -> str:
    return json.dumps({"feature_set": feature_set, "params": params}, sort_keys=True)


def evaluate_candidate_records(
    records: list[dict[str, Any]],
    tickers: list[str],
    end_date: pd.Timestamp,
    train_window: str,
    include_time_index: bool,
    feature_set_file: str,
    notional: float,
    baseline_ticker_summaries: dict[str, dict[str, Any]],
    drawdown_worsen_limit: float,
    prepared_cache: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluated = []
    for idx, record in enumerate(records, start=1):
        params = record["params"]
        result = evaluate_candidate_on_end_date(
            tickers=tickers,
            end_date=end_date,
            train_window=train_window,
            include_time_index=include_time_index,
            feature_set=record["feature_set"],
            feature_set_file=feature_set_file,
            lgbm_params=params,
            lgbm_param_preset=record.get("lgbm_param_preset", "baseline"),
            lgbm_params_json=record.get("lgbm_params_json"),
            notional=notional,
            prepared_cache=prepared_cache,
        )
        violations = drawdown_guardrail_violations(
            candidate_summaries=result["tickers"],
            baseline_summaries=baseline_ticker_summaries,
            worsen_limit=drawdown_worsen_limit,
        )
        evaluated_record = dict(record)
        evaluated_record["index"] = idx
        evaluated_record["aggregate"] = result["aggregate"]
        evaluated_record["tickers"] = result["tickers"]
        evaluated_record["guardrail_violations"] = violations
        evaluated_record["guardrail_pass"] = len(violations) == 0
        evaluated.append(evaluated_record)
    return evaluated


def ranking_value(record: dict[str, Any]) -> tuple[float, float]:
    guardrail = 1.0 if record.get("guardrail_pass") else 0.0
    mean_ret = record.get("aggregate", {}).get("basket_mean_avg_return_pct")
    if mean_ret is None:
        mean_ret = float("-inf")
    return (guardrail, float(mean_ret))


def sort_by_rank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=ranking_value, reverse=True)


def parameter_neighbors(param: str, value: Any) -> list[Any]:
    if param == "learning_rate":
        vals = [max(0.005, round(float(value) - 0.01, 4)), round(float(value), 4), round(float(value) + 0.01, 4)]
        return sorted(set(vals))
    if param == "n_estimators":
        vals = [max(100, int(value) - 200), int(value), int(value) + 200]
        return sorted(set(vals))
    if param == "num_leaves":
        vals = [max(5, int(value) - 8), int(value), int(value) + 8]
        return sorted(set(vals))
    if param == "min_data_in_leaf":
        vals = [max(5, int(value) - 10), int(value), int(value) + 10]
        return sorted(set(vals))
    if param in {"feature_fraction", "bagging_fraction"}:
        vals = [
            max(0.5, round(float(value) - 0.1, 3)),
            round(float(value), 3),
            min(1.0, round(float(value) + 0.1, 3)),
        ]
        return sorted(set(vals))
    if param in {"lambda_l1", "lambda_l2"}:
        vals = [max(0.0, round(float(value) - 0.5, 3)), round(float(value), 3), round(float(value) + 0.5, 3)]
        return sorted(set(vals))
    return [value]


def build_refine_candidates(
    top_records: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    dedup: dict[str, dict[str, Any]] = {}
    for record in top_records:
        base_params = record["params"]
        feature_set = record["feature_set"]
        base_key = candidate_key(feature_set, base_params)
        dedup[base_key] = {
            "stage": "refine",
            "feature_set": feature_set,
            "params": dict(base_params),
            "origin": "top_base",
        }
        for param in COARSE_GRID.keys():
            for value in parameter_neighbors(param, base_params[param]):
                if value == base_params[param]:
                    continue
                varied = dict(base_params)
                varied[param] = value
                key = candidate_key(feature_set, varied)
                dedup[key] = {
                    "stage": "refine",
                    "feature_set": feature_set,
                    "params": varied,
                    "origin": f"{record.get('stage', 'coarse')}:{param}",
                }
    candidates = list(dedup.values())
    if max_candidates > 0 and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    return candidates


def collect_gp_reference(tickers: list[str]) -> dict[str, Any]:
    gp = {}
    for ticker in tickers:
        path = ROOT_DIR / "gp_return" / "artifacts" / ticker / "regular" / "gp_return_summary.json"
        if path.exists():
            gp[ticker] = json.loads(path.read_text())
    return gp


def validate_evaluation_dates(
    baseline_end: pd.Timestamp,
    ablation_end: pd.Timestamp,
    tune_end: pd.Timestamp,
    holdout_end: pd.Timestamp,
):
    if not (ablation_end < tune_end < holdout_end):
        raise ValueError(
            "Date order must be strict: ablation_end < tune_end < holdout_end. "
            f"Got ablation_end={ablation_end.date()}, tune_end={tune_end.date()}, holdout_end={holdout_end.date()}."
        )
    if baseline_end != holdout_end:
        raise ValueError(
            "baseline_end must match holdout_end so baseline and finalists are compared on the same holdout date. "
            f"Got baseline_end={baseline_end.date()}, holdout_end={holdout_end.date()}."
        )


def append_tech_results(
    tech_path: Path,
    report: dict[str, Any],
):
    winner = report["winner"]
    holdout = winner["holdout"]
    summary = holdout["aggregate"]
    run_id = report["run_id"]
    ts = report["generated_at"]
    section = [
        "",
        f"## Optimization Run {run_id}",
        "",
        f"- Generated: `{ts}`",
        f"- Winner feature set: `{winner['feature_set']}`",
        f"- Winner params: `{json.dumps(winner['params'], sort_keys=True)}`",
        f"- Winner tuning basket mean avg_return_pct: `{winner['tune']['aggregate']['basket_mean_avg_return_pct']}`",
        f"- Winner holdout basket mean avg_return_pct: `{summary['basket_mean_avg_return_pct']}`",
        f"- Winner holdout basket worst max_drawdown: `{summary['basket_worst_max_drawdown']}`",
        f"- Baseline holdout basket mean avg_return_pct: `{report['baseline_holdout']['aggregate']['basket_mean_avg_return_pct']}`",
        f"- F1 drops: `{', '.join(report['feature_sets']['F1']) if report['feature_sets']['F1'] else '(none)'}`",
        f"- F2 drops: `{', '.join(report['feature_sets']['F2']) if report['feature_sets']['F2'] else '(none)'}`",
    ]
    existing = tech_path.read_text()
    tech_path.write_text(existing.rstrip() + "\n" + "\n".join(section) + "\n")


def retrain_winner(
    tickers: list[str],
    winner: dict[str, Any],
    feature_set_file: str,
    train_window: str,
    test_window: str,
    step_window: str,
    include_time_index: bool,
) -> dict[str, Any]:
    config = {
        "data_years": gbm_train.DATA_YEARS,
        "window_ret": gbm_train.WINDOW_RET,
        "train_window": train_window,
        "test_window": test_window,
        "step_window": step_window,
        "artifact_dir": str(gbm_train.ARTIFACT_DIR_DEFAULT),
        "drop_time_index": not include_time_index,
        "feature_set": winner["feature_set"],
        "feature_set_file": feature_set_file,
        "lgbm_param_preset": "baseline",
        "lgbm_params_json": None,
        "lgbm_params": winner["params"],
        "regime_score": {
            "enabled": True,
            "score_window": gbm_train.REGIME_SCORE_WINDOW,
            "score_clip": gbm_train.REGIME_SCORE_CLIP,
            "weights": gbm_train.REGIME_SCORE_WEIGHTS,
        },
    }
    history_cache: dict[str, Any] = {}
    summaries = {}
    for ticker in tickers:
        summaries[ticker] = gbm_train.train_for_ticker(ticker, config, history_cache)
    return summaries


def main():
    args = parse_args()
    tickers = parse_tickers(args.tickers)

    random.seed(args.seed)
    np.random.seed(args.seed)

    baseline_end = pd.Timestamp(args.baseline_end).normalize()
    ablation_end = pd.Timestamp(args.ablation_end).normalize()
    tune_end = pd.Timestamp(args.tune_end).normalize()
    holdout_end = pd.Timestamp(args.holdout_end).normalize()
    validate_evaluation_dates(
        baseline_end=baseline_end,
        ablation_end=ablation_end,
        tune_end=tune_end,
        holdout_end=holdout_end,
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_dir)
    run_dir = output_root / "optimization" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_set_file = str(Path(args.feature_set_file))

    print(f"Optimization run: {run_id}")
    print(f"Tickers: {', '.join(tickers)}")
    print("Running baseline backtests...")

    prepared_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    baseline_params = resolve_lgbm_params("baseline")
    baseline_holdout = evaluate_candidate_on_end_date(
        tickers=tickers,
        end_date=baseline_end,
        train_window=args.train_window,
        include_time_index=args.include_time_index,
        feature_set=FEATURE_SET_F0,
        feature_set_file=feature_set_file,
        lgbm_params=baseline_params,
        lgbm_param_preset="baseline",
        lgbm_params_json=None,
        notional=args.notional,
        prepared_cache=prepared_cache,
    )
    baseline_tune = evaluate_candidate_on_end_date(
        tickers=tickers,
        end_date=tune_end,
        train_window=args.train_window,
        include_time_index=args.include_time_index,
        feature_set=FEATURE_SET_F0,
        feature_set_file=feature_set_file,
        lgbm_params=baseline_params,
        lgbm_param_preset="baseline",
        lgbm_params_json=None,
        notional=args.notional,
        prepared_cache=prepared_cache,
    )
    write_json(run_dir / "baseline_holdout.json", baseline_holdout)
    write_json(run_dir / "baseline_tune.json", baseline_tune)

    print("Running basket feature ablation...")
    ablation_output_dir = output_root
    ablation_payloads = {}
    for ticker in tickers:
        print(f"  Ablation: {ticker}")
        ablation_payloads[ticker] = run_ablation_for_ticker(
            ticker=ticker,
            end_date=ablation_end,
            train_window=args.train_window,
            test_window=args.test_window,
            step_window=args.step_window,
            include_time_index=args.include_time_index,
            output_dir=ablation_output_dir,
            lgbm_params=baseline_params,
            lgbm_param_preset="baseline",
            lgbm_params_json=None,
            feature_set_file=feature_set_file,
        )
    harmful_ranked = rank_harmful_features(ablation_payloads)
    f1_drops, f2_drops, selected_pool = choose_feature_sets(harmful_ranked, len(tickers))
    feature_set_path = write_feature_set_file(
        output_path=feature_set_file,
        f1_drop_features=f1_drops,
        f2_drop_features=f2_drops,
        metadata={
            "run_id": run_id,
            "tickers": tickers,
            "ablation_end": str(ablation_end.date()),
            "selection_pool_size": len(selected_pool),
        },
    )
    ablation_summary = {
        "ablation_end": str(ablation_end.date()),
        "tickers": tickers,
        "harmful_ranked": harmful_ranked,
        "feature_set_file": str(feature_set_path),
        "feature_sets": {"F1": f1_drops, "F2": f2_drops},
    }
    write_json(run_dir / "ablation_summary.json", ablation_summary)

    print("Building coarse hyperparameter candidates...")
    coarse_grid = generate_coarse_configs()
    coarse_grid = maybe_cap_configs(coarse_grid, args.coarse_max_configs, args.seed)
    coarse_records = []
    for feature_set in (FEATURE_SET_F0, FEATURE_SET_F1, FEATURE_SET_F2):
        for params in coarse_grid:
            coarse_records.append(
                {
                    "stage": "coarse",
                    "feature_set": feature_set,
                    "params": params,
                    "lgbm_param_preset": "baseline",
                    "lgbm_params_json": None,
                }
            )
    print(f"Evaluating coarse candidates: {len(coarse_records)}")
    coarse_evaluated = evaluate_candidate_records(
        records=coarse_records,
        tickers=tickers,
        end_date=tune_end,
        train_window=args.train_window,
        include_time_index=args.include_time_index,
        feature_set_file=feature_set_file,
        notional=args.notional,
        baseline_ticker_summaries=baseline_tune["tickers"],
        drawdown_worsen_limit=args.drawdown_worsen_limit,
        prepared_cache=prepared_cache,
    )
    coarse_ranked = sort_by_rank(coarse_evaluated)
    write_json(
        run_dir / "coarse_results.json",
        {
            "candidate_count": len(coarse_ranked),
            "top_50": coarse_ranked[:50],
        },
    )

    top_for_refine = coarse_ranked[: max(1, args.refine_top_n)]
    print(f"Building refine candidates from top {len(top_for_refine)} coarse configs...")
    refine_records = build_refine_candidates(top_for_refine, args.refine_max_configs)
    print(f"Evaluating refine candidates: {len(refine_records)}")
    refine_evaluated = evaluate_candidate_records(
        records=refine_records,
        tickers=tickers,
        end_date=tune_end,
        train_window=args.train_window,
        include_time_index=args.include_time_index,
        feature_set_file=feature_set_file,
        notional=args.notional,
        baseline_ticker_summaries=baseline_tune["tickers"],
        drawdown_worsen_limit=args.drawdown_worsen_limit,
        prepared_cache=prepared_cache,
    )
    refine_ranked = sort_by_rank(refine_evaluated)
    write_json(
        run_dir / "refine_results.json",
        {
            "candidate_count": len(refine_ranked),
            "top_50": refine_ranked[:50],
        },
    )

    finalists = [r for r in refine_ranked if r.get("guardrail_pass")] or refine_ranked
    finalists = finalists[: max(1, args.holdout_top_n)]
    print(f"Validating finalists on holdout date: {holdout_end.date()} ({len(finalists)} configs)")

    holdout_validations = []
    promoted = None
    baseline_holdout_mean = baseline_holdout["aggregate"]["basket_mean_avg_return_pct"]
    for record in finalists:
        holdout_eval = evaluate_candidate_on_end_date(
            tickers=tickers,
            end_date=holdout_end,
            train_window=args.train_window,
            include_time_index=args.include_time_index,
            feature_set=record["feature_set"],
            feature_set_file=feature_set_file,
            lgbm_params=record["params"],
            lgbm_param_preset=record.get("lgbm_param_preset", "baseline"),
            lgbm_params_json=record.get("lgbm_params_json"),
            notional=args.notional,
            prepared_cache=prepared_cache,
        )
        holdout_violations = drawdown_guardrail_violations(
            candidate_summaries=holdout_eval["tickers"],
            baseline_summaries=baseline_holdout["tickers"],
            worsen_limit=args.drawdown_worsen_limit,
        )
        holdout_pass = len(holdout_violations) == 0
        holdout_mean = holdout_eval["aggregate"]["basket_mean_avg_return_pct"]
        beats_baseline = (
            holdout_pass
            and holdout_mean is not None
            and baseline_holdout_mean is not None
            and holdout_mean > baseline_holdout_mean
        )
        item = {
            "candidate": record,
            "holdout": holdout_eval,
            "holdout_guardrail_pass": holdout_pass,
            "holdout_guardrail_violations": holdout_violations,
            "beats_holdout_baseline": beats_baseline,
        }
        holdout_validations.append(item)
        if beats_baseline:
            promoted = item

    if promoted is None:
        best_item = max(
            holdout_validations,
            key=lambda x: (
                1 if x["holdout_guardrail_pass"] else 0,
                x["holdout"]["aggregate"]["basket_mean_avg_return_pct"]
                if x["holdout"]["aggregate"]["basket_mean_avg_return_pct"] is not None
                else float("-inf"),
            ),
        )
        promoted = best_item

    winner_candidate = promoted["candidate"]
    winner = {
        "feature_set": winner_candidate["feature_set"],
        "params": winner_candidate["params"],
        "tune": {
            "aggregate": winner_candidate["aggregate"],
            "tickers": winner_candidate["tickers"],
            "guardrail_pass": winner_candidate["guardrail_pass"],
            "guardrail_violations": winner_candidate["guardrail_violations"],
        },
        "holdout": promoted["holdout"],
        "holdout_guardrail_pass": promoted["holdout_guardrail_pass"],
        "holdout_guardrail_violations": promoted["holdout_guardrail_violations"],
        "beats_holdout_baseline": promoted["beats_holdout_baseline"],
    }

    retrain_summaries = None
    if not args.skip_retrain:
        print("Retraining winner artifacts...")
        retrain_summaries = retrain_winner(
            tickers=tickers,
            winner=winner,
            feature_set_file=feature_set_file,
            train_window=args.train_window,
            test_window=args.test_window,
            step_window=args.step_window,
            include_time_index=args.include_time_index,
        )

    optimized_backtests = {}
    optimized_output_dir = run_dir / "optimized_backtests"
    for ticker in tickers:
        backtest_result = gbm_backtest.run_backtest(
            ticker=ticker,
            end_date=holdout_end,
            train_window=args.train_window,
            notional=args.notional,
            include_time_index=args.include_time_index,
            feature_set=winner["feature_set"],
            feature_set_file=feature_set_file,
            lgbm_params=winner["params"],
            output_dir=optimized_output_dir,
            lgbm_param_preset="baseline",
            lgbm_params_json=None,
            write_outputs=True,
            verbose=False,
        )
        optimized_backtests[ticker] = backtest_result["summary"]

    gp_reference = collect_gp_reference(tickers)

    report = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "tickers": tickers,
        "dates": {
            "baseline_end": str(baseline_end.date()),
            "ablation_end": str(ablation_end.date()),
            "tune_end": str(tune_end.date()),
            "holdout_end": str(holdout_end.date()),
        },
        "feature_set_file": feature_set_file,
        "feature_sets": {"F0": [], "F1": f1_drops, "F2": f2_drops},
        "baseline_holdout": baseline_holdout,
        "baseline_tune": baseline_tune,
        "winner": winner,
        "coarse_top_10": coarse_ranked[:10],
        "refine_top_10": refine_ranked[:10],
        "holdout_validation_top": holdout_validations[:10],
        "optimized_backtests": optimized_backtests,
        "gp_reference": gp_reference,
        "retrain_summaries": retrain_summaries,
    }
    write_json(run_dir / "final_report.json", report)

    if not args.skip_tech_update:
        tech_path = ROOT_DIR / "gbm_return" / "TECH.md"
        append_tech_results(tech_path, report)
        print(f"Updated TECH.md: {tech_path}")

    print("\nOptimization complete.")
    print(f"Run directory: {run_dir}")
    print(f"Winner feature set: {winner['feature_set']}")
    print(f"Winner holdout mean avg_return_pct: {winner['holdout']['aggregate']['basket_mean_avg_return_pct']}")


if __name__ == "__main__":
    main()
