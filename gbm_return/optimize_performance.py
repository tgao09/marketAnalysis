import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

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
DEFAULT_OPTUNA_STUDY = "gbm_return_optuna"
DEFAULT_HOLDOUT_TOP_N = 15
DEFAULT_MIN_BASKET_TRADES = 80
DEFAULT_N_TRIALS = 300
HARD_REJECT_SCORE = -1.0e12
OBJECTIVE_TSTAT_WEIGHT = 0.001


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize gbm_return using Optuna + feature ablation."
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ticker basket for tuning and validation.",
    )
    parser.add_argument("--train-window", default=gbm_train.DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--test-window", default=gbm_train.DEFAULT_TEST_WINDOW)
    parser.add_argument("--step-window", default=gbm_train.DEFAULT_STEP_WINDOW)
    parser.add_argument(
        "--baseline-end",
        default=None,
        help="Optional override (YYYY-MM-DD). Defaults to auto holdout end.",
    )
    parser.add_argument(
        "--ablation-end",
        default=None,
        help="Optional override (YYYY-MM-DD). Defaults to holdout_end - 6 months.",
    )
    parser.add_argument(
        "--tune-end",
        default=None,
        help="Optional override (YYYY-MM-DD). Defaults to holdout_end - 3 months.",
    )
    parser.add_argument(
        "--holdout-end",
        default=None,
        help="Optional override (YYYY-MM-DD). Defaults to latest available trading day.",
    )
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
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--study-name", default=DEFAULT_OPTUNA_STUDY)
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URI or sqlite file path. Defaults to run_dir/optuna_study.db.",
    )
    parser.add_argument("--min-basket-trades", type=int, default=DEFAULT_MIN_BASKET_TRADES)
    parser.add_argument("--holdout-top-n", type=int, default=DEFAULT_HOLDOUT_TOP_N)
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
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def json_ready(value: Any):
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
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
        return [
            float(summary[field])
            for summary in ticker_summaries.values()
            if summary.get(field) is not None
        ]

    avg_return_values = collect("avg_return_pct")
    win_rate_values = collect("win_rate")
    avg_pnl_values = collect("avg_pnl")
    max_drawdowns = collect("max_drawdown")
    return_tstats = collect("return_tstat")
    trade_rates = collect("trade_rate")
    total_trades = sum(
        int(summary.get("total_trades", 0) or 0) for summary in ticker_summaries.values()
    )
    aggregate = {
        "basket_mean_avg_return_pct": float(np.mean(avg_return_values)) if avg_return_values else None,
        "basket_mean_win_rate": float(np.mean(win_rate_values)) if win_rate_values else None,
        "basket_mean_avg_pnl": float(np.mean(avg_pnl_values)) if avg_pnl_values else None,
        "basket_worst_max_drawdown": float(np.min(max_drawdowns)) if max_drawdowns else None,
        "basket_mean_return_tstat": float(np.mean(return_tstats)) if return_tstats else None,
        "basket_mean_trade_rate": float(np.mean(trade_rates)) if trade_rates else None,
        "basket_total_trades": int(total_trades),
    }
    aggregate["basket_objective_score"] = compute_objective_score(aggregate)
    return aggregate


def compute_objective_score(aggregate: dict[str, Any]) -> float | None:
    mean_ret = aggregate.get("basket_mean_avg_return_pct")
    if mean_ret is None or not np.isfinite(float(mean_ret)):
        return None
    mean_tstat = aggregate.get("basket_mean_return_tstat")
    tstat_bonus = 0.0
    if mean_tstat is not None and np.isfinite(float(mean_tstat)):
        tstat_bonus = OBJECTIVE_TSTAT_WEIGHT * float(mean_tstat)
    return float(mean_ret) + tstat_bonus


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


def assess_candidate(
    result: dict[str, Any],
    baseline_ticker_summaries: dict[str, dict[str, Any]],
    drawdown_worsen_limit: float,
    min_basket_trades: int,
) -> dict[str, Any]:
    violations = drawdown_guardrail_violations(
        candidate_summaries=result["tickers"],
        baseline_summaries=baseline_ticker_summaries,
        worsen_limit=drawdown_worsen_limit,
    )
    guardrail_pass = len(violations) == 0
    total_trades = int(result["aggregate"].get("basket_total_trades") or 0)
    min_trades_pass = total_trades >= min_basket_trades
    objective_score = result["aggregate"].get("basket_objective_score")
    has_objective = objective_score is not None and np.isfinite(float(objective_score))
    hard_reject = (not guardrail_pass) or (not min_trades_pass) or (not has_objective)
    return {
        "guardrail_pass": guardrail_pass,
        "guardrail_violations": violations,
        "min_trades_pass": min_trades_pass,
        "min_basket_trades": int(min_basket_trades),
        "basket_total_trades": total_trades,
        "basket_objective_score": objective_score,
        "hard_reject": hard_reject,
    }


def evaluate_candidate_on_end_date(
    tickers: list[str],
    end_date: pd.Timestamp,
    train_window: str,
    include_time_index: bool,
    feature_set: str,
    feature_set_file: str,
    lgbm_params: dict[str, Any],
    training_policy: dict[str, Any],
    direction_mode: str,
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
            training_policy=training_policy,
            direction_mode=direction_mode,
            verbose=False,
        )
        summary["training_policy"] = training_policy
        summary["direction_mode"] = direction_mode
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
            improvements = row.get("improvements") or {}
            mae_improvement = improvements.get("mae_simple_improvement")
            dir_improvement = improvements.get("directional_improvement")
            if feature is None or mae_improvement is None or dir_improvement is None:
                continue
            if feature not in stats:
                stats[feature] = {
                    "feature": feature,
                    "support_count": 0,
                    "ticker_hits": [],
                    "mae_simple_improvements": [],
                    "directional_improvements": [],
                }
            item = stats[feature]
            item["support_count"] += 1
            item["ticker_hits"].append(ticker)
            item["mae_simple_improvements"].append(float(mae_improvement))
            item["directional_improvements"].append(float(dir_improvement))

    ranked: list[dict[str, Any]] = []
    for feature, item in stats.items():
        support_count = int(item["support_count"])
        mae_values = item["mae_simple_improvements"]
        dir_values = item["directional_improvements"]
        ranked.append(
            {
                "feature": feature,
                "support_count": support_count,
                "support_ratio": support_count / basket_size if basket_size else 0.0,
                "ticker_hits": item["ticker_hits"],
                "median_mae_simple_improvement": float(np.median(mae_values)),
                "median_directional_improvement": float(np.median(dir_values)),
                "avg_mae_simple_improvement": float(np.mean(mae_values)),
                "avg_directional_improvement": float(np.mean(dir_values)),
            }
        )
    ranked.sort(
        key=lambda x: (
            -x["support_count"],
            -x["median_mae_simple_improvement"],
            -x["median_directional_improvement"],
            x["feature"],
        )
    )
    return ranked


def choose_feature_sets(
    harmful_ranked: list[dict[str, Any]],
    basket_size: int,
) -> tuple[list[str], list[str], list[dict[str, Any]], int]:
    support_min = max(2, math.ceil(0.5 * basket_size))
    stable = [
        item
        for item in harmful_ranked
        if (
            item["support_count"] >= support_min
            and item["median_mae_simple_improvement"] > 0.0
            and item["median_directional_improvement"] >= 0.0
        )
    ]
    f1 = [item["feature"] for item in stable[:3]]
    f2 = [item["feature"] for item in stable[:5]]
    return f1, f2, stable, support_min


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
    return


def latest_available_holdout_end(tickers: list[str], train_window: str) -> pd.Timestamp:
    anchor = pd.Timestamp.today().normalize()
    latest: pd.Timestamp | None = None
    for ticker in tickers:
        prepared = gbm_backtest.prepare_backtest_data(
            ticker=ticker,
            end_date=anchor,
            train_window=train_window,
        )
        dataset_index = prepared.get("dataset_index")
        if dataset_index is None or len(dataset_index) == 0:
            continue
        ticker_latest = pd.Timestamp(dataset_index.max()).normalize()
        if latest is None or ticker_latest < latest:
            latest = ticker_latest
    return latest if latest is not None else anchor


def resolve_evaluation_dates(
    args: argparse.Namespace,
    tickers: list[str],
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    auto_holdout_end = latest_available_holdout_end(tickers, args.train_window)
    holdout_end = (
        pd.Timestamp(args.holdout_end).normalize()
        if args.holdout_end
        else auto_holdout_end
    )
    baseline_end = (
        pd.Timestamp(args.baseline_end).normalize()
        if args.baseline_end
        else holdout_end
    )
    tune_end = (
        pd.Timestamp(args.tune_end).normalize()
        if args.tune_end
        else holdout_end - pd.DateOffset(months=3)
    )
    ablation_end = (
        pd.Timestamp(args.ablation_end).normalize()
        if args.ablation_end
        else holdout_end - pd.DateOffset(months=6)
    )
    return baseline_end, ablation_end, tune_end, holdout_end


def resolve_storage_uri(storage_arg: str | None, run_dir: Path) -> tuple[str, str]:
    if storage_arg:
        if "://" in storage_arg:
            return storage_arg, storage_arg
        storage_path = Path(storage_arg).resolve()
    else:
        storage_path = (run_dir / "optuna_study.db").resolve()
    storage_uri = f"sqlite:///{storage_path.as_posix()}"
    return storage_uri, str(storage_path)


def trial_params_from_optuna_params(
    params: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    feature_set = str(params.get("feature_set", FEATURE_SET_F0))
    lgbm_overrides = {
        "learning_rate": float(params["learning_rate"]),
        "n_estimators": int(params["n_estimators"]),
        "num_leaves": int(params["num_leaves"]),
        "min_data_in_leaf": int(params["min_data_in_leaf"]),
        "feature_fraction": float(params["feature_fraction"]),
        "bagging_fraction": float(params["bagging_fraction"]),
        "lambda_l1": float(params["lambda_l1"]),
        "lambda_l2": float(params["lambda_l2"]),
    }
    resolved_lgbm = resolve_lgbm_params("baseline", overrides=lgbm_overrides)
    clip_upper = float(params["target_clip_upper_quantile"])
    training_policy = gbm_train.resolve_training_policy(
        {
            "target_clip_lower_quantile": 1.0 - clip_upper,
            "target_clip_upper_quantile": clip_upper,
            "recency_min_weight": float(params["recency_min_weight"]),
        }
    )
    direction_mode = gbm_train.resolve_direction_mode(str(params["direction_mode"]))
    return feature_set, lgbm_overrides, resolved_lgbm, training_policy, direction_mode


def trial_record_for_report(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_number": record.get("trial_number"),
        "objective_value": record.get("objective_value"),
        "feature_set": record.get("feature_set"),
        "params": record.get("params"),
        "training_policy": record.get("training_policy"),
        "direction_mode": record.get("direction_mode"),
        "aggregate": record.get("aggregate"),
        "assessment": record.get("assessment"),
        "error": record.get("error"),
    }


def append_tech_results(
    tech_path: Path,
    report: dict[str, Any],
):
    winner = report["winner"]
    if not winner or not winner.get("holdout"):
        return
    holdout = winner["holdout"]
    summary = holdout["aggregate"]
    run_id = report["run_id"]
    ts = report["generated_at"]
    section = [
        "",
        f"## Optimization Run {run_id}",
        "",
        f"- Generated: `{ts}`",
        f"- Search backend: `{report['search_backend']}`",
        f"- Winner feature set: `{winner['feature_set']}`",
        f"- Winner params: `{json.dumps(winner['params'], sort_keys=True)}`",
        f"- Winner training policy: `{json.dumps(winner['training_policy'], sort_keys=True)}`",
        f"- Winner direction mode: `{winner['direction_mode']}`",
        f"- Winner tuning basket mean avg_return_pct: `{winner['tune']['aggregate']['basket_mean_avg_return_pct']}`",
        f"- Winner tuning objective score: `{winner['tune']['aggregate']['basket_objective_score']}`",
        f"- Winner holdout basket mean avg_return_pct: `{summary['basket_mean_avg_return_pct']}`",
        f"- Winner holdout objective score: `{summary['basket_objective_score']}`",
        f"- Winner holdout basket worst max_drawdown: `{summary['basket_worst_max_drawdown']}`",
        f"- Baseline holdout basket mean avg_return_pct: `{report['baseline_holdout']['aggregate']['basket_mean_avg_return_pct']}`",
        f"- Baseline holdout objective score: `{report['baseline_holdout']['aggregate']['basket_objective_score']}`",
        f"- F1 drops: `{', '.join(report['feature_sets']['F1']) if report['feature_sets']['F1'] else '(none)'}`",
        f"- F2 drops: `{', '.join(report['feature_sets']['F2']) if report['feature_sets']['F2'] else '(none)'}`",
    ]
    if tech_path.exists():
        existing = tech_path.read_text().rstrip()
    else:
        existing = "# GBM Return Technical Notes"
    tech_path.write_text(existing + "\n" + "\n".join(section) + "\n")


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
        "training_policy": winner["training_policy"],
        "direction_mode": winner["direction_mode"],
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
    tickers = parse_tickers(args.tickers) or list(DEFAULT_TICKERS)

    np.random.seed(args.seed)

    baseline_end, ablation_end, tune_end, holdout_end = resolve_evaluation_dates(args, tickers)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_dir)
    run_dir = output_root / "optimization" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_set_file = str(Path(args.feature_set_file))
    storage_uri, storage_path = resolve_storage_uri(args.storage, run_dir)
    study_name = f"{args.study_name}_{run_id}"

    print(f"Optimization run: {run_id}")
    print(f"Tickers: {', '.join(tickers)}")
    print(
        f"Dates | baseline={baseline_end.date()} | ablation={ablation_end.date()} | "
        f"tune={tune_end.date()} | holdout={holdout_end.date()}"
    )
    print("Running baseline backtests...")

    prepared_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    baseline_params = resolve_lgbm_params("baseline")
    baseline_training_policy = gbm_train.resolve_training_policy()
    baseline_direction_mode = gbm_train.resolve_direction_mode()
    baseline_holdout = evaluate_candidate_on_end_date(
        tickers=tickers,
        end_date=baseline_end,
        train_window=args.train_window,
        include_time_index=args.include_time_index,
        feature_set=FEATURE_SET_F0,
        feature_set_file=feature_set_file,
        lgbm_params=baseline_params,
        training_policy=baseline_training_policy,
        direction_mode=baseline_direction_mode,
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
        training_policy=baseline_training_policy,
        direction_mode=baseline_direction_mode,
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
    f1_drops, f2_drops, stable_pool, support_min = choose_feature_sets(harmful_ranked, len(tickers))
    feature_set_path = write_feature_set_file(
        output_path=feature_set_file,
        f1_drop_features=f1_drops,
        f2_drop_features=f2_drops,
        metadata={
            "run_id": run_id,
            "tickers": tickers,
            "ablation_end": str(ablation_end.date()),
            "stability_support_min": support_min,
            "stable_pool_size": len(stable_pool),
        },
    )
    ablation_summary = {
        "ablation_end": str(ablation_end.date()),
        "tickers": tickers,
        "harmful_ranked": harmful_ranked,
        "feature_set_file": str(feature_set_path),
        "feature_sets": {"F1": f1_drops, "F2": f2_drops},
        "stability_filter": {
            "support_count_min": support_min,
            "median_mae_simple_improvement_gt": 0.0,
            "median_directional_improvement_gte": 0.0,
            "selected_count": len(stable_pool),
            "selected_features": [item["feature"] for item in stable_pool],
        },
    }
    write_json(run_dir / "ablation_summary.json", ablation_summary)

    print(f"Starting Optuna search with {args.n_trials} trials...")
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage_uri,
        load_if_exists=True,
    )

    tune_trial_records: dict[int, dict[str, Any]] = {}

    def objective(trial):
        feature_set = trial.suggest_categorical(
            "feature_set",
            [FEATURE_SET_F0, FEATURE_SET_F1, FEATURE_SET_F2],
        )
        lgbm_overrides = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1400, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 120),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0, step=0.05),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0, step=0.05),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        }
        lgbm_params = resolve_lgbm_params("baseline", overrides=lgbm_overrides)
        clip_upper = trial.suggest_float("target_clip_upper_quantile", 0.96, 0.995, step=0.005)
        training_policy = gbm_train.resolve_training_policy(
            {
                "target_clip_lower_quantile": 1.0 - clip_upper,
                "target_clip_upper_quantile": clip_upper,
                "recency_min_weight": trial.suggest_float("recency_min_weight", 0.2, 0.8, step=0.1),
            }
        )
        direction_mode = gbm_train.resolve_direction_mode(
            trial.suggest_categorical(
                "direction_mode",
                ["long_short", "long_only", "short_only"],
            )
        )
        result = evaluate_candidate_on_end_date(
            tickers=tickers,
            end_date=tune_end,
            train_window=args.train_window,
            include_time_index=args.include_time_index,
            feature_set=feature_set,
            feature_set_file=feature_set_file,
            lgbm_params=lgbm_params,
            training_policy=training_policy,
            direction_mode=direction_mode,
            lgbm_param_preset="baseline",
            lgbm_params_json=None,
            notional=args.notional,
            prepared_cache=prepared_cache,
        )
        assessment = assess_candidate(
            result=result,
            baseline_ticker_summaries=baseline_tune["tickers"],
            drawdown_worsen_limit=args.drawdown_worsen_limit,
            min_basket_trades=args.min_basket_trades,
        )
        candidate_score = result["aggregate"].get("basket_objective_score")
        if assessment["hard_reject"] or candidate_score is None:
            score = HARD_REJECT_SCORE
        else:
            score = float(candidate_score)

        record = {
            "trial_number": trial.number,
            "feature_set": feature_set,
            "params": lgbm_params,
            "training_policy": training_policy,
            "direction_mode": direction_mode,
            "aggregate": result["aggregate"],
            "tickers": result["tickers"],
            "assessment": assessment,
            "objective_value": float(score),
        }
        tune_trial_records[trial.number] = record

        trial.set_user_attr("feature_set", feature_set)
        trial.set_user_attr("aggregate", json_ready(result["aggregate"]))
        trial.set_user_attr("assessment", json_ready(assessment))
        trial.set_user_attr("objective_value", float(score))
        return float(score)

    study.optimize(objective, n_trials=args.n_trials)

    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    tune_ranked = []
    for trial in completed_trials:
        cached = tune_trial_records.get(trial.number)
        if cached is not None:
            tune_ranked.append(cached)
            continue
        feature_set, _, resolved, training_policy, direction_mode = trial_params_from_optuna_params(
            trial.params
        )
        tune_ranked.append(
            {
                "trial_number": trial.number,
                "feature_set": feature_set,
                "params": resolved,
                "training_policy": training_policy,
                "direction_mode": direction_mode,
                "aggregate": trial.user_attrs.get("aggregate"),
                "assessment": trial.user_attrs.get("assessment"),
                "objective_value": float(trial.value if trial.value is not None else HARD_REJECT_SCORE),
                "error": trial.user_attrs.get("error"),
            }
        )

    tune_ranked.sort(
        key=lambda x: float(x.get("objective_value", HARD_REJECT_SCORE)),
        reverse=True,
    )
    write_json(
        run_dir / "optuna_trials_top.json",
        {
            "candidate_count": len(tune_ranked),
            "top_50": [trial_record_for_report(item) for item in tune_ranked[:50]],
        },
    )

    top_trials = sorted(
        completed_trials,
        key=lambda trial: float(trial.value if trial.value is not None else HARD_REJECT_SCORE),
        reverse=True,
    )[: max(1, args.holdout_top_n)]

    print(
        f"Validating top {len(top_trials)} Optuna trials on holdout date: {holdout_end.date()}..."
    )
    holdout_validations = []
    baseline_holdout_score = baseline_holdout["aggregate"]["basket_objective_score"]
    promoted = None
    for trial in top_trials:
        tune_record = next((item for item in tune_ranked if item["trial_number"] == trial.number), None)
        if tune_record is None:
            feature_set, _, resolved, training_policy, direction_mode = trial_params_from_optuna_params(
                trial.params
            )
            tune_record = {
                "trial_number": trial.number,
                "feature_set": feature_set,
                "params": resolved,
                "training_policy": training_policy,
                "direction_mode": direction_mode,
                "objective_value": float(trial.value if trial.value is not None else HARD_REJECT_SCORE),
            }

        holdout_eval = evaluate_candidate_on_end_date(
            tickers=tickers,
            end_date=holdout_end,
            train_window=args.train_window,
            include_time_index=args.include_time_index,
            feature_set=tune_record["feature_set"],
            feature_set_file=feature_set_file,
            lgbm_params=tune_record["params"],
            training_policy=tune_record["training_policy"],
            direction_mode=tune_record["direction_mode"],
            lgbm_param_preset="baseline",
            lgbm_params_json=None,
            notional=args.notional,
            prepared_cache=prepared_cache,
        )
        holdout_assessment = assess_candidate(
            result=holdout_eval,
            baseline_ticker_summaries=baseline_holdout["tickers"],
            drawdown_worsen_limit=args.drawdown_worsen_limit,
            min_basket_trades=args.min_basket_trades,
        )
        holdout_pass = not holdout_assessment["hard_reject"]
        holdout_score = holdout_eval["aggregate"]["basket_objective_score"]
        beats_baseline = (
            holdout_pass
            and holdout_score is not None
            and baseline_holdout_score is not None
            and float(holdout_score) > float(baseline_holdout_score)
        )
        item = {
            "trial_number": int(trial.number),
            "objective_value": float(trial.value if trial.value is not None else HARD_REJECT_SCORE),
            "candidate": tune_record,
            "holdout": holdout_eval,
            "holdout_assessment": holdout_assessment,
            "beats_holdout_baseline": beats_baseline,
        }
        holdout_validations.append(item)
        if promoted is None and beats_baseline:
            promoted = item

    if promoted is None:
        eligible = [item for item in holdout_validations if not item["holdout_assessment"]["hard_reject"]]
        population = eligible if eligible else holdout_validations
        if population:
            promoted = max(
                population,
                key=lambda item: (
                    float(
                        item["holdout"]["aggregate"]["basket_objective_score"]
                        if item["holdout"]["aggregate"]["basket_objective_score"] is not None
                        else float("-inf")
                    ),
                    float(
                        item["holdout"]["aggregate"]["basket_mean_avg_return_pct"]
                        if item["holdout"]["aggregate"]["basket_mean_avg_return_pct"] is not None
                        else float("-inf")
                    ),
                    float(item["objective_value"]),
                ),
            )

    write_json(
        run_dir / "holdout_validations.json",
        {
            "count": len(holdout_validations),
            "items": holdout_validations,
        },
    )

    winner = None
    if promoted is not None:
        winner_candidate = promoted["candidate"]
        winner = {
            "trial_number": winner_candidate["trial_number"],
            "objective_value": winner_candidate["objective_value"],
            "feature_set": winner_candidate["feature_set"],
            "params": winner_candidate["params"],
            "training_policy": winner_candidate["training_policy"],
            "direction_mode": winner_candidate["direction_mode"],
            "tune": {
                "aggregate": winner_candidate.get("aggregate"),
                "tickers": winner_candidate.get("tickers"),
                "assessment": winner_candidate.get("assessment"),
            },
            "holdout": promoted["holdout"],
            "holdout_assessment": promoted["holdout_assessment"],
            "beats_holdout_baseline": promoted["beats_holdout_baseline"],
        }

    retrain_summaries = None
    if winner is not None and not args.skip_retrain:
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
    if winner is not None:
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
                training_policy=winner["training_policy"],
                direction_mode=winner["direction_mode"],
                output_dir=optimized_output_dir,
                lgbm_param_preset="baseline",
                lgbm_params_json=None,
                write_outputs=True,
                verbose=False,
            )
            optimized_backtests[ticker] = backtest_result["summary"]

    gp_reference = collect_gp_reference(tickers)

    tune_assessments = [
        item.get("assessment")
        for item in tune_ranked
        if isinstance(item.get("assessment"), dict)
    ]
    holdout_assessments = [item["holdout_assessment"] for item in holdout_validations]
    tune_guardrail_pass = sum(1 for item in tune_assessments if item.get("guardrail_pass"))
    holdout_guardrail_pass = sum(1 for item in holdout_assessments if item.get("guardrail_pass"))
    tune_hard_reject = sum(1 for item in tune_assessments if item.get("hard_reject"))
    holdout_hard_reject = sum(1 for item in holdout_assessments if item.get("hard_reject"))

    report = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "search_backend": "optuna",
        "tickers": tickers,
        "dates": {
            "baseline_end": str(baseline_end.date()),
            "ablation_end": str(ablation_end.date()),
            "tune_end": str(tune_end.date()),
            "holdout_end": str(holdout_end.date()),
        },
        "run_config": {
            "train_window": args.train_window,
            "test_window": args.test_window,
            "step_window": args.step_window,
            "notional": args.notional,
            "drawdown_worsen_limit": args.drawdown_worsen_limit,
            "min_basket_trades": args.min_basket_trades,
            "seed": args.seed,
            "n_trials": args.n_trials,
            "holdout_top_n": args.holdout_top_n,
            "include_time_index": bool(args.include_time_index),
        },
        "feature_set_file": feature_set_file,
        "feature_sets": {"F0": [], "F1": f1_drops, "F2": f2_drops},
        "baseline_holdout": baseline_holdout,
        "baseline_tune": baseline_tune,
        "optuna": {
            "study_name": study.study_name,
            "storage_uri": storage_uri,
            "storage_path": storage_path,
            "seed": args.seed,
            "n_trials_requested": args.n_trials,
            "n_trials_completed": len(completed_trials),
            "best_trial_number": int(study.best_trial.number),
            "best_value": float(study.best_value),
            "best_params": study.best_trial.params,
        },
        "guardrail_stats": {
            "tune_guardrail_pass_count": tune_guardrail_pass,
            "tune_guardrail_fail_count": len(tune_assessments) - tune_guardrail_pass,
            "tune_hard_reject_count": tune_hard_reject,
            "holdout_guardrail_pass_count": holdout_guardrail_pass,
            "holdout_guardrail_fail_count": len(holdout_assessments) - holdout_guardrail_pass,
            "holdout_hard_reject_count": holdout_hard_reject,
        },
        "winner": winner,
        "tune_top_10": [trial_record_for_report(item) for item in tune_ranked[:10]],
        "holdout_validation_top": holdout_validations[:10],
        "optimized_backtests": optimized_backtests,
        "gp_reference": gp_reference,
        "retrain_summaries": retrain_summaries,
    }
    write_json(run_dir / "final_report.json", report)

    if winner is not None and not args.skip_tech_update:
        tech_path = ROOT_DIR / "gbm_return" / "TECH.md"
        append_tech_results(tech_path, report)
        print(f"Updated TECH.md: {tech_path}")

    print("\nOptimization complete.")
    print(f"Run directory: {run_dir}")
    if winner is None:
        print("No holdout winner selected.")
    else:
        print(f"Winner trial: {winner['trial_number']}")
        print(f"Winner feature set: {winner['feature_set']}")
        print(
            f"Winner holdout mean avg_return_pct: {winner['holdout']['aggregate']['basket_mean_avg_return_pct']}"
        )


if __name__ == "__main__":
    main()
