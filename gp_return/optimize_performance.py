import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.trial import TrialState

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.walk_forward import walk_forward_splits
from gp_return.backtest_walk_forward import (
    DEFAULT_TEST_YEARS,
    build_backtest_pca_transformer,
    build_dataset,
    compute_dataset_start,
    summarize_trades,
)
from gp_return.train import ARTIFACT_DIR_DEFAULT, WINDOW_RET, normalize_features, set_time_index, train_gp


DEFAULT_TICKERS = ["AAPL", "NVDA", "AMZN", "KO"]
DEFAULT_NOTIONAL = 10000.0
DEFAULT_TRIALS = 120
DEFAULT_HOLDOUT_TOP_N = 20
DEFAULT_DRAWDOWN_WORSEN_LIMIT = 0.10
MIN_TRAIN_ROWS = 60
PENALTY_SCORE = -1e9


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize gp_return with Optuna using backtest return.")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--train-window", default="2y")
    parser.add_argument("--test-window", default="1m")
    parser.add_argument("--step-window", default="1m")
    parser.add_argument("--tune-end", default=None, help="Tune end date YYYY-MM-DD. Default: holdout_end - 3 months.")
    parser.add_argument("--holdout-end", default=None, help="Holdout end date YYYY-MM-DD. Default: today.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--holdout-top-n", type=int, default=DEFAULT_HOLDOUT_TOP_N)
    parser.add_argument("--drawdown-worsen-limit", type=float, default=DEFAULT_DRAWDOWN_WORSEN_LIMIT)
    parser.add_argument("--notional", type=float, default=DEFAULT_NOTIONAL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR_DEFAULT / "optuna_runs"))
    return parser.parse_args()


def parse_tickers(raw: str) -> list[str]:
    tokens = [token.strip().upper() for token in raw.split(",")]
    tickers: list[str] = []
    seen: set[str] = set()
    for ticker in tokens:
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


def prepare_backtest_data(
    ticker: str,
    end_date: pd.Timestamp,
    train_window: str,
    test_window: str,
    step_window: str,
):
    dataset_start = compute_dataset_start(end_date, train_window)
    eval_start = end_date - pd.DateOffset(years=DEFAULT_TEST_YEARS)
    data = build_dataset(
        ticker=ticker,
        start_date=dataset_start,
        end_date=end_date,
    )

    dataset = data["dataset"]
    all_splits = list(
        walk_forward_splits(
            data=dataset,
            train_window=train_window,
            test_window=test_window,
            embargo=WINDOW_RET,
            step=step_window,
            min_train_rows=MIN_TRAIN_ROWS,
        )
    )
    selected_splits = [
        split
        for split in all_splits
        if split.test_start >= eval_start and split.test_end <= end_date
    ]

    return {
        "ticker": ticker,
        "end_date": end_date,
        "train_window": train_window,
        "test_window": test_window,
        "step_window": step_window,
        "eval_start": eval_start,
        "dataset": dataset,
        "splits": selected_splits,
        "base_feature_columns": [col for col in dataset.columns if col != "target"],
    }


def run_backtest_prepared_gp(
    prepared: dict[str, Any],
    feature_columns: list[str],
    model_params: dict[str, Any],
    pca_enabled: bool,
    notional: float,
    device: torch.device,
):
    trades: list[dict[str, Any]] = []
    for split in prepared["splits"]:
        train_df = split.train.copy()
        test_df = split.test.copy()
        fold_start = train_df.index.min()
        train_df = set_time_index(train_df, fold_start)
        test_df = set_time_index(test_df, fold_start)

        if pca_enabled:
            fold_pca = build_backtest_pca_transformer()
            train_x_df, test_x_df = fold_pca.transform_train_test(train_df, test_df, feature_columns)
        else:
            train_x_df, test_x_df, _ = normalize_features(train_df, test_df, feature_columns)

        train_x = torch.tensor(train_x_df.values, dtype=torch.float32, device=device)
        train_y = torch.tensor(train_df["target"].values, dtype=torch.float32, device=device)
        model, likelihood = train_gp(
            train_x,
            train_y,
            train_iters=int(model_params["train_iters"]),
            device=device,
            learning_rate=float(model_params["learning_rate"]),
            weight_decay=float(model_params["weight_decay"]),
            matern_nu=float(model_params["matern_nu"]),
            use_rq=bool(model_params["use_rq"]),
            use_linear=bool(model_params["use_linear"]),
        )

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            test_x = torch.tensor(test_x_df.values, dtype=torch.float32, device=device)
            preds = likelihood(model(test_x))
            mean_logs = preds.mean.detach().cpu().numpy()

        actual_simple = np.exp(test_df["target"].values) - 1.0
        for idx, test_date in enumerate(test_df.index):
            mean_log = float(mean_logs[idx])
            direction = "long" if mean_log > 0.0 else "short"
            signed_return = float(actual_simple[idx]) if direction == "long" else float(-actual_simple[idx])
            pnl = notional * signed_return
            trades.append(
                {
                    "symbol": prepared["ticker"],
                    "trade_date": test_date,
                    "direction": direction,
                    "pnl": pnl,
                    "return_pct": signed_return,
                    "fold": int(split.fold),
                }
            )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("trade_date")
    summary = summarize_trades(trades_df)
    avg_return_pct = float(trades_df["return_pct"].mean()) if not trades_df.empty else None
    summary.update(
        {
            "ticker": prepared["ticker"],
            "start_date": str(prepared["eval_start"].date()),
            "end_date": str(prepared["end_date"].date()),
            "train_window": prepared["train_window"],
            "test_window": prepared["test_window"],
            "step_window": prepared["step_window"],
            "test_years": DEFAULT_TEST_YEARS,
            "window_ret": WINDOW_RET,
            "notional": notional,
            "avg_return_pct": avg_return_pct,
            "feature_count": len(feature_columns),
            "pca_enabled": bool(pca_enabled),
            "fold_count": int(len(prepared["splits"])),
        }
    )
    return summary


def evaluate_candidate_on_end_date(
    tickers: list[str],
    end_date: pd.Timestamp,
    train_window: str,
    test_window: str,
    step_window: str,
    selected_features: list[str],
    model_params: dict[str, Any],
    pca_enabled: bool,
    notional: float,
    device: torch.device,
    prepared_cache: dict[tuple[str, str, str, str, bool], dict[str, Any]],
    on_ticker_done=None,
):
    ticker_summaries: dict[str, dict[str, Any]] = {}
    for idx, ticker in enumerate(tickers):
        cache_key = (
            ticker,
            str(end_date.date()),
            train_window,
            f"{test_window}|{step_window}",
            bool(pca_enabled),
        )
        if cache_key not in prepared_cache:
            prepared_cache[cache_key] = prepare_backtest_data(
                ticker=ticker,
                end_date=end_date,
                train_window=train_window,
                test_window=test_window,
                step_window=step_window,
            )
        prepared = prepared_cache[cache_key]
        feature_columns = [col for col in selected_features if col in prepared["base_feature_columns"]]
        if not feature_columns:
            summary = {
                "ticker": ticker,
                "total_trades": 0,
                "win_rate": None,
                "avg_pnl": None,
                "median_pnl": None,
                "std_pnl": None,
                "max_drawdown": None,
                "avg_return_pct": None,
            }
        else:
            summary = run_backtest_prepared_gp(
                prepared=prepared,
                feature_columns=feature_columns,
                model_params=model_params,
                pca_enabled=pca_enabled,
                notional=notional,
                device=device,
            )
        ticker_summaries[ticker] = summary
        if callable(on_ticker_done):
            on_ticker_done(idx, ticker, summary)

    aggregate = aggregate_basket_summary(ticker_summaries)
    return {"aggregate": aggregate, "tickers": ticker_summaries}


def sample_model_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "train_iters": trial.suggest_int("train_iters", 40, 180, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        "matern_nu": trial.suggest_categorical("matern_nu", [0.5, 1.5, 2.5]),
        "use_rq": trial.suggest_categorical("use_rq", [True, False]),
        "use_linear": trial.suggest_categorical("use_linear", [True, False]),
    }


def sample_selected_features(trial: optuna.Trial, base_feature_columns: list[str]) -> list[str]:
    selected: list[str] = []
    for col in base_feature_columns:
        keep = trial.suggest_categorical(f"feat__{col}", [True, False])
        if keep:
            selected.append(col)
    return selected


def trial_to_result(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
    }


def main():
    args = parse_args()
    tickers = parse_tickers(args.tickers) or list(DEFAULT_TICKERS)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    print(f"Using device: {device.type}")

    holdout_end = pd.Timestamp(args.holdout_end).normalize() if args.holdout_end else pd.Timestamp.today().normalize()
    tune_end = pd.Timestamp(args.tune_end).normalize() if args.tune_end else (holdout_end - pd.DateOffset(months=3))

    run_dir = Path(args.output_dir) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared_cache: dict[tuple[str, str, str, str, bool], dict[str, Any]] = {}
    base_prepared = prepare_backtest_data(
        ticker=tickers[0],
        end_date=tune_end,
        train_window=args.train_window,
        test_window=args.test_window,
        step_window=args.step_window,
    )
    base_feature_columns = list(base_prepared["base_feature_columns"])
    baseline_features = [col for col in base_feature_columns if col != "time_index"] or list(base_feature_columns)
    baseline_params = {
        "train_iters": 160,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "matern_nu": 0.5,
        "use_rq": True,
        "use_linear": True,
    }

    baseline_tune = evaluate_candidate_on_end_date(
        tickers=tickers,
        end_date=tune_end,
        train_window=args.train_window,
        test_window=args.test_window,
        step_window=args.step_window,
        selected_features=baseline_features,
        model_params=baseline_params,
        pca_enabled=args.pca,
        notional=args.notional,
        device=device,
        prepared_cache=prepared_cache,
    )
    baseline_holdout = evaluate_candidate_on_end_date(
        tickers=tickers,
        end_date=holdout_end,
        train_window=args.train_window,
        test_window=args.test_window,
        step_window=args.step_window,
        selected_features=baseline_features,
        model_params=baseline_params,
        pca_enabled=args.pca,
        notional=args.notional,
        device=device,
        prepared_cache=prepared_cache,
    )

    write_json(run_dir / "baseline_tune.json", baseline_tune)
    write_json(run_dir / "baseline_holdout.json", baseline_holdout)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        model_params = sample_model_params(trial)
        selected_features = sample_selected_features(trial, base_feature_columns)
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr("selected_features", selected_features)
        if not selected_features:
            trial.set_user_attr("guardrail_pass", False)
            trial.set_user_attr("guardrail_violations", [{"reason": "no_features_selected"}])
            return PENALTY_SCORE

        rolling_returns: list[float] = []

        def on_ticker_done(idx: int, _ticker: str, summary: dict[str, Any]):
            avg_return = summary.get("avg_return_pct")
            if avg_return is not None:
                rolling_returns.append(float(avg_return))
            intermediate = float(np.mean(rolling_returns)) if rolling_returns else PENALTY_SCORE
            trial.report(intermediate, step=idx + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = evaluate_candidate_on_end_date(
            tickers=tickers,
            end_date=tune_end,
            train_window=args.train_window,
            test_window=args.test_window,
            step_window=args.step_window,
            selected_features=selected_features,
            model_params=model_params,
            pca_enabled=args.pca,
            notional=args.notional,
            device=device,
            prepared_cache=prepared_cache,
            on_ticker_done=on_ticker_done,
        )
        violations = drawdown_guardrail_violations(
            candidate_summaries=result["tickers"],
            baseline_summaries=baseline_tune["tickers"],
            worsen_limit=args.drawdown_worsen_limit,
        )
        guardrail_pass = len(violations) == 0
        score = result["aggregate"]["basket_mean_avg_return_pct"]
        if score is None:
            score = PENALTY_SCORE
        elif not guardrail_pass:
            score = PENALTY_SCORE

        trial.set_user_attr("aggregate", result["aggregate"])
        trial.set_user_attr("tickers", result["tickers"])
        trial.set_user_attr("guardrail_pass", guardrail_pass)
        trial.set_user_attr("guardrail_violations", violations)
        return float(score)

    print(f"Running Optuna: trials={args.trials}, tickers={','.join(tickers)}")
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    trial_results = {"trial_count": len(study.trials), "trials": [trial_to_result(t) for t in study.trials]}
    write_json(run_dir / "trial_results.json", trial_results)

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: float(t.value if t.value is not None else PENALTY_SCORE), reverse=True)
    finalists = completed_trials[: max(1, args.holdout_top_n)]

    baseline_holdout_mean = baseline_holdout["aggregate"]["basket_mean_avg_return_pct"]
    holdout_validations: list[dict[str, Any]] = []
    for trial in finalists:
        model_params = dict(trial.user_attrs.get("model_params", {}))
        selected_features = list(trial.user_attrs.get("selected_features", []))
        holdout_result = evaluate_candidate_on_end_date(
            tickers=tickers,
            end_date=holdout_end,
            train_window=args.train_window,
            test_window=args.test_window,
            step_window=args.step_window,
            selected_features=selected_features,
            model_params=model_params,
            pca_enabled=args.pca,
            notional=args.notional,
            device=device,
            prepared_cache=prepared_cache,
        )
        holdout_violations = drawdown_guardrail_violations(
            candidate_summaries=holdout_result["tickers"],
            baseline_summaries=baseline_holdout["tickers"],
            worsen_limit=args.drawdown_worsen_limit,
        )
        holdout_pass = len(holdout_violations) == 0
        holdout_mean = holdout_result["aggregate"]["basket_mean_avg_return_pct"]
        beats_baseline = (
            holdout_pass
            and holdout_mean is not None
            and baseline_holdout_mean is not None
            and holdout_mean > baseline_holdout_mean
        )
        holdout_validations.append(
            {
                "trial_number": trial.number,
                "trial_value": trial.value,
                "selected_features": selected_features,
                "model_params": model_params,
                "tune": {
                    "aggregate": trial.user_attrs.get("aggregate"),
                    "tickers": trial.user_attrs.get("tickers"),
                    "guardrail_pass": trial.user_attrs.get("guardrail_pass"),
                    "guardrail_violations": trial.user_attrs.get("guardrail_violations"),
                },
                "holdout": holdout_result,
                "holdout_guardrail_pass": holdout_pass,
                "holdout_guardrail_violations": holdout_violations,
                "beats_holdout_baseline": beats_baseline,
            }
        )

    holdout_validations.sort(
        key=lambda x: (
            1 if x["holdout_guardrail_pass"] else 0,
            x["holdout"]["aggregate"]["basket_mean_avg_return_pct"]
            if x["holdout"]["aggregate"]["basket_mean_avg_return_pct"] is not None
            else PENALTY_SCORE,
        ),
        reverse=True,
    )
    write_json(
        run_dir / "holdout_validations.json",
        {"count": len(holdout_validations), "items": holdout_validations},
    )

    winner = holdout_validations[0] if holdout_validations else None
    final_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_config": {
            "tickers": tickers,
            "train_window": args.train_window,
            "test_window": args.test_window,
            "step_window": args.step_window,
            "tune_end": str(tune_end.date()),
            "holdout_end": str(holdout_end.date()),
            "trials": args.trials,
            "drawdown_worsen_limit": args.drawdown_worsen_limit,
            "holdout_top_n": args.holdout_top_n,
            "notional": args.notional,
            "seed": args.seed,
            "pca": bool(args.pca),
        },
        "baseline_tune": baseline_tune,
        "baseline_holdout": baseline_holdout,
        "winner": winner,
        "top_10_trials": [trial_to_result(t) for t in completed_trials[:10]],
        "holdout_validation_top_10": holdout_validations[:10],
    }
    write_json(run_dir / "final_report.json", final_report)

    run_config = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tickers": tickers,
        "train_window": args.train_window,
        "test_window": args.test_window,
        "step_window": args.step_window,
        "tune_end": str(tune_end.date()),
        "holdout_end": str(holdout_end.date()),
        "trials": args.trials,
        "drawdown_worsen_limit": args.drawdown_worsen_limit,
        "holdout_top_n": args.holdout_top_n,
        "notional": args.notional,
        "seed": args.seed,
        "pca": bool(args.pca),
        "output_dir": str(run_dir),
        "base_feature_columns": base_feature_columns,
        "baseline_features": baseline_features,
        "baseline_model_params": baseline_params,
    }
    write_json(run_dir / "run_config.json", run_config)

    print(f"Run directory: {run_dir}")
    if winner is None:
        print("No winner selected (no completed trials).")
    else:
        holdout_mean = winner["holdout"]["aggregate"]["basket_mean_avg_return_pct"]
        print(f"Winner trial: {winner['trial_number']}")
        print(f"Winner holdout basket mean avg_return_pct: {holdout_mean}")


if __name__ == "__main__":
    main()
