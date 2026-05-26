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

from common import walk_forward_splits
from hmm_regime.train import (
    DEFAULT_MIN_TRAIN_ROWS as HMM_DEFAULT_MIN_TRAIN_ROWS,
    build_market_dataset,
)
from lstm_return.backtest_walk_forward import DEFAULT_TEST_YEARS, summarize_trades
from lstm_return.train import (
    ARTIFACT_DIR_DEFAULT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_HMM_N_INIT,
    DEFAULT_HMM_N_ITER,
    DEFAULT_HMM_TRAIN_WINDOW,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_TRAIN_SEQUENCES,
    DEFAULT_NUM_LAYERS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SEQ_LEN,
    DEFAULT_TRAIN_WINDOW,
    DEFAULT_WEIGHT_DECAY,
    HMM_FEATURE_COLUMNS,
    WINDOW_RET,
    build_model_dataset,
    compute_dataset_start,
    fit_lstm_model,
    predict_sequences,
    prepare_fold_data,
    resolve_base_feature_columns,
    resolve_device,
)


DEFAULT_TICKERS = ["AAPL", "NVDA", "AMZN", "KO"]
DEFAULT_NOTIONAL = 10000.0
DEFAULT_TRIALS = 120
DEFAULT_HOLDOUT_TOP_N = 20
DEFAULT_DRAWDOWN_WORSEN_LIMIT = 0.10
MIN_TRAIN_ROWS = HMM_DEFAULT_MIN_TRAIN_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize lstm_return with Optuna using backtest return."
    )
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--train-window", default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--test-window", default="1m")
    parser.add_argument("--step-window", default="1m")
    parser.add_argument(
        "--tune-end",
        default=None,
        help="Tune end date YYYY-MM-DD. Default: holdout_end - 3 months.",
    )
    parser.add_argument(
        "--holdout-end",
        default=None,
        help="Holdout end date YYYY-MM-DD. Default: today.",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--holdout-top-n", type=int, default=DEFAULT_HOLDOUT_TOP_N)
    parser.add_argument(
        "--drawdown-worsen-limit",
        type=float,
        default=DEFAULT_DRAWDOWN_WORSEN_LIMIT,
    )
    parser.add_argument("--notional", type=float, default=DEFAULT_NOTIONAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2))


def compute_objective_score(aggregate: dict[str, Any]) -> float | None:
    mean_pnl = aggregate.get("basket_mean_avg_pnl")
    if mean_pnl is None or not np.isfinite(float(mean_pnl)):
        return None
    max_drawdown = aggregate.get("basket_worst_max_drawdown")
    if max_drawdown is None or not np.isfinite(float(max_drawdown)):
        return float(mean_pnl)
    return float(mean_pnl) - abs(float(max_drawdown))


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

    aggregate = {
        "basket_mean_avg_return_pct": float(np.mean(avg_return_values)) if avg_return_values else None,
        "basket_mean_win_rate": float(np.mean(win_rate_values)) if win_rate_values else None,
        "basket_mean_avg_pnl": float(np.mean(avg_pnl_values)) if avg_pnl_values else None,
        "basket_worst_max_drawdown": float(np.min(max_drawdowns)) if max_drawdowns else None,
        "basket_total_trades": int(total_trades),
    }
    aggregate["basket_objective_score"] = compute_objective_score(aggregate)
    return aggregate


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
    hmm_train_window: str,
):
    dataset_start = compute_dataset_start(end_date, train_window, hmm_train_window)
    eval_start = end_date - pd.DateOffset(years=DEFAULT_TEST_YEARS)
    data = build_model_dataset(
        ticker=ticker,
        start_date=dataset_start,
        end_date=end_date,
        history_cache={},
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
        "market_dataset": build_market_dataset(dataset_start, end_date),
        "splits": selected_splits,
        "base_feature_columns": resolve_base_feature_columns(
            dataset=dataset,
            drop_time_index=False,
        ),
    }


def run_backtest_prepared_lstm(
    prepared: dict[str, Any],
    feature_columns: list[str],
    model_params: dict[str, Any],
    notional: float,
    device: torch.device,
):
    feature_cols = list(feature_columns) + list(HMM_FEATURE_COLUMNS)
    model_kwargs = {
        "input_size": len(feature_cols),
        "hidden_size": int(model_params["hidden_size"]),
        "num_layers": int(model_params["num_layers"]),
        "dropout": float(model_params["dropout"]),
    }
    config = {
        "seq_len": int(model_params["seq_len"]),
        "hmm_train_window": model_params["hmm_train_window"],
        "hmm_n_iter": int(model_params["hmm_n_iter"]),
        "hmm_n_init": int(model_params["hmm_n_init"]),
        "random_state": int(model_params["random_state"]),
    }

    trade_frames: list[pd.DataFrame] = []
    for split in prepared["splits"]:
        train_x, train_y, _, test_x, test_y, test_dates, _, _ = prepare_fold_data(
            dataset=prepared["dataset"],
            market_dataset=prepared["market_dataset"],
            split_train_start=split.train_start,
            split_train_end=split.train_end,
            split_test_end=split.test_end,
            split_train_dates=split.train.index,
            split_test_dates=split.test.index,
            base_feature_cols=feature_columns,
            config=config,
        )
        if len(train_x) < DEFAULT_MIN_TRAIN_SEQUENCES or len(test_x) == 0:
            continue

        model, _ = fit_lstm_model(
            train_x=train_x,
            train_y=train_y,
            model_kwargs=model_kwargs,
            device=device,
            epochs=int(model_params["epochs"]),
            batch_size=int(model_params["batch_size"]),
            learning_rate=float(model_params["learning_rate"]),
            weight_decay=float(model_params["weight_decay"]),
            seed=int(model_params["random_state"]),
        )
        preds = predict_sequences(model, test_x, device=device)

        actual_simple = np.exp(test_y) - 1.0
        directions = np.where(preds > 0.0, "long", "short")
        signed_returns = np.where(preds > 0.0, actual_simple, -actual_simple)
        trade_frames.append(
            pd.DataFrame(
                {
                    "symbol": prepared["ticker"],
                    "trade_date": test_dates,
                    "direction": directions,
                    "predicted_log_return": preds,
                    "actual_log_return": test_y,
                    "pnl": notional * signed_returns,
                    "return_pct": signed_returns,
                    "fold": int(split.fold),
                }
            )
        )

    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
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
    notional: float,
    device: torch.device,
    prepared_cache: dict[tuple[str, str, str, str, str], dict[str, Any]],
    on_ticker_done=None,
):
    ticker_summaries: dict[str, dict[str, Any]] = {}
    for idx, ticker in enumerate(tickers):
        cache_key = (
            ticker,
            str(end_date.date()),
            train_window,
            f"{test_window}|{step_window}",
            str(model_params["hmm_train_window"]),
        )
        if cache_key not in prepared_cache:
            prepared_cache[cache_key] = prepare_backtest_data(
                ticker=ticker,
                end_date=end_date,
                train_window=train_window,
                test_window=test_window,
                step_window=step_window,
                hmm_train_window=str(model_params["hmm_train_window"]),
            )
        prepared = prepared_cache[cache_key]
        available_feature_columns = resolve_base_feature_columns(
            dataset=prepared["dataset"],
            drop_time_index=bool(model_params["drop_time_index"]),
        )
        feature_columns = [col for col in selected_features if col in available_feature_columns]
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
            summary = run_backtest_prepared_lstm(
                prepared=prepared,
                feature_columns=feature_columns,
                model_params=model_params,
                notional=notional,
                device=device,
            )
        ticker_summaries[ticker] = summary
        if callable(on_ticker_done):
            on_ticker_done(idx, ticker, summary)

    aggregate = aggregate_basket_summary(ticker_summaries)
    return {"aggregate": aggregate, "tickers": ticker_summaries}


def sample_model_params(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    include_time_index = trial.suggest_categorical("include_time_index", [False, True])
    return {
        "seq_len": trial.suggest_categorical("seq_len", [20, 40, 60, 80, 120]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "epochs": trial.suggest_categorical("epochs", [20, 30, 40, 60]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "drop_time_index": not include_time_index,
        "random_state": int(seed),
        "hmm_train_window": trial.suggest_categorical("hmm_train_window", ["2y", "3y", "4y"]),
        "hmm_n_iter": trial.suggest_categorical("hmm_n_iter", [150, 250, 400]),
        "hmm_n_init": trial.suggest_categorical("hmm_n_init", [2, 3, 4]),
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


def main() -> None:
    args = parse_args()
    tickers = parse_tickers(args.tickers) or list(DEFAULT_TICKERS)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device()
    print(f"Using device: {device.type}")

    holdout_end = pd.Timestamp(args.holdout_end).normalize() if args.holdout_end else pd.Timestamp.today().normalize()
    tune_end = pd.Timestamp(args.tune_end).normalize() if args.tune_end else (holdout_end - pd.DateOffset(months=3))

    run_dir = Path(args.output_dir) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared_cache: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    baseline_params = {
        "seq_len": DEFAULT_SEQ_LEN,
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "num_layers": DEFAULT_NUM_LAYERS,
        "dropout": DEFAULT_DROPOUT,
        "epochs": DEFAULT_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "drop_time_index": True,
        "random_state": int(args.seed),
        "hmm_train_window": DEFAULT_HMM_TRAIN_WINDOW,
        "hmm_n_iter": DEFAULT_HMM_N_ITER,
        "hmm_n_init": DEFAULT_HMM_N_INIT,
    }
    base_prepared = prepare_backtest_data(
        ticker=tickers[0],
        end_date=tune_end,
        train_window=args.train_window,
        test_window=args.test_window,
        step_window=args.step_window,
        hmm_train_window=baseline_params["hmm_train_window"],
    )
    base_feature_columns = list(base_prepared["base_feature_columns"])
    baseline_features = list(base_feature_columns)

    baseline_tune = evaluate_candidate_on_end_date(
        tickers=tickers,
        end_date=tune_end,
        train_window=args.train_window,
        test_window=args.test_window,
        step_window=args.step_window,
        selected_features=baseline_features,
        model_params=baseline_params,
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
        model_params = sample_model_params(trial, args.seed)
        selected_features = sample_selected_features(trial, base_feature_columns)
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr("selected_features", selected_features)
        if not selected_features:
            trial.set_user_attr("guardrail_pass", False)
            trial.set_user_attr("guardrail_violations", [{"reason": "no_features_selected"}])
            raise optuna.TrialPruned("No features selected.")

        rolling_summaries: dict[str, dict[str, Any]] = {}

        def on_ticker_done(idx: int, _ticker: str, summary: dict[str, Any]):
            rolling_summaries[_ticker] = summary
            intermediate = aggregate_basket_summary(rolling_summaries).get("basket_objective_score")
            if intermediate is not None and np.isfinite(float(intermediate)):
                trial.report(float(intermediate), step=idx + 1)
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
        score = result["aggregate"].get("basket_objective_score")
        if score is None or not np.isfinite(float(score)):
            raise optuna.TrialPruned("Candidate produced no objective score.")

        trial.set_user_attr("aggregate", result["aggregate"])
        trial.set_user_attr("tickers", result["tickers"])
        trial.set_user_attr("guardrail_pass", guardrail_pass)
        trial.set_user_attr("guardrail_violations", violations)
        return float(score)

    print(f"Running Optuna: trials={args.trials}, tickers={','.join(tickers)}")
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    trial_results = {"trial_count": len(study.trials), "trials": [trial_to_result(t) for t in study.trials]}
    write_json(run_dir / "trial_results.json", trial_results)

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    completed_trials.sort(key=lambda t: float(t.value), reverse=True)
    finalists = completed_trials[: max(1, args.holdout_top_n)]

    baseline_holdout_score = baseline_holdout["aggregate"].get("basket_objective_score")
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
        holdout_score = holdout_result["aggregate"].get("basket_objective_score")
        beats_baseline = (
            holdout_score is not None
            and baseline_holdout_score is not None
            and holdout_score > baseline_holdout_score
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
                "holdout_score": holdout_score,
                "holdout_guardrail_pass": holdout_pass,
                "holdout_guardrail_violations": holdout_violations,
                "beats_holdout_baseline": beats_baseline,
            }
        )

    holdout_validations.sort(
        key=lambda x: (
            float(x["holdout_score"]) if x["holdout_score"] is not None else float("-inf"),
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
