import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import TrialState

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from hmm_regime.backtest_walk_forward import (
    DEFAULT_AUC_THRESHOLD,
    DEFAULT_MIN_STATE_OCCUPANCY,
    DEFAULT_STEP_BDAYS,
    DEFAULT_TEST_YEARS,
    DEFAULT_VOL_RATIO_THRESHOLD,
    average_dwell_length,
    evaluate_predictions,
    summarize_transition_stability,
)
from hmm_regime.train import (
    ARTIFACT_DIR_DEFAULT,
    DEFAULT_MIN_TRAIN_ROWS,
    DEFAULT_N_INIT,
    DEFAULT_N_ITER,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RETRAIN_CADENCE,
    DEFAULT_TRAIN_WINDOW,
    MARKET_NON_FEATURE_COLUMNS,
    apply_scaler,
    build_market_dataset,
    build_state_output,
    compute_dataset_start,
    compute_filtered_state_probs,
    compute_shift_probability,
    fit_hmm_bundle,
    save_artifacts,
    select_training_features,
)


DEFAULT_TRIALS = 8
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR_DEFAULT / "optuna_runs"
DEFAULT_STUDY_NAME = "hmm_regime_optuna"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize HMM regime hyperparameters with Optuna.")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today).")
    parser.add_argument("--test-years", type=int, default=DEFAULT_TEST_YEARS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR_DEFAULT))
    parser.add_argument("--auc-threshold", type=float, default=DEFAULT_AUC_THRESHOLD)
    parser.add_argument("--vol-ratio-threshold", type=float, default=DEFAULT_VOL_RATIO_THRESHOLD)
    parser.add_argument("--min-state-occupancy", type=float, default=DEFAULT_MIN_STATE_OCCUPANCY)
    parser.add_argument(
        "--skip-final-train",
        action="store_true",
        help="Skip retraining final market artifacts with the winning configuration.",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
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
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2))


def trial_to_result(trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    return {
        "number": int(trial.number),
        "value": (None if trial.value is None else float(trial.value)),
        "state": trial.state.name,
        "params": trial.params,
        "user_attrs": dict(trial.user_attrs),
    }


def sample_selected_features(trial: optuna.Trial, base_feature_columns: List[str]) -> List[str]:
    selected: List[str] = []
    for column in base_feature_columns:
        keep = trial.suggest_categorical(f"feat__{column}", [True, False])
        if keep:
            selected.append(column)
    return selected

def compute_objective(summary: Dict[str, Any]) -> float:
    gates = summary["gates"]
    alignment = summary["alignment_metrics"]
    drawdown_alignment = summary["drawdown_alignment"]

    regime_auc = float(gates["shift_prob_auc_regime_change_next_bday"]["value"] or 0.0)
    stress_vol_ratio = float(gates["stress_like_vol_ratio"]["value"] or 0.0)
    min_occupancy = float(min(gates["min_state_occupancy"]["value"].values()))
    vol_jump_auc = float(alignment["shift_prob_auc_vol_jump_5d"]["value"] or 0.0)
    stress_drawdown_share = float(drawdown_alignment["stress_share_in_deep_drawdown_decile"] or 0.0)
    transition_std_mean = float(summary["transition_stability"]["transition_std_mean"] or 0.0)
    total_folds = int(summary["n_predictions"]) + int(summary["n_skipped"])
    skipped_rate = float(summary["n_skipped"] / total_folds) if total_folds > 0 else 1.0

    score = (
        (3.0 * regime_auc)
        + (1.5 * vol_jump_auc)
        + (1.0 * min(stress_vol_ratio, 3.0) / 3.0)
        + (1.0 * stress_drawdown_share)
        + (0.5 * min_occupancy)
        - (2.0 * skipped_rate)
        - (0.5 * transition_std_mean)
    )
    if not summary["acceptance_pass"]:
        score -= 2.0
    return float(score)


def infer_state_for_date(
    bundle: Dict[str, Any],
    train_features: pd.DataFrame,
    dataset: pd.DataFrame,
    target_date: pd.Timestamp,
    feature_columns: List[str],
) -> pd.Series:
    eval_features = dataset.drop(columns=MARKET_NON_FEATURE_COLUMNS, errors="ignore")
    eval_features = eval_features.loc[
        (eval_features.index >= train_features.index.min()) & (eval_features.index <= target_date)
    ]
    eval_features = eval_features.loc[:, feature_columns].dropna()
    scaled = apply_scaler(eval_features, bundle["scaler"])
    state_probs = compute_filtered_state_probs(bundle["model"], scaled.values)
    transition_matrix = np.asarray(bundle["model"].transmat_, dtype=float)
    shift_probability = compute_shift_probability(state_probs, transition_matrix)
    states = build_state_output(
        index=eval_features.index,
        state_probs=state_probs,
        shift_probability=shift_probability,
        asof_date=target_date,
        stress_state_id=int(bundle["stress_state_id"]),
    )
    return states.iloc[-1]


def run_walk_forward(
    *,
    dataset: pd.DataFrame,
    end_date: pd.Timestamp,
    test_years: int,
    feature_columns: List[str],
    train_window: str,
    step_bdays: int,
    n_iter: int,
    n_init: int,
    random_state: int,
    min_train_rows: int,
    min_state_occupancy: float,
    auc_threshold: float,
    vol_ratio_threshold: float,
) -> Dict[str, Any]:
    test_start = end_date - pd.DateOffset(years=int(test_years))
    usable_idx = dataset.loc[:, feature_columns].dropna().index
    pred_dates = usable_idx[(usable_idx >= test_start) & (usable_idx <= end_date)]
    pred_dates = pred_dates[:: int(step_bdays)]

    prediction_rows: List[Dict[str, Any]] = []
    transition_matrices: List[np.ndarray] = []
    skipped: List[Dict[str, str]] = []

    for fold, asof_date in enumerate(pred_dates, start=1):
        try:
            train_features = select_training_features(
                dataset=dataset,
                asof_date=asof_date,
                train_window=train_window,
                min_train_rows=min_train_rows,
            )
            bundle = fit_hmm_bundle(
                train_features=train_features,
                train_targets=dataset.loc[train_features.index],
                n_iter=n_iter,
                random_state=random_state,
                n_init=n_init,
            )

            state_df = build_state_output(
                index=train_features.index,
                state_probs=bundle["state_probs"],
                shift_probability=bundle["shift_probability"],
                asof_date=asof_date,
                stress_state_id=int(bundle["stress_state_id"]),
            )
            latest = state_df.iloc[-1].to_dict()
            latest["date"] = pd.Timestamp(latest["date"])
            latest["asof"] = pd.Timestamp(latest["asof"])
            latest["train_start"] = pd.Timestamp(train_features.index.min())
            latest["train_end"] = pd.Timestamp(train_features.index.max())
            latest["fold"] = int(fold)

            next_pos = usable_idx.searchsorted(asof_date, side="right")
            if next_pos < len(usable_idx):
                next_business_date = pd.Timestamp(usable_idx[next_pos])
                next_state = infer_state_for_date(
                    bundle,
                    train_features,
                    dataset,
                    next_business_date,
                    feature_columns,
                )
                latest["next_business_date"] = next_business_date
                latest["next_business_state_label"] = str(next_state["state_label"])
            else:
                latest["next_business_date"] = pd.NaT
                latest["next_business_state_label"] = None
            prediction_rows.append(latest)
            transition_matrices.append(np.asarray(bundle["transition_matrix"], dtype=float))
        except Exception as exc:
            skipped.append(
                {
                    "fold": str(fold),
                    "asof": pd.Timestamp(asof_date).date().isoformat(),
                    "reason": str(exc),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ValueError("All optimization folds were skipped.")
    predictions = predictions.sort_values("date").reset_index(drop=True)

    eval_report = evaluate_predictions(
        predictions=predictions,
        dataset=dataset,
        min_state_occupancy=min_state_occupancy,
        auc_threshold=auc_threshold,
        vol_ratio_threshold=vol_ratio_threshold,
    )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "feature_columns": list(feature_columns),
        "feature_count": int(len(feature_columns)),
        "train_window": train_window,
        "step_bdays": int(step_bdays),
        "n_iter": int(n_iter),
        "n_init": int(n_init),
        "random_state": int(random_state),
        "min_train_rows": int(min_train_rows),
        "test_start": str(pd.Timestamp(test_start).date()),
        "test_end": str(pd.Timestamp(end_date).date()),
        "n_predictions": int(len(predictions)),
        "n_skipped": int(len(skipped)),
        "acceptance_pass": bool(eval_report["acceptance_pass"]),
        "gates": eval_report["gates"],
        "alignment_metrics": eval_report["alignment_metrics"],
        "transition_stability": summarize_transition_stability(transition_matrices),
        "persistence": average_dwell_length(predictions["state_label"].tolist()),
        "drawdown_alignment": eval_report["drawdown_alignment"],
        "skipped_samples": skipped[:20],
    }
    summary["objective_score"] = compute_objective(summary)
    return summary


def maybe_retrain_final(
    *,
    artifact_dir: Path,
    end_date: pd.Timestamp,
    feature_columns: List[str],
    train_window: str,
    n_iter: int,
    n_init: int,
    random_state: int,
    min_train_rows: int,
) -> Dict[str, Any]:
    start_date = compute_dataset_start(end_date, train_window, test_years=0)
    dataset = build_market_dataset(start_date, end_date)
    usable = dataset.loc[:, feature_columns].dropna()
    asof_date = usable.index.max()
    train_features = select_training_features(
        dataset=dataset,
        asof_date=asof_date,
        train_window=train_window,
        min_train_rows=min_train_rows,
    )
    bundle = fit_hmm_bundle(
        train_features=train_features,
        train_targets=dataset.loc[train_features.index],
        n_iter=n_iter,
        random_state=random_state,
        n_init=n_init,
    )
    train_states = build_state_output(
        index=train_features.index,
        state_probs=bundle["state_probs"],
        shift_probability=bundle["shift_probability"],
        asof_date=asof_date,
        stress_state_id=int(bundle["stress_state_id"]),
    )
    save_artifacts(
        artifact_dir=artifact_dir,
        bundle=bundle,
        train_states=train_states,
        train_window=train_window,
        retrain_cadence=DEFAULT_RETRAIN_CADENCE,
        n_iter=n_iter,
        n_init=n_init,
        random_state=random_state,
    )
    return {
        "feature_columns": list(feature_columns),
        "feature_count": int(len(feature_columns)),
        "train_start": str(pd.Timestamp(bundle["train_start"]).date()),
        "train_end": str(pd.Timestamp(bundle["train_end"]).date()),
        "selected_seed": int(bundle["seed"]),
        "candidate_count": int(bundle["candidate_count"]),
        "artifact_dir": str(artifact_dir),
    }


def main() -> None:
    args = parse_args()
    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache: Dict[Tuple[str, int], pd.DataFrame] = {}

    def get_dataset(train_window: str, test_years: int) -> pd.DataFrame:
        key = (train_window, int(test_years))
        if key not in dataset_cache:
            start_date = compute_dataset_start(end_date, train_window, test_years=test_years)
            dataset_cache[key] = build_market_dataset(start_date, end_date)
        return dataset_cache[key]

    base_feature_columns = list(
        get_dataset(DEFAULT_TRAIN_WINDOW, args.test_years)
        .drop(columns=MARKET_NON_FEATURE_COLUMNS, errors="ignore")
        .columns
    )

    baseline_config = {
        "feature_columns": list(base_feature_columns),
        "train_window": DEFAULT_TRAIN_WINDOW,
        "step_bdays": DEFAULT_STEP_BDAYS,
        "n_iter": DEFAULT_N_ITER,
        "n_init": DEFAULT_N_INIT,
        "random_state": DEFAULT_RANDOM_STATE,
        "min_train_rows": DEFAULT_MIN_TRAIN_ROWS,
    }
    baseline_summary = run_walk_forward(
        dataset=get_dataset(baseline_config["train_window"], args.test_years),
        end_date=end_date,
        test_years=args.test_years,
        min_state_occupancy=args.min_state_occupancy,
        auc_threshold=args.auc_threshold,
        vol_ratio_threshold=args.vol_ratio_threshold,
        **baseline_config,
    )
    write_json(output_dir / "baseline_summary.json", baseline_summary)

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        config = {
            "feature_columns": sample_selected_features(trial, base_feature_columns),
            "train_window": trial.suggest_categorical("train_window", ["2y", "3y", "4y"]),
            "step_bdays": trial.suggest_categorical("step_bdays", [3, 5, 10]),
            "n_iter": trial.suggest_categorical("n_iter", [250, 500, 750]),
            "n_init": trial.suggest_categorical("n_init", [4, 6, 8]),
            "random_state": trial.suggest_int("random_state", 0, 96, step=8),
            "min_train_rows": trial.suggest_categorical("min_train_rows", [252, 378, 504]),
        }
        trial.set_user_attr("selected_features", config["feature_columns"])
        if not config["feature_columns"]:
            raise optuna.TrialPruned("No features selected.")
        try:
            summary = run_walk_forward(
                dataset=get_dataset(config["train_window"], args.test_years),
                end_date=end_date,
                test_years=args.test_years,
                min_state_occupancy=args.min_state_occupancy,
                auc_threshold=args.auc_threshold,
                vol_ratio_threshold=args.vol_ratio_threshold,
                **config,
            )
        except Exception as exc:
            trial.set_user_attr("feature_error", str(exc))
            raise optuna.TrialPruned(str(exc))
        trial.set_user_attr("summary", summary)
        return float(summary["objective_score"])

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]
    top_trials = sorted(completed, key=lambda item: item.value, reverse=True)[:10]
    trial_payload = [trial_to_result(trial) for trial in top_trials]
    write_json(output_dir / "optuna_trials_top.json", {"trials": trial_payload})
    if not completed:
        raise ValueError(
            "No optimization trials completed successfully. "
            "Increase --trials or relax feature guardrails."
        )

    best_trial = study.best_trial
    best_summary = best_trial.user_attrs["summary"]
    best_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "end_date": str(pd.Timestamp(end_date).date()),
        "baseline_summary": baseline_summary,
        "best_trial": trial_to_result(best_trial),
        "best_summary": best_summary,
    }

    if not args.skip_final_train:
        final_train = maybe_retrain_final(
            artifact_dir=Path(args.artifact_dir),
            end_date=end_date,
            feature_columns=list(best_trial.user_attrs.get("selected_features", base_feature_columns)),
            train_window=best_trial.params["train_window"],
            n_iter=int(best_trial.params["n_iter"]),
            n_init=int(best_trial.params["n_init"]),
            random_state=int(best_trial.params["random_state"]),
            min_train_rows=int(best_trial.params["min_train_rows"]),
        )
        best_payload["final_train"] = final_train

    write_json(output_dir / "best_result.json", best_payload)
    print(f"Baseline objective: {baseline_summary['objective_score']:.6f}")
    print(f"Best objective: {best_summary['objective_score']:.6f}")
    print(f"Best params: {json.dumps(best_trial.params, sort_keys=True)}")
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    np.random.seed(DEFAULT_RANDOM_STATE)
    main()
