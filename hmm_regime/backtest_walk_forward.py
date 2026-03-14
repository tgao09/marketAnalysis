import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from hmm_regime.train import (
    ARTIFACT_DIR_DEFAULT,
    DEFAULT_MIN_TRAIN_ROWS,
    DEFAULT_N_INIT,
    DEFAULT_N_ITER,
    MARKET_NON_FEATURE_COLUMNS,
    N_STATES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RETRAIN_CADENCE,
    DEFAULT_TRAIN_WINDOW,
    apply_scaler,
    build_market_dataset,
    build_state_output,
    compute_filtered_state_probs,
    compute_dataset_start,
    compute_shift_probability,
    fit_hmm_bundle,
    select_training_features,
    state_label,
)


DEFAULT_TEST_YEARS = 1
DEFAULT_STEP_BDAYS = 5
DEFAULT_AUC_THRESHOLD = 0.55
DEFAULT_VOL_RATIO_THRESHOLD = 1.25
DEFAULT_MIN_STATE_OCCUPANCY = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly walk-forward evaluation for market HMM regime model.")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today).")
    parser.add_argument("--test-years", type=int, default=DEFAULT_TEST_YEARS)
    parser.add_argument("--train-window", default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--step-bdays", type=int, default=DEFAULT_STEP_BDAYS)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--n-init", type=int, default=DEFAULT_N_INIT)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    parser.add_argument("--min-state-occupancy", type=float, default=DEFAULT_MIN_STATE_OCCUPANCY)
    parser.add_argument("--auc-threshold", type=float, default=DEFAULT_AUC_THRESHOLD)
    parser.add_argument("--vol-ratio-threshold", type=float, default=DEFAULT_VOL_RATIO_THRESHOLD)
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR_DEFAULT))
    return parser.parse_args()


def average_dwell_length(labels: List[str]) -> Dict[str, float]:
    if not labels:
        return {"avg_dwell": 0.0, "max_dwell": 0.0}
    run_lengths: List[int] = []
    prev = labels[0]
    current_len = 1
    for label in labels[1:]:
        if label == prev:
            current_len += 1
        else:
            run_lengths.append(current_len)
            current_len = 1
            prev = label
    run_lengths.append(current_len)
    return {
        "avg_dwell": float(np.mean(run_lengths)),
        "max_dwell": float(np.max(run_lengths)),
    }


def summarize_transition_stability(matrices: List[np.ndarray]) -> Dict[str, float | None]:
    if len(matrices) < 2:
        return {"transition_std_mean": None, "transition_std_max": None}
    arr = np.stack(matrices, axis=0)
    std = arr.std(axis=0)
    return {
        "transition_std_mean": float(std.mean()),
        "transition_std_max": float(std.max()),
    }


def binary_auc_metric(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str,
    threshold: float | None = None,
) -> Dict[str, float | int | bool | None]:
    metric: Dict[str, float | int | bool | None] = {
        "value": None,
        "threshold": threshold,
        "pass": False,
        "n": 0,
        "positive_rate": None,
    }
    auc_df = frame.dropna(subset=[score_col, target_col])
    metric["n"] = int(len(auc_df))
    if auc_df.empty:
        return metric

    target = auc_df[target_col].astype(int)
    metric["positive_rate"] = float(target.mean())
    if target.nunique() < 2:
        return metric

    value = float(roc_auc_score(target, auc_df[score_col]))
    metric["value"] = value
    if threshold is not None:
        metric["pass"] = bool(value >= threshold)
    return metric


def evaluate_predictions(
    predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    min_state_occupancy: float,
    auc_threshold: float,
    vol_ratio_threshold: float,
) -> Dict[str, object]:
    eval_cols = ["forward_ret_5d", "forward_vol_5d", "drawdown", "vol_jump_5d"]
    eval_frame = predictions.set_index("date").join(dataset[eval_cols], how="left").sort_index()
    eval_frame["regime_change_next_bday"] = (
        eval_frame["state_label"] != eval_frame["next_business_state_label"]
    ).astype(float)
    eval_frame.loc[eval_frame["next_business_state_label"].isna(), "regime_change_next_bday"] = np.nan

    regime_change_auc = binary_auc_metric(
        frame=eval_frame,
        score_col="shift_prob",
        target_col="regime_change_next_bday",
        threshold=auc_threshold,
    )
    vol_jump_auc = binary_auc_metric(
        frame=eval_frame,
        score_col="shift_prob",
        target_col="vol_jump_5d",
    )

    occupancy = {
        state_label(state_id): float((predictions["state_id"] == state_id).mean()) if not predictions.empty else 0.0
        for state_id in range(N_STATES)
    }
    occupancy_pass = all(value >= min_state_occupancy for value in occupancy.values())

    vol_df = eval_frame.dropna(subset=["forward_vol_5d"])
    stress_conditional_vol = None
    unconditional_vol = None
    stress_vol_ratio = None
    stress_vol_pass = False
    if not vol_df.empty:
        unconditional_vol = float(vol_df["forward_vol_5d"].mean())
        stress_rows = vol_df[vol_df["state_id"] == vol_df["stress_state_id"]]
        if not stress_rows.empty and unconditional_vol > 0:
            stress_conditional_vol = float(stress_rows["forward_vol_5d"].mean())
            stress_vol_ratio = float(stress_conditional_vol / unconditional_vol)
            stress_vol_pass = bool(stress_vol_ratio >= vol_ratio_threshold)

    dd_df = eval_frame.dropna(subset=["drawdown"])
    drawdown_alignment = {
        "stress_mean_drawdown": None,
        "overall_mean_drawdown": None,
        "stress_share_in_deep_drawdown_decile": None,
    }
    if not dd_df.empty:
        drawdown_alignment["overall_mean_drawdown"] = float(dd_df["drawdown"].mean())
        stress_dd = dd_df.loc[dd_df["state_id"] == dd_df["stress_state_id"], "drawdown"]
        if not stress_dd.empty:
            drawdown_alignment["stress_mean_drawdown"] = float(stress_dd.mean())
        deep_cut = float(dd_df["drawdown"].quantile(0.10))
        deep_mask = dd_df["drawdown"] <= deep_cut
        if deep_mask.any():
            drawdown_alignment["stress_share_in_deep_drawdown_decile"] = float(
                (dd_df.loc[deep_mask, "state_id"] == dd_df.loc[deep_mask, "stress_state_id"]).mean()
            )

    acceptance_pass = bool(regime_change_auc["pass"] and occupancy_pass and stress_vol_pass)
    return {
        "eval_frame": eval_frame.reset_index(),
        "acceptance_pass": acceptance_pass,
        "gates": {
            "shift_prob_auc_regime_change_next_bday": regime_change_auc,
            "stress_like_vol_ratio": {
                "label": "dynamic_stress_state",
                "value": stress_vol_ratio,
                "threshold": vol_ratio_threshold,
                "pass": stress_vol_pass,
                "stress_conditional_vol": stress_conditional_vol,
                "unconditional_vol": unconditional_vol,
            },
            "min_state_occupancy": {
                "value": occupancy,
                "threshold": min_state_occupancy,
                "pass": occupancy_pass,
            },
        },
        "alignment_metrics": {
            "shift_prob_auc_vol_jump_5d": vol_jump_auc,
        },
        "drawdown_alignment": drawdown_alignment,
    }


def infer_state_for_date(
    bundle: Dict[str, object],
    train_features: pd.DataFrame,
    dataset: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.Series:
    eval_features = dataset.drop(columns=MARKET_NON_FEATURE_COLUMNS, errors="ignore")
    eval_features = eval_features.loc[
        (eval_features.index >= train_features.index.min()) & (eval_features.index <= target_date)
    ].dropna()
    if bundle.get("feature_columns"):
        eval_features = eval_features.loc[:, list(bundle["feature_columns"])]
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


def main() -> None:
    args = parse_args()
    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    start_date = compute_dataset_start(end_date, args.train_window, test_years=args.test_years)
    test_start = end_date - pd.DateOffset(years=args.test_years)

    print(f"Building dataset from {start_date.date()} to {end_date.date()}...")
    dataset = build_market_dataset(start_date, end_date)
    usable_features = dataset.drop(columns=MARKET_NON_FEATURE_COLUMNS, errors="ignore")
    usable_idx = usable_features.dropna().index
    pred_dates = usable_idx[(usable_idx >= test_start) & (usable_idx <= end_date)]
    pred_dates = pred_dates[:: args.step_bdays]

    prediction_rows: List[Dict[str, object]] = []
    transition_rows: List[Dict[str, object]] = []
    transition_matrices: List[np.ndarray] = []
    skipped: List[Dict[str, str]] = []

    for fold, asof_date in enumerate(pred_dates, start=1):
        try:
            train_features = select_training_features(
                dataset=dataset,
                asof_date=asof_date,
                train_window=args.train_window,
                min_train_rows=args.min_train_rows,
            )
            bundle = fit_hmm_bundle(
                train_features=train_features,
                train_targets=dataset.loc[train_features.index],
                n_iter=args.n_iter,
                random_state=args.random_state,
                n_init=args.n_init,
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
                next_state = infer_state_for_date(bundle, train_features, dataset, next_business_date)
                latest["next_business_date"] = next_business_date
                latest["next_business_state_label"] = str(next_state["state_label"])
            else:
                latest["next_business_date"] = pd.NaT
                latest["next_business_state_label"] = None
            prediction_rows.append(latest)

            trans = np.asarray(bundle["transition_matrix"], dtype=float)
            transition_matrices.append(trans)
            for from_id in range(N_STATES):
                for to_id in range(N_STATES):
                    transition_rows.append(
                        {
                            "fold": int(fold),
                            "asof": pd.Timestamp(asof_date).date().isoformat(),
                            "from_state_id": int(from_id),
                            "from_state_label": state_label(from_id),
                            "to_state_id": int(to_id),
                            "to_state_label": state_label(to_id),
                            "probability": float(trans[from_id, to_id]),
                        }
                    )
        except Exception as exc:
            skipped.append(
                {
                    "fold": str(fold),
                    "asof": pd.Timestamp(asof_date).date().isoformat(),
                    "reason": str(exc),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    transition_df = pd.DataFrame(transition_rows)
    if predictions.empty:
        raise ValueError("All walk-forward folds were skipped.")
    predictions = predictions.sort_values("date").reset_index(drop=True)

    eval_report = evaluate_predictions(
        predictions=predictions,
        dataset=dataset,
        min_state_occupancy=args.min_state_occupancy,
        auc_threshold=args.auc_threshold,
        vol_ratio_threshold=args.vol_ratio_threshold,
    )
    eval_frame = eval_report["eval_frame"]
    transition_stability = summarize_transition_stability(transition_matrices)
    dwell = average_dwell_length(predictions["state_label"].tolist())

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_type": "GaussianHMM",
        "n_states": N_STATES,
        "feature_columns": list(usable_features.columns),
        "train_window": args.train_window,
        "retrain_cadence": DEFAULT_RETRAIN_CADENCE,
        "test_start": str(pd.Timestamp(test_start).date()),
        "test_end": str(pd.Timestamp(end_date).date()),
        "step_bdays": int(args.step_bdays),
        "n_init": int(args.n_init),
        "n_predictions": int(len(predictions)),
        "n_skipped": int(len(skipped)),
        "acceptance_pass": bool(eval_report["acceptance_pass"]),
        "gates": eval_report["gates"],
        "alignment_metrics": eval_report["alignment_metrics"],
        "transition_stability": transition_stability,
        "persistence": dwell,
        "drawdown_alignment": eval_report["drawdown_alignment"],
        "skipped_samples": skipped[:20],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = output_dir / "walk_forward_states.csv"
    eval_path = output_dir / "walk_forward_eval.csv"
    transition_path = output_dir / "transition_matrices.csv"
    summary_path = output_dir / "walk_forward_summary.json"

    ordered_cols = [
        "date",
        "state_id",
        "state_label",
        "p_state_0",
        "p_state_1",
        "p_state_2",
        "p_state_3",
        "shift_prob",
        "next_business_date",
        "next_business_state_label",
        "asof",
        "fold",
        "train_start",
        "train_end",
    ]
    predictions[ordered_cols].to_csv(states_path, index=False)
    eval_frame.to_csv(eval_path, index=False)
    transition_df.to_csv(transition_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"States saved to: {states_path}")
    print(f"Eval rows saved to: {eval_path}")
    print(f"Transition matrices saved to: {transition_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Acceptance pass: {summary['acceptance_pass']}")


if __name__ == "__main__":
    np.random.seed(DEFAULT_RANDOM_STATE)
    main()
