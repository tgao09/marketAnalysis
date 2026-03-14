import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import parse_window, walk_forward_splits
from gbm_return.configuration import (
    FEATURE_SET_CHOICES,
    FEATURE_SET_F0,
    resolve_lgbm_params,
)
from gbm_return.train import (
    ARTIFACT_DIR_DEFAULT,
    DATA_YEARS,
    DEFAULT_STEP_WINDOW,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
    FEATURE_LOOKBACK_MAX,
    MIN_TRAIN_ROWS,
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    REGIME_SCORE_WINDOW,
    WINDOW_RET,
    build_features,
    build_target,
    compute_regime_score,
    evaluate,
    extract_field,
    fetch_history_cached,
    prepare_lgbm_training_data,
    resolve_sector_etf,
    select_feature_columns,
    set_time_index,
    summarize_fold_metrics,
    train_lgbm,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Leave-one-out feature ablation for return GBM.")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--train-window", default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--test-window", default=DEFAULT_TEST_WINDOW)
    parser.add_argument("--step-window", default=DEFAULT_STEP_WINDOW)
    parser.add_argument(
        "--include-time-index",
        action="store_true",
        help="Include time_index in features (default is to exclude).",
    )
    parser.add_argument(
        "--feature-set",
        default=FEATURE_SET_F0,
        choices=FEATURE_SET_CHOICES,
        help="Feature set variant. F1/F2 are loaded from --feature-set-file.",
    )
    parser.add_argument(
        "--feature-set-file",
        default=str(ARTIFACT_DIR_DEFAULT / "feature_sets.json"),
        help="Path to feature_sets.json used for F1/F2 feature drops.",
    )
    parser.add_argument(
        "--lgbm-param-preset",
        default="baseline",
        help="Named LightGBM preset from gbm_return.configuration.",
    )
    parser.add_argument(
        "--lgbm-params-json",
        default=None,
        help="Optional JSON file with LightGBM params to merge on top of preset.",
    )
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR_DEFAULT))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prompt_tickers():
    raw = input("Tickers for feature ablation (comma/space separated): ").strip()
    tokens = [token.strip().upper() for token in re.split(r"[,\s]+", raw) if token.strip()]
    seen = set()
    tickers = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            tickers.append(token)
    return tickers


def compute_dataset_start(end_date: pd.Timestamp, train_window: str):
    train_offset = parse_window(train_window)
    base_start = end_date - pd.DateOffset(years=DATA_YEARS)
    buffer_days = max(FEATURE_LOOKBACK_MAX, REGIME_SCORE_WINDOW) + (2 * WINDOW_RET) + 5
    min_start = end_date - train_offset - pd.DateOffset(days=buffer_days)
    return min(base_start, min_start)


def build_dataset(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    history_cache: dict[str, pd.DataFrame] = {}
    sector_etf, sector_name, sector_error = resolve_sector_etf(ticker)
    if sector_error:
        print(f"{ticker}: sector fallback to {sector_etf} ({sector_error})")

    stock_history = fetch_history_cached(ticker, start_date, end_date, history_cache)
    sector_history = fetch_history_cached(sector_etf, start_date, end_date, history_cache)
    gld_history = fetch_history_cached("GLD", start_date, end_date, history_cache)
    spy_history = fetch_history_cached("SPY", start_date, end_date, history_cache)
    vix_history = fetch_history_cached("^VIX", start_date, end_date, history_cache)

    price_stock = extract_field(stock_history, "Close", ticker)
    price_sector = extract_field(sector_history, "Close", sector_etf)
    price_gld = extract_field(gld_history, "Close", "GLD")
    price_spy = extract_field(spy_history, "Close", "SPY")
    price_vix = extract_field(vix_history, "Close", "^VIX")

    features = build_features(price_stock, price_sector, price_gld, price_spy, price_vix)
    target = build_target(price_stock)
    regime_score = compute_regime_score(
        price_vix.reindex(price_stock.index).ffill(),
        price_spy.reindex(price_stock.index).ffill(),
        REGIME_SCORE_WINDOW,
        REGIME_SCORE_CLIP,
        REGIME_SCORE_WEIGHTS,
    )
    dataset = features.join([target])
    dataset["regime_score"] = regime_score
    dataset = dataset.dropna()
    if dataset.empty:
        raise ValueError(f"{ticker}: No rows left after feature/target alignment.")
    if dataset.index.has_duplicates:
        dataset = dataset.loc[~dataset.index.duplicated(keep="last")]
    return {"dataset": dataset, "sector_etf": sector_etf, "sector_name": sector_name}


def select_last_6m_folds(all_splits, end_date, test_window, step_window):
    latest_available_end = max(split.test_end for split in all_splits)
    eval_end = min(end_date, latest_available_end)
    if test_window == "1m" and step_window == "1m":
        all_sorted = sorted(all_splits, key=lambda s: s.test_end)
        selected = all_sorted[-6:]
        if len(selected) < 6:
            dates = [f"fold{split.fold}:{split.test_start.date()}->{split.test_end.date()}" for split in all_sorted]
            raise ValueError(
                "Last-6-month fold enforcement failed. "
                f"Expected 6 monthly folds but found {len(selected)}. "
                f"All test windows: {dates}"
            )
        eval_start = selected[0].test_start
    else:
        eval_start = eval_end - pd.DateOffset(months=6)
        selected = [split for split in all_splits if split.test_start >= eval_start and split.test_end <= eval_end]
        if not selected:
            raise ValueError(
                f"No walk-forward folds found within last 6 months ({eval_start.date()}..{eval_end.date()})."
            )
    return eval_start, eval_end, selected


def run_candidate(candidate_feature_cols, selected_splits, lgbm_params):
    fold_metrics = []
    gain_frames = []
    split_frames = []
    shap_abs_frames = []

    for split in selected_splits:
        train_df = set_time_index(split.train.copy(), split.train_start)
        test_df = set_time_index(split.test.copy(), split.train_start)
        train_x, train_y, sample_weight, _ = prepare_lgbm_training_data(
            train_df,
            candidate_feature_cols,
        )
        test_x = test_df[candidate_feature_cols]
        test_y = test_df["target"]

        model = train_lgbm(train_x, train_y, lgbm_params, sample_weight=sample_weight)
        metrics = evaluate(model, test_x, test_y)
        fold_metrics.append(
            {
                "fold": split.fold,
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "mae_simple": metrics["mae_simple"],
                "directional": metrics["directional"],
                "corr": metrics["corr"],
                "coverage_95": metrics["coverage_95"],
                "avg_interval_width": metrics["avg_interval_width"],
            }
        )

        booster = model.booster_
        gain = pd.Series(booster.feature_importance(importance_type="gain"), index=candidate_feature_cols)
        split_imp = pd.Series(
            booster.feature_importance(importance_type="split"),
            index=candidate_feature_cols,
        )
        gain_frames.append(gain)
        split_frames.append(split_imp)

        contrib = booster.predict(test_x, pred_contrib=True)
        if contrib.ndim == 1:
            contrib = contrib.reshape(1, -1)
        # Last column is bias term.
        shap_values = np.abs(contrib[:, :-1]).mean(axis=0)
        shap_abs_frames.append(pd.Series(shap_values, index=candidate_feature_cols))

    summary = summarize_fold_metrics(fold_metrics)
    mean_gain = pd.concat(gain_frames, axis=1).mean(axis=1).sort_values(ascending=False)
    mean_split = pd.concat(split_frames, axis=1).mean(axis=1).sort_values(ascending=False)
    mean_shap = pd.concat(shap_abs_frames, axis=1).mean(axis=1).sort_values(ascending=False)
    return summary, mean_gain.to_dict(), mean_split.to_dict(), mean_shap.to_dict()


def metric_triplet(summary_metrics):
    return {
        "directional_mean": summary_metrics["directional_mean"],
        "mae_simple_mean": summary_metrics["mae_simple_mean"],
        "coverage_95_mean": summary_metrics["coverage_95_mean"],
    }


def deltas_from_baseline(candidate_metrics, baseline_metrics):
    directional_delta = candidate_metrics["directional_mean"] - baseline_metrics["directional_mean"]
    mae_simple_delta = candidate_metrics["mae_simple_mean"] - baseline_metrics["mae_simple_mean"]
    coverage_95_base = baseline_metrics["coverage_95_mean"]
    coverage_95_cand = candidate_metrics["coverage_95_mean"]
    coverage_95_delta = None
    if coverage_95_base is not None and coverage_95_cand is not None:
        coverage_95_delta = coverage_95_cand - coverage_95_base

    mae_simple_improvement = baseline_metrics["mae_simple_mean"] - candidate_metrics["mae_simple_mean"]
    directional_improvement = directional_delta
    if coverage_95_base is not None and coverage_95_cand is not None:
        coverage_target_improvement = abs(coverage_95_base - 0.95) - abs(coverage_95_cand - 0.95)
    else:
        coverage_target_improvement = 0.0

    return {
        "deltas": {
            "directional_delta": directional_delta,
            "mae_simple_delta": mae_simple_delta,
            "coverage_95_delta": coverage_95_delta,
        },
        "improvements": {
            "mae_simple_improvement": mae_simple_improvement,
            "directional_improvement": directional_improvement,
            "coverage_target_improvement": coverage_target_improvement,
        },
    }


def build_ranked_outputs(results, baseline_metrics):
    successes = [item for item in results if item["status"] == "ok"]
    failures = [item for item in results if item["status"] != "ok"]
    for item in successes:
        extra = deltas_from_baseline(item["candidate_metrics"], baseline_metrics)
        item["deltas"] = extra["deltas"]
        item["improvements"] = extra["improvements"]

    leaderboard_sorted = sorted(
        successes,
        key=lambda x: (
            -x["improvements"]["mae_simple_improvement"],
            -x["improvements"]["directional_improvement"],
            -x["improvements"]["coverage_target_improvement"],
            x["dropped_feature"],
        ),
    )
    leaderboard = []
    for rank, item in enumerate(leaderboard_sorted, start=1):
        leaderboard.append(
            {
                "rank": rank,
                "dropped_feature": item["dropped_feature"],
                "candidate_metrics": item["candidate_metrics"],
                "deltas": item["deltas"],
                "improvements": item["improvements"],
            }
        )

    per_feature_deltas = [
        {
            "dropped_feature": item["dropped_feature"],
            "status": "ok",
            "deltas": item["deltas"],
            "improvements": item["improvements"],
        }
        for item in sorted(successes, key=lambda x: x["dropped_feature"])
    ]
    failed_diagnostics = []
    for item in sorted(failures, key=lambda x: x["dropped_feature"]):
        failed_diagnostics.append({"dropped_feature": item["dropped_feature"], "error": item["error"]})
        per_feature_deltas.append(
            {
                "dropped_feature": item["dropped_feature"],
                "status": "failed",
                "error": item["error"],
            }
        )
    return leaderboard, per_feature_deltas, failed_diagnostics


def write_ablation_json(
    output_dir,
    ticker,
    eval_start,
    eval_end,
    args,
    baseline_feature_cols,
    baseline_metrics,
    baseline_gain,
    baseline_split,
    baseline_shap,
    selected_splits,
    leaderboard,
    per_feature_deltas,
    failed_diagnostics,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ablation.json"
    payload = {
        "metadata": {
            "ticker": ticker,
            "generated_at": datetime.now(UTC).isoformat(),
            "eval_start_date": str(eval_start.date()),
            "eval_end_date": str(eval_end.date()),
            "train_window": args.train_window,
            "test_window": args.test_window,
            "step_window": args.step_window,
            "include_time_index": args.include_time_index,
            "feature_set": args.feature_set,
            "feature_set_file": args.feature_set_file,
            "lgbm_param_preset": args.lgbm_param_preset,
            "lgbm_params_json": args.lgbm_params_json,
            "fold_count": len(selected_splits),
            "features_tested_count": len(baseline_feature_cols),
        },
        "baseline_metrics": baseline_metrics,
        "baseline_importance": {
            "gain": baseline_gain,
            "split": baseline_split,
        },
        "baseline_shap_summary": baseline_shap,
        "leaderboard": leaderboard,
        "per_feature_deltas": per_feature_deltas,
        "failed_feature_diagnostics": failed_diagnostics,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main():
    args = parse_args()
    tickers = prompt_tickers()

    np.random.seed(args.seed)
    lgbm_params = resolve_lgbm_params(
        preset_name=args.lgbm_param_preset,
        params_json=args.lgbm_params_json,
    )
    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()

    for ticker in tickers:
        dataset_start = compute_dataset_start(end_date, args.train_window)
        print(f"\nBuilding dataset for {ticker}...")
        data = build_dataset(ticker, dataset_start, end_date)
        dataset = data["dataset"]
        baseline_feature_cols = select_feature_columns(
            dataset=dataset,
            drop_time_index=not args.include_time_index,
            feature_set=args.feature_set,
            feature_set_file=args.feature_set_file,
        )

        all_splits = list(
            walk_forward_splits(
                dataset,
                train_window=args.train_window,
                test_window=args.test_window,
                embargo=WINDOW_RET,
                step=args.step_window,
                min_train_rows=MIN_TRAIN_ROWS,
            )
        )

        eval_start, eval_end, selected_splits = select_last_6m_folds(
            all_splits,
            end_date,
            args.test_window,
            args.step_window,
        )
        print(
            f"Selected {len(selected_splits)} folds within last 6 months "
            f"({eval_start.date()}..{eval_end.date()})."
        )

        baseline_summary, baseline_gain, baseline_split, baseline_shap = run_candidate(
            baseline_feature_cols,
            selected_splits,
            lgbm_params,
        )
        baseline_metrics = metric_triplet(baseline_summary)
        print(
            "Baseline metrics | "
            f"Dir mean: {baseline_metrics['directional_mean']:.2%} | "
            f"MAE(simple) mean: {baseline_metrics['mae_simple_mean']:.4%}"
        )

        results = []
        for idx, feature in enumerate(baseline_feature_cols, start=1):
            candidate_feature_cols = [col for col in baseline_feature_cols if col != feature]
            print(f"[{idx}/{len(baseline_feature_cols)}] Testing drop: {feature}")
            candidate_summary, _, _, _ = run_candidate(
                candidate_feature_cols,
                selected_splits,
                lgbm_params,
            )
            results.append(
                {
                    "dropped_feature": feature,
                    "status": "ok",
                    "candidate_metrics": metric_triplet(candidate_summary),
                }
            )

        leaderboard, per_feature_deltas, failed_diagnostics = build_ranked_outputs(results, baseline_metrics)
        ticker_output_dir = Path(args.output_dir) / ticker
        out_path = write_ablation_json(
            ticker_output_dir,
            ticker,
            eval_start,
            eval_end,
            args,
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
        print(f"\nAblation saved to: {out_path}")
        print(f"Leaderboard rows: {len(leaderboard)}")
        print(f"Per-feature delta rows: {len(per_feature_deltas)}")


if __name__ == "__main__":
    main()
