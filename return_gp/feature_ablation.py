import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import parse_window, walk_forward_splits
from return_gp.train import (
    ARTIFACT_DIR_DEFAULT,
    DATA_YEARS,
    DEFAULT_STEP_WINDOW,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_ITERS,
    DEFAULT_TRAIN_WINDOW,
    FEATURE_LOOKBACK_MAX,
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    REGIME_SCORE_WINDOW,
    WINDOW_RET,
    build_features,
    build_target,
    compute_regime_score,
    extract_field,
    fetch_history_cached,
    normalize_features,
    resolve_sector_etf,
    resolve_device,
    set_time_index,
    summarize_fold_metrics,
    train_gp,
    evaluate,
)


MIN_TRAIN_ROWS = 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leave-one-out feature ablation for return GP."
    )
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--train-window", default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--test-window", default=DEFAULT_TEST_WINDOW)
    parser.add_argument("--step-window", default=DEFAULT_STEP_WINDOW)
    parser.add_argument("--train-iters", type=int, default=DEFAULT_TRAIN_ITERS)
    parser.add_argument(
        "--include-time-index",
        action="store_true",
        help="Include time_index in features (default is to exclude).",
    )
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR_DEFAULT))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prompt_tickers():
    raw = input("Tickers for feature ablation (comma/space separated): ").strip()
    if not raw:
        return []

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
    volume_stock = extract_field(stock_history, "Volume", ticker)
    price_sector = extract_field(sector_history, "Close", sector_etf)
    price_gld = extract_field(gld_history, "Close", "GLD")
    price_spy = extract_field(spy_history, "Close", "SPY")
    price_vix = extract_field(vix_history, "Close", "^VIX")

    features = build_features(
        price_stock,
        volume_stock,
        price_sector,
        price_gld,
        price_spy,
        price_vix,
    )
    target = build_target(price_stock)

    # Keep regime inputs causal to avoid leaking future values.
    price_spy_regime = price_spy.reindex(price_stock.index).ffill()
    price_vix_regime = price_vix.reindex(price_stock.index).ffill()
    regime_score = compute_regime_score(
        price_vix_regime,
        price_spy_regime,
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

    return {
        "dataset": dataset,
        "sector_etf": sector_etf,
        "sector_name": sector_name,
    }


def select_last_6m_folds(all_splits, end_date, test_window, step_window):
    latest_available_end = max(split.test_end for split in all_splits)
    eval_end = min(end_date, latest_available_end)
    eval_start = eval_end - pd.DateOffset(months=6)
    selected = [
        split
        for split in all_splits
        if split.test_start >= eval_start and split.test_end <= eval_end
    ]

    if test_window == "1m" and step_window == "1m":
        expected = 6
        if len(selected) != expected:
            dates = [
                f"fold{split.fold}:{split.test_start.date()}->{split.test_end.date()}"
                for split in all_splits
            ]
            raise ValueError(
                "Last-6-month fold enforcement failed. "
                f"Expected {expected} monthly folds but found {len(selected)} within "
                f"{eval_start.date()}..{eval_end.date()}. "
                f"All available test windows: {dates}"
            )
    elif not selected:
        raise ValueError(
            f"No walk-forward folds found within last 6 months "
            f"({eval_start.date()}..{eval_end.date()})."
        )

    return eval_start, eval_end, selected


def run_candidate(candidate_feature_cols, selected_splits, train_iters, device):
    fold_metrics = []
    for split in selected_splits:
        train_df = set_time_index(split.train.copy(), split.train_start)
        test_df = set_time_index(split.test.copy(), split.train_start)
        train_x_df, test_x_df, _ = normalize_features(train_df, test_df, candidate_feature_cols)

        train_x = torch.tensor(train_x_df.values, dtype=torch.float32, device=device)
        train_y = torch.tensor(train_df["target"].values, dtype=torch.float32, device=device)
        test_x = torch.tensor(test_x_df.values, dtype=torch.float32, device=device)
        test_y = torch.tensor(test_df["target"].values, dtype=torch.float32, device=device)

        model, likelihood = train_gp(
            train_x,
            train_y,
            train_iters=train_iters,
            device=device,
        )
        metrics = evaluate(model, likelihood, test_x, test_y)
        fold_metrics.append(
            {
                "fold": split.fold,
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "mae_simple": metrics["mae_simple"],
                "directional": metrics["directional"],
                "coverage_95": metrics["coverage_95"],
                "avg_interval_width": metrics["avg_interval_width"],
            }
        )

    return summarize_fold_metrics(fold_metrics)


def metric_triplet(summary_metrics):
    return {
        "directional_mean": summary_metrics["directional_mean"],
        "mae_simple_mean": summary_metrics["mae_simple_mean"],
        "coverage_95_mean": summary_metrics["coverage_95_mean"],
    }


def deltas_from_baseline(candidate_metrics, baseline_metrics):
    directional_delta = candidate_metrics["directional_mean"] - baseline_metrics["directional_mean"]
    mae_simple_delta = candidate_metrics["mae_simple_mean"] - baseline_metrics["mae_simple_mean"]
    coverage_95_delta = candidate_metrics["coverage_95_mean"] - baseline_metrics["coverage_95_mean"]

    mae_simple_improvement = baseline_metrics["mae_simple_mean"] - candidate_metrics["mae_simple_mean"]
    directional_improvement = directional_delta
    coverage_target_improvement = abs(baseline_metrics["coverage_95_mean"] - 0.95) - abs(
        candidate_metrics["coverage_95_mean"] - 0.95
    )

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

    feature_delta_successes = sorted(
        successes,
        key=lambda x: (abs(x["deltas"]["mae_simple_delta"]), x["dropped_feature"]),
    )

    feature_deltas = [
        {
            "dropped_feature": item["dropped_feature"],
            "status": "ok",
            "deltas": item["deltas"],
            "improvements": item["improvements"],
        }
        for item in feature_delta_successes
    ]

    for item in sorted(failures, key=lambda x: x["dropped_feature"]):
        feature_deltas.append(
            {
                "dropped_feature": item["dropped_feature"],
                "status": "failed",
                "error": item["error"],
            }
        )

    return leaderboard, feature_deltas


def write_ablation_json(
    output_dir,
    ticker,
    eval_start,
    eval_end,
    args,
    baseline_feature_cols,
    baseline_metrics,
    selected_splits,
    leaderboard,
    feature_deltas,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ablation.json"

    payload = {
        "meta": {
            "ticker": ticker,
            "generated_at": datetime.now(UTC).isoformat(),
            "end_date": str(eval_end.date()),
            "eval_start_date": str(eval_start.date()),
            "eval_end_date": str(eval_end.date()),
            "train_window": args.train_window,
            "test_window": args.test_window,
            "step_window": args.step_window,
            "train_iters": args.train_iters,
            "include_time_index": args.include_time_index,
            "fold_count": len(selected_splits),
            "features_tested_count": len(baseline_feature_cols),
            "ranking_rule": (
                "mae_simple_improvement desc, directional_improvement desc, "
                "coverage_target_improvement desc"
            ),
            "bottom_sort_rule": "abs(mae_simple_delta) asc",
        },
        "baseline": {
            "feature_count": len(baseline_feature_cols),
            "feature_columns": baseline_feature_cols,
            "metrics": baseline_metrics,
        },
        "leaderboard": leaderboard,
        "feature_deltas": feature_deltas,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main():
    args = parse_args()
    if args.train_iters <= 0:
        raise ValueError("--train-iters must be positive.")

    tickers = prompt_tickers()
    if not tickers:
        print("No tickers provided. Exiting.")
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device()
    print(f"Using device: {device.type}")

    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()

    for ticker in tickers:
        try:
            dataset_start = compute_dataset_start(end_date, args.train_window)
            print(f"\nBuilding dataset for {ticker}...")

            data = build_dataset(ticker, dataset_start, end_date)
            dataset = data["dataset"]

            baseline_feature_cols = [col for col in dataset.columns if col != "target"]
            if not args.include_time_index:
                baseline_feature_cols = [col for col in baseline_feature_cols if col != "time_index"]
            if not baseline_feature_cols:
                raise ValueError("No features available for ablation.")

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
            if not all_splits:
                raise ValueError("No walk-forward splits produced.")

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

            baseline_summary = run_candidate(
                baseline_feature_cols,
                selected_splits,
                args.train_iters,
                device,
            )
            baseline_metrics = metric_triplet(baseline_summary)
            print(
                "Baseline metrics | "
                f"Dir mean: {baseline_metrics['directional_mean']:.2%} | "
                f"MAE(simple) mean: {baseline_metrics['mae_simple_mean']:.4%} | "
                f"Coverage95 mean: {baseline_metrics['coverage_95_mean']:.2%}"
            )

            results = []
            for idx, feature in enumerate(baseline_feature_cols, start=1):
                candidate_feature_cols = [col for col in baseline_feature_cols if col != feature]
                print(f"[{idx}/{len(baseline_feature_cols)}] Testing drop: {feature}")
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
                    candidate_summary = run_candidate(
                        candidate_feature_cols,
                        selected_splits,
                        args.train_iters,
                        device,
                    )
                    results.append(
                        {
                            "dropped_feature": feature,
                            "status": "ok",
                            "candidate_metrics": metric_triplet(candidate_summary),
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

            leaderboard, feature_deltas = build_ranked_outputs(results, baseline_metrics)
            ticker_output_dir = Path(args.output_dir) / ticker
            out_path = write_ablation_json(
                ticker_output_dir,
                ticker,
                eval_start,
                eval_end,
                args,
                baseline_feature_cols,
                baseline_metrics,
                selected_splits,
                leaderboard,
                feature_deltas,
            )

            print(f"\nAblation saved to: {out_path}")
            print(f"Leaderboard rows: {len(leaderboard)}")
            print(f"Feature delta rows: {len(feature_deltas)}")
        except Exception as exc:
            print(f"{ticker}: ablation failed - {exc}")


if __name__ == "__main__":
    main()
