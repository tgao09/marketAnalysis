import argparse
import os
import sys
from datetime import datetime
from typing import List

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Ensure we can import project modules
sys.path.append(os.path.join(REPO_ROOT, "dataset"))
sys.path.append(os.path.join(REPO_ROOT, "xgboost"))

from construct_dataset import compile_stock_dataset  # type: ignore
from lag_features import create_lag_features  # type: ignore
from feature_engineering import load_and_engineer_features  # type: ignore
from train_xgboost import XGBoostTrainingPipeline  # type: ignore
from hyperparameter_tuning import HyperparameterTuner  # type: ignore
from forecast_xgboost import XGBoostForecaster  # type: ignore
from stock_screener import XGBoostScreener  # type: ignore


def run_pipeline(
    horizons: List[int],
    years: int,
    n_lags: int,
    run_train_tune: bool,
    skip_fetch: bool,
    min_sharpe: float,
    num_screen: int,
    recompute_technical: bool,
) -> None:
    """Run end-to-end XGBoost workflow."""
    base_path = os.path.join(REPO_ROOT, "dataset", "stock_dataset.csv")
    lagged_path = os.path.join(REPO_ROOT, "dataset", "stock_dataset_with_lags.csv")

    if skip_fetch:
        print("\n=== Skipping dataset fetch/lag (using existing files) ===")
        if not os.path.exists(base_path) or not os.path.exists(lagged_path):
            raise FileNotFoundError(
                "Cannot skip fetch/lag: required files missing "
                f"({base_path if not os.path.exists(base_path) else ''} "
                f"{lagged_path if not os.path.exists(lagged_path) else ''})"
            )
    else:
        print("\n=== Step 1: Construct base dataset ===")
        compile_stock_dataset(
            tickers_file=os.path.join(REPO_ROOT, "dataset", "training_stocks.txt"),
            years=years,
            output_file=base_path,
            normalize=False,
        )

        print("\n=== Step 2: Create lag features ===")
        create_lag_features(
            input_file=base_path,
            n_lags=n_lags,
            features_to_lag=None,
            output_file=lagged_path,
        )

    print("\n=== Step 3: Feature engineering ===")
    features_path = os.path.join(REPO_ROOT, "xgboost", "features_engineered.csv")
    load_and_engineer_features(
        input_file=lagged_path,
        output_file=features_path,
        forward_horizons=horizons,
    )

    models_dir = os.path.join(REPO_ROOT, "xgboost", "models")
    tuned_models_dir = os.path.join(models_dir, "tuned")

    if run_train_tune:
        print("\n=== Step 4: Train baseline models ===")
        trainer = XGBoostTrainingPipeline(
            data_file=features_path,
            models_dir=models_dir,
            results_dir=os.path.join(REPO_ROOT, "xgboost", "results", "training"),
            horizons=horizons,
            cross_sectional=True,
        )
        trainer.train_all_horizons(use_cv=True, use_sample_weights=True)

        print("\n=== Step 5: Hyperparameter tuning ===")
        tuner = HyperparameterTuner(
            data_file=features_path,
            models_dir=tuned_models_dir,
            results_dir=os.path.join(REPO_ROOT, "xgboost", "results", "tuning"),
        )
        tuning_results = {}
        for horizon in horizons:
            tuning_results[horizon] = tuner.tune_single_horizon(
                horizon=horizon,
                preset_names=None,
                test_weeks=52,
                use_cv=True,
            )
        tuner.save_best_hyperparameters(tuning_results)

    # prefer tuned models if tuning ran; otherwise use baseline
    models_to_use = tuned_models_dir if run_train_tune else models_dir

    print("\n=== Step 6: Forecasting ===")
    forecaster = XGBoostForecaster(
        models_dir=models_to_use,
        results_dir=os.path.join(REPO_ROOT, "xgboost", "results", "forecasts"),
        recompute_technical=recompute_technical,
    )
    forecasts_df = forecaster.forecast_multi_horizon(features_path, horizons=horizons)

    forecast_dir = os.path.join(REPO_ROOT, "xgboost", "results", "forecasts")
    latest_forecast_file = os.path.join(
        forecast_dir,
        sorted(
            f
            for f in os.listdir(forecast_dir)
            if f.startswith("forecasts_") and f.endswith(".csv")
        )[-1],
    )
    print(f"Forecast file: {latest_forecast_file}")

    print("\n=== Step 7: Stock screening ===")
    screener = XGBoostScreener(min_return=0.003, min_sharpe=min_sharpe)
    top = screener.screen(latest_forecast_file, n=num_screen)

    print("\nTop opportunities:")
    for _, row in top.iterrows():
        direction = "LONG" if row["ensemble_return"] > 0 else "SHORT"
        print(
            f"{direction:5} {row['ticker']:6} | 1W: {row['weekly_return']*100:6.2f}% | "
            f"Ensemble: {row['ensemble_return']*100:6.2f}% | Strength: {row['signal_strength']:.4f} | "
            f"Sharpe: {row.get('sharpe_ratio_3m', float('nan')):.2f}"
        )

    print("\nPipeline complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end XGBoost pipeline (dataset -> lags -> features -> optional train/tune -> forecast -> screen)."
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8], help="Forecast horizons in weeks.")
    parser.add_argument("--years", type=int, default=3, help="Years of history to fetch.")
    parser.add_argument("--n-lags", type=int, default=3, help="Number of lag steps to create.")
    parser.add_argument(
        "--train-and-tune",
        action="store_true",
        help="Train baseline models and run hyperparameter tuning (uses tuned models for forecasting).",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip dataset construction and lagging (use existing CSVs in dataset/).",
    )
    parser.add_argument("--min-sharpe", type=float, default=0.3, help="Minimum Sharpe ratio filter for screening.")
    parser.add_argument("--num-screen", type=int, default=20, help="Number of opportunities to output from screener.")
    parser.add_argument(
        "--no-recompute-technical",
        action="store_true",
        help="Skip recomputing technical indicators during forecasting (faster but less accurate).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        horizons=args.horizons,
        years=args.years,
        n_lags=args.n_lags,
        run_train_tune=args.train_and_tune,
        skip_fetch=args.no_fetch,
        min_sharpe=args.min_sharpe,
        num_screen=args.num_screen,
        recompute_technical=not args.no_recompute_technical,
    )
