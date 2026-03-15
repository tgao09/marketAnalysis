import argparse
import json
import math
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import parse_window
from gp_return.train import (
    DEFAULT_TRAIN_ITERS as GP_DEFAULT_TRAIN_ITERS,
    DEFAULT_TRAIN_WINDOW as GP_DEFAULT_TRAIN_WINDOW,
    FEATURE_LOOKBACK_MAX as GP_FEATURE_LOOKBACK_MAX,
    REGIME_SCORE_CLIP,
    REGIME_SCORE_WEIGHTS,
    REGIME_SCORE_WINDOW,
    TICKER_GOLD,
    TICKER_SPY,
    TICKER_VIX,
    WINDOW_RET,
    build_features as build_gp_features,
    build_target,
    extract_field,
    fetch_history_cached,
    resolve_device,
    resolve_sector_etf,
    train_gp,
)
from hmm_regime.train import (
    DEFAULT_TRAIN_WINDOW as HMM_DEFAULT_TRAIN_WINDOW,
    DEFAULT_N_INIT as HMM_DEFAULT_N_INIT,
    DEFAULT_N_ITER as HMM_DEFAULT_N_ITER,
    DEFAULT_MIN_TRAIN_ROWS as HMM_DEFAULT_MIN_TRAIN_ROWS,
    DEFAULT_RANDOM_STATE as HMM_DEFAULT_RANDOM_STATE,
    apply_scaler,
    build_market_dataset,
    build_state_output,
    compute_filtered_state_probs,
    compute_shift_probability,
    fit_hmm_bundle,
    select_training_features,
)


ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_VARIANT_REGULAR = "regular"
DEFAULT_META_TRAIN_WINDOW = "1y"
DEFAULT_TEST_YEARS = 1
MIN_META_TRAIN_ROWS = 60
DEFAULT_RIDGE_ALPHAS = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], dtype=float)
PREDICTIONS_FILENAME = "ensemble_return_predictions.csv"
TRADES_FILENAME = "ensemble_return_trades.csv"
SUMMARY_FILENAME = "ensemble_return_summary.json"
MODEL_STATE_FILENAME = "model_state.pkl"
CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HMM + GP residual ensemble return model.")
    parser.add_argument("--ticker", default=None, help="Ticker symbol.")
    parser.add_argument("--end", default=None, help="Training end date (YYYY-MM-DD).")
    parser.add_argument("--base-train-window", default=GP_DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--meta-train-window", default=DEFAULT_META_TRAIN_WINDOW)
    parser.add_argument("--output-dir", default=ARTIFACT_DIR_DEFAULT)
    return parser.parse_args()


def prompt_ticker() -> str | None:
    raw = input("Ticker to train: ").strip()
    if not raw:
        return None
    return raw.upper()


def resolve_artifact_variant() -> str:
    return ARTIFACT_VARIANT_REGULAR


def resolve_artifact_dir(output_dir: str | Path, ticker: str) -> Path:
    return Path(output_dir) / ticker.upper() / resolve_artifact_variant()


def compute_strategy_start(
    end_date: pd.Timestamp,
    base_train_window: str,
    meta_train_window: str,
    hmm_train_window: str = HMM_DEFAULT_TRAIN_WINDOW,
    test_years: int = DEFAULT_TEST_YEARS,
) -> pd.Timestamp:
    base_start = pd.Timestamp(end_date).normalize() - parse_window(base_train_window)
    hmm_start = pd.Timestamp(end_date).normalize() - parse_window(hmm_train_window)
    meta_offset = parse_window(meta_train_window)
    start = min(base_start, hmm_start) - meta_offset
    start = start - pd.DateOffset(years=test_years)
    buffer_days = max(GP_FEATURE_LOOKBACK_MAX, REGIME_SCORE_WINDOW, 252) + (2 * WINDOW_RET) + 10
    start = start - pd.DateOffset(days=buffer_days)
    return start


def build_strategy_dataset(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    history_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    if history_cache is None:
        history_cache = {}
    regime_config = {
        "enabled": False,
        "score_window": REGIME_SCORE_WINDOW,
        "score_clip": REGIME_SCORE_CLIP,
        "weights": REGIME_SCORE_WEIGHTS,
    }

    sector_etf, sector_name, sector_error = resolve_sector_etf(ticker)
    if sector_error:
        print(f"{ticker}: sector fallback to {sector_etf} ({sector_error})")

    stock_history = fetch_history_cached(ticker, start_date, end_date, history_cache)
    sector_history = fetch_history_cached(sector_etf, start_date, end_date, history_cache)
    gld_history = fetch_history_cached(TICKER_GOLD, start_date, end_date, history_cache)
    spy_history = fetch_history_cached(TICKER_SPY, start_date, end_date, history_cache)
    vix_history = fetch_history_cached(TICKER_VIX, start_date, end_date, history_cache)

    price_stock = extract_field(stock_history, "Close", ticker)
    volume_stock = extract_field(stock_history, "Volume", ticker)
    price_sector = extract_field(sector_history, "Close", sector_etf)
    price_gld = extract_field(gld_history, "Close", TICKER_GOLD)
    price_spy = extract_field(spy_history, "Close", TICKER_SPY)
    price_vix = extract_field(vix_history, "Close", TICKER_VIX)

    features = build_gp_features(
        price_stock,
        volume_stock,
        price_sector,
        price_gld,
        price_spy,
        price_vix,
        regime_config,
    )
    target = build_target(price_stock)

    dataset = features.join([target]).dropna()
    if dataset.index.has_duplicates:
        dataset = dataset.loc[~dataset.index.duplicated(keep="last")]
    close_stock = price_stock.reindex(dataset.index)

    return {
        "dataset": dataset,
        "close": close_stock,
        "sector_etf": sector_etf,
        "sector_name": sector_name,
    }


def latest_strategy_features(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    history_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    if history_cache is None:
        history_cache = {}
    regime_config = {
        "enabled": False,
        "score_window": REGIME_SCORE_WINDOW,
        "score_clip": REGIME_SCORE_CLIP,
        "weights": REGIME_SCORE_WEIGHTS,
    }

    sector_etf, sector_name, sector_error = resolve_sector_etf(ticker)
    if sector_error:
        print(f"{ticker}: sector fallback to {sector_etf} ({sector_error})")

    stock_history = fetch_history_cached(ticker, start_date, end_date, history_cache)
    sector_history = fetch_history_cached(sector_etf, start_date, end_date, history_cache)
    gld_history = fetch_history_cached(TICKER_GOLD, start_date, end_date, history_cache)
    spy_history = fetch_history_cached(TICKER_SPY, start_date, end_date, history_cache)
    vix_history = fetch_history_cached(TICKER_VIX, start_date, end_date, history_cache)

    price_stock = extract_field(stock_history, "Close", ticker)
    volume_stock = extract_field(stock_history, "Volume", ticker)
    price_sector = extract_field(sector_history, "Close", sector_etf)
    price_gld = extract_field(gld_history, "Close", TICKER_GOLD)
    price_spy = extract_field(spy_history, "Close", TICKER_SPY)
    price_vix = extract_field(vix_history, "Close", TICKER_VIX)

    features = build_gp_features(
        price_stock,
        volume_stock,
        price_sector,
        price_gld,
        price_spy,
        price_vix,
        regime_config,
    )
    if features.index.has_duplicates:
        features = features.loc[~features.index.duplicated(keep="last")]
    close_stock = price_stock.reindex(features.index)

    return {
        "features": features,
        "close": close_stock,
        "sector_etf": sector_etf,
        "sector_name": sector_name,
    }


def build_candidate_frame(
    dataset: pd.DataFrame,
    close_stock: pd.Series,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    index_series = pd.Series(dataset.index, index=dataset.index)
    exit_date = index_series.shift(-WINDOW_RET)
    exit_close = close_stock.shift(-WINDOW_RET)

    frame = pd.DataFrame(
        {
            "entry_close": close_stock,
            "exit_date": exit_date,
            "exit_close": exit_close,
            "actual_target": dataset["target"],
        }
    )
    if start_date is not None:
        frame = frame.loc[pd.Timestamp(start_date).normalize() :]
    if end_date is not None:
        frame = frame.loc[: pd.Timestamp(end_date).normalize()]
    frame = frame.dropna(subset=["entry_close", "exit_date", "exit_close", "actual_target"])
    return frame


def compute_gp_prediction_for_date(
    dataset: pd.DataFrame,
    test_date: pd.Timestamp,
    base_train_window: str,
    device: torch.device,
    train_iters: int = GP_DEFAULT_TRAIN_ITERS,
    all_feature_index: pd.DatetimeIndex | None = None,
    test_features: pd.Series | None = None,
) -> dict[str, float | int | str]:
    dataset_index = pd.DatetimeIndex(all_feature_index if all_feature_index is not None else dataset.index)
    test_date = pd.Timestamp(test_date).normalize()
    test_pos = dataset_index.searchsorted(test_date, side="left")
    if test_pos >= len(dataset_index) or dataset_index[test_pos] != test_date:
        raise ValueError(f"Test date {test_date.date()} not present in feature index.")
    train_end_pos = test_pos - WINDOW_RET - 1
    if train_end_pos < 0:
        raise ValueError("Not enough embargoed rows for GP training.")

    train_end = dataset_index[train_end_pos]
    train_start = test_date - parse_window(base_train_window) - pd.offsets.BDay(WINDOW_RET)
    train_df = dataset.loc[(dataset.index > train_start) & (dataset.index <= train_end)]
    if len(train_df) < 60:
        raise ValueError(f"Not enough GP train rows ({len(train_df)} < 60).")

    train_x_df = train_df.drop(columns=["target"])
    train_mean = train_x_df.mean()
    train_std = train_x_df.std().replace(0.0, 1.0)
    train_x_df = (train_x_df - train_mean) / train_std
    if test_features is None:
        test_features = dataset.loc[test_date].drop(labels=["target"])
    else:
        missing_feature_cols = [col for col in train_x_df.columns if col not in test_features.index]
        if missing_feature_cols:
            missing_list = ", ".join(missing_feature_cols)
            raise ValueError(f"Missing GP inference features for {test_date.date()}: {missing_list}")
        test_features = test_features.reindex(train_x_df.columns)

    test_x_series = ((test_features - train_mean) / train_std).replace([np.inf, -np.inf], np.nan)
    if test_x_series.isna().any():
        bad_cols = ", ".join(test_x_series.index[test_x_series.isna()])
        raise ValueError(f"Invalid GP inference features for {test_date.date()}: {bad_cols}")
    test_x = test_x_series.to_numpy(dtype=float, copy=False)

    train_x = torch.tensor(train_x_df.values, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_df["target"].values, dtype=torch.float32, device=device)
    model, likelihood = train_gp(train_x, train_y, train_iters=train_iters, device=device)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32, device=device).unsqueeze(0)
        preds = likelihood(model(test_x_tensor))
        mean_log = preds.mean.item()
        std_log = preds.variance.sqrt().item()

    return {
        "gp_pred_mean_log": mean_log,
        "gp_pred_std_log": std_log,
        "gp_train_rows": len(train_df),
        "gp_train_start": str(train_df.index.min().date()),
        "gp_train_end": str(train_df.index.max().date()),
    }


def compute_hmm_state_for_date(
    market_dataset: pd.DataFrame,
    test_date: pd.Timestamp,
    hmm_train_window: str = HMM_DEFAULT_TRAIN_WINDOW,
) -> dict[str, float | int | str]:
    state_features = select_training_features(
        dataset=market_dataset,
        asof_date=test_date,
        train_window=hmm_train_window,
        min_train_rows=HMM_DEFAULT_MIN_TRAIN_ROWS,
    )
    test_date = pd.Timestamp(test_date).normalize()
    test_pos = state_features.index.searchsorted(test_date, side="left")
    if test_pos >= len(state_features.index) or state_features.index[test_pos] != test_date:
        raise ValueError(f"Test date {test_date.date()} not present in HMM feature index.")
    train_end_pos = test_pos - WINDOW_RET - 1
    if train_end_pos < 0:
        raise ValueError("Not enough embargoed rows for HMM training.")
    train_features = state_features.iloc[: train_end_pos + 1].copy()
    if len(train_features) < HMM_DEFAULT_MIN_TRAIN_ROWS:
        raise ValueError(
            "Not enough causally labeled HMM train rows "
            f"({len(train_features)} < {HMM_DEFAULT_MIN_TRAIN_ROWS})."
        )
    bundle = fit_hmm_bundle(
        train_features=train_features,
        train_targets=market_dataset.loc[train_features.index],
        n_iter=HMM_DEFAULT_N_ITER,
        random_state=HMM_DEFAULT_RANDOM_STATE,
        n_init=HMM_DEFAULT_N_INIT,
    )
    scaled_state_features = apply_scaler(state_features, bundle["scaler"])
    state_probs = compute_filtered_state_probs(bundle["model"], scaled_state_features.values)
    shift_probability = compute_shift_probability(state_probs, bundle["transition_matrix"])
    states = build_state_output(
        index=state_features.index,
        state_probs=state_probs,
        shift_probability=shift_probability,
        asof_date=test_date,
        stress_state_id=bundle["stress_state_id"],
    )
    latest = states.iloc[-1]
    out: dict[str, float | int | str] = {
        "state_id": latest["state_id"],
        "state_label": latest["state_label"],
        "shift_prob": latest["shift_prob"],
        "hmm_train_rows": len(train_features),
        "hmm_train_start": str(train_features.index.min().date()),
        "hmm_train_end": str(train_features.index.max().date()),
    }
    for idx in range(4):
        out[f"p_state_{idx}"] = latest[f"p_state_{idx}"]
    return out


def build_base_prediction_rows(
    ticker: str,
    end_date: pd.Timestamp,
    base_train_window: str,
    meta_train_window: str,
    device: torch.device,
    test_years: int = DEFAULT_TEST_YEARS,
    gp_train_iters: int = GP_DEFAULT_TRAIN_ITERS,
) -> pd.DataFrame:
    end_date = pd.Timestamp(end_date).normalize()
    start_date = compute_strategy_start(
        end_date=end_date,
        base_train_window=base_train_window,
        meta_train_window=meta_train_window,
        test_years=test_years,
    )
    print(f"Building base datasets for {ticker} from {start_date.date()} to {end_date.date()}...")
    strategy_data = build_strategy_dataset(ticker, start_date, end_date)
    market_dataset = build_market_dataset(start_date, end_date)
    dataset = strategy_data["dataset"]
    close_stock = strategy_data["close"]
    candidates = build_candidate_frame(dataset, close_stock)
    earliest_base_row_date = end_date - pd.DateOffset(years=test_years) - parse_window(meta_train_window)
    candidates = candidates.loc[candidates.index >= earliest_base_row_date].copy()

    rows: list[dict[str, Any]] = []
    total = len(candidates)
    for idx, (test_date, candidate) in enumerate(candidates.iterrows(), start=1):
        try:
            gp_row = compute_gp_prediction_for_date(
                dataset=dataset,
                test_date=test_date,
                base_train_window=base_train_window,
                device=device,
                train_iters=gp_train_iters,
            )
            hmm_row = compute_hmm_state_for_date(
                market_dataset=market_dataset,
                test_date=test_date,
            )
        except Exception as exc:
            print(f"{test_date.date()} | skipped base row: {exc}")
            continue

        row = {
            "symbol": ticker.upper(),
            "date": test_date,
            "exit_date": candidate["exit_date"],
            "entry_close": candidate["entry_close"],
            "exit_close": candidate["exit_close"],
            "actual_target": candidate["actual_target"],
            "actual_simple_return": math.exp(candidate["actual_target"]) - 1.0,
        }
        row.update(gp_row)
        row.update(hmm_row)
        rows.append(row)

        if idx == 1 or idx % 25 == 0 or idx == total:
            print(
                f"{test_date.date()} | built {len(rows)}/{total} causal base rows "
                f"for {ticker}"
            )

    base_rows = pd.DataFrame(rows)
    if base_rows.empty:
        raise ValueError("No causal base prediction rows were generated.")
    base_rows = base_rows.sort_values("date").reset_index(drop=True)
    base_rows["meta_residual_target"] = base_rows["actual_target"] - base_rows["gp_pred_mean_log"]
    return base_rows


def add_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gp_pred_mean_log_x_shift_prob"] = out["gp_pred_mean_log"] * out["shift_prob"]
    for idx in range(4):
        out[f"gp_pred_mean_log_x_p_state_{idx}"] = out["gp_pred_mean_log"] * out[f"p_state_{idx}"]
    return out


def build_meta_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_meta_features(frame)
    return enriched[
        [
            "gp_pred_mean_log",
            "gp_pred_std_log",
            "p_state_0",
            "p_state_1",
            "p_state_2",
            "p_state_3",
            "shift_prob",
            "gp_pred_mean_log_x_shift_prob",
            "gp_pred_mean_log_x_p_state_0",
            "gp_pred_mean_log_x_p_state_1",
            "gp_pred_mean_log_x_p_state_2",
            "gp_pred_mean_log_x_p_state_3",
        ]
    ].copy()


def fit_meta_model(train_rows: pd.DataFrame) -> tuple[StandardScaler, RidgeCV]:
    x_train = build_meta_feature_frame(train_rows)
    y_train = train_rows["meta_residual_target"].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = RidgeCV(alphas=DEFAULT_RIDGE_ALPHAS, fit_intercept=True)
    model.fit(x_scaled, y_train)
    return scaler, model


def predict_meta_residual(
    scaler: StandardScaler,
    model: RidgeCV,
    rows: pd.DataFrame,
) -> np.ndarray:
    x = build_meta_feature_frame(rows)
    x_scaled = scaler.transform(x)
    preds = model.predict(x_scaled)
    return np.asarray(preds, dtype=float)


def summarize_forecast_metrics(
    frame: pd.DataFrame,
    pred_col: str,
    actual_col: str = "actual_target",
) -> dict[str, float | None]:
    if frame.empty:
        return {
            "count": 0,
            "mae": None,
            "mse": None,
            "mae_simple": None,
            "directional": None,
        }

    actual = frame[actual_col].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    actual_simple = np.exp(actual) - 1.0
    pred_simple = np.exp(pred) - 1.0
    return {
        "count": len(frame),
        "mae": float(np.mean(np.abs(pred - actual))),
        "mse": float(np.mean((pred - actual) ** 2)),
        "mae_simple": float(np.mean(np.abs(pred_simple - actual_simple))),
        "directional": float(np.mean(np.sign(pred) == np.sign(actual))),
    }


def build_uplift_metrics(
    ensemble_metrics: dict[str, float | None],
    baseline_metrics: dict[str, float | None],
) -> dict[str, float | None]:
    uplift: dict[str, float | None] = {}
    for key in ("mae", "mse", "mae_simple"):
        ensemble_value = ensemble_metrics.get(key)
        baseline_value = baseline_metrics.get(key)
        if ensemble_value is None or baseline_value is None:
            uplift[f"{key}_uplift_vs_gp"] = None
        else:
            uplift[f"{key}_uplift_vs_gp"] = float(baseline_value - ensemble_value)

    ensemble_dir = ensemble_metrics.get("directional")
    baseline_dir = baseline_metrics.get("directional")
    uplift["directional_uplift_vs_gp"] = (
        None
        if ensemble_dir is None or baseline_dir is None
        else float(ensemble_dir - baseline_dir)
    )
    return uplift


def summarize_trades(trades: pd.DataFrame) -> dict[str, float | int | None]:
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": None,
            "avg_pnl": None,
            "median_pnl": None,
            "std_pnl": None,
            "max_drawdown": None,
        }

    pnl = trades["pnl"]
    daily = trades.groupby("trade_date")["pnl"].sum().sort_index()
    equity = daily.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "total_trades": len(trades),
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "std_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else None,
    }


def build_trade_frame(predictions: pd.DataFrame, notional: float) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "exit_date",
                "direction",
                "entry_close",
                "exit_close",
                "notional",
                "shares",
                "pred_mean_log",
                "pred_mean_simple",
                "actual_simple_return",
                "pnl",
                "return_pct",
            ]
        )

    trades_df = predictions[
        [
            "symbol",
            "date",
            "exit_date",
            "entry_close",
            "exit_close",
            "ensemble_pred_mean_log",
            "actual_simple_return",
        ]
    ].rename(
        columns={
            "date": "trade_date",
            "ensemble_pred_mean_log": "pred_mean_log",
        }
    ).copy()

    long_mask = trades_df["pred_mean_log"] >= 0.0
    trades_df["direction"] = np.where(long_mask, "long", "short")
    trades_df["notional"] = notional
    trades_df["shares"] = notional / trades_df["entry_close"]
    long_pnl_per_share = trades_df["exit_close"] - trades_df["entry_close"]
    short_pnl_per_share = trades_df["entry_close"] - trades_df["exit_close"]
    trades_df["pnl"] = trades_df["shares"] * np.where(long_mask, long_pnl_per_share, short_pnl_per_share)
    trades_df["return_pct"] = trades_df["pnl"] / notional
    trades_df["pred_mean_simple"] = np.exp(trades_df["pred_mean_log"]) - 1.0

    return trades_df.sort_values("trade_date").reset_index(drop=True)


def run_meta_backtest(
    base_rows: pd.DataFrame,
    meta_train_window: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    min_meta_train_rows: int = MIN_META_TRAIN_ROWS,
    notional: float = 10000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_rows = add_meta_features(base_rows.sort_values("date").reset_index(drop=True))
    exit_date_norm = pd.to_datetime(base_rows["exit_date"]).dt.normalize()
    eval_start = pd.Timestamp(eval_start).normalize()
    eval_end = pd.Timestamp(eval_end).normalize()
    meta_offset = parse_window(meta_train_window)

    prediction_rows: list[dict[str, Any]] = []
    skipped = 0
    eval_candidates = base_rows.loc[
        (base_rows["date"] >= eval_start) & (base_rows["date"] <= eval_end)
    ].copy()

    for row_idx, row in eval_candidates.iterrows():
        row_date = pd.Timestamp(row["date"]).normalize()
        train_start = row_date - meta_offset
        train_rows = base_rows.loc[
            (base_rows["date"] > train_start)
            & (base_rows["date"] < row_date)
            & (exit_date_norm < row_date)
        ].copy()
        if len(train_rows) < min_meta_train_rows:
            skipped += 1
            continue

        scaler, model = fit_meta_model(train_rows)
        row_features = build_meta_feature_frame(eval_candidates.loc[[row_idx]])
        residual_pred = model.predict(scaler.transform(row_features))[0]
        ensemble_pred = row["gp_pred_mean_log"] + residual_pred

        row_dict = row.to_dict()
        row_dict["meta_train_rows"] = len(train_rows)
        row_dict["meta_train_start"] = str(train_rows["date"].min().date())
        row_dict["meta_train_end"] = str(train_rows["date"].max().date())
        row_dict["ridge_alpha"] = float(model.alpha_)
        row_dict["meta_residual_pred"] = residual_pred
        row_dict["ensemble_pred_mean_log"] = ensemble_pred
        row_dict["ensemble_pred_mean_simple"] = math.exp(ensemble_pred) - 1.0
        prediction_rows.append(row_dict)

    predictions = pd.DataFrame(prediction_rows)
    if not predictions.empty:
        predictions = predictions.sort_values("date").reset_index(drop=True)
    trades = build_trade_frame(predictions, notional=notional)

    ensemble_metrics = summarize_forecast_metrics(predictions, "ensemble_pred_mean_log")
    gp_metrics = summarize_forecast_metrics(predictions, "gp_pred_mean_log")
    uplift_metrics = build_uplift_metrics(ensemble_metrics, gp_metrics)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "eval_start": str(eval_start.date()),
        "eval_end": str(eval_end.date()),
        "meta_train_window": meta_train_window,
        "min_meta_train_rows": min_meta_train_rows,
        "candidate_rows": len(eval_candidates),
        "predicted_rows": len(predictions),
        "skipped_rows": skipped,
        "ensemble_metrics": ensemble_metrics,
        "gp_baseline_metrics": gp_metrics,
        "uplift_vs_gp": uplift_metrics,
        "trade_metrics": summarize_trades(trades),
    }
    return predictions, trades, summary


def fit_final_meta_model(
    base_rows: pd.DataFrame,
    meta_train_window: str,
    asof_date: pd.Timestamp,
    min_meta_train_rows: int = MIN_META_TRAIN_ROWS,
) -> tuple[StandardScaler, RidgeCV, dict[str, Any]]:
    if base_rows.empty:
        raise ValueError("Cannot fit final meta-model on empty base rows.")

    base_rows = base_rows.sort_values("date").reset_index(drop=True)
    asof_date = pd.Timestamp(asof_date).normalize()
    train_start = asof_date - parse_window(meta_train_window)
    exit_date_norm = pd.to_datetime(base_rows["exit_date"]).dt.normalize()
    train_rows = base_rows.loc[
        (base_rows["date"] > train_start) & (exit_date_norm < asof_date)
    ].copy()
    if len(train_rows) < min_meta_train_rows:
        raise ValueError(
            f"Not enough meta rows for final model ({len(train_rows)} < {min_meta_train_rows})."
        )

    scaler, model = fit_meta_model(train_rows)
    diagnostics = {
        "asof_date": str(asof_date.date()),
        "train_start": str(train_rows["date"].min().date()),
        "train_end": str(train_rows["date"].max().date()),
        "train_rows": len(train_rows),
        "ridge_alpha": float(model.alpha_),
    }
    return scaler, model, diagnostics


def save_artifacts(
    artifact_dir: Path,
    model: RidgeCV,
    scaler: StandardScaler,
    config: dict[str, Any],
    metrics: dict[str, Any],
    meta_feature_columns: list[str],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_blob = {
        "model": model,
        "scaler": scaler,
        "meta_feature_columns": meta_feature_columns,
    }
    with (artifact_dir / MODEL_STATE_FILENAME).open("wb") as fh:
        pickle.dump(model_blob, fh)
    (artifact_dir / CONFIG_FILENAME).write_text(json.dumps(config, indent=2))
    (artifact_dir / METRICS_FILENAME).write_text(json.dumps(metrics, indent=2))


def load_model_blob(artifact_dir: Path) -> dict[str, Any]:
    with (artifact_dir / MODEL_STATE_FILENAME).open("rb") as fh:
        return pickle.load(fh)


def save_backtest_outputs(
    artifact_dir: Path,
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(artifact_dir / PREDICTIONS_FILENAME, index=False)
    trades.to_csv(artifact_dir / TRADES_FILENAME, index=False)
    (artifact_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    ticker = args.ticker or prompt_ticker()
    if not ticker:
        print("No ticker provided. Exiting.")
        return

    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    device = resolve_device()
    print(f"Using device: {device.type}")

    base_rows = build_base_prediction_rows(
        ticker=ticker,
        end_date=end_date,
        base_train_window=args.base_train_window,
        meta_train_window=args.meta_train_window,
        device=device,
    )

    eval_start = end_date - pd.DateOffset(years=DEFAULT_TEST_YEARS)
    predictions, trades, backtest_summary = run_meta_backtest(
        base_rows=base_rows,
        meta_train_window=args.meta_train_window,
        eval_start=eval_start,
        eval_end=end_date,
    )
    scaler, model, final_model_meta = fit_final_meta_model(
        base_rows=base_rows,
        meta_train_window=args.meta_train_window,
        asof_date=end_date,
    )
    meta_feature_columns = list(build_meta_feature_frame(base_rows).columns)

    config = {
        "ticker": ticker.upper(),
        "end_date": str(end_date.date()),
        "artifact_variant": resolve_artifact_variant(),
        "base_train_window": args.base_train_window,
        "meta_train_window": args.meta_train_window,
        "min_meta_train_rows": MIN_META_TRAIN_ROWS,
        "meta_feature_columns": meta_feature_columns,
    }
    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "backtest_summary": backtest_summary,
        "final_model": final_model_meta,
    }

    artifact_dir = resolve_artifact_dir(args.output_dir, ticker)
    save_artifacts(
        artifact_dir=artifact_dir,
        model=model,
        scaler=scaler,
        config=config,
        metrics=metrics,
        meta_feature_columns=meta_feature_columns,
    )

    print(f"Artifacts saved to: {artifact_dir}")
    print(f"Backtest predictions generated: {len(predictions)}")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
