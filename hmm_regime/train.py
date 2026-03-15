import argparse
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import get_history, parse_window


ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "artifacts" / "market"
TICKER_SPY = "SPY"
TICKER_VIX = "^VIX"
WINDOW_RET = 5
WINDOW_RET_MEDIUM = 20
WINDOW_RET_LONG = 63
ZSCORE_WINDOW = 252
FEATURE_LOOKBACK_MAX = 252
DEFAULT_TRAIN_WINDOW = "3y"
DEFAULT_RETRAIN_CADENCE = "weekly"
N_STATES = 4
DEFAULT_N_ITER = 500
DEFAULT_N_INIT = 8
DEFAULT_RANDOM_STATE = 42
DEFAULT_MIN_TRAIN_ROWS = 252
MIN_COLLAPSE_OCCUPANCY = 0.01
WINSOR_QUANTILE = 0.01
MARKET_TARGET_COLUMNS = (
    "forward_ret_5d",
    "sign_flip_5d",
    "forward_vol_5d",
    "vol_jump_5d",
    "drawdown",
)
MARKET_NON_FEATURE_COLUMNS = (*MARKET_TARGET_COLUMNS, "spy_close")

def state_label(state_id: int) -> str:
    return f"state_{int(state_id)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone 4-state market HMM regime model.")
    parser.add_argument("--train-window", default=DEFAULT_TRAIN_WINDOW, help="Rolling train window like '3y'.")
    parser.add_argument("--end", default=None, help="Training end date YYYY-MM-DD (default: today).")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR_DEFAULT))
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--n-init", type=int, default=DEFAULT_N_INIT)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    parser.add_argument("--retrain-cadence", default=DEFAULT_RETRAIN_CADENCE)
    return parser.parse_args()


def fetch_history_cached(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache: Dict[Tuple[str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    start_norm = pd.Timestamp(start_date).normalize()
    end_norm = pd.Timestamp(end_date).normalize()
    key = (symbol.upper(), str(start_norm.date()), str(end_norm.date()))
    if key not in cache:
        end_exclusive = end_norm + pd.Timedelta(days=1)
        history = get_history(
            symbol,
            period=None,
            start=str(start_norm.date()),
            end=str(end_exclusive.date()),
            interval="1d",
            auto_adjust=True,
        )
        idx = history.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        history = history.copy()
        history.index = idx.normalize()
        history = history.sort_index()
        cache[key] = history
    return cache[key]


def extract_field(history: pd.DataFrame, field: str, symbol: str) -> pd.Series:
    if field not in history.columns:
        raise KeyError(f"Missing field {field} for {symbol}.")
    series = history[field].copy()
    if series.empty:
        raise ValueError(f"No {field} data for {symbol}.")
    return series


def compute_log_return(series: pd.Series, window: int = 1) -> pd.Series:
    ratio = series / series.shift(window)
    ratio = ratio.replace(0.0, np.nan)
    out = np.log(ratio)
    return out.replace([np.inf, -np.inf], np.nan)


def zscore_trailing(series: pd.Series, window: int, min_periods: int = 40) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def compute_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0.0, np.nan)
    ratio = numerator / denom
    ratio = ratio.replace(0.0, np.nan)
    out = np.log(ratio)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_drawdown(series: pd.Series) -> pd.Series:
    return (series / series.cummax()) - 1.0


def compute_dataset_start(end_date: pd.Timestamp, train_window: str, test_years: int = 0) -> pd.Timestamp:
    train_offset = parse_window(train_window)
    buffer_days = max(FEATURE_LOOKBACK_MAX, ZSCORE_WINDOW) + (2 * WINDOW_RET) + 10
    start = pd.Timestamp(end_date).normalize() - train_offset - pd.DateOffset(days=buffer_days)
    if test_years > 0:
        start = start - pd.DateOffset(years=int(test_years))
    return start


def build_features(price_spy: pd.Series, price_vix: pd.Series) -> pd.DataFrame:
    index = price_spy.index
    price_vix = price_vix.reindex(index).ffill()

    spy_ret_1d = compute_log_return(price_spy, 1)
    vix_chg_1d = compute_log_return(price_vix, 1)
    spy_vol_5d = spy_ret_1d.rolling(WINDOW_RET).std()
    spy_vol_20d = spy_ret_1d.rolling(WINDOW_RET_MEDIUM).std()
    spy_vol_63d = spy_ret_1d.rolling(WINDOW_RET_LONG).std()
    vix_vol_5d = vix_chg_1d.rolling(WINDOW_RET).std()
    vix_vol_20d = vix_chg_1d.rolling(WINDOW_RET_MEDIUM).std()

    features = pd.DataFrame(index=index)
    features["spy_ret_1d"] = spy_ret_1d
    features["spy_ret_5d"] = compute_log_return(price_spy, WINDOW_RET)
    features["spy_ret_20d"] = compute_log_return(price_spy, WINDOW_RET_MEDIUM)
    features["spy_ret_63d"] = compute_log_return(price_spy, WINDOW_RET_LONG)
    features["spy_vol_5d"] = spy_vol_5d
    features["spy_vol_20d"] = spy_vol_20d
    features["spy_vol_63d"] = spy_vol_63d
    features["spy_vol_ratio_5_20"] = compute_log_ratio(spy_vol_5d, spy_vol_20d)
    features["spy_vol_ratio_20_63"] = compute_log_ratio(spy_vol_20d, spy_vol_63d)
    features["vix_level_z"] = zscore_trailing(price_vix, ZSCORE_WINDOW)
    features["vix_chg_1d"] = vix_chg_1d
    features["vix_chg_5d"] = compute_log_return(price_vix, 5)
    features["vix_chg_20d"] = compute_log_return(price_vix, WINDOW_RET_MEDIUM)
    features["vix_vol_5d"] = vix_vol_5d
    features["vix_vol_20d"] = vix_vol_20d
    features["vix_trend_gap_20d"] = (price_vix / price_vix.rolling(WINDOW_RET_MEDIUM).mean()) - 1.0
    features["trend_gap_20d"] = (price_spy / price_spy.rolling(20).mean()) - 1.0
    features["trend_gap_63d"] = (price_spy / price_spy.rolling(WINDOW_RET_LONG).mean()) - 1.0
    features["spy_drawdown"] = compute_drawdown(price_spy)
    return features


def build_market_targets(price_spy: pd.Series) -> pd.DataFrame:
    spy_ret_1d = compute_log_return(price_spy, 1)
    prev_ret_5d = compute_log_return(price_spy, WINDOW_RET)
    forward_ret_5d = np.log(price_spy.shift(-WINDOW_RET) / price_spy)
    forward_vol_5d = spy_ret_1d.rolling(WINDOW_RET).std().shift(-WINDOW_RET)
    drawdown = compute_drawdown(price_spy)

    prev_sign = np.sign(prev_ret_5d)
    next_sign = np.sign(forward_ret_5d)
    sign_flip_5d = ((prev_sign * next_sign) < 0).astype(float)
    ambiguous = (prev_sign == 0) | (next_sign == 0) | prev_sign.isna() | next_sign.isna()
    sign_flip_5d.loc[ambiguous] = np.nan

    realized_vol_5d = spy_ret_1d.rolling(WINDOW_RET).std()
    vol_jump_baseline = realized_vol_5d.rolling(63, min_periods=20).median().shift(1)
    vol_jump_5d = (forward_vol_5d > (1.25 * vol_jump_baseline)).astype(float)
    vol_jump_5d.loc[vol_jump_baseline.isna()] = np.nan

    out = pd.DataFrame(index=price_spy.index)
    out["forward_ret_5d"] = forward_ret_5d
    out["sign_flip_5d"] = sign_flip_5d
    out["forward_vol_5d"] = forward_vol_5d
    out["vol_jump_5d"] = vol_jump_5d
    out["drawdown"] = drawdown
    return out


def build_market_dataset(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    history_cache: Dict[Tuple[str, str, str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    cache = history_cache or {}
    spy_history = fetch_history_cached(TICKER_SPY, start_date, end_date, cache)
    vix_history = fetch_history_cached(TICKER_VIX, start_date, end_date, cache)

    price_spy = extract_field(spy_history, "Close", TICKER_SPY)
    price_vix = extract_field(vix_history, "Close", TICKER_VIX)

    features = build_features(price_spy, price_vix)
    targets = build_market_targets(price_spy)

    out = features.join(targets, how="left")
    out["spy_close"] = price_spy.reindex(out.index).ffill()
    out = out.sort_index()
    return out


def select_training_features(
    dataset: pd.DataFrame,
    asof_date: pd.Timestamp,
    train_window: str,
    min_train_rows: int,
) -> pd.DataFrame:
    asof_date = pd.Timestamp(asof_date).normalize()
    train_start = asof_date - parse_window(train_window)
    train_df = dataset.drop(columns=MARKET_NON_FEATURE_COLUMNS, errors="ignore")
    train_df = train_df.loc[(train_df.index > train_start) & (train_df.index <= asof_date)]
    train_df = train_df.dropna()
    if train_df.empty or len(train_df) < int(min_train_rows):
        raise ValueError(
            f"Insufficient training rows for asof={asof_date.date()}. "
            f"rows={len(train_df)} min_required={min_train_rows}"
        )
    return train_df


def fit_scaler(train_features: pd.DataFrame) -> Dict[str, pd.Series]:
    lower = train_features.quantile(WINSOR_QUANTILE)
    upper = train_features.quantile(1.0 - WINSOR_QUANTILE)
    clipped = train_features.clip(lower=lower, upper=upper, axis=1)
    mean = clipped.mean()
    std = clipped.std(ddof=0).replace(0.0, 1.0)
    return {"mean": mean, "std": std, "lower": lower, "upper": upper}


def apply_scaler(
    frame: pd.DataFrame,
    scaler: Dict[str, Any],
) -> pd.DataFrame:
    scaler_columns = list(pd.Series(scaler["mean"]).index) if "mean" in scaler else []
    active_feature_columns = scaler_columns or list(frame.columns)
    mean = pd.Series(scaler["mean"], dtype=float).reindex(active_feature_columns)
    std = pd.Series(scaler["std"], dtype=float).replace(0.0, 1.0).reindex(active_feature_columns)
    clipped = frame[active_feature_columns].copy()
    lower_raw = scaler.get("lower")
    upper_raw = scaler.get("upper")
    if lower_raw is not None and upper_raw is not None:
        lower = pd.Series(lower_raw, dtype=float).reindex(active_feature_columns)
        upper = pd.Series(upper_raw, dtype=float).reindex(active_feature_columns)
        clipped = clipped.clip(lower=lower, upper=upper, axis=1)
    scaled = (clipped - mean) / std
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    return scaled


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights)
    if not np.any(mask):
        return float("nan")
    masked_weights = weights[mask]
    denom = float(masked_weights.sum())
    if denom <= 0:
        return float("nan")
    return float(np.dot(masked_weights, values[mask]) / denom)


def compute_state_statistics(raw_probs: np.ndarray, train_frame: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    metric_arrays = {
        "spy_ret_5d_mean": train_frame["spy_ret_5d"].to_numpy(dtype=float),
        "spy_ret_20d_mean": train_frame["spy_ret_20d"].to_numpy(dtype=float),
        "spy_vol_5d_mean": train_frame["spy_vol_5d"].to_numpy(dtype=float),
        "spy_vol_20d_mean": train_frame["spy_vol_20d"].to_numpy(dtype=float),
        "forward_ret_5d_mean": train_frame["forward_ret_5d"].to_numpy(dtype=float),
        "forward_vol_5d_mean": train_frame["forward_vol_5d"].to_numpy(dtype=float),
        "vol_jump_5d_rate": train_frame["vol_jump_5d"].to_numpy(dtype=float),
        "drawdown_mean": train_frame["drawdown"].to_numpy(dtype=float),
    }
    stats: Dict[int, Dict[str, float]] = {}
    n_rows = float(len(train_frame))

    for state in range(raw_probs.shape[1]):
        weights = raw_probs[:, state]
        denom = float(weights.sum())
        if denom <= 0:
            stats[state] = {
                "occupancy": 0.0,
                **{metric: float("nan") for metric in metric_arrays},
            }
            continue
        state_metrics = {"occupancy": denom / n_rows}
        for metric, values in metric_arrays.items():
            state_metrics[metric] = _weighted_nanmean(values, weights)
        stats[state] = state_metrics
    return stats


def identify_stress_state(state_stats: Dict[int, Dict[str, float]]) -> int:
    def score(state_id: int) -> Tuple[float, float, float]:
        metrics = state_stats[state_id]
        forward_vol = float(metrics.get("forward_vol_5d_mean", float("nan")))
        vol_jump = float(metrics.get("vol_jump_5d_rate", float("nan")))
        drawdown = float(metrics.get("drawdown_mean", float("nan")))
        return (
            forward_vol if np.isfinite(forward_vol) else float("-inf"),
            vol_jump if np.isfinite(vol_jump) else float("-inf"),
            -(drawdown if np.isfinite(drawdown) else 0.0),
        )

    return max(sorted(state_stats.keys()), key=score)


def _logsumexp(vec: np.ndarray) -> float:
    vmax = float(np.max(vec))
    if not np.isfinite(vmax):
        return vmax
    return vmax + float(np.log(np.sum(np.exp(vec - vmax))))


def compute_filtered_state_probs(model: GaussianHMM, scaled_values: np.ndarray) -> np.ndarray:
    """Forward-only state posteriors p(s_t | x_1..x_t) in raw model state order."""
    values = np.asarray(scaled_values, dtype=float)
    log_emlik = model._compute_log_likelihood(values)

    start = np.clip(np.asarray(model.startprob_, dtype=float), 1e-300, 1.0)
    trans = np.clip(np.asarray(model.transmat_, dtype=float), 1e-300, 1.0)

    log_start = np.log(start)
    log_trans = np.log(trans)
    n_obs, n_states = log_emlik.shape
    log_alpha = np.empty((n_obs, n_states), dtype=float)

    log_alpha[0] = log_start + log_emlik[0]
    log_alpha[0] = log_alpha[0] - _logsumexp(log_alpha[0])

    for row in range(1, n_obs):
        for col in range(n_states):
            log_alpha[row, col] = log_emlik[row, col] + _logsumexp(log_alpha[row - 1] + log_trans[:, col])
        log_alpha[row] = log_alpha[row] - _logsumexp(log_alpha[row])

    probs = np.exp(log_alpha)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, np.inf)
    return probs / row_sums


def compute_shift_probability(state_probs: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    """Predictive shift score aligned to current MAP state and next-state uncertainty."""
    transition = np.asarray(transition_matrix, dtype=float)
    next_probs = np.asarray(state_probs, dtype=float) @ transition
    row_sums = np.clip(next_probs.sum(axis=1, keepdims=True), 1e-12, np.inf)
    next_probs = next_probs / row_sums

    current_state = np.argmax(state_probs, axis=1).astype(int)
    stay_prob = next_probs[np.arange(len(next_probs)), current_state]
    stay_prob = np.clip(stay_prob, 0.0, 1.0)

    entropy = -np.sum(next_probs * np.log(np.clip(next_probs, 1e-12, 1.0)), axis=1)
    entropy = entropy / np.log(next_probs.shape[1])
    shift_score = (0.7 * (1.0 - stay_prob)) + (0.3 * entropy)
    return np.clip(shift_score, 0.0, 1.0)


def build_state_output(
    index: pd.DatetimeIndex,
    state_probs: np.ndarray,
    shift_probability: np.ndarray,
    asof_date: pd.Timestamp,
    stress_state_id: int,
) -> pd.DataFrame:
    state_id = np.argmax(state_probs, axis=1).astype(int)
    out = pd.DataFrame(index=index)
    out["date"] = pd.to_datetime(index).normalize()
    out["state_id"] = state_id
    out["state_label"] = [state_label(item) for item in state_id]
    for item in range(state_probs.shape[1]):
        out[f"p_state_{item}"] = state_probs[:, item]
    out["shift_prob"] = shift_probability
    out["stress_state_id"] = int(stress_state_id)
    out["stress_state_label"] = state_label(stress_state_id)
    out["asof"] = pd.Timestamp(asof_date).normalize()
    return out.reset_index(drop=True)


def _safe_auc(scores: pd.Series, target: pd.Series) -> float | None:
    frame = pd.DataFrame({"score": scores, "target": target}).dropna()
    if frame.empty:
        return None
    y = frame["target"].astype(int)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, frame["score"]))


def compute_candidate_selection_metrics(
    train_frame: pd.DataFrame,
    state_probs: np.ndarray,
    transition_matrix: np.ndarray,
    shift_probability: np.ndarray,
    log_likelihood_per_row: float,
    stress_state_id: int,
) -> Dict[str, float | None]:
    state_id = np.argmax(state_probs, axis=1).astype(int)

    eval_frame = pd.DataFrame(index=train_frame.index)
    eval_frame["state_id"] = state_id
    eval_frame["shift_prob"] = shift_probability
    eval_frame["forward_vol_5d"] = train_frame["forward_vol_5d"]
    eval_frame["vol_jump_5d"] = train_frame["vol_jump_5d"]
    eval_frame["drawdown"] = train_frame["drawdown"]
    eval_frame["regime_change_next_bday"] = (eval_frame["state_id"] != eval_frame["state_id"].shift(-1)).astype(float)
    eval_frame.loc[eval_frame["state_id"].shift(-1).isna(), "regime_change_next_bday"] = np.nan

    vol_df = eval_frame.dropna(subset=["forward_vol_5d"])
    unconditional_vol = float(vol_df["forward_vol_5d"].mean()) if not vol_df.empty else None
    stress_vol_ratio = None
    if unconditional_vol and unconditional_vol > 0:
        stress_rows = vol_df[vol_df["state_id"] == int(stress_state_id)]
        if not stress_rows.empty:
            stress_vol_ratio = float(stress_rows["forward_vol_5d"].mean() / unconditional_vol)

    stress_drawdown_share = None
    dd_df = eval_frame.dropna(subset=["drawdown"])
    if not dd_df.empty:
        deep_cut = float(dd_df["drawdown"].quantile(0.10))
        deep_rows = dd_df[dd_df["drawdown"] <= deep_cut]
        if not deep_rows.empty:
            stress_drawdown_share = float((deep_rows["state_id"] == int(stress_state_id)).mean())

    min_occupancy = float(state_probs.mean(axis=0).min())
    transition_persistence = float(np.mean(np.diag(np.asarray(transition_matrix, dtype=float))))

    return {
        "log_likelihood_per_row": log_likelihood_per_row,
        "min_occupancy": min_occupancy,
        "stress_vol_ratio": stress_vol_ratio,
        "stress_drawdown_share": stress_drawdown_share,
        "regime_change_auc": _safe_auc(eval_frame["shift_prob"], eval_frame["regime_change_next_bday"]),
        "vol_jump_auc": _safe_auc(eval_frame["shift_prob"], eval_frame["vol_jump_5d"]),
        "transition_persistence": transition_persistence,
    }


def select_best_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_weights = [
        ("min_occupancy", 2.0),
        ("stress_vol_ratio", 2.0),
        ("stress_drawdown_share", 1.5),
        ("regime_change_auc", 1.25),
        ("vol_jump_auc", 1.0),
        ("transition_persistence", 0.5),
        ("log_likelihood_per_row", 0.5),
    ]

    for candidate in candidates:
        candidate["selection_score"] = 0.0

    for metric, weight in metric_weights:
        valid = [
            candidate
            for candidate in candidates
            if np.isfinite(candidate["selection_metrics"].get(metric, float("nan")))
        ]
        if not valid:
            continue
        ordered = sorted(valid, key=lambda item: item["selection_metrics"][metric], reverse=True)
        if len(ordered) == 1:
            ordered[0]["selection_score"] += weight
            continue
        denom = len(ordered) - 1
        for rank, candidate in enumerate(ordered):
            candidate["selection_score"] += weight * (1.0 - (rank / denom))

    return max(
        candidates,
        key=lambda item: (
            item["selection_score"],
            item["selection_metrics"]["min_occupancy"],
            item["selection_metrics"].get("stress_vol_ratio", float("-inf")),
            item["selection_metrics"]["log_likelihood_per_row"],
        ),
    )


def fit_hmm_bundle(
    train_features: pd.DataFrame,
    train_targets: pd.DataFrame,
    n_iter: int,
    random_state: int,
    n_init: int,
) -> Dict[str, Any]:
    feature_frame = train_features.drop(columns=MARKET_NON_FEATURE_COLUMNS, errors="ignore").copy()
    feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated(keep="last")]
    target_frame = train_targets.loc[
        :,
        ~train_targets.columns.duplicated(keep="last"),
    ][["forward_ret_5d", "forward_vol_5d", "vol_jump_5d", "drawdown"]].copy()
    scaler = fit_scaler(feature_frame)
    train_scaled = apply_scaler(feature_frame, scaler)
    train_frame = feature_frame.copy()
    for column in target_frame.columns:
        train_frame[column] = target_frame[column]

    candidates: List[Dict[str, Any]] = []
    errors: List[str] = []
    for seed in range(int(random_state), int(random_state) + max(int(n_init), 1)):
        try:
            model = GaussianHMM(
                n_components=N_STATES,
                covariance_type="diag",
                n_iter=int(n_iter),
                random_state=int(seed),
            )
            model.fit(train_scaled.values)
            raw_probs = model.predict_proba(train_scaled.values)
            occupancy = raw_probs.mean(axis=0)
            if np.any(occupancy < MIN_COLLAPSE_OCCUPANCY):
                raise ValueError(
                    "State collapse detected. "
                    f"Occupancy={np.round(occupancy, 6).tolist()} min={MIN_COLLAPSE_OCCUPANCY}"
                )

            state_stats = compute_state_statistics(raw_probs, train_frame)
            state_probs = compute_filtered_state_probs(model, train_scaled.values)
            transition_matrix = np.asarray(model.transmat_, dtype=float)
            shift_probability = compute_shift_probability(state_probs, transition_matrix)
            log_likelihood_per_row = float(model.score(train_scaled.values) / len(train_scaled))
            stress_state_id = identify_stress_state(state_stats)

            candidate = {
                "model": model,
                "seed": int(seed),
                "scaler": scaler,
                "feature_columns": list(feature_frame.columns),
                "raw_probs": raw_probs,
                "state_probs": state_probs,
                "transition_matrix": transition_matrix,
                "shift_probability": shift_probability,
                "stress_state_id": int(stress_state_id),
                "occupancy_raw": occupancy,
                "state_stats": state_stats,
                "selection_metrics": compute_candidate_selection_metrics(
                    train_frame=train_frame,
                    state_probs=state_probs,
                    transition_matrix=transition_matrix,
                    shift_probability=shift_probability,
                    log_likelihood_per_row=log_likelihood_per_row,
                    stress_state_id=stress_state_id,
                ),
                "train_start": feature_frame.index.min(),
                "train_end": feature_frame.index.max(),
            }
            candidates.append(candidate)
        except Exception as exc:
            errors.append(f"seed={seed}: {exc}")

    if not candidates:
        message = "; ".join(errors[-5:]) if errors else "no candidates produced"
        raise ValueError(f"All HMM fits failed. {message}")

    best_candidate = select_best_candidate(candidates)
    best_candidate["candidate_count"] = len(candidates)
    best_candidate["fit_errors"] = errors[-10:]
    best_candidate["candidate_summaries"] = [
        {
            "seed": candidate["seed"],
            "selection_score": float(candidate["selection_score"]),
            "selection_metrics": {
                key: (None if value is None else float(value))
                for key, value in candidate["selection_metrics"].items()
            },
            "stress_state_id": int(candidate["stress_state_id"]),
        }
        for candidate in candidates
    ]
    return best_candidate


def load_model_blob(artifact_dir: Path) -> Dict[str, Any]:
    model_path = Path(artifact_dir) / "model_state.pkl"
    with model_path.open("rb") as fh:
        blob = pickle.load(fh)
    return blob


def save_artifacts(
    artifact_dir: Path,
    bundle: Dict[str, Any],
    train_states: pd.DataFrame,
    train_window: str,
    retrain_cadence: str,
    n_iter: int,
    n_init: int,
    random_state: int,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_blob = {
        "model": bundle["model"],
        "feature_columns": bundle["feature_columns"],
        "state_labels": [state_label(idx) for idx in range(N_STATES)],
        "transition_matrix": np.asarray(bundle["transition_matrix"]).tolist(),
        "stress_state_id": int(bundle["stress_state_id"]),
        "scaler": {
            "mean": bundle["scaler"]["mean"].to_dict(),
            "std": bundle["scaler"]["std"].to_dict(),
            "lower": bundle["scaler"]["lower"].to_dict(),
            "upper": bundle["scaler"]["upper"].to_dict(),
        },
        "train_window": train_window,
        "retrain_cadence": retrain_cadence,
        "n_states": N_STATES,
        "n_iter": int(n_iter),
        "n_init": int(n_init),
        "random_state": int(random_state),
        "selected_seed": int(bundle["seed"]),
        "train_start": str(pd.Timestamp(bundle["train_start"]).date()),
        "train_end": str(pd.Timestamp(bundle["train_end"]).date()),
        "model_type": "GaussianHMM",
    }

    config = {
        "model_type": "GaussianHMM",
        "n_states": N_STATES,
        "feature_columns": bundle["feature_columns"],
        "train_window": train_window,
        "retrain_cadence": retrain_cadence,
        "state_label_map": {str(idx): state_label(idx) for idx in range(N_STATES)},
        "generated_at": datetime.now(UTC).isoformat(),
        "n_iter": int(n_iter),
        "n_init": int(n_init),
    }

    diagnostics = {
        "train_start": str(pd.Timestamp(bundle["train_start"]).date()),
        "train_end": str(pd.Timestamp(bundle["train_end"]).date()),
        "occupancy_raw": [float(x) for x in bundle["occupancy_raw"]],
        "state_stats": {
            str(key): {metric: float(value) for metric, value in values.items()}
            for key, values in bundle["state_stats"].items()
        },
        "transition_matrix": np.asarray(bundle["transition_matrix"]).tolist(),
        "stress_state_id": int(bundle["stress_state_id"]),
        "selected_seed": int(bundle["seed"]),
        "candidate_count": int(bundle["candidate_count"]),
        "selection_metrics": {
            key: (None if value is None else float(value))
            for key, value in bundle["selection_metrics"].items()
        },
        "candidate_summaries": bundle["candidate_summaries"],
        "fit_errors": bundle["fit_errors"],
        "generated_at": datetime.now(UTC).isoformat(),
    }

    with (artifact_dir / "model_state.pkl").open("wb") as fh:
        pickle.dump(model_blob, fh)
    (artifact_dir / "config.json").write_text(json.dumps(config, indent=2))
    (artifact_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    train_states.to_csv(artifact_dir / "train_states.csv", index=False)


def main() -> None:
    args = parse_args()
    end_date = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    start_date = compute_dataset_start(end_date, args.train_window, test_years=0)

    print(f"Building market dataset from {start_date.date()} to {end_date.date()}...")
    dataset = build_market_dataset(start_date, end_date)
    usable = dataset.drop(columns=MARKET_NON_FEATURE_COLUMNS, errors="ignore").dropna()
    asof_date = usable.index.max()
    train_features = select_training_features(
        dataset,
        asof_date=asof_date,
        train_window=args.train_window,
        min_train_rows=args.min_train_rows,
    )
    print(
        "Training window: "
        f"{train_features.index.min().date()} -> {train_features.index.max().date()} "
        f"({len(train_features)} rows)"
    )

    bundle = fit_hmm_bundle(
        train_features=train_features,
        train_targets=dataset.loc[train_features.index],
        n_iter=args.n_iter,
        random_state=args.random_state,
        n_init=args.n_init,
    )
    train_states = build_state_output(
        index=train_features.index,
        state_probs=bundle["state_probs"],
        shift_probability=bundle["shift_probability"],
        asof_date=asof_date,
        stress_state_id=bundle["stress_state_id"],
    )
    save_artifacts(
        artifact_dir=Path(args.artifact_dir),
        bundle=bundle,
        train_states=train_states,
        train_window=args.train_window,
        retrain_cadence=args.retrain_cadence,
        n_iter=args.n_iter,
        n_init=args.n_init,
        random_state=args.random_state,
    )

    occ = [float(x) for x in bundle["occupancy_raw"]]
    print(f"Raw state occupancy: {np.round(np.asarray(occ), 4).tolist()}")
    print(
        "Selected seed / candidates: "
        f"{bundle['seed']} / {bundle['candidate_count']}"
    )
    print(f"Artifacts saved to: {Path(args.artifact_dir)}")


if __name__ == "__main__":
    np.random.seed(DEFAULT_RANDOM_STATE)
    main()


