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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import get_history, parse_window


ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "artifacts" / "market"
TICKER_SPY = "SPY"
TICKER_VIX = "^VIX"
WINDOW_RET = 5
ZSCORE_WINDOW = 252
FEATURE_LOOKBACK_MAX = 252
DEFAULT_TRAIN_WINDOW = "3y"
DEFAULT_RETRAIN_CADENCE = "weekly"
N_STATES = 4
DEFAULT_N_ITER = 500
DEFAULT_RANDOM_STATE = 42
DEFAULT_MIN_TRAIN_ROWS = 252
MIN_COLLAPSE_OCCUPANCY = 0.01

STATE_LABELS = [
    "calm_bull",
    "choppy_bull",
    "calm_bear",
    "choppy_bear",
]

FEATURE_COLUMNS = [
    "spy_ret_1d",
    "spy_ret_5d",
    "spy_vol_1d",
    "spy_vol_5d",
    "spy_vol_20d",
    "vix_level_z",
    "vix_chg_1d",
    "vix_chg_5d",
    "vix_vol_1d",
    "vix_vol_5d",
    "trend_gap_20d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone 4-state market HMM regime model.")
    parser.add_argument("--train-window", default=DEFAULT_TRAIN_WINDOW, help="Rolling train window like '3y'.")
    parser.add_argument("--end", default=None, help="Training end date YYYY-MM-DD (default: today).")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR_DEFAULT))
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
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


def compute_dataset_start(end_date: pd.Timestamp, train_window: str, test_years: int = 0) -> pd.Timestamp:
    train_offset = parse_window(train_window)
    buffer_days = max(FEATURE_LOOKBACK_MAX, ZSCORE_WINDOW) + (2 * WINDOW_RET) + 10
    start = pd.Timestamp(end_date).normalize() - train_offset - pd.DateOffset(days=buffer_days)
    if test_years > 0:
        start = start - pd.DateOffset(years=int(test_years))
    return start


def build_market_features(price_spy: pd.Series, price_vix: pd.Series) -> pd.DataFrame:
    index = price_spy.index
    price_vix = price_vix.reindex(index).ffill()

    spy_ret_1d = compute_log_return(price_spy, 1)
    vix_chg_1d = compute_log_return(price_vix, 1)

    features = pd.DataFrame(index=index)
    features["spy_ret_1d"] = spy_ret_1d
    features["spy_ret_5d"] = compute_log_return(price_spy, WINDOW_RET)
    features["spy_vol_1d"] = spy_ret_1d.abs()
    features["spy_vol_5d"] = spy_ret_1d.rolling(5).std()
    features["spy_vol_20d"] = spy_ret_1d.rolling(20).std()
    features["vix_level_z"] = zscore_trailing(price_vix, ZSCORE_WINDOW)
    features["vix_chg_1d"] = vix_chg_1d
    features["vix_chg_5d"] = compute_log_return(price_vix, 5)
    features["vix_vol_1d"] = vix_chg_1d.abs()
    features["vix_vol_5d"] = vix_chg_1d.rolling(5).std()
    features["trend_gap_20d"] = (price_spy / price_spy.rolling(20).mean()) - 1.0
    return features


def build_market_targets(price_spy: pd.Series) -> pd.DataFrame:
    spy_ret_1d = compute_log_return(price_spy, 1)
    prev_ret_5d = compute_log_return(price_spy, WINDOW_RET)
    forward_ret_5d = np.log(price_spy.shift(-WINDOW_RET) / price_spy)
    forward_vol_5d = spy_ret_1d.rolling(WINDOW_RET).std().shift(-WINDOW_RET)
    drawdown = (price_spy / price_spy.cummax()) - 1.0

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

    features = build_market_features(price_spy, price_vix)
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
    train_df = dataset.loc[(dataset.index > train_start) & (dataset.index <= asof_date), FEATURE_COLUMNS]
    train_df = train_df.dropna()
    if train_df.empty or len(train_df) < int(min_train_rows):
        raise ValueError(
            f"Insufficient training rows for asof={asof_date.date()}. "
            f"rows={len(train_df)} min_required={min_train_rows}"
        )
    return train_df


def fit_scaler(train_features: pd.DataFrame) -> Dict[str, pd.Series]:
    mean = train_features.mean()
    std = train_features.std(ddof=0).replace(0.0, 1.0)
    return {"mean": mean, "std": std}


def apply_scaler(frame: pd.DataFrame, scaler: Dict[str, Any]) -> pd.DataFrame:
    mean = pd.Series(scaler["mean"], dtype=float).reindex(FEATURE_COLUMNS)
    std = pd.Series(scaler["std"], dtype=float).replace(0.0, 1.0).reindex(FEATURE_COLUMNS)
    scaled = (frame[FEATURE_COLUMNS] - mean) / std
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    return scaled


def compute_state_statistics(raw_probs: np.ndarray, train_features: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    ret = train_features["spy_ret_5d"].to_numpy(dtype=float)
    vol = train_features["spy_vol_5d"].to_numpy(dtype=float)
    stats: Dict[int, Dict[str, float]] = {}
    n_rows = float(len(train_features))

    for state in range(raw_probs.shape[1]):
        weights = raw_probs[:, state]
        denom = float(weights.sum())
        if denom <= 0:
            stats[state] = {
                "occupancy": 0.0,
                "spy_ret_5d_mean": float("nan"),
                "spy_vol_5d_mean": float("nan"),
            }
            continue
        stats[state] = {
            "occupancy": denom / n_rows,
            "spy_ret_5d_mean": float(np.dot(weights, ret) / denom),
            "spy_vol_5d_mean": float(np.dot(weights, vol) / denom),
        }
    return stats


def canonicalize_state_order(state_stats: Dict[int, Dict[str, float]]) -> Tuple[List[int], Dict[int, int]]:
    states = sorted(state_stats.keys())
    by_ret_desc = sorted(states, key=lambda s: state_stats[s]["spy_ret_5d_mean"], reverse=True)
    bull_states = by_ret_desc[:2]
    bear_states = by_ret_desc[2:]

    calm_bull_raw = min(bull_states, key=lambda s: state_stats[s]["spy_vol_5d_mean"])
    choppy_bull_raw = max(bull_states, key=lambda s: state_stats[s]["spy_vol_5d_mean"])
    calm_bear_raw = min(bear_states, key=lambda s: state_stats[s]["spy_vol_5d_mean"])
    choppy_bear_raw = max(bear_states, key=lambda s: state_stats[s]["spy_vol_5d_mean"])

    canonical_to_raw = [calm_bull_raw, choppy_bull_raw, calm_bear_raw, choppy_bear_raw]
    if len(set(canonical_to_raw)) != len(canonical_to_raw):
        raise ValueError(f"Canonical mapping is not unique: {canonical_to_raw}")

    raw_to_canonical = {raw: canonical for canonical, raw in enumerate(canonical_to_raw)}
    return canonical_to_raw, raw_to_canonical


def remap_transition_matrix(raw_transition: np.ndarray, canonical_to_raw: List[int]) -> np.ndarray:
    raw_transition = np.asarray(raw_transition, dtype=float)
    return raw_transition[np.ix_(canonical_to_raw, canonical_to_raw)]


def _logsumexp(vec: np.ndarray) -> float:
    vmax = float(np.max(vec))
    if not np.isfinite(vmax):
        return vmax
    return vmax + float(np.log(np.sum(np.exp(vec - vmax))))


def compute_filtered_canonical_probs(
    model: GaussianHMM,
    scaled_values: np.ndarray,
    canonical_to_raw: List[int],
) -> np.ndarray:
    """Forward-only state posteriors p(s_t | x_1..x_t) in canonical state order."""
    values = np.asarray(scaled_values, dtype=float)
    raw_to_use = [int(idx) for idx in canonical_to_raw]
    log_emlik_raw = model._compute_log_likelihood(values)
    log_emlik = log_emlik_raw[:, raw_to_use]

    start_raw = np.asarray(model.startprob_, dtype=float)
    trans_raw = np.asarray(model.transmat_, dtype=float)
    start = np.clip(start_raw[raw_to_use], 1e-300, 1.0)
    trans = np.clip(trans_raw[np.ix_(raw_to_use, raw_to_use)], 1e-300, 1.0)

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


def compute_shift_probability(canonical_probs: np.ndarray, canonical_transition: np.ndarray) -> np.ndarray:
    """Same-date shift estimate: P(s_{t+1} != s_t | x_1..x_t)."""
    diag = np.clip(np.diag(np.asarray(canonical_transition, dtype=float)), 0.0, 1.0)
    same_prob = np.sum(canonical_probs * diag, axis=1)
    same_prob = np.clip(same_prob, 0.0, 1.0)
    return 1.0 - same_prob


def build_state_output(
    index: pd.DatetimeIndex,
    canonical_probs: np.ndarray,
    shift_probability: np.ndarray,
    asof_date: pd.Timestamp,
) -> pd.DataFrame:
    state_id = np.argmax(canonical_probs, axis=1).astype(int)
    out = pd.DataFrame(index=index)
    out["date"] = pd.to_datetime(index).normalize()
    out["state_id"] = state_id
    out["state_label"] = [STATE_LABELS[item] for item in state_id]
    out["p_state_0"] = canonical_probs[:, 0]
    out["p_state_1"] = canonical_probs[:, 1]
    out["p_state_2"] = canonical_probs[:, 2]
    out["p_state_3"] = canonical_probs[:, 3]
    out["shift_prob"] = shift_probability
    out["asof"] = pd.Timestamp(asof_date).normalize()
    return out.reset_index(drop=True)


def fit_hmm_bundle(
    train_features: pd.DataFrame,
    n_iter: int,
    random_state: int,
) -> Dict[str, Any]:
    scaler = fit_scaler(train_features)
    train_scaled = apply_scaler(train_features, scaler)

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=int(n_iter),
        random_state=int(random_state),
    )
    model.fit(train_scaled.values)
    raw_probs = model.predict_proba(train_scaled.values)
    occupancy = raw_probs.mean(axis=0)
    if np.any(occupancy < MIN_COLLAPSE_OCCUPANCY):
        raise ValueError(
            "State collapse detected. "
            f"Occupancy={np.round(occupancy, 6).tolist()} min={MIN_COLLAPSE_OCCUPANCY}"
        )

    state_stats = compute_state_statistics(raw_probs, train_features)
    canonical_to_raw, raw_to_canonical = canonicalize_state_order(state_stats)
    canonical_transmat = remap_transition_matrix(model.transmat_, canonical_to_raw)
    canonical_probs = compute_filtered_canonical_probs(model, train_scaled.values, canonical_to_raw)
    shift_probability = compute_shift_probability(canonical_probs, canonical_transmat)

    return {
        "model": model,
        "scaler": scaler,
        "raw_probs": raw_probs,
        "canonical_probs": canonical_probs,
        "canonical_transmat": canonical_transmat,
        "shift_probability": shift_probability,
        "canonical_to_raw": canonical_to_raw,
        "raw_to_canonical": raw_to_canonical,
        "occupancy_raw": occupancy,
        "state_stats_raw": state_stats,
        "train_start": train_features.index.min(),
        "train_end": train_features.index.max(),
    }


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
    random_state: int,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_blob = {
        "model": bundle["model"],
        "feature_columns": FEATURE_COLUMNS,
        "state_labels": STATE_LABELS,
        "canonical_to_raw": bundle["canonical_to_raw"],
        "raw_to_canonical": bundle["raw_to_canonical"],
        "canonical_transmat": bundle["canonical_transmat"],
        "scaler": {
            "mean": bundle["scaler"]["mean"].to_dict(),
            "std": bundle["scaler"]["std"].to_dict(),
        },
        "train_window": train_window,
        "retrain_cadence": retrain_cadence,
        "n_states": N_STATES,
        "n_iter": int(n_iter),
        "random_state": int(random_state),
        "train_start": str(pd.Timestamp(bundle["train_start"]).date()),
        "train_end": str(pd.Timestamp(bundle["train_end"]).date()),
        "model_type": "GaussianHMM",
    }

    config = {
        "model_type": "GaussianHMM",
        "n_states": N_STATES,
        "feature_columns": FEATURE_COLUMNS,
        "train_window": train_window,
        "retrain_cadence": retrain_cadence,
        "state_label_map": {str(idx): label for idx, label in enumerate(STATE_LABELS)},
        "generated_at": datetime.now(UTC).isoformat(),
    }

    diagnostics = {
        "train_start": str(pd.Timestamp(bundle["train_start"]).date()),
        "train_end": str(pd.Timestamp(bundle["train_end"]).date()),
        "occupancy_raw": [float(x) for x in bundle["occupancy_raw"]],
        "state_stats_raw": {
            str(key): {metric: float(value) for metric, value in values.items()}
            for key, values in bundle["state_stats_raw"].items()
        },
        "canonical_to_raw": [int(x) for x in bundle["canonical_to_raw"]],
        "canonical_transition_matrix": np.asarray(bundle["canonical_transmat"]).tolist(),
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
    usable = dataset[FEATURE_COLUMNS].dropna()
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
        n_iter=args.n_iter,
        random_state=args.random_state,
    )
    train_states = build_state_output(
        index=train_features.index,
        canonical_probs=bundle["canonical_probs"],
        shift_probability=bundle["shift_probability"],
        asof_date=asof_date,
    )
    save_artifacts(
        artifact_dir=Path(args.artifact_dir),
        bundle=bundle,
        train_states=train_states,
        train_window=args.train_window,
        retrain_cadence=args.retrain_cadence,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )

    occ = [float(x) for x in bundle["occupancy_raw"]]
    print(f"Raw state occupancy: {np.round(np.asarray(occ), 4).tolist()}")
    print(f"Artifacts saved to: {Path(args.artifact_dir)}")


if __name__ == "__main__":
    np.random.seed(DEFAULT_RANDOM_STATE)
    main()


