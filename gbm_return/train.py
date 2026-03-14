import argparse
import json
import pickle
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import (
    DEFAULT_SECTOR_ETF_MAP,
    canonicalize_sector_name,
    get_history,
    get_info,
    parse_window,
    walk_forward_splits,
)
from gbm_return.configuration import (
    BASE_LGBM_PARAMS,
    FEATURE_SET_CHOICES,
    FEATURE_SET_F0,
    apply_feature_set,
    resolve_feature_drops,
    resolve_lgbm_params,
)


ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_VARIANT_REGULAR = "regular"
TICKER_GOLD = "GLD"
TICKER_SPY = "SPY"
TICKER_VIX = "^VIX"
WINDOW_RET = 5
DATA_YEARS = 3
DEFAULT_TRAIN_WINDOW = "2y"
DEFAULT_TEST_WINDOW = "1m"
DEFAULT_STEP_WINDOW = "1m"
FEATURE_LOOKBACK_MAX = 60
REGIME_SCORE_WINDOW = 252
REGIME_SCORE_CLIP = 4.0
REGIME_SCORE_WEIGHTS = {"vix": 0.5, "spy_vol": 0.5}
MIN_TRAIN_ROWS = 60
LGBM_PARAMS = dict(BASE_LGBM_PARAMS)
FEATURE_SET_FILE_DEFAULT = ARTIFACT_DIR_DEFAULT / "feature_sets.json"
VALID_DIRECTION_MODES = ("long_short", "long_only", "short_only")
DEFAULT_DIRECTION_MODE = "long_short"
DEFAULT_TRAINING_POLICY = {
    "target_clip_lower_quantile": 0.02,
    "target_clip_upper_quantile": 0.98,
    "recency_min_weight": 0.35,
}


def resolve_artifact_variant() -> str:
    return ARTIFACT_VARIANT_REGULAR


def resolve_direction_mode(direction_mode: str | None = None) -> str:
    direction_mode = str(direction_mode or DEFAULT_DIRECTION_MODE)
    if direction_mode not in VALID_DIRECTION_MODES:
        valid = ", ".join(VALID_DIRECTION_MODES)
        raise ValueError(f"Unknown direction_mode '{direction_mode}'. Valid values: {valid}")
    return direction_mode


def resolve_training_policy(overrides: dict | None = None) -> dict:
    resolved = dict(DEFAULT_TRAINING_POLICY)
    if overrides:
        resolved.update(overrides)

    lower_q = float(resolved["target_clip_lower_quantile"])
    upper_q = float(resolved["target_clip_upper_quantile"])
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("Target clip quantiles must satisfy 0 <= lower < upper <= 1.")

    recency_min_weight = float(resolved["recency_min_weight"])
    if not 0.0 < recency_min_weight <= 1.0:
        raise ValueError("recency_min_weight must be in (0, 1].")

    resolved["target_clip_lower_quantile"] = lower_q
    resolved["target_clip_upper_quantile"] = upper_q
    resolved["recency_min_weight"] = recency_min_weight
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(description="Train GBM return model.")
    parser.add_argument(
        "--train-window",
        default=DEFAULT_TRAIN_WINDOW,
        help="Rolling train window like '2y', '18m', or '260d'.",
    )
    parser.add_argument(
        "--drop-time-index",
        action="store_true",
        help="Exclude time_index from training features (default).",
    )
    parser.add_argument(
        "--include-time-index",
        action="store_true",
        help="Include time_index in training features.",
    )
    parser.add_argument(
        "--feature-set",
        default=FEATURE_SET_F0,
        choices=FEATURE_SET_CHOICES,
        help="Feature set variant. F1/F2 are loaded from --feature-set-file.",
    )
    parser.add_argument(
        "--feature-set-file",
        default=str(FEATURE_SET_FILE_DEFAULT),
        help="Path to feature_sets.json used for F1/F2 drop lists.",
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
    parser.set_defaults(drop_time_index=True)
    return parser.parse_args()


def prompt_tickers():
    raw = input("Enter tickers (comma-separated): ").strip()
    tokens = [token.strip().upper() for token in re.split(r"[,\s]+", raw) if token.strip()]
    seen = set()
    tickers = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            tickers.append(token)
    return tickers


def resolve_sector_etf(ticker):
    sector = None
    etf = None
    error = None
    try:
        info = get_info(ticker)
        sector_raw = info.get("sector") or info.get("sectorKey")
        if sector_raw:
            sector_key = canonicalize_sector_name(sector_raw) or sector_raw.strip()
            if sector_key in DEFAULT_SECTOR_ETF_MAP:
                etf = DEFAULT_SECTOR_ETF_MAP[sector_key]
                sector = sector_key
            else:
                error = f"Sector '{sector_raw}' resolved to '{sector_key}' not in DEFAULT_SECTOR_ETF_MAP."
        else:
            error = "Sector missing from ticker info."
    except Exception as exc:
        error = str(exc)

    if etf is None:
        return TICKER_SPY, sector, error
    return etf, sector, None


def fetch_history_cached(symbol, start_date, end_date, cache):
    key = symbol.upper()
    if key not in cache:
        end_exclusive = (
            pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
            if end_date is not None
            else None
        )
        history = get_history(
            symbol,
            period=None,
            start=str(pd.Timestamp(start_date).date()),
            end=str(end_exclusive.date()) if end_exclusive is not None else None,
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


def extract_field(history, field, symbol):
    if field not in history.columns:
        raise KeyError(f"Missing field {field} in data for {symbol}.")
    series = history[field].copy()
    if series.empty:
        raise ValueError(f"No {field} data returned for {symbol}.")
    return series


def compute_log_return(series, window=1):
    ratio = series / series.shift(window)
    ratio = ratio.replace(0.0, np.nan)
    log_ret = np.log(ratio)
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
    return log_ret


def zscore_trailing(series, window, min_periods=20):
    roll_mean = series.rolling(window, min_periods=min_periods).mean()
    roll_std = series.rolling(window, min_periods=min_periods).std()
    z = (series - roll_mean) / roll_std
    return z.replace([np.inf, -np.inf], np.nan)


def normalize_regime_weights(weights):
    if not weights:
        return {"vix": 0.5, "spy_vol": 0.5}
    w_vix = float(weights.get("vix", 0.5))
    w_spy = float(weights.get("spy_vol", 0.5))
    total = w_vix + w_spy
    if total <= 0:
        return {"vix": 0.5, "spy_vol": 0.5}
    return {"vix": w_vix / total, "spy_vol": w_spy / total}


def compute_regime_score(price_vix, price_spy, window, clip, weights):
    log_ret_spy = compute_log_return(price_spy, 1)
    spy_vol_20d = log_ret_spy.rolling(20).std()

    z_vix = zscore_trailing(price_vix, window)
    z_spy = zscore_trailing(spy_vol_20d, window)

    z_vix = z_vix.clip(lower=0, upper=clip)
    z_spy = z_spy.clip(lower=0, upper=clip)

    weights = normalize_regime_weights(weights)
    score = (weights["vix"] * (z_vix / clip)) + (weights["spy_vol"] * (z_spy / clip))
    score = score.clip(lower=0, upper=1)
    return score.rename("regime_score")


def trading_day_in_quarter(index):
    positions = pd.Series(np.arange(len(index)), index=index)
    quarters = index.to_period("Q")
    first_pos = positions.groupby(quarters).transform("min")
    last_pos = positions.groupby(quarters).transform("max")
    day_in_quarter = (positions - first_pos).astype(int)
    quarter_len = (last_pos - first_pos + 1).astype(int)
    return day_in_quarter, quarter_len


def build_features(price_stock, price_sector, price_gld, price_spy, price_vix):
    index = price_stock.index

    price_sector = price_sector.reindex(index).ffill()
    price_gld = price_gld.reindex(index).ffill()
    price_spy = price_spy.reindex(index).ffill()
    price_vix = price_vix.reindex(index).ffill()

    log_ret_stock = compute_log_return(price_stock, 1)
    log_ret_sector = compute_log_return(price_sector, 1)
    log_ret_gld = compute_log_return(price_gld, 1)
    log_ret_spy = compute_log_return(price_spy, 1)

    features = pd.DataFrame(index=price_stock.index)
    features["time_index"] = (features.index - features.index[0]).days.astype(int)
    features["ret_1d"] = log_ret_stock
    features["ret_5d"] = compute_log_return(price_stock, WINDOW_RET)
    features["ret_10d"] = compute_log_return(price_stock, 10)
    ret_20d = compute_log_return(price_stock, 20)
    features["ret_20d"] = ret_20d
    features["ret_60d"] = compute_log_return(price_stock, 60)
    features["vol_5d"] = log_ret_stock.rolling(WINDOW_RET).std()
    features["vol_10d"] = log_ret_stock.rolling(10).std()
    features["vol_20d"] = log_ret_stock.rolling(20).std()
    features["vol_60d"] = log_ret_stock.rolling(60).std()
    features["skew_20d"] = log_ret_stock.rolling(20).skew()
    features["stock_ma20_gap"] = (price_stock / price_stock.rolling(20).mean()) - 1.0
    features["stock_ma60_gap"] = (price_stock / price_stock.rolling(60).mean()) - 1.0
    roll_max_60 = price_stock.rolling(60).max()
    features["drawdown_60d"] = (price_stock / roll_max_60) - 1.0
    features["vol_ratio_5_20"] = (features["vol_5d"] / features["vol_20d"]).replace(
        [np.inf, -np.inf], np.nan
    )
    features["sector_ret_5d"] = compute_log_return(price_sector, WINDOW_RET)
    features["sector_vol_5d"] = log_ret_sector.rolling(WINDOW_RET).std()
    features["rel_strength_sector_20d"] = ret_20d - compute_log_return(price_sector, 20)
    features["gld_ret_5d"] = compute_log_return(price_gld, WINDOW_RET)
    features["gld_vol_5d"] = log_ret_gld.rolling(WINDOW_RET).std()
    features["spy_ret_5d"] = compute_log_return(price_spy, WINDOW_RET)
    features["spy_vol_20d"] = log_ret_spy.rolling(20).std()
    features["spy_ma20_gap"] = (price_spy / price_spy.rolling(20).mean()) - 1.0
    features["vix_chg_1d"] = compute_log_return(price_vix, 1)
    day_in_quarter, quarter_len = trading_day_in_quarter(features.index)
    quarter_len = quarter_len.replace(0, 1)
    phase = (2.0 * np.pi * day_in_quarter) / quarter_len
    features["q_phase_sin"] = np.sin(phase)
    features["q_phase_cos"] = np.cos(phase)

    # features["vol_chg_1d"] = volume_stock.pct_change()
    # features["momentum_5_20"] = features["ret_5d"] - ret_20d
    # features["sector_ret_1d"] = log_ret_sector
    # features["gld_ret_1d"] = log_ret_gld
    # features["vix_level"] = price_vix
    # features["corr_spy_60d"] = log_ret_stock.rolling(60).corr(log_ret_spy)

    return features


def set_time_index(features, start_date):
    features = features.copy()
    features["time_index"] = (features.index - start_date).days.astype(int)
    return features


def build_target(price_stock):
    forward_price = price_stock.shift(-WINDOW_RET)
    ratio = forward_price / price_stock
    target = np.log(ratio.replace(0.0, np.nan))
    target = target.replace([np.inf, -np.inf], np.nan)
    return target.rename("target")


def find_all_nan_columns(frame):
    return [col for col in frame.columns if frame[col].isna().all()]


def summarize_series(series):
    non_na = series.dropna()
    if non_na.empty:
        return {"count": 0, "start": None, "end": None}
    return {
        "count": int(non_na.shape[0]),
        "start": str(non_na.index.min().date()),
        "end": str(non_na.index.max().date()),
    }


def validate_alignment_and_nan(
    ticker,
    features,
    target,
    price_stock,
    price_sector,
    price_gld,
    price_spy,
    price_vix,
):
    if not features.index.equals(target.index):
        raise ValueError(
            f"{ticker}: Feature/target index mismatch. "
            f"features={features.index.min().date()}..{features.index.max().date()} "
            f"target={target.index.min().date()}..{target.index.max().date()}"
        )

    all_nan_features = find_all_nan_columns(features)
    if all_nan_features:
        coverage = {
            "stock": summarize_series(price_stock),
            "sector": summarize_series(price_sector),
            "gld": summarize_series(price_gld),
            "spy": summarize_series(price_spy),
            "vix": summarize_series(price_vix),
        }
        raise ValueError(
            f"{ticker}: All-NaN feature columns detected: {', '.join(all_nan_features)}. "
            f"Series coverage: {coverage}"
        )

    if target.isna().all():
        coverage = summarize_series(price_stock)
        raise ValueError(
            f"{ticker}: Target is all NaN. "
            f"Check price coverage and window settings. Stock coverage: {coverage}"
        )

def select_feature_columns(
    dataset: pd.DataFrame,
    drop_time_index: bool,
    feature_set: str = FEATURE_SET_F0,
    feature_set_file: str | None = None,
) -> list[str]:
    feature_cols = [col for col in dataset.columns if col != "target"]
    if drop_time_index:
        feature_cols = [col for col in feature_cols if col != "time_index"]
    feature_cols, missing = apply_feature_set(
        feature_cols=feature_cols,
        feature_set=feature_set,
        feature_set_file=feature_set_file,
    )
    if missing:
        print(
            f"feature_set={feature_set}: ignoring drop features not present in current columns: "
            f"{', '.join(missing)}"
    )
    return feature_cols

def select_feature_columns(
    dataset: pd.DataFrame,
    drop_time_index: bool,
    feature_set: str = FEATURE_SET_F0,
    feature_set_file: str | None = None,
) -> list[str]:
    feature_cols = [col for col in dataset.columns if col != "target"]
    if drop_time_index:
        feature_cols = [col for col in feature_cols if col != "time_index"]
    feature_cols, missing = apply_feature_set(
        feature_cols=feature_cols,
        feature_set=feature_set,
        feature_set_file=feature_set_file,
    )
    if missing:
        print(
            f"feature_set={feature_set}: ignoring drop features not present in current columns: "
            f"{', '.join(missing)}"
        )
    return feature_cols


def latest_train_window(data, train_window, min_train_rows=MIN_TRAIN_ROWS):
    if data.empty:
        raise ValueError("Cannot build final training window from empty dataset.")

    train_offset = parse_window(train_window)
    latest_end = data.index.max()
    latest_start = latest_end - train_offset
    train_df = data.loc[(data.index > latest_start) & (data.index <= latest_end)]

    if len(train_df) < min_train_rows:
        raise ValueError(
            f"Not enough rows for final training window ({len(train_df)} < {min_train_rows}). "
            f"window={train_window}, latest_end={latest_end.date()}"
        )

    return train_df


def clip_target_series(target: pd.Series, training_policy: dict | None = None):
    policy = resolve_training_policy(training_policy)
    lower_bound = float(target.quantile(policy["target_clip_lower_quantile"]))
    upper_bound = float(target.quantile(policy["target_clip_upper_quantile"]))
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        clipped = target.copy()
        lower_bound = None
        upper_bound = None
    else:
        clipped = target.clip(lower=lower_bound, upper=upper_bound)
    return clipped, {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_quantile": policy["target_clip_lower_quantile"],
        "upper_quantile": policy["target_clip_upper_quantile"],
    }


def compute_recency_weights(index: pd.Index, min_weight: float) -> pd.Series:
    if len(index) <= 1:
        return pd.Series(1.0, index=index, dtype=float)
    progress = np.linspace(0.0, 1.0, len(index), dtype=float)
    weights = min_weight + (1.0 - min_weight) * np.power(progress, 1.5)
    return pd.Series(weights, index=index, dtype=float)


def prepare_lgbm_training_data(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    training_policy: dict | None = None,
):
    policy = resolve_training_policy(training_policy)
    train_x = train_df[feature_cols]
    clipped_target, clip_info = clip_target_series(train_df["target"], policy)
    sample_weight = compute_recency_weights(train_df.index, policy["recency_min_weight"])
    metadata = {
        "target_clip": clip_info,
        "sample_weight": {
            "min_weight": policy["recency_min_weight"],
            "max_weight": 1.0,
        },
    }
    return train_x, clipped_target, sample_weight, metadata


def train_lgbm(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    params: dict,
    sample_weight: pd.Series | np.ndarray | None = None,
):
    model = lgb.LGBMRegressor(**params)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)
    model.fit(train_x, train_y, **fit_kwargs)
    return model


def evaluate(model, test_x: pd.DataFrame, test_y: pd.Series):
    pred = pd.Series(model.predict(test_x), index=test_y.index)
    mae = float(np.mean(np.abs(pred - test_y)))
    mse = float(np.mean((pred - test_y) ** 2))
    pred_simple = np.exp(pred) - 1.0
    actual_simple = np.exp(test_y) - 1.0
    mae_simple = float(np.mean(np.abs(pred_simple - actual_simple)))
    directional = float(np.mean(np.sign(pred) == np.sign(test_y)))
    corr = float(pred.corr(test_y)) if len(pred) > 1 else None
    return {
        "mae": mae,
        "mse": mse,
        "mae_simple": mae_simple,
        "directional": directional,
        "corr": corr,
        "coverage_95": None,
        "avg_interval_width": None,
    }


def summarize_fold_metrics(fold_metrics):
    mae_values = [fold["mae"] for fold in fold_metrics]
    mse_values = [fold["mse"] for fold in fold_metrics]
    mae_simple_values = [fold["mae_simple"] for fold in fold_metrics]
    dir_values = [fold["directional"] for fold in fold_metrics]
    corr_values = [fold["corr"] for fold in fold_metrics if fold.get("corr") is not None]
    summary = {
        "folds": len(fold_metrics),
        "mae_mean": float(np.mean(mae_values)) if mae_values else None,
        "mae_median": float(np.median(mae_values)) if mae_values else None,
        "mse_mean": float(np.mean(mse_values)) if mse_values else None,
        "mse_median": float(np.median(mse_values)) if mse_values else None,
        "mae_simple_mean": float(np.mean(mae_simple_values)) if mae_simple_values else None,
        "mae_simple_median": float(np.median(mae_simple_values)) if mae_simple_values else None,
        "directional_mean": float(np.mean(dir_values)) if dir_values else None,
        "directional_median": float(np.median(dir_values)) if dir_values else None,
        "corr_mean": float(np.mean(corr_values)) if corr_values else None,
        "corr_median": float(np.median(corr_values)) if corr_values else None,
        "coverage_95_mean": None,
        "coverage_95_median": None,
        "avg_interval_width_mean": None,
        "avg_interval_width_median": None,
    }
    return summary


def save_artifacts(
    artifact_dir,
    model,
    fold_metrics,
    summary_metrics,
    config,
    feature_cols,
):
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model_state.pt"
    scaler_path = artifact_dir / "scaler.json"
    metrics_path = artifact_dir / "metrics.json"
    config_path = artifact_dir / "config.json"

    payload = {
        "model_str": model.booster_.model_to_string(),
        "feature_columns": feature_cols,
    }
    with model_path.open("wb") as fh:
        pickle.dump(payload, fh)

    if scaler_path.exists():
        scaler_path.unlink()

    metrics_out = {
        "summary": summary_metrics,
        "folds": fold_metrics,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    config_path.write_text(json.dumps(config, indent=2))
    print(f"\nArtifacts saved to: {artifact_dir}")


def compute_start_date(end_date, data_years, train_window, test_window, regime_score_window):
    start_date = end_date - pd.DateOffset(years=data_years)
    train_offset = parse_window(train_window)
    test_offset = parse_window(test_window)
    buffer_days = max(FEATURE_LOOKBACK_MAX, regime_score_window) + (2 * WINDOW_RET) + 5
    min_start = end_date - train_offset
    min_start = min_start - test_offset
    min_start = min_start - pd.DateOffset(days=buffer_days)
    if min_start < start_date:
        start_date = min_start
    return start_date


def build_model_dataset(ticker, config, history_cache):
    end_date = pd.Timestamp.today().normalize()
    start_date = compute_start_date(
        end_date,
        config["data_years"],
        config["train_window"],
        config["test_window"],
        config["regime_score"]["score_window"],
    )

    sector_etf, sector_name, sector_error = resolve_sector_etf(ticker)
    if sector_error:
        print(f"{ticker}: sector fallback to {sector_etf} ({sector_error})")

    stock_history = fetch_history_cached(ticker, start_date, end_date, history_cache)
    sector_history = fetch_history_cached(sector_etf, start_date, end_date, history_cache)
    gld_history = fetch_history_cached(TICKER_GOLD, start_date, end_date, history_cache)
    spy_history = fetch_history_cached(TICKER_SPY, start_date, end_date, history_cache)
    vix_history = fetch_history_cached(TICKER_VIX, start_date, end_date, history_cache)

    price_stock = extract_field(stock_history, "Close", ticker)
    price_sector = extract_field(sector_history, "Close", sector_etf)
    price_gld = extract_field(gld_history, "Close", TICKER_GOLD)
    price_spy = extract_field(spy_history, "Close", TICKER_SPY)
    price_vix = extract_field(vix_history, "Close", TICKER_VIX)

    features = build_features(price_stock, price_sector, price_gld, price_spy, price_vix)
    target = build_target(price_stock)
    regime_cfg = config["regime_score"]

    if regime_cfg.get("enabled", True):
        regime_score = compute_regime_score(
            price_vix.reindex(price_stock.index).ffill(),
            price_spy.reindex(price_stock.index).ffill(),
            regime_cfg["score_window"],
            regime_cfg["score_clip"],
            regime_cfg["weights"],
        )
    else:
        regime_score = pd.Series(0.0, index=price_stock.index, name="regime_score")

    validate_alignment_and_nan(
        ticker,
        features,
        target,
        price_stock,
        price_sector,
        price_gld,
        price_spy,
        price_vix,
    )

    dataset = features.join([target])
    dataset["regime_score"] = regime_score
    before_rows = int(dataset.shape[0])
    nan_counts = dataset.isna().sum().sort_values(ascending=False)
    dataset = dataset.dropna()
    if dataset.empty:
        nan_summary = nan_counts.head(5).to_dict()
        raise ValueError(
            f"{ticker}: No rows left after feature/target alignment. "
            f"Rows before dropna={before_rows}. Top NaN columns: {nan_summary}"
        )

    train_offset = parse_window(config["train_window"])
    test_offset = parse_window(config["test_window"])
    min_required_end = dataset.index.min() + train_offset + test_offset
    last_usable = dataset.index.max()
    if last_usable <= min_required_end:
        raise ValueError(
            f"{ticker}: Not enough data after alignment for walk-forward windows. "
            f"Last usable date={last_usable.date()} but need >= {min_required_end.date()} "
            f"for train_window={config['train_window']} and test_window={config['test_window']}. "
            f"Forward target trims the last {WINDOW_RET} rows."
        )

    return dataset, sector_etf, sector_name


def train_for_ticker(ticker, config, history_cache):
    dataset, sector_etf, sector_name = build_model_dataset(ticker, config, history_cache)
    feature_cols = select_feature_columns(
        dataset=dataset,
        drop_time_index=bool(config.get("drop_time_index", True)),
        feature_set=str(config.get("feature_set", FEATURE_SET_F0)),
        feature_set_file=config.get("feature_set_file"),
    )

    splits = list(
        walk_forward_splits(
            dataset,
            train_window=config["train_window"],
            test_window=config["test_window"],
            embargo=config["window_ret"],
            step=config["step_window"],
            min_train_rows=MIN_TRAIN_ROWS,
        )
    )
    fold_metrics = []
    print(f"\nTraining {ticker} (sector ETF: {sector_etf})")
    for split in splits:
        train_df = set_time_index(split.train.copy(), split.train_start)
        test_df = set_time_index(split.test.copy(), split.train_start)
        train_x, train_y, sample_weight, train_meta = prepare_lgbm_training_data(
            train_df,
            feature_cols,
            config.get("training_policy"),
        )
        test_x = test_df[feature_cols]
        test_y = test_df["target"]

        model = train_lgbm(
            train_x,
            train_y,
            config["lgbm_params"],
            sample_weight=sample_weight,
        )
        metrics = evaluate(model, test_x, test_y)
        print(
            f"Fold {split.fold} | Train: {split.train_start.date()} -> {split.train_end.date()} | "
            f"Test: {split.test_start.date()} -> {split.test_end.date()} | "
            f"MAE(log): {metrics['mae']:.6f} | MAE(simple): {metrics['mae_simple']:.4%} | "
            f"MSE: {metrics['mse']:.6f} | Dir: {metrics['directional']:.2%}"
        )
        fold_metrics.append(
            {
                "fold": split.fold,
                "train_start": str(split.train_start.date()),
                "train_end": str(split.train_end.date()),
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "mae_simple": metrics["mae_simple"],
                "directional": metrics["directional"],
                "corr": metrics["corr"],
                "coverage_95": metrics["coverage_95"],
                "avg_interval_width": metrics["avg_interval_width"],
                "train_target_clip": train_meta["target_clip"],
            }
        )

    summary_metrics = summarize_fold_metrics(fold_metrics)
    print(
        f"\nSummary | {ticker} | Folds: {summary_metrics['folds']} | "
        f"MAE(log) mean: {summary_metrics['mae_mean']:.6f} | "
        f"MAE(simple) mean: {summary_metrics['mae_simple_mean']:.4%} | "
        f"MSE mean: {summary_metrics['mse_mean']:.6f} | "
        f"Dir mean: {summary_metrics['directional_mean']:.2%} | "
        f"Corr mean: {summary_metrics['corr_mean'] if summary_metrics['corr_mean'] is not None else float('nan'):.4f}"
    )

    final_train_raw = latest_train_window(dataset, config["train_window"], min_train_rows=MIN_TRAIN_ROWS)
    final_train_df = set_time_index(final_train_raw.copy(), final_train_raw.index.min())
    final_train_x, final_train_y, final_sample_weight, final_train_meta = prepare_lgbm_training_data(
        final_train_df,
        feature_cols,
        config.get("training_policy"),
    )
    final_model = train_lgbm(
        final_train_x,
        final_train_y,
        config["lgbm_params"],
        sample_weight=final_sample_weight,
    )
    config_out = dict(config)
    configured_drop_features = resolve_feature_drops(
        feature_set=str(config.get("feature_set", FEATURE_SET_F0)),
        feature_set_file=config.get("feature_set_file"),
    )
    config_out.update(
        {
            "ticker": ticker,
            "sector": sector_name,
            "sector_etf": sector_etf,
            "feature_columns": feature_cols,
            "feature_set": config.get("feature_set", FEATURE_SET_F0),
            "feature_set_drop_features": configured_drop_features,
            "direction_mode": resolve_direction_mode(config.get("direction_mode")),
            "training_policy": resolve_training_policy(config.get("training_policy")),
            "final_target_clip": final_train_meta["target_clip"],
            "final_train_window": {
                "start": str(final_train_df.index.min().date()),
                "end": str(final_train_df.index.max().date()),
                "rows": int(len(final_train_df)),
            },
        }
    )

    artifact_dir = Path(config["artifact_dir"]) / ticker / resolve_artifact_variant()
    save_artifacts(
        artifact_dir=artifact_dir,
        model=final_model,
        fold_metrics=fold_metrics,
        summary_metrics=summary_metrics,
        config=config_out,
        feature_cols=feature_cols,
    )
    return summary_metrics


def main():
    args = parse_args()
    if args.include_time_index:
        args.drop_time_index = False
    parse_window(args.train_window)
    lgbm_params = resolve_lgbm_params(
        preset_name=args.lgbm_param_preset,
        params_json=args.lgbm_params_json,
    )

    tickers = prompt_tickers()
    config = {
        "data_years": DATA_YEARS,
        "window_ret": WINDOW_RET,
        "train_window": args.train_window,
        "test_window": DEFAULT_TEST_WINDOW,
        "step_window": DEFAULT_STEP_WINDOW,
        "artifact_dir": str(ARTIFACT_DIR_DEFAULT),
        "drop_time_index": args.drop_time_index,
        "feature_set": args.feature_set,
        "feature_set_file": args.feature_set_file,
        "lgbm_param_preset": args.lgbm_param_preset,
        "lgbm_params_json": args.lgbm_params_json,
        "lgbm_params": lgbm_params,
        "training_policy": resolve_training_policy(),
        "direction_mode": DEFAULT_DIRECTION_MODE,
        "regime_score": {
            "enabled": True,
            "score_window": REGIME_SCORE_WINDOW,
            "score_clip": REGIME_SCORE_CLIP,
            "weights": REGIME_SCORE_WEIGHTS,
        },
    }

    history_cache = {}
    summaries = {}
    for ticker in tickers:
        summary = train_for_ticker(ticker, config, history_cache)
        summaries[ticker] = summary

    if summaries:
        print("\nFinished training:")
        for ticker, summary in summaries.items():
            print(
                f"  {ticker} | MAE(log) mean: {summary['mae_mean']:.6f} | "
                f"MAE(simple) mean: {summary['mae_simple_mean']:.4%} | "
                f"MSE mean: {summary['mse_mean']:.6f} | "
                f"Dir mean: {summary['directional_mean']:.2%}"
            )


if __name__ == "__main__":
    np.random.seed(42)
    main()
