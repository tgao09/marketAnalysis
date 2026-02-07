import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gpytorch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common import (
    DEFAULT_SECTOR_ETF_MAP,
    PCATransformer,
    get_history,
    get_info,
    parse_window,
    save_pca_json,
    walk_forward_splits,
)


ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "artifacts"
TICKER_GOLD = "GLD"
TICKER_SPY = "SPY"
TICKER_VIX = "^VIX"
WINDOW_RET = 5
DATA_YEARS = 3
DEFAULT_TRAIN_ITERS = 400
DEFAULT_TRAIN_WINDOW = "2y"
DEFAULT_TEST_WINDOW = "1m"
DEFAULT_STEP_WINDOW = "1m"
FEATURE_LOOKBACK_MAX = 60
REGIME_SCORE_WINDOW = 252
REGIME_SCORE_CLIP = 4.0
REGIME_SCORE_WEIGHTS = {"vix": 0.5, "spy_vol": 0.5}
PCA_VAR_THRESHOLD = 0.80
PCA_MAX_PCS = 12
PCA_IMPUTE_STRATEGY = "median"
PCA_MODE = "replace"
PCA_PC_PREFIX = "pc_"
ARTIFACT_VARIANT_REGULAR = "regular"
ARTIFACT_VARIANT_PCA = "pca"
PCA_ANALYSIS_EXTRA_FEATURES = [
    "ret_20d",
    "vol_chg_1d",
    "momentum_5_20",
    "sector_ret_1d",
    "gld_ret_1d",
    "vix_level",
    "corr_spy_60d",
]


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GP return model.")
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
        "--pca",
        action="store_true",
        help="Enable PCA preprocessing pipeline (saved under ticker/pca artifacts).",
    )
    parser.set_defaults(drop_time_index=True)
    return parser.parse_args()


def resolve_artifact_variant(pca_enabled: bool) -> str:
    return ARTIFACT_VARIANT_PCA if pca_enabled else ARTIFACT_VARIANT_REGULAR


def build_pca_transformer() -> PCATransformer:
    return PCATransformer(
        threshold=PCA_VAR_THRESHOLD,
        max_pcs=PCA_MAX_PCS,
        impute_strategy=PCA_IMPUTE_STRATEGY,
        mode=PCA_MODE,
        pc_prefix=PCA_PC_PREFIX,
    )


def prompt_tickers():
    raw = input("Enter tickers (comma-separated): ").strip()
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


def resolve_sector_etf(ticker):
    sector = None
    etf = None
    error = None
    try:
        info = get_info(ticker)
        sector = info.get("sector") or info.get("sectorKey")
        if sector:
            sector_key = sector.strip()
            if sector_key not in DEFAULT_SECTOR_ETF_MAP:
                title_key = sector_key.title()
                if title_key in DEFAULT_SECTOR_ETF_MAP:
                    sector_key = title_key
            if sector_key in DEFAULT_SECTOR_ETF_MAP:
                etf = DEFAULT_SECTOR_ETF_MAP[sector_key]
                sector = sector_key
            else:
                error = f"Sector '{sector_key}' not in DEFAULT_SECTOR_ETF_MAP."
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
        end_exclusive = None
        if end_date is not None:
            # yfinance treats `end` as exclusive; bump by one day so the
            # caller's inclusive end_date is respected.
            end_exclusive = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
        history = get_history(
            symbol,
            period=None,
            start=str(pd.Timestamp(start_date).date()),
            end=str(end_exclusive.date()) if end_exclusive is not None else None,
            interval="1d",
            auto_adjust=True,
        )
        if isinstance(history.index, pd.DatetimeIndex):
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


def build_features(price_stock, volume_stock, price_sector, price_gld, price_spy, price_vix):
    index = price_stock.index

    # Forward-fill only to avoid leaking future information via backfill.
    volume_stock = volume_stock.reindex(index).ffill()
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

    # Stock returns (log) and summary stats
    features["ret_1d"] = log_ret_stock
    features["ret_5d"] = compute_log_return(price_stock, WINDOW_RET)
    features["ret_10d"] = compute_log_return(price_stock, 10)
    ret_20d = compute_log_return(price_stock, 20)
    features["ret_20d"] = ret_20d
    features["ret_60d"] = compute_log_return(price_stock, 60)

    # Stock volatility and standardized returns
    features["vol_5d"] = log_ret_stock.rolling(WINDOW_RET).std()
    features["vol_10d"] = log_ret_stock.rolling(10).std()
    features["vol_20d"] = log_ret_stock.rolling(20).std()
    features["vol_60d"] = log_ret_stock.rolling(60).std()

    # Volume and distribution shape
    features["vol_chg_1d"] = volume_stock.pct_change()
    features["skew_20d"] = log_ret_stock.rolling(20).skew()

    # Trend, range, and volatility structure
    features["stock_ma20_gap"] = (price_stock / price_stock.rolling(20).mean()) - 1.0
    features["stock_ma60_gap"] = (price_stock / price_stock.rolling(60).mean()) - 1.0
    roll_min_60 = price_stock.rolling(60).min()
    roll_max_60 = price_stock.rolling(60).max()
    features["drawdown_60d"] = (price_stock / roll_max_60) - 1.0
    features["momentum_5_20"] = features["ret_5d"] - ret_20d
    features["vol_ratio_5_20"] = (features["vol_5d"] / features["vol_20d"]).replace(
        [np.inf, -np.inf], np.nan
    )

    # Sector ETF features
    features["sector_ret_1d"] = log_ret_sector
    features["sector_ret_5d"] = compute_log_return(price_sector, WINDOW_RET)
    features["sector_vol_5d"] = log_ret_sector.rolling(WINDOW_RET).std()
    features["rel_strength_sector_20d"] = ret_20d - compute_log_return(price_sector, 20)

    # GLD features
    features["gld_ret_1d"] = log_ret_gld
    features["gld_ret_5d"] = compute_log_return(price_gld, WINDOW_RET)
    features["gld_vol_5d"] = log_ret_gld.rolling(WINDOW_RET).std()

    # Market regime (SPY + VIX)
    features["spy_ret_5d"] = compute_log_return(price_spy, WINDOW_RET)
    features["spy_vol_20d"] = log_ret_spy.rolling(20).std()
    features["spy_ma20_gap"] = (price_spy / price_spy.rolling(20).mean()) - 1.0
    features["vix_level"] = price_vix
    features["vix_chg_1d"] = compute_log_return(price_vix, 1)
    features["corr_spy_60d"] = log_ret_stock.rolling(60).corr(log_ret_spy)

    # Calendar features (cyclical encoding within quarter)
    day_in_quarter, quarter_len = trading_day_in_quarter(features.index)
    quarter_len = quarter_len.replace(0, 1)
    phase = (2.0 * np.pi * day_in_quarter) / quarter_len
    features["q_phase_sin"] = np.sin(phase)
    features["q_phase_cos"] = np.cos(phase)

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
    return {"count": int(non_na.shape[0]), "start": str(non_na.index.min().date()), "end": str(non_na.index.max().date())}


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


def normalize_features(train_df, test_df, feature_cols):
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0.0, 1.0)

    train_x = (train_df[feature_cols] - mean) / std
    test_x = (test_df[feature_cols] - mean) / std

    scaler = {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }
    return train_x, test_x, scaler


def select_feature_columns(dataset: pd.DataFrame, drop_time_index: bool, pca_enabled: bool) -> list[str]:
    feature_cols = [col for col in dataset.columns if col != "target"]
    if drop_time_index:
        feature_cols = [col for col in feature_cols if col != "time_index"]
    if not pca_enabled:
        feature_cols = [col for col in feature_cols if col not in PCA_ANALYSIS_EXTRA_FEATURES]
    return feature_cols


def latest_train_window(data, train_window, min_train_rows=60):
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


class ReturnGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        ard_num_dims = train_x.shape[-1]
        matern = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=ard_num_dims)
        rq = gpytorch.kernels.RQKernel(ard_num_dims=ard_num_dims)
        linear = gpytorch.kernels.LinearKernel(ard_num_dims=ard_num_dims)
        self.covar_module = (
            gpytorch.kernels.ScaleKernel(matern)
            + gpytorch.kernels.ScaleKernel(rq)
            + gpytorch.kernels.ScaleKernel(linear)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, train_iters=DEFAULT_TRAIN_ITERS, device=None):
    if device is None:
        device = train_x.device
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ReturnGPModel(train_x, train_y, likelihood).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(1, train_iters + 1):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if i == 1 or i % 50 == 0 or i == train_iters:
            print(f"Iter {i}/{train_iters} - Loss: {loss.item():.4f}")

    return model, likelihood


def iter_base_kernels(kernel):
    if hasattr(kernel, "base_kernel"):
        yield from iter_base_kernels(kernel.base_kernel)
        return
    if hasattr(kernel, "kernels"):
        for sub_kernel in kernel.kernels:
            yield from iter_base_kernels(sub_kernel)
        return
    yield kernel


def kernel_display_name(kernel):
    if isinstance(kernel, gpytorch.kernels.MaternKernel):
        return f"Matern(nu={kernel.nu})"
    if isinstance(kernel, gpytorch.kernels.LinearKernel):
        return "Linear"
    return kernel.__class__.__name__


def collect_ard_importance(model, feature_cols):
    num_features = len(feature_cols)
    kernels = list(iter_base_kernels(model.covar_module))
    results = []

    for kernel in kernels:
        if not hasattr(kernel, "lengthscale"):
            continue
        lengthscale = kernel.lengthscale
        if lengthscale is None:
            continue
        lengthscale = lengthscale.detach().cpu().numpy().reshape(-1)
        if lengthscale.size != num_features:
            continue
        lengthscale = np.clip(lengthscale, 1e-8, None)
        importance = 1.0 / lengthscale
        results.append(
            {
                "name": kernel_display_name(kernel),
                "lengthscale": lengthscale,
                "importance": importance,
            }
        )

    return results


def print_ard_importance(results, feature_cols):
    if not results:
        print("\nARD feature importance: no ARD lengthscales found in kernel.")
        return

    for result in results:
        name = result["name"]
        lengthscale = result["lengthscale"]
        importance = result["importance"]
        order = np.argsort(-importance)
        print(f"\nARD feature importance ({name}) - higher = more important (1/lengthscale):")
        for rank, idx in enumerate(order, start=1):
            print(
                f"  {rank}. {feature_cols[idx]}: {importance[idx]:.6f} "
                f"(lengthscale={lengthscale[idx]:.6f})"
            )


def bottom_feature_sets(results, feature_cols, bottom_n=10):
    feature_sets = []
    if not results:
        return feature_sets
    num_features = len(feature_cols)
    take_n = min(bottom_n, num_features)
    for result in results:
        importance = result["importance"]
        order = np.argsort(importance)
        bottom_idx = order[:take_n]
        feature_sets.append({feature_cols[idx] for idx in bottom_idx})
    return feature_sets


def evaluate(model, likelihood, test_x, test_y):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(test_x))
        mean = preds.mean
        std = preds.variance.sqrt()
        mae = torch.mean(torch.abs(mean - test_y)).item()
        mse = torch.mean((mean - test_y) ** 2).item()
        mean_simple = torch.exp(mean) - 1.0
        actual_simple = torch.exp(test_y) - 1.0
        mae_simple = torch.mean(torch.abs(mean_simple - actual_simple)).item()
        directional = torch.mean((torch.sign(mean) == torch.sign(test_y)).float()).item()
        lower = mean - (1.96 * std)
        upper = mean + (1.96 * std)
        coverage_95 = torch.mean(((test_y >= lower) & (test_y <= upper)).float()).item()
        avg_interval_width = torch.mean((upper - lower)).item()
    return {
        "mae": mae,
        "mse": mse,
        "mae_simple": mae_simple,
        "directional": directional,
        "coverage_95": coverage_95,
        "avg_interval_width": avg_interval_width,
    }


def summarize_fold_metrics(fold_metrics):
    mae_values = [fold["mae"] for fold in fold_metrics]
    mse_values = [fold["mse"] for fold in fold_metrics]
    mae_simple_values = [fold["mae_simple"] for fold in fold_metrics]
    dir_values = [fold["directional"] for fold in fold_metrics]
    coverage_values = [fold["coverage_95"] for fold in fold_metrics]
    width_values = [fold["avg_interval_width"] for fold in fold_metrics]
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
        "coverage_95_mean": float(np.mean(coverage_values)) if coverage_values else None,
        "coverage_95_median": float(np.median(coverage_values)) if coverage_values else None,
        "avg_interval_width_mean": float(np.mean(width_values)) if width_values else None,
        "avg_interval_width_median": float(np.median(width_values)) if width_values else None,
    }
    return summary


def save_artifacts(
    artifact_dir,
    model,
    likelihood,
    fold_metrics,
    summary_metrics,
    config,
    feature_cols,
    raw_feature_cols,
    scaler=None,
):
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model_state.pt"
    scaler_path = artifact_dir / "scaler.json"
    metrics_path = artifact_dir / "metrics.json"
    config_path = artifact_dir / "config.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "feature_columns": feature_cols,
            "raw_feature_columns": raw_feature_cols,
        },
        model_path,
    )

    if scaler is not None:
        scaler_out = {"mean": scaler["mean"], "std": scaler["std"]}
        scaler_path.write_text(json.dumps(scaler_out, indent=2))
    elif scaler_path.exists():
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


def train_for_ticker(ticker, config, history_cache, device):
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
    volume_stock = extract_field(stock_history, "Volume", ticker)
    price_sector = extract_field(sector_history, "Close", sector_etf)
    price_gld = extract_field(gld_history, "Close", TICKER_GOLD)
    price_spy = extract_field(spy_history, "Close", TICKER_SPY)
    price_vix = extract_field(vix_history, "Close", TICKER_VIX)

    features = build_features(price_stock, volume_stock, price_sector, price_gld, price_spy, price_vix)
    target = build_target(price_stock)

    regime_config = config["regime_score"]
    # Keep regime inputs causal (no backward fill from future observations).
    price_spy_regime = price_spy.reindex(price_stock.index).ffill()
    price_vix_regime = price_vix.reindex(price_stock.index).ffill()

    if regime_config.get("enabled", True):
        regime_score = compute_regime_score(
            price_vix_regime,
            price_spy_regime,
            regime_config["score_window"],
            regime_config["score_clip"],
            regime_config["weights"],
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
            f"Note the forward target uses {WINDOW_RET}d, which trims the last {WINDOW_RET} days. "
            "Try a shorter test window or wait for more recent data."
        )

    pca_enabled = bool(config.get("pca", {}).get("enabled", False))
    feature_cols = select_feature_columns(
        dataset=dataset,
        drop_time_index=bool(config.get("drop_time_index", True)),
        pca_enabled=pca_enabled,
    )
    artifact_variant = resolve_artifact_variant(pca_enabled)

    splits = list(
        walk_forward_splits(
            dataset,
            train_window=config["train_window"],
            test_window=config["test_window"],
            embargo=config["window_ret"],
            step=config["step_window"],
            min_train_rows=60,
        )
    )
    if not splits:
        raise ValueError(f"{ticker}: No walk-forward splits produced.")

    fold_metrics = []
    ard_bottom_sets = []
    importance_feature_cols_ref = None
    importance_feature_cols_unstable = False

    print(f"\nTraining {ticker} (sector ETF: {sector_etf}, variant: {artifact_variant})")
    for split in splits:
        train_df = set_time_index(split.train.copy(), split.train_start)
        test_df = set_time_index(split.test.copy(), split.train_start)

        if pca_enabled:
            fold_pca = build_pca_transformer()
            train_x_df, test_x_df = fold_pca.transform_train_test(train_df, test_df, feature_cols)
            model_feature_cols = train_x_df.columns.tolist()
            fold_pca_k = int(fold_pca.k_selected_ or 0)
        else:
            train_x_df, test_x_df, _ = normalize_features(train_df, test_df, feature_cols)
            model_feature_cols = list(feature_cols)
            fold_pca_k = None

        if importance_feature_cols_ref is None:
            importance_feature_cols_ref = list(model_feature_cols)
        elif model_feature_cols != importance_feature_cols_ref:
            importance_feature_cols_unstable = True

        train_x = torch.tensor(train_x_df.values, dtype=torch.float32, device=device)
        train_y = torch.tensor(train_df["target"].values, dtype=torch.float32, device=device)
        test_x = torch.tensor(test_x_df.values, dtype=torch.float32, device=device)
        test_y = torch.tensor(test_df["target"].values, dtype=torch.float32, device=device)

        print(
            f"\nFold {split.fold} | Train: {split.train_start.date()} -> {split.train_end.date()} | "
            f"Test: {split.test_start.date()} -> {split.test_end.date()}"
        )

        model, likelihood = train_gp(
            train_x,
            train_y,
            train_iters=config["train_iters"],
            device=device,
        )

        metrics = evaluate(model, likelihood, test_x, test_y)
        print(
            f"MAE(log): {metrics['mae']:.6f} | MAE(simple): {metrics['mae_simple']:.4%} | "
            f"MSE: {metrics['mse']:.6f} | "
            f"Dir: {metrics['directional']:.2%} | Coverage95: {metrics['coverage_95']:.2%}"
        )
        ard_results = collect_ard_importance(model, model_feature_cols)
        print_ard_importance(ard_results, model_feature_cols)
        ard_bottom_sets.extend(bottom_feature_sets(ard_results, model_feature_cols, bottom_n=10))

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
                "coverage_95": metrics["coverage_95"],
                "avg_interval_width": metrics["avg_interval_width"],
                "pca_k": fold_pca_k,
            }
        )

    summary_metrics = summarize_fold_metrics(fold_metrics)
    if ard_bottom_sets and not importance_feature_cols_unstable:
        shared_bottom = set.intersection(*ard_bottom_sets)
        summary_metrics["low_importance_features"] = [
            col for col in (importance_feature_cols_ref or []) if col in shared_bottom
        ]
    elif ard_bottom_sets and importance_feature_cols_unstable:
        summary_metrics["low_importance_features"] = []
        summary_metrics["low_importance_note"] = (
            "Skipped shared-bottom ARD summary because PCA fold component counts varied."
        )
    else:
        summary_metrics["low_importance_features"] = []
    print(
        f"\nSummary | {ticker} | Folds: {summary_metrics['folds']} | "
        f"MAE(log) mean: {summary_metrics['mae_mean']:.6f} | "
        f"MAE(simple) mean: {summary_metrics['mae_simple_mean']:.4%} | "
        f"MSE mean: {summary_metrics['mse_mean']:.6f} | "
        f"Dir mean: {summary_metrics['directional_mean']:.2%}"
    )

    final_train_raw = latest_train_window(
        dataset,
        train_window=config["train_window"],
        min_train_rows=60,
    )
    final_fold_start = final_train_raw.index.min()
    final_train_df = set_time_index(final_train_raw.copy(), final_fold_start)
    final_pca_transformer = None
    if pca_enabled:
        final_pca_transformer = build_pca_transformer().fit(final_train_df, feature_cols)
        final_train_x_df = final_pca_transformer.transform(final_train_df)
        final_scaler = None
        final_model_feature_cols = final_train_x_df.columns.tolist()
    else:
        final_train_x_df, _, final_scaler = normalize_features(
            final_train_df,
            final_train_df,
            feature_cols,
        )
        final_model_feature_cols = list(feature_cols)
    final_train_x = torch.tensor(final_train_x_df.values, dtype=torch.float32, device=device)
    final_train_y = torch.tensor(final_train_df["target"].values, dtype=torch.float32, device=device)

    print(
        f"\nFinal model fit | Train: {final_train_df.index.min().date()} -> "
        f"{final_train_df.index.max().date()}"
    )
    final_model, final_likelihood = train_gp(
        final_train_x,
        final_train_y,
        train_iters=config["train_iters"],
        device=device,
    )

    artifact_dir = Path(config["artifact_dir"]) / ticker / artifact_variant
    config_out = dict(config)
    config_out.update(
        {
            "ticker": ticker,
            "sector": sector_name,
            "sector_etf": sector_etf,
            "artifact_variant": artifact_variant,
            "kernel": {
                "matern_nu": 0.5,
                "matern_ard": True,
                "rational_quadratic_ard": True,
            },
            "noise_model": "gaussian",
            "final_train_window": {
                "start": str(final_train_df.index.min().date()),
                "end": str(final_train_df.index.max().date()),
                "rows": int(len(final_train_df)),
            },
        }
    )

    save_artifacts(
        artifact_dir,
        final_model,
        final_likelihood,
        fold_metrics,
        summary_metrics,
        config_out,
        final_model_feature_cols,
        feature_cols,
        scaler=final_scaler,
    )
    pca_path = artifact_dir / "pca.json"
    if pca_enabled:
        if final_pca_transformer is None:
            raise RuntimeError("Expected final_pca_transformer for PCA-enabled training.")
        save_pca_json(pca_path, final_pca_transformer)
    elif pca_path.exists():
        pca_path.unlink()

    return summary_metrics


def main():
    args = parse_args()
    if args.include_time_index:
        args.drop_time_index = False
    tickers = prompt_tickers()
    if not tickers:
        print("No tickers provided. Exiting.")
        return
    device = resolve_device()
    print(f"Using device: {device.type}")

    config = {
        "data_years": DATA_YEARS,
        "window_ret": WINDOW_RET,
        "train_window": DEFAULT_TRAIN_WINDOW,
        "test_window": DEFAULT_TEST_WINDOW,
        "step_window": DEFAULT_STEP_WINDOW,
        "train_iters": DEFAULT_TRAIN_ITERS,
        "artifact_dir": str(ARTIFACT_DIR_DEFAULT),
        "drop_time_index": args.drop_time_index,
        "pca": {
            "enabled": args.pca,
            "threshold": PCA_VAR_THRESHOLD,
            "max_pcs": PCA_MAX_PCS,
            "impute_strategy": PCA_IMPUTE_STRATEGY,
            "mode": PCA_MODE,
            "pc_prefix": PCA_PC_PREFIX,
        },
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
        try:
            summary = train_for_ticker(ticker, config, history_cache, device)
            summaries[ticker] = summary
        except Exception as exc:
            print(f"{ticker}: training failed - {exc}")

    if summaries:
        print("\nBottom-10 features across all folds and ARD kernels:")
        for ticker, summary in summaries.items():
            low_features = summary.get("low_importance_features", [])
            if low_features:
                print(f"  {ticker}: {', '.join(low_features)}")
            else:
                print(f"  {ticker}: none")
        print("\nFinished training:")
        for ticker, summary in summaries.items():
            print(
                f"  {ticker} | MAE(log) mean: {summary['mae_mean']:.6f} | "
                f"MAE(simple) mean: {summary['mae_simple_mean']:.4%} | "
                f"MSE mean: {summary['mse_mean']:.6f} | "
                f"Dir mean: {summary['directional_mean']:.2%}"
            )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
