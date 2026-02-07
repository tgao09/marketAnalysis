import argparse
import json
import math
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

from common import get_history, parse_window, walk_forward_splits


ARTIFACT_DIR_DEFAULT = Path(__file__).resolve().parent / "artifacts"
TICKER_TARGET = "XLK"
TICKER_GOLD = "GLD"
TICKER_SPY = "SPY"
TICKER_VIX = "^VIX"
ANNUALIZATION = 252
WINDOW_VOL = 5
NOISE_WINDOW = 20
DATA_YEARS = 2
DEFAULT_TRAIN_ITERS = 200
DEFAULT_KERNEL_EQUATION = "1*2*4"
KERNEL_BUILDERS = [
    (
        "Matern (nu=1.5)",
        lambda ard_num_dims=None: gpytorch.kernels.MaternKernel(
            nu=1.5, ard_num_dims=ard_num_dims
        ),
    ),
    ("Rational Quadratic", lambda ard_num_dims=None: gpytorch.kernels.RQKernel()),
    ("Periodic", lambda ard_num_dims=None: gpytorch.kernels.PeriodicKernel()),
    (
        "RBF (squared exp)",
        lambda ard_num_dims=None: gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims),
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train GP volatility model.")
    parser.add_argument(
        "--drop-time-index",
        action="store_true",
        help="Exclude time_index from training features.",
    )
    return parser.parse_args()


def choose_option(prompt, options, default_index=0):
    while True:
        print("\n" + prompt)
        for i, opt in enumerate(options, start=1):
            suffix = " (default)" if i - 1 == default_index else ""
            print(f"  {i}) {opt}{suffix}")
        raw = input("Select option number: ").strip()
        if raw == "":
            return default_index
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        print("Invalid choice. Try again.")


def prompt_float(prompt, default_value):
    while True:
        raw = input(f"{prompt} [default {default_value}]: ").strip()
        if raw == "":
            return float(default_value)
        try:
            return float(raw)
        except ValueError:
            print("Invalid number. Try again.")


def tokenize_kernel_equation(expr):
    tokens = []
    number = ""
    for ch in expr:
        if ch.isdigit():
            number += ch
            continue
        if number:
            tokens.append(int(number))
            number = ""
        if ch.isspace():
            continue
        if ch in ("+", "*", "(", ")"):
            tokens.append(ch)
            continue
        raise ValueError(f"Unexpected character '{ch}'.")
    if number:
        tokens.append(int(number))
    return tokens


def parse_kernel_equation(expr, max_index):
    tokens = tokenize_kernel_equation(expr)
    if not tokens:
        raise ValueError("Equation is empty.")

    output = []
    ops = []
    precedence = {"+": 1, "*": 2}

    for tok in tokens:
        if isinstance(tok, int):
            if tok < 1 or tok > max_index:
                raise ValueError(f"Kernel index {tok} out of range 1-{max_index}.")
            output.append(tok)
            continue
        if tok in ("+", "*"):
            while ops and ops[-1] in ("+", "*") and precedence[ops[-1]] >= precedence[tok]:
                output.append(ops.pop())
            ops.append(tok)
            continue
        if tok == "(":
            ops.append(tok)
            continue
        if tok == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if not ops or ops[-1] != "(":
                raise ValueError("Mismatched parentheses.")
            ops.pop()
            continue

    while ops:
        if ops[-1] == "(":
            raise ValueError("Mismatched parentheses.")
        output.append(ops.pop())

    stack = []
    for tok in output:
        if isinstance(tok, int):
            stack.append(tok)
            continue
        if len(stack) < 2:
            raise ValueError("Invalid operator placement.")
        right = stack.pop()
        left = stack.pop()
        stack.append((tok, left, right))

    if len(stack) != 1:
        raise ValueError("Invalid equation.")

    return stack[0]


def build_kernel_from_ast(ast, kernels):
    if isinstance(ast, int):
        return kernels[ast - 1]
    op, left, right = ast
    left_kernel = build_kernel_from_ast(left, kernels)
    right_kernel = build_kernel_from_ast(right, kernels)
    if op == "+":
        return left_kernel + right_kernel
    if op == "*":
        return left_kernel * right_kernel
    raise ValueError(f"Unsupported operator '{op}'.")


def kernel_indices_from_ast(ast, indices=None):
    if indices is None:
        indices = set()
    if isinstance(ast, int):
        indices.add(ast)
        return indices
    _, left, right = ast
    kernel_indices_from_ast(left, indices)
    kernel_indices_from_ast(right, indices)
    return indices


def prompt_kernel_equation(kernel_labels, default_equation):
    print("\nKernel components:")
    for i, label in enumerate(kernel_labels, start=1):
        print(f"  {i}) {label}")
    print("Combine using +, *, and parentheses. Example: (1+2)*3+4")

    while True:
        raw = input(f"Kernel equation [default {default_equation}]: ").strip()
        if raw == "":
            return default_equation
        try:
            parse_kernel_equation(raw, len(kernel_labels))
            return raw
        except ValueError as exc:
            print(f"Invalid kernel equation: {exc}")


def get_config_interactive():
    print("GP Volatility Training (XLK)\n")

    test_window_options = ["1m", "2m"]
    test_window_idx = choose_option("Walk-forward test window:", test_window_options, default_index=0)
    test_window = test_window_options[test_window_idx]

    kernel_options = ["Use default kernel hyperparameters", "Set custom kernel hyperparameters"]
    kernel_idx = choose_option("Kernel hyperparameters:", kernel_options, default_index=0)

    kernel_config = {
        "custom": kernel_idx == 1,
        "lengthscale": None,
        "period_length": None,
        "outputscale": None,
    }

    if kernel_config["custom"]:
        kernel_config["lengthscale"] = prompt_float("Lengthscale", 1.0)
        kernel_config["period_length"] = prompt_float("Period length", 5.0)
        kernel_config["outputscale"] = prompt_float("Output scale", 1.0)

    kernel_config["equation"] = prompt_kernel_equation(
        [label for label, _ in KERNEL_BUILDERS],
        DEFAULT_KERNEL_EQUATION,
    )

    config = {
        "ticker_target": TICKER_TARGET,
        "ticker_gold": TICKER_GOLD,
        "ticker_spy": TICKER_SPY,
        "ticker_vix": TICKER_VIX,
        "data_years": DATA_YEARS,
        "window_vol": WINDOW_VOL,
        "noise_window": NOISE_WINDOW,
        "annualization": ANNUALIZATION,
        "train_window": "2y",
        "test_window": test_window,
        "step_window": test_window,
        "train_iters": DEFAULT_TRAIN_ITERS,
        "artifact_dir": str(ARTIFACT_DIR_DEFAULT),
        "kernel": kernel_config,
    }
    return config


def fetch_data(tickers, start_date, end_date):
    frames = {}
    for symbol in tickers:
        history = get_history(
            symbol,
            period=None,
            start=str(pd.Timestamp(start_date).date()),
            end=str(pd.Timestamp(end_date).date()),
            interval="1d",
            auto_adjust=True,
        )
        frames[symbol] = history

    data = pd.concat(frames, axis=1, sort=False)
    data.columns = data.columns.swaplevel(0, 1)
    data = data.sort_index(axis=1)
    if data.empty:
        raise ValueError("No data returned from yfinance.")
    return data


def extract_field(data, field, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        if field not in data.columns.get_level_values(0):
            raise KeyError(f"Missing field {field} in data.")
        return data[field][ticker].copy()
    if field not in data.columns:
        raise KeyError(f"Missing field {field} in data.")
    return data[field].copy()


def build_features(price_xlk, volume_xlk, price_gld, price_spy, price_vix):
    index = price_xlk.index
    volume_xlk = volume_xlk.reindex(index).ffill().bfill()
    price_gld = price_gld.reindex(index).ffill().bfill()
    price_spy = price_spy.reindex(index).ffill().bfill()
    price_vix = price_vix.reindex(index).ffill().bfill()

    returns_xlk = price_xlk.pct_change()
    returns_gld = price_gld.pct_change()
    returns_spy = price_spy.pct_change()

    features = pd.DataFrame(index=price_xlk.index)
    features["time_index"] = (features.index - features.index[0]).days.astype(int)

    features["ret_1d"] = returns_xlk
    features["ret_5d"] = price_xlk.pct_change(5)
    features["vol_5d"] = returns_xlk.rolling(WINDOW_VOL).std()
    features["mean_ret_5d"] = returns_xlk.rolling(WINDOW_VOL).mean()
    features["mean_abs_ret_5d"] = returns_xlk.abs().rolling(WINDOW_VOL).mean()
    features["vol_chg_1d"] = volume_xlk.pct_change()

    features["gld_vol_5d"] = returns_gld.rolling(WINDOW_VOL).std()
    features["spy_vol_5d"] = returns_spy.rolling(WINDOW_VOL).std()
    features["vix_level"] = price_vix

    return features


def build_target(price_xlk):
    returns = price_xlk.pct_change()
    forward_vol = returns.rolling(WINDOW_VOL).std().shift(-WINDOW_VOL)
    forward_vol = forward_vol * math.sqrt(ANNUALIZATION)
    forward_vol = forward_vol.replace(0.0, np.nan)
    target = np.log(forward_vol)

    noise = target.rolling(NOISE_WINDOW).std()
    noise = noise.pow(2)
    return target.rename("target"), noise.rename("noise")


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


class VolGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_config):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        ard_num_dims = train_x.shape[-1]
        kernels = []
        for idx, (_, builder) in enumerate(KERNEL_BUILDERS, start=1):
            kernel = builder(ard_num_dims=ard_num_dims)
            kernel._kernel_index = idx
            kernels.append(kernel)

        if kernel_config.get("custom"):
            lengthscale = kernel_config.get("lengthscale")
            period_length = kernel_config.get("period_length")
            outputscale = kernel_config.get("outputscale")

            if lengthscale is not None:
                for kernel in kernels:
                    if hasattr(kernel, "lengthscale"):
                        kernel.lengthscale = lengthscale
            if period_length is not None:
                for kernel in kernels:
                    if hasattr(kernel, "period_length"):
                        kernel.period_length = period_length

        equation = kernel_config.get("equation") or DEFAULT_KERNEL_EQUATION
        kernel_ast = parse_kernel_equation(equation, len(kernels))
        self.used_kernel_indices = kernel_indices_from_ast(kernel_ast)
        base_kernel = build_kernel_from_ast(kernel_ast, kernels)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        if kernel_config.get("custom") and kernel_config.get("outputscale") is not None:
            self.covar_module.outputscale = kernel_config["outputscale"]

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(
    train_x,
    train_y,
    train_noise,
    kernel_config,
    train_iters=DEFAULT_TRAIN_ITERS,
):
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train_noise,
        learn_additional_noise=True,
    )
    model = VolGPModel(train_x, train_y, likelihood, kernel_config)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
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
    if isinstance(kernel, gpytorch.kernels.RBFKernel):
        return "RBF"
    if isinstance(kernel, gpytorch.kernels.RQKernel):
        return "RationalQuadratic"
    if isinstance(kernel, gpytorch.kernels.PeriodicKernel):
        return "Periodic"
    return kernel.__class__.__name__


def print_ard_importance(model, feature_cols):
    num_features = len(feature_cols)
    base_kernel = model.covar_module
    kernels = list(iter_base_kernels(base_kernel))
    used_indices = getattr(model, "used_kernel_indices", None)

    results = []
    for kernel in kernels:
        kernel_index = getattr(kernel, "_kernel_index", None)
        if used_indices is not None and kernel_index is not None:
            if kernel_index not in used_indices:
                continue
        if not hasattr(kernel, "lengthscale"):
            continue
        lengthscale = kernel.lengthscale.detach().cpu().numpy().reshape(-1)
        if lengthscale.size != num_features:
            continue
        lengthscale = np.clip(lengthscale, 1e-8, None)
        importance = 1.0 / lengthscale
        results.append((kernel_display_name(kernel), lengthscale, importance))

    if not results:
        print("\nARD feature importance: no ARD lengthscales found in kernel.")
        return

    combined = []
    for name, lengthscale, importance in results:
        order = np.argsort(-importance)
        print(f"\nARD feature importance ({name}) - higher = more important (1/lengthscale):")
        for rank, idx in enumerate(order, start=1):
            print(
                f"  {rank}. {feature_cols[idx]}: {importance[idx]:.6f} "
                f"(lengthscale={lengthscale[idx]:.6f})"
            )
        combined.append(importance / importance.sum())

    if len(combined) > 1:
        combined_mean = np.mean(np.vstack(combined), axis=0)
        order = np.argsort(-combined_mean)
        print("\nCombined ARD feature importance (mean normalized across ARD kernels):")
        for rank, idx in enumerate(order, start=1):
            print(f"  {rank}. {feature_cols[idx]}: {combined_mean[idx]:.6f}")


def evaluate(model, likelihood, test_x, test_y, test_noise):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(test_x), noise=test_noise)
        mean = preds.mean
        std = preds.variance.sqrt()
        mae = torch.mean(torch.abs(mean - test_y)).item()
        mse = torch.mean((mean - test_y) ** 2).item()
        lower = mean - (1.96 * std)
        upper = mean + (1.96 * std)
        coverage_95 = torch.mean(((test_y >= lower) & (test_y <= upper)).float()).item()
        avg_interval_width = torch.mean((upper - lower)).item()
    return {
        "mae": mae,
        "mse": mse,
        "coverage_95": coverage_95,
        "avg_interval_width": avg_interval_width,
    }


def summarize_fold_metrics(fold_metrics):
    mae_values = [fold["mae"] for fold in fold_metrics]
    mse_values = [fold["mse"] for fold in fold_metrics]
    coverage_values = [fold["coverage_95"] for fold in fold_metrics]
    width_values = [fold["avg_interval_width"] for fold in fold_metrics]
    summary = {
        "folds": len(fold_metrics),
        "mae_mean": float(np.mean(mae_values)) if mae_values else None,
        "mae_median": float(np.median(mae_values)) if mae_values else None,
        "mse_mean": float(np.mean(mse_values)) if mse_values else None,
        "mse_median": float(np.median(mse_values)) if mse_values else None,
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
    scaler,
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

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "feature_columns": feature_cols,
        },
        model_path,
    )

    scaler_out = {"mean": scaler["mean"], "std": scaler["std"]}
    scaler_path.write_text(json.dumps(scaler_out, indent=2))

    metrics_out = {
        "summary": summary_metrics,
        "folds": fold_metrics,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    metrics_path.write_text(json.dumps(metrics_out, indent=2))

    config_path.write_text(json.dumps(config, indent=2))

    print(f"\nArtifacts saved to: {artifact_dir}")


def main():
    args = parse_args()
    config = get_config_interactive()
    config["drop_time_index"] = args.drop_time_index

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=config["data_years"])
    train_offset = parse_window(config["train_window"])
    test_offset = parse_window(config["test_window"])
    buffer_days = NOISE_WINDOW + (2 * WINDOW_VOL) + 5
    min_start = end_date - train_offset
    min_start = min_start - test_offset
    min_start = min_start - pd.DateOffset(days=buffer_days)
    if min_start < start_date:
        start_date = min_start
        print(f"Extended lookback for walk-forward windows: start={start_date.date()}")

    print("\nDownloading data...")
    data = fetch_data([TICKER_TARGET, TICKER_GOLD, TICKER_SPY, TICKER_VIX], start_date, end_date)

    price_xlk = extract_field(data, "Close", TICKER_TARGET)
    volume_xlk = extract_field(data, "Volume", TICKER_TARGET)
    price_gld = extract_field(data, "Close", TICKER_GOLD)
    price_spy = extract_field(data, "Close", TICKER_SPY)
    price_vix = extract_field(data, "Close", TICKER_VIX)

    features = build_features(price_xlk, volume_xlk, price_gld, price_spy, price_vix)
    target, noise = build_target(price_xlk)

    dataset = features.join([target, noise]).dropna()
    if dataset.empty:
        raise ValueError("No rows left after feature/target alignment.")

    min_required_end = dataset.index.min() + train_offset
    min_required_end = min_required_end + test_offset
    if dataset.index.max() <= min_required_end:
        raise ValueError(
            "Not enough usable data for the requested walk-forward windows after feature/target alignment. "
            "Try a shorter train/test window or increase the lookback."
        )

    feature_cols = [col for col in dataset.columns if col not in ("target", "noise")]
    if config.get("drop_time_index"):
        feature_cols = [col for col in feature_cols if col != "time_index"]

    fold_metrics = []
    last_model = None
    last_likelihood = None
    last_scaler = None
    # Target uses forward WINDOW_VOL days, so embargo must match to prevent leakage.
    horizon_embargo = WINDOW_VOL
    if horizon_embargo < 0:
        raise ValueError("WINDOW_VOL must be non-negative for walk-forward embargo.")

    splits = walk_forward_splits(
        dataset,
        train_window=config["train_window"],
        test_window=config["test_window"],
        embargo=horizon_embargo,
        step=config["step_window"],
        min_train_rows=60,
    )

    print("\nTraining GP model (walk-forward)...")
    for split in splits:
        train_df = split.train.copy()
        test_df = split.test.copy()

        # Reset time index per fold to avoid absolute calendar leakage.
        fold_start = split.train_start
        train_df["time_index"] = (train_df.index - fold_start).days.astype(int)
        test_df["time_index"] = (test_df.index - fold_start).days.astype(int)

        train_x_df, test_x_df, scaler = normalize_features(train_df, test_df, feature_cols)

        train_x = torch.tensor(train_x_df.values, dtype=torch.float32)
        train_y = torch.tensor(train_df["target"].values, dtype=torch.float32)
        train_noise = torch.tensor(train_df["noise"].values, dtype=torch.float32).clamp_min(1e-8)
        test_x = torch.tensor(test_x_df.values, dtype=torch.float32)
        test_y = torch.tensor(test_df["target"].values, dtype=torch.float32)
        test_noise = torch.tensor(test_df["noise"].values, dtype=torch.float32).clamp_min(1e-8)

        print(f"\nFold {split.fold} | Train: {split.train_start.date()} -> {split.train_end.date()} | "
              f"Test: {split.test_start.date()} -> {split.test_end.date()}")

        model, likelihood = train_gp(
            train_x,
            train_y,
            train_noise,
            config["kernel"],
            config["train_iters"],
        )

        metrics = evaluate(model, likelihood, test_x, test_y, test_noise)
        print(
            f"MAE: {metrics['mae']:.6f} | MSE: {metrics['mse']:.6f} | "
            f"Coverage95: {metrics['coverage_95']:.2%}"
        )
        print_ard_importance(model, feature_cols)

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
                "coverage_95": metrics["coverage_95"],
                "avg_interval_width": metrics["avg_interval_width"],
            }
        )

        last_model = model
        last_likelihood = likelihood
        last_scaler = scaler

    if not fold_metrics:
        raise ValueError("No walk-forward splits produced; check date range and windows.")

    summary_metrics = summarize_fold_metrics(fold_metrics)
    print(
        f"\nSummary | Folds: {summary_metrics['folds']} | "
        f"MAE mean: {summary_metrics['mae_mean']:.6f} | "
        f"MSE mean: {summary_metrics['mse_mean']:.6f} | "
        f"Coverage95 mean: {summary_metrics['coverage_95_mean']:.2%}"
    )

    save_artifacts(
        Path(config["artifact_dir"]),
        last_model,
        last_likelihood,
        last_scaler,
        fold_metrics,
        summary_metrics,
        config,
        feature_cols,
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
