"""Validation-only neighborhood and seed stability checks for selected PPO config."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from common.backtesting import PortfolioReplay, PortfolioWalkForward, PortfolioWalkForwardConfig, portfolio_metrics
from rl_portfolio_management.data_pipeline import load_snapshot
from rl_portfolio_management.optimize_ppo import _clean, robust_validation_score
from rl_portfolio_management.ppo import PPOConfig, SB3PortfolioPolicy, train_ppo
from rl_portfolio_management.rl_env import RewardConfig


def classify_isolated_spike(best_score: float, nearby_scores: Sequence[float]) -> bool:
    """Flag unsupported optimum: statistical spike or majority invalid/negative neighbors."""
    values = np.asarray(nearby_scores, dtype=float)
    usable = values[np.isfinite(values) & (values > -1_000)]
    mostly_bad = len(values) > 0 and np.count_nonzero(
        ~np.isfinite(values) | (values <= 0) | (values <= -1_000)
    ) > len(values) / 2
    if not len(usable):
        return True
    median = float(np.median(usable))
    std = float(np.std(usable, ddof=1)) if len(usable) > 1 else 0.0
    statistical_spike = best_score > median and best_score - median > 2 * std
    return bool(statistical_spike or mostly_bad)


def _config(params: Mapping[str, object], timesteps: int, seed: int) -> PPOConfig:
    width = int(params.get("net_width", 128))
    return PPOConfig(
        seed=seed, timesteps=timesteps, net_arch=(width, width),
        learning_rate=float(params.get("learning_rate", 3e-4)),
        gamma=float(params.get("gamma", 0.99)),
        gae_lambda=float(params.get("gae_lambda", 0.95)),
        ent_coef=float(params.get("ent_coef", 0.0)),
        n_steps=int(params.get("n_steps", 128)),
        action_cadence=int(params.get("action_cadence", 1)),
        gross_budget=float(params.get("gross_budget", 0.8)),
        reward=RewardConfig(
            drawdown=float(params.get("reward_drawdown", 0.05)),
            turnover=float(params.get("reward_turnover", 0.001)),
        ),
    )


def _variants(best: PPOConfig) -> list[tuple[str, PPOConfig]]:
    return [
        ("best", best),
        ("learning_rate_x0.75", replace(best, learning_rate=best.learning_rate * 0.75)),
        ("learning_rate_x1.25", replace(best, learning_rate=best.learning_rate * 1.25)),
        ("gross_budget_minus0.1", replace(best, gross_budget=max(0.0, best.gross_budget - 0.1))),
        ("gross_budget_plus0.1", replace(best, gross_budget=min(1.0, best.gross_budget + 0.1))),
        ("reward_turnover_x0.5", replace(best, reward=replace(best.reward, turnover=best.reward.turnover * 0.5))),
        ("reward_turnover_x2", replace(best, reward=replace(best.reward, turnover=best.reward.turnover * 2))),
        ("best_seed_29", replace(best, seed=29)),
        ("best_seed_43", replace(best, seed=43)),
    ]


def run_stability(manifest_path: str | Path, best_path: str | Path, output: str | Path,
                  *, selection_folds: int = 2, timesteps: int = 1_500,
                  seeds_only: bool = False) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    best_record = json.loads(Path(best_path).read_text(encoding="utf-8"))
    best = _config(best_record["params"], timesteps, seed=17)
    frames, manifest = load_snapshot(manifest_path, verify=True)
    symbols = tuple(manifest["config"]["symbols"])
    panel = {symbol: frames[symbol] for symbol in symbols}
    folds = tuple(PortfolioWalkForward(PortfolioWalkForwardConfig(
        purge_bars=60, embargo_bars=5, starting_equity=10_000, max_folds=selection_folds,
    )).selection_folds(panel))

    runs = []
    variants = _variants(best)
    if seeds_only:
        variants = [item for item in variants if item[0] in {"best", "best_seed_29", "best_seed_43"}]
    for name, config in variants:
        run_dir = output / name
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "result.json"
        if result_path.exists():
            runs.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        (run_dir / "config.json").write_text(
            json.dumps(_clean(asdict(config)), indent=2, sort_keys=True), encoding="utf-8")
        fold_rows = []
        try:
            for item in folds:
                fold_dir = run_dir / f"fold_{item.fold:02d}"
                model, scaler = train_ppo(item.train, item.validation, fold_dir, config)
                policy = SB3PortfolioPolicy(
                    model, context=item.train, scaler=scaler, lookback=config.lookback,
                    gross_budget=config.gross_budget, starting_equity=10_000,
                    action_cadence=config.action_cadence,
                )
                replay = PortfolioReplay(starting_cash=10_000).run(item.validation, policy)
                fold_rows.append({
                    "fold": item.fold, "train_start": item.train_start,
                    "train_end": item.train_end, "validation_start": item.validation_start,
                    "validation_end": item.validation_end,
                    "train_hashes": dict(item.train_hashes),
                    "validation_hashes": dict(item.validation_hashes),
                    "metrics": portfolio_metrics(replay), "test_accessed": False,
                })
            score = robust_validation_score([row["metrics"] for row in fold_rows])
            error = None
        except Exception as exc:
            score, error = -1_000.0, f"{type(exc).__name__}: {exc}"
        row = {"name": name, "config": _clean(asdict(config)), "folds": _clean(fold_rows),
               "robust_score": score, "invalid_reason": error, "test_accessed": False}
        result_path.write_text(json.dumps(_clean(row), indent=2, sort_keys=True), encoding="utf-8")
        runs.append(row)

    by_name = {row["name"]: float(row["robust_score"]) for row in runs}
    neighborhood = [v for k, v in by_name.items() if k not in {"best", "best_seed_29", "best_seed_43"}]
    seeds = [by_name[k] for k in ("best", "best_seed_29", "best_seed_43")]
    summary = {
        "source_best": str(Path(best_path).resolve()), "timesteps": timesteps,
        "selection_folds": selection_folds, "runs": _clean(runs),
        "neighborhood_median": float(np.median(neighborhood)) if neighborhood else None,
        "neighborhood_std": float(np.std(neighborhood, ddof=1)) if len(neighborhood) > 1 else 0.0,
        "seed_median": float(np.median(seeds)),
        "seed_std": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else 0.0,
        "isolated_spike": classify_isolated_spike(by_name["best"], neighborhood) if neighborhood else None,
        "selection_only": True, "test_accessed": False,
    }
    (output / "stability_summary.json").write_text(
        json.dumps(_clean(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("best_validation", type=Path)
    parser.add_argument("--output", type=Path, default=Path("rl_portfolio_management/runs/stability"))
    parser.add_argument("--selection-folds", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=1_500)
    parser.add_argument("--seeds-only", action="store_true")
    args = parser.parse_args()
    run_stability(args.manifest, args.best_validation, args.output,
                  selection_folds=args.selection_folds, timesteps=args.timesteps,
                  seeds_only=args.seeds_only)


if __name__ == "__main__":
    main()
