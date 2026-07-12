"""Validation-only Optuna search for shared cross-asset PPO.

No call path in this module invokes ``PortfolioWalkForward.run`` or receives a
test slice. Final test evaluation belongs to a separate, frozen-config command.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from common.backtesting import PortfolioReplay, PortfolioWalkForward, PortfolioWalkForwardConfig, portfolio_metrics
from rl_portfolio_management.data_pipeline import load_snapshot
from rl_portfolio_management.ppo import PPOConfig, SB3PortfolioPolicy, train_ppo
from rl_portfolio_management.rl_env import RewardConfig


def robust_validation_score(metrics: Sequence[Mapping[str, object]]) -> float:
    """Robust fold score; returns finite invalid penalty for unusable trials."""
    if not metrics:
        return -1_000.0
    rows = []
    for value in metrics:
        try:
            calmar = float(value["calmar"])
            sharpe = float(value["sharpe"])
            drawdown = float(value["maximum_drawdown"])
        except (KeyError, TypeError, ValueError):
            return -1_000.0
        if not np.isfinite([calmar, sharpe, drawdown]).all() or not 0 <= drawdown <= 1:
            return -1_000.0
        # Bounds prevent tiny drawdowns or one explosive fold dominating selection.
        rows.append(float(np.clip(calmar, -5, 5) + np.clip(sharpe, -4, 4) / 2 - drawdown))
    dispersion = float(np.std(rows, ddof=1)) if len(rows) > 1 else 0.0
    return float(np.median(rows) - dispersion)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def sampled_config(trial, *, timesteps: int, seed: int) -> PPOConfig:
    """Small search space containing only high-impact supported parameters."""
    width = trial.suggest_categorical("net_width", [64, 128, 256])
    return PPOConfig(
        seed=seed, timesteps=timesteps, net_arch=(width, width),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        gamma=trial.suggest_float("gamma", 0.95, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.85, 0.99),
        ent_coef=trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
        n_steps=trial.suggest_categorical("n_steps", [64, 128, 256]),
        action_cadence=trial.suggest_categorical("action_cadence", [1, 3, 5]),
        gross_budget=trial.suggest_float("gross_budget", 0.4, 0.9),
        reward=RewardConfig(
            drawdown=trial.suggest_float("reward_drawdown", 0.0, 0.2),
            turnover=trial.suggest_float("reward_turnover", 0.0, 0.01),
        ),
    )


def optimize(manifest_path: str | Path, output: str | Path, *, trials: int = 8,
             selection_folds: int = 2, timesteps: int = 1_500, seed: int = 17):
    import optuna

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames, manifest = load_snapshot(manifest_path, verify=True)
    symbols = tuple(manifest["config"]["symbols"])
    panel = {symbol: frames[symbol] for symbol in symbols}
    wf = PortfolioWalkForward(PortfolioWalkForwardConfig(
        purge_bars=60, embargo_bars=5, starting_equity=10_000,
        max_folds=selection_folds,
    ))
    # Materialize selection views only. Object schema cannot represent test dates/data/hashes.
    folds = tuple(wf.selection_folds(panel))
    database = output / "optuna.sqlite3"
    study = optuna.create_study(
        study_name="ppo_validation", direction="maximize", load_if_exists=True,
        storage=f"sqlite:///{database.as_posix()}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )

    def objective(trial):
        config = sampled_config(trial, timesteps=timesteps, seed=seed)
        trial_dir = output / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "config.json").write_text(
            json.dumps(_clean(asdict(config)), indent=2, sort_keys=True), encoding="utf-8")
        fold_rows = []
        try:
            for item in folds:
                fold_dir = trial_dir / f"fold_{item.fold:02d}"
                model, scaler = train_ppo(item.train, item.validation, fold_dir, config)
                frozen = SB3PortfolioPolicy(
                    model, context=item.train, scaler=scaler, lookback=config.lookback,
                    gross_budget=config.gross_budget, starting_equity=10_000,
                    action_cadence=config.action_cadence,
                )
                replay = PortfolioReplay(starting_cash=10_000).run(item.validation, frozen)
                metrics = portfolio_metrics(replay)
                fold_rows.append({
                    "fold": item.fold, "train_start": item.train_start,
                    "train_end": item.train_end, "validation_start": item.validation_start,
                    "validation_end": item.validation_end,
                    "train_hashes": dict(item.train_hashes),
                    "validation_hashes": dict(item.validation_hashes),
                    "metrics": metrics, "test_accessed": False,
                })
                score = robust_validation_score([row["metrics"] for row in fold_rows])
                trial.report(score, step=len(fold_rows))
                if trial.should_prune():
                    raise optuna.TrialPruned()
            score = robust_validation_score([row["metrics"] for row in fold_rows])
            if score <= -1_000:
                raise optuna.TrialPruned("invalid validation metrics")
            return score
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("invalid_reason", f"{type(exc).__name__}: {exc}")
            return -1_000.0
        finally:
            (trial_dir / "validation_metrics.json").write_text(
                json.dumps(_clean(fold_rows), indent=2, sort_keys=True), encoding="utf-8")

    study.optimize(objective, n_trials=trials, gc_after_trial=True)
    study.trials_dataframe().to_csv(output / "study_trials.csv", index=False)
    best = {"number": study.best_trial.number, "value": study.best_value,
            "params": study.best_params, "selection_only": True, "test_accessed": False}
    (output / "best_validation.json").write_text(
        json.dumps(_clean(best), indent=2, sort_keys=True), encoding="utf-8")
    return study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("rl_portfolio_management/runs/optuna_rough"))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--selection-folds", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    optimize(args.manifest, args.output, trials=args.trials,
             selection_folds=args.selection_folds, timesteps=args.timesteps, seed=args.seed)


if __name__ == "__main__":
    main()
