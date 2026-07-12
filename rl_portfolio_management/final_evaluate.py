"""Frozen, untouched-test walk-forward evaluation for selected PPO config."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from common.backtesting import PortfolioWalkForward, PortfolioWalkForwardConfig
from rl_portfolio_management.data_pipeline import load_snapshot
from rl_portfolio_management.optimize_ppo import _clean
from rl_portfolio_management.ppo import PPOConfig, SB3PortfolioPolicy, train_ppo
from rl_portfolio_management.rl_env import RewardConfig


DEFAULT_SEEDS = (17, 29, 43)


def config_from_params(params: Mapping[str, object], *, timesteps: int, seed: int) -> PPOConfig:
    """Construct fixed PPO config using search defaults for absent parameters."""
    width = int(params.get("net_width", 128))
    return PPOConfig(
        seed=seed, timesteps=timesteps, net_arch=(width, width),
        lookback=int(params.get("lookback", 60)),
        learning_rate=float(params.get("learning_rate", 3e-4)),
        gamma=float(params.get("gamma", 0.99)),
        gae_lambda=float(params.get("gae_lambda", 0.95)),
        ent_coef=float(params.get("ent_coef", 0.0)),
        n_steps=int(params.get("n_steps", 128)),
        batch_size=int(params.get("batch_size", 64)),
        action_cadence=int(params.get("action_cadence", 1)),
        gross_budget=float(params.get("gross_budget", 0.8)),
        reward=RewardConfig(
            drawdown=float(params.get("reward_drawdown", 0.05)),
            turnover=float(params.get("reward_turnover", 0.001)),
            exposure_instability=float(params.get("reward_exposure_instability", 0.001)),
            holding_time=float(params.get("reward_holding_time", 0.001)),
            holding_target_min=int(params.get("holding_target_min", 3)),
            holding_target_max=int(params.get("holding_target_max", 5)),
        ),
    )


def exclude_development_fold(panel: Mapping[str, pd.DataFrame], config: PortfolioWalkForwardConfig):
    """Drop all data before original fold 1 train start, excluding fold 0 dev test."""
    wf = PortfolioWalkForward(config)
    folds = tuple(wf.selection_folds(panel))
    if len(folds) < 2:
        raise ValueError("need at least two original complete folds to exclude development fold 0")
    cutoff = folds[1].train_start
    sliced = {symbol: frame.loc[frame.index >= cutoff].copy() for symbol, frame in panel.items()}
    audit = {
        "rule": "panel sliced from original selection fold 1 train_start",
        "original_development_fold": 0,
        "excluded_through": (cutoff - pd.Timedelta(nanoseconds=1)).isoformat(),
        "remaining_panel_start": cutoff.isoformat(),
        "original_fold_1_train_start": cutoff.isoformat(),
    }
    return sliced, audit


def _sha256(value: object) -> str:
    payload = json.dumps(_clean(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _equity_frame(seed: int, fold) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [timestamp.isoformat() for timestamp, _ in fold.replay.snapshots],
        "equity": [snapshot.equity for _, snapshot in fold.replay.snapshots],
        "cash": [snapshot.cash for _, snapshot in fold.replay.snapshots],
        "gross_exposure": [snapshot.gross_exposure for _, snapshot in fold.replay.snapshots],
        "net_exposure": [snapshot.net_exposure for _, snapshot in fold.replay.snapshots],
        "seed": seed, "fold": fold.context.fold,
    })


def _dispersion(rows: Sequence[Mapping[str, object]]) -> dict:
    metric_names = sorted(set.intersection(*(
        {key for key, value in row["metrics"].items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        for row in rows
    ))) if rows else []
    output = {}
    for name in metric_names:
        values = np.asarray([row["metrics"][name] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            output[name] = {"count": int(values.size), "mean": float(values.mean()),
                            "median": float(np.median(values)),
                            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                            "min": float(values.min()), "max": float(values.max())}
    return output


def run_final(manifest_path: str | Path, best_path: str | Path, output: str | Path,
              *, seeds: Sequence[int] = DEFAULT_SEEDS, timesteps: int = 1_500) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    best_path = Path(best_path).resolve()
    best_record = json.loads(best_path.read_text(encoding="utf-8"))
    if best_record.get("test_accessed") is not False:
        raise ValueError("best-validation record must explicitly state test_accessed=false")
    params = dict(best_record["params"])
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")

    base_config = PortfolioWalkForwardConfig(
        purge_bars=60, embargo_bars=5, starting_equity=10_000)
    frames, manifest = load_snapshot(manifest_path, verify=True)
    symbols = tuple(manifest["config"]["symbols"])
    original_panel = {symbol: frames[symbol] for symbol in symbols}
    panel, exclusion = exclude_development_fold(original_panel, base_config)

    frozen_payload = {
        "source_best_validation": str(best_path),
        "source_best_validation_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "snapshot_id": manifest["snapshot_id"], "snapshot_content_sha256": manifest["content_sha256"],
        "params": params, "timesteps": timesteps, "seeds": list(seeds),
        "walk_forward": _clean(asdict(base_config)), "development_exclusion": exclusion,
        "selection_frozen_before_test": True, "test_tuning_permitted": False,
    }
    frozen_payload["frozen_config_sha256"] = _sha256(frozen_payload)
    frozen_path = output / "frozen_config.json"
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text(encoding="utf-8"))
        if existing != frozen_payload:
            raise ValueError("output contains different frozen configuration")
    else:
        frozen_path.write_text(json.dumps(_clean(frozen_payload), indent=2, sort_keys=True), encoding="utf-8")

    seed_results = []
    for seed in seeds:
        seed_dir = output / "final" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        result_path = seed_dir / "result.json"
        if result_path.exists():
            completed = json.loads(result_path.read_text(encoding="utf-8"))
            if completed.get("completed") is True and completed.get("frozen_config_sha256") == frozen_payload["frozen_config_sha256"]:
                seed_results.append(completed)
                continue
        config = config_from_params(params, timesteps=timesteps, seed=seed)

        def factory(train, validation, context):
            fold_dir = seed_dir / f"fold_{context.fold:02d}"
            model, scaler = train_ppo(train, validation, fold_dir, config)
            return SB3PortfolioPolicy(
                model, context=train, scaler=scaler, lookback=config.lookback,
                gross_budget=config.gross_budget, starting_equity=10_000,
                action_cadence=config.action_cadence,
            )

        result = PortfolioWalkForward(base_config).run(panel, factory)
        fold_rows = []
        for fold in result.folds:
            equity_path = seed_dir / f"fold_{fold.context.fold:02d}" / "test_equity.csv"
            _equity_frame(seed, fold).to_csv(equity_path, index=False)
            fold_rows.append({
                "seed": seed, "fold": fold.context.fold,
                "train_start": fold.context.train_start, "train_end": fold.context.train_end,
                "validation_start": fold.context.validation_start, "validation_end": fold.context.validation_end,
                "test_start": fold.test_start, "test_end": fold.test_end,
                "train_hashes": dict(fold.context.train_hashes),
                "validation_hashes": dict(fold.context.validation_hashes),
                "test_hashes": dict(fold.test_hashes), "metrics": dict(fold.metrics),
                "test_accessed": fold.test_accessed, "equity_csv": str(equity_path),
            })
        seed_result = {
            "seed": seed, "completed": True, "panel_hash": result.panel_hash,
            "frozen_config_sha256": frozen_payload["frozen_config_sha256"],
            "folds": _clean(fold_rows), "test_accessed": True,
            "test_used_for_tuning": False,
        }
        result_path.write_text(json.dumps(seed_result, indent=2, sort_keys=True), encoding="utf-8")
        seed_results.append(seed_result)

    rows = [fold for seed_result in seed_results for fold in seed_result["folds"]]
    flat = [{"seed": row["seed"], "fold": row["fold"], **row["metrics"]} for row in rows]
    pd.DataFrame(flat).to_csv(output / "final_metrics.csv", index=False)
    summary = {
        "completed": True, "frozen_config_sha256": frozen_payload["frozen_config_sha256"],
        "seeds": list(seeds), "fold_count": len(rows), "seed_results": seed_results,
        "aggregate_seed_fold_dispersion": _dispersion(rows),
        "development_exclusion": exclusion, "test_accessed": True,
        "test_used_for_tuning": False,
    }
    (output / "final_summary.json").write_text(json.dumps(_clean(summary), indent=2, sort_keys=True), encoding="utf-8")
    (output / "test_access_audit.json").write_text(json.dumps({
        "selection_source_test_accessed": best_record.get("test_accessed"),
        "selection_frozen_before_test": True, "test_accessed_for_final_reporting": True,
        "test_used_for_tuning": False, "development_exclusion": exclusion,
        "frozen_config_sha256": frozen_payload["frozen_config_sha256"],
    }, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("best_validation", type=Path)
    parser.add_argument("--output", type=Path, default=Path("rl_portfolio_management/runs/final_evaluation"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--timesteps", type=int, default=1_500)
    args = parser.parse_args()
    run_final(args.manifest, args.best_validation, args.output, seeds=args.seeds, timesteps=args.timesteps)


if __name__ == "__main__":
    main()
