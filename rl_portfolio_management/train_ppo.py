"""Train one shared price-only PPO policy through PortfolioWalkForward."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.backtesting import PortfolioWalkForward, PortfolioWalkForwardConfig
from rl_portfolio_management.data_pipeline import load_snapshot
from rl_portfolio_management.ppo import PPOConfig, SB3PortfolioPolicy, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("rl_portfolio_management/runs/ppo"))
    parser.add_argument("--small", action="store_true", help="2k-step correctness run")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--one-fold", action="store_true")
    args = parser.parse_args()
    frames, manifest = load_snapshot(args.manifest)
    benchmark = manifest["config"].get("benchmark", "SPY")
    panel = {s: f for s, f in frames.items() if s != benchmark}
    timesteps = args.timesteps or (2_000 if args.small else 25_000)
    config = PPOConfig(timesteps=timesteps)
    wf = PortfolioWalkForwardConfig(purge_bars=60, embargo_bars=5, starting_equity=10_000,
                                    max_folds=1 if args.one_fold else None)

    def factory(train, validation, context):
        fold_dir = args.output / f"fold_{context.fold:02d}"
        model, scaler = train_ppo(train, validation, fold_dir, config)
        return SB3PortfolioPolicy(model, context=validation, scaler=scaler,
                                  lookback=config.lookback, gross_budget=config.gross_budget,
                                  action_cadence=config.action_cadence)

    result = PortfolioWalkForward(wf).run(panel, factory)
    metrics = [{"fold": fold.context.fold, "test_start": str(fold.test_start),
                "test_end": str(fold.test_end), "metrics": dict(fold.metrics)}
               for fold in result.folds]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
