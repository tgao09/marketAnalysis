"""Run deterministic portfolio baselines through Universal Backtester."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from common.backtesting import PortfolioWalkForward, PortfolioWalkForwardConfig
from rl_portfolio_management.data_pipeline import load_snapshot
from rl_portfolio_management.policies import AlwaysCashPolicy, EqualWeightLongPolicy, MomentumPolicy, RandomValidPolicy


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _run(name, panel, factory, config, output):
    result = PortfolioWalkForward(config).run(panel, lambda train, validation, context: factory())
    folds = []
    curves = []
    for fold in result.folds:
        folds.append({
            "fold": fold.context.fold,
            "train_start": fold.context.train_start,
            "train_end": fold.context.train_end,
            "validation_start": fold.context.validation_start,
            "validation_end": fold.context.validation_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "train_hashes": dict(fold.context.train_hashes),
            "validation_hashes": dict(fold.context.validation_hashes),
            "test_hashes": dict(fold.test_hashes),
            "test_accessed": fold.test_accessed,
            "metrics": dict(fold.metrics),
        })
        for timestamp, snapshot in fold.replay.snapshots:
            curves.append({"strategy": name, "fold": fold.context.fold, "timestamp": timestamp, "equity": snapshot.equity, "drawdown": snapshot.drawdown, "gross_exposure": snapshot.gross_exposure / snapshot.equity, "net_exposure": snapshot.net_exposure / snapshot.equity})
    payload = _clean({"strategy": name, "panel_hash": result.panel_hash, "config": vars(config), "folds": folds})
    (output / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return curves, payload


def run(manifest_path: str | Path, output_root: str | Path | None = None):
    frames, manifest = load_snapshot(manifest_path, verify=True)
    universe = tuple(manifest["config"]["symbols"])
    panel = {symbol: frames[symbol] for symbol in universe}
    output = Path(output_root or Path("rl_portfolio_management/results/baselines") / manifest["snapshot_id"])
    output.mkdir(parents=True, exist_ok=True)
    config = PortfolioWalkForwardConfig(purge_bars=60, embargo_bars=5, starting_equity=10_000)
    definitions = {
        "always_cash": AlwaysCashPolicy,
        "equal_weight_long": EqualWeightLongPolicy,
        "momentum_20d": lambda: MomentumPolicy(lookback=20, rebalance_every=5, long_fraction=0.3),
        "random_seed_7": lambda: RandomValidPolicy(seed=7),
        "random_seed_42": lambda: RandomValidPolicy(seed=42),
        "random_seed_101": lambda: RandomValidPolicy(seed=101),
    }
    all_curves, summaries = [], {}
    for name, factory in definitions.items():
        curves, summaries[name] = _run(name, panel, factory, config, output)
        all_curves.extend(curves)
    curves, summaries["spy_buy_hold"] = _run("spy_buy_hold", {"SPY": frames[manifest["config"]["benchmark"]]}, EqualWeightLongPolicy, config, output)
    all_curves.extend(curves)
    pd.DataFrame(all_curves).to_csv(output / "equity_curves.csv", index=False)
    aggregate = {name: {"folds": len(value["folds"]), "median_sharpe": float(pd.Series([fold["metrics"]["sharpe"] for fold in value["folds"]]).median()), "median_calmar": float(pd.Series([fold["metrics"]["calmar"] for fold in value["folds"]]).median()), "median_return": float(pd.Series([fold["metrics"]["cumulative_return"] for fold in value["folds"]]).median()), "worst_drawdown": float(pd.Series([fold["metrics"]["maximum_drawdown"] for fold in value["folds"]]).max())} for name, value in summaries.items()}
    (output / "aggregate.json").write_text(json.dumps(_clean(aggregate), indent=2, sort_keys=True), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    print(run(args.manifest, args.output_root))


if __name__ == "__main__":
    main()
