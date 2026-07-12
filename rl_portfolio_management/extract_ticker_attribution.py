"""Reporting-only per-ticker attribution from frozen final PPO checkpoints.

This module never trains or selects a model. It reconstructs the exact final
panel/folds, fits each training-fold scaler, loads ``best_model.zip``, and
replays only the already-designated test slices through Universal Backtester.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from common.backtesting import Order, OrderSide, OrderStatus, PortfolioWalkForward, PortfolioWalkForwardConfig
from rl_portfolio_management.data_pipeline import load_snapshot
from rl_portfolio_management.features import TrainingFoldScaler
from rl_portfolio_management.final_evaluate import DEFAULT_SEEDS, exclude_development_fold
from rl_portfolio_management.ppo import SB3PortfolioPolicy, prepare_fold_features


ROOT = Path(__file__).resolve().parent
DEFAULT_FINAL = ROOT / "runs" / "final_evaluation"


def fifo_trade_stats(orders: Iterable[Order], bar_index: pd.DatetimeIndex) -> dict[str, dict[str, float]]:
    """Match filled quantities FIFO and return completed-lot statistics.

    A reversal closes old lots before opening a residual lot. Holding duration
    counts observed trading bars between fill timestamps, not calendar days.
    """
    positions: dict[str, deque[list[object]]] = defaultdict(deque)
    pnl: dict[str, list[float]] = defaultdict(list)
    holding: dict[str, list[int]] = defaultdict(list)
    indexer = {pd.Timestamp(ts): i for i, ts in enumerate(bar_index)}
    fills = sorted(
        (o for o in orders if o.status is OrderStatus.FILLED and o.filled_at is not None),
        key=lambda o: (pd.Timestamp(o.filled_at), o.order_id),
    )
    for order in fills:
        symbol = order.request.symbol
        signed = float(order.filled_quantity) * (1.0 if order.request.side is OrderSide.BUY else -1.0)
        remaining = signed
        lots = positions[symbol]
        while lots and remaining * float(lots[0][0]) < 0:
            lot_qty, lot_price, lot_time = lots[0]
            close_qty = min(abs(remaining), abs(float(lot_qty)))
            direction = 1.0 if float(lot_qty) > 0 else -1.0
            pnl[symbol].append(close_qty * (float(order.fill_price) - float(lot_price)) * direction)
            holding[symbol].append(indexer[pd.Timestamp(order.filled_at)] - indexer[pd.Timestamp(lot_time)])
            new_lot = float(lot_qty) - direction * close_qty
            remaining += direction * close_qty
            if abs(new_lot) < 1e-12:
                lots.popleft()
            else:
                lots[0][0] = new_lot
        if abs(remaining) >= 1e-12:
            lots.append([remaining, float(order.fill_price), pd.Timestamp(order.filled_at)])

    output = {}
    for symbol in sorted(set(positions) | set(pnl)):
        values = np.asarray(pnl[symbol], dtype=float)
        ages = np.asarray(holding[symbol], dtype=float)
        output[symbol] = {
            "completed_trade_count": int(values.size),
            "completed_trade_win_rate": float(np.mean(values > 0)) if values.size else 0.0,
            "completed_trade_avg_pnl": float(values.mean()) if values.size else 0.0,
            "completed_trade_median_pnl": float(np.median(values)) if values.size else 0.0,
            "completed_trade_avg_holding_bars": float(ages.mean()) if ages.size else 0.0,
            "completed_trade_median_holding_bars": float(np.median(ages)) if ages.size else 0.0,
            "long_matched_contribution": float(sum(v for v, o in zip(pnl[symbol], _matched_sides(fills, symbol)) if o > 0)),
            "short_matched_contribution": float(sum(v for v, o in zip(pnl[symbol], _matched_sides(fills, symbol)) if o < 0)),
        }
    return output


def _matched_sides(fills: Sequence[Order], symbol: str) -> list[int]:
    """Originating lot directions for FIFO matches; mirrors matcher cheaply."""
    lots: deque[list[float]] = deque()
    sides: list[int] = []
    for order in fills:
        if order.request.symbol != symbol:
            continue
        remaining = float(order.filled_quantity) * (1 if order.request.side is OrderSide.BUY else -1)
        while lots and remaining * lots[0][0] < 0:
            quantity = min(abs(remaining), abs(lots[0][0]))
            direction = 1 if lots[0][0] > 0 else -1
            sides.append(direction)
            lots[0][0] -= direction * quantity
            remaining += direction * quantity
            if abs(lots[0][0]) < 1e-12:
                lots.popleft()
        if abs(remaining) >= 1e-12:
            lots.append([remaining])
    return sides


def attribution_rows(seed: int, fold: int, symbols: Sequence[str], replay) -> list[dict[str, object]]:
    """Aggregate one replay and enforce symbol P&L/equity conservation."""
    if not replay.snapshots:
        raise ValueError("replay has no snapshots")
    timestamps = pd.DatetimeIndex([timestamp for timestamp, _ in replay.snapshots])
    final = replay.snapshots[-1][1]
    fifo = fifo_trade_stats(replay.orders, timestamps)
    filled = [o for o in replay.orders if o.status is OrderStatus.FILLED]
    rows = []
    for symbol in symbols:
        position = final.positions.get(symbol)
        realized = 0.0 if position is None else float(position.realized_pnl)
        unrealized = 0.0 if position is None else float(position.quantity * (
            # Snapshot marks at test_end close; infer mark from equity accounting
            # is unnecessary because caller supplies it below after replay.
            0.0 - position.average_cost))
        stats = fifo.get(symbol, {})
        symbol_orders = [o for o in filled if o.request.symbol == symbol]
        row = {
            "seed": seed, "fold": fold, "symbol": symbol,
            "filled_order_count": len(symbol_orders),
            "filled_notional": float(sum(float(o.filled_quantity) * float(o.fill_price) for o in symbol_orders)),
            "buy_filled_notional": float(sum(float(o.filled_quantity) * float(o.fill_price) for o in symbol_orders if o.request.side is OrderSide.BUY)),
            "sell_filled_notional": float(sum(float(o.filled_quantity) * float(o.fill_price) for o in symbol_orders if o.request.side is OrderSide.SELL)),
            "realized_pnl": realized, "ending_unrealized_pnl": unrealized,
            **{k: stats.get(k, 0 if k == "completed_trade_count" else 0.0) for k in (
                "completed_trade_count", "completed_trade_win_rate", "completed_trade_avg_pnl",
                "completed_trade_median_pnl", "completed_trade_avg_holding_bars",
                "completed_trade_median_holding_bars", "long_matched_contribution",
                "short_matched_contribution")},
        }
        rows.append(row)
    return rows


def reconcile_rows(rows: Sequence[Mapping[str, object]], ending_equity: float,
                   starting_equity: float = 10_000.0, tolerance: float = 1e-6) -> float:
    """Return residual after asserting total symbol contribution equals equity P&L."""
    contribution = sum(float(row["total_contribution"]) for row in rows)
    residual = contribution - (float(ending_equity) - float(starting_equity))
    if abs(residual) > tolerance:
        raise AssertionError(f"ticker contribution residual {residual} exceeds {tolerance}")
    return residual


def run(final_root: str | Path = DEFAULT_FINAL, output: str | Path | None = None) -> dict:
    """Replay frozen checkpoints and export reporting-only attribution."""
    from stable_baselines3 import PPO

    final_root = Path(final_root).resolve()
    output = Path(output or final_root / "ticker_attribution").resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((final_root / "frozen_config.json").read_text(encoding="utf-8"))
    snapshot = ROOT / "data" / "snapshots" / frozen["snapshot_id"] / "manifest.json"
    frames, manifest = load_snapshot(snapshot, verify=True)
    if manifest["content_sha256"] != frozen["snapshot_content_sha256"]:
        raise ValueError("frozen snapshot hash mismatch")
    symbols = tuple(manifest["config"]["symbols"])
    config = PortfolioWalkForwardConfig(**frozen["walk_forward"])
    panel, exclusion = exclude_development_fold({s: frames[s] for s in symbols}, config)
    if exclusion != frozen["development_exclusion"]:
        raise ValueError("development-fold exclusion mismatch")

    all_rows: list[dict[str, object]] = []
    reconciliations = []
    current_seed = None

    for seed in frozen["seeds"]:
        current_seed = int(seed)
        seed_dir = final_root / "final" / f"seed_{seed}"

        def factory(train, validation, context):
            fold_dir = seed_dir / f"fold_{context.fold:02d}"
            saved = json.loads((fold_dir / "config.json").read_text(encoding="utf-8"))
            _, _, _, _, scaler = prepare_fold_features(train, validation, int(saved["lookback"]))
            # Stronger than trusting serialized scaler: reconstructed fingerprint
            # must equal training-time artifact before checkpoint inference.
            stored = json.loads((fold_dir / "scaler.json").read_text(encoding="utf-8"))
            if scaler.fingerprint != stored["fingerprint"]:
                raise ValueError(f"scaler mismatch seed={seed} fold={context.fold}")
            model = PPO.load(fold_dir / "best_model.zip")
            return SB3PortfolioPolicy(model, context=train, scaler=scaler,
                                      lookback=int(saved["lookback"]),
                                      gross_budget=float(saved["gross_budget"]),
                                      action_cadence=int(saved["action_cadence"]),
                                      starting_equity=config.starting_equity)

        result = PortfolioWalkForward(config).run(panel, factory)
        for fold_result in result.folds:
            test_end = fold_result.test_end
            rows = attribution_rows(current_seed, fold_result.context.fold, symbols, fold_result.replay)
            final_snapshot = fold_result.replay.snapshots[-1][1]
            for row in rows:
                symbol = str(row["symbol"])
                position = final_snapshot.positions.get(symbol)
                close = float(panel[symbol].loc[test_end, "Close"])
                unrealized = 0.0 if position is None else float(position.quantity * (close - position.average_cost))
                row["ending_unrealized_pnl"] = unrealized
                row["total_contribution"] = float(row["realized_pnl"]) + unrealized
            residual = reconcile_rows(rows, final_snapshot.equity, config.starting_equity)
            reconciliations.append({"seed": current_seed, "fold": fold_result.context.fold,
                                    "ending_equity": final_snapshot.equity, "residual": residual})
            all_rows.extend(rows)

    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "attribution_rows.csv", index=False)
    by_symbol = []
    for symbol, group in frame.groupby("symbol", sort=True):
        trades = int(group.completed_trade_count.sum())
        by_symbol.append({
            "symbol": symbol, "seed_fold_count": len(group),
            "filled_order_count": int(group.filled_order_count.sum()),
            "filled_notional": float(group.filled_notional.sum()),
            "completed_trade_count": trades,
            "completed_trade_win_rate_weighted": float(np.average(group.completed_trade_win_rate,
                weights=group.completed_trade_count)) if trades else 0.0,
            "realized_pnl": float(group.realized_pnl.sum()),
            "ending_unrealized_pnl": float(group.ending_unrealized_pnl.sum()),
            "total_contribution": float(group.total_contribution.sum()),
            "long_matched_contribution": float(group.long_matched_contribution.sum()),
            "short_matched_contribution": float(group.short_matched_contribution.sum()),
        })
    aggregate = {
        "reporting_only": True, "retrained": False, "test_used_for_tuning": False,
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "snapshot_content_sha256": frozen["snapshot_content_sha256"],
        "seed_fold_count": len(reconciliations), "row_count": len(all_rows),
        "max_abs_reconciliation_residual": float(max(abs(x["residual"]) for x in reconciliations)),
        "reconciliations": reconciliations, "by_symbol": by_symbol,
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = run(args.final_root, args.output)
    print(json.dumps({k: summary[k] for k in ("seed_fold_count", "row_count", "max_abs_reconciliation_residual")}, indent=2))


if __name__ == "__main__":
    main()
