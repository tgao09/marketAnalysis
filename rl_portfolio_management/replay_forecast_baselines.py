"""Replay frozen GP/GBM forecast artifacts as causal multi-asset portfolios.

This does not retrain either forecaster. A forecast is visible only when its
``trade_date`` equals the current observation date. Orders then execute under
``PortfolioReplay`` next-bar rules.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from common.backtesting import PortfolioObservation, PortfolioReplay, TargetPosition, portfolio_metrics
from rl_portfolio_management.data_pipeline import load_snapshot
from rl_portfolio_management.policies import EqualWeightLongPolicy, MomentumPolicy


SYMBOLS = ("AMT", "CAT", "JNJ", "JPM", "NEE", "WMT", "XOM")
STARTING_EQUITY = 10_000.0
GROSS_BUDGET = 0.5
PROVENANCE_LIMITATION = (
    "Forecast CSVs are frozen legacy backtest outputs, not independently regenerated predictions. "
    "trade_date is treated as forecast availability time, but artifacts contain no model-fit cutoff, "
    "training-data fingerprint, or per-row prediction-created timestamp; strict OOS provenance cannot "
    "be independently proven from these CSVs. Original overlapping same-close trade PnL is ignored."
)


def _date_key(values) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return pd.DatetimeIndex(parsed).tz_convert("America/New_York").normalize()


def load_forecasts(root: str | Path, model: str, symbols=SYMBOLS) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    rows, files = [], {}
    for symbol in symbols:
        matches = list((root / model / symbol / "regular").glob("*_return_trades.csv"))
        if len(matches) != 1:
            raise ValueError(f"expected one trade CSV for {model}/{symbol}, found {len(matches)}")
        path = matches[0]
        frame = pd.read_csv(path)
        required = {"symbol", "trade_date", "pred_mean_log"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        frame = frame.loc[:, ["symbol", "trade_date", "pred_mean_log"]].copy()
        frame["date"] = _date_key(frame["trade_date"])
        frame["pred_mean_log"] = pd.to_numeric(frame["pred_mean_log"], errors="coerce")
        if frame["date"].isna().any() or not np.isfinite(frame["pred_mean_log"]).all():
            raise ValueError(f"{path} has invalid trade_date/pred_mean_log")
        if set(frame["symbol"].astype(str)) != {symbol}:
            raise ValueError(f"{path} symbol mismatch")
        if frame["date"].duplicated().any():
            raise ValueError(f"{path} has duplicate forecast dates")
        rows.append(frame.loc[:, ["symbol", "date", "pred_mean_log"]])
        files[symbol] = {"path": str(path), "rows": len(frame), "first": frame.date.min(), "last": frame.date.max()}
    forecasts = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])
    return forecasts, {"model": model, "files": files, "provenance_limitation": PROVENANCE_LIMITATION}


class ForecastTargetPolicy:
    """Date-exact forecast policy; never carries a forecast into a later date."""

    def __init__(self, forecasts: pd.DataFrame, symbols=SYMBOLS, gross_budget: float = GROSS_BUDGET,
                 magnitude_clip: float = 0.05):
        if not 0 < gross_budget <= 1:
            raise ValueError("gross_budget must be in (0, 1]")
        self.symbols = tuple(symbols)
        self.gross_budget = float(gross_budget)
        self.magnitude_clip = float(magnitude_clip)
        self._by_date = {
            timestamp: dict(zip(group.symbol, group.pred_mean_log))
            for timestamp, group in forecasts.groupby("date", sort=False)
        }
        self.seen_dates: list[pd.Timestamp] = []
        self.target_weights: list[tuple[pd.Timestamp, dict[str, float]]] = []

    def weights_at(self, timestamp: pd.Timestamp) -> dict[str, float]:
        key = pd.Timestamp(timestamp).tz_convert("America/New_York").normalize()
        current = self._by_date.get(key, {})
        raw = {symbol: float(np.clip(current.get(symbol, 0.0), -self.magnitude_clip, self.magnitude_clip))
               for symbol in self.symbols}
        active = [symbol for symbol, value in raw.items() if abs(value) > 0]
        if not active:
            return {symbol: 0.0 for symbol in self.symbols}
        # Sparse cross-sections have no reliable magnitude ranking.
        if len(active) < math.ceil(len(self.symbols) / 2):
            allocation = self.gross_budget / len(active)
            return {symbol: (math.copysign(allocation, raw[symbol]) if symbol in active else 0.0)
                    for symbol in self.symbols}
        scale = self.gross_budget / sum(abs(raw[symbol]) for symbol in active)
        return {symbol: raw[symbol] * scale for symbol in self.symbols}

    def act(self, observation: PortfolioObservation):
        weights = self.weights_at(observation.timestamp)
        self.seen_dates.append(observation.timestamp)
        self.target_weights.append((observation.timestamp, weights))
        return tuple(TargetPosition(symbol, target_notional=weights[symbol] * observation.portfolio.equity)
                     for symbol in self.symbols)


def _clean(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if is_dataclass(value):
        return _clean(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _aligned(panel: Mapping[str, pd.DataFrame], dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    common = dates
    for frame in panel.values():
        common = common.intersection(frame.index, sort=False)
    common = common.sort_values()
    if common.empty:
        raise ValueError("no common replay timestamps")
    return {symbol: frame.loc[common].copy() for symbol, frame in panel.items()}


def _write_replay(name: str, replay, output: Path, metadata: dict | None = None) -> dict:
    metrics = portfolio_metrics(replay)
    equity = pd.DataFrame([
        {"strategy": name, "timestamp": timestamp, "equity": snap.equity, "cash": snap.cash,
         "drawdown": snap.drawdown, "gross_exposure": snap.gross_exposure,
         "net_exposure": snap.net_exposure}
        for timestamp, snap in replay.snapshots
    ])
    equity.to_csv(output / f"{name}_equity.csv", index=False)
    (output / f"{name}_equity.json").write_text(
        json.dumps(_clean(equity.to_dict(orient="records")), indent=2), encoding="utf-8"
    )
    orders = [_clean(order) for order in replay.orders]
    (output / f"{name}_orders.json").write_text(json.dumps(orders, indent=2, sort_keys=True), encoding="utf-8")
    pd.json_normalize(orders).to_csv(output / f"{name}_orders.csv", index=False)
    payload = {"strategy": name, "metrics": metrics, "metadata": metadata or {},
               "first_timestamp": replay.snapshots[0][0], "last_timestamp": replay.snapshots[-1][0]}
    (output / f"{name}_metrics.json").write_text(json.dumps(_clean(payload), indent=2, sort_keys=True), encoding="utf-8")
    return _clean(payload)


def _extract_rl_curves(exact_dates: pd.DatetimeIndex, root: Path, output: Path) -> dict:
    wanted = set(exact_dates)
    parts, summaries = [], []
    for path in sorted(root.glob("seed_*/fold_*/test_equity.csv")):
        frame = pd.read_csv(path)
        frame["timestamp"] = _date_key(frame["timestamp"])
        overlap = frame[frame.timestamp.isin(wanted)].copy()
        if overlap.empty:
            continue
        seed = int(path.parents[1].name.split("_")[1])
        fold = int(path.parent.name.split("_")[1])
        overlap["seed"], overlap["fold"], overlap["source"] = seed, fold, str(path)
        parts.append(overlap)
        summaries.append({"seed": seed, "fold": fold, "rows": len(overlap),
                          "first": overlap.timestamp.min(), "last": overlap.timestamp.max(),
                          "period_return": float(overlap.equity.iloc[-1] / overlap.equity.iloc[0] - 1)})
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    combined.to_csv(output / "rl_2025_overlap_equity.csv", index=False)
    by_seed = {}
    if not combined.empty:
        for seed, group in pd.DataFrame(summaries).groupby("seed"):
            by_seed[str(int(seed))] = {"fold_segments": len(group), "overlap_rows": int(group.rows.sum()),
                                      "mean_segment_return": float(group.period_return.mean()),
                                      "segment_returns": group.period_return.tolist()}
    return {"segments": _clean(summaries), "by_seed": by_seed,
            "comparability_warning": "RL used 13 assets and separately reset six-month test folds; GP/GBM use 7 assets in one 2025 replay. Curves are context only, not like-for-like portfolio metrics."}


def run(manifest: str | Path, artifact_root: str | Path = ".baseline_runs/basket",
        output_root: str | Path = "rl_portfolio_management/results/forecast_replay") -> Path:
    frames, snapshot = load_snapshot(manifest, verify=True)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for label, directory in (("gbm", "gbm_backtest"), ("gp", "gp_backtest")):
        loaded[label] = load_forecasts(artifact_root, directory)

    gbm_dates = pd.DatetimeIndex(sorted(loaded["gbm"][0].date.unique()))
    start, end = gbm_dates.min(), gbm_dates.max()
    requested_dates = frames[SYMBOLS[0]].index[(frames[SYMBOLS[0]].index >= start) & (frames[SYMBOLS[0]].index <= end)]
    seven = _aligned({symbol: frames[symbol] for symbol in SYMBOLS}, requested_dates)
    exact_dates = next(iter(seven.values())).index
    results = {}
    for label in ("gbm", "gp"):
        forecasts, audit = loaded[label]
        policy = ForecastTargetPolicy(forecasts)
        replay = PortfolioReplay(STARTING_EQUITY).run(seven, policy)
        audit.update({"gross_budget": GROSS_BUDGET, "symbols": list(SYMBOLS),
                      "forecast_rows": len(forecasts), "replay_dates": len(exact_dates),
                      "date_rule": "Only forecast rows whose trade_date equals observation date are visible; absent symbols target zero."})
        results[label] = _write_replay(label, replay, output, audit)

    for name, panel, policy in (
        ("equal_weight_7", seven, EqualWeightLongPolicy()),
        ("momentum_20d_7", seven, MomentumPolicy(lookback=20, rebalance_every=5, long_fraction=0.3)),
        ("spy_buy_hold", _aligned({"SPY": frames["SPY"]}, exact_dates), EqualWeightLongPolicy(rebalance_every=10_000)),
    ):
        results[name] = _write_replay(name, PortfolioReplay(STARTING_EQUITY).run(panel, policy), output,
                                      {"exact_date_match": True})

    rl = _extract_rl_curves(exact_dates, Path("rl_portfolio_management/runs/final_evaluation/final"), output)
    summary = {"snapshot_id": snapshot["snapshot_id"], "snapshot_content_sha256": snapshot["content_sha256"],
               "timestamps": {"count": len(exact_dates), "first": exact_dates.min(), "last": exact_dates.max()},
               "strategies": results, "rl_overlap": rl, "provenance_limitation": PROVENANCE_LIMITATION}
    (output / "summary.json").write_text(json.dumps(_clean(summary), indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([{"strategy": key, **value["metrics"]} for key, value in results.items()]).to_csv(output / "metrics.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--artifact-root", default=".baseline_runs/basket")
    parser.add_argument("--output-root", default="rl_portfolio_management/results/forecast_replay")
    args = parser.parse_args()
    print(run(args.manifest, args.artifact_root, args.output_root))


if __name__ == "__main__":
    main()
