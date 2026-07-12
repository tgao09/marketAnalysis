"""Lean, leakage-safe SB3 PPO training and replay adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from common.backtesting import Hold, PortfolioObservation, TargetPosition
from .features import TrainingFoldScaler, build_asset_features
from .rl_env import CrossAssetPortfolioEnv, RewardConfig

Panel = Mapping[str, pd.DataFrame]


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 17
    timesteps: int = 25_000
    net_arch: tuple[int, ...] = (128, 128)
    lookback: int = 60
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    n_steps: int = 256
    batch_size: int = 64
    gross_budget: float = 0.9
    action_cadence: int = 1
    reward: RewardConfig = RewardConfig()


def _feature_panel(bars: Panel) -> dict[str, pd.DataFrame]:
    """Price-only features; SPY-relative inputs intentionally deferred."""
    return {symbol: build_asset_features(frame) for symbol, frame in sorted(bars.items())}


def prepare_fold_features(train: Panel, validation: Panel, lookback: int = 60):
    """Fit train-only scaler and construct validation with causal train tail."""
    if lookback < 60:
        raise ValueError("lookback must cover 60-bar feature warm-up")
    raw_train = _feature_panel(train)
    usable = next(iter(train.values())).index[lookback:]
    train_features = {s: f.loc[usable] for s, f in raw_train.items()}
    if any(not np.isfinite(f.to_numpy(float)).all() for f in train_features.values()):
        raise ValueError("non-finite training features remain after warm-up")
    scaler = TrainingFoldScaler.fit(train_features)
    train_scaled = {s: scaler.transform(f) for s, f in train_features.items()}
    validation_scaled = {}
    for symbol in sorted(train):
        joined = pd.concat([train[symbol].iloc[-lookback:], validation[symbol]])
        causal = build_asset_features(joined).loc[validation[symbol].index]
        validation_scaled[symbol] = scaler.transform(causal)
        if not np.isfinite(validation_scaled[symbol].to_numpy(float)).all():
            raise ValueError(f"non-finite validation features: {symbol}")
    train_bars = {s: f.loc[usable].copy() for s, f in train.items()}
    return train_bars, train_scaled, {s: f.copy() for s, f in validation.items()}, validation_scaled, scaler


def prepare_replay_features(context: Panel, replay: Panel, scaler: TrainingFoldScaler,
                            lookback: int = 60) -> dict[str, pd.DataFrame]:
    """Build replay features from preceding context and bars through each t only."""
    output = {}
    for symbol in sorted(replay):
        joined = pd.concat([context[symbol].iloc[-lookback:], replay[symbol]])
        output[symbol] = scaler.transform(build_asset_features(joined).loc[replay[symbol].index])
        if not np.isfinite(output[symbol].to_numpy(float)).all():
            raise ValueError(f"non-finite replay features: {symbol}")
    return output


class SB3PortfolioPolicy:
    """Frozen shared cross-asset model exposed as PortfolioPolicy."""

    def __init__(self, model, features: Mapping[str, pd.DataFrame] | None = None, *,
                 context: Panel | None = None, scaler: TrainingFoldScaler | None = None,
                 lookback: int = 60, gross_budget: float = 0.9,
                 action_cadence: int = 1,
                 starting_equity: float = 10_000.0):
        self.model = model
        if features is None and (context is None or scaler is None):
            raise ValueError("provide frozen features or streaming context plus scaler")
        source = features if features is not None else context
        self.features = None if features is None else {s: f.copy(deep=True) for s, f in sorted(features.items())}
        self.symbols = tuple(sorted(source))
        self._history = None if context is None else {s: f.iloc[-lookback:].copy(deep=True) for s, f in sorted(context.items())}
        self.scaler = scaler
        self.lookback = lookback
        self.gross_budget = float(gross_budget)
        if isinstance(action_cadence, bool) or not isinstance(action_cadence, int) or action_cadence <= 0:
            raise ValueError("action_cadence must be a positive integer")
        self.action_cadence = action_cadence
        self._action_counter = 0
        self.starting_equity = float(starting_equity)
        self._peak = starting_equity

    def prepare_replay(self, warmup: Panel) -> None:
        """Reset causal feature history after model selection, before test."""
        if set(warmup) != set(self.symbols):
            raise ValueError("warmup symbols differ from policy symbols")
        self._history = {
            symbol: warmup[symbol].iloc[-self.lookback:].copy(deep=True)
            for symbol in self.symbols
        }
        self.features = None
        self._action_counter = 0

    def observation_vector(self, observation: PortfolioObservation) -> np.ndarray:
        """Match CrossAssetPortfolioEnv observation layout exactly."""
        pstate, state = observation.portfolio, []
        if self.features is None:
            rows = {}
            for symbol in self.symbols:
                bar = observation.bars[symbol]
                row = pd.DataFrame({"Open": [bar.open], "High": [bar.high], "Low": [bar.low],
                                    "Close": [bar.close], "Volume": [bar.volume]},
                                   index=pd.DatetimeIndex([observation.timestamp]))
                history = pd.concat([self._history[symbol], row])
                history = history[~history.index.duplicated(keep="last")].sort_index().iloc[-self.lookback-1:]
                self._history[symbol] = history
                rows[symbol] = self.scaler.transform(build_asset_features(history)).iloc[-1]
        else:
            rows = {s: self.features[s].loc[observation.timestamp] for s in self.symbols}
        equity = pstate.equity
        self._peak = max(self._peak, equity)
        for symbol in self.symbols:
            state.extend(rows[symbol].to_numpy(float))
            position = pstate.positions.get(symbol)
            quantity = 0.0 if position is None else position.quantity
            close = observation.bars[symbol].close
            weight = quantity * close / max(equity, 1e-12)
            distance = 0.0 if not quantity else np.sign(quantity) * (close / position.average_cost - 1)
            age = 0 if position is None else position.age_bars
            state.extend((weight, distance, age / 20.0))
        state.extend((pstate.cash/max(equity, 1e-12), equity/self.starting_equity,
                      pstate.gross_exposure/max(equity, 1e-12),
                      pstate.net_exposure/max(equity, 1e-12), equity/self._peak-1,
                      pstate.recent_turnover, float(len(pstate.pending_orders))))
        return np.asarray(state, dtype=np.float32)

    def act(self, observation: PortfolioObservation):
        vector = self.observation_vector(observation)
        decision = self._action_counter % self.action_cadence == 0
        self._action_counter += 1
        if not decision:
            return (Hold(),)
        raw, _ = self.model.predict(vector, deterministic=True)
        weights = np.clip(np.asarray(raw, dtype=float), -1, 1)
        gross = float(np.abs(weights).sum())
        if gross > self.gross_budget:
            weights *= self.gross_budget / gross
        return tuple(TargetPosition(s, target_notional=float(weights[i] * observation.portfolio.equity))
                     for i, s in enumerate(self.symbols))


def save_scaler(scaler: TrainingFoldScaler, path: Path) -> None:
    payload = {"feature_names": list(scaler.feature_names), "means": dict(scaler.means),
               "stds": dict(scaler.stds), "fingerprint": scaler.fingerprint}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train_ppo(train: Panel, validation: Panel, output: str | Path, config: PPOConfig):
    """Train/resume PPO; validation reward selects frozen checkpoint."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tb, tf, vb, vf, scaler = prepare_fold_features(train, validation, config.lookback)
    train_env = Monitor(CrossAssetPortfolioEnv(tb, tf, gross_budget=config.gross_budget,
                                               reward=config.reward, action_cadence=config.action_cadence))
    validation_env = Monitor(CrossAssetPortfolioEnv(vb, vf, gross_budget=config.gross_budget,
                                                    reward=config.reward, action_cadence=config.action_cadence))
    checkpoint = output / "latest_model.zip"
    if checkpoint.exists():
        model = PPO.load(checkpoint, env=train_env)
    else:
        model = PPO("MlpPolicy", train_env, seed=config.seed, verbose=0,
                    policy_kwargs={"net_arch": list(config.net_arch)}, n_steps=config.n_steps,
                    batch_size=config.batch_size, learning_rate=config.learning_rate,
                    gamma=config.gamma, gae_lambda=config.gae_lambda, ent_coef=config.ent_coef)
    callback = EvalCallback(validation_env, best_model_save_path=str(output), log_path=str(output),
                            eval_freq=max(config.n_steps, 500), n_eval_episodes=1,
                            deterministic=True)
    model.learn(total_timesteps=config.timesteps, callback=callback, reset_num_timesteps=False)
    model.save(checkpoint)
    best = output / "best_model.zip"
    frozen = PPO.load(best if best.exists() else checkpoint)
    save_scaler(scaler, output / "scaler.json")
    payload = asdict(config)
    (output / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    metadata = {"config_sha256": fingerprint, "scaler_sha256": scaler.fingerprint,
                "price_only": True, "selection_data": "validation", "test_accessed": False}
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frozen, scaler
