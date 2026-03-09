from __future__ import annotations

from pathlib import Path
import re
from typing import Any


ARTIFACT_RE = re.compile(r"^(?P<label>.+?) saved to: (?P<path>.+)$")
RUN_DIR_RE = re.compile(r"^Run directory: (?P<path>.+)$")
ITER_LOSS_RE = re.compile(r"Iter (?P<step>\d+)/(?P<total>\d+) - Loss: (?P<loss>[-+0-9.eE]+)")
FOLD_RE = re.compile(r"Fold (?P<fold>\d+)")
TRAIN_METRICS_RE = re.compile(
    r"MAE(?:\(log\))?: (?P<mae>[-+0-9.eE]+)"
    r"(?: \| MAE\(simple\): (?P<mae_simple>[-+0-9.eE%]+))?"
    r" \| MSE: (?P<mse>[-+0-9.eE]+)"
    r"(?: \| Dir: (?P<directional>[-+0-9.eE%]+))?"
    r"(?: \| Coverage95: (?P<coverage>[-+0-9.eE%]+))?"
)
SUMMARY_RE = re.compile(r"Summary \|.*Folds: (?P<folds>\d+)")
TRADE_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2}).*Pred: (?P<pred>[-+0-9.]+)% \| PnL: (?P<pnl>[-+0-9.]+)"
)
OPTUNA_TRIAL_RE = re.compile(r"Trial (?P<trial>\d+) finished with value: (?P<value>[-+0-9.eE]+)")
WINNER_TRIAL_RE = re.compile(r"Winner trial: (?P<trial>\d+)")
WINNER_VALUE_RE = re.compile(r"Winner .*avg_return_pct: (?P<value>[-+0-9.eE]+)")
PROBABILITIES_RE = re.compile(
    r"p_state_0=(?P<p0>[-+0-9.eE]+), "
    r"p_state_1=(?P<p1>[-+0-9.eE]+), "
    r"p_state_2=(?P<p2>[-+0-9.eE]+), "
    r"p_state_3=(?P<p3>[-+0-9.eE]+)"
)
MEAN_SIMPLE_RE = re.compile(r"Mean simple return: (?P<value>[-+0-9.]+)%")
MEAN_LOG_RE = re.compile(r"Mean log return: (?P<value>[-+0-9.eE]+)")
STD_LOG_RE = re.compile(r"Log-return std: (?P<value>[-+0-9.eE]+)")
VOL_MEAN_RE = re.compile(r"Annualized vol \(mean\): (?P<value>[-+0-9.eE]+)")
VOL_INTERVAL_RE = re.compile(r"95% interval: \[(?P<lower>[-+0-9.eE]+), (?P<upper>[-+0-9.eE]+)\]")
SIMPLE_INTERVAL_RE = re.compile(
    r"95% interval \(simple return\): \[(?P<lower>[-+0-9.]+)%, (?P<upper>[-+0-9.]+)%\]"
)
STATE_RE = re.compile(r"State: (?P<label>.+?) \(id=(?P<state_id>\d+)\)")
SHIFT_PROB_RE = re.compile(r"Shift probability: (?P<value>[-+0-9.eE]+)")
DEVICE_RE = re.compile(r"Using device: (?P<device>\w+)")
ACCEPTANCE_RE = re.compile(r"Acceptance pass: (?P<value>True|False)")
FORECAST_LABEL_RE = re.compile(r"^(?P<label>.+?)\s+5-day forward .+ forecast$")


def _to_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    percent = raw.endswith("%")
    numeric = float(raw.rstrip("%"))
    return numeric / 100.0 if percent else numeric


class LineParser:
    def __init__(self, spec_id: str):
        self.spec_id = spec_id
        self.current_fold: int | None = None
        self.trade_index = 0
        self.trade_wins = 0
        self.trade_cumulative_pnl = 0.0
        self.trade_peak_cumulative_pnl = 0.0
        self.trade_max_drawdown = 0.0
        self.prediction_index = 0
        self.best_objective: float | None = None
        self.current_prediction_label: str | None = None

    def parse(self, line: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = [{"kind": "log", "line": line}]
        stripped = line.strip()
        if not stripped:
            return events

        forecast_label_match = FORECAST_LABEL_RE.match(stripped)
        if forecast_label_match:
            self.current_prediction_label = forecast_label_match.group("label").strip()

        artifact_match = ARTIFACT_RE.match(stripped)
        if artifact_match:
            label = artifact_match.group("label").lower().replace(" ", "_")
            events.append(
                {
                    "kind": "artifact",
                    "label": label,
                    "path": Path(artifact_match.group("path").strip()),
                }
            )

        run_dir_match = RUN_DIR_RE.match(stripped)
        if run_dir_match:
            events.append(
                {
                    "kind": "artifact",
                    "label": "run_directory",
                    "path": Path(run_dir_match.group("path").strip()),
                }
            )

        device_match = DEVICE_RE.match(stripped)
        if device_match:
            events.append({"kind": "scalar", "name": "device", "value": device_match.group("device")})

        if stripped.startswith(("Building ", "Training ", "Running ", "Retraining ", "Validating ")):
            events.append({"kind": "phase", "value": stripped})

        loss_match = ITER_LOSS_RE.search(stripped)
        if loss_match:
            events.append(
                {
                    "kind": "series",
                    "name": "loss",
                    "x": int(loss_match.group("step")),
                    "y": float(loss_match.group("loss")),
                }
            )

        fold_match = FOLD_RE.search(stripped)
        if fold_match:
            self.current_fold = int(fold_match.group("fold"))
            events.append({"kind": "scalar", "name": "current_fold", "value": self.current_fold})

        metrics_match = TRAIN_METRICS_RE.search(stripped)
        if metrics_match and self.current_fold is not None:
            fold = self.current_fold
            for name, group in (
                ("mae", "mae"),
                ("mae_simple", "mae_simple"),
                ("mse", "mse"),
                ("directional", "directional"),
                ("coverage_95", "coverage"),
            ):
                value = _to_float(metrics_match.group(group))
                if value is not None:
                    events.append({"kind": "series", "name": name, "x": fold, "y": value})
                    events.append({"kind": "scalar", "name": name, "value": value})

        summary_match = SUMMARY_RE.search(stripped)
        if summary_match:
            events.append({"kind": "scalar", "name": "folds", "value": int(summary_match.group("folds"))})

        trade_match = TRADE_RE.search(stripped)
        if trade_match:
            self.trade_index += 1
            pnl = float(trade_match.group("pnl"))
            pred = float(trade_match.group("pred")) / 100.0
            self.trade_cumulative_pnl += pnl
            self.trade_peak_cumulative_pnl = max(self.trade_peak_cumulative_pnl, self.trade_cumulative_pnl)
            current_drawdown = self.trade_cumulative_pnl - self.trade_peak_cumulative_pnl
            self.trade_max_drawdown = min(self.trade_max_drawdown, current_drawdown)
            if pnl > 0:
                self.trade_wins += 1
            events.append({"kind": "series", "name": "trade_pnl", "x": self.trade_index, "y": pnl})
            events.append({"kind": "series", "name": "trade_pred", "x": self.trade_index, "y": pred})
            events.append({"kind": "scalar", "name": "total_trades", "value": self.trade_index})
            events.append({"kind": "scalar", "name": "cumulative_pnl", "value": self.trade_cumulative_pnl})
            events.append({"kind": "scalar", "name": "avg_pnl", "value": self.trade_cumulative_pnl / self.trade_index})
            events.append({"kind": "scalar", "name": "win_rate", "value": self.trade_wins / self.trade_index})
            events.append({"kind": "scalar", "name": "max_drawdown", "value": self.trade_max_drawdown})
            if self.trade_peak_cumulative_pnl > 0:
                events.append(
                    {
                        "kind": "scalar",
                        "name": "max_drawdown_pct",
                        "value": abs(self.trade_max_drawdown) / self.trade_peak_cumulative_pnl,
                    }
                )

        trial_match = OPTUNA_TRIAL_RE.search(stripped)
        if trial_match:
            trial = int(trial_match.group("trial"))
            value = float(trial_match.group("value"))
            self.best_objective = value if self.best_objective is None else max(self.best_objective, value)
            events.append({"kind": "series", "name": "objective", "x": trial, "y": value})
            events.append({"kind": "series", "name": "best_objective", "x": trial, "y": self.best_objective})

        winner_match = WINNER_TRIAL_RE.search(stripped)
        if winner_match:
            events.append({"kind": "scalar", "name": "winner_trial", "value": int(winner_match.group("trial"))})

        winner_value_match = WINNER_VALUE_RE.search(stripped)
        if winner_value_match:
            events.append(
                {
                    "kind": "scalar",
                    "name": "winner_holdout_avg_return_pct",
                    "value": float(winner_value_match.group("value")),
                }
            )

        mean_simple_match = MEAN_SIMPLE_RE.search(stripped)
        if mean_simple_match:
            self.prediction_index += 1
            value = float(mean_simple_match.group("value")) / 100.0
            events.append(
                {
                    "kind": "prediction",
                    "metric": "mean_simple_return",
                    "value": value,
                    "index": self.prediction_index,
                    "label": self.current_prediction_label or f"Prediction {self.prediction_index}",
                }
            )
            events.append({"kind": "series", "name": "prediction_value", "x": self.prediction_index, "y": value})

        mean_log_match = MEAN_LOG_RE.search(stripped)
        if mean_log_match:
            events.append({"kind": "scalar", "name": "mean_log_return", "value": float(mean_log_match.group("value"))})

        std_match = STD_LOG_RE.search(stripped)
        if std_match:
            events.append({"kind": "scalar", "name": "log_return_std", "value": float(std_match.group("value"))})

        vol_mean_match = VOL_MEAN_RE.search(stripped)
        if vol_mean_match:
            self.prediction_index += 1
            value = float(vol_mean_match.group("value"))
            events.append(
                {
                    "kind": "prediction",
                    "metric": "annualized_vol",
                    "value": value,
                    "index": self.prediction_index,
                    "label": self.current_prediction_label or f"Prediction {self.prediction_index}",
                }
            )
            events.append({"kind": "series", "name": "prediction_value", "x": self.prediction_index, "y": value})

        vol_interval_match = VOL_INTERVAL_RE.search(stripped)
        if vol_interval_match:
            events.append(
                {
                    "kind": "interval",
                    "name": "prediction_interval",
                    "lower": float(vol_interval_match.group("lower")),
                    "upper": float(vol_interval_match.group("upper")),
                    "label": self.current_prediction_label or f"Prediction {self.prediction_index}",
                }
            )

        simple_interval_match = SIMPLE_INTERVAL_RE.search(stripped)
        if simple_interval_match:
            events.append(
                {
                    "kind": "interval",
                    "name": "prediction_interval",
                    "lower": float(simple_interval_match.group("lower")) / 100.0,
                    "upper": float(simple_interval_match.group("upper")) / 100.0,
                    "label": self.current_prediction_label or f"Prediction {self.prediction_index}",
                }
            )

        state_match = STATE_RE.search(stripped)
        if state_match:
            events.append({"kind": "scalar", "name": "state_label", "value": state_match.group("label")})
            events.append({"kind": "scalar", "name": "state_id", "value": int(state_match.group("state_id"))})

        probabilities_match = PROBABILITIES_RE.search(stripped)
        if probabilities_match:
            events.append(
                {
                    "kind": "probabilities",
                    "label": self.current_prediction_label or "Regime",
                    "values": {
                        "p_state_0": float(probabilities_match.group("p0")),
                        "p_state_1": float(probabilities_match.group("p1")),
                        "p_state_2": float(probabilities_match.group("p2")),
                        "p_state_3": float(probabilities_match.group("p3")),
                    },
                }
            )

        shift_match = SHIFT_PROB_RE.search(stripped)
        if shift_match:
            events.append({"kind": "scalar", "name": "shift_probability", "value": float(shift_match.group("value"))})

        acceptance_match = ACCEPTANCE_RE.search(stripped)
        if acceptance_match:
            events.append({"kind": "scalar", "name": "acceptance_pass", "value": acceptance_match.group("value") == "True"})

        return events
