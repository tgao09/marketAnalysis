from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    field_type: str = "text"
    default: Any = ""
    choices: tuple[str, ...] = ()
    required: bool = False
    help_text: str = ""
    advanced: bool = False


@dataclass(frozen=True)
class WorkflowSpec:
    id: str
    tab: str
    label: str
    model_family: str
    script_path: str
    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)
    history_kind: str = "none"
    build_command: Callable[["WorkflowSpec", dict[str, Any]], list[str]] | None = None

    @property
    def script(self) -> Path:
        return REPO_ROOT / self.script_path

    def command(self, params: dict[str, Any]) -> list[str]:
        if self.build_command is None:
            raise ValueError(f"No command builder configured for {self.id}.")
        return self.build_command(self, params)


def _append_value(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value.strip() == "":
            return
        cmd.extend([flag, value.strip()])
        return
    cmd.extend([flag, str(value)])


def _append_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def _base_command(spec: WorkflowSpec) -> list[str]:
    return [sys.executable, "-u", str(spec.script)]


def _build_training_gbm(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--tickers", params.get("tickers"))
    _append_value(cmd, "--train-window", params.get("train_window"))
    _append_value(cmd, "--feature-set", params.get("feature_set"))
    _append_value(cmd, "--lgbm-param-preset", params.get("lgbm_param_preset"))
    _append_flag(cmd, "--include-time-index", bool(params.get("include_time_index")))
    return cmd


def _build_training_gp_return(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--tickers", params.get("tickers"))
    _append_value(cmd, "--train-window", params.get("train_window"))
    _append_flag(cmd, "--include-time-index", bool(params.get("include_time_index")))
    _append_flag(cmd, "--pca", bool(params.get("pca")))
    return cmd


def _build_training_gp_vol(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--train-window", params.get("train_window"))
    _append_value(cmd, "--test-window", params.get("test_window"))
    _append_value(cmd, "--step-window", params.get("step_window"))
    _append_value(cmd, "--train-iters", params.get("train_iters"))
    _append_value(cmd, "--kernel-mode", params.get("kernel_mode"))
    _append_value(cmd, "--kernel-equation", params.get("kernel_equation"))
    _append_value(cmd, "--kernel-lengthscale", params.get("kernel_lengthscale"))
    _append_value(cmd, "--kernel-period-length", params.get("kernel_period_length"))
    _append_value(cmd, "--kernel-outputscale", params.get("kernel_outputscale"))
    _append_flag(cmd, "--drop-time-index", bool(params.get("drop_time_index")))
    return cmd


def _build_training_hmm(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--train-window", params.get("train_window"))
    _append_value(cmd, "--end", params.get("end"))
    _append_value(cmd, "--n-iter", params.get("n_iter"))
    _append_value(cmd, "--random-state", params.get("random_state"))
    _append_value(cmd, "--min-train-rows", params.get("min_train_rows"))
    _append_value(cmd, "--retrain-cadence", params.get("retrain_cadence"))
    return cmd


def _build_optimization_gbm(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--tickers", params.get("tickers"))
    _append_value(cmd, "--train-window", params.get("train_window"))
    _append_value(cmd, "--test-window", params.get("test_window"))
    _append_value(cmd, "--step-window", params.get("step_window"))
    _append_value(cmd, "--n-trials", params.get("n_trials"))
    _append_value(cmd, "--holdout-top-n", params.get("holdout_top_n"))
    _append_value(cmd, "--notional", params.get("notional"))
    _append_value(cmd, "--drawdown-worsen-limit", params.get("drawdown_worsen_limit"))
    _append_flag(cmd, "--include-time-index", bool(params.get("include_time_index")))
    _append_flag(cmd, "--skip-retrain", bool(params.get("skip_retrain")))
    return cmd


def _build_optimization_gp(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--tickers", params.get("tickers"))
    _append_value(cmd, "--train-window", params.get("train_window"))
    _append_value(cmd, "--test-window", params.get("test_window"))
    _append_value(cmd, "--step-window", params.get("step_window"))
    _append_value(cmd, "--trials", params.get("trials"))
    _append_value(cmd, "--holdout-top-n", params.get("holdout_top_n"))
    _append_value(cmd, "--notional", params.get("notional"))
    _append_value(cmd, "--drawdown-worsen-limit", params.get("drawdown_worsen_limit"))
    _append_flag(cmd, "--pca", bool(params.get("pca")))
    return cmd


def _build_backtest_gbm(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--ticker", params.get("ticker"))
    _append_value(cmd, "--end", params.get("end"))
    _append_value(cmd, "--notional", params.get("notional"))
    _append_value(cmd, "--train-window", params.get("train_window"))
    _append_value(cmd, "--feature-set", params.get("feature_set"))
    _append_value(cmd, "--lgbm-param-preset", params.get("lgbm_param_preset"))
    _append_flag(cmd, "--include-time-index", bool(params.get("include_time_index")))
    return cmd


def _build_backtest_gp(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--ticker", params.get("ticker"))
    _append_value(cmd, "--end", params.get("end"))
    _append_value(cmd, "--notional", params.get("notional"))
    _append_value(cmd, "--train-iters", params.get("train_iters"))
    _append_value(cmd, "--train-window", params.get("train_window"))
    _append_value(cmd, "--threshold", params.get("threshold"))
    _append_flag(cmd, "--include-time-index", bool(params.get("include_time_index")))
    _append_flag(cmd, "--pca", bool(params.get("pca")))
    return cmd


def _build_backtest_gp_vol(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--symbols", params.get("symbols"))
    _append_value(cmd, "--start", params.get("start"))
    _append_value(cmd, "--end", params.get("end"))
    _append_value(cmd, "--threshold", params.get("threshold"))
    _append_value(cmd, "--train-window", params.get("train_window"))
    _append_value(cmd, "--test-window", params.get("test_window"))
    _append_value(cmd, "--step-window", params.get("step_window"))
    _append_value(cmd, "--train-iters", params.get("train_iters"))
    _append_value(cmd, "--iv-window", params.get("iv_window"))
    _append_value(cmd, "--exit-days", params.get("exit_days"))
    _append_value(cmd, "--fees", params.get("fees"))
    _append_value(cmd, "--slippage", params.get("slippage"))
    return cmd


def _build_backtest_hmm(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--end", params.get("end"))
    _append_value(cmd, "--test-years", params.get("test_years"))
    _append_value(cmd, "--train-window", params.get("train_window"))
    _append_value(cmd, "--step-bdays", params.get("step_bdays"))
    _append_value(cmd, "--n-iter", params.get("n_iter"))
    _append_value(cmd, "--random-state", params.get("random_state"))
    _append_value(cmd, "--min-train-rows", params.get("min_train_rows"))
    _append_value(cmd, "--min-state-occupancy", params.get("min_state_occupancy"))
    _append_value(cmd, "--auc-threshold", params.get("auc_threshold"))
    _append_value(cmd, "--vol-ratio-threshold", params.get("vol_ratio_threshold"))
    return cmd


def _build_predict_gbm(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--tickers", params.get("tickers"))
    return cmd


def _build_predict_gp(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--tickers", params.get("tickers"))
    _append_flag(cmd, "--pca", bool(params.get("pca")))
    return cmd


def _build_predict_gp_vol(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    return _base_command(spec)


def _build_predict_hmm(spec: WorkflowSpec, params: dict[str, Any]) -> list[str]:
    cmd = _base_command(spec)
    _append_value(cmd, "--date", params.get("date"))
    _append_value(cmd, "--output-csv", params.get("output_csv"))
    return cmd


TABS = ("Training", "Optimization", "Backtesting", "Prediction")


WORKFLOWS: tuple[WorkflowSpec, ...] = (
    WorkflowSpec(
        id="training_gbm_return",
        tab="Training",
        label="GBM Return",
        model_family="gbm_return",
        script_path="gbm_return/train.py",
        history_kind="training_metrics",
        build_command=_build_training_gbm,
        fields=(
            FieldSpec("tickers", "Tickers", default="AAPL,NVDA", required=True),
            FieldSpec("train_window", "Train Window", default="2y"),
            FieldSpec("feature_set", "Feature Set", field_type="choice", default="F0", choices=("F0", "F1", "F2")),
            FieldSpec("lgbm_param_preset", "LGBM Preset", default="baseline"),
            FieldSpec("include_time_index", "Include Time Index", field_type="bool", default=False),
        ),
    ),
    WorkflowSpec(
        id="training_gp_return",
        tab="Training",
        label="GP Return",
        model_family="gp_return",
        script_path="gp_return/train.py",
        history_kind="training_metrics",
        build_command=_build_training_gp_return,
        fields=(
            FieldSpec("tickers", "Tickers", default="AAPL,NVDA", required=True),
            FieldSpec("train_window", "Train Window", default="2y"),
            FieldSpec("pca", "Enable PCA", field_type="bool", default=False),
            FieldSpec("include_time_index", "Include Time Index", field_type="bool", default=False),
        ),
    ),
    WorkflowSpec(
        id="training_gp_vol",
        tab="Training",
        label="GP Volatility",
        model_family="gp_vol",
        script_path="gp_vol/train.py",
        history_kind="training_metrics",
        build_command=_build_training_gp_vol,
        fields=(
            FieldSpec("train_window", "Train Window", default="2y"),
            FieldSpec("test_window", "Test Window", field_type="choice", default="1m", choices=("1m", "2m")),
            FieldSpec("step_window", "Step Window", field_type="choice", default="1m", choices=("1m", "2m")),
            FieldSpec("train_iters", "Train Iterations", field_type="int", default=200),
            FieldSpec("kernel_mode", "Kernel Mode", field_type="choice", default="default", choices=("default", "custom")),
            FieldSpec("kernel_equation", "Kernel Equation", default="1*2*4"),
            FieldSpec("kernel_lengthscale", "Kernel Lengthscale", field_type="float", default=""),
            FieldSpec("kernel_period_length", "Kernel Period", field_type="float", default=""),
            FieldSpec("kernel_outputscale", "Kernel Outputscale", field_type="float", default=""),
            FieldSpec("drop_time_index", "Drop Time Index", field_type="bool", default=False),
        ),
    ),
    WorkflowSpec(
        id="training_hmm_regime",
        tab="Training",
        label="HMM Regime",
        model_family="hmm_regime",
        script_path="hmm_regime/train.py",
        history_kind="hmm_training",
        build_command=_build_training_hmm,
        fields=(
            FieldSpec("train_window", "Train Window", default="3y"),
            FieldSpec("end", "End Date", default=""),
            FieldSpec("n_iter", "Iterations", field_type="int", default=250),
            FieldSpec("retrain_cadence", "Retrain Cadence", default="1m"),
            FieldSpec("random_state", "Random State", field_type="int", default=42),
            FieldSpec("min_train_rows", "Min Train Rows", field_type="int", default=252),
        ),
    ),
    WorkflowSpec(
        id="optimization_gbm_return",
        tab="Optimization",
        label="GBM Return",
        model_family="gbm_return",
        script_path="gbm_return/optimize_performance.py",
        history_kind="optimization_report",
        build_command=_build_optimization_gbm,
        fields=(
            FieldSpec("tickers", "Tickers", default="AAPL,NVDA,AMZN,KO", required=True),
            FieldSpec("train_window", "Train Window", default="2y"),
            FieldSpec("test_window", "Test Window", default="1m"),
            FieldSpec("step_window", "Step Window", default="1m"),
            FieldSpec("n_trials", "Trials", field_type="int", default=300),
            FieldSpec("holdout_top_n", "Holdout Top N", field_type="int", default=15),
            FieldSpec("notional", "Notional", field_type="float", default=10000),
            FieldSpec("drawdown_worsen_limit", "Drawdown Worsen Limit", field_type="float", default=0.10),
            FieldSpec("include_time_index", "Include Time Index", field_type="bool", default=False),
            FieldSpec("skip_retrain", "Skip Retrain", field_type="bool", default=False),
        ),
    ),
    WorkflowSpec(
        id="optimization_gp_return",
        tab="Optimization",
        label="GP Return",
        model_family="gp_return",
        script_path="gp_return/optimize_performance.py",
        history_kind="optimization_report",
        build_command=_build_optimization_gp,
        fields=(
            FieldSpec("tickers", "Tickers", default="AAPL,NVDA,AMZN,KO", required=True),
            FieldSpec("train_window", "Train Window", default="2y"),
            FieldSpec("test_window", "Test Window", default="1m"),
            FieldSpec("step_window", "Step Window", default="1m"),
            FieldSpec("trials", "Trials", field_type="int", default=120),
            FieldSpec("holdout_top_n", "Holdout Top N", field_type="int", default=20),
            FieldSpec("notional", "Notional", field_type="float", default=10000),
            FieldSpec("drawdown_worsen_limit", "Drawdown Worsen Limit", field_type="float", default=0.10),
            FieldSpec("pca", "Enable PCA", field_type="bool", default=False),
        ),
    ),
    WorkflowSpec(
        id="backtesting_gbm_return",
        tab="Backtesting",
        label="GBM Return",
        model_family="gbm_return",
        script_path="gbm_return/backtest_walk_forward.py",
        history_kind="backtest_summary",
        build_command=_build_backtest_gbm,
        fields=(
            FieldSpec("ticker", "Ticker", default="AAPL", required=True),
            FieldSpec("end", "End Date", default=""),
            FieldSpec("train_window", "Train Window", default="2y"),
            FieldSpec("notional", "Notional", field_type="float", default=10000),
            FieldSpec("feature_set", "Feature Set", field_type="choice", default="F0", choices=("F0", "F1", "F2")),
            FieldSpec("lgbm_param_preset", "LGBM Preset", default="baseline"),
            FieldSpec("include_time_index", "Include Time Index", field_type="bool", default=False),
        ),
    ),
    WorkflowSpec(
        id="backtesting_gp_return",
        tab="Backtesting",
        label="GP Return",
        model_family="gp_return",
        script_path="gp_return/backtest_walk_forward.py",
        history_kind="backtest_summary",
        build_command=_build_backtest_gp,
        fields=(
            FieldSpec("ticker", "Ticker", default="AAPL", required=True),
            FieldSpec("end", "End Date", default=""),
            FieldSpec("train_window", "Train Window", default="2y"),
            FieldSpec("train_iters", "Train Iterations", field_type="int", default=140),
            FieldSpec("threshold", "Threshold", field_type="float", default=0.01),
            FieldSpec("notional", "Notional", field_type="float", default=10000),
            FieldSpec("pca", "Enable PCA", field_type="bool", default=False),
            FieldSpec("include_time_index", "Include Time Index", field_type="bool", default=False),
        ),
    ),
    WorkflowSpec(
        id="backtesting_gp_vol",
        tab="Backtesting",
        label="GP Volatility Options Proxy",
        model_family="gp_vol",
        script_path="gp_vol/backtest_options_proxy.py",
        history_kind="backtest_summary",
        build_command=_build_backtest_gp_vol,
        fields=(
            FieldSpec("symbols", "Symbols", default="XLK,PLTR,NVDA", required=True),
            FieldSpec("start", "Start Date", default=""),
            FieldSpec("end", "End Date", default=""),
            FieldSpec("threshold", "Threshold", field_type="float", default=0.02),
            FieldSpec("train_window", "Train Window", default="2y"),
            FieldSpec("test_window", "Test Window", default="1m"),
            FieldSpec("step_window", "Step Window", default="1m"),
            FieldSpec("train_iters", "Train Iterations", field_type="int", default=200),
            FieldSpec("iv_window", "IV Window", field_type="int", default=20),
            FieldSpec("exit_days", "Exit Days", field_type="int", default=5),
            FieldSpec("fees", "Fees", field_type="float", default=0.0),
            FieldSpec("slippage", "Slippage", field_type="float", default=0.0),
        ),
    ),
    WorkflowSpec(
        id="backtesting_hmm_regime",
        tab="Backtesting",
        label="HMM Regime",
        model_family="hmm_regime",
        script_path="hmm_regime/backtest_walk_forward.py",
        history_kind="hmm_backtest",
        build_command=_build_backtest_hmm,
        fields=(
            FieldSpec("end", "End Date", default=""),
            FieldSpec("test_years", "Test Years", field_type="int", default=2),
            FieldSpec("train_window", "Train Window", default="3y"),
            FieldSpec("step_bdays", "Step BDays", field_type="int", default=21),
            FieldSpec("n_iter", "Iterations", field_type="int", default=250),
            FieldSpec("random_state", "Random State", field_type="int", default=42),
            FieldSpec("min_train_rows", "Min Train Rows", field_type="int", default=252),
            FieldSpec("min_state_occupancy", "Min State Occupancy", field_type="float", default=0.08),
            FieldSpec("auc_threshold", "AUC Threshold", field_type="float", default=0.55),
            FieldSpec("vol_ratio_threshold", "Vol Ratio Threshold", field_type="float", default=1.10),
        ),
    ),
    WorkflowSpec(
        id="prediction_gbm_return",
        tab="Prediction",
        label="GBM Return",
        model_family="gbm_return",
        script_path="gbm_return/predict.py",
        build_command=_build_predict_gbm,
        fields=(FieldSpec("tickers", "Tickers", default="AAPL,NVDA", required=True),),
    ),
    WorkflowSpec(
        id="prediction_gp_return",
        tab="Prediction",
        label="GP Return",
        model_family="gp_return",
        script_path="gp_return/predict.py",
        build_command=_build_predict_gp,
        fields=(
            FieldSpec("tickers", "Tickers", default="AAPL,NVDA", required=True),
            FieldSpec("pca", "Enable PCA", field_type="bool", default=False),
        ),
    ),
    WorkflowSpec(
        id="prediction_gp_vol",
        tab="Prediction",
        label="GP Volatility",
        model_family="gp_vol",
        script_path="gp_vol/predict.py",
        build_command=_build_predict_gp_vol,
    ),
    WorkflowSpec(
        id="prediction_hmm_regime",
        tab="Prediction",
        label="HMM Regime",
        model_family="hmm_regime",
        script_path="hmm_regime/predict.py",
        build_command=_build_predict_hmm,
        fields=(
            FieldSpec("date", "As Of Date", default=""),
            FieldSpec("output_csv", "Output CSV", default=""),
        ),
    ),
)


def workflows_for_tab(tab: str) -> list[WorkflowSpec]:
    return [spec for spec in WORKFLOWS if spec.tab == tab]


def workflow_by_id(spec_id: str) -> WorkflowSpec:
    for spec in WORKFLOWS:
        if spec.id == spec_id:
            return spec
    raise KeyError(spec_id)
