from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import queue
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from telemetry_dashboard.artifacts import collect_artifact_snapshot, scan_recent_history
from telemetry_dashboard.registry import REPO_ROOT, FieldSpec, TABS, WorkflowSpec, workflows_for_tab
from telemetry_dashboard.runner import ProcessRunner


CARD_LABELS = {
    "device": "Device",
    "current_fold": "Fold",
    "mae": "MAE",
    "mse": "MSE",
    "mae_simple": "MAE(Simple)",
    "directional": "Directional",
    "coverage_95": "Coverage95",
    "winner_trial": "Winner Trial",
    "winner_holdout_avg_return_pct": "Winner Holdout Avg Return",
    "avg_return_pct": "Avg Return",
    "cumulative_pnl": "Cumulative PnL",
    "avg_pnl": "Avg PnL",
    "win_rate": "Win Rate",
    "max_drawdown": "Max Drawdown",
    "total_trades": "Trades",
    "state_label": "State",
    "shift_probability": "Shift Probability",
    "acceptance_pass": "Acceptance",
}


@dataclass
class RunState:
    spec: WorkflowSpec
    params: dict[str, Any]
    command: list[str]
    started_at: datetime
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=900))
    scalars: dict[str, Any] = field(default_factory=dict)
    series: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    artifact_paths: list[Path] = field(default_factory=list)
    prediction_points: list[dict[str, Any]] = field(default_factory=list)
    prediction_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    probability_snapshot: dict[str, float] = field(default_factory=dict)
    status: str = "Launching"
    phase: str = "Starting"
    returncode: int | None = None
    pid: int | None = None
    interval: tuple[float, float] | None = None


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        magnitude = abs(value)
        if 0 < magnitude < 1:
            return f"{value:.2%}" if magnitude < 0.5 else f"{value:.4f}"
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def _open_path(path: Path) -> None:
    if not path.exists():
        return
    if os.name == "nt":
        os.startfile(str(path))
        return
    subprocess.Popen(["xdg-open", str(path)])


class WorkflowTab(ttk.Frame):
    def __init__(self, master: ttk.Notebook, app: "TelemetryDashboard", tab_name: str):
        super().__init__(master, padding=10, style="Shell.TFrame")
        self.app = app
        self.tab_name = tab_name
        self.specs = workflows_for_tab(tab_name)
        self.current_spec = self.specs[0]
        self.variables: dict[str, tk.Variable] = {}
        self.history_items: list[dict[str, Any]] = []
        self.last_phase_text: str | None = None
        self.last_run_card: tuple[str, str] | None = None
        self.last_card_values: dict[str, Any] | None = None
        self.last_logs_snapshot: list[str] = []
        self.last_chart_signature: Any = None
        self.is_idle_view = False

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.left = ttk.Frame(self, width=296, style="Sidebar.TFrame")
        self.left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        self.left.grid_propagate(False)
        self.right = ttk.Frame(self, style="Shell.TFrame")
        self.right.grid(row=0, column=1, sticky="nsew")
        self.right.columnconfigure(0, weight=1)
        self.right.rowconfigure(2, weight=1)

        selector_frame = ttk.Frame(self.left, style="Sidebar.TFrame")
        selector_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(selector_frame, text="Workflow", style="Section.TLabel").pack(anchor="w")
        self.spec_var = tk.StringVar(value=self.current_spec.label)
        self.spec_select = ttk.Combobox(
            selector_frame,
            textvariable=self.spec_var,
            values=[spec.label for spec in self.specs],
            state="readonly",
        )
        self.spec_select.pack(fill="x", pady=(4, 0), ipady=2)
        self.spec_select.bind("<<ComboboxSelected>>", self._on_spec_change)

        self.form_frame = ttk.LabelFrame(self.left, text="Launch", padding=10, style="Minimal.TLabelframe")
        self.form_frame.pack(fill="x")
        self.fields_frame = ttk.Frame(self.form_frame, style="Sidebar.TFrame")
        self.fields_frame.pack(fill="x")

        self.start_button = ttk.Button(self.form_frame, text=f"Start {tab_name} Run", command=self._start_run)
        self.start_button.pack(fill="x", pady=(10, 0), ipady=4)

        history_frame = ttk.LabelFrame(self.left, text="Recent Outputs", padding=10, style="Minimal.TLabelframe")
        history_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.history_list = tk.Listbox(
            history_frame,
            height=14,
            activestyle="none",
            bg="#fbfbfa",
            fg="#1f2937",
            bd=0,
            highlightthickness=1,
            highlightbackground="#d6d3d1",
            selectbackground="#dbeafe",
            selectforeground="#111827",
        )
        self.history_list.pack(fill="both", expand=True)
        self.history_list.bind("<Double-Button-1>", self._open_selected_history)
        ttk.Button(history_frame, text="Refresh", command=self.refresh_history).pack(fill="x", pady=(8, 0))

        header = ttk.Frame(self.right, style="Shell.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.phase_label = ttk.Label(header, text="Idle", style="Muted.TLabel")
        self.phase_label.grid(row=0, column=0, sticky="w")

        self.cards = ttk.Frame(self.right, style="Shell.TFrame")
        self.cards.grid(row=1, column=0, sticky="ew", pady=(8, 10))
        for idx in range(6):
            self.cards.columnconfigure(idx, weight=1)
        self.run_card = ttk.Frame(self.cards, style="Card.TFrame", padding=10)
        self.run_card.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.run_card_title_var = tk.StringVar(value="Active Run")
        self.run_card_value_var = tk.StringVar(value="Idle")
        self.run_card_detail_var = tk.StringVar(value="No active process")
        ttk.Label(self.run_card, textvariable=self.run_card_title_var, style="Muted.TLabel").pack(anchor="w")
        ttk.Label(self.run_card, textvariable=self.run_card_value_var, style="CardValue.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Label(self.run_card, textvariable=self.run_card_detail_var, style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
        self.card_frames: list[ttk.Frame] = []
        self.card_label_vars: list[tk.StringVar] = []
        self.card_value_vars: list[tk.StringVar] = []
        for idx in range(5):
            card = ttk.Frame(self.cards, style="Card.TFrame", padding=10)
            card.grid(row=0, column=idx + 1, sticky="ew", padx=(0, 8))
            label_var = tk.StringVar(value="")
            value_var = tk.StringVar(value="")
            ttk.Label(card, textvariable=label_var, style="Muted.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=value_var, style="CardValue.TLabel").pack(anchor="w", pady=(4, 0))
            self.card_frames.append(card)
            self.card_label_vars.append(label_var)
            self.card_value_vars.append(value_var)
        self.empty_cards_label = ttk.Label(self.cards, text="No metrics yet.", style="Muted.TLabel")

        content = ttk.Panedwindow(self.right, orient=tk.VERTICAL)
        content.grid(row=2, column=0, sticky="nsew")

        chart_frame = ttk.LabelFrame(content, text="Charts", padding=8, style="Minimal.TLabelframe")
        log_frame = ttk.LabelFrame(content, text="Streaming Terminal", padding=8, style="Minimal.TLabelframe")
        content.add(chart_frame, weight=3)
        content.add(log_frame, weight=2)

        self.figure = Figure(figsize=(9, 4.8), dpi=100)
        self.axis_primary = self.figure.add_subplot(121)
        self.axis_secondary = self.figure.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.log_widget = ScrolledText(
            log_frame,
            height=16,
            wrap="word",
            bg="#111315",
            fg="#e7e5e4",
            insertbackground="#e7e5e4",
            bd=0,
            highlightthickness=1,
            highlightbackground="#292524",
        )
        self.log_widget.pack(fill="both", expand=True)
        self.log_widget.configure(state="disabled")

        self._render_fields()
        self.refresh_history()

    def _on_spec_change(self, *_args: object) -> None:
        label = self.spec_var.get()
        self.current_spec = next(spec for spec in self.specs if spec.label == label)
        self._render_fields()
        self.refresh_history()

    def _clear_frame(self, frame: ttk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _render_fields(self) -> None:
        self.variables.clear()
        self._clear_frame(self.fields_frame)

        for field in self.current_spec.fields:
            self._create_field_widget(self.fields_frame, field)

    def _create_field_widget(self, frame: ttk.Frame, field: FieldSpec) -> None:
        container = ttk.Frame(frame, style="Sidebar.TFrame")
        container.pack(fill="x", pady=(0, 8))
        if field.field_type == "bool":
            variable = tk.BooleanVar(value=bool(field.default))
            widget = ttk.Checkbutton(container, variable=variable, text=field.label, style="Minimal.TCheckbutton")
            widget.pack(anchor="w", padx=(1, 0))
            if field.help_text:
                ttk.Label(container, text=field.help_text, style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
        else:
            container.columnconfigure(1, weight=1)
            ttk.Label(container, text=field.label, style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
            default_value = "" if field.default is None else str(field.default)
            variable = tk.StringVar(value=default_value)
            if field.field_type == "choice":
                widget = ttk.Combobox(container, textvariable=variable, values=list(field.choices), state="readonly")
            else:
                widget = ttk.Entry(container, textvariable=variable)
            widget.grid(row=0, column=1, sticky="ew")
            if field.help_text:
                ttk.Label(container, text=field.help_text, style="Muted.TLabel").grid(row=1, column=1, sticky="w", pady=(2, 0))
        self.variables[field.name] = variable

    def _collect_params(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for field in self.current_spec.fields:
            variable = self.variables[field.name]
            raw = variable.get()
            if field.field_type == "bool":
                value = bool(raw)
            elif field.field_type == "int":
                text = str(raw).strip()
                value = None if text == "" else int(text)
            elif field.field_type == "float":
                text = str(raw).strip()
                value = None if text == "" else float(text)
            else:
                value = str(raw).strip()
            if field.required and (value is None or value == ""):
                raise ValueError(f"{field.label} is required.")
            values[field.name] = value
        return values

    def _start_run(self) -> None:
        try:
            params = self._collect_params()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc), parent=self)
            return
        self.app.start_run(self.current_spec, params)

    def set_run_enabled(self, enabled: bool) -> None:
        self.start_button.configure(state="normal" if enabled else "disabled")

    def refresh_history(self) -> None:
        self.history_items = scan_recent_history(self.current_spec)
        self.history_list.delete(0, tk.END)
        for item in self.history_items:
            stamp = item["updated_at"].strftime("%Y-%m-%d %H:%M")
            self.history_list.insert(tk.END, f"{stamp} | {item['label']} | {item['summary']}")

    def _open_selected_history(self, *_args: object) -> None:
        selection = self.history_list.curselection()
        if not selection:
            return
        item = self.history_items[selection[0]]
        _open_path(Path(item["path"]))

    def render(self, state: RunState | None) -> None:
        if state is None:
            if self.is_idle_view:
                return
            self.phase_label.configure(text="Idle")
            self.last_phase_text = "Idle"
            self._render_run_card(None)
            self._render_cards({})
            self._render_logs([])
            self._draw_empty()
            self.is_idle_view = True
            return
        self.is_idle_view = False
        phase_text = f"{state.status} | {state.phase}"
        if phase_text != self.last_phase_text:
            self.phase_label.configure(text=phase_text)
            self.last_phase_text = phase_text
        self._render_run_card(state)
        self._render_cards(state.scalars)
        self._render_logs(list(state.logs))
        self._draw_charts(state)

    def _render_run_card(self, state: RunState | None) -> None:
        if state is None:
            snapshot = ("Idle", "No active process")
        else:
            status_text = f"{state.spec.label} | {state.status}"
            elapsed = datetime.now() - state.started_at
            pid_text = f"PID {state.pid}" if state.pid is not None else "PID pending"
            snapshot = (status_text, f"{pid_text} | {elapsed.seconds}s")
        if snapshot == self.last_run_card:
            return
        self.run_card_value_var.set(snapshot[0])
        self.run_card_detail_var.set(snapshot[1])
        self.last_run_card = snapshot

    def _render_cards(self, scalars: dict[str, Any]) -> None:
        visible_keys = [key for key in CARD_LABELS if key in scalars][: len(self.card_frames)]
        card_values = {key: scalars[key] for key in visible_keys}
        if card_values == self.last_card_values:
            return
        self.last_card_values = dict(card_values)
        if not visible_keys:
            for card in self.card_frames:
                card.grid_remove()
            self.empty_cards_label.grid(row=0, column=1, sticky="w")
            return
        self.empty_cards_label.grid_remove()
        for idx, key in enumerate(visible_keys):
            self.card_frames[idx].grid()
            self.card_label_vars[idx].set(CARD_LABELS[key])
            self.card_value_vars[idx].set(_format_value(scalars[key]))
        for idx in range(len(visible_keys), len(self.card_frames)):
            self.card_frames[idx].grid_remove()

    def _render_logs(self, lines: list[str]) -> None:
        if lines == self.last_logs_snapshot:
            return
        self.log_widget.configure(state="normal")
        previous = self.last_logs_snapshot
        if previous and len(lines) >= len(previous) and lines[: len(previous)] == previous:
            new_lines = lines[len(previous) :]
            if new_lines:
                prefix = "\n" if previous else ""
                self.log_widget.insert(tk.END, prefix + "\n".join(new_lines))
                self.log_widget.see(tk.END)
        else:
            self.log_widget.delete("1.0", tk.END)
            if lines:
                self.log_widget.insert(tk.END, "\n".join(lines))
                self.log_widget.see(tk.END)
        self.log_widget.configure(state="disabled")
        self.last_logs_snapshot = list(lines)

    def _draw_empty(self) -> None:
        if self.last_chart_signature == "idle":
            return
        self.axis_primary.clear()
        self.axis_secondary.clear()
        self.axis_primary.set_title("No active run")
        self.axis_secondary.set_title("")
        self.canvas.draw_idle()
        self.last_chart_signature = "idle"

    def _draw_charts(self, state: RunState) -> None:
        chart_signature = self._chart_signature(state)
        if chart_signature == self.last_chart_signature:
            return
        self.axis_primary.clear()
        self.axis_secondary.clear()
        self.axis_primary.set_axis_on()
        self.axis_secondary.set_axis_on()
        if self.tab_name == "Training":
            self._draw_training(state)
        elif self.tab_name == "Optimization":
            self._draw_optimization(state)
        elif self.tab_name == "Backtesting":
            self._draw_backtesting(state)
        else:
            self._draw_prediction(state)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.last_chart_signature = chart_signature

    def _chart_signature(self, state: RunState) -> Any:
        series_signature = tuple(
            (name, len(points), points[-1] if points else None)
            for name, points in sorted(state.series.items())
        )
        scalar_signature = tuple(
            (name, state.scalars.get(name))
            for name in ("winner_trial", "winner_holdout_avg_return_pct")
            if name in state.scalars
        )
        prediction_signature = tuple(
            (item.get("label"), item.get("metric"), item.get("value"))
            for item in state.prediction_points
        )
        interval_signature = tuple(sorted(state.prediction_intervals.items()))
        probability_signature = tuple(sorted(state.probability_snapshot.items()))
        return (
            self.tab_name,
            series_signature,
            scalar_signature,
            prediction_signature,
            interval_signature,
            probability_signature,
            state.interval,
        )

    def _draw_training(self, state: RunState) -> None:
        loss_points = state.series.get("loss", [])
        if loss_points:
            self.axis_primary.plot([x for x, _ in loss_points], [y for _, y in loss_points], color="#0f766e", linewidth=2)
        else:
            self.axis_primary.text(0.5, 0.5, "Waiting for loss output", ha="center", va="center", transform=self.axis_primary.transAxes)
        self.axis_primary.set_title("Training Loss")

        metric_keys = [key for key in ("mae", "mse", "mae_simple", "directional", "coverage_95") if key in state.series]
        colors = ["#2563eb", "#f59e0b", "#7c3aed", "#ef4444", "#059669"]
        for key, color in zip(metric_keys, colors):
            points = state.series[key]
            self.axis_secondary.plot([x for x, _ in points], [y for _, y in points], marker="o", label=key, color=color)
        self.axis_secondary.set_title("Fold Metrics")
        if metric_keys:
            self.axis_secondary.legend(loc="best")
        else:
            self.axis_secondary.text(0.5, 0.5, "Fold metrics will appear here", ha="center", va="center", transform=self.axis_secondary.transAxes)

    def _draw_optimization(self, state: RunState) -> None:
        objectives = state.series.get("objective", [])
        if objectives:
            self.axis_primary.plot([x for x, _ in objectives], [y for _, y in objectives], marker="o", linestyle="", color="#2563eb", label="trial")
            best_points = state.series.get("best_objective", [])
            if best_points:
                self.axis_primary.plot([x for x, _ in best_points], [y for _, y in best_points], color="#0f766e", linewidth=2, label="best")
            self.axis_primary.legend(loc="best")
        else:
            self.axis_primary.text(0.5, 0.5, "Optuna trial results will appear here", ha="center", va="center", transform=self.axis_primary.transAxes)
        self.axis_primary.set_title("Objective by Trial")

        winner_value = state.scalars.get("winner_holdout_avg_return_pct")
        winner_trial = state.scalars.get("winner_trial")
        self.axis_secondary.set_title("Selection Summary")
        lines = []
        if winner_trial is not None:
            lines.append(f"Winner trial: {winner_trial}")
        if winner_value is not None:
            lines.append(f"Winner holdout avg return: {_format_value(winner_value)}")
        if not lines:
            lines.append("Winner information will appear after validation.")
        self.axis_secondary.text(0.03, 0.95, "\n".join(lines), va="top", transform=self.axis_secondary.transAxes)
        self.axis_secondary.set_axis_off()

    def _draw_backtesting(self, state: RunState) -> None:
        cumulative = state.series.get("cumulative_pnl") or []
        if not cumulative and "trade_pnl" in state.series:
            running = []
            total = 0.0
            for idx, pnl in state.series["trade_pnl"]:
                total += pnl
                running.append((idx, total))
            cumulative = running
        if cumulative:
            self.axis_primary.plot([x for x, _ in cumulative], [y for _, y in cumulative], color="#0f766e", linewidth=2)
        else:
            self.axis_primary.text(0.5, 0.5, "Backtest PnL will appear here", ha="center", va="center", transform=self.axis_primary.transAxes)
        self.axis_primary.set_title("Cumulative PnL")

        trades = state.series.get("trade_pnl", [])
        if trades:
            colors = ["#059669" if pnl >= 0 else "#dc2626" for _, pnl in trades]
            self.axis_secondary.bar([x for x, _ in trades], [y for _, y in trades], color=colors)
        else:
            self.axis_secondary.text(0.5, 0.5, "Trade-level PnL bars will appear here", ha="center", va="center", transform=self.axis_secondary.transAxes)
        self.axis_secondary.set_title("Per-Trade PnL")

    def _draw_prediction(self, state: RunState) -> None:
        predictions = state.prediction_points
        if predictions:
            labels = [str(item.get("label") or f"Prediction {idx + 1}") for idx, item in enumerate(predictions)]
            values = [float(item["value"]) for item in predictions]
            xs = list(range(len(predictions)))
            self.axis_primary.bar(xs, values, color="#2563eb", width=0.55)
            errors = []
            has_intervals = True
            for label, value in zip(labels, values):
                interval = state.prediction_intervals.get(label)
                if interval is None:
                    has_intervals = False
                    break
                lower, upper = interval
                errors.append((value - lower, upper - value))
            if has_intervals and errors:
                lower_err = [max(0.0, item[0]) for item in errors]
                upper_err = [max(0.0, item[1]) for item in errors]
                self.axis_primary.errorbar(
                    xs,
                    values,
                    yerr=[lower_err, upper_err],
                    fmt="none",
                    ecolor="#0f766e",
                    elinewidth=2,
                    capsize=5,
                )
            self.axis_primary.set_xticks(xs)
            self.axis_primary.set_xticklabels(labels)
        else:
            self.axis_primary.text(0.5, 0.5, "Prediction values will appear here", ha="center", va="center", transform=self.axis_primary.transAxes)
        self.axis_primary.set_title("Latest Forecasts")

        if state.probability_snapshot:
            labels = list(state.probability_snapshot.keys())
            values = [state.probability_snapshot[key] for key in labels]
            self.axis_secondary.bar(labels, values, color="#7c3aed")
            self.axis_secondary.set_ylim(0, 1)
            self.axis_secondary.set_title("State Probabilities")
        elif state.prediction_intervals:
            labels = list(state.prediction_intervals.keys())
            lowers = [state.prediction_intervals[label][0] for label in labels]
            uppers = [state.prediction_intervals[label][1] for label in labels]
            ys = list(range(len(labels)))
            self.axis_secondary.hlines(ys, lowers, uppers, color="#0f766e", linewidth=3)
            self.axis_secondary.scatter(lowers, ys, color="#0f766e", s=24)
            self.axis_secondary.scatter(uppers, ys, color="#0f766e", s=24)
            self.axis_secondary.set_yticks(ys)
            self.axis_secondary.set_yticklabels(labels)
            self.axis_secondary.set_title("Prediction Intervals")
        else:
            lines = []
            if "mean_log_return" in state.scalars:
                lines.append(f"Mean log return: {_format_value(state.scalars['mean_log_return'])}")
            if "log_return_std" in state.scalars:
                lines.append(f"Log-return std: {_format_value(state.scalars['log_return_std'])}")
            if "state_label" in state.scalars:
                lines.append(f"State: {state.scalars['state_label']}")
            if "shift_probability" in state.scalars:
                lines.append(f"Shift probability: {_format_value(state.scalars['shift_probability'])}")
            if not lines:
                lines.append("Prediction intervals or state probabilities will appear here")
            self.axis_secondary.text(0.04, 0.95, "\n".join(lines), va="top", transform=self.axis_secondary.transAxes)
            self.axis_secondary.set_title("Prediction Details")


class TelemetryDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MarketAnalysis Telemetry Dashboard")
        self.geometry("1460x930")
        self.configure(bg="#f5f5f4")
        self._configure_style()

        self.event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.runner = ProcessRunner(self.event_queue)
        self.active_state: RunState | None = None
        self.active_tab_name: str | None = None
        self.last_rendered_tab_name: str | None = None

        shell = ttk.Frame(self, padding=12, style="Shell.TFrame")
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell, style="Shell.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        controls = ttk.Frame(header, style="Shell.TFrame")
        controls.grid(row=0, column=1, sticky="e")
        self.stop_button = ttk.Button(controls, text="Stop Run", command=self.stop_run, state="disabled")
        self.stop_button.grid(row=0, column=0, padx=(0, 8))
        self.open_artifacts_button = ttk.Button(
            controls,
            text="Open Artifact Folder",
            command=self.open_artifact_folder,
            state="disabled",
        )
        self.open_artifacts_button.grid(row=0, column=1)

        self.notebook = ttk.Notebook(shell)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.tabs: dict[str, WorkflowTab] = {}
        for tab_name in TABS:
            tab = WorkflowTab(self.notebook, self, tab_name)
            self.tabs[tab_name] = tab
            self.notebook.add(tab, text=tab_name)

        self.after(150, self._poll_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#f5f5f4", foreground="#1c1917", font=("Segoe UI", 9))
        style.configure("Shell.TFrame", background="#f5f5f4")
        style.configure("Sidebar.TFrame", background="#f5f5f4")
        style.configure("TFrame", background="#f5f5f4")
        style.configure("TLabel", background="#f5f5f4", foreground="#1c1917")
        style.configure("TLabelframe", background="#f5f5f4", bordercolor="#d6d3d1", relief="solid")
        style.configure("TLabelframe.Label", background="#f5f5f4", foreground="#44403c", font=("Segoe UI Semibold", 9))
        style.configure("Minimal.TLabelframe", background="#f5f5f4", bordercolor="#d6d3d1", relief="solid")
        style.configure("Minimal.TLabelframe.Label", background="#f5f5f4", foreground="#44403c", font=("Segoe UI Semibold", 9))
        style.configure("DashboardTitle.TLabel", background="#f5f5f4", foreground="#1c1917", font=("Bahnschrift SemiCondensed", 18))
        style.configure("Title.TLabel", background="#f5f5f4", foreground="#1c1917", font=("Bahnschrift SemiCondensed", 13))
        style.configure("Section.TLabel", background="#f5f5f4", foreground="#57534e", font=("Segoe UI Semibold", 9))
        style.configure("FieldLabel.TLabel", background="#f5f5f4", foreground="#44403c", font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background="#f5f5f4", foreground="#78716c", font=("Segoe UI", 8))
        style.configure("Card.TFrame", background="#fcfcfb", relief="solid", borderwidth=1)
        style.configure("CardValue.TLabel", background="#fcfcfb", foreground="#1c1917", font=("Bahnschrift", 12))
        style.configure("TButton", padding=6, background="#f8fafc", foreground="#111827", bordercolor="#d6d3d1")
        style.map("TButton", background=[("active", "#eff6ff")], bordercolor=[("active", "#93c5fd")])
        style.configure("TEntry", fieldbackground="#fcfcfb", bordercolor="#d6d3d1", lightcolor="#d6d3d1", darkcolor="#d6d3d1", padding=6)
        style.configure("TCombobox", fieldbackground="#fcfcfb", bordercolor="#d6d3d1", lightcolor="#d6d3d1", darkcolor="#d6d3d1", padding=4)
        style.configure("Minimal.TCheckbutton", background="#f5f5f4", foreground="#1c1917")
        style.map("Minimal.TCheckbutton", background=[("active", "#f5f5f4")])
        style.configure("TNotebook", background="#f5f5f4", borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure(
            "TNotebook.Tab",
            background="#ece7e1",
            foreground="#44403c",
            padding=(14, 7),
            borderwidth=0,
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#fcfcfb"), ("active", "#f5f5f4")],
            foreground=[("selected", "#111827"), ("active", "#1f2937")],
        )

    def start_run(self, spec: WorkflowSpec, params: dict[str, Any]) -> None:
        if self.active_state is not None and self.active_state.returncode is None:
            messagebox.showwarning("Run In Progress", "Only one active run is supported at a time.", parent=self)
            return
        command = spec.command(params)
        self.active_state = RunState(spec=spec, params=params, command=command, started_at=datetime.now())
        self.active_tab_name = spec.tab
        self.stop_button.configure(state="normal")
        self.open_artifacts_button.configure(state="disabled")
        for tab in self.tabs.values():
            tab.set_run_enabled(False)
        self.tabs[spec.tab].render(self.active_state)
        self.runner.start(spec.id, command, REPO_ROOT)

    def stop_run(self) -> None:
        if self.active_state is None or self.active_state.returncode is not None:
            return
        self.runner.stop()
        self.active_state.phase = "Stopping"
        self.active_state.status = "Stopping"
        self._render_active_tab()

    def open_artifact_folder(self) -> None:
        if self.active_state is None or not self.active_state.artifact_paths:
            return
        latest = self.active_state.artifact_paths[-1]
        target = latest if latest.is_dir() else latest.parent
        _open_path(target)

    def _poll_events(self) -> None:
        changed = False
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            changed = True
            self._handle_event(event)
        if changed:
            self._render_active_tab()
        self.after(150, self._poll_events)

    def _handle_event(self, event: dict[str, Any]) -> None:
        if self.active_state is None:
            return
        kind = event["kind"]
        if kind == "process_started":
            self.active_state.pid = event["pid"]
            self.active_state.status = "Running"
            self.active_state.phase = "Streaming telemetry"
            return
        if kind == "log":
            self.active_state.logs.append(event["line"])
            return
        if kind == "phase":
            self.active_state.phase = event["value"]
            return
        if kind == "scalar":
            self.active_state.scalars[event["name"]] = event["value"]
            return
        if kind == "series":
            self.active_state.series.setdefault(event["name"], []).append((float(event["x"]), float(event["y"])))
            return
        if kind == "artifact":
            self.active_state.artifact_paths.append(Path(event["path"]))
            self.open_artifacts_button.configure(state="normal")
            return
        if kind == "prediction":
            self.active_state.prediction_points.append(event)
            return
        if kind == "interval":
            interval = (float(event["lower"]), float(event["upper"]))
            self.active_state.interval = interval
            label = str(event.get("label") or f"Prediction {len(self.active_state.prediction_intervals) + 1}")
            self.active_state.prediction_intervals[label] = interval
            return
        if kind == "probabilities":
            self.active_state.probability_snapshot = dict(event["values"])
            return
        if kind == "run_finished":
            self.active_state.returncode = int(event["returncode"])
            self.active_state.status = "Succeeded" if self.active_state.returncode == 0 else "Failed"
            self.active_state.phase = "Completed"
            snapshot = collect_artifact_snapshot(self.active_state.spec, self.active_state.artifact_paths)
            for key, value in snapshot.get("cards", {}).items():
                self.active_state.scalars[key] = value
            for key, points in snapshot.get("series", {}).items():
                if key not in self.active_state.series:
                    self.active_state.series[key] = [(float(x), float(y)) for x, y in points]
            self.stop_button.configure(state="disabled")
            for tab in self.tabs.values():
                tab.set_run_enabled(True)
                tab.refresh_history()
            return

    def _render_active_tab(self) -> None:
        if self.last_rendered_tab_name and self.last_rendered_tab_name != self.active_tab_name:
            self.tabs[self.last_rendered_tab_name].render(None)
        if self.active_tab_name is not None:
            self.tabs[self.active_tab_name].render(self.active_state)
        elif self.last_rendered_tab_name is not None:
            self.tabs[self.last_rendered_tab_name].render(None)
        self.last_rendered_tab_name = self.active_tab_name


def main() -> None:
    app = TelemetryDashboard()
    app.mainloop()
