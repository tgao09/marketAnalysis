# Telemetry Dashboard

Local Tkinter dashboard for launching market-analysis workflows, streaming live logs, parsing telemetry from stdout, and opening the latest artifacts.

## Purpose

The dashboard is a thin orchestration layer over the repo's training, optimization, backtesting, and prediction scripts. It gives you:

- A GUI launcher for supported workflows
- Live terminal output in one place
- Parsed metrics rendered as cards and charts
- Recent artifact history per workflow
- One-click access to the latest output folder

## Launch

```powershell
python telemetry_dashboard.py
```

The root launcher forwards into `telemetry_dashboard.app.main()`.

## Requirements

- Python 3.11+ is the safe assumption.
- Install repo dependencies before launching:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dashboard depends on:

- `tkinter` for the desktop UI
- `matplotlib` for charts
- The rest of the repo's model dependencies because it launches the underlying scripts directly

## Dashboard structure

The UI is organized into four tabs:

- `Training`
- `Optimization`
- `Backtesting`
- `Prediction`

Supported workflow families currently exposed in the dashboard:

- GBM Return
- GP Return
- GP Volatility
- HMM Regime

Each workflow is defined in `telemetry_dashboard/registry.py` as a `WorkflowSpec` with:

- The script path to run
- The tab it belongs to
- The form fields shown in the sidebar
- A command builder that converts UI input into CLI flags
- The artifact history type used by the recent-output panel

## How it works

### Command generation

The dashboard does not call internal Python APIs from the modeling code. It builds subprocess commands and runs the scripts exactly as if you launched them from the terminal.

Command definitions live in `telemetry_dashboard/registry.py`.

### Process execution

`telemetry_dashboard/runner.py` starts one active subprocess at a time with:

- Unbuffered Python output
- `stdout` and `stderr` merged into a single stream
- A background thread that reads output line by line

That line stream is turned into dashboard events and pushed onto a queue for the UI thread to render.

### Telemetry parsing

`telemetry_dashboard/parsers.py` extracts structured signals from plain-text logs, including:

- Training loss
- Fold metrics such as MAE, MSE, directional accuracy, and coverage
- Optuna trial objectives and running best objective
- Prediction means and intervals
- HMM state labels and regime probabilities
- Artifact paths announced by scripts
- Acceptance flags and summary fields

This means dashboard compatibility depends on log format stability. If a script's printed output changes, parser rules may need to change too.

### Artifact history

`telemetry_dashboard/artifacts.py` scans known artifact patterns and builds the "Recent Outputs" panel. It also loads snapshots from generated JSON and CSV files so finished runs can populate cards and charts even after the process exits.

Current artifact patterns include:

- `gbm_return/artifacts/*/regular/metrics.json`
- `gp_return/artifacts/*/*/metrics.json`
- `gp_vol/artifacts/metrics.json`
- `hmm_regime/artifacts/market/diagnostics.json`
- `gbm_return/artifacts/optimization/*/final_report.json`
- `gp_return/artifacts/optuna_runs/*/final_report.json`
- `gbm_return/artifacts/*/regular/gbm_return_summary.json`
- `gp_return/artifacts/*/*/gp_return_summary.json`
- `gp_vol/artifacts/variance_proxy_summary.json`
- `hmm_regime/artifacts/market/walk_forward_summary.json`

If output filenames or folder layouts change, update the artifact scanner and tests.

## UI behavior

The main window provides:

- A left sidebar with workflow selection and parameter inputs
- A recent-output list for the selected workflow
- Status cards for active-run and summary metrics
- Two charts that update during execution
- A scrolling terminal pane with raw logs
- Global controls to stop the run and open the latest artifact folder

Only one active run is supported at a time.

## Files

- `telemetry_dashboard.py`: root launcher
- `telemetry_dashboard/app.py`: Tkinter app, rendering, layout, and event loop
- `telemetry_dashboard/registry.py`: workflow definitions and command builders
- `telemetry_dashboard/runner.py`: subprocess management and streaming output
- `telemetry_dashboard/parsers.py`: stdout-to-event parsing
- `telemetry_dashboard/artifacts.py`: artifact discovery and snapshot loading

## Testing

Relevant tests are under `tests/`:

- `test_dashboard_registry.py`
- `test_dashboard_parsers.py`
- `test_dashboard_artifacts.py`

Run them with:

```powershell
python -m pytest tests
```

Or:

```powershell
python -m unittest discover -s tests
```

## Maintenance notes

- Keep CLI flags in `telemetry_dashboard/registry.py` aligned with the underlying scripts.
- Keep parser regexes aligned with the exact log lines emitted by those scripts.
- Keep artifact path patterns aligned with actual saved outputs.
- If you add a new dashboard workflow, update the registry first, then add parser coverage and artifact discovery as needed.
