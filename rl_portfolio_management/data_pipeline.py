"""Leakage-safe market-data snapshot builder for portfolio experiments.

Universe is intentionally fixed and known today. Historical results therefore
have survivorship/selection bias; manifests carry this disclosure explicitly.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

UNIVERSE = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "XOM",
    "JNJ", "WMT", "CAT", "NEE", "AMT",
)
BENCHMARK = "SPY"
SURVIVORSHIP_DISCLOSURE = (
    "Fixed sector-diverse universe selected using information known at snapshot "
    "creation time; historical membership is unavailable, so results have "
    "survivorship and selection bias."
)
CANONICAL_COLUMNS = ("open", "high", "low", "close", "volume", "dividends", "stock_splits")


@dataclass(frozen=True)
class SnapshotConfig:
    start: str
    end: str
    interval: str = "1d"
    symbols: tuple[str, ...] = UNIVERSE
    benchmark: str = BENCHMARK
    auto_adjust: bool = True
    actions: bool = True
    regular_session_only: bool = True
    timezone: str = "America/New_York"

    def __post_init__(self) -> None:
        if self.interval not in {"1d", "1h", "60m"}:
            raise ValueError("interval must be 1d, 1h, or 60m")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be non-empty and unique")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def canonicalize(raw: pd.DataFrame, symbol: str, config: SnapshotConfig) -> pd.DataFrame:
    """Return sorted, unique, session-filtered frame without filling future values."""
    if raw.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS).rename_axis("timestamp")
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        levels = [set(map(str, frame.columns.get_level_values(i))) for i in range(frame.columns.nlevels)]
        candidate = next((i for i, values in enumerate(levels) if symbol in values), None)
        if candidate is not None:
            frame = frame.xs(symbol, axis=1, level=candidate, drop_level=True)
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    frame = frame.rename(columns={"stock_splits": "stock_splits"})
    parsed = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
    if config.interval == "1d":
        # Yahoo daily labels are exchange trading dates, not UTC instants.
        if parsed.tz is None:
            idx = parsed.tz_localize(config.timezone)
        else:
            idx = parsed.tz_convert(config.timezone).normalize()
    else:
        idx = parsed.tz_localize("UTC") if parsed.tz is None else parsed.tz_convert("UTC")
    valid = ~idx.isna()
    frame = frame.loc[valid].copy()
    idx = idx[valid].tz_convert(config.timezone)
    frame.index = idx
    frame.index.name = "timestamp"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if config.regular_session_only and config.interval in {"1h", "60m"}:
        # Yahoo timestamps bars by bar-open. 09:30 through 15:30 are regular-session bars.
        frame = frame.between_time("09:30", "15:30", inclusive="both")
    for column in CANONICAL_COLUMNS:
        if column not in frame:
            frame[column] = 0.0 if column in {"dividends", "stock_splits"} else float("nan")
    frame = frame.loc[:, CANONICAL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return frame


def align_frames(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align to intersection of observed timestamps; never forward/back fill."""
    nonempty = [frame.index for frame in frames.values() if not frame.empty]
    if not nonempty:
        return {symbol: frame.copy() for symbol, frame in frames.items()}
    common = nonempty[0]
    for index in nonempty[1:]:
        common = common.intersection(index)
    common = common.sort_values()
    return {symbol: frame.loc[frame.index.intersection(common)].reindex(common) for symbol, frame in frames.items()}


def download(config: SnapshotConfig) -> dict[str, pd.DataFrame]:
    """Download symbols independently to isolate missing/delisted failures."""
    import yfinance as yf

    frames: dict[str, pd.DataFrame] = {}
    for symbol in (*config.symbols, config.benchmark):
        raw = yf.download(
            symbol, start=config.start, end=config.end, interval=config.interval,
            auto_adjust=config.auto_adjust, actions=config.actions,
            prepost=not config.regular_session_only, progress=False, threads=False,
        )
        frames[symbol] = canonicalize(raw, symbol, config)
    return frames


def frame_fingerprint(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized.index = normalized.index.map(lambda value: value.isoformat())
    payload = normalized.to_csv(index=True, lineterminator="\n", float_format="%.12g").encode()
    return hashlib.sha256(payload).hexdigest()


def persist_snapshot(frames: Mapping[str, pd.DataFrame], config: SnapshotConfig, root: str | Path) -> Path:
    """Persist immutable version directory and return manifest path."""
    root = Path(root)
    config_dict = asdict(config)
    config_dict["symbols"] = list(config.symbols)
    config_hash = sha256_json(config_dict)
    data_hash = sha256_json({s: frame_fingerprint(f) for s, f in sorted(frames.items())})
    snapshot_id = f"{config.interval}_{config.start}_{config.end}_{config_hash[:8]}_{data_hash[:8]}"
    destination = root / snapshot_id
    destination.mkdir(parents=True, exist_ok=True)

    storage: dict[str, dict[str, str]] = {}
    for symbol, frame in sorted(frames.items()):
        parquet = destination / f"{symbol}.parquet"
        try:
            frame.to_parquet(parquet, index=True)
            storage[symbol] = {"format": "parquet", "file": parquet.name}
        except (ImportError, ModuleNotFoundError):
            csv = destination / f"{symbol}.csv"
            frame.to_csv(csv, index=True, date_format="%Y-%m-%dT%H:%M:%S%z")
            storage[symbol] = {"format": "csv", "file": csv.name}

    details = {}
    for symbol, frame in sorted(frames.items()):
        details[symbol] = {
            "rows": len(frame),
            "first_timestamp": None if frame.empty else frame.index.min().isoformat(),
            "last_timestamp": None if frame.empty else frame.index.max().isoformat(),
            "missing_fraction": {c: float(frame[c].isna().mean()) if len(frame) else None for c in frame.columns},
            "content_sha256": frame_fingerprint(frame),
            **storage[symbol],
        }
    manifest = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_dict,
        "config_sha256": config_hash,
        "content_sha256": data_hash,
        "universe_rule": "explicit fixed sector-diverse large-cap US universe",
        "survivorship_disclosure": SURVIVORSHIP_DISCLOSURE,
        "adjustment_semantics": "yfinance auto_adjust OHLC; actions retained separately when supplied",
        "ideal_fill_disclosure": "Yahoo has no reliable historical bid/ask; no spread observations are invented.",
        "alignment": "per-symbol canonical frames; intersection alignment available; no filling",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "yfinance": _package_version("yfinance")},
        "symbols": details,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def build_snapshot(config: SnapshotConfig, root: str | Path, align: bool = False) -> Path:
    frames = download(config)
    return persist_snapshot(align_frames(frames) if align else frames, config, root)

# verified snapshot loading lives below


def load_snapshot(manifest_path: str | Path, verify: bool = True) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load persisted bars and verify manifest, config, and content hashes."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema")
    if verify and sha256_json(manifest["config"]) != manifest.get("config_sha256"):
        raise ValueError("snapshot config hash mismatch")
    frames: dict[str, pd.DataFrame] = {}
    fingerprints: dict[str, str] = {}
    for symbol, details in sorted(manifest["symbols"].items()):
        data_path = (path.parent / details["file"]).resolve()
        if data_path.parent != path.parent:
            raise ValueError(f"snapshot file escapes manifest directory: {symbol}")
        if details["format"] == "parquet":
            frame = pd.read_parquet(data_path)
        elif details["format"] == "csv":
            frame = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
        else:
            raise ValueError(f"unsupported snapshot format for {symbol}")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        frame.index.name = "timestamp"
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        frame = frame.loc[:, CANONICAL_COLUMNS].sort_index()
        fingerprint = frame_fingerprint(frame)
        if verify and fingerprint != details.get("content_sha256"):
            raise ValueError(f"snapshot content hash mismatch: {symbol}")
        if verify and len(frame) != details.get("rows"):
            raise ValueError(f"snapshot row count mismatch: {symbol}")
        frames[symbol] = frame.rename(columns={column: column.title() for column in CANONICAL_COLUMNS})
        fingerprints[symbol] = fingerprint
    if verify and sha256_json(fingerprints) != manifest.get("content_sha256"):
        raise ValueError("snapshot aggregate content hash mismatch")
    return frames, manifest
