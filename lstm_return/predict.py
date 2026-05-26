import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from gp_return.train import resolve_device
from lstm_return.train import (
    ARTIFACT_DIR_DEFAULT,
    ARTIFACT_VARIANT_REGULAR,
    build_latest_sequence,
    load_artifacts,
    predict_sequences,
)


def prompt_tickers() -> list[str]:
    raw = input("Tickers to predict (comma/space separated): ").strip()
    if not raw:
        return []
    tokens = [token.strip().upper() for token in re.split(r"[,\s]+", raw) if token.strip()]
    seen: set[str] = set()
    tickers: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            tickers.append(token)
    return tickers


def predict_next_window(artifact_dir: Path, ticker: str, device: torch.device) -> dict[str, float | pd.Timestamp]:
    model_blob, config, model = load_artifacts(artifact_dir, device=device)
    asof_date, latest_sequence = build_latest_sequence(
        ticker=ticker,
        config=config,
        model_blob=model_blob,
        end_date=pd.Timestamp.today().normalize(),
    )
    preds = predict_sequences(model, latest_sequence[np.newaxis, ...], device=device)
    mean_log = float(preds[0])
    mean_simple = math.exp(mean_log) - 1.0
    action = "long" if mean_log >= 0.0 else "short"
    return {
        "asof_date": asof_date,
        "mean_log": mean_log,
        "mean_simple": mean_simple,
        "action": action,
    }


def main() -> None:
    tickers = prompt_tickers()
    if not tickers:
        print("No ticker provided. Exiting.")
        return

    device = resolve_device()
    print(f"Using device: {device.type}")

    for idx, ticker in enumerate(tickers):
        if idx:
            print("")
        artifact_dir = ARTIFACT_DIR_DEFAULT / ticker / ARTIFACT_VARIANT_REGULAR
        result = predict_next_window(artifact_dir, ticker=ticker, device=device)
        print(f"{ticker} 5-day forward log-return forecast")
        print(f"As of: {pd.Timestamp(result['asof_date']).date()} (last trading day with full features)")
        print(f"Mean log return: {result['mean_log']:.6f}")
        print(f"Mean simple return: {result['mean_simple']:.2%}")
        print(f"Action: {result['action']}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
