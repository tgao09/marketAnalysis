import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from hmm_regime.train import (
    ARTIFACT_DIR_DEFAULT,
    FEATURE_COLUMNS,
    build_market_dataset,
    build_state_output,
    compute_dataset_start,
    compute_filtered_canonical_probs,
    compute_shift_probability,
    apply_scaler,
    load_model_blob,
    remap_transition_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict latest market regime from trained HMM artifacts.")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR_DEFAULT))
    parser.add_argument("--date", default=None, help="As-of date YYYY-MM-DD (default: latest available).")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path for the latest row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    blob = load_model_blob(artifact_dir)

    asof_date = pd.Timestamp(args.date).normalize() if args.date else pd.Timestamp.today().normalize()
    train_window = blob["train_window"]
    start_date = compute_dataset_start(asof_date, train_window, test_years=0)
    dataset = build_market_dataset(start_date, asof_date)

    features = dataset[FEATURE_COLUMNS].dropna()
    features = features.loc[:asof_date]

    scaled = apply_scaler(features, blob["scaler"])
    model = blob["model"]
    canonical_to_raw = [int(x) for x in blob["canonical_to_raw"]]

    canonical_probs = compute_filtered_canonical_probs(model, scaled.values, canonical_to_raw)
    canonical_transition = remap_transition_matrix(model.transmat_, canonical_to_raw)
    shift_probability = compute_shift_probability(canonical_probs, canonical_transition)

    states = build_state_output(
        index=features.index,
        canonical_probs=canonical_probs,
        shift_probability=shift_probability,
        asof_date=features.index.max(),
    )
    latest = states.iloc[-1]

    print(f"As of: {pd.Timestamp(latest['date']).date()}")
    print(f"State: {latest['state_label']} (id={int(latest['state_id'])})")
    print(
        "Probabilities: "
        f"p_state_0={latest['p_state_0']:.4f}, "
        f"p_state_1={latest['p_state_1']:.4f}, "
        f"p_state_2={latest['p_state_2']:.4f}, "
        f"p_state_3={latest['p_state_3']:.4f}"
    )
    print(f"Shift probability: {latest['shift_prob']:.4f}")

    output_path = Path(args.output_csv) if args.output_csv else artifact_dir / "latest_state.csv"
    latest_frame = pd.DataFrame([latest])
    latest_frame.to_csv(output_path, index=False)
    print(f"Latest state row saved to: {output_path}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
