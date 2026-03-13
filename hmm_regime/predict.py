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
    apply_scaler,
    compute_dataset_start,
    compute_filtered_state_probs,
    compute_shift_probability,
    load_model_blob,
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
    feature_columns = list(blob.get("feature_columns", FEATURE_COLUMNS))
    start_date = compute_dataset_start(asof_date, train_window, test_years=0)
    dataset = build_market_dataset(start_date, asof_date)

    features = dataset[feature_columns].dropna()
    features = features.loc[:asof_date]

    scaled = apply_scaler(features, blob["scaler"], feature_columns=feature_columns)
    model = blob["model"]
    state_probs = compute_filtered_state_probs(model, scaled.values)
    transition_matrix = np.asarray(model.transmat_, dtype=float)
    shift_probability = compute_shift_probability(state_probs, transition_matrix)

    states = build_state_output(
        index=features.index,
        state_probs=state_probs,
        shift_probability=shift_probability,
        asof_date=features.index.max(),
        stress_state_id=int(blob.get("stress_state_id", 0)),
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
