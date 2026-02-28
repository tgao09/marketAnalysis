import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BASE_LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "n_estimators": 400,
    "seed": 42,
    "verbosity": -1,
}

LGBM_PARAM_PRESETS = {
    "baseline": {},
    "stability": {
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "n_estimators": 600,
        "learning_rate": 0.03,
    },
    "expressive": {
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "n_estimators": 1000,
        "learning_rate": 0.03,
    },
}

FEATURE_SET_F0 = "F0"
FEATURE_SET_F1 = "F1"
FEATURE_SET_F2 = "F2"
FEATURE_SET_CHOICES = (FEATURE_SET_F0, FEATURE_SET_F1, FEATURE_SET_F2)


def _load_json_dict(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def resolve_lgbm_params(
    preset_name: str = "baseline",
    params_json: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if preset_name not in LGBM_PARAM_PRESETS:
        valid = ", ".join(sorted(LGBM_PARAM_PRESETS))
        raise ValueError(f"Unknown lgbm preset '{preset_name}'. Valid presets: {valid}")

    resolved = dict(BASE_LGBM_PARAMS)
    resolved.update(LGBM_PARAM_PRESETS[preset_name])
    if params_json:
        resolved.update(_load_json_dict(params_json))
    if overrides:
        resolved.update(overrides)
    return resolved


def resolve_feature_drops(
    feature_set: str,
    feature_set_file: str | Path | None,
) -> list[str]:
    if feature_set not in FEATURE_SET_CHOICES:
        valid = ", ".join(FEATURE_SET_CHOICES)
        raise ValueError(f"Unknown feature set '{feature_set}'. Valid values: {valid}")
    if feature_set == FEATURE_SET_F0:
        return []
    if not feature_set_file:
        raise ValueError(f"feature_set={feature_set} requires --feature-set-file.")

    payload = _load_json_dict(feature_set_file)
    drops = payload["feature_sets"][feature_set]["drop_features"]
    return [item.strip() for item in drops if item.strip()]


def apply_feature_set(
    feature_cols: list[str],
    feature_set: str,
    feature_set_file: str | Path | None,
) -> tuple[list[str], list[str]]:
    drop_features = resolve_feature_drops(feature_set, feature_set_file)
    if not drop_features:
        return list(feature_cols), []
    drop_lookup = set(drop_features)
    selected = [col for col in feature_cols if col not in drop_lookup]
    missing = [col for col in drop_features if col not in feature_cols]
    if not selected:
        raise ValueError(
            f"Applying feature_set={feature_set} dropped all features. "
            f"Drops={drop_features}"
        )
    return selected, missing


def write_feature_set_file(
    output_path: str | Path,
    f1_drop_features: list[str],
    f2_drop_features: list[str],
    metadata: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "feature_sets": {
            FEATURE_SET_F1: {"drop_features": list(f1_drop_features)},
            FEATURE_SET_F2: {"drop_features": list(f2_drop_features)},
        },
    }
    if metadata:
        payload["metadata"] = metadata
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
