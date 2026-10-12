"""Loading the saved pipeline and making predictions (used by the app and the tests)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from data_validation import FEATURES

MODEL_FILE = "crop_pipeline.joblib"
METADATA_FILE = "model_metadata.json"


class ModelNotFoundError(FileNotFoundError):
    """The fitted pipeline file does not exist (training has not been run yet)."""


class ModelLoadError(RuntimeError):
    """The pipeline file exists but could not be loaded (corrupt or incompatible version)."""


def load_model(models_dir: str | Path):
    path = Path(models_dir) / MODEL_FILE
    if not path.is_file():
        raise ModelNotFoundError(
            f"Model file '{path}' not found. Run `python src/train_model.py` first."
        )
    try:
        return joblib.load(path)
    except Exception as exc:  # corrupt file, scikit-learn version mismatch, ...
        raise ModelLoadError(
            f"Could not load '{path}' ({type(exc).__name__}: {exc}). "
            "Re-run `python src/train_model.py` with the current environment."
        ) from exc


def load_metadata(models_dir: str | Path) -> dict:
    """Return the training metadata, or an empty dict if it is unavailable."""
    path = Path(models_dir) / METADATA_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def predict_row(pipeline, values: dict, features: list[str] | None = None):
    """Predict one row. Returns ``(label, probabilities)``.

    ``probabilities`` is a Series sorted high to low, or None if the model has no
    ``predict_proba``. The input row is built in the training feature order.
    """
    features = features or FEATURES
    row = pd.DataFrame([[values[f] for f in features]], columns=features)
    label = str(pipeline.predict(row)[0])
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(row)[0]
        probabilities = pd.Series(proba, index=[str(c) for c in pipeline.classes_]).sort_values(
            ascending=False
        )
    return label, probabilities


def out_of_training_range(values: dict, metadata: dict) -> list[str]:
    """Messages for inputs outside the min/max seen in the training data."""
    ranges = metadata.get("training_feature_ranges", {})
    messages = []
    for feature, (lo, hi) in ranges.items():
        if feature in values and not lo <= values[feature] <= hi:
            messages.append(
                f"{feature} = {values[feature]:g} is outside the range seen in training "
                f"({lo:g} to {hi:g}); the prediction may be unreliable."
            )
    return messages
