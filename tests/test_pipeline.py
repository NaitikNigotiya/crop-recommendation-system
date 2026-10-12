"""Tests for the crop recommendation pipeline.

These use a small SYNTHETIC dataset generated on the fly purely to exercise the code.
It has nothing to do with real crops, and no metric from it belongs in the report.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import holm_adjust  # noqa: E402
from data_validation import (  # noqa: E402
    FEATURES,
    TARGET,
    DataValidationError,
    load_data,
    validate_dataframe,
    validate_inputs,
)
from prediction import (  # noqa: E402
    ModelLoadError,
    ModelNotFoundError,
    load_metadata,
    load_model,
    out_of_training_range,
    predict_row,
)
from train_model import run, split_data  # noqa: E402


def make_synthetic(n_per_class=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = {
        "alpha": [20, 20, 20, 15, 40, 5.5, 50],
        "beta": [80, 40, 40, 25, 70, 6.5, 120],
        "gamma": [120, 60, 80, 30, 85, 7.0, 220],
        "delta": [40, 100, 150, 20, 60, 6.0, 90],
    }
    frames = []
    for label, mean in centers.items():
        data = rng.normal(mean, [6, 6, 6, 2, 5, 0.3, 15], size=(n_per_class, len(FEATURES)))
        frame = pd.DataFrame(np.clip(data, 0.1, None), columns=FEATURES)
        frame[TARGET] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    base = tmp_path_factory.mktemp("run")
    data_path = base / "synthetic.csv"
    make_synthetic().to_csv(data_path, index=False)
    out = run(data_path, base / "models", base / "reports", seed=42)
    return base, out


# ---- data validation -------------------------------------------------------
def test_missing_csv_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(tmp_path / "nope.csv")


def test_missing_column_detected():
    df = make_synthetic().drop(columns=["ph"])
    with pytest.raises(DataValidationError, match="ph"):
        validate_dataframe(df)


def test_missing_values_and_duplicates_reported():
    df = make_synthetic()
    df.loc[0, "N"] = np.nan
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    clean, report = validate_dataframe(df)
    assert report["rows_dropped_missing"] == 1
    assert report["duplicate_rows_kept"] >= 1
    assert len(clean) == len(df) - 1


def test_non_numeric_cells_become_missing():
    df = make_synthetic().astype({"N": object})
    df.loc[3, "N"] = "abc"
    _, report = validate_dataframe(df)
    assert report["non_numeric_cells_set_to_missing"] == {"N": 1}


# ---- user input validation -------------------------------------------------
GOOD = {"N": "90", "P": "42", "K": "43", "temperature": "21", "humidity": "82", "ph": "6.5", "rainfall": "200"}


def test_valid_input_passes():
    values, errors = validate_inputs(GOOD)
    assert errors == [] and values["ph"] == 6.5


@pytest.mark.parametrize(
    "field,bad",
    [("N", "abc"), ("ph", "20"), ("humidity", "-5"), ("rainfall", ""), ("K", "nan"), ("P", "inf")],
)
def test_invalid_input_rejected_not_clamped(field, bad):
    values, errors = validate_inputs({**GOOD, field: bad})
    assert len(errors) == 1 and field not in values


def test_missing_key_rejected():
    raw = dict(GOOD)
    raw.pop("K")
    assert validate_inputs(raw)[1]


# ---- statistics ------------------------------------------------------------
def test_holm_adjust_matches_hand_calculation():
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


# ---- training protocol -----------------------------------------------------
def test_split_is_stratified_and_disjoint():
    df = make_synthetic()
    X_train, X_test, y_train, y_test = split_data(df, 0.2, 42)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(df)
    assert y_test.value_counts().nunique() == 1  # equal classes stay equal when stratified


def test_split_reproducible_with_seed():
    df = make_synthetic()
    a = split_data(df, 0.2, 42)[1].index
    b = split_data(df, 0.2, 42)[1].index
    assert list(a) == list(b)


def test_training_outputs_exist_and_beat_baseline(trained):
    base, out = trained
    for name in ("model_comparison.csv", "classification_report.csv", "confusion_matrix.csv", "test_metrics.json"):
        assert (base / "reports" / name).is_file()
    assert (base / "reports" / "figures" / "confusion_matrix.png").is_file()
    assert (base / "models" / "crop_pipeline.joblib").is_file()
    m = out["metrics"]
    assert m["test_accuracy"] > m["baseline_test_accuracy"]
    assert m["selected_model"] != "Dummy (most frequent)"


def test_training_is_reproducible(tmp_path):
    data_path = tmp_path / "d.csv"
    make_synthetic().to_csv(data_path, index=False)
    a = run(data_path, tmp_path / "m1", tmp_path / "r1", seed=7)["comparison"]
    b = run(data_path, tmp_path / "m2", tmp_path / "r2", seed=7)["comparison"]
    pd.testing.assert_frame_equal(a, b)


# ---- prediction ------------------------------------------------------------
def test_saved_pipeline_reloads_and_predicts(trained):
    base, _ = trained
    pipeline = load_model(base / "models")
    metadata = load_metadata(base / "models")
    assert metadata["features"] == FEATURES  # consistent feature order
    values, errors = validate_inputs({"N": "80", "P": "40", "K": "40", "temperature": "25", "humidity": "70", "ph": "6.5", "rainfall": "120"})
    assert not errors
    label, probs = predict_row(pipeline, values, metadata["features"])
    assert label in metadata["classes"]
    assert probs is not None and abs(probs.sum() - 1) < 1e-6


def test_out_of_training_range_warning(trained):
    base, _ = trained
    metadata = load_metadata(base / "models")
    values = {f: 50.0 for f in FEATURES}
    values["rainfall"] = 999.0
    assert any("rainfall" in msg for msg in out_of_training_range(values, metadata))


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path)


def test_corrupt_model_file(tmp_path):
    (tmp_path / "crop_pipeline.joblib").write_bytes(b"not a model")
    with pytest.raises(ModelLoadError):
        load_model(tmp_path)
