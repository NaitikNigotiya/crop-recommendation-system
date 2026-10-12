"""Dataset and user-input validation for the crop recommendation project.

Nothing here silently modifies data: dataset checks *report* problems, and
user-input checks return error messages instead of clamping values.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
EXPECTED_COLUMNS = FEATURES + [TARGET]

# Display labels. Units are only stated where the project blueprint says they are
# "commonly" reported; the dataset documentation does not establish them.
FEATURE_LABELS = {
    "N": "Nitrogen (N), dataset units",
    "P": "Phosphorus (P), dataset units",
    "K": "Potassium (K), dataset units",
    "temperature": "Temperature (commonly °C)",
    "humidity": "Relative humidity (commonly %)",
    "ph": "Soil pH",
    "rainfall": "Rainfall, dataset units",
}

# Configurable sanity bounds used to (a) reject impossible user input and
# (b) flag suspicious dataset rows. They are deliberately wide and are NOT
# agronomic recommendations. Adjust them if your dataset documentation says otherwise.
INPUT_RANGES = {
    "N": (0.0, 300.0),
    "P": (0.0, 300.0),
    "K": (0.0, 400.0),
    "temperature": (-10.0, 60.0),
    "humidity": (0.0, 100.0),
    "ph": (0.0, 14.0),
    "rainfall": (0.0, 1000.0),
}


class DataValidationError(ValueError):
    """Raised when the dataset cannot be used as-is (wrong columns, no usable rows...)."""


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the CSV, with a clear error if the file is not where we expect it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download Crop_recommendation.csv "
            "(see data/README.md) and place it there."
        )
    return pd.read_csv(path)


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Check the dataset and return ``(clean_df, report)``.

    Cleaning policy (documented in the report):
      * columns are checked against the expected names (extra columns are ignored);
      * feature columns are converted to numbers; unparseable cells become missing;
      * rows with a missing feature or label are dropped and counted;
      * duplicate rows and out-of-range values are REPORTED but NOT removed.
    """
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Missing expected column(s): {missing_cols}. Found columns: {list(df.columns)}"
        )

    report: dict = {
        "rows_loaded": int(len(df)),
        "original_dtypes": {c: str(df[c].dtype) for c in EXPECTED_COLUMNS},
        "extra_columns_ignored": [c for c in df.columns if c not in EXPECTED_COLUMNS],
    }

    work = df[EXPECTED_COLUMNS].copy()

    non_numeric_cells = {}
    for col in FEATURES:
        converted = pd.to_numeric(work[col], errors="coerce")
        bad = int((converted.isna() & work[col].notna()).sum())
        if bad:
            non_numeric_cells[col] = bad
        work[col] = converted
    report["non_numeric_cells_set_to_missing"] = non_numeric_cells

    work[TARGET] = work[TARGET].astype("string").str.strip()
    work.loc[work[TARGET] == "", TARGET] = pd.NA

    report["missing_values"] = {c: int(work[c].isna().sum()) for c in EXPECTED_COLUMNS}

    before = len(work)
    work = work.dropna().reset_index(drop=True)
    work[TARGET] = work[TARGET].astype(str)
    report["rows_dropped_missing"] = int(before - len(work))
    report["rows_after_cleaning"] = int(len(work))
    if work.empty:
        raise DataValidationError("No usable rows remain after removing missing values.")

    report["duplicate_rows_kept"] = int(work.duplicated().sum())

    out_of_range = {}
    for col, (lo, hi) in INPUT_RANGES.items():
        n_bad = int(((work[col] < lo) | (work[col] > hi)).sum())
        if n_bad:
            out_of_range[col] = n_bad
    report["rows_outside_sanity_ranges_kept"] = out_of_range

    counts = work[TARGET].value_counts()
    report["n_classes"] = int(counts.size)
    report["class_counts"] = {str(k): int(v) for k, v in counts.sort_index().items()}
    report["class_imbalance_ratio_max_over_min"] = float(counts.max() / counts.min())

    return work, report


def validate_inputs(raw: dict) -> tuple[dict, list[str]]:
    """Validate raw user input (strings or numbers). Returns ``(values, errors)``.

    Values are never clamped or altered: anything invalid produces a message.
    """
    values: dict[str, float] = {}
    errors: list[str] = []
    for feature in FEATURES:
        label = FEATURE_LABELS[feature]
        entry = raw.get(feature)
        if entry is None or (isinstance(entry, str) and not entry.strip()):
            errors.append(f"{label}: a value is required.")
            continue
        try:
            number = float(entry)
        except (TypeError, ValueError):
            errors.append(f"{label}: '{entry}' is not a number.")
            continue
        if not math.isfinite(number):
            errors.append(f"{label}: value must be a finite number.")
            continue
        lo, hi = INPUT_RANGES[feature]
        if not lo <= number <= hi:
            errors.append(f"{label}: {number:g} is outside the accepted range {lo:g} to {hi:g}.")
            continue
        values[feature] = number
    return values, errors
