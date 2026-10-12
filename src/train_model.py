"""Train, compare, evaluate and save the crop classifier.

Run from the project root:  python src/train_model.py

Protocol
  1. Stratified train/test split (default 80/20, fixed seed).
  2. Candidate models are scikit-learn Pipelines, so scaling is fitted on training folds only.
  3. Models are compared with stratified cross-validation on the TRAINING data.
  4. The best model by mean CV macro-F1 (dummy baseline excluded) is chosen.
  5. The chosen model is fitted on the training set and evaluated ONCE on the test set.
  6. That same fitted pipeline is saved, so reported metrics describe the saved model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_validation import FEATURES, TARGET, DataValidationError, load_data, validate_dataframe
from evaluation import (
    compare_models,
    evaluate_on_test,
    most_confused_pairs,
    save_confusion_matrix_figure,
)
from prediction import METADATA_FILE, MODEL_FILE

ROOT = Path(__file__).resolve().parents[1]
BASELINE_NAME = "Dummy (most frequent)"


def build_candidates(seed: int) -> dict[str, Pipeline]:
    return {
        BASELINE_NAME: Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
        ),
        "K-Nearest Neighbors": Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]
        ),
        "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(random_state=seed))]),
        "Random Forest": Pipeline(
            [("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))]
        ),
    }


def split_data(df: pd.DataFrame, test_size: float, seed: int):
    X, y = df[FEATURES], df[TARGET]
    try:
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    except ValueError as exc:
        raise DataValidationError(
            f"Stratified split failed (a class may have too few rows): {exc}"
        ) from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    data_path: Path,
    models_dir: Path,
    reports_dir: Path,
    seed: int = 42,
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    df, quality = validate_dataframe(load_data(data_path))
    X_train, X_test, y_train, y_test = split_data(df, test_size, seed)

    n_splits = min(cv_folds, int(y_train.value_counts().min()))
    if n_splits < 2:
        raise DataValidationError("Too few rows in the smallest class for cross-validation.")

    candidates = build_candidates(seed)
    comparison = compare_models(candidates, X_train, y_train, n_splits, seed)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    contenders = comparison[comparison["model"] != BASELINE_NAME]
    best_name = contenders.iloc[0]["model"]  # table is sorted by cv_macro_f1_mean
    best = clone(candidates[best_name]).fit(X_train, y_train)

    # The held-out test set is used only here, once.
    result = evaluate_on_test(best, X_test, y_test)
    result["classification_report"].to_csv(reports_dir / "classification_report.csv")
    result["confusion_matrix"].to_csv(reports_dir / "confusion_matrix.csv")
    save_confusion_matrix_figure(
        result["confusion_matrix"],
        reports_dir / "figures" / "confusion_matrix.png",
        f"{best_name}: confusion matrix (test set)",
    )

    baseline = clone(candidates[BASELINE_NAME]).fit(X_train, y_train)
    baseline_eval = evaluate_on_test(baseline, X_test, y_test)

    test_metrics = {
        "selected_model": best_name,
        "selection_criterion": "highest mean cross-validated macro-F1 on training data (baseline excluded)",
        "test_accuracy": result["accuracy"],
        "test_macro_f1": result["macro_f1"],
        "test_weighted_f1": result["weighted_f1"],
        "baseline_test_accuracy": baseline_eval["accuracy"],
        "baseline_test_macro_f1": baseline_eval["macro_f1"],
        "most_confused_pairs": most_confused_pairs(result["confusion_matrix"]),
    }
    (reports_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    joblib.dump(best, models_dir / MODEL_FILE)
    metadata = {
        **test_metrics,
        "features": FEATURES,
        "target": TARGET,
        "classes": [str(c) for c in best.classes_],
        "training_feature_ranges": {
            f: [float(X_train[f].min()), float(X_train[f].max())] for f in FEATURES
        },
        "random_seed": seed,
        "test_size": test_size,
        "cv_folds": n_splits,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dataset_file": Path(data_path).name,
        "dataset_sha256": _sha256(Path(data_path)),
        "data_quality": quality,
        "scikit_learn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (models_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))

    return {"comparison": comparison, "metrics": test_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Crop_recommendation.csv")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    out = run(args.data, args.models_dir, args.reports_dir, args.seed, args.test_size, args.cv_folds)
    pd.set_option("display.width", 140)
    print("\nCross-validation comparison (training data):")
    print(out["comparison"].round(4).to_string(index=False))
    m = out["metrics"]
    print(f"\nSelected: {m['selected_model']}")
    print(f"Test accuracy {m['test_accuracy']:.4f} | macro-F1 {m['test_macro_f1']:.4f} "
          f"| baseline accuracy {m['baseline_test_accuracy']:.4f}")
    print(f"Saved model to {args.models_dir / MODEL_FILE}")
