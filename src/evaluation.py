"""Model comparison and final evaluation helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def compare_models(models: dict, X_train, y_train, n_splits: int, seed: int) -> pd.DataFrame:
    """Stratified k-fold cross-validation on the TRAINING data only."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=1,
        )
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "cv_accuracy_std": scores["test_accuracy"].std(),
                "cv_macro_f1_mean": scores["test_macro_f1"].mean(),
                "cv_macro_f1_std": scores["test_macro_f1"].std(),
                "cv_folds": n_splits,
            }
        )
    table = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    return table.reset_index(drop=True)


def evaluate_on_test(pipeline, X_test, y_test) -> dict:
    """Evaluate a fitted pipeline on the held-out test set (call this once)."""
    labels = sorted(str(c) for c in pipeline.classes_)
    predictions = pipeline.predict(X_test)
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "classification_report": pd.DataFrame(report).T,
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels
        ),
    }


def save_confusion_matrix_figure(cm: pd.DataFrame, path: str | Path, title: str) -> None:
    n = len(cm)
    size = max(6, 0.45 * n + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        cm,
        annot=n <= 30,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"size": 6},
    )
    ax.set_xlabel("Predicted crop")
    ax.set_ylabel("Actual crop")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def most_confused_pairs(cm: pd.DataFrame, top: int = 5) -> list[dict]:
    """Largest off-diagonal confusion-matrix cells (for the error review)."""
    values = cm.to_numpy().copy()
    np.fill_diagonal(values, 0)
    flat = [
        (int(values[i, j]), cm.index[i], cm.columns[j])
        for i in range(values.shape[0])
        for j in range(values.shape[1])
        if values[i, j] > 0
    ]
    flat.sort(reverse=True)
    return [{"count": c, "actual": a, "predicted": p} for c, a, p in flat[:top]]
