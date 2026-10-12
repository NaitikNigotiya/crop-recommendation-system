"""Data-quality checks, descriptive statistics, figures and hypothesis tests.

Run from the project root:  python src/analysis.py

Everything produced here is descriptive. No preprocessing parameters are learned
from this step, so it cannot leak information into model evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from data_validation import FEATURES, TARGET, load_data, validate_dataframe

ROOT = Path(__file__).resolve().parents[1]


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjustment for multiple comparisons."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for f in FEATURES:
        s = df[f]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows[f] = {
            "count": int(s.count()),
            "mean": s.mean(),
            "median": s.median(),
            "variance": s.var(),
            "std": s.std(),
            "min": s.min(),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "max": s.max(),
            "skewness": s.skew(),
        }
    return pd.DataFrame(rows).T


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """For each feature, test whether its distribution differs across crop labels.

    H0: all crop groups have the same mean (ANOVA) / same distribution (Kruskal-Wallis).
    H1: at least one group differs.
    Holm adjustment is applied across the features within each test family.
    """
    groups_by_feature = {
        f: [g[f].to_numpy() for _, g in df.groupby(TARGET)] for f in FEATURES
    }
    n = len(df)
    rows = []
    for f in FEATURES:
        groups = groups_by_feature[f]
        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw = stats.kruskal(*groups)
        levene_w, p_levene = stats.levene(*groups, center="median")
        grand_mean = df[f].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((df[f] - grand_mean) ** 2).sum()
        rows.append(
            {
                "feature": f,
                "n_groups": len(groups),
                "anova_F": f_stat,
                "anova_p": p_anova,
                "eta_squared": ss_between / ss_total,
                "kruskal_H": h_stat,
                "kruskal_p": p_kw,
                "epsilon_squared": h_stat / (n - 1),
                "levene_W": levene_w,
                "levene_p": p_levene,
            }
        )
    table = pd.DataFrame(rows)
    table["anova_p_holm"] = holm_adjust(table["anova_p"])
    table["kruskal_p_holm"] = holm_adjust(table["kruskal_p"])
    return table


def make_figures(df: pd.DataFrame, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved = []

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(figures_dir / name, dpi=150)
        plt.close(fig)
        saved.append(name)

    # 1. class counts
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(counts))))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#4c9f70")
    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Crop label")
    ax.set_title("Class counts")
    _save(fig, "class_counts.png")

    # 2. histograms and 3. boxplots (overall)
    for kind, name in (("hist", "feature_histograms.png"), ("box", "feature_boxplots.png")):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        for ax, f in zip(axes.ravel(), FEATURES):
            if kind == "hist":
                sns.histplot(df[f], kde=True, ax=ax, color="#4c72b0")
            else:
                sns.boxplot(y=df[f], ax=ax, color="#dd8452")
            ax.set_title(f)
        for ax in axes.ravel()[len(FEATURES):]:
            ax.axis("off")
        _save(fig, name)

    # 4. correlation heatmap
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson correlation of input features")
    _save(fig, "correlation_heatmap.png")

    # 5. feature distributions by crop
    order = sorted(df[TARGET].unique())
    for f in FEATURES:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(order))))
        sns.boxplot(data=df, x=f, y=TARGET, order=order, ax=ax, color="#8172b3", fliersize=2)
        ax.set_title(f"{f} by crop")
        _save(fig, f"by_crop_{f}.png")
    return saved


def write_summary(path: Path, quality: dict, tests: pd.DataFrame, corr: pd.DataFrame) -> None:
    """A short Markdown note you can adapt for the report (facts only, from this run)."""
    pairs = (
        corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        .stack()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(3)
    )
    lines = [
        "# Analysis summary (generated by src/analysis.py)",
        "",
        "## Data quality",
        f"- Rows loaded: {quality['rows_loaded']}; rows after cleaning: {quality['rows_after_cleaning']} "
        f"(dropped for missing values: {quality['rows_dropped_missing']})",
        f"- Duplicate rows (kept): {quality['duplicate_rows_kept']}",
        f"- Classes: {quality['n_classes']}; max/min class-count ratio: "
        f"{quality['class_imbalance_ratio_max_over_min']:.2f}",
        f"- Rows outside configured sanity ranges (kept): {quality['rows_outside_sanity_ranges_kept'] or 'none'}",
        "",
        "## Strongest pairwise linear correlations among inputs",
        *[f"- {a} vs {b}: r = {v:.2f}" for (a, b), v in pairs.items()],
        "",
        "Correlation shows association, not causation, and a weak linear correlation does not mean "
        "a feature is useless for prediction.",
        "",
        "## Group-difference tests (feature vs crop label)",
        "H0: the crop groups do not differ on the feature. H1: at least one group differs.",
        "p-values are Holm-adjusted across the seven features within each test family.",
        "",
        "| feature | ANOVA F | ANOVA p (Holm) | eta^2 | Kruskal H | Kruskal p (Holm) | epsilon^2 | Levene p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in tests.itertuples():
        lines.append(
            f"| {r.feature} | {r.anova_F:.2f} | {r.anova_p_holm:.3g} | {r.eta_squared:.3f} | "
            f"{r.kruskal_H:.2f} | {r.kruskal_p_holm:.3g} | {r.epsilon_squared:.3f} | {r.levene_p:.3g} |"
        )
    lines += [
        "",
        "Assumptions and cautions: ANOVA assumes roughly normal groups with similar variances "
        "(a small Levene p suggests unequal variances, so lean on Kruskal-Wallis). With large "
        "samples almost any difference is 'significant', so look at the effect sizes. This dataset "
        "is observational or synthetic, so these tests do not establish what causes a crop to grow.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(data_path: Path, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    df, quality = validate_dataframe(load_data(data_path))

    (reports_dir / "data_quality_report.json").write_text(json.dumps(quality, indent=2))
    pd.Series(quality["class_counts"], name="count").rename_axis("label").to_csv(
        reports_dir / "class_counts.csv"
    )

    descriptive_statistics(df).to_csv(reports_dir / "descriptive_statistics.csv")
    corr = df[FEATURES].corr()
    corr.to_csv(reports_dir / "correlation_matrix.csv")
    tests = hypothesis_tests(df)
    tests.to_csv(reports_dir / "hypothesis_tests.csv", index=False)
    figures = make_figures(df, reports_dir / "figures")
    write_summary(reports_dir / "analysis_summary.md", quality, tests, corr)

    print(f"Rows: {quality['rows_after_cleaning']} | classes: {quality['n_classes']}")
    print(f"Saved tables, {len(figures)} figures and analysis_summary.md to {reports_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Crop_recommendation.csv")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    main(args.data, args.reports_dir)
