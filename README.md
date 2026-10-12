# Crop Recommendation System Using Statistical Machine Learning

A prototype that predicts a crop label from soil and climate measurements (N, P, K, temperature,
humidity, pH, rainfall). It covers the full SML workflow: data validation, statistical analysis,
model comparison, a saved pipeline and a Streamlit app.

> **Limitation:** this is an educational decision-support prototype. A prediction is only as good as
> the dataset and its labels, and it is not a substitute for local agronomic advice.

## 1. Setup

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Get the dataset

Download `Crop_recommendation.csv` from Kaggle and save it as `data/Crop_recommendation.csv`.
Details and the checklist of what to record (source, download date, license) are in
[`data/README.md`](data/README.md).

## 3. Run

From the project root:

```bash
python src/analysis.py      # data quality, descriptive stats, figures, hypothesis tests -> reports/
python src/train_model.py   # CV comparison, final test evaluation, saves models/crop_pipeline.joblib
streamlit run app.py        # interactive prediction app
python -m pytest            # tests (use a small synthetic dataset, not your real one)
```

Useful options for `train_model.py`: `--seed 42`, `--test-size 0.2`, `--cv-folds 5`, `--data PATH`.

## 4. Project Structure & What each file does

```text
crop-recommendation-system/
├── app.py                      # Interactive Streamlit web interface
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & execution guide
├── data/
│   ├── Crop_recommendation.csv # Clean dataset (2,200 rows, 22 crops)
│   └── README.md               # Dataset source, hash and license notes
├── src/
│   ├── data_validation.py      # Validation rules for inputs and raw data
│   ├── analysis.py             # EDA, stats & hypothesis tests -> reports/
│   ├── evaluation.py           # Cross-validation & test evaluation metrics
│   ├── train_model.py          # ML pipeline training, comparison & export
│   └── prediction.py           # Inference engine with range warnings
├── models/
│   ├── crop_pipeline.joblib    # Trained Random Forest pipeline
│   └── model_metadata.json     # Training configuration, ranges & metrics
├── reports/
│   ├── analysis_summary.md     # Statistical hypothesis test report
│   ├── test_metrics.json       # Measured held-out test evaluation scores
│   ├── *.csv                   # Metrics, comparisons, confusion matrices
│   └── figures/                # 12 EDA distribution & confusion matrix plots
└── tests/
    └── test_pipeline.py        # Automated testing suite with synthetic data
```

| Path | Purpose |
|---|---|
| `src/data_validation.py` | Column/type/missing/duplicate checks, sanity ranges, validation of app input |
| `src/analysis.py` | Descriptive statistics, figures, correlation, ANOVA and Kruskal-Wallis with Holm correction |
| `src/evaluation.py` | Cross-validated model comparison, test-set metrics, confusion matrix |
| `src/train_model.py` | Split, train, compare, select, evaluate once, save pipeline and metadata |
| `src/prediction.py` | Load the saved pipeline, predict one row, training-range warnings |
| `app.py` | Streamlit interface |
| `tests/test_pipeline.py` | Checks from the testing plan (see below) |

Outputs (created when you run the scripts):

- `reports/data_quality_report.json`, `descriptive_statistics.csv`, `correlation_matrix.csv`,
  `hypothesis_tests.csv`, `analysis_summary.md`, `figures/*.png`
- `reports/model_comparison.csv`, `classification_report.csv`, `confusion_matrix.csv`,
  `test_metrics.json`, `figures/confusion_matrix.png`
- `models/crop_pipeline.joblib` and `models/model_metadata.json` (features, classes, seed, dataset
  SHA-256, training ranges, scikit-learn version)

## 5. Method in brief (useful for the viva)

1. **Validation:** required columns are checked; features are made numeric; rows with missing values
   are dropped and counted; duplicates and out-of-range rows are reported but kept.
2. **Statistics:** mean, median, variance, SD, quartiles, IQR, skewness per feature; class counts;
   histograms, boxplots, correlation heatmap, feature-by-crop boxplots. For each feature, ANOVA and
   Kruskal-Wallis test whether crop groups differ (H0: no difference), with Holm correction across
   the seven features, eta-squared / epsilon-squared effect sizes, and Levene's test for the equal-
   variance assumption. A small p-value shows a difference in the data, not a cause.
3. **Models:** Dummy baseline, Logistic Regression and KNN (both standardized inside the pipeline),
   Decision Tree, Random Forest. No hyperparameter search is done in this version; defaults are used
   (Random Forest: 300 trees). This is an easy extension (`GridSearchCV` on the training data only).
4. **Protocol:** stratified 80/20 split with a fixed seed; stratified 5-fold CV on the training data
   only; selection by mean CV macro-F1 (baseline excluded); the test set is touched once, for the
   selected model. The saved pipeline is that same fitted model, so reported metrics match it.
5. **Metrics:** accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrix, and
   the most-confused class pairs for error review.
6. **App:** text inputs validated for numeric type and sanity range (never clamped), prediction with
   top-5 probabilities, a warning when an input is outside the training range, and a clear message if
   the model file is missing or unloadable.

EDA runs on the full dataset but is descriptive only: nothing learned from it feeds into the models,
so it cannot leak into the test evaluation.

## 6. Testing plan coverage

| Plan item | Where |
|---|---|
| CSV missing / wrong path | `test_missing_csv_gives_clear_error` |
| Unexpected or missing columns | `test_missing_column_detected` |
| Missing values, duplicates, non-numeric cells | `test_missing_values_and_duplicates_reported`, `test_non_numeric_cells_become_missing` |
| Invalid / out-of-range app inputs | `test_invalid_input_rejected_not_clamped`, `test_missing_key_rejected` |
| Out-of-training-range inputs | `test_out_of_training_range_warning` |
| Model file missing or incompatible | `test_missing_model_file`, `test_corrupt_model_file` |
| Reproducibility with fixed seed | `test_split_reproducible_with_seed`, `test_training_is_reproducible` |
| No train/test leakage | `test_split_is_stratified_and_disjoint` (plus scaling inside Pipelines) |
| Feature order consistency | `test_saved_pipeline_reloads_and_predicts` |
| Prediction on a correctly shaped row | `test_saved_pipeline_reloads_and_predicts` |

## 7. Definition of done: status

- [x] Dataset source and usage terms documented (in `data/README.md`)
- [x] Validation and EDA are reproducible (`src/analysis.py`)
- [x] Statistical methods explained and applied with assumptions noted
- [x] Multiple models compared against a baseline
- [x] Test set kept separate from model selection
- [x] Metrics and confusion matrix generated from actual runs (`src/train_model.py`)
- [x] Fitted pipeline saved and reloadable
- [x] App validates inputs and returns predictions
- [x] Limitations visible in the app and documented here
- [x] README includes setup and run commands
- [x] Report and viva notes match measured results (`reports/test_metrics.json`)

Do not quote any accuracy figure until you have run the scripts on the real dataset and read it from
`reports/test_metrics.json`.

## 8. Future work

Regional validation data, season and location features, permutation-importance explanations,
hyperparameter search on training data, and external validation on data not used in development.
