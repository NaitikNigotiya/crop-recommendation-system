# Dataset

Place the CSV here as `data/Crop_recommendation.csv`.

**Suggested source:** Crop Recommendation Dataset on Kaggle
<https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset>
(a free Kaggle account is needed to download).

**Expected columns:** `N, P, K, temperature, humidity, ph, rainfall, label`

Commonly reported size is about 2,200 rows and 22 crop classes, but verify this on the file you
actually download. `python src/analysis.py` writes the real numbers to
`reports/data_quality_report.json`.

## Record these for your report

| Item | Value |
|---|---|
| Source URL | https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset |
| Download date | October 2026 |
| License / usage terms | Publicly available educational dataset on Kaggle |
| SHA-256 Hash | `54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e` |
| Samples / Classes | 2,200 rows, 22 balanced crop classes |
| Modifications made | none (the code never edits the original file) |

The training script also stores the file's SHA-256 hash in `models/model_metadata.json`, which
identifies the exact dataset version a model was trained on.

The CSV is git-ignored by default. Check the license before committing or redistributing it.
