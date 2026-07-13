# Heart Disease AI-MED

Heart Disease AI-MED is an academic machine-learning project for binary heart-disease prediction using the processed UCI Heart Disease cohorts. The project uses the Cleveland cohort for model development, keeps a held-out Cleveland test split for internal final evaluation, and uses the Hungarian and Switzerland cohorts as external-site tests.

The repository snapshot includes raw and processed data, four analysis notebooks, saved model artifacts, generated tables, and generated figures. It is intended for research and reproducibility, not clinical deployment.

> Safety note: this project is not a medical device, is not prospectively validated, and must not be used to make patient-care decisions.

## What This Project Does

- Converts the original multi-class UCI disease label into a binary target.
- Splits Cleveland into train, validation, and held-out test partitions with stratification.
- Treats Hungarian and Switzerland as external cohorts for transportability testing.
- Compares Logistic Regression, XGBoost with Optuna tuning, and a compact MLP.
- Evaluates discrimination, threshold metrics, calibration, cross-site shift, CORAL domain adaptation, and SHAP explanations.

## Repository Layout

```text
.
|-- data/
|   |-- raw/                  # UCI processed source files
|   `-- processed/            # Canonical split CSV files and split metadata
|-- notebooks/
|   |-- 01_dataset.ipynb      # Data loading, target conversion, split design
|   |-- 02_eda.ipynb          # Missingness, balance, distributions, shift analysis
|   |-- 03_modeling.ipynb     # Feature engineering, tuning, model comparison
|   `-- 04_evaluation.ipynb   # Final metrics, calibration, DeLong, CORAL, SHAP
|-- outputs/
|   |-- figures/              # EDA, ROC/PR, reliability, and SHAP figures
|   |-- models/               # Saved Phase 3 model artifacts
|   `-- tables/               # Metrics, metadata, tuning, and analysis tables
|-- requirements.txt
`-- README.md
```

Current snapshot note: this repository does not contain a `src/` package or a `reports/` directory. A few notebook cells and saved-artifact references still mention an earlier script-oriented layout such as `src/03_modeling.py`. The files actually present in this checkout are the notebooks, data, models, figures, and tables listed above.

## Quick Start

Create a Python environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Open the notebooks:

```bash
jupyter notebook
```

Recommended execution and reading order:

1. `notebooks/01_dataset.ipynb`
2. `notebooks/02_eda.ipynb`
3. `notebooks/03_modeling.ipynb`
4. `notebooks/04_evaluation.ipynb`

The committed artifacts under `outputs/` are the canonical generated outputs for this repository snapshot. Re-running the notebooks may require resolving the Phase 4 model-loading caveat described in the reproducibility notes below.

## Data

The raw files are the processed UCI Heart Disease files:

- `data/raw/processed.cleveland.data`
- `data/raw/processed.hungarian.data`
- `data/raw/processed.switzerland.data`

The 13 raw predictors are:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal
```

The original UCI target is mapped to a binary label:

- `0`: no heart disease
- `1`: any original target value greater than `0`

The processed UCI files do not include explicit patient identifiers. The project preserves `source_row_index` for row-level checks and confirms no overlap across Cleveland partitions, but true patient-level independence cannot be proven from these files alone.

## Dataset Splits

Cleveland is split with `StratifiedShuffleSplit` using `random_state = 42` into 70% train, 15% validation, and 15% held-out test. Hungarian and Switzerland are reserved for external evaluation.

| Split | Rows | Negative | Positive | Positive rate |
| --- | ---: | ---: | ---: | ---: |
| Cleveland train | 212 | 115 | 97 | 45.8% |
| Cleveland validation | 45 | 24 | 21 | 46.7% |
| Cleveland test | 46 | 25 | 21 | 45.7% |
| Hungarian external | 294 | 188 | 106 | 36.1% |
| Swiss external | 123 | 8 | 115 | 93.5% |

Split metadata is stored in `data/processed/phase1_split_metadata.json`, and class balance is summarized in `outputs/tables/class_balance.csv`.

## Analysis Workflow

### Phase 1: Dataset Preparation

`01_dataset.ipynb` loads the three processed UCI cohorts, coerces numeric values, converts `?` to missing values, creates the binary target, and prepares the split strategy.

### Phase 2: Exploratory Analysis

`02_eda.ipynb` summarizes class balance, missingness, Cleveland train feature distributions, correlations, and cross-site covariate shift. Shift screening uses chi-square tests for low-cardinality variables and Kolmogorov-Smirnov tests for higher-cardinality variables.

Important EDA outputs:

- `outputs/tables/class_balance.csv`
- `outputs/tables/missingness_summary.csv`
- `outputs/tables/distribution_shift_summary.csv`
- `outputs/figures/class_balance.png`
- `outputs/figures/missingness_heatmap.png`
- `outputs/figures/cleveland_train_correlation_matrix.png`
- `outputs/figures/top_distribution_shifts.png`

Notable missingness:

- Cleveland train has 1 missing `thal` value.
- Hungarian has high missingness in `ca` (291/294), `thal` (266/294), and `slope` (190/294).
- Switzerland has high missingness in `ca` (118/123), `fbs` (75/123), and `thal` (52/123).

Notable external shifts versus Cleveland train:

- Hungarian shifts most strongly in `restecg`, `age`, `oldpeak`, `slope`, and `thalach`.
- Switzerland shifts most strongly in `chol`, `restecg`, `thalach`, `cp`, and `sex`.

### Phase 3: Modeling

`03_modeling.ipynb` compares three model families:

- Logistic Regression
- XGBoost tuned with Optuna
- Small MLP

All models use the same preprocessing design:

```text
ClinicalFeatureEngineer -> SimpleImputer(strategy="median") -> StandardScaler
```

The feature engineer expands 13 raw predictors to 19 model features:

```text
age_x_thalach
age_x_oldpeak
trestbps_x_chol
cp_x_thal
exang_x_oldpeak
framingham_proxy
```

Training policy:

- Tune and cross-validate on Cleveland train only.
- Use Cleveland validation for model-selection sanity checking.
- Fit final saved models on Cleveland train plus validation.
- Keep Cleveland test, Hungarian, and Switzerland untouched until Phase 4 evaluation.

Feature metadata is stored in `outputs/tables/phase3_feature_metadata.json`, and selected hyperparameters are stored in `outputs/tables/phase3_best_params.json`.

### Phase 4: Final Evaluation

`04_evaluation.ipynb` evaluates the final saved models on:

- Cleveland held-out test
- Hungarian external cohort
- Switzerland external cohort

The evaluation reports ROC-AUC, average precision, F1, sensitivity, and specificity. Threshold-dependent metrics use a fixed probability threshold of `0.5`.

Phase 4 also includes:

- DeLong tests between the top two ROC-AUC models per split.
- Platt sigmoid and isotonic calibration fitted from Cleveland development data.
- Reliability diagrams.
- CORAL covariance-alignment ablations.
- SHAP explanations for the optimized XGBoost model.

## Saved Model Artifacts

Saved models are stored in `outputs/models/`:

- `phase3_logistic_regression.joblib`
- `phase3_xgboost_optuna.joblib`
- `phase3_small_mlp.joblib`
- `phase3_best_model.joblib`
- `phase3_coral_adapter_class.pkl`

The selected best model by Cleveland validation ROC-AUC is Logistic Regression.

## Development Results

Cross-validation on Cleveland train:

| Model | ROC-AUC mean | ROC-AUC std | Avg precision mean | F1 mean | Sensitivity mean | Specificity mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small MLP | 0.912 | 0.036 | 0.919 | 0.784 | 0.701 | 0.930 |
| Logistic Regression | 0.905 | 0.030 | 0.899 | 0.791 | 0.765 | 0.861 |
| XGBoost + Optuna | 0.901 | 0.026 | 0.900 | 0.814 | 0.795 | 0.861 |

Cleveland validation:

| Model | ROC-AUC | Avg precision | F1 | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.903 | 0.918 | 0.769 | 0.714 | 0.875 |
| XGBoost + Optuna | 0.897 | 0.902 | 0.789 | 0.714 | 0.917 |
| Small MLP | 0.867 | 0.885 | 0.757 | 0.667 | 0.917 |

## Final Evaluation Results

Results from `outputs/tables/phase4_external_metrics.csv`:

| Split | Model | ROC-AUC | Avg precision | F1 | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cleveland test | Logistic Regression | 0.949 | 0.926 | 0.864 | 0.905 | 0.840 |
| Cleveland test | XGBoost + Optuna | 0.935 | 0.915 | 0.837 | 0.857 | 0.840 |
| Cleveland test | Small MLP | 0.935 | 0.930 | 0.791 | 0.810 | 0.800 |
| Hungarian | Logistic Regression | 0.869 | 0.830 | 0.663 | 0.547 | 0.941 |
| Hungarian | XGBoost + Optuna | 0.876 | 0.834 | 0.720 | 0.632 | 0.931 |
| Hungarian | Small MLP | 0.833 | 0.767 | 0.513 | 0.368 | 0.963 |
| Switzerland | Logistic Regression | 0.654 | 0.954 | 0.573 | 0.409 | 0.750 |
| Switzerland | XGBoost + Optuna | 0.720 | 0.970 | 0.632 | 0.470 | 0.750 |
| Switzerland | Small MLP | 0.607 | 0.959 | 0.203 | 0.113 | 1.000 |

Main result pattern:

- Logistic Regression performs best by ROC-AUC on the held-out Cleveland test.
- XGBoost + Optuna performs best by ROC-AUC on both external cohorts.
- Switzerland has a 93.5% positive rate, so its high average precision values must be interpreted against a high-prevalence baseline.

## Statistical Comparison

DeLong test results are stored in `outputs/tables/phase4_delong_tests.csv`.

| Split | Model 1 | Model 2 | AUC difference | p-value |
| --- | --- | --- | ---: | ---: |
| Cleveland test | Logistic Regression | XGBoost + Optuna | 0.013 | 0.405 |
| Hungarian | XGBoost + Optuna | Logistic Regression | 0.006 | 0.659 |
| Switzerland | XGBoost + Optuna | Logistic Regression | 0.065 | 0.031 |

Only the Switzerland XGBoost-versus-Logistic comparison is significant at `p < 0.05` in the saved outputs. This should be interpreted cautiously because the Switzerland cohort has only 8 negative cases.

## Calibration

Calibration experiments compare uncalibrated probabilities, Platt sigmoid calibration, and isotonic calibration. Expected calibration error (ECE) is computed with 10 uniform bins.

Selected ECE values from `outputs/tables/phase4_calibration_metrics.csv`:

| Split | Model | Calibration | ECE |
| --- | --- | --- | ---: |
| Cleveland test | Logistic Regression | uncalibrated | 0.106 |
| Cleveland test | Logistic Regression | isotonic | 0.072 |
| Hungarian | XGBoost + Optuna | uncalibrated | 0.089 |
| Hungarian | XGBoost + Optuna | Platt sigmoid | 0.080 |
| Hungarian | Small MLP | isotonic | 0.034 |
| Switzerland | XGBoost + Optuna | uncalibrated | 0.452 |
| Switzerland | XGBoost + Optuna | isotonic | 0.394 |

Calibration improves some model/split combinations, but probability calibration remains weak on Switzerland.

## Domain Adaptation

The project includes a CORAL covariance-alignment ablation. CORAL is not used to select or train the Phase 3 models; it is evaluated only as a Phase 4 ablation.

Selected XGBoost + Optuna ablation results:

| Split | Metric | Without CORAL | With CORAL | Delta |
| --- | --- | ---: | ---: | ---: |
| Hungarian | ROC-AUC | 0.876 | 0.875 | -0.000 |
| Hungarian | F1 | 0.720 | 0.627 | -0.094 |
| Hungarian | Sensitivity | 0.632 | 0.491 | -0.142 |
| Switzerland | ROC-AUC | 0.720 | 0.628 | -0.091 |
| Switzerland | F1 | 0.632 | 0.408 | -0.223 |
| Switzerland | Sensitivity | 0.470 | 0.261 | -0.209 |

For the optimized XGBoost model, CORAL does not improve external performance in this snapshot.

## Explainability

SHAP TreeExplainer is applied to the optimized XGBoost model.

Main SHAP outputs:

- `outputs/figures/phase4_shap_summary_xgboost.png`
- `outputs/figures/phase4_shap_feature_importance_xgboost.png`
- `outputs/tables/phase4_shap_feature_importance_xgboost.csv`

Top SHAP features:

| Rank | Feature | Mean absolute SHAP |
| ---: | --- | ---: |
| 1 | `ca` | 0.750 |
| 2 | `cp_x_thal` | 0.667 |
| 3 | `cp` | 0.320 |
| 4 | `age_x_oldpeak` | 0.293 |
| 5 | `thal` | 0.262 |
| 6 | `sex` | 0.241 |
| 7 | `age` | 0.198 |
| 8 | `thalach` | 0.195 |
| 9 | `chol` | 0.156 |
| 10 | `age_x_thalach` | 0.131 |

## Reproducibility Notes

- Main seed: `random_state = 42`.
- Cross-validation: 5-fold stratified CV in Phase 3.
- XGBoost tuning: 50 Optuna trials.
- Processed split metadata: `data/processed/phase1_split_metadata.json`.
- Feature metadata: `outputs/tables/phase3_feature_metadata.json`.
- Model-selection metadata: `outputs/tables/phase3_best_params.json`.
- Final metrics and analysis tables: `outputs/tables/`.
- Generated figures: `outputs/figures/`.

Important caveats:

- `outputs/tables/phase4_output_manifest.json` contains absolute paths from another local workspace and references a few older artifact names. Use the actual files present in `outputs/tables/` and `outputs/figures/`.
- The saved `.joblib` models reference custom Phase 3 classes such as `ClinicalFeatureEngineer`. To reload them in a fresh environment, those class definitions must be importable under the expected module name before calling `joblib.load`.
- `notebooks/04_evaluation.ipynb` currently contains cells that expect `src/03_modeling.py`, but this checkout does not include a `src/` directory. The committed outputs remain useful, but full notebook re-execution may need that script restored or the class definitions loaded directly from the Phase 3 notebook.

## Limitations

- The dataset is small, especially the 46-row Cleveland held-out test set.
- The Switzerland cohort is extremely imbalanced with 115 positives and only 8 negatives.
- The processed files do not include patient identifiers, so only row-level split independence can be checked.
- External cohorts have substantial missingness and feature-distribution shift.
- Threshold metrics use a fixed `0.5` threshold instead of a clinically selected operating point.
- Calibration on external cohorts, especially Switzerland, remains weak.
- The repository is a notebook research snapshot, not a packaged production pipeline.
- No temporal validation, prospective validation, external clinical validation, or deployment validation is included.

## Bottom Line

This project shows that models trained on Cleveland can achieve strong internal discrimination, but external performance varies meaningfully by cohort. Logistic Regression is strongest on the held-out Cleveland test, while XGBoost + Optuna has the best ROC-AUC on Hungarian and Switzerland. The external results should be interpreted through the lens of cohort shift, missingness, calibration error, and the severe class imbalance in the Switzerland cohort.
