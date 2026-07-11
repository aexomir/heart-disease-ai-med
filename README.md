# Heart Disease AI in Medicine Project

This repository contains a reproducible machine-learning workflow for binary heart-disease prediction using the processed UCI Heart Disease files for Cleveland, Hungarian, and Switzerland cohorts.

The project treats Cleveland as the source cohort, splits it into train, validation, and held-out test sets, and reserves Hungarian and Swiss as external-site evaluation cohorts. The code covers dataset preparation, exploratory data analysis, tuned model training, held-out and external evaluation, calibration, DeLong statistical testing, CORAL domain-adaptation ablation, and SHAP explainability.

This is an academic/experimental project. The repository does not provide evidence for clinical deployment or prospective validation.

## Repository Structure

```text
.
├── data/
│   ├── raw/                       # UCI processed source files
│   └── processed/                 # Canonical split CSVs and split metadata
├── notebooks/                     # Notebook versions of the phase workflow
├── outputs/
│   ├── figures/                   # EDA, calibration, SHAP, ROC/PR figures
│   ├── models/                    # Saved joblib model artifacts
│   └── tables/                    # Metrics, metadata, tuning, and analysis tables
├── reports/
│   ├── critical_discussion_methods_results.md
│   └── project_requirement_manual.md
├── src/
│   ├── 01_dataset.py              # Phase 1: data loading and canonical splits
│   ├── 02_eda.py                  # Phase 2: EDA and distribution-shift analysis
│   ├── 03_modeling.py             # Phase 3: model tuning, selection, and saving
│   └── 04_evaluation.py           # Phase 4: held-out/external evaluation
└── requirements.txt
```

## Data

The project uses the processed UCI Heart Disease files listed in `src/01_dataset.py`:

- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`

If the files are missing from `data/raw/`, Phase 1 attempts to download them from the UCI archive URLs defined in `SITE_FILES`.

The target is binarized in `src/01_dataset.py`: original target value `0` remains `0`, and any original target value greater than `0` becomes `1`.

## Canonical Splits

Phase 1 uses `StratifiedShuffleSplit` with `random_state = 42`. Cleveland is split into 70% train, 15% validation, and 15% held-out test. Hungarian and Swiss are not used for model tuning; they are reserved for external evaluation.

Split counts from `data/processed/phase1_split_metadata.json` and `outputs/tables/class_balance.csv`:

| Split | Rows | Negative | Positive | Positive Rate |
| --- | ---: | ---: | ---: | ---: |
| Cleveland train | 212 | 115 | 97 | 45.8% |
| Cleveland validation | 45 | 24 | 21 | 46.7% |
| Cleveland held-out test | 46 | 25 | 21 | 45.7% |
| Hungarian external | 294 | 188 | 106 | 36.1% |
| Swiss external | 123 | 8 | 115 | 93.5% |

The processed UCI files do not include explicit patient IDs. The project preserves `source_row_index` and verifies no Cleveland row overlap across train, validation, and test, but true patient-level independence cannot be proven from the available files.

## Features

The raw features are:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal
```

`src/03_modeling.py` applies a shared preprocessing pipeline to all models:

```text
ClinicalFeatureEngineer -> SimpleImputer(strategy="median") -> StandardScaler
```

The feature engineer adds six derived features:

```text
age_x_thalach
age_x_oldpeak
trestbps_x_chol
cp_x_thal
exang_x_oldpeak
framingham_proxy
```

The final model feature count is 19, documented in `outputs/tables/phase3_feature_metadata.json`.

## Workflow

Run the phases from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the project phases:

```bash
python src/01_dataset.py
python src/02_eda.py
python src/03_modeling.py
python src/04_evaluation.py
```

Notes from the code:

- Phase 2 automatically runs Phase 1 if required processed files are missing.
- Phase 3 automatically runs Phase 1 if required processed files are missing.
- Phase 4 automatically runs Phase 3 if saved model files are missing.
- `src/03_modeling.py` uses `n_jobs=1` throughout for single-process stability on macOS/Python environments.

## Phase 1: Dataset Setup

Implemented in `src/01_dataset.py`.

Main outputs:

- `data/processed/cleveland_train.csv`
- `data/processed/cleveland_validation.csv`
- `data/processed/cleveland_test.csv`
- `data/processed/hungarian.csv`
- `data/processed/swiss.csv`
- `data/processed/phase1_split_metadata.json`

Phase 1 records:

- UCI raw file paths and URLs
- binary target conversion
- stratified Cleveland split ratios
- random state
- row-overlap check across Cleveland partitions
- patient-ID limitation

## Phase 2: Exploratory Data Analysis

Implemented in `src/02_eda.py`.

Main table outputs:

- `outputs/tables/class_balance.csv`
- `outputs/tables/missingness_summary.csv`
- `outputs/tables/distribution_shift_summary.csv`

Main figure outputs:

- `outputs/figures/class_balance.png`
- `outputs/figures/missingness_heatmap.png`
- `outputs/figures/cleveland_train_correlation_matrix.png`
- `outputs/figures/top_distribution_shifts.png`
- per-feature Cleveland train histograms in `outputs/figures/`

The distribution-shift analysis compares Cleveland train against Cleveland validation, Cleveland test, Hungarian, and Swiss. It uses chi-square tests for low-cardinality features and Kolmogorov-Smirnov tests for higher-cardinality numeric features.

Examples from `outputs/tables/distribution_shift_summary.csv`:

- Cleveland train vs Hungarian: large shifts include `restecg` (`p = 3.58e-40`), `age` (`p = 1.76e-13`), `oldpeak` (`p = 7.97e-12`), `slope` (`p = 1.77e-11`), and `thalach` (`p = 3.07e-08`).
- Cleveland train vs Swiss: large shifts include `chol` (`p = 9.94e-95`), `restecg` (`p = 2.19e-22`), `thalach` (`p = 1.17e-15`), `cp` (`p = 3.30e-07`), and `sex` (`p = 2.47e-06`).

Missingness examples from `outputs/tables/missingness_summary.csv`:

- Cleveland train has one missing `thal` value.
- Hungarian has high missingness in `ca` (291/294 rows), `thal` (266/294), and `slope` (190/294).
- Swiss has high missingness in `ca` (118/123), `fbs` (75/123), and `thal` (52/123).

## Phase 3: Modeling and Optimization

Implemented in `src/03_modeling.py`.

Training policy from the code:

- Tune and cross-validate only on Cleveland train.
- Use Cleveland validation for model-selection sanity checking.
- Refit final models on Cleveland train plus validation.
- Do not evaluate Cleveland test, Hungarian, or Swiss during Phase 3.

Models:

- Logistic Regression
- XGBoost + Optuna
- Small MLP

All three use the same preprocessing pipeline.

Optimization details from `src/03_modeling.py` and `outputs/tables/phase3_best_params.json`:

| Model | Search Method | Selected Settings |
| --- | --- | --- |
| Logistic Regression | Grid search | `C = 0.5`, `class_weight = None`, `penalty = l2`, `solver = lbfgs` |
| XGBoost + Optuna | 50 Optuna trials | `n_estimators = 130`, `max_depth = 5`, `learning_rate = 0.0554`, `subsample = 0.6564`, `colsample_bytree = 0.6657`, `min_child_weight = 2.7522`, `gamma = 1.8122`, `reg_alpha = 0.0510`, `reg_lambda = 0.5056`, `scale_pos_weight = 1.0295` |
| Small MLP | Randomized search | one hidden layer with 32 units, `relu`, `adam`, `alpha = 0.001`, `batch_size = 16`, `learning_rate_init = 0.0005` |

Saved outputs:

- `outputs/tables/phase3_cv_summary.csv`
- `outputs/tables/phase3_validation_metrics.csv`
- `outputs/tables/phase3_xgboost_optuna_trials.csv`
- `outputs/tables/phase3_feature_names.csv`
- `outputs/tables/phase3_best_params.json`
- `outputs/tables/phase3_feature_metadata.json`
- `outputs/tables/phase3_domain_adaptation_metadata.json`
- `outputs/models/phase3_logistic_regression.joblib`
- `outputs/models/phase3_xgboost_optuna.joblib`
- `outputs/models/phase3_small_mlp.joblib`
- `outputs/models/phase3_best_model.joblib`

The selected best model by Cleveland validation ROC-AUC is Logistic Regression.

### Cross-Validation Results

From `outputs/tables/phase3_cv_summary.csv`:

| Model | ROC-AUC Mean | ROC-AUC Std | Average Precision Mean | F1 Mean | Sensitivity Mean | Specificity Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small MLP | 0.912 | 0.036 | 0.919 | 0.784 | 0.701 | 0.930 |
| Logistic Regression | 0.905 | 0.030 | 0.899 | 0.791 | 0.765 | 0.861 |
| XGBoost + Optuna | 0.901 | 0.026 | 0.900 | 0.814 | 0.795 | 0.861 |

### Cleveland Validation Results

From `outputs/tables/phase3_validation_metrics.csv`:

| Model | ROC-AUC | Average Precision | F1 | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.903 | 0.918 | 0.769 | 0.714 | 0.875 |
| XGBoost + Optuna | 0.897 | 0.902 | 0.789 | 0.714 | 0.917 |
| Small MLP | 0.867 | 0.885 | 0.757 | 0.667 | 0.917 |

## Phase 4: Evaluation

Implemented in `src/04_evaluation.py`.

Evaluation splits:

- Cleveland held-out test
- Hungarian external test
- Swiss external test

Metrics:

- ROC-AUC
- average precision
- F1
- sensitivity
- specificity

Threshold metrics use a fixed probability threshold of `0.5`.

Saved table outputs:

- `outputs/tables/phase4_external_metrics.csv`
- `outputs/tables/phase4_delong_tests.csv`
- `outputs/tables/phase4_calibration_metrics.csv`
- `outputs/tables/phase4_domain_adaptation_ablation.csv`
- `outputs/tables/phase4_shap_feature_importance_xgboost.csv`

Saved figure outputs include:

- reliability diagrams named like `outputs/figures/reliability_<split>_<model>.png`
- `outputs/figures/phase4_shap_summary_xgboost.png`
- older/generated ROC and PR curve figures are also present in `outputs/figures/`

### Held-Out and External Metrics

From `outputs/tables/phase4_external_metrics.csv`:

| Split | Model | ROC-AUC | Average Precision | F1 | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cleveland test | Logistic Regression | 0.949 | 0.926 | 0.864 | 0.905 | 0.840 |
| Cleveland test | XGBoost + Optuna | 0.935 | 0.915 | 0.837 | 0.857 | 0.840 |
| Cleveland test | Small MLP | 0.935 | 0.930 | 0.791 | 0.810 | 0.800 |
| Hungarian | Logistic Regression | 0.869 | 0.830 | 0.663 | 0.547 | 0.941 |
| Hungarian | XGBoost + Optuna | 0.876 | 0.834 | 0.720 | 0.632 | 0.931 |
| Hungarian | Small MLP | 0.833 | 0.767 | 0.513 | 0.368 | 0.963 |
| Swiss | Logistic Regression | 0.654 | 0.954 | 0.573 | 0.409 | 0.750 |
| Swiss | XGBoost + Optuna | 0.720 | 0.970 | 0.632 | 0.470 | 0.750 |
| Swiss | Small MLP | 0.607 | 0.959 | 0.203 | 0.113 | 1.000 |

Observed pattern in the generated outputs:

- Logistic Regression has the highest ROC-AUC on the Cleveland held-out test.
- XGBoost + Optuna has the highest ROC-AUC on both external cohorts.
- Swiss average precision is high for all models, but Swiss has a 93.5% positive rate, so average precision must be interpreted with that prevalence in mind.

### DeLong Tests

From `outputs/tables/phase4_delong_tests.csv`:

| Split | Model 1 | Model 2 | AUC Difference | p-value |
| --- | --- | --- | ---: | ---: |
| Cleveland test | Logistic Regression | XGBoost + Optuna | 0.013 | 0.405 |
| Hungarian | XGBoost + Optuna | Logistic Regression | 0.006 | 0.659 |
| Swiss | XGBoost + Optuna | Logistic Regression | 0.065 | 0.031 |

The current DeLong output supports a statistically significant XGBoost vs Logistic Regression AUC difference on Swiss only among these three comparisons.

## Calibration

Phase 4 evaluates:

- uncalibrated probabilities
- Platt sigmoid calibration
- isotonic calibration

Calibration is measured using expected calibration error (ECE) with 10 uniform bins. The calibration workflow trains base models on Cleveland train and calibrates on Cleveland validation.

Selected ECE values from `outputs/tables/phase4_calibration_metrics.csv`:

| Split | Model | Calibration | ECE |
| --- | --- | --- | ---: |
| Cleveland test | Logistic Regression | uncalibrated | 0.106 |
| Cleveland test | Logistic Regression | isotonic | 0.072 |
| Cleveland test | XGBoost + Optuna | uncalibrated | 0.117 |
| Hungarian | XGBoost + Optuna | uncalibrated | 0.089 |
| Hungarian | XGBoost + Optuna | Platt sigmoid | 0.080 |
| Hungarian | Small MLP | isotonic | 0.034 |
| Swiss | XGBoost + Optuna | uncalibrated | 0.452 |
| Swiss | XGBoost + Optuna | isotonic | 0.394 |

The generated calibration results show that calibration can improve ECE, especially in some model/split combinations, but Swiss remains poorly calibrated across models.

## Domain Adaptation

The project implements CORAL covariance alignment in `CORALAdapter` in `src/03_modeling.py`.

Phase 3 does not use CORAL during model training. Phase 4 uses it only as an ablation, comparing predictions without CORAL against predictions after target features are transformed into the source covariance space.

From `outputs/tables/phase3_domain_adaptation_metadata.json`:

```text
method: CORAL covariance alignment
used_during_phase3_training: false
purpose: Phase 4 ablation only
```

Selected XGBoost + Optuna deltas from `outputs/tables/phase4_domain_adaptation_ablation.csv`:

| Split | Metric | Without CORAL | With CORAL | Delta |
| --- | --- | ---: | ---: | ---: |
| Hungarian | ROC-AUC | 0.876 | 0.875 | -0.000 |
| Hungarian | F1 | 0.720 | 0.627 | -0.094 |
| Hungarian | Sensitivity | 0.632 | 0.491 | -0.142 |
| Swiss | ROC-AUC | 0.720 | 0.628 | -0.091 |
| Swiss | F1 | 0.632 | 0.408 | -0.223 |
| Swiss | Sensitivity | 0.470 | 0.261 | -0.209 |

For the optimized XGBoost model, the current CORAL ablation does not improve external performance.

## Explainability

Phase 4 uses SHAP `TreeExplainer` on the optimized XGBoost model with up to 150 Cleveland train-plus-validation samples.

Outputs:

- `outputs/figures/phase4_shap_summary_xgboost.png`
- `outputs/tables/phase4_shap_feature_importance_xgboost.csv`

Top SHAP features from `outputs/tables/phase4_shap_feature_importance_xgboost.csv`:

| Rank | Feature | Mean Absolute SHAP |
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

The project uses fixed seeds:

- Phase 1 split: `RANDOM_STATE = 42`
- Phase 3 training/CV/Optuna sampling: `RANDOM_STATE = 42`

Core reproducibility artifacts:

- canonical processed splits in `data/processed/`
- saved model artifacts in `outputs/models/`
- saved metrics and metadata in `outputs/tables/`
- saved figures in `outputs/figures/`
- script versions in `src/`
- notebook versions in `notebooks/`

One repository note: `outputs/tables/phase4_output_manifest.json` contains absolute paths from another local workspace and references some older file names. The current Phase 4 code writes the table names listed in the Phase 4 section above.

## Limitations Reflected in the Current Code and Outputs

- The processed UCI files do not include patient IDs, so the code can verify row-level non-overlap for Cleveland splits but cannot prove patient-level independence beyond row identity.
- The dataset is small: Cleveland held-out test has 46 rows, Hungarian has 294 rows, and Swiss has 123 rows.
- Swiss has only 8 negative cases and a 93.5% positive rate, making specificity, calibration, and average precision especially sensitive to prevalence.
- External cohorts show substantial feature-distribution shift relative to Cleveland train.
- Threshold metrics use a fixed `0.5` threshold; there is no validation-derived clinical threshold selection in the current scripts.
- CORAL is implemented as a full-target unsupervised feature-alignment ablation; the code does not implement a half-target adaptation sensitivity design.
- There is no temporal validation in the repository.
- The current report draft is in `reports/critical_discussion_methods_results.md`; it is not a full polished manuscript with all formal sections.

## Key Takeaways From the  Results

- Logistic Regression is selected as the best model by Cleveland validation ROC-AUC and performs best on the Cleveland held-out test by ROC-AUC.
- XGBoost + Optuna is the strongest external-site ranking model by ROC-AUC on Hungarian and Swiss.
- The Hungarian XGBoost vs Logistic Regression AUC difference is small and not statistically significant in the current DeLong table.
- The Swiss XGBoost vs Logistic Regression AUC difference is statistically significant in the current DeLong table, but Swiss has severe prevalence imbalance and only 8 negative examples.
- Calibration should be reported alongside discrimination; Swiss probability calibration remains poor even after calibration.
- CORAL covariance alignment is a useful negative-result ablation here: it does not improve the optimized XGBoost model on the external cohorts.
