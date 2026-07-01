# Project Requirement Manual

This is a human-readable checklist for understanding what the project already does, where to find each part, and how strong each part is. I am using three statuses:

- **Done**: implemented in code and/or clearly saved in outputs.
- **Partial**: mostly present, but there is a caveat or it could be stronger.
- **Not done**: I could not find evidence in the project files.

## Quick Project Map

- Phase 1 dataset and splitting: `src/01_dataset.py`
- Phase 2 EDA: `src/02_eda.py`
- Phase 3 modeling and optimization: `src/03_modeling.py`
- Phase 4 evaluation, calibration, SHAP, DeLong, CORAL: `src/04_evaluation.py`
- Processed datasets: `data/processed/`
- Saved tables: `outputs/tables/`
- Saved figures: `outputs/figures/`
- Saved models: `outputs/models/`
- Current written draft: `reports/critical_discussion_methods_results.md`

## 1. Dataset & Splitting

**Overall status: Done.**

The project uses the UCI Heart Disease processed files and treats Cleveland as the source dataset, with Hungarian and Swiss held back as external test datasets. The target is binary: `0` means no disease, and any original disease label greater than zero becomes `1`.

**Where to find it**

- UCI source files and URLs: `src/01_dataset.py`, lines 37-50.
- Binary target conversion: `src/01_dataset.py`, lines 70-78.
- Cleveland 70/15/15 split: `src/01_dataset.py`, lines 87-106.
- Held-out Cleveland test, Hungarian, and Swiss saved separately: `src/01_dataset.py`, lines 130-144.
- Split metadata and documentation: `src/01_dataset.py`, lines 146-168.
- Actual split counts: `data/processed/phase1_split_metadata.json`.

**How well it is done**

This is one of the strongest parts of the project. The code creates Cleveland train, validation, and held-out test before modeling. Hungarian and Swiss are separate files, so they can remain untouched until Phase 4.

**Technical note**

The split is stratified, so the disease/no-disease ratio stays similar across Cleveland train, validation, and test. The project also preserves `source_row_index` and checks no Cleveland row appears in multiple Cleveland splits.

**Important caveat**

There are no patient IDs in the processed UCI files. The project handles this correctly by stating the assumption instead of pretending true patient-level independence can be proven. See `src/01_dataset.py`, lines 12-13 and 149-154.

## 2. Exploratory Data Analysis

**Overall status: Done, with one small caveat.**

The EDA covers class balance, missingness, distributions, correlation, and cross-site distribution shift.

**Where to find it**

- Loads all canonical splits: `src/02_eda.py`, lines 47-55.
- Class balance table: `src/02_eda.py`, lines 58-69.
- Missing-value summary: `src/02_eda.py`, lines 72-82.
- Cross-site distribution shift testing: `src/02_eda.py`, lines 89-143.
- Class balance plot: `src/02_eda.py`, lines 146-156.
- Missingness heatmap: `src/02_eda.py`, lines 159-173.
- Feature histograms: `src/02_eda.py`, lines 176-193.
- Correlation matrix: `src/02_eda.py`, lines 196-209.
- Top distribution shift plot: `src/02_eda.py`, lines 212-222.
- Saved EDA outputs: `src/02_eda.py`, lines 238-246.

**Useful output files**

- `outputs/tables/class_balance.csv`
- `outputs/tables/missingness_summary.csv`
- `outputs/tables/distribution_shift_summary.csv`
- `outputs/figures/missingness_heatmap.png`
- `outputs/figures/cleveland_train_correlation_matrix.png`
- `outputs/figures/top_distribution_shifts.png`

**How well it is done**

The EDA is good for this assignment because it does more than describe Cleveland. It compares Cleveland train against Hungarian and Swiss, which directly supports the generalization/domain adaptation story.

**Technical note**

The shift analysis uses chi-square tests for low-cardinality features and Kolmogorov-Smirnov tests for more continuous features. This is useful because some UCI variables are categorical codes, while others are numeric measurements.

**Small caveat**

There is no separate saved table of full descriptive statistics like mean, standard deviation, median, and quartiles for every feature. The project still has dataset statistics through class balance, missingness, histograms, and shift tables, but a full `describe()` table would make this even cleaner.

## 3. Modeling

**Overall status: Done.**

The project implements the three required optimized models: Logistic Regression, XGBoost, and Small MLP.

**Where to find it**

- Shared model imports: `src/03_modeling.py`, lines 33-47.
- Shared preprocessing pipeline: `src/03_modeling.py`, lines 226-231.
- Logistic Regression pipeline: `src/03_modeling.py`, lines 234-238.
- XGBoost pipeline: `src/03_modeling.py`, lines 241-253.
- Small MLP pipeline: `src/03_modeling.py`, lines 256-266.
- Three models trained in Phase 3: `src/03_modeling.py`, lines 508-510.
- Final models saved: `src/03_modeling.py`, lines 465-468.

**How well it is done**

This is solid. All models use the same preprocessing, which makes the comparison fair. The final models are saved with `joblib`, so Phase 4 can reload them instead of retraining manually.

**Technical note**

The preprocessing is inside the model pipeline. That matters because imputation and scaling are learned only from the training fold during cross-validation, which reduces leakage risk.

## 4. Feature Engineering

**Overall status: Done.**

The project adds clinical interaction features and a Framingham-inspired risk proxy.

**Where to find it**

- Feature engineering class: `src/03_modeling.py`, lines 79-132.
- Interaction terms: `src/03_modeling.py`, lines 99-103.
- Framingham-inspired proxy: `src/03_modeling.py`, lines 105-111.
- Engineered feature count saved: `src/03_modeling.py`, lines 447-455.
- Feature metadata output: `outputs/tables/phase3_feature_metadata.json`.

**How well it is done**

This is a real project contribution beyond simple benchmarking. The engineered features make clinical sense because heart disease risk often depends on combinations, such as age with exercise capacity or chest pain type with thalassemia result.

**Technical note**

The final feature count is 19: 13 raw features plus 6 engineered features. This is documented in `outputs/tables/phase3_feature_metadata.json`.

## 5. Optimization

**Overall status: Done.**

The professor asked for no default hyperparameters. The project satisfies this through grid search, randomized search, and Optuna.

**Where to find it**

- Five-fold CV constant: `src/03_modeling.py`, lines 56-58.
- Logistic Regression grid search: `src/03_modeling.py`, lines 305-346.
- XGBoost Optuna search space: `src/03_modeling.py`, lines 349-382.
- Optuna trial count set to 50: `src/03_modeling.py`, line 58.
- XGBoost actually optimized with `n_trials=XGB_OPTUNA_TRIALS`: `src/03_modeling.py`, line 371.
- MLP randomized search: `src/03_modeling.py`, lines 385-414.
- Best parameters saved: `outputs/tables/phase3_best_params.json`.
- Optuna trial table saved: `outputs/tables/phase3_xgboost_optuna_trials.csv`.

**How well it is done**

Strong. The project does not just instantiate models with defaults. It tunes all three, and XGBoost satisfies the professor's specific 50-trial Optuna requirement.

**Technical note**

Optimization is done only on Cleveland train. The code explicitly says Cleveland test, Hungarian, and Swiss are not evaluated in Phase 3, which protects the external evaluation. See `src/03_modeling.py`, lines 10-16 and 500-504.

## 6. Explainability

**Overall status: Done.**

The project uses SHAP TreeExplainer for the optimized XGBoost model, saves a SHAP summary plot, and saves a feature-importance table.

**Where to find it**

- SHAP runner: `src/04_evaluation.py`, lines 457-512.
- TreeExplainer call: `src/04_evaluation.py`, lines 485-487.
- SHAP summary plot saved: `src/04_evaluation.py`, lines 489-500.
- SHAP feature importance table created: `src/04_evaluation.py`, lines 503-506.
- SHAP output saved in Phase 4: `src/04_evaluation.py`, lines 553-557.
- SHAP result table: `outputs/tables/phase4_shap_feature_importance_xgboost.csv`.
- SHAP plot: `outputs/figures/phase4_shap_summary_xgboost.png`.

**How well it is done**

Good. The top SHAP features are clinically believable: `ca`, `cp_x_thal`, `cp`, `age_x_oldpeak`, `thal`, `sex`, `age`, and `thalach`.

**Technical note**

The strongest SHAP feature is `ca` with mean absolute SHAP 0.750. That means the model heavily uses the number of major vessels, which clinically makes sense for coronary disease. The current discussion explains this in `reports/critical_discussion_methods_results.md`, line 13.

## 7. Calibration

**Overall status: Done.**

The project implements uncalibrated, Platt sigmoid, and isotonic calibration comparisons with ECE and reliability diagrams.

**Where to find it**

- ECE calculation: `src/04_evaluation.py`, lines 301-323.
- Platt and isotonic calibrator creation: `src/04_evaluation.py`, lines 326-341.
- Calibration experiment: `src/04_evaluation.py`, lines 344-377.
- Reliability diagrams: `src/04_evaluation.py`, lines 380-411.
- Calibration run in Phase 4: `src/04_evaluation.py`, lines 540-543.
- Calibration metrics saved: `src/04_evaluation.py`, line 550.
- Calibration output table: `outputs/tables/phase4_calibration_metrics.csv`.
- Reliability figures: files named like `outputs/figures/phase4_reliability_hungarian_xgboost_plus_optuna.png`.

**How well it is done**

Good and clinically important. The project does not only report AUC; it checks whether predicted probabilities are trustworthy.

**Technical note**

On Hungarian, Platt-calibrated XGBoost has ECE 0.080, better than uncalibrated XGBoost ECE 0.089. On Swiss, calibration remains poor even after isotonic calibration: XGBoost improves from ECE 0.452 to 0.394, but that is still high. This supports the argument that the model can rank patients but should not be trusted blindly as a calibrated risk calculator.

## 8. Domain Adaptation

**Overall status: Done, with one professor-suggestion caveat.**

The project treats Cleveland to Hungarian/Swiss as a domain adaptation problem and implements CORAL as an isolated module. It then runs an ablation comparing no CORAL versus CORAL.

**Where to find it**

- CORAL adapter class: `src/03_modeling.py`, lines 135-168.
- Domain adaptation metadata: `src/03_modeling.py`, lines 457-463.
- CORAL ablation runner: `src/04_evaluation.py`, lines 414-454.
- CORAL fit/transform/predict logic: `src/04_evaluation.py`, lines 430-442.
- CORAL metrics and deltas: `src/04_evaluation.py`, lines 444-452.
- CORAL ablation run in Phase 4: `src/04_evaluation.py`, line 545.
- Ablation output table: `outputs/tables/phase4_domain_adaptation_ablation.csv`.

**How well it is done**

The implementation is clear and useful, especially because it reports negative results instead of hiding them. For XGBoost, CORAL does not help: Swiss ROC-AUC drops from 0.720 to 0.628, F1 drops from 0.632 to 0.408, and sensitivity drops from 0.470 to 0.261. This is discussed in `reports/critical_discussion_methods_results.md`, lines 11 and 39.

**Technical note**

CORAL aligns covariance structure between source and target feature spaces. It can help when feature distributions shift but labels mean the same thing. It can fail when the target prevalence, missingness, or clinical coding is very different. That is probably what happens with Swiss, where the positive rate is 93.5%.

**Caveat**

The professor additionally suggested adapting using only half of the target dataset and then evaluating on the full target dataset. I did not find that exact half-target adaptation design in the code. Current CORAL uses the full target feature matrix in an unsupervised way. Because it does not use target labels, this is not label leakage, but the half-target suggestion is still not implemented exactly.

## 9. Performance Evaluation

**Overall status: Mostly done.**

The required metrics are implemented: ROC-AUC, PR-AUC/average precision, F1, sensitivity, and specificity.

**Where to find it**

- Specificity from confusion matrix: `src/04_evaluation.py`, lines 139-141.
- Metrics function: `src/04_evaluation.py`, lines 144-155.
- Model evaluation loop: `src/04_evaluation.py`, lines 158-175.
- Evaluation splits: `src/04_evaluation.py`, lines 529-533.
- Metrics saved: `src/04_evaluation.py`, line 548.
- Main results table: `outputs/tables/phase4_external_metrics.csv`.
- PR and ROC curve figures: `outputs/figures/phase4_pr_curves_hungarian.png`, `outputs/figures/phase4_pr_curves_swiss.png`, `outputs/figures/phase4_roc_curves_hungarian.png`, `outputs/figures/phase4_roc_curves_swiss.png`.

**How well it is done**

The numerical evaluation is comprehensive. The best external model is XGBoost + Optuna: Hungarian ROC-AUC 0.876 and Swiss ROC-AUC 0.720. Logistic Regression is strongest on Cleveland held-out test with ROC-AUC 0.949.

**Technical note**

Sensitivity and specificity are calculated from predictions at a 0.5 probability threshold. That is simple and reproducible.

**Caveat**

The threshold method is currently a fixed 0.5 threshold, not an optimized clinical threshold. If the professor expects explicit threshold selection, this is only partially satisfied. A stronger version would choose a threshold on Cleveland validation, for example by maximizing Youden's J statistic or setting a minimum sensitivity target, then lock that threshold before external testing.

## 10. Statistical Evaluation

**Overall status: Done.**

The project implements DeLong tests between the top two models by ROC-AUC for each evaluation split.

**Where to find it**

- DeLong helper functions: `src/04_evaluation.py`, lines 178-252.
- DeLong comparison runner: `src/04_evaluation.py`, lines 255-295.
- DeLong tests run in Phase 4: `src/04_evaluation.py`, line 538.
- DeLong output saved: `src/04_evaluation.py`, line 549.
- DeLong table: `outputs/tables/phase4_delong_tests.csv`.

**How well it is done**

Good. It reports p-values and supports a careful interpretation. For example, XGBoost beats Logistic Regression on Swiss by AUC difference 0.065 with p = 0.031, but the Hungarian difference is tiny and not significant: AUC difference 0.006 with p = 0.659.

**Technical note**

This matters because two AUCs can look different just by random noise, especially with small datasets. DeLong testing asks whether the AUC difference is likely meaningful.

## 11. Generalization

**Overall status: Done.**

The project evaluates in-domain held-out generalization and external-site generalization.

**Where to find it**

- Phase 4 evaluates Cleveland test, Hungarian, and Swiss: `src/04_evaluation.py`, lines 9-12 and 529-533.
- External metrics table: `outputs/tables/phase4_external_metrics.csv`.
- Discussion of generalization: `reports/critical_discussion_methods_results.md`, lines 7, 11, 31, 33, and 35.

**How well it is done**

Good. The project does not stop at Cleveland validation. It checks whether models transfer to Hungarian and Swiss, which is exactly the hard clinical question.

**Technical note**

The results show mixed generalization. XGBoost generalizes best externally, especially Hungarian ROC-AUC 0.876 and Swiss ROC-AUC 0.720. But Swiss remains difficult because the cohort is extremely positive-heavy: 115 positives and only 8 negatives.

## 12. Overfitting

**Overall status: Done, but the write-up should explain it clearly.**

The project checks overfitting indirectly by comparing cross-validation, validation, held-out Cleveland test, and external test performance.

**Where to find it**

- Five-fold CV scores saved: `outputs/tables/phase3_cv_summary.csv`.
- Cleveland validation scores saved: `outputs/tables/phase3_validation_metrics.csv`.
- Held-out/external scores saved: `outputs/tables/phase4_external_metrics.csv`.
- CV logic: `src/03_modeling.py`, lines 506-522.
- Final held-out evaluation: `src/04_evaluation.py`, lines 529-548.
- Overfitting discussion in draft: `reports/critical_discussion_methods_results.md`, lines 29-35.

**How well it is done**

Reasonable. The held-out Cleveland test does not collapse compared with CV. For example, Logistic Regression CV ROC-AUC is 0.905 and Cleveland held-out ROC-AUC is 0.949. XGBoost CV ROC-AUC is 0.901 and Cleveland held-out ROC-AUC is 0.935. That pattern does not suggest obvious overfitting to Cleveland train.

**Technical note**

Overfitting would look like excellent train/CV performance but much worse held-out performance. Here, the Cleveland held-out performance is similar or even higher than CV. The performance drop on Swiss is more likely domain shift than ordinary overfitting, because Swiss has a very different positive rate and strong feature shifts.

**Caveat**

There is no explicit training-set metric table beside CV. A stricter overfitting check could add train-vs-validation-vs-test metrics for every model.

## 13. Reproducibility

**Overall status: Done.**

The project has notebooks, Python runners, fixed seeds, saved outputs, and saved models.

**Where to find it**

- Fixed seed in Phase 1: `src/01_dataset.py`, line 27.
- Fixed seed and CV settings in Phase 3: `src/03_modeling.py`, lines 56-58.
- Python runners: `src/01_dataset.py`, `src/02_eda.py`, `src/03_modeling.py`, `src/04_evaluation.py`.
- Separate notebooks: `notebooks/01_dataset_v2.ipynb`, `notebooks/02_eda_v2.ipynb`, `notebooks/03_modeling_v2.ipynb`, `notebooks/04_evaluation_v2.ipynb`.
- Saved models: `outputs/models/`.
- Saved tables: `outputs/tables/`.
- Saved figures: `outputs/figures/`.

**How well it is done**

Strong. The project is not only notebook-based. It has scripts that can regenerate the major artifacts.

**Technical note**

Using `random_state = 42` in splitting, model training, and Optuna sampling makes the results much more reproducible.

## 14. Final Report

**Overall status: Partial.**

There is now a strong draft of critical discussion, methods, results, limitations, and a rubric check. It is not yet a fully polished final submission with every formal section.

**Where to find it**

- Current draft: `reports/critical_discussion_methods_results.md`.
- Related work and contribution framing: `reports/critical_discussion_methods_results.md`, line 5.
- Methods: `reports/critical_discussion_methods_results.md`, lines 17-25.
- Results: `reports/critical_discussion_methods_results.md`, lines 27-39.
- Limitations: `reports/critical_discussion_methods_results.md`, line 15.
- Rubric check: `reports/critical_discussion_methods_results.md`, lines 41-46.

**How well it is done**

The hard scientific content is there. What is missing is mostly final packaging: a clean introduction, a short conclusion, maybe a formal related-work section, and figure/table callouts.

**Technical note**

The best report structure would be: Introduction, Related Work, Dataset, Methods, Experimental Setup, Results, Explainability, Calibration, Domain Adaptation, Discussion, Limitations, Conclusion.

## 15. Required Limitations

**Overall status: Done.**

The current draft explicitly mentions the major limitations: small dataset, older 1980s data, no temporal validation, no patient IDs, limited external cohorts, and the Swiss prevalence issue.

**Where to find it**

- Limitations paragraph: `reports/critical_discussion_methods_results.md`, line 15.
- Dataset age source: [UCI Heart Disease dataset page](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease).
- Patient ID assumption in code: `src/01_dataset.py`, lines 12-13 and 149-154.

**How well it is done**

Good. The limitations are not hidden. They are tied to actual numbers: Cleveland test has only 46 rows, Swiss has only 8 negatives, and the dataset was donated in 1988.

**Technical note**

These limitations matter because calibration, specificity, and p-values can be unstable when the number of negatives is tiny. Swiss specificity is especially fragile because it is based on only 8 negative cases.

## 16. Administrative Requirement

**Requirement: Reserve a presentation slot in the shared Excel file.**

**Status: Not done / cannot verify from this repo.**

I did not find a shared Excel file or any local evidence that a presentation slot was reserved.

**Where to find it**

- No file found in the workspace for this requirement.

**How well it is done**

This is outside the code project. You should manually confirm it in the shared Excel file used by the class.

## Short Professor-Requirement Summary

| Requirement                                                   |                  Status | Best evidence                                                                                  |
| ------------------------------------------------------------- | ----------------------: | ---------------------------------------------------------------------------------------------- |
| UCI Heart Disease, Cleveland source, Hungarian/Swiss external |                    Done | `src/01_dataset.py`, lines 37-50 and 171-183                                                   |
| Held-out Cleveland test before modeling                       |                    Done | `src/01_dataset.py`, lines 87-106                                                              |
| No data leakage / untouched external tests                    |                    Done | `src/03_modeling.py`, lines 10-16 and 500-504                                                  |
| Subject independence documented                               |                    Done | `src/01_dataset.py`, lines 12-13 and 149-154                                                   |
| EDA and distribution shift                                    |                    Done | `src/02_eda.py`, lines 58-143 and 238-246                                                      |
| Three optimized models                                        |                    Done | `src/03_modeling.py`, lines 305-414                                                            |
| Shared preprocessing                                          |                    Done | `src/03_modeling.py`, lines 226-231                                                            |
| Feature engineering                                           |                    Done | `src/03_modeling.py`, lines 79-111                                                             |
| 50 Optuna trials                                              |                    Done | `src/03_modeling.py`, lines 56-58 and 371                                                      |
| SHAP explainability                                           |                    Done | `src/04_evaluation.py`, lines 457-512                                                          |
| Calibration with Platt/isotonic/ECE                           |                    Done | `src/04_evaluation.py`, lines 301-377                                                          |
| Reliability diagrams                                          |                    Done | `src/04_evaluation.py`, lines 380-411                                                          |
| Domain adaptation with CORAL                                  |                    Done | `src/03_modeling.py`, lines 135-168; `src/04_evaluation.py`, lines 417-454                     |
| Half-target adaptation suggestion                             |        Not done exactly | Current CORAL uses full target features unsupervised                                           |
| ROC-AUC, PR-AUC, F1, sensitivity, specificity                 |                    Done | `src/04_evaluation.py`, lines 144-155                                                          |
| PR curves                                                     |   Done as saved figures | `outputs/figures/phase4_pr_curves_hungarian.png`, `outputs/figures/phase4_pr_curves_swiss.png` |
| Threshold selection                                           |                 Partial | Fixed 0.5 threshold in `src/04_evaluation.py`, line 147                                        |
| DeLong test                                                   |                    Done | `src/04_evaluation.py`, lines 178-295                                                          |
| External validation                                           |                    Done | `src/04_evaluation.py`, lines 529-533                                                          |
| Overfitting check                                             | Done, could be stronger | `outputs/tables/phase3_cv_summary.csv` vs `outputs/tables/phase4_external_metrics.csv`         |
| Reproducibility                                               |                    Done | Scripts, notebooks, saved outputs, fixed seeds                                                 |
| Final report                                                  |                 Partial | `reports/critical_discussion_methods_results.md`                                               |
| Presentation slot                                             |            Not verified | Needs manual check in shared Excel                                                             |

## What I Would Improve Before Submission

1. Add a small threshold-selection section if the professor expects more than a fixed 0.5 threshold. Use Cleveland validation only, then lock the threshold before external tests.
2. Add an explicit train-vs-validation-vs-test table to make the overfitting argument even stronger.
3. Implement the professor's half-target CORAL suggestion as a sensitivity analysis. Use half of Hungarian/Swiss features to fit CORAL, then evaluate on the whole target or a held-out target half, depending on what the professor meant.
4. Turn `reports/critical_discussion_methods_results.md` into the final polished report with Introduction and Conclusion sections.
5. Manually reserve and document the presentation slot in the shared Excel file.
