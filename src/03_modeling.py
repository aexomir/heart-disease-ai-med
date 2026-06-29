"""
Phase 3 v2 — Stable Single-Process Modeling Runner
AI in Medicine Project: Heart Disease ML

Why this version exists
-----------------------
This version avoids joblib multiprocessing errors on macOS / Python 3.14 by using
n_jobs=1 everywhere. It is slower but much more stable.

Rules
-----
- Tune/CV only on Cleveland train.
- Use Cleveland validation only for model-selection sanity check.
- Refit final models on Cleveland train + validation.
- Do not touch Cleveland test, Hungarian, or Swiss in Phase 3.
- Final model refitting uses sklearn.base.clone().
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

# Stable alias for later joblib loading in Phase 4.
STABLE_MODULE_NAME = "phase3_modeling"
sys.modules[STABLE_MODULE_NAME] = sys.modules[__name__]

RANDOM_STATE = 42
N_SPLITS = 5
XGB_OPTUNA_TRIALS = 50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

RAW_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal",
]

REQUIRED_PHASE1_FILES = [
    "cleveland_train.csv",
    "cleveland_validation.csv",
    "cleveland_test.csv",
    "hungarian.csv",
    "swiss.csv",
]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add clinical interaction terms and a Framingham-inspired proxy."""

    def __init__(self):
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        X_df = self._as_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        transformed = self.transform(X_df)
        self.feature_names_out_ = list(transformed.columns)
        return self

    def transform(self, X):
        X_df = self._as_dataframe(X).copy()

        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        X_df["age_x_thalach"] = X_df["age"] * X_df["thalach"]
        X_df["age_x_oldpeak"] = X_df["age"] * X_df["oldpeak"]
        X_df["trestbps_x_chol"] = X_df["trestbps"] * X_df["chol"]
        X_df["cp_x_thal"] = X_df["cp"] * X_df["thal"]
        X_df["exang_x_oldpeak"] = X_df["exang"] * X_df["oldpeak"]

        X_df["framingham_proxy"] = (
            X_df["age"] / 10.0
            + X_df["sex"] * 2.0
            + (X_df["chol"] - 200.0) / 40.0
            + (X_df["trestbps"] - 120.0) / 20.0
            + X_df["fbs"] * 1.5
        )

        return X_df

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_, dtype=object)
        if input_features is not None:
            return np.array(input_features, dtype=object)
        return np.array([], dtype=object)

    @staticmethod
    def _as_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()

        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        if arr.shape[1] != len(RAW_FEATURES):
            raise ValueError(f"Expected {len(RAW_FEATURES)} features, got {arr.shape[1]}.")
        return pd.DataFrame(arr, columns=RAW_FEATURES)


class CORALAdapter:
    """CORAL covariance alignment for Phase 4 ablation only."""

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.source_mean_ = None
        self.target_mean_ = None
        self.target_whitening_ = None
        self.source_coloring_ = None

    def fit(self, X_source, X_target):
        Xs = np.asarray(X_source, dtype=float)
        Xt = np.asarray(X_target, dtype=float)

        self.source_mean_ = Xs.mean(axis=0)
        self.target_mean_ = Xt.mean(axis=0)

        Cs = np.cov(Xs, rowvar=False) + self.eps * np.eye(Xs.shape[1])
        Ct = np.cov(Xt, rowvar=False) + self.eps * np.eye(Xt.shape[1])

        self.source_coloring_ = self._matrix_power(Cs, 0.5)
        self.target_whitening_ = self._matrix_power(Ct, -0.5)
        return self

    def transform_target_to_source(self, X_target):
        Xt = np.asarray(X_target, dtype=float)
        Xt_centered = Xt - self.target_mean_
        return Xt_centered @ self.target_whitening_ @ self.source_coloring_ + self.source_mean_

    @staticmethod
    def _matrix_power(matrix, power):
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


ClinicalFeatureEngineer.__module__ = STABLE_MODULE_NAME
CORALAdapter.__module__ = STABLE_MODULE_NAME


def import_phase1_runner():
    phase1_path = PROJECT_ROOT / "src" / "01_dataset.py"
    if not phase1_path.exists():
        raise FileNotFoundError(f"Cannot find Phase 1 runner at {phase1_path}")

    spec = importlib.util.spec_from_file_location("phase1_dataset", phase1_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_phase1_outputs():
    missing = [f for f in REQUIRED_PHASE1_FILES if not (PROCESSED_DATA_DIR / f).exists()]
    if not missing:
        return

    print("Missing Phase 1 processed files. Running Phase 1 first...")
    phase1 = import_phase1_runner()

    if hasattr(phase1, "run_dataset_setup_v2"):
        phase1.run_dataset_setup_v2()
    elif hasattr(phase1, "run_dataset_setup"):
        phase1.run_dataset_setup()
    else:
        raise AttributeError("Phase 1 runner must expose run_dataset_setup_v2() or run_dataset_setup().")


def load_processed_split(filename: str) -> pd.DataFrame:
    ensure_phase1_outputs()
    path = PROCESSED_DATA_DIR / filename
    df = pd.read_csv(path)

    required = RAW_FEATURES + ["target"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{filename} missing columns: {missing_cols}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["target"].isna().any():
        raise ValueError(f"{filename} contains missing target values.")

    df["target"] = df["target"].astype(int)
    return df


def get_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df[RAW_FEATURES].copy(), df["target"].copy()


def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("features", ClinicalFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def make_logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])


def make_xgboost_pipeline(params: dict | None = None) -> Pipeline:
    params = params or {}
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            **params,
        )),
    ])


def make_mlp_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", MLPClassifier(
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=50,
            random_state=RANDOM_STATE,
        )),
    ])


def specificity_score(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan


SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "sensitivity": "recall",
    "specificity": make_scorer(specificity_score),
}


def summarize_cv_results(name: str, results: dict) -> dict:
    row = {"model": name}
    for metric in SCORING:
        values = np.asarray(results[f"test_{metric}"], dtype=float)
        row[f"{metric}_mean"] = float(np.nanmean(values))
        row[f"{metric}_std"] = float(np.nanstd(values))
    return row


def evaluate_validation(name: str, model: Pipeline, X_val, y_val) -> dict:
    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_val, prob)),
        "average_precision": float(average_precision_score(y_val, prob)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_val, pred, zero_division=0)),
        "specificity": specificity_score(y_val, pred),
    }


def train_logistic_regression(X_train, y_train, cv):
    print("\nTraining optimized Logistic Regression...")

    grid = [
        {
            "model__solver": ["liblinear"],
            "model__penalty": ["l1", "l2"],
            "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            "model__class_weight": [None, "balanced"],
        },
        {
            "model__solver": ["lbfgs"],
            "model__penalty": ["l2"],
            "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            "model__class_weight": [None, "balanced"],
        },
    ]

    search = GridSearchCV(
        make_logistic_pipeline(),
        grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    cv_results = cross_validate(
        search.best_estimator_,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )

    print(f"Best LR AUC: {search.best_score_:.6f}")
    print(f"Best LR params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, summarize_cv_results("Logistic Regression", cv_results)


def train_xgboost_optuna(X_train, y_train, cv):
    print("\nTraining XGBoost with Optuna...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 350),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.5),
        }
        model = make_xgboost_pipeline(params)
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1, error_score="raise")
        return float(np.mean(scores["test_score"]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=XGB_OPTUNA_TRIALS, show_progress_bar=True)

    best = make_xgboost_pipeline(study.best_params)
    best.fit(X_train, y_train)

    cv_results = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise")
    trials_df = study.trials_dataframe()

    print(f"Completed Optuna trials: {len(study.trials)}")
    print(f"Best XGBoost AUC: {study.best_value:.6f}")
    print(f"Best XGBoost params: {study.best_params}")
    return best, study.best_params, trials_df, summarize_cv_results("XGBoost + Optuna", cv_results)


def train_mlp(X_train, y_train, cv):
    print("\nTraining optimized small MLP...")

    grid = {
        "model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8), (32, 16)],
        "model__activation": ["relu", "tanh"],
        "model__solver": ["adam"],
        "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "model__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "model__batch_size": [16, 32, 64],
    }

    search = RandomizedSearchCV(
        make_mlp_pipeline(),
        grid,
        n_iter=25,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    cv_results = cross_validate(search.best_estimator_, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise")

    print(f"Best MLP AUC: {search.best_score_:.6f}")
    print(f"Best MLP params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, summarize_cv_results("Small MLP", cv_results)


def refit_final_model(best_estimator: Pipeline, X_train_val, y_train_val) -> Pipeline:
    model = clone(best_estimator)
    model.fit(X_train_val, y_train_val)
    return model


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    pre = fitted_pipeline.named_steps["preprocess"]
    engineer = pre.named_steps["features"]
    names = list(engineer.get_feature_names_out())
    if names:
        return names
    temp = ClinicalFeatureEngineer()
    temp.fit(pd.DataFrame(np.zeros((1, len(RAW_FEATURES))), columns=RAW_FEATURES))
    return list(temp.get_feature_names_out())


def save_outputs(final_models, best_model_name, cv_summary, validation_metrics, xgb_trials, best_params, feature_names):
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cv_summary.to_csv(OUTPUT_TABLE_DIR / "phase3_cv_summary.csv", index=False)
    validation_metrics.to_csv(OUTPUT_TABLE_DIR / "phase3_validation_metrics.csv", index=False)
    xgb_trials.to_csv(OUTPUT_TABLE_DIR / "phase3_xgboost_optuna_trials.csv", index=False)
    pd.DataFrame({"feature": feature_names}).to_csv(OUTPUT_TABLE_DIR / "phase3_feature_names.csv", index=False)

    best_params = dict(best_params)
    best_params["best_model_by_validation_auc"] = best_model_name
    (OUTPUT_TABLE_DIR / "phase3_best_params.json").write_text(json.dumps(best_params, indent=2))

    feature_metadata = {
        "raw_feature_count": len(RAW_FEATURES),
        "engineered_feature_count": len(feature_names),
        "raw_features": RAW_FEATURES,
        "engineered_features": feature_names,
        "preprocessing": "ClinicalFeatureEngineer -> SimpleImputer(median) -> StandardScaler",
        "training_policy": "Tune/CV on Cleveland train; final fit on Cleveland train + validation; do not touch Cleveland test, Hungarian, Swiss.",
    }
    (OUTPUT_TABLE_DIR / "phase3_feature_metadata.json").write_text(json.dumps(feature_metadata, indent=2))

    domain_metadata = {
        "method": "CORAL covariance alignment",
        "implemented_class": "CORALAdapter",
        "used_during_phase3_training": False,
        "purpose": "Phase 4 ablation only",
    }
    (OUTPUT_TABLE_DIR / "phase3_domain_adaptation_metadata.json").write_text(json.dumps(domain_metadata, indent=2))

    joblib.dump(final_models["Logistic Regression"], OUTPUT_MODEL_DIR / "phase3_logistic_regression.joblib")
    joblib.dump(final_models["XGBoost + Optuna"], OUTPUT_MODEL_DIR / "phase3_xgboost_optuna.joblib")
    joblib.dump(final_models["Small MLP"], OUTPUT_MODEL_DIR / "phase3_small_mlp.joblib")
    joblib.dump(final_models[best_model_name], OUTPUT_MODEL_DIR / "phase3_best_model.joblib")

    print("\nSaved Phase 3 outputs:")
    for path in [
        OUTPUT_TABLE_DIR / "phase3_cv_summary.csv",
        OUTPUT_TABLE_DIR / "phase3_validation_metrics.csv",
        OUTPUT_TABLE_DIR / "phase3_xgboost_optuna_trials.csv",
        OUTPUT_TABLE_DIR / "phase3_feature_names.csv",
        OUTPUT_TABLE_DIR / "phase3_best_params.json",
        OUTPUT_TABLE_DIR / "phase3_feature_metadata.json",
        OUTPUT_TABLE_DIR / "phase3_domain_adaptation_metadata.json",
        OUTPUT_MODEL_DIR / "phase3_logistic_regression.joblib",
        OUTPUT_MODEL_DIR / "phase3_xgboost_optuna.joblib",
        OUTPUT_MODEL_DIR / "phase3_small_mlp.joblib",
        OUTPUT_MODEL_DIR / "phase3_best_model.joblib",
    ]:
        print(path)


def run_modeling_v2():
    print("\nPhase 3 v2 — Stable Single-Process Modeling Runner")
    print("=" * 72)

    train_df = load_processed_split("cleveland_train.csv")
    val_df = load_processed_split("cleveland_validation.csv")

    X_train, y_train = get_xy(train_df)
    X_val, y_val = get_xy(val_df)

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    print(f"Cleveland train shape: {X_train.shape}")
    print(f"Cleveland validation shape: {X_val.shape}")
    print(f"Cleveland train positive rate: {y_train.mean():.2%}")
    print(f"Cleveland validation positive rate: {y_val.mean():.2%}")
    print("Cleveland test, Hungarian, and Swiss are not evaluated in Phase 3.")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    lr_best, lr_params, lr_cv = train_logistic_regression(X_train, y_train, cv)
    xgb_best, xgb_params, xgb_trials, xgb_cv = train_xgboost_optuna(X_train, y_train, cv)
    mlp_best, mlp_params, mlp_cv = train_mlp(X_train, y_train, cv)

    tuned_models = {
        "Logistic Regression": lr_best,
        "XGBoost + Optuna": xgb_best,
        "Small MLP": mlp_best,
    }

    cv_summary = pd.DataFrame([lr_cv, xgb_cv, mlp_cv]).sort_values("roc_auc_mean", ascending=False)

    validation_metrics = pd.DataFrame(
        [evaluate_validation(name, model, X_val, y_val) for name, model in tuned_models.items()]
    ).sort_values("roc_auc", ascending=False)

    best_model_name = validation_metrics.iloc[0]["model"]

    final_models = {
        name: refit_final_model(model, X_train_val, y_train_val)
        for name, model in tuned_models.items()
    }

    feature_names = get_feature_names(final_models["Logistic Regression"])

    best_params = {
        "logistic_regression": lr_params,
        "xgboost_optuna": xgb_params,
        "small_mlp": mlp_params,
    }

    print("\nPhase 3 CV summary:")
    print(cv_summary.to_string(index=False))

    print("\nCleveland validation metrics:")
    print(validation_metrics.to_string(index=False))

    print(f"\nSelected best model by validation ROC-AUC: {best_model_name}")

    save_outputs(
        final_models=final_models,
        best_model_name=best_model_name,
        cv_summary=cv_summary,
        validation_metrics=validation_metrics,
        xgb_trials=xgb_trials,
        best_params=best_params,
        feature_names=feature_names,
    )

    print("\nCompletion checks:")
    print(f"LR optimized: {bool(lr_params)}")
    print(f"XGBoost Optuna trials: {len(xgb_trials)}")
    print(f"MLP optimized: {bool(mlp_params)}")
    print(f"Engineered feature count: {len(feature_names)}")
    print("Domain adaptation module available: True")
    print("Held-out/external test sets untouched: True")

    return {
        "cv_summary": cv_summary,
        "validation_metrics": validation_metrics,
        "best_model_name": best_model_name,
        "final_models": final_models,
    }


run_modeling = run_modeling_v2


if __name__ == "__main__":
    run_modeling_v2()
