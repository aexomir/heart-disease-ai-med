"""
Phase 3 — Modeling, Feature Engineering, and Domain Adaptation
AI in Medicine Project: Heart Disease ML

This is the production runner for Phase 3.

What it does
------------
1. Loads Cleveland, Hungarian, and Swiss using Phase 1.
2. Reads Phase 2 EDA tables from outputs/tables/.
3. Adds clinically motivated engineered features.
4. Builds a shared preprocessing pipeline:
   feature engineering -> median imputation -> standard scaling.
5. Implements CORAL as an isolated domain-adaptation module.
6. Trains and tunes three mandatory models using Cleveland only:
   - Logistic Regression
   - XGBoost + Optuna, with at least 50 trials
   - Small MLP
7. Computes 5-fold Cleveland CV metrics.
8. Saves model checkpoints and tables for Phase 4.

Important leakage rule
----------------------
Hungarian and Swiss are loaded only to confirm availability and to prepare
Phase 4. They are NOT used for model fitting, hyperparameter tuning,
threshold selection, calibration, or CV scoring in Phase 3.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

try:
    import optuna
except ImportError as exc:
    raise ImportError("Optuna is required. Install with: pip install optuna") from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("XGBoost is required. Install with: pip install xgboost") from exc


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
XGB_OPTUNA_TRIALS = 50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

for directory in [OUTPUT_DIR, TABLE_DIR, MODEL_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

PHASE1_PATH = SRC_DIR / "01_dataset.py"

RAW_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

EDA_TABLE_NAMES = {
    "class_balance": "class_balance.csv",
    "distribution_shift": "distribution_shift_summary.csv",
    "missingness": "missingness_summary.csv",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def load_phase1_runner():
    """Load src/01_dataset.py using importlib."""
    if not PHASE1_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {PHASE1_PATH}. Expected Phase 1 runner at src/01_dataset.py"
        )

    spec = importlib.util.spec_from_file_location("phase1_dataset", PHASE1_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load module from {PHASE1_PATH}")
    spec.loader.exec_module(module)
    return module


def save_json(obj: dict, path: Path) -> None:
    """Save a JSON file with NumPy-safe conversion."""
    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return value

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def specificity_score(y_true, y_pred) -> float:
    """Specificity = true negative rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def summarize_cv_results(model_name: str, cv_results: dict, scoring_keys: List[str]) -> dict:
    """Convert sklearn cross_validate output into one summary row."""
    row = {"model": model_name}
    for metric in scoring_keys:
        values = cv_results[f"test_{metric}"]
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    return row


# ---------------------------------------------------------------------
# Phase 2 EDA outputs
# ---------------------------------------------------------------------

def load_eda_outputs() -> Dict[str, pd.DataFrame]:
    """Load fixed-name Phase 2 EDA tables if present."""
    eda_outputs: Dict[str, pd.DataFrame] = {}

    for key, filename in EDA_TABLE_NAMES.items():
        path = TABLE_DIR / filename
        if path.exists():
            eda_outputs[key] = pd.read_csv(path)
            print(f"Loaded EDA table: {path}")
        else:
            print(f"EDA table not found: {path}")

    return eda_outputs


def extract_shifted_features(eda_outputs: Dict[str, pd.DataFrame], top_k: int = 8) -> List[str]:
    """Extract most shifted features from distribution_shift_summary.csv."""
    if "distribution_shift" not in eda_outputs:
        return []

    shift_df = eda_outputs["distribution_shift"].copy()
    if "feature" not in shift_df.columns or "p_value" not in shift_df.columns:
        return []

    shift_df["p_value"] = pd.to_numeric(shift_df["p_value"], errors="coerce")
    shifted_features = (
        shift_df.dropna(subset=["p_value"])
        .sort_values("p_value")
        ["feature"]
        .astype(str)
        .drop_duplicates()
        .head(top_k)
        .tolist()
    )
    return shifted_features


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add clinically motivated interactions and a Framingham-inspired proxy."""

    def __init__(self):
        self.feature_names_in_: List[str] | None = None
        self.feature_names_out_: List[str] | None = None

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        self.feature_names_out_ = list(self.transform(X_df).columns)
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X).copy()

        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        X_df["age_x_thalach"] = X_df["age"] * X_df["thalach"]
        X_df["age_x_oldpeak"] = X_df["age"] * X_df["oldpeak"]
        X_df["trestbps_x_chol"] = X_df["trestbps"] * X_df["chol"]
        X_df["cp_x_thal"] = X_df["cp"] * X_df["thal"]
        X_df["exang_x_oldpeak"] = X_df["exang"] * X_df["oldpeak"]

        # Framingham-inspired proxy, not the true Framingham Risk Score.
        # HDL, smoking, and medication status are unavailable in UCI.
        age_component = X_df["age"] / 10.0
        male_component = X_df["sex"] * 2.0
        chol_component = (X_df["chol"] - 200.0) / 40.0
        bp_component = (X_df["trestbps"] - 120.0) / 20.0
        fbs_component = X_df["fbs"] * 1.5
        X_df["framingham_proxy"] = (
            age_component + male_component + chol_component + bp_component + fbs_component
        )

        return X_df

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
            if input_features is None:
                return np.array([])
            return np.array(input_features)
        return np.array(self.feature_names_out_)

    @staticmethod
    def _to_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=RAW_FEATURES[: X.shape[1]])


def make_preprocessor() -> Pipeline:
    """Shared preprocessing pipeline used by all models."""
    return Pipeline(
        steps=[
            ("features", ClinicalFeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def get_engineered_feature_names() -> List[str]:
    """Return engineered feature names from the feature engineering module."""
    sample = pd.DataFrame(np.zeros((1, len(RAW_FEATURES))), columns=RAW_FEATURES)
    fe = ClinicalFeatureEngineer().fit(sample)
    return list(fe.get_feature_names_out())


# ---------------------------------------------------------------------
# Domain adaptation
# ---------------------------------------------------------------------

class CORALAdapter:
    """CORAL covariance alignment for unsupervised domain adaptation.

    The adapter is isolated so Phase 4 can compare external evaluation with
    and without domain adaptation.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.source_mean_ = None
        self.target_mean_ = None
        self.source_transform_ = None
        self.target_transform_ = None

    def fit(self, X_source, X_target):
        Xs = np.asarray(X_source, dtype=float)
        Xt = np.asarray(X_target, dtype=float)

        self.source_mean_ = Xs.mean(axis=0)
        self.target_mean_ = Xt.mean(axis=0)

        Cs = np.cov(Xs, rowvar=False) + self.eps * np.eye(Xs.shape[1])
        Ct = np.cov(Xt, rowvar=False) + self.eps * np.eye(Xt.shape[1])

        self.source_transform_ = self._matrix_power(Cs, -0.5)
        self.target_transform_ = self._matrix_power(Ct, 0.5)
        return self

    def transform_source_to_target(self, X_source):
        if self.source_mean_ is None:
            raise RuntimeError("CORALAdapter must be fitted before transform.")
        Xs = np.asarray(X_source, dtype=float)
        Xs_centered = Xs - self.source_mean_
        return Xs_centered @ self.source_transform_ @ self.target_transform_ + self.target_mean_

    @staticmethod
    def _matrix_power(matrix, power):
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


def run_coral_smoke_test(cleveland: pd.DataFrame, hungarian: pd.DataFrame) -> dict:
    """Smoke-test CORAL shape compatibility without using labels."""
    preprocessor = make_preprocessor()
    X_source = preprocessor.fit_transform(cleveland[RAW_FEATURES])
    X_target = preprocessor.transform(hungarian[RAW_FEATURES])

    adapter = CORALAdapter()
    adapter.fit(X_source, X_target)
    X_aligned = adapter.transform_source_to_target(X_source)

    return {
        "processed_source_shape": list(X_source.shape),
        "processed_target_shape": list(X_target.shape),
        "aligned_source_shape": list(X_aligned.shape),
    }


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------

def make_lr_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def make_xgb_pipeline(params: dict) -> Pipeline:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )
    return Pipeline(steps=[("preprocess", make_preprocessor()), ("model", model)])


def make_mlp_pipeline(params: dict | None = None) -> Pipeline:
    default_params = {
        "max_iter": 1200,
        "early_stopping": True,
        "validation_fraction": 0.20,
        "n_iter_no_change": 40,
        "random_state": RANDOM_STATE,
    }
    if params:
        default_params.update(params)

    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", MLPClassifier(**default_params)),
        ]
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_logistic_regression(X: pd.DataFrame, y: pd.Series, cv, scoring: dict) -> Tuple[Pipeline, dict, dict]:
    print("\nTraining optimized Logistic Regression...")

    lr_param_grid = [
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
        estimator=make_lr_pipeline(),
        param_grid=lr_param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)

    cv_results = cross_validate(search.best_estimator_, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    summary = summarize_cv_results("Logistic Regression", cv_results, list(scoring.keys()))

    print(f"Best LR AUC: {search.best_score_:.6f}")
    print(f"Best LR params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, summary


def train_xgboost_optuna(X: pd.DataFrame, y: pd.Series, cv, scoring: dict) -> Tuple[Pipeline, dict, dict, pd.DataFrame]:
    print("\nTraining XGBoost with Optuna...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 450),
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
        pipe = make_xgb_pipeline(params)
        scores = cross_validate(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return float(scores["test_score"].mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=XGB_OPTUNA_TRIALS, show_progress_bar=True)

    if len(study.trials) < 50:
        raise RuntimeError("Professor requirement not met: XGBoost Optuna trials < 50.")

    best_model = make_xgb_pipeline(study.best_params)
    best_model.fit(X, y)

    cv_results = cross_validate(best_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    summary = summarize_cv_results("XGBoost + Optuna", cv_results, list(scoring.keys()))
    trials_df = study.trials_dataframe()

    print(f"Completed Optuna trials: {len(study.trials)}")
    print(f"Best XGBoost AUC: {study.best_value:.6f}")
    print(f"Best XGBoost params: {study.best_params}")

    xgb_metadata = {
        "best_params": study.best_params,
        "best_auc": float(study.best_value),
        "n_trials": len(study.trials),
    }

    return best_model, xgb_metadata, summary, trials_df


def fit_pipeline_with_sample_weight(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit sklearn Pipeline with class-balanced sample weights for final MLP model."""
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    pipeline.fit(X, y, model__sample_weight=sample_weight)
    return pipeline


def train_mlp(X: pd.DataFrame, y: pd.Series, cv, scoring: dict) -> Tuple[Pipeline, dict, dict]:
    print("\nTraining optimized small MLP...")

    # Search uses MLPClassifier's own early stopping. Final fit below uses sample weights.
    mlp_search_pipeline = make_mlp_pipeline()

    mlp_param_grid = {
        "model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8), (32, 16)],
        "model__activation": ["relu", "tanh"],
        "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "model__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "model__batch_size": [16, 32, 64],
        "model__solver": ["adam"],
    }

    search = RandomizedSearchCV(
        estimator=mlp_search_pipeline,
        param_distributions=mlp_param_grid,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)

    best_params = search.best_params_
    clean_params = {key.replace("model__", ""): value for key, value in best_params.items()}
    final_model = make_mlp_pipeline(clean_params)
    final_model = fit_pipeline_with_sample_weight(final_model, X, y)

    # Cross-validation with the selected hyperparameters and class-balanced sample weights.
    fold_results = {
        "roc_auc": [],
        "average_precision": [],
        "f1": [],
        "sensitivity": [],
        "specificity": [],
    }

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = make_mlp_pipeline(clean_params)
        fold_model = fit_pipeline_with_sample_weight(fold_model, X_train, y_train)

        y_proba = fold_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        fold_results["roc_auc"].append(roc_auc_score(y_test, y_proba))
        fold_results["average_precision"].append(average_precision_score(y_test, y_proba))
        fold_results["f1"].append(f1_score(y_test, y_pred))
        fold_results["sensitivity"].append(recall_score(y_test, y_pred))
        fold_results["specificity"].append(specificity_score(y_test, y_pred))

    summary = {"model": "Small MLP"}
    for metric, values in fold_results.items():
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))

    print("MLP fitted with class-balanced sample weights.")
    print(f"Best MLP AUC from search: {search.best_score_:.6f}")
    print(f"Best MLP params: {best_params}")

    return final_model, best_params, summary


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def run_modeling() -> Dict[str, object]:
    print("\nPhase 3 — Modeling")
    print("=" * 72)

    phase1 = load_phase1_runner()
    phase1_outputs = phase1.run_dataset_setup()

    cleveland = phase1_outputs["cleveland"].copy()
    hungarian = phase1_outputs["hungarian"].copy()
    swiss = phase1_outputs["swiss"].copy()

    eda_outputs = load_eda_outputs()
    shifted_features = extract_shifted_features(eda_outputs)
    print("\nMost shifted features from Phase 2:")
    print(shifted_features if shifted_features else "No shift features loaded.")

    X = cleveland[RAW_FEATURES].copy()
    y = cleveland["target"].copy()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "sensitivity": "recall",
        "specificity": make_scorer(specificity_score),
    }

    engineered_feature_names = get_engineered_feature_names()
    feature_metadata = {
        "raw_features": RAW_FEATURES,
        "engineered_features": engineered_feature_names,
        "n_raw_features": len(RAW_FEATURES),
        "n_engineered_features": len(engineered_feature_names),
        "shifted_features_from_eda": shifted_features,
        "framingham_proxy_note": (
            "This is a Framingham-inspired proxy, not the true Framingham Risk Score, "
            "because HDL, smoking, and blood-pressure medication status are unavailable."
        ),
    }

    coral_metadata = run_coral_smoke_test(cleveland, hungarian)

    lr_model, lr_params, lr_summary = train_logistic_regression(X, y, cv, scoring)
    xgb_model, xgb_metadata, xgb_summary, xgb_trials = train_xgboost_optuna(X, y, cv, scoring)
    mlp_model, mlp_params, mlp_summary = train_mlp(X, y, cv, scoring)

    cv_summary = pd.DataFrame([lr_summary, xgb_summary, mlp_summary]).sort_values(
        "roc_auc_mean", ascending=False
    )

    best_model_name = str(cv_summary.iloc[0]["model"])
    best_model_map = {
        "Logistic Regression": lr_model,
        "XGBoost + Optuna": xgb_model,
        "Small MLP": mlp_model,
    }
    best_model = best_model_map[best_model_name]

    model_params = {
        "logistic_regression": lr_params,
        "xgboost_optuna": xgb_metadata,
        "small_mlp": mlp_params,
        "best_internal_cv_model": best_model_name,
    }

    # Save tables.
    cv_summary_path = TABLE_DIR / "phase3_cv_summary.csv"
    xgb_trials_path = TABLE_DIR / "phase3_xgboost_optuna_trials.csv"
    feature_names_path = TABLE_DIR / "phase3_feature_names.csv"

    cv_summary.to_csv(cv_summary_path, index=False)
    xgb_trials.to_csv(xgb_trials_path, index=False)
    pd.DataFrame({"feature": engineered_feature_names}).to_csv(feature_names_path, index=False)

    # Save metadata.
    save_json(model_params, TABLE_DIR / "phase3_best_params.json")
    save_json(feature_metadata, TABLE_DIR / "phase3_feature_metadata.json")
    save_json(coral_metadata, TABLE_DIR / "phase3_domain_adaptation_metadata.json")

    # Save models.
    joblib.dump(lr_model, MODEL_DIR / "phase3_logistic_regression.joblib")
    joblib.dump(xgb_model, MODEL_DIR / "phase3_xgboost_optuna.joblib")
    joblib.dump(mlp_model, MODEL_DIR / "phase3_small_mlp.joblib")
    joblib.dump(best_model, MODEL_DIR / "phase3_best_model.joblib")

    # Save CORAL class metadata for Phase 4; the adapter is fitted per external site later.
    with (MODEL_DIR / "phase3_coral_adapter_class.pkl").open("wb") as f:
        pickle.dump(CORALAdapter, f)

    print("\n" + "=" * 72)
    print("Phase 3 CV summary")
    print("=" * 72)
    print(cv_summary.to_string(index=False))

    print("\nSaved Phase 3 outputs:")
    print(cv_summary_path)
    print(xgb_trials_path)
    print(feature_names_path)
    print(TABLE_DIR / "phase3_best_params.json")
    print(TABLE_DIR / "phase3_feature_metadata.json")
    print(TABLE_DIR / "phase3_domain_adaptation_metadata.json")
    print(MODEL_DIR / "phase3_logistic_regression.joblib")
    print(MODEL_DIR / "phase3_xgboost_optuna.joblib")
    print(MODEL_DIR / "phase3_small_mlp.joblib")
    print(MODEL_DIR / "phase3_best_model.joblib")

    print("\nCompletion checks:")
    print(f"LR optimized: {bool(lr_params)}")
    print(f"XGBoost Optuna trials: {xgb_metadata['n_trials']}")
    print(f"MLP optimized: {bool(mlp_params)}")
    print(f"Engineered feature count: {len(engineered_feature_names)}")
    print("Domain adaptation module available: True")
    print(f"External splits loaded but not evaluated in Phase 3: {hungarian.shape}, {swiss.shape}")

    return {
        "cv_summary": cv_summary,
        "best_model_name": best_model_name,
        "models": {
            "logistic_regression": lr_model,
            "xgboost_optuna": xgb_model,
            "small_mlp": mlp_model,
            "best_model": best_model,
        },
        "model_params": model_params,
        "feature_metadata": feature_metadata,
        "coral_metadata": coral_metadata,
        "datasets": {
            "cleveland": cleveland,
            "hungarian": hungarian,
            "swiss": swiss,
        },
    }


if __name__ == "__main__":
    run_modeling()
