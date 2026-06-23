"""
Phase 4 — Evaluation, Calibration, Explainability, and Domain Adaptation
AI in Medicine Project: Heart Disease ML

This is the production runner for Phase 4.

What it does
------------
1. Loads Phase 1 data using src/01_dataset.py.
2. Loads Phase 3 trained model checkpoints from outputs/models/.
3. Evaluates Logistic Regression, XGBoost, and Small MLP on Hungarian and Swiss.
4. Computes AUC-ROC, AUC-PR, F1, sensitivity, and specificity.
5. Runs DeLong tests between the top two external models.
6. Fits Platt and isotonic calibration on a Cleveland holdout only.
7. Computes reliability/ECE metrics and saves reliability diagrams.
8. Runs CORAL domain-adaptation ablation on external splits.
9. Generates SHAP TreeExplainer results for the XGBoost model.

Important leakage rule
----------------------
No model, imputer, scaler, threshold, calibrator, or hyperparameter is fitted
using Hungarian or Swiss labels. Hungarian and Swiss are used only for external
evaluation. CORAL uses external features without labels.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from sklearn.frozen import FrozenEstimator
except Exception:  # Older scikit-learn fallback
    FrozenEstimator = None

try:
    import shap
except ImportError:
    shap = None

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
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

MODEL_FILES = {
    "Logistic Regression": "phase3_logistic_regression.joblib",
    "XGBoost + Optuna": "phase3_xgboost_optuna.joblib",
    "Small MLP": "phase3_small_mlp.joblib",
}


# ---------------------------------------------------------------------
# Classes required for loading Phase 3 joblib pipelines
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


class CORALAdapter:
    """CORAL covariance alignment for unsupervised domain adaptation."""

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


# ---------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------

def load_phase1_runner():
    if not PHASE1_PATH.exists():
        raise FileNotFoundError(f"Cannot find Phase 1 runner at {PHASE1_PATH}")
    spec = importlib.util.spec_from_file_location("phase1_dataset", PHASE1_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load module from {PHASE1_PATH}")
    spec.loader.exec_module(module)
    return module


def load_phase3_models() -> Dict[str, Pipeline]:
    models = {}
    for model_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model checkpoint: {path}. Run src/03_modeling.py first."
            )
        models[model_name] = joblib.load(path)
        print(f"Loaded model: {model_name} <- {path}")
    return models


def save_json(obj: dict, path: Path) -> None:
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


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def specificity_score(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_score(y_true, y_pred)),
        "threshold": threshold,
    }


def predict_proba_positive(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------
# DeLong test implementation
# ---------------------------------------------------------------------

def compute_midrank(x):
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]
    midranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x), dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true, pred_one, pred_two):
    y_true = np.asarray(y_true).astype(int)
    pred_one = np.asarray(pred_one, dtype=float)
    pred_two = np.asarray(pred_two, dtype=float)

    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true))
    if label_1_count == 0 or label_1_count == len(y_true):
        return np.array([np.nan, np.nan]), np.nan, np.nan, np.nan

    preds = np.vstack([pred_one, pred_two])[:, order]
    aucs, cov = fast_delong(preds, label_1_count)

    cov = np.atleast_2d(cov)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        return aucs, diff, np.nan, np.nan
    z = abs(diff) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return aucs, diff, float(z), float(p)


# ---------------------------------------------------------------------
# External evaluation
# ---------------------------------------------------------------------

def evaluate_external_models(models: Dict[str, Pipeline], external_sets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, dict]:
    rows = []
    probabilities = {}

    for split_name, df in external_sets.items():
        X_ext = df[RAW_FEATURES].copy()
        y_ext = df["target"].copy()
        probabilities[split_name] = {}

        for model_name, model in models.items():
            y_prob = predict_proba_positive(model, X_ext)
            probabilities[split_name][model_name] = y_prob
            metric_row = compute_metrics(y_ext, y_prob)
            metric_row.update({"split": split_name, "model": model_name, "mode": "standard"})
            rows.append(metric_row)

    metrics_df = pd.DataFrame(rows).sort_values(["split", "auc_roc"], ascending=[True, False])
    return metrics_df, probabilities


def run_delong_tests(external_metrics: pd.DataFrame, probabilities: dict, external_sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for split_name, df in external_sets.items():
        ranking = (
            external_metrics[external_metrics["split"] == split_name]
            .sort_values("auc_roc", ascending=False)
            ["model"]
            .tolist()
        )
        if len(ranking) < 2:
            continue

        model_a, model_b = ranking[0], ranking[1]
        y_true = df["target"].to_numpy()
        pred_a = probabilities[split_name][model_a]
        pred_b = probabilities[split_name][model_b]
        aucs, diff, z, p = delong_roc_test(y_true, pred_a, pred_b)

        rows.append({
            "split": split_name,
            "model_a": model_a,
            "model_b": model_b,
            "auc_a": float(aucs[0]) if not np.isnan(aucs[0]) else np.nan,
            "auc_b": float(aucs[1]) if not np.isnan(aucs[1]) else np.nan,
            "auc_difference_a_minus_b": diff,
            "z": z,
            "p_value": p,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += abs(bin_accuracy - bin_confidence) * np.mean(mask)
    return float(ece)


def fit_model_maybe_weighted(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    final_estimator = model.named_steps.get("model")
    if isinstance(final_estimator, MLPClassifier):
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, model__sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return model


def make_calibrated_prefit_model(prefit_model, X_cal, y_cal, method):
    # New scikit-learn API
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(prefit_model),
            method=method,
            cv=None,
        )
        calibrator.fit(X_cal, y_cal)
        return calibrator

    # Older scikit-learn fallback
    try:
        calibrator = CalibratedClassifierCV(estimator=prefit_model, method=method, cv="prefit")
    except TypeError:
        calibrator = CalibratedClassifierCV(base_estimator=prefit_model, method=method, cv="prefit")
    calibrator.fit(X_cal, y_cal)
    return calibrator


def build_calibrated_models(models: Dict[str, Pipeline], cleveland: pd.DataFrame) -> Dict[Tuple[str, str], Pipeline]:
    X_cleveland = cleveland[RAW_FEATURES].copy()
    y_cleveland = cleveland["target"].copy()

    X_train_cal, X_holdout_cal, y_train_cal, y_holdout_cal = train_test_split(
        X_cleveland,
        y_cleveland,
        test_size=0.20,
        stratify=y_cleveland,
        random_state=RANDOM_STATE,
    )

    calibrated_models = {}
    for model_name, final_model in models.items():
        base = clone(final_model)
        base = fit_model_maybe_weighted(base, X_train_cal, y_train_cal)
        calibrated_models[(model_name, "uncalibrated")] = base
        calibrated_models[(model_name, "platt_sigmoid")] = make_calibrated_prefit_model(
            base, X_holdout_cal, y_holdout_cal, method="sigmoid"
        )
        calibrated_models[(model_name, "isotonic")] = make_calibrated_prefit_model(
            base, X_holdout_cal, y_holdout_cal, method="isotonic"
        )

    return calibrated_models


def evaluate_calibration(calibrated_models: dict, external_sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, df in external_sets.items():
        X_ext = df[RAW_FEATURES].copy()
        y_ext = df["target"].copy()
        for (model_name, calibration_method), model in calibrated_models.items():
            y_prob = predict_proba_positive(model, X_ext)
            metric_row = compute_metrics(y_ext, y_prob)
            metric_row.update({
                "split": split_name,
                "model": model_name,
                "calibration": calibration_method,
                "ece_10": expected_calibration_error(y_ext, y_prob, n_bins=10),
                "brier_like_mean_squared_error": float(np.mean((np.asarray(y_ext) - y_prob) ** 2)),
            })
            rows.append(metric_row)
    return pd.DataFrame(rows).sort_values(["split", "model", "ece_10"])


def save_reliability_diagrams(calibrated_models: dict, external_sets: Dict[str, pd.DataFrame]) -> List[Path]:
    saved_paths = []
    for split_name, df in external_sets.items():
        X_ext = df[RAW_FEATURES].copy()
        y_ext = df["target"].copy()

        for model_name in MODEL_FILES.keys():
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

            for method in ["uncalibrated", "platt_sigmoid", "isotonic"]:
                model = calibrated_models[(model_name, method)]
                y_prob = predict_proba_positive(model, X_ext)
                frac_pos, mean_pred = calibration_curve(y_ext, y_prob, n_bins=10, strategy="uniform")
                ece = expected_calibration_error(y_ext, y_prob, n_bins=10)
                ax.plot(mean_pred, frac_pos, marker="o", label=f"{method} | ECE={ece:.3f}")

            ax.set_title(f"Reliability diagram — {model_name} on {split_name}")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed positive fraction")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            fig.tight_layout()

            safe_model = model_name.lower().replace(" ", "_").replace("+", "plus")
            path = FIGURE_DIR / f"phase4_reliability_{split_name}_{safe_model}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            saved_paths.append(path)
    return saved_paths


# ---------------------------------------------------------------------
# Domain adaptation ablation
# ---------------------------------------------------------------------

def predict_with_coral_target_to_source(model: Pipeline, X_source_raw: pd.DataFrame, X_target_raw: pd.DataFrame) -> np.ndarray:
    """Predict target examples after aligning target processed features to source covariance.

    The model was trained on Cleveland/source processed features, so for external
    testing we transform target features toward the source feature distribution.
    No target labels are used.
    """
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]

    X_source_processed = preprocessor.transform(X_source_raw)
    X_target_processed = preprocessor.transform(X_target_raw)

    # Transform target -> source by fitting CORAL with source=target, target=source.
    adapter = CORALAdapter()
    adapter.fit(X_target_processed, X_source_processed)
    X_target_aligned = adapter.transform_source_to_target(X_target_processed)

    return estimator.predict_proba(X_target_aligned)[:, 1]


def run_domain_adaptation_ablation(models: Dict[str, Pipeline], cleveland: pd.DataFrame, external_sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    X_source = cleveland[RAW_FEATURES].copy()

    for split_name, df in external_sets.items():
        X_target = df[RAW_FEATURES].copy()
        y_target = df["target"].copy()

        for model_name, model in models.items():
            y_prob_standard = predict_proba_positive(model, X_target)
            standard = compute_metrics(y_target, y_prob_standard)
            standard.update({"split": split_name, "model": model_name, "domain_adaptation": "none"})
            rows.append(standard)

            y_prob_coral = predict_with_coral_target_to_source(model, X_source, X_target)
            coral = compute_metrics(y_target, y_prob_coral)
            coral.update({"split": split_name, "model": model_name, "domain_adaptation": "coral_target_to_source"})
            rows.append(coral)

    ablation = pd.DataFrame(rows)

    delta_rows = []
    for (split_name, model_name), group in ablation.groupby(["split", "model"]):
        if set(group["domain_adaptation"]) >= {"none", "coral_target_to_source"}:
            base = group[group["domain_adaptation"] == "none"].iloc[0]
            coral = group[group["domain_adaptation"] == "coral_target_to_source"].iloc[0]
            delta_rows.append({
                "split": split_name,
                "model": model_name,
                "auc_roc_delta_coral_minus_none": coral["auc_roc"] - base["auc_roc"],
                "auc_pr_delta_coral_minus_none": coral["auc_pr"] - base["auc_pr"],
                "f1_delta_coral_minus_none": coral["f1"] - base["f1"],
                "sensitivity_delta_coral_minus_none": coral["sensitivity"] - base["sensitivity"],
                "specificity_delta_coral_minus_none": coral["specificity"] - base["specificity"],
            })

    delta_df = pd.DataFrame(delta_rows)
    return ablation, delta_df


# ---------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------

def get_engineered_feature_names_from_pipeline(model: Pipeline) -> List[str]:
    feature_engineer = model.named_steps["preprocess"].named_steps["features"]
    names = list(feature_engineer.get_feature_names_out())
    if not names:
        # Fallback if old pickle did not retain feature names.
        sample = pd.DataFrame(np.zeros((1, len(RAW_FEATURES))), columns=RAW_FEATURES)
        names = list(ClinicalFeatureEngineer().fit(sample).get_feature_names_out())
    return names


def run_shap_tree_explainer(models: Dict[str, Pipeline], cleveland: pd.DataFrame) -> pd.DataFrame:
    if shap is None:
        print("SHAP is not installed. Skipping SHAP analysis. Install with: pip install shap")
        return pd.DataFrame()

    if "XGBoost + Optuna" not in models:
        print("XGBoost model not found. Skipping TreeExplainer.")
        return pd.DataFrame()

    model = models["XGBoost + Optuna"]
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    feature_names = get_engineered_feature_names_from_pipeline(model)

    X_explain_raw = cleveland[RAW_FEATURES].copy()
    X_explain_processed = preprocessor.transform(X_explain_raw)
    X_explain_df = pd.DataFrame(X_explain_processed, columns=feature_names)

    # Keep runtime reasonable on small machines.
    if len(X_explain_df) > 300:
        X_explain_df = X_explain_df.sample(n=300, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_explain_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_explain_df, show=False, max_display=15)
    plt.tight_layout()
    summary_path = FIGURE_DIR / "phase4_shap_summary_xgboost.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    top = importance.head(15).sort_values("mean_abs_shap")
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_title("SHAP feature importance — XGBoost TreeExplainer")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    bar_path = FIGURE_DIR / "phase4_shap_feature_importance_xgboost.png"
    fig.savefig(bar_path, dpi=200)
    plt.close(fig)

    print(f"Saved SHAP summary plot: {summary_path}")
    print(f"Saved SHAP bar plot: {bar_path}")
    return importance


# ---------------------------------------------------------------------
# Plotting external ROC/PR curves
# ---------------------------------------------------------------------

def save_external_curves(models: Dict[str, Pipeline], external_sets: Dict[str, pd.DataFrame]) -> List[Path]:
    saved_paths = []
    for split_name, df in external_sets.items():
        X_ext = df[RAW_FEATURES].copy()
        y_ext = df["target"].copy()

        fig, ax = plt.subplots(figsize=(6, 5))
        for model_name, model in models.items():
            y_prob = predict_proba_positive(model, X_ext)
            fpr, tpr, _ = roc_curve(y_ext, y_prob)
            auc = roc_auc_score(y_ext, y_prob)
            ax.plot(fpr, tpr, label=f"{model_name} AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f"ROC curves — {split_name}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIGURE_DIR / f"phase4_roc_curves_{split_name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved_paths.append(path)

        fig, ax = plt.subplots(figsize=(6, 5))
        for model_name, model in models.items():
            y_prob = predict_proba_positive(model, X_ext)
            precision, recall, _ = precision_recall_curve(y_ext, y_prob)
            ap = average_precision_score(y_ext, y_prob)
            ax.plot(recall, precision, label=f"{model_name} AP={ap:.3f}")
        ax.set_title(f"Precision-recall curves — {split_name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIGURE_DIR / f"phase4_pr_curves_{split_name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def run_evaluation() -> Dict[str, object]:
    print("\nPhase 4 — Evaluation")
    print("=" * 72)

    phase1 = load_phase1_runner()
    phase1_outputs = phase1.run_dataset_setup()

    cleveland = phase1_outputs["cleveland"].copy()
    hungarian = phase1_outputs["hungarian"].copy()
    swiss = phase1_outputs["swiss"].copy()
    external_sets = {"hungarian": hungarian, "swiss": swiss}

    models = load_phase3_models()

    print("\nEvaluating external test sets...")
    external_metrics, probabilities = evaluate_external_models(models, external_sets)
    external_metrics_path = TABLE_DIR / "phase4_external_metrics.csv"
    external_metrics.to_csv(external_metrics_path, index=False)
    print(external_metrics.to_string(index=False))

    print("\nRunning DeLong tests between top two models per external split...")
    delong_df = run_delong_tests(external_metrics, probabilities, external_sets)
    delong_path = TABLE_DIR / "phase4_delong_results.csv"
    delong_df.to_csv(delong_path, index=False)
    print(delong_df.to_string(index=False))

    print("\nRunning calibration analysis using Cleveland holdout only...")
    calibrated_models = build_calibrated_models(models, cleveland)
    calibration_df = evaluate_calibration(calibrated_models, external_sets)
    calibration_path = TABLE_DIR / "phase4_calibration_metrics.csv"
    calibration_df.to_csv(calibration_path, index=False)
    reliability_paths = save_reliability_diagrams(calibrated_models, external_sets)
    print(calibration_df.to_string(index=False))

    print("\nRunning CORAL domain-adaptation ablation...")
    ablation_df, delta_df = run_domain_adaptation_ablation(models, cleveland, external_sets)
    ablation_path = TABLE_DIR / "phase4_domain_adaptation_ablation.csv"
    delta_path = TABLE_DIR / "phase4_domain_adaptation_delta.csv"
    ablation_df.to_csv(ablation_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    print(delta_df.to_string(index=False))

    print("\nRunning SHAP TreeExplainer for XGBoost...")
    shap_importance = run_shap_tree_explainer(models, cleveland)
    shap_path = TABLE_DIR / "phase4_shap_feature_importance.csv"
    shap_importance.to_csv(shap_path, index=False)
    if not shap_importance.empty:
        print(shap_importance.head(15).to_string(index=False))

    print("\nSaving external ROC and PR curves...")
    curve_paths = save_external_curves(models, external_sets)

    output_manifest = {
        "external_metrics": str(external_metrics_path),
        "delong_results": str(delong_path),
        "calibration_metrics": str(calibration_path),
        "domain_adaptation_ablation": str(ablation_path),
        "domain_adaptation_delta": str(delta_path),
        "shap_feature_importance": str(shap_path),
        "reliability_diagrams": [str(p) for p in reliability_paths],
        "external_curves": [str(p) for p in curve_paths],
    }
    manifest_path = TABLE_DIR / "phase4_output_manifest.json"
    save_json(output_manifest, manifest_path)

    print("\nSaved Phase 4 outputs:")
    for key, value in output_manifest.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} files")
        else:
            print(value)
    print(manifest_path)

    print("\nCompletion checks:")
    print("Performance metrics computed for all models and external splits: True")
    print(f"DeLong tests saved: {delong_path.exists()}")
    print(f"Calibration metrics saved: {calibration_path.exists()}")
    print(f"Domain adaptation ablation saved: {ablation_path.exists()}")
    print(f"SHAP feature importance saved: {shap_path.exists()}")

    return {
        "external_metrics": external_metrics,
        "delong_results": delong_df,
        "calibration_metrics": calibration_df,
        "domain_adaptation_ablation": ablation_df,
        "domain_adaptation_delta": delta_df,
        "shap_feature_importance": shap_importance,
        "manifest": output_manifest,
    }


if __name__ == "__main__":
    run_evaluation()
