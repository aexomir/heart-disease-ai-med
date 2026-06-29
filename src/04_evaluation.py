"""
Phase 4 v2 — Stable Evaluation Runner
AI in Medicine Project: Heart Disease ML

Fix included:
- Imports src/03_modeling.py safely by registering it in sys.modules before execution.
- This prevents KeyError: 'phase3_modeling' and allows joblib to load custom classes.

Evaluates:
- Cleveland held-out test
- Hungarian external test
- Swiss external test

Produces:
- performance metrics
- DeLong tests
- calibration metrics and reliability plots
- CORAL domain-adaptation ablation
- SHAP TreeExplainer for XGBoost
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

try:
    from sklearn.frozen import FrozenEstimator
except Exception:
    FrozenEstimator = None

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

RAW_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal",
]


def import_module_from_src(module_filename: str, module_name: str):
    """Import a src module safely.

    Critical detail: we register the module in sys.modules before exec_module().
    This is required because src/03_modeling.py defines custom classes and gives
    them the stable module name 'phase3_modeling' for joblib serialization.
    """
    path = PROJECT_ROOT / "src" / module_filename

    if not path.exists():
        raise FileNotFoundError(f"Cannot find required module: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_phase3_outputs():
    required = [
        OUTPUT_MODEL_DIR / "phase3_logistic_regression.joblib",
        OUTPUT_MODEL_DIR / "phase3_xgboost_optuna.joblib",
        OUTPUT_MODEL_DIR / "phase3_small_mlp.joblib",
    ]

    if all(path.exists() for path in required):
        return

    print("Phase 3 model files missing. Running Phase 3 first...")
    phase3 = import_module_from_src("03_modeling.py", "phase3_modeling")
    phase3.run_modeling_v2()


def load_split(filename: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed split: {path}. Run src/01_dataset.py first."
        )

    df = pd.read_csv(path)

    for col in RAW_FEATURES + ["target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["target"] = df["target"].astype(int)
    return df


def get_xy(df: pd.DataFrame):
    return df[RAW_FEATURES].copy(), df["target"].copy()


def load_models():
    # Register custom classes from Phase 3 before joblib.load().
    import_module_from_src("03_modeling.py", "phase3_modeling")
    ensure_phase3_outputs()

    return {
        "Logistic Regression": joblib.load(OUTPUT_MODEL_DIR / "phase3_logistic_regression.joblib"),
        "XGBoost + Optuna": joblib.load(OUTPUT_MODEL_DIR / "phase3_xgboost_optuna.joblib"),
        "Small MLP": joblib.load(OUTPUT_MODEL_DIR / "phase3_small_mlp.joblib"),
    }


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan


def evaluate_predictions(y_true, prob):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = (prob >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "average_precision": float(average_precision_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": specificity_score(y_true, pred),
    }


def evaluate_models(models, eval_splits):
    rows = []
    predictions = {}

    for split_name, df in eval_splits.items():
        X, y = get_xy(df)

        for model_name, model in models.items():
            prob = model.predict_proba(X)[:, 1]
            predictions[(split_name, model_name)] = prob

            rows.append({
                "split": split_name,
                "model": model_name,
                **evaluate_predictions(y, prob),
            })

    return pd.DataFrame(rows), predictions


# ---------------------------------------------------------------------
# DeLong test
# ---------------------------------------------------------------------
def compute_midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    if m == 0 or n == 0:
        raise ValueError("DeLong test requires both positive and negative labels.")

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

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    delong_cov = sx / m + sy / n
    return aucs, np.atleast_2d(delong_cov)


def delong_roc_test(y_true, pred_one, pred_two):
    y_true = np.asarray(y_true)
    pred_one = np.asarray(pred_one)
    pred_two = np.asarray(pred_two)

    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true))

    preds = np.vstack([pred_one, pred_two])[:, order]
    aucs, cov = fast_delong(preds, label_1_count)

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]

    if var <= 0:
        return aucs, float(diff), np.nan, np.nan

    z = abs(diff) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))

    return aucs, float(diff), float(z), float(p)


def run_delong_tests(metrics, predictions, eval_splits):
    rows = []

    for split_name, df in eval_splits.items():
        y = df["target"].values

        if len(np.unique(y)) < 2:
            continue

        split_metrics = metrics[metrics["split"] == split_name].sort_values(
            "roc_auc", ascending=False
        )

        if len(split_metrics) < 2:
            continue

        model_1 = split_metrics.iloc[0]["model"]
        model_2 = split_metrics.iloc[1]["model"]

        try:
            aucs, diff, z, p = delong_roc_test(
                y,
                predictions[(split_name, model_1)],
                predictions[(split_name, model_2)],
            )
        except Exception as exc:
            aucs, diff, z, p = [np.nan, np.nan], np.nan, np.nan, np.nan
            print(f"DeLong failed for {split_name}: {exc}")

        rows.append({
            "split": split_name,
            "model_1": model_1,
            "model_2": model_2,
            "auc_1": float(aucs[0]),
            "auc_2": float(aucs[1]),
            "auc_diff": float(diff),
            "z": z,
            "p_value": p,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
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


def make_calibrated_prefit_model(prefit_model, X_cal, y_cal, method):
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(prefit_model),
            method=method,
            cv=None,
        )
    else:
        calibrator = CalibratedClassifierCV(
            estimator=prefit_model,
            method=method,
            cv="prefit",
        )

    calibrator.fit(X_cal, y_cal)
    return calibrator


def run_calibration(models, train_df, validation_df, eval_splits):
    X_train, y_train = get_xy(train_df)
    X_val, y_val = get_xy(validation_df)

    calibrated_models = {}

    for model_name, final_model in models.items():
        base = clone(final_model)
        base.fit(X_train, y_train)

        calibrated_models[(model_name, "uncalibrated")] = base
        calibrated_models[(model_name, "platt_sigmoid")] = make_calibrated_prefit_model(
            base, X_val, y_val, method="sigmoid"
        )
        calibrated_models[(model_name, "isotonic")] = make_calibrated_prefit_model(
            base, X_val, y_val, method="isotonic"
        )

    rows = []

    for split_name, df in eval_splits.items():
        X_eval, y_eval = get_xy(df)

        for (model_name, calibration), model in calibrated_models.items():
            prob = model.predict_proba(X_eval)[:, 1]
            rows.append({
                "split": split_name,
                "model": model_name,
                "calibration": calibration,
                "ece": expected_calibration_error(y_eval, prob),
                **evaluate_predictions(y_eval, prob),
            })

    return pd.DataFrame(rows), calibrated_models


def save_reliability_diagrams(calibrated_models, eval_splits):
    for split_name, df in eval_splits.items():
        X_eval, y_eval = get_xy(df)

        for model_name in sorted({key[0] for key in calibrated_models.keys()}):
            fig, ax = plt.subplots(figsize=(5, 5))

            for calibration in ["uncalibrated", "platt_sigmoid", "isotonic"]:
                model = calibrated_models[(model_name, calibration)]
                prob = model.predict_proba(X_eval)[:, 1]

                try:
                    frac_pos, mean_pred = calibration_curve(
                        y_eval, prob, n_bins=10, strategy="uniform"
                    )
                    ax.plot(mean_pred, frac_pos, marker="o", label=calibration)
                except Exception as exc:
                    print(f"Reliability curve failed for {model_name}/{split_name}/{calibration}: {exc}")

            ax.plot([0, 1], [0, 1], linestyle="--", label="perfect")
            ax.set_title(f"Reliability: {model_name} on {split_name}")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction positive")
            ax.legend()
            plt.tight_layout()

            safe_model = model_name.lower().replace(" ", "_").replace("+", "plus")
            fig.savefig(
                OUTPUT_FIGURE_DIR / f"reliability_{split_name}_{safe_model}.png",
                dpi=200,
            )
            plt.close(fig)


# ---------------------------------------------------------------------
# CORAL domain adaptation ablation
# ---------------------------------------------------------------------
def run_coral_ablation(models, train_validation_df, eval_splits):
    phase3 = import_module_from_src("03_modeling.py", "phase3_modeling")
    rows = []

    X_source, _ = get_xy(train_validation_df)

    for split_name, df in eval_splits.items():
        X_target, y_target = get_xy(df)

        for model_name, pipe in models.items():
            pre = pipe.named_steps["preprocess"]
            clf = pipe.named_steps["model"]

            X_source_processed = pre.transform(X_source)
            X_target_processed = pre.transform(X_target)

            coral = phase3.CORALAdapter()
            coral.fit(X_source_processed, X_target_processed)

            X_target_coral = coral.transform_target_to_source(X_target_processed)

            prob_no_adapt = pipe.predict_proba(X_target)[:, 1]
            prob_coral = clf.predict_proba(X_target_coral)[:, 1]

            no_adapt = evaluate_predictions(y_target, prob_no_adapt)
            adapted = evaluate_predictions(y_target, prob_coral)

            for metric in ["roc_auc", "average_precision", "f1", "sensitivity", "specificity"]:
                rows.append({
                    "split": split_name,
                    "model": model_name,
                    "metric": metric,
                    "without_coral": no_adapt[metric],
                    "with_coral": adapted[metric],
                    "delta": adapted[metric] - no_adapt[metric],
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------
def run_shap_xgboost(models, train_validation_df, max_samples=150):
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Skipping SHAP.")
        return pd.DataFrame()

    if "XGBoost + Optuna" not in models:
        print("XGBoost model not found. Skipping SHAP.")
        return pd.DataFrame()

    xgb_pipe = models["XGBoost + Optuna"]
    pre = xgb_pipe.named_steps["preprocess"]
    model = xgb_pipe.named_steps["model"]

    X_train_val, _ = get_xy(train_validation_df)

    if len(X_train_val) > max_samples:
        X_sample = X_train_val.sample(max_samples, random_state=RANDOM_STATE)
    else:
        X_sample = X_train_val.copy()

    X_processed = pre.transform(X_sample)
    feature_names = list(pre.named_steps["features"].get_feature_names_out())

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)

        shap.summary_plot(
            shap_values,
            X_processed,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            OUTPUT_FIGURE_DIR / "phase4_shap_summary_xgboost.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        importance = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        return importance

    except Exception as exc:
        print(f"SHAP failed: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_evaluation_v2():
    print("\nPhase 4 v2 — Stable Evaluation")
    print("=" * 72)

    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("cleveland_train.csv")
    validation_df = load_split("cleveland_validation.csv")
    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)

    eval_splits = {
        "cleveland_test": load_split("cleveland_test.csv"),
        "hungarian": load_split("hungarian.csv"),
        "swiss": load_split("swiss.csv"),
    }

    models = load_models()

    metrics, predictions = evaluate_models(models, eval_splits)
    delong = run_delong_tests(metrics, predictions, eval_splits)

    calibration_metrics, calibrated_models = run_calibration(
        models, train_df, validation_df, eval_splits
    )
    save_reliability_diagrams(calibrated_models, eval_splits)

    coral_ablation = run_coral_ablation(models, train_validation_df, eval_splits)
    shap_importance = run_shap_xgboost(models, train_validation_df)

    metrics.to_csv(OUTPUT_TABLE_DIR / "phase4_external_metrics.csv", index=False)
    delong.to_csv(OUTPUT_TABLE_DIR / "phase4_delong_tests.csv", index=False)
    calibration_metrics.to_csv(OUTPUT_TABLE_DIR / "phase4_calibration_metrics.csv", index=False)
    coral_ablation.to_csv(OUTPUT_TABLE_DIR / "phase4_domain_adaptation_ablation.csv", index=False)

    if not shap_importance.empty:
        shap_importance.to_csv(
            OUTPUT_TABLE_DIR / "phase4_shap_feature_importance_xgboost.csv",
            index=False,
        )

    print("\nExternal / held-out metrics:")
    print(metrics.sort_values(["split", "roc_auc"], ascending=[True, False]).to_string(index=False))

    print("\nDeLong tests:")
    if delong.empty:
        print("No DeLong tests available.")
    else:
        print(delong.to_string(index=False))

    print("\nSaved Phase 4 outputs:")
    for path in [
        OUTPUT_TABLE_DIR / "phase4_external_metrics.csv",
        OUTPUT_TABLE_DIR / "phase4_delong_tests.csv",
        OUTPUT_TABLE_DIR / "phase4_calibration_metrics.csv",
        OUTPUT_TABLE_DIR / "phase4_domain_adaptation_ablation.csv",
    ]:
        print(path)

    return {
        "metrics": metrics,
        "delong": delong,
        "calibration_metrics": calibration_metrics,
        "coral_ablation": coral_ablation,
        "shap_importance": shap_importance,
    }


run_evaluation = run_evaluation_v2


if __name__ == "__main__":
    run_evaluation_v2()
