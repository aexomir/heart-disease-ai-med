
"""
Phase 2 v2 — EDA runner.

Reads the canonical Phase 1 v2 processed splits and saves EDA tables/figures.
EDA is based mainly on Cleveland train to avoid using validation/test evidence for
model-design decisions, while cross-site summaries compare train against external sites.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal",
]


def ensure_phase1_outputs() -> None:
    required = [
        "cleveland_train.csv", "cleveland_validation.csv", "cleveland_test.csv",
        "hungarian.csv", "swiss.csv",
    ]
    if all((PROCESSED_DATA_DIR / f).exists() for f in required):
        return

    phase1_path = PROJECT_ROOT / "src" / "01_dataset.py"
    spec = importlib.util.spec_from_file_location("phase1_dataset", phase1_path)
    phase1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase1)
    phase1.run_dataset_setup_v2()


def load_processed_splits():
    ensure_phase1_outputs()
    return {
        "cleveland_train": pd.read_csv(PROCESSED_DATA_DIR / "cleveland_train.csv"),
        "cleveland_validation": pd.read_csv(PROCESSED_DATA_DIR / "cleveland_validation.csv"),
        "cleveland_test": pd.read_csv(PROCESSED_DATA_DIR / "cleveland_test.csv"),
        "hungarian": pd.read_csv(PROCESSED_DATA_DIR / "hungarian.csv"),
        "swiss": pd.read_csv(PROCESSED_DATA_DIR / "swiss.csv"),
    }


def class_balance_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        counts = df["target"].value_counts().sort_index()
        rows.append({
            "split": name,
            "n_rows": int(len(df)),
            "n_negative": int(counts.get(0, 0)),
            "n_positive": int(counts.get(1, 0)),
            "positive_rate": float(df["target"].mean()),
        })
    return pd.DataFrame(rows)


def missingness_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        for col in FEATURE_COLUMNS:
            rows.append({
                "split": name,
                "feature": col,
                "missing_count": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()),
            })
    return pd.DataFrame(rows)


def is_low_cardinality(series: pd.Series, threshold: int = 10) -> bool:
    return series.dropna().nunique() <= threshold


def compare_feature_shift(source: pd.DataFrame, target: pd.DataFrame, feature: str) -> dict:
    a = pd.to_numeric(source[feature], errors="coerce").dropna()
    b = pd.to_numeric(target[feature], errors="coerce").dropna()

    if len(a) == 0 or len(b) == 0:
        return {
            "test": "not_enough_data",
            "statistic": np.nan,
            "p_value": np.nan,
            "source_mean": np.nan,
            "target_mean": np.nan,
            "mean_delta": np.nan,
        }

    if is_low_cardinality(pd.concat([a, b])):
        values = sorted(set(a.unique()).union(set(b.unique())))
        observed = np.array([
            [(a == v).sum() for v in values],
            [(b == v).sum() for v in values],
        ])
        try:
            stat, p, _, _ = stats.chi2_contingency(observed)
        except ValueError:
            stat, p = np.nan, np.nan
        test = "chi_square"
    else:
        stat, p = stats.ks_2samp(a, b)
        test = "ks_2samp"

    return {
        "test": test,
        "statistic": float(stat) if pd.notna(stat) else np.nan,
        "p_value": float(p) if pd.notna(p) else np.nan,
        "source_mean": float(a.mean()),
        "target_mean": float(b.mean()),
        "mean_delta": float(b.mean() - a.mean()),
    }


def distribution_shift_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source = splits["cleveland_train"]
    rows = []
    for target_name in ["cleveland_validation", "cleveland_test", "hungarian", "swiss"]:
        target = splits[target_name]
        for feature in FEATURE_COLUMNS:
            result = compare_feature_shift(source, target, feature)
            rows.append({
                "comparison": f"cleveland_train_vs_{target_name}",
                "feature": feature,
                **result,
            })

    out = pd.DataFrame(rows)
    out["minus_log10_p"] = -np.log10(out["p_value"].replace(0, np.nextafter(0, 1)))
    return out.sort_values(["comparison", "p_value"])


def save_class_balance_plot(class_balance: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    class_balance.set_index("split")[["n_negative", "n_positive"]].plot(kind="bar", ax=ax)
    ax.set_title("Class balance by split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of patients")
    ax.legend(["No disease", "Disease present"])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_FIGURE_DIR / "class_balance.png", dpi=200)
    plt.close(fig)


def save_missingness_heatmap(splits: dict[str, pd.DataFrame]) -> None:
    combined = pd.concat(
        [df.assign(split=name) for name, df in splits.items()],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(combined[FEATURE_COLUMNS].isna(), aspect="auto")
    ax.set_title("Missingness matrix across all splits")
    ax.set_xlabel("Features")
    ax.set_ylabel("Rows")
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90)
    plt.tight_layout()
    fig.savefig(OUTPUT_FIGURE_DIR / "missingness_heatmap.png", dpi=200)
    plt.close(fig)


def save_histograms(cleveland_train: pd.DataFrame) -> None:
    for feature in FEATURE_COLUMNS + ["target_original"]:
        if feature not in cleveland_train.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for label, label_name in [(0, "No disease"), (1, "Disease present")]:
            values = pd.to_numeric(
                cleveland_train.loc[cleveland_train["target"] == label, feature],
                errors="coerce",
            ).dropna()
            ax.hist(values, bins=20, alpha=0.6, label=label_name)
        ax.set_title(f"Cleveland train distribution by label: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUTPUT_FIGURE_DIR / f"hist_cleveland_train_{feature}.png", dpi=200)
        plt.close(fig)


def save_correlation_matrix(cleveland_train: pd.DataFrame) -> None:
    cols = FEATURE_COLUMNS + ["target"]
    corr = cleveland_train[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, aspect="auto")
    ax.set_title("Cleveland train correlation matrix")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(OUTPUT_FIGURE_DIR / "cleveland_train_correlation_matrix.png", dpi=200)
    plt.close(fig)


def save_shift_plot(shift: pd.DataFrame) -> None:
    top = shift.dropna(subset=["p_value"]).sort_values("p_value").head(12).copy()
    top["label"] = top["comparison"] + " — " + top["feature"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["label"], top["minus_log10_p"])
    ax.invert_yaxis()
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("Top distribution shifts relative to Cleveland train")
    plt.tight_layout()
    fig.savefig(OUTPUT_FIGURE_DIR / "top_distribution_shifts.png", dpi=200)
    plt.close(fig)


def run_eda_v2() -> dict[str, pd.DataFrame]:
    print("\nPhase 2 v2 — EDA")
    print("=" * 72)

    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_processed_splits()

    class_balance = class_balance_table(splits)
    missingness = missingness_summary(splits)
    shift = distribution_shift_summary(splits)

    class_balance.to_csv(OUTPUT_TABLE_DIR / "class_balance.csv", index=False)
    missingness.to_csv(OUTPUT_TABLE_DIR / "missingness_summary.csv", index=False)
    shift.to_csv(OUTPUT_TABLE_DIR / "distribution_shift_summary.csv", index=False)

    save_class_balance_plot(class_balance)
    save_missingness_heatmap(splits)
    save_histograms(splits["cleveland_train"])
    save_correlation_matrix(splits["cleveland_train"])
    save_shift_plot(shift)

    print("Saved EDA outputs:")
    print(OUTPUT_TABLE_DIR / "class_balance.csv")
    print(OUTPUT_TABLE_DIR / "missingness_summary.csv")
    print(OUTPUT_TABLE_DIR / "distribution_shift_summary.csv")
    print(OUTPUT_FIGURE_DIR)

    print("\nMost shifted features:")
    print(shift[["comparison", "feature", "p_value", "mean_delta"]].head(10).to_string(index=False))

    return {
        "class_balance": class_balance,
        "missingness_summary": missingness,
        "distribution_shift_summary": shift,
    }


if __name__ == "__main__":
    run_eda_v2()
