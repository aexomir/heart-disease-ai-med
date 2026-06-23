"""
Phase 2 - Exploratory Data Analysis (EDA)

AI in Medicine Project
Heart Disease Classification

Responsibilities
----------------
1. Load Cleveland, Hungarian, and Swiss splits using the Phase 1 data module.
2. Check class balance, missingness, feature distributions, and correlations.
3. Quantify cross-site distribution shift between Cleveland and external sites.
4. Save EDA tables and figures to outputs/.
5. Return all important tables for use in notebooks and later phases.

Expected project usage
----------------------
Run from the project root:

    python src/02_eda.py

This file assumes src/01-dataset.py exists and exposes run_dataset_setup().
Because the filename contains a hyphen, this module loads it by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

RANDOM_STATE = 42
TARGET_COL = "target"


# ---------------------------------------------------------------------
# Paths / setup
# ---------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create output folders used by the EDA phase."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_phase1_runner():
    """
    Load src/01-dataset.py by path.

    The filename uses a hyphen, so it cannot be imported with a normal
    Python import statement.
    """
    data_path = SRC_DIR / "01-dataset.py"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Make sure Phase 1 file is located at src/01-dataset.py."
        )

    spec = importlib.util.spec_from_file_location("phase1_dataset", data_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Phase 1 module from {data_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_dataset_setup


# ---------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------

def standardize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the binary label column is named 'target'.

    Some Phase 1 versions may use 'num'. This function accepts either.
    """
    df = df.copy()

    if TARGET_COL in df.columns:
        return df

    if "num" in df.columns:
        return df.rename(columns={"num": TARGET_COL})

    raise KeyError("Dataset must contain either 'target' or 'num' label column.")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns excluding the binary target."""
    return [col for col in df.columns if col != TARGET_COL]


def coerce_features_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all feature columns to numeric.

    UCI files may contain '?' missing values or object dtypes.
    Converting with errors='coerce' safely turns invalid values into NaN.
    """
    df = standardize_target_column(df)
    df = df.copy()

    for col in get_feature_columns(df):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(int)
    return df


def load_clean_splits() -> Dict[str, pd.DataFrame]:
    """Load Phase 1 datasets and convert all features to numeric."""
    run_dataset_setup = load_phase1_runner()
    datasets = run_dataset_setup()

    splits = {
        "Cleveland": datasets["cleveland"],
        "Hungarian": datasets["hungarian"],
        "Swiss": datasets["swiss"],
    }

    return {name: coerce_features_to_numeric(df) for name, df in splits.items()}


# ---------------------------------------------------------------------
# EDA tables
# ---------------------------------------------------------------------

def class_balance_table(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a class balance summary for all sites."""
    rows = []

    for name, df in splits.items():
        counts = df[TARGET_COL].value_counts().sort_index()
        total = len(df)
        no_disease = int(counts.get(0, 0))
        disease = int(counts.get(1, 0))

        rows.append(
            {
                "site": name,
                "n_rows": total,
                "no_disease_count": no_disease,
                "disease_count": disease,
                "no_disease_rate": no_disease / total if total else np.nan,
                "disease_rate": disease / total if total else np.nan,
            }
        )

    return pd.DataFrame(rows)


def missingness_table(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a missingness summary by site and feature."""
    rows = []

    for site, df in splits.items():
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            rows.append(
                {
                    "site": site,
                    "feature": col,
                    "missing_count": missing_count,
                    "missing_rate": missing_count / len(df) if len(df) else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["missing_rate", "site", "feature"], ascending=[False, True, True]
    )


def compare_feature_shift(source: pd.DataFrame, target: pd.DataFrame, feature: str) -> dict:
    """
    Compare one feature between Cleveland and an external site.

    Continuous-looking features are compared with the KS test.
    Discrete/binary/categorical-looking numeric features are compared with
    Mann-Whitney U. This is an EDA screening tool, not a causal test.
    """
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

    unique_count = pd.concat([a, b]).nunique()

    if unique_count <= 10:
        test = "mannwhitneyu"
        stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    else:
        test = "ks_2samp"
        stat, p_value = stats.ks_2samp(a, b)

    return {
        "test": test,
        "statistic": float(stat),
        "p_value": float(p_value),
        "source_mean": float(a.mean()),
        "target_mean": float(b.mean()),
        "mean_delta": float(b.mean() - a.mean()),
    }


def distribution_shift_table(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create Cleveland-vs-external distribution shift table."""
    cleveland = splits["Cleveland"]
    feature_cols = get_feature_columns(cleveland)
    rows = []

    for external_name in ["Hungarian", "Swiss"]:
        external_df = splits[external_name]
        for feature in feature_cols:
            result = compare_feature_shift(cleveland, external_df, feature)
            rows.append(
                {
                    "comparison": f"Cleveland_vs_{external_name}",
                    "feature": feature,
                    **result,
                    "abs_mean_delta": abs(result["mean_delta"])
                    if pd.notna(result["mean_delta"])
                    else np.nan,
                }
            )

    shift_df = pd.DataFrame(rows)
    return shift_df.sort_values(["p_value", "abs_mean_delta"], ascending=[True, False])


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def save_class_balance_plot(balance_df: pd.DataFrame) -> Path:
    """Save class balance bar plot."""
    plot_df = balance_df.melt(
        id_vars="site",
        value_vars=["no_disease_count", "disease_count"],
        var_name="class",
        value_name="count",
    )
    plot_df["class"] = plot_df["class"].map(
        {"no_disease_count": "No disease", "disease_count": "Disease"}
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="site", y="count", hue="class")
    plt.title("Class Balance Across Sites")
    plt.xlabel("Site")
    plt.ylabel("Count")
    plt.tight_layout()

    path = FIGURE_DIR / "class_balance.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def save_missingness_heatmap(splits: Dict[str, pd.DataFrame]) -> Path:
    """Save missingness heatmap for all sites combined."""
    combined = []

    for site, df in splits.items():
        temp = df.copy()
        temp["site"] = site
        combined.append(temp)

    combined_df = pd.concat(combined, axis=0, ignore_index=True)

    plt.figure(figsize=(12, 6))
    msno.matrix(combined_df, sparkline=False)
    plt.title("Missingness Pattern Across All Sites")
    plt.tight_layout()

    path = FIGURE_DIR / "missingness_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def save_cleveland_correlation_matrix(cleveland: pd.DataFrame) -> Path:
    """Save Cleveland-only correlation matrix."""
    corr = cleveland.corr(numeric_only=True)

    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, square=True)
    plt.title("Cleveland Correlation Matrix")
    plt.tight_layout()

    path = FIGURE_DIR / "cleveland_correlation_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def save_feature_histograms(cleveland: pd.DataFrame) -> list[Path]:
    """
    Save Cleveland feature histograms split by binary label.

    These plots are Cleveland-only because Cleveland is the training source site.
    """
    paths = []
    feature_cols = get_feature_columns(cleveland)

    for feature in feature_cols:
        plt.figure(figsize=(7, 4))
        sns.histplot(
            data=cleveland,
            x=feature,
            hue=TARGET_COL,
            kde=False,
            bins=20,
            element="step",
            stat="count",
            common_norm=False,
        )
        plt.title(f"Cleveland Distribution by Label: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()

        safe_feature = feature.replace("/", "_").replace(" ", "_")
        path = FIGURE_DIR / f"hist_cleveland_{safe_feature}.png"
        plt.savefig(path, dpi=300)
        plt.close()
        paths.append(path)

    return paths


def save_shift_barplot(shift_df: pd.DataFrame, top_n: int = 10) -> Path:
    """Save barplot of largest absolute mean deltas."""
    plot_df = shift_df.dropna(subset=["abs_mean_delta"]).copy()
    plot_df = plot_df.sort_values("abs_mean_delta", ascending=False).head(top_n)
    plot_df["label"] = plot_df["comparison"] + " | " + plot_df["feature"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, y="label", x="abs_mean_delta")
    plt.title(f"Top {top_n} Cross-Site Feature Shifts by Absolute Mean Delta")
    plt.xlabel("Absolute Mean Delta")
    plt.ylabel("Comparison | Feature")
    plt.tight_layout()

    path = FIGURE_DIR / "top_distribution_shifts.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


# ---------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------

def summarize_eda_findings(
    balance_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    top_n: int = 8,
) -> None:
    """Print concise EDA findings for Phase 3 decisions."""
    print("\n" + "=" * 70)
    print("EDA SUMMARY FOR PHASE 3")
    print("=" * 70)

    print("\nClass balance:")
    print(balance_df.to_string(index=False))

    print("\nFeatures with missing values:")
    missing_nonzero = missing_df[missing_df["missing_count"] > 0]
    if missing_nonzero.empty:
        print("No missing values detected.")
    else:
        print(
            missing_nonzero.head(20).to_string(
                index=False,
                formatters={"missing_rate": "{:.2%}".format},
            )
        )

    print("\nStrongest cross-site shifts:")
    cols = [
        "comparison",
        "feature",
        "test",
        "p_value",
        "source_mean",
        "target_mean",
        "mean_delta",
    ]
    print(
        shift_df[cols]
        .head(top_n)
        .to_string(
            index=False,
            formatters={
                "p_value": "{:.3e}".format,
                "source_mean": "{:.3f}".format,
                "target_mean": "{:.3f}".format,
                "mean_delta": "{:.3f}".format,
            },
        )
    )

    print("\nPhase 3 implications:")
    print("- Use imputation in the preprocessing pipeline, fitted only on Cleveland train.")
    print("- Keep stratified Cleveland train/validation splitting because class balance matters clinically.")
    print("- Treat Hungarian and Swiss only as external test sites; do not refit preprocessing on them.")
    print("- Use the strongest shifted features to motivate the domain adaptation ablation.")


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run_eda(save_outputs: bool = True) -> Dict[str, object]:
    """
    Run the full Phase 2 EDA pipeline.

    Parameters
    ----------
    save_outputs:
        If True, saves tables and figures to outputs/.

    Returns
    -------
    Dictionary containing cleaned splits, EDA tables, and saved file paths.
    """
    ensure_output_dirs()

    print("\nLoading cleaned Phase 1 splits...\n")
    splits = load_clean_splits()

    balance_df = class_balance_table(splits)
    missing_df = missingness_table(splits)
    shift_df = distribution_shift_table(splits)

    saved_paths: Dict[str, object] = {}

    if save_outputs:
        balance_path = TABLE_DIR / "class_balance.csv"
        missing_path = TABLE_DIR / "missingness_summary.csv"
        shift_path = TABLE_DIR / "distribution_shift_summary.csv"

        balance_df.to_csv(balance_path, index=False)
        missing_df.to_csv(missing_path, index=False)
        shift_df.to_csv(shift_path, index=False)

        saved_paths["class_balance_table"] = balance_path
        saved_paths["missingness_table"] = missing_path
        saved_paths["distribution_shift_table"] = shift_path

        saved_paths["class_balance_plot"] = save_class_balance_plot(balance_df)
        saved_paths["missingness_heatmap"] = save_missingness_heatmap(splits)
        saved_paths["cleveland_correlation_matrix"] = save_cleveland_correlation_matrix(
            splits["Cleveland"]
        )
        saved_paths["feature_histograms"] = save_feature_histograms(splits["Cleveland"])
        saved_paths["top_distribution_shifts"] = save_shift_barplot(shift_df)

    summarize_eda_findings(balance_df, missing_df, shift_df)

    if save_outputs:
        print("\nSaved outputs:")
        for name, path in saved_paths.items():
            if isinstance(path, list):
                print(f"- {name}: {len(path)} files")
            else:
                print(f"- {name}: {path}")

    return {
        "splits": splits,
        "class_balance": balance_df,
        "missingness": missing_df,
        "distribution_shift": shift_df,
        "saved_paths": saved_paths,
    }


if __name__ == "__main__":
    run_eda(save_outputs=True)
