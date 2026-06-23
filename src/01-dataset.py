"""
Phase 1 — Dataset Setup
AI in Medicine Project: Heart Disease ML

This module is the Phase 1 implementation/runner.

What it does
------------
1. Downloads the original UCI processed Heart Disease site files if missing.
2. Stores them in data/raw/.
3. Loads Cleveland, Hungarian, and Swiss as separate external-site datasets.
4. Assigns the standard 14 published attributes.
5. Converts '?' to missing values and casts columns to numeric.
6. Collapses the original target scale into binary labels:
   0 = no disease, 1 = disease present.
7. Verifies column alignment across the three splits.
8. Prints shapes, missingness, and label distributions.
9. Creates a leakage-safe Cleveland train/validation split.

Important project rule
----------------------
Cleveland is used for training/validation.
Hungarian and Swiss are kept untouched as external test sets.
No preprocessing is fitted on Hungarian or Swiss in this phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

COLUMN_NAMES = [
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
    "target",
]

SITE_FILES = {
    "cleveland": {
        "filename": "processed.cleveland.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    },
    "hungarian": {
        "filename": "processed.hungarian.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    },
    "swiss": {
        "filename": "processed.switzerland.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    },
}


def ensure_raw_data(raw_data_dir: Path = RAW_DATA_DIR) -> Dict[str, Path]:
    """
    Download the three UCI processed files if they do not already exist locally.

    Returns
    -------
    dict
        Mapping from site name to local file path.
    """

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    local_paths: Dict[str, Path] = {}

    for site, info in SITE_FILES.items():
        path = raw_data_dir / info["filename"]
        local_paths[site] = path

        if path.exists() and path.stat().st_size > 0:
            print(f"Found local file for {site}: {path}")
            continue

        print(f"Downloading {site} from UCI...")
        urlretrieve(info["url"], path)
        print(f"Saved to: {path}")

    return local_paths


def load_site_file(path: Path, site_name: str) -> pd.DataFrame:
    """
    Load one processed UCI Heart Disease file.
    """

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
    )

    # All columns are numeric in the processed 14-attribute files.
    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["site"] = site_name

    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse original UCI target encoding:
    0 = no disease, 1-4 = disease present.
    """

    df = df.copy()
    df["target_original"] = df["target"]
    df["target"] = (df["target"] > 0).astype(int)
    return df


def load_all_splits(raw_data_dir: Path = RAW_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Download if needed, then load Cleveland, Hungarian, and Swiss.
    """

    paths = ensure_raw_data(raw_data_dir)

    datasets = {
        site: binarize_target(load_site_file(path, site))
        for site, path in paths.items()
    }

    return datasets


def verify_column_alignment(datasets: Dict[str, pd.DataFrame]) -> None:
    """
    Verify that all sites contain the same modeling columns.
    """

    expected_model_columns = COLUMN_NAMES

    for site, df in datasets.items():
        actual_model_columns = [col for col in df.columns if col in expected_model_columns]
        if actual_model_columns != expected_model_columns:
            raise ValueError(
                f"Column mismatch in {site}.\n"
                f"Expected: {expected_model_columns}\n"
                f"Actual:   {actual_model_columns}"
            )

    print("Column alignment check passed for Cleveland, Hungarian, and Swiss.")


def dataset_summary(name: str, df: pd.DataFrame) -> None:
    """
    Print a compact clinical ML dataset summary.
    """

    label_counts = df["target"].value_counts().sort_index()
    positive_rate = df["target"].mean()
    missing_counts = df[COLUMN_NAMES].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    print("\n" + "=" * 72)
    print(name.upper())
    print("=" * 72)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nBinary label distribution:")
    print(label_counts.to_string())
    print(f"Positive disease rate: {positive_rate:.2%}")

    print("\nOriginal target values before binary collapse:")
    print(df["target_original"].value_counts(dropna=False).sort_index().to_string())

    print("\nMissing values in modeling columns:")
    if missing_counts.empty:
        print("No missing values detected.")
    else:
        print(missing_counts.to_string())


def create_cleveland_train_validation_split(
    cleveland: pd.DataFrame,
    validation_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a stratified Cleveland train/validation split.

    Hungarian and Swiss must not be used here because they are external test sets.
    """

    feature_columns = [col for col in COLUMN_NAMES if col != "target"]

    X = cleveland[feature_columns].copy()
    y = cleveland["target"].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_val, y_train, y_val


def print_split_summary(y_train: pd.Series, y_val: pd.Series) -> None:
    """
    Print the Cleveland train/validation label balance.
    """

    print("\n" + "=" * 72)
    print("CLEVELAND TRAIN/VALIDATION SPLIT")
    print("=" * 72)
    print(f"Train size: {len(y_train)}")
    print(f"Validation size: {len(y_val)}")
    print(f"Train positive rate: {y_train.mean():.2%}")
    print(f"Validation positive rate: {y_val.mean():.2%}")


def run_dataset_setup(raw_data_dir: Path = RAW_DATA_DIR) -> Dict[str, object]:
    """
    Full Phase 1 runner.

    This function is safe to call from notebooks. It does not create processed
    output files; it only downloads/caches raw UCI files if missing.
    """

    print("\nPhase 1 — Dataset setup")
    print("Loading Cleveland, Hungarian, and Swiss UCI Heart Disease splits...")

    datasets = load_all_splits(raw_data_dir)
    verify_column_alignment(datasets)

    for site, df in datasets.items():
        dataset_summary(site, df)

    X_train, X_val, y_train, y_val = create_cleveland_train_validation_split(
        datasets["cleveland"]
    )
    print_split_summary(y_train, y_val)

    print("\nLeakage rule confirmed:")
    print("Cleveland is used for train/validation only.")
    print("Hungarian and Swiss remain untouched external test sets for Phase 4.")

    return {
        "cleveland": datasets["cleveland"],
        "hungarian": datasets["hungarian"],
        "swiss": datasets["swiss"],
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }


if __name__ == "__main__":
    run_dataset_setup()
