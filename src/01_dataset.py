
"""
Phase 1 v2 — Dataset setup and canonical splitting.

Creates the single source of truth for all later phases:
- Cleveland train: 70%
- Cleveland validation: 15%
- Cleveland held-out test: 15%
- Hungarian external test
- Swiss external test

No patient ID column exists in the processed UCI Heart Disease files, so this runner
uses StratifiedShuffleSplit and documents the row-independence assumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve
import json

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target",
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
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for site, info in SITE_FILES.items():
        path = raw_data_dir / info["filename"]
        paths[site] = path
        if path.exists() and path.stat().st_size > 0:
            print(f"Found local file for {site}: {path}")
        else:
            print(f"Downloading {site}...")
            urlretrieve(info["url"], path)
            print(f"Saved: {path}")

    return paths


def load_site(path: Path, site: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["target_original"] = df["target"]
    df["target"] = (df["target"] > 0).astype(int)
    df["site"] = site
    df["source_row_index"] = df.index.astype(int)
    return df


def load_all_sites() -> Dict[str, pd.DataFrame]:
    paths = ensure_raw_data()
    return {site: load_site(path, site) for site, path in paths.items()}


def split_cleveland_70_15_15(cleveland: pd.DataFrame):
    df = cleveland.copy()

    splitter_1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=RANDOM_STATE
    )
    train_idx, temp_idx = next(splitter_1.split(df, df["target"]))

    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    splitter_2 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=RANDOM_STATE
    )
    val_rel_idx, test_rel_idx = next(splitter_2.split(temp, temp["target"]))

    validation = temp.iloc[val_rel_idx].copy()
    test = temp.iloc[test_rel_idx].copy()

    return train, validation, test


def assert_no_overlap(*dfs: pd.DataFrame) -> None:
    index_sets = [set(df["source_row_index"].tolist()) for df in dfs]
    names = ["train", "validation", "test"]
    for i in range(len(index_sets)):
        for j in range(i + 1, len(index_sets)):
            overlap = index_sets[i].intersection(index_sets[j])
            if overlap:
                raise ValueError(f"Overlap detected between {names[i]} and {names[j]}: {overlap}")
    print("No source row index overlap between Cleveland train/validation/test.")


def summarize_split(name: str, df: pd.DataFrame) -> dict:
    return {
        "name": name,
        "n_rows": int(len(df)),
        "n_positive": int(df["target"].sum()),
        "n_negative": int((df["target"] == 0).sum()),
        "positive_rate": float(df["target"].mean()),
    }


def save_outputs(train, validation, test, hungarian, swiss) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "cleveland_train.csv": train,
        "cleveland_validation.csv": validation,
        "cleveland_test.csv": test,
        "hungarian.csv": hungarian,
        "swiss.csv": swiss,
    }

    for filename, df in outputs.items():
        path = PROCESSED_DATA_DIR / filename
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    metadata = {
        "random_state": RANDOM_STATE,
        "split_strategy": "StratifiedShuffleSplit",
        "patient_id_available": False,
        "patient_id_note": (
            "The processed UCI Heart Disease files do not include explicit patient IDs. "
            "Rows are therefore treated as independent patient records; source_row_index "
            "is preserved only to verify no row overlap between Cleveland partitions."
        ),
        "cleveland_split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "splits": {
            "cleveland_train": summarize_split("cleveland_train", train),
            "cleveland_validation": summarize_split("cleveland_validation", validation),
            "cleveland_test": summarize_split("cleveland_test", test),
            "hungarian": summarize_split("hungarian", hungarian),
            "swiss": summarize_split("swiss", swiss),
        },
        "no_overlap_confirmed": True,
    }

    metadata_path = PROCESSED_DATA_DIR / "phase1_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved: {metadata_path}")


def run_dataset_setup_v2() -> Dict[str, pd.DataFrame]:
    print("\nPhase 1 v2 — Dataset setup")
    print("=" * 72)

    sites = load_all_sites()
    cleveland = sites["cleveland"]
    hungarian = sites["hungarian"]
    swiss = sites["swiss"]

    train, validation, test = split_cleveland_70_15_15(cleveland)
    assert_no_overlap(train, validation, test)

    save_outputs(train, validation, test, hungarian, swiss)

    print("\nSplit summary")
    print("=" * 72)
    for name, df in {
        "cleveland_train": train,
        "cleveland_validation": validation,
        "cleveland_test": test,
        "hungarian": hungarian,
        "swiss": swiss,
    }.items():
        s = summarize_split(name, df)
        print(
            f"{name:22s} rows={s['n_rows']:3d} "
            f"positive_rate={s['positive_rate']:.2%} "
            f"pos={s['n_positive']:3d} neg={s['n_negative']:3d}"
        )

    return {
        "cleveland_train": train,
        "cleveland_validation": validation,
        "cleveland_test": test,
        "hungarian": hungarian,
        "swiss": swiss,
    }


if __name__ == "__main__":
    run_dataset_setup_v2()
