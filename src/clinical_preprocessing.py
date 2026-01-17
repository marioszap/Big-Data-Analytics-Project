from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ======================================================
# Paths
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# Clinical preprocessing (Part A1)
# ======================================================
def remove_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace erroneous values with NaN:
    - numeric sentinel 999
    - strings starting with 'test' (case-insensitive)
    """
    out = df.copy()

    # numeric placeholder -> NaN
    out.replace(999, np.nan, inplace=True)

    # strings like "test ..." -> NaN (only on object columns)
    obj_cols = out.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        out[obj_cols] = out[obj_cols].replace(
            to_replace=re.compile(r"^\s*test.*", flags=re.IGNORECASE),
            value=np.nan,
        )

    return out


def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.6,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop columns with missing ratio > threshold.
    Returns:
      - cleaned_df (with dropped columns)
      - report_df (missing ratio per column, sorted desc)
    """
    exclude_cols = set(exclude_cols or [])
    out = df.copy()

    miss_ratio = out.isna().mean().sort_values(ascending=False)
    report = miss_ratio.reset_index()
    report.columns = ["column", "missing_ratio"]

    to_drop = [c for c, r in miss_ratio.items() if (r > threshold and c not in exclude_cols)]
    out = out.drop(columns=to_drop)

    return out, report


def drop_rows_with_missing(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """Drop rows that have missing values in required columns (e.g., the target)."""
    return df.dropna(subset=required_cols).copy()


def run_preprocessing_part_a() -> None:
    clinical_path = DATA_DIR / "clinical_dataset.csv"
    if not clinical_path.exists():
        raise FileNotFoundError(f"Could not find: {clinical_path}")

    clinical = pd.read_csv(clinical_path, sep=";")

    # 1) Remove erroneous values
    clinical_clean = remove_errors(clinical)

    # 2) Drop columns with too much missing data
    clinical_clean, miss_report = drop_high_missing_columns(
        clinical_clean,
        threshold=0.6,
        exclude_cols=["part_id", "fried"],
    )

    # 3) Drop rows missing the target
    clinical_clean = drop_rows_with_missing(clinical_clean, required_cols=["fried"])

    # Save outputs
    miss_report.to_csv(OUT_DIR / "clinical_missing_ratio_report.csv", index=False)
    clinical_clean.to_csv(OUT_DIR / "clinical_clean.csv", sep=";", index=False)

    print("✅ Part A preprocessing completed.")
    print(f"Saved: {OUT_DIR / 'clinical_clean.csv'}")
    print(f"Saved: {OUT_DIR / 'clinical_missing_ratio_report.csv'}")


if __name__ == "__main__":
    run_preprocessing_part_a()
