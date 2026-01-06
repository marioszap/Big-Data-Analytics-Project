import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from difflib import get_close_matches


# ======================================================
# Paths 
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent   # project-root/
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


# ======================================================
# Clinical preprocessing -->  PART A1
# ======================================================

def remove_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace erroneous values with NaN:
    - numeric sentinel 999
    - strings starting with 'test' (case-insensitive)
    """
    out = df.copy()

    out.replace(999, np.nan, inplace=True)

    obj_cols = out.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        out[obj_cols] = out[obj_cols].replace(
            to_replace=re.compile(r"^\s*test.*", flags=re.IGNORECASE),
            value=np.nan
        )

    return out


def nominal_to_numerical(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    mapping_out: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Convert nominal (object) columns to numeric codes.
    Stable mapping using sorted unique values.
    """
    exclude_cols = exclude_cols or []
    out = df.copy()
    mapping: Dict[str, Dict[str, int]] = {}

    obj_cols = [c for c in out.columns if out[c].dtype == "object" and c not in exclude_cols]

    for col in obj_cols:
        uniques = sorted(out[col].dropna().astype(str).unique())
        col_map = {v: i for i, v in enumerate(uniques)}
        mapping[col] = col_map

        out[col] = out[col].astype(str).where(out[col].notna(), np.nan)
        out[col] = out[col].map(col_map)

    if mapping_out:
        save_json(mapping, mapping_out)

    return out, mapping


def count_nan_per_column(df: pd.DataFrame, out_path: Optional[str | Path] = None) -> Dict[str, int]:
    counts = df.isna().sum().to_dict()
    if out_path:
        save_json(counts, out_path)
    return counts


# ======================================================
# Missing values handling
# ======================================================

def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.4,
    exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Drop columns with missing ratio > threshold.
    Example: threshold=0.4 => drop columns with >40% missing.
    """
    exclude_cols = set(exclude_cols or [])
    out = df.copy()

    miss_ratio = out.isna().mean()
    to_drop = [c for c, r in miss_ratio.items() if (r > threshold and c not in exclude_cols)]

    return out.drop(columns=to_drop)


def drop_rows_with_missing(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """Drop rows with missing values in required columns (e.g., target)."""
    return df.dropna(subset=required_cols).copy()


def impute_missing_values(
    df: pd.DataFrame,
    strategy_num: str = "median",
    strategy_cat: str = "mode",
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Impute missing values:
      - numeric: median (default) or mean
      - categorical/object: mode (most frequent)
    Returns (imputed_df, numeric_imputation_report)
    """
    exclude_cols = set(exclude_cols or [])
    out = df.copy()
    report: Dict[str, float] = {}

    # numeric columns
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c]) and c not in exclude_cols]
    for c in num_cols:
        if out[c].isna().any():
            if strategy_num == "mean":
                fill_val = float(out[c].mean())
            else:
                fill_val = float(out[c].median())
            out[c] = out[c].fillna(fill_val)
            report[c] = fill_val

    # categorical/object columns
    cat_cols = [c for c in out.columns if out[c].dtype == "object" and c not in exclude_cols]
    for c in cat_cols:
        if out[c].isna().any():
            if strategy_cat == "mode":
                modes = out[c].mode(dropna=True)
                fill_val = modes.iloc[0] if len(modes) > 0 else "UNKNOWN"
            else:
                fill_val = "UNKNOWN"
            out[c] = out[c].fillna(fill_val)

    return out, report


# ======================================================
# Beacons preprocessing --> PART B1
# ======================================================

def normalize_room_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace(r"[^a-z]+", "", regex=True)
    return s.replace("", np.nan)


def fix_room_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    reference_rooms = ["bathroom", "laundryroom", "livingroom", "kitchen", "bedroom", "office"]

    keyword_map = {
        "bath": "bathroom",
        "laundry": "laundryroom",
        "sit": "livingroom",
        "tv": "livingroom",
        "living": "livingroom",
        "diner": "kitchen",
        "kitch": "kitchen",
        "desk": "office",
        "work": "office",
        "bed": "bedroom",
    }

    out["room"] = normalize_room_text(out["room"])

    for key, val in keyword_map.items():
        mask = out["room"].notna() & out["room"].str.contains(key, regex=False)
        out.loc[mask, "room"] = val

    def fuzzy(x):
        if pd.isna(x) or x in reference_rooms:
            return x
        match = get_close_matches(x, reference_rooms, n=1, cutoff=0.65)
        return match[0] if match else x

    out["room"] = out["room"].apply(fuzzy)
    return out


def remove_erroneous_users(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s = out["part_id"].astype(str).str.strip()
    mask = s.str.fullmatch(r"\d{4}")
    out = out[mask].copy()
    out["part_id"] = out["part_id"].astype(str)
    return out


def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date = out["ts_date"].astype(str).str.zfill(8)
    time = out["ts_time"].astype(str).str.strip()
    out["timestamp"] = pd.to_datetime(
        date + " " + time,
        format="%Y%m%d %H:%M:%S",
        errors="coerce"
    )
    return out


def get_percent_of_time_per_room(
    df: pd.DataFrame,
    keep_rooms: Optional[List[str]] = None,
    daily: bool = False
) -> pd.DataFrame:
    """
    If daily=False -> returns 1 row per user (weighted across days).
    If daily=True  -> returns 1 row per (user, day).
    """
    keep_rooms = keep_rooms or ["kitchen", "bedroom", "bathroom", "livingroom"]

    out = build_timestamp(df)
    out = out.dropna(subset=["timestamp", "room", "part_id", "ts_date"]).copy()
    out["ts_date"] = out["ts_date"].astype(int)

    out.sort_values(["part_id", "ts_date", "timestamp"], inplace=True)

    out["next_ts"] = out.groupby(["part_id", "ts_date"])["timestamp"].shift(-1)
    out["duration_s"] = (out["next_ts"] - out["timestamp"]).dt.total_seconds()

    out = out.dropna(subset=["duration_s"]).copy()
    out = out[out["duration_s"] > 0].copy()

    grouped = (
        out.groupby(["part_id", "ts_date", "room"], as_index=False)["duration_s"]
        .sum()
    )

    totals = (
        grouped.groupby(["part_id", "ts_date"], as_index=False)["duration_s"]
        .sum()
        .rename(columns={"duration_s": "total_s"})
    )

    merged = grouped.merge(totals, on=["part_id", "ts_date"], how="left")
    merged["pct"] = (merged["duration_s"] / merged["total_s"]) * 100.0

    pivot_daily = merged.pivot_table(
        index=["part_id", "ts_date"],
        columns="room",
        values="pct",
        fill_value=0.0,
        aggfunc="sum"
    ).reset_index()

    for r in keep_rooms:
        if r not in pivot_daily.columns:
            pivot_daily[r] = 0.0

    pivot_daily = pivot_daily[["part_id", "ts_date"] + keep_rooms].copy()

    if daily:
        return pivot_daily.rename(columns={"ts_date": "date"})

    # Weighted aggregation across days per user (weights = total time per day)
    total_daily = totals.rename(columns={"ts_date": "date"}).copy()
    pivot_daily = pivot_daily.rename(columns={"ts_date": "date"}).merge(total_daily, on=["part_id", "date"], how="left")

    def weighted_avg(group: pd.DataFrame) -> pd.Series:
        w = group["total_s"].to_numpy()
        wsum = w.sum()
        if wsum <= 0:
            return pd.Series({r: 0.0 for r in keep_rooms})
        return pd.Series({r: np.average(group[r].to_numpy(), weights=w) for r in keep_rooms})

    per_user = pivot_daily.groupby("part_id").apply(weighted_avg).reset_index()
    return per_user


# ======================================================
# Main pipeline
# ======================================================

if __name__ == "__main__":

    clinical_path = DATA_DIR / "clinical_dataset.csv"
    beacons_path = DATA_DIR / "beacons_dataset.csv"

    clinical = pd.read_csv(clinical_path, sep=";")
    beacons = pd.read_csv(beacons_path, sep=";")

    # ----------------------------
    # Clinical (Part A1)
    # ----------------------------
    clinical_clean = remove_errors(clinical)

    # Drop columns with too many missing
    clinical_clean = drop_high_missing_columns(
        clinical_clean,
        threshold=0.4,
        exclude_cols=["part_id", "fried"]
    )

    # Drop rows missing the target (needed for classification)
    clinical_clean = drop_rows_with_missing(clinical_clean, required_cols=["fried"])

    # Impute remaining missing values
    clinical_imputed, impute_report = impute_missing_values(
        clinical_clean,
        strategy_num="median",
        strategy_cat="mode",
        exclude_cols=["part_id"]
    )
    save_json(impute_report, OUT_DIR / "imputation_report.json")

    # Convert nominal -> numeric (after imputation)
    clinical_numeric, mapping = nominal_to_numerical(
        clinical_imputed,
        exclude_cols=["part_id"],
        mapping_out=OUT_DIR / "nominal_to_numeric_map.json"
    )

    save_json(count_nan_per_column(clinical_numeric), OUT_DIR / "num_nan_by_column_after_impute.json")
    clinical_numeric.to_csv(OUT_DIR / "clinical_clean.csv", sep=";", index=False)

    # ----------------------------
    # Beacons (Part B1)
    # ----------------------------
    beacons_clean = fix_room_names(beacons)
    beacons_clean = remove_erroneous_users(beacons_clean)

    beacons_features = get_percent_of_time_per_room(
        beacons_clean,
        keep_rooms=["kitchen", "bedroom", "bathroom", "livingroom"],
        daily=False   # 1 row per user
    )

    beacons_features.to_csv(OUT_DIR / "beacons_features.csv", sep=";", index=False)

    print("✅ Preprocessing completed successfully.")
    print(f"Saved: {OUT_DIR / 'clinical_clean.csv'}")
    print(f"Saved: {OUT_DIR / 'beacons_features.csv'}")
