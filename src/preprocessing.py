from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from difflib import get_close_matches


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


# Clinical cleaning 
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


def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.6,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop columns with missing ratio > threshold.
    Returns: (df_dropped, report_df)
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
    return df.dropna(subset=required_cols).copy()


# Beacons preprocessing + features
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


def _entropy_from_pct(row: pd.Series, room_cols: List[str]) -> float:
    p = row[room_cols].to_numpy(dtype=float) / 100.0
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def get_beacons_features_per_user(
    beacons_df: pd.DataFrame,
    keep_rooms: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Per user features:
    - % time in each room (weighted by daily total time)
    - transitions_per_day (weighted)
    - entropy_rooms (weighted)
    """
    keep_rooms = keep_rooms or ["kitchen", "bedroom", "bathroom", "livingroom"]

    out = build_timestamp(beacons_df)
    out = out.dropna(subset=["timestamp", "room", "part_id", "ts_date"]).copy()
    out["ts_date"] = out["ts_date"].astype(int)
    out.sort_values(["part_id", "ts_date", "timestamp"], inplace=True)

    # duration between events
    out["next_ts"] = out.groupby(["part_id", "ts_date"])["timestamp"].shift(-1)
    out["duration_s"] = (out["next_ts"] - out["timestamp"]).dt.total_seconds()
    out = out.dropna(subset=["duration_s"]).copy()
    out = out[out["duration_s"] > 0].copy()

    # totals per day for weighting
    totals = (
        out.groupby(["part_id", "ts_date"], as_index=False)["duration_s"]
        .sum()
        .rename(columns={"duration_s": "total_s"})
    )

    # room duration per day
    grouped = (
        out.groupby(["part_id", "ts_date", "room"], as_index=False)["duration_s"]
        .sum()
        .merge(totals, on=["part_id", "ts_date"], how="left")
    )
    grouped["pct"] = (grouped["duration_s"] / grouped["total_s"]) * 100.0

    pivot_daily = grouped.pivot_table(
        index=["part_id", "ts_date"],
        columns="room",
        values="pct",
        fill_value=0.0,
        aggfunc="sum"
    ).reset_index()

    # ensure columns exist
    for r in keep_rooms:
        if r not in pivot_daily.columns:
            pivot_daily[r] = 0.0
    pivot_daily = pivot_daily[["part_id", "ts_date"] + keep_rooms].copy()

    # transitions per day: count room changes
    out["prev_room"] = out.groupby(["part_id", "ts_date"])["room"].shift(1)
    out["is_transition"] = (out["room"] != out["prev_room"]).astype(int)
    transitions = (
        out.groupby(["part_id", "ts_date"], as_index=False)["is_transition"]
        .sum()
        .rename(columns={"is_transition": "transitions"})
    )

    daily = pivot_daily.merge(totals, on=["part_id", "ts_date"], how="left") \
                      .merge(transitions, on=["part_id", "ts_date"], how="left")

    daily["transitions"] = daily["transitions"].fillna(0.0)

    # entropy per day
    daily["entropy_rooms"] = daily.apply(lambda row: _entropy_from_pct(row, keep_rooms), axis=1)

    # weighted aggregation across days
    def weighted_avg(group: pd.DataFrame) -> pd.Series:
        w = group["total_s"].to_numpy(dtype=float)
        wsum = w.sum()
        if wsum <= 0:
            base = {r: 0.0 for r in keep_rooms}
            base["transitions_per_day"] = 0.0
            base["entropy_rooms"] = 0.0
            return pd.Series(base)

        outd = {r: float(np.average(group[r].to_numpy(dtype=float), weights=w)) for r in keep_rooms}
        outd["transitions_per_day"] = float(np.average(group["transitions"].to_numpy(dtype=float), weights=w))
        outd["entropy_rooms"] = float(np.average(group["entropy_rooms"].to_numpy(dtype=float), weights=w))
        return pd.Series(outd)

    per_user = daily.groupby("part_id").apply(weighted_avg).reset_index()
    return per_user


def run_preprocessing() -> None:
    clinical_path = DATA_DIR / "clinical_dataset.csv"
    beacons_path = DATA_DIR / "beacons_dataset.csv"

    clinical = pd.read_csv(clinical_path, sep=";")
    beacons = pd.read_csv(beacons_path, sep=";")

    # Clinical: only cleaning + dropping
    clinical_clean = remove_errors(clinical)

    clinical_clean, miss_report = drop_high_missing_columns(
        clinical_clean,
        threshold=0.6,
        exclude_cols=["part_id", "fried"]
    )

    clinical_clean = drop_rows_with_missing(clinical_clean, required_cols=["fried"])

    miss_report.to_csv(OUT_DIR / "clinical_missing_ratio_report.csv", index=False)
    clinical_clean.to_csv(OUT_DIR / "clinical_clean.csv", sep=";", index=False)

    # Beacons: cleaning + features
    beacons_clean = fix_room_names(beacons)
    beacons_clean = remove_erroneous_users(beacons_clean)

    beacons_features = get_beacons_features_per_user(
        beacons_clean,
        keep_rooms=["kitchen", "bedroom", "bathroom", "livingroom"]
    )
    beacons_features.to_csv(OUT_DIR / "beacons_features.csv", sep=";", index=False)

    print("✅ Preprocessing done.")
    print(f"Saved: {OUT_DIR / 'clinical_clean.csv'}")
    print(f"Saved: {OUT_DIR / 'beacons_features.csv'}")


if __name__ == "__main__":
    run_preprocessing()
