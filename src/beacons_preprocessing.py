from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from difflib import get_close_matches


# ======================================================
# Paths
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEACONS_PATH = DATA_DIR / "beacons_dataset.csv"


# ======================================================
# Helpers: String similarity (from your original code)
# ======================================================
def replace_with_most_similar(value, choices) -> str:
    if isinstance(value, str):
        match = get_close_matches(value, choices, n=1, cutoff=0.65)
        if match:
            return match[0]
    return value


def replace_with_most_similar_dict(value, choices: dict) -> str:
    if isinstance(value, str):
        match = get_close_matches(value, choices.keys(), n=1, cutoff=0.7)
        if match:
            return choices[match[0]]
    return value


def replace_if_contains(text, mapping) -> str:
    if isinstance(text, str):
        for key, value in mapping.items():
            if key in text:
                return value
    return text


# ======================================================
# Beacons cleaning
# ======================================================
def normalize_room_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace(r"[^a-z]+", "", regex=True)
    return s.replace("", np.nan)


def fix_room_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Integrated version:
    - normalize text (lowercase + remove non-letters)
    - fuzzy match to reference rooms
    - keyword mapping and substring mapping (your original approach)
    """
    out = df.copy()

    rooms_reference = ["bathroom", "laundryroom", "livingroom", "kitchen", "bedroom", "office"]

    keywords_dict = {
        "bath": "bathroom",
        "laundry": "laundryroom",
        "tv": "livingroom",
        "sit": "livingroom",
        "seat": "livingroom",
        "dinner": "kitchen",
        "diner": "kitchen",
        "desk": "office",
        "work": "office",
        "hall": "hall",
        "out": "garden",
        "entr": "entrance",
        "bed": "bedroom",
        # extra robust keys
        "living": "livingroom",
        "kitch": "kitchen",
    }

    out["room"] = normalize_room_text(out["room"])

    # Step 1: fuzzy match to known rooms (catches typos like "kithcen")
    out["room"] = out["room"].apply(lambda x: replace_with_most_similar(x, rooms_reference))

    # Step 2: map close keywords to canonical labels
    out["room"] = out["room"].apply(lambda x: replace_with_most_similar_dict(x, keywords_dict))

    # Step 3: substring heuristic (e.g. "bathrm1" -> "bathroom" after normalization)
    out["room"] = out["room"].apply(lambda x: replace_if_contains(x, keywords_dict))

    return out


def remove_erroneous_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only valid 4-digit numeric participant IDs.
    (Equivalent to your original logic, but more robust.)
    """
    out = df.copy()
    s = out["part_id"].astype(str).str.strip()
    mask = s.str.fullmatch(r"\d{4}")
    out = out[mask].copy()
    out["part_id"] = out["part_id"].astype(str)
    return out


def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build timestamp from ts_date (YYYYMMDD) and ts_time (HH:MM:SS).
    Invalid parses become NaT and will be dropped downstream.
    """
    out = df.copy()
    date = out["ts_date"].astype(str).str.zfill(8)
    time = out["ts_time"].astype(str).str.strip()

    out["timestamp"] = pd.to_datetime(
        date + " " + time,
        format="%Y%m%d %H:%M:%S",
        errors="coerce",
    )
    return out


# ======================================================
# Feature engineering
# ======================================================
def _entropy_from_pct(row: pd.Series, room_cols: List[str]) -> float:
    """
    Shannon entropy (bits) from room time distribution.
    Higher entropy => more evenly spread time among rooms.
    """
    p = row[room_cols].to_numpy(dtype=float) / 100.0
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def get_beacons_features_per_user(
    beacons_df: pd.DataFrame,
    keep_rooms: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Produces 1 row per user with:
      - weighted % time per room (weights = total observed time per day)
      - transitions_per_day (weighted)
      - entropy_rooms (weighted)
    """
    keep_rooms = keep_rooms or ["kitchen", "bedroom", "bathroom", "livingroom"]

    out = build_timestamp(beacons_df)
    out = out.dropna(subset=["timestamp", "room", "part_id", "ts_date"]).copy()
    out["ts_date"] = out["ts_date"].astype(int)
    out.sort_values(["part_id", "ts_date", "timestamp"], inplace=True)

    # durations between consecutive events within the same user-day
    out["next_ts"] = out.groupby(["part_id", "ts_date"])["timestamp"].shift(-1)
    out["duration_s"] = (out["next_ts"] - out["timestamp"]).dt.total_seconds()

    out = out.dropna(subset=["duration_s"]).copy()
    out = out[out["duration_s"] > 0].copy()

    # total observed time per user-day (for weighting)
    totals = (
        out.groupby(["part_id", "ts_date"], as_index=False)["duration_s"]
        .sum()
        .rename(columns={"duration_s": "total_s"})
    )

    # % time per room per day
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
        aggfunc="sum",
    ).reset_index()

    for r in keep_rooms:
        if r not in pivot_daily.columns:
            pivot_daily[r] = 0.0

    pivot_daily = pivot_daily[["part_id", "ts_date"] + keep_rooms].copy()

    # transitions per day: count room changes (FIXED: do not count first record)
    out["prev_room"] = out.groupby(["part_id", "ts_date"])["room"].shift(1)
    out["is_transition"] = ((out["prev_room"].notna()) & (out["room"] != out["prev_room"])).astype(int)

    transitions = (
        out.groupby(["part_id", "ts_date"], as_index=False)["is_transition"]
        .sum()
        .rename(columns={"is_transition": "transitions"})
    )

    daily = (
        pivot_daily
        .merge(totals, on=["part_id", "ts_date"], how="left")
        .merge(transitions, on=["part_id", "ts_date"], how="left")
    )
    daily["transitions"] = daily["transitions"].fillna(0.0)

    # entropy per day
    daily["entropy_rooms"] = daily.apply(lambda row: _entropy_from_pct(row, keep_rooms), axis=1)

    # weighted aggregation across days (weights = total_s)
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


# ======================================================
# Main entry
# ======================================================
def run_part_b1() -> None:
    if not BEACONS_PATH.exists():
        raise FileNotFoundError(f"Could not find: {BEACONS_PATH}")

    df = pd.read_csv(BEACONS_PATH, sep=";")

    # 1) remove invalid ids
    df = remove_erroneous_users(df)

    # 2) clean room names
    df = fix_room_names(df)

    # 3) compute per-user features
    features = get_beacons_features_per_user(
        df,
        keep_rooms=["kitchen", "bedroom", "bathroom", "livingroom"],
    )

    out_path = OUT_DIR / "beacons_features.csv"
    features.to_csv(out_path, sep=";", index=False)

    print("✅ Part B1 (Beacons preprocessing) completed.")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_part_b1()
