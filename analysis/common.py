"""Shared loaders for the LifeSnaps feasibility assessment.

Everything here is deliberately assumption-free: schemas are read off the
files, not off the paper.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

HOURLY_CSV = ROOT / "hourly_fitbit_sema_df_unprocessed.csv"
DAILY_CSV = ROOT / "daily_fitbit_sema_df_unprocessed.csv"
SEMA_BSON = ROOT / "sema.bson"
SURVEYS_BSON = ROOT / "surveys.bson"

# Channels that are numeric and vary within a day in the hourly table.
HOURLY_CHANNELS = [
    "bpm",
    "steps",
    "calories",
    "distance",
    "temperature",
    "scl_avg",
    "minutes_in_default_zone_1",
    "minutes_below_default_zone_1",
    "minutes_in_default_zone_2",
    "minutes_in_default_zone_3",
]

MOOD_COLS = ["ALERT", "HAPPY", "NEUTRAL", "RESTED/RELAXED", "SAD", "TENSE/ANXIOUS", "TIRED"]
PLACE_COLS = ["ENTERTAINMENT", "GYM", "HOME", "HOME_OFFICE", "OTHER", "OUTDOORS", "TRANSIT", "WORK/SCHOOL"]


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(HOURLY_CSV, index_col=0, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["ts"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df["mindfulness_session"] = df["mindfulness_session"].map(
        {True: True, False: False, "True": True, "False": False}
    )
    for c in HOURLY_CHANNELS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["id", "ts"]).reset_index(drop=True)


def load_daily() -> pd.DataFrame:
    df = pd.read_csv(DAILY_CSV, index_col=0, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values(["id", "date"]).reset_index(drop=True)


def _decode_bson(path: Path) -> list[dict]:
    from bson import decode_file_iter

    with open(path, "rb") as f:
        return list(decode_file_iter(f))


def load_sema() -> pd.DataFrame:
    docs = _decode_bson(SEMA_BSON)
    rows = []
    for d in docs:
        r = dict(d.get("data", {}))
        r["_id"] = str(d.get("_id"))
        r["user_id"] = str(d.get("user_id"))
        rows.append(r)
    df = pd.DataFrame(rows)
    for c in ["CREATED_TS", "SCHEDULED_TS", "STARTED_TS", "COMPLETED_TS", "EXPIRED_TS", "UPLOADED_TS"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["ts"] = df["COMPLETED_TS"].fillna(df.get("STARTED_TS")).fillna(df.get("CREATED_TS"))
    df["date"] = df["ts"].dt.normalize()
    df["hour"] = df["ts"].dt.hour
    return df


def load_surveys() -> pd.DataFrame:
    docs = _decode_bson(SURVEYS_BSON)
    rows = []
    for d in docs:
        rows.append(
            {
                "_id": str(d.get("_id")),
                "user_id": str(d.get("user_id")),
                "type": d.get("type"),
                "n_fields": len(d.get("data", {})),
                "submitdate": d.get("data", {}).get("submitdate"),
            }
        )
    return pd.DataFrame(rows)


def parse_listcol(v):
    """activityType / badgeType are stringified python lists."""
    if not isinstance(v, str) or not v.startswith("["):
        return []
    try:
        out = ast.literal_eval(v)
        return list(out) if isinstance(out, (list, tuple)) else [out]
    except Exception:
        return []


def write(df: pd.DataFrame, name: str) -> pd.DataFrame:
    p = OUT / name
    df.to_csv(p, index=False)
    print(f"  wrote {p.relative_to(ROOT)}  ({len(df)} rows)")
    return df
