"""Builds the dense per-participant hourly panel that every Phase-2 metric and
the dashboard are computed from.

The panel is dense over each participant's own observation span: every hour
between their first and last hourly row exists as a slot, whether or not the
CSV had a row for it. That distinction is the whole point -- a missing row is
"no data at all for that hour", which is different from a row that exists but
carries NaN for one channel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, load_hourly, parse_listcol

# Channels that genuinely vary within a participant-day in the hourly table.
TRUE_HOURLY = ["bpm", "steps", "calories", "distance", "temperature"]
# Present in the hourly table but constant across every hour of a participant-day:
# these are daily aggregates stamped onto a single hour, not time series.
DAILY_STAMPED = [
    "scl_avg",
    "minutes_in_default_zone_1",
    "minutes_below_default_zone_1",
    "minutes_in_default_zone_2",
    "minutes_in_default_zone_3",
]
MOOD_COLS = ["ALERT", "HAPPY", "NEUTRAL", "RESTED/RELAXED", "SAD", "TENSE/ANXIOUS", "TIRED"]
PLACE_COLS = ["ENTERTAINMENT", "GYM", "HOME", "HOME_OFFICE", "OTHER", "OUTDOORS", "TRANSIT", "WORK/SCHOOL"]
PANEL_CHANNELS = TRUE_HOURLY + DAILY_STAMPED

EXERCISE_LABELS = {
    "Walk", "Bike", "Workout", "Sport", "Aerobic Workout", "Run", "Swim",
    "Yoga/Pilates", "Weights", "Treadmill", "Circuit Training", "Elliptical",
    "Hike", "Martial Arts", "Interval Workout", "Spinning", "Bootcamp", "Tennis",
}
# "Walk" dominates and is auto-detected rather than user-initiated; keep it
# separable from deliberate training sessions.
VIGOROUS = {"Bike", "Workout", "Sport", "Aerobic Workout", "Run", "Swim", "Weights",
            "Treadmill", "Circuit Training", "Elliptical", "Hike", "Martial Arts",
            "Interval Workout", "Spinning", "Bootcamp", "Tennis"}


def build_panel() -> tuple[pd.DataFrame, dict]:
    h = load_hourly()
    h["id"] = h["id"].astype(str)
    # 70 exact-duplicate (id, ts) pairs across 2 participants, all at hour 0 --
    # an upstream merge artifact. Recorded in the inventory; dropped here.
    n_dup = int(h.duplicated(["id", "ts"], keep="first").sum())
    h = h.drop_duplicates(["id", "ts"], keep="first")

    frames = []
    for pid, g in h.groupby("id", sort=True):
        idx = pd.date_range(g["ts"].min(), g["ts"].max(), freq="h")
        gg = g.set_index("ts").reindex(idx)
        gg.index.name = "ts"
        gg["id"] = pid
        gg["row_present"] = g.set_index("ts").reindex(idx)["date"].notna().values
        frames.append(gg.reset_index())
    p = pd.concat(frames, ignore_index=True)

    p["date"] = p["ts"].dt.normalize()
    p["hour"] = p["ts"].dt.hour
    p["dow"] = p["ts"].dt.dayofweek

    # ---- events -----------------------------------------------------------
    acts = p["activityType"].map(parse_listcol)
    p["ev_exercise_any"] = acts.str.len() > 0
    p["ev_exercise_vigorous"] = acts.map(lambda l: any(x in VIGOROUS for x in l))
    p["ev_walk"] = acts.map(lambda l: "Walk" in l)
    p["ev_mindfulness"] = p["mindfulness_session"].fillna(False).astype(bool)
    p["activity_labels"] = acts.map(lambda l: "|".join(l))

    # ---- EMA --------------------------------------------------------------
    def onehot_label(block: pd.DataFrame) -> pd.Series:
        arr = block.to_numpy(dtype="float64")
        hit = np.nanmax(np.where(np.isnan(arr), -1.0, arr), axis=1) == 1.0
        lab = np.array(block.columns)[np.argmax(np.nan_to_num(arr, nan=-1.0), axis=1)]
        return pd.Series(np.where(hit, lab, None), index=block.index, dtype=object)

    mood = p[MOOD_COLS]
    p["ema_present"] = mood.notna().any(axis=1)
    p["ema_mood"] = onehot_label(mood)
    p["ema_valid"] = p["ema_mood"].notna()
    place = p[PLACE_COLS]
    p["ema_place"] = onehot_label(place)

    meta = {"n_participants": p["id"].nunique(), "n_slots": len(p), "n_duplicate_rows_dropped": n_dup}
    return p, meta


def presence_matrix(p: pd.DataFrame, channels=PANEL_CHANNELS):
    """Per participant: (n_hours x n_channels) bool presence, plus hour offsets."""
    out = {}
    for pid, g in p.groupby("id", sort=True):
        g = g.sort_values("ts")
        out[pid] = {
            "t0": g["ts"].iloc[0],
            "n": len(g),
            "M": g[channels].notna().to_numpy(),
            "ts": g["ts"].to_numpy(),
        }
    return out


if __name__ == "__main__":
    p, meta = build_panel()
    print(meta)
    print(p[["id", "ts", "row_present", "bpm", "steps", "ev_exercise_any", "ev_mindfulness", "ema_valid"]].head())
    print("dense slots:", len(p), " rows actually present:", int(p["row_present"].sum()))
    p.to_parquet(OUT / "panel.parquet")
    print("wrote", OUT / "panel.parquet")
