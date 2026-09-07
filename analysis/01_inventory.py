"""Phase 1 - discovery inventory. Reads schemas off the files, assumes nothing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (DAILY_CSV, HOURLY_CSV, MOOD_COLS, OUT, PLACE_COLS, ROOT,
                    SEMA_BSON, SURVEYS_BSON, load_daily, load_hourly,
                    load_sema, load_surveys, parse_listcol, write)

pd.set_option("display.width", 200)

# ---------------------------------------------------------------- file manifest
print("\n== FILE MANIFEST ==")
rows = []
for p in sorted(ROOT.glob("*")):
    if p.is_dir() or p.name.startswith("."):
        continue
    n = None
    if p.suffix == ".csv":
        with open(p) as f:
            n = sum(1 for _ in f) - 1
    elif p.suffix == ".bson":
        from bson import decode_file_iter
        with open(p, "rb") as f:
            n = sum(1 for _ in decode_file_iter(f))
    rows.append({"file": p.name, "bytes": p.stat().st_size,
                 "mb": round(p.stat().st_size / 1e6, 2), "records": n})
manifest = pd.DataFrame(rows)
print(manifest.to_string(index=False))
write(manifest, "01_file_manifest.csv")

# fitbit.bson absence is the single most consequential fact in this inventory.
missing_collection = not (ROOT / "fitbit.bson").exists() and (ROOT / "fitbit.metadata.json").exists()
print(f"\n!! fitbit.metadata.json present but fitbit.bson absent: {missing_collection}")
if missing_collection:
    meta = json.loads((ROOT / "fitbit.metadata.json").read_text())
    keys = sorted({k for ix in meta["indexes"] for k in ix["key"]})
    print("   indexed fields of the ABSENT collection:", keys)

hourly = load_hourly()
daily = load_daily()
sema = load_sema()
surveys = load_surveys()

# ------------------------------------------------------------- channel profile
def profile(df: pd.DataFrame, source: str, time_col: str, grain: str) -> pd.DataFrame:
    recs = []
    for c in df.columns:
        if c in ("ts",):
            continue
        s = df[c]
        nn = s.notna().sum()
        r = {"source": source, "column": c, "dtype": str(s.dtype), "grain": grain,
             "n_nonnull": int(nn), "pct_nonnull": round(100 * nn / len(df), 2),
             "n_unique": int(s.nunique(dropna=True))}
        num = pd.to_numeric(s, errors="coerce")
        is_dt = pd.api.types.is_datetime64_any_dtype(s)
        if num.notna().sum() > 0 and not is_dt:
            r |= {"min": num.min(), "p50": num.median(), "max": num.max()}
            # does it vary within a participant-day? -> genuine sub-daily series
            if grain == "hourly" and num.notna().sum() > 100:
                g = df.assign(_v=num).groupby(["id", "date"])["_v"]
                nun = g.nunique()
                r["frac_pdays_varying"] = round(float((nun > 1).mean()), 3)
        else:
            ex = s.dropna().astype(str).unique()[:3]
            r["example"] = " | ".join(ex)[:80]
        recs.append(r)
    return pd.DataFrame(recs)

print("\n== CHANNEL PROFILE (hourly table) ==")
ph = profile(hourly, "hourly_fitbit_sema", "ts", "hourly")
print(ph.to_string(index=False))
print("\n== CHANNEL PROFILE (daily table) ==")
pd_ = profile(daily, "daily_fitbit_sema", "date", "daily")
print(pd_.to_string(index=False))
chan = pd.concat([ph, pd_], ignore_index=True)
write(chan, "01_channel_profile.csv")

# ----------------------------------------------------- per-participant coverage
print("\n== PER-PARTICIPANT COVERAGE ==")
cov = hourly.groupby("id").agg(
    first_ts=("ts", "min"), last_ts=("ts", "max"),
    n_hour_rows=("ts", "size"), n_days_present=("date", "nunique"),
).reset_index()
cov["span_days"] = (cov["last_ts"] - cov["first_ts"]).dt.total_seconds() / 86400
cov["day_density"] = (cov["n_days_present"] / cov["span_days"].clip(lower=1)).round(3)
d_cov = daily.groupby("id").agg(daily_rows=("date", "size")).reset_index()
s_cov = sema.groupby("user_id").agg(
    sema_first=("ts", "min"), sema_last=("ts", "max"),
    n_sema=("ts", "size"), sema_days=("date", "nunique")).reset_index().rename(columns={"user_id": "id"})
cov = cov.merge(d_cov, on="id", how="outer").merge(s_cov, on="id", how="outer")
print(f"participants in hourly: {hourly['id'].nunique()}  daily: {daily['id'].nunique()}  sema: {sema['user_id'].nunique()}")
print(f"union: {cov['id'].nunique()}")
print(cov.describe(include="all").to_string())
write(cov, "01_participant_coverage.csv")

# --------------------------------------------------------------- timezone check
print("\n== TIMEZONE / TIMESTAMP CONSISTENCY ==")
tz_notes = []
tz_notes.append(("hourly.date", "date string YYYY-MM-DD, no tz, hour is separate float column"))
tz_notes.append(("daily.date", "date string YYYY-MM-DD, no tz"))
for c in ["CREATED_TS", "SCHEDULED_TS", "STARTED_TS", "COMPLETED_TS", "UPLOADED_TS"]:
    ex = sema[c].dropna().iloc[0] if c in sema and sema[c].notna().any() else None
    tz_notes.append((f"sema.{c}", f"naive ISO local time, minute resolution, e.g. {ex}"))
tz = pd.DataFrame(tz_notes, columns=["field", "note"])
print(tz.to_string(index=False))
write(tz, "01_timezone_notes.csv")

# Cross-source alignment: do sema hours line up with hourly rows that have data?
sema_ids = set(sema["user_id"]); hourly_ids = set(hourly["id"])
print(f"\nsema ids not in hourly: {len(sema_ids - hourly_ids)}; hourly ids not in sema: {len(hourly_ids - sema_ids)}")
m = sema.merge(hourly[["id", "date", "hour", "bpm"]], left_on=["user_id", "date", "hour"],
               right_on=["id", "date", "hour"], how="left")
print(f"SEMA responses that land on an existing hourly row: {m['id'].notna().sum()} / {len(sema)} "
      f"({100*m['id'].notna().mean():.1f}%)")
print(f"  ...and that hourly row has a bpm value: {m['bpm'].notna().sum()} ({100*m['bpm'].notna().mean():.1f}%)")

# The hourly table also carries one-hot MOOD/PLACE columns copied from SEMA.
print("\n== MOOD/PLACE ONE-HOTS INSIDE THE HOURLY TABLE ==")
mp = pd.DataFrame({
    "column": MOOD_COLS + PLACE_COLS,
    "n_nonnull": [int(hourly[c].notna().sum()) for c in MOOD_COLS + PLACE_COLS],
})
print(mp.to_string(index=False))
write(mp, "01_mood_place_onehot_counts.csv")

# ------------------------------------------------------------- event-like fields
print("\n== EVENT-LIKE FIELDS ==")
ev = []
act = hourly["activityType"].map(parse_listcol)
ev.append({"event_family": "activityType (hourly bucket flag)", "n_participant_hours": int((act.str.len() > 0).sum()),
           "n_individual_labels": int(act.str.len().sum()), "timestamped": "NO - hour bucket only"})
ev.append({"event_family": "mindfulness_session == True", "n_participant_hours": int((hourly["mindfulness_session"] == True).sum()),
           "n_individual_labels": int((hourly["mindfulness_session"] == True).sum()), "timestamped": "NO - hour bucket only"})
ev.append({"event_family": "badgeType (hourly)", "n_participant_hours": int(hourly["badgeType"].map(parse_listcol).str.len().gt(0).sum()),
           "n_individual_labels": int(hourly["badgeType"].map(parse_listcol).str.len().sum()), "timestamped": "NO - hour bucket only"})
ev.append({"event_family": "sleep onset (sleep_start)", "n_participant_hours": 0, "n_individual_labels": 0,
           "timestamped": "ABSENT - only daily sleep aggregates exist"})
evdf = pd.DataFrame(ev)
print(evdf.to_string(index=False))
write(evdf, "01_event_like_fields.csv")

from collections import Counter
cnt = Counter(x for l in act for x in l)
adf = pd.DataFrame(sorted(cnt.items(), key=lambda kv: -kv[1]), columns=["activity_label", "n"])
print("\nactivityType labels:")
print(adf.to_string(index=False))
write(adf, "01_activity_labels.csv")

# --------------------------------------------------------------- survey inventory
print("\n== SURVEY / QUESTIONNAIRE INVENTORY ==")
sv = surveys.groupby("type").agg(n_docs=("_id", "size"), n_participants=("user_id", "nunique")).reset_index()
print(sv.to_string(index=False))
write(sv, "01_survey_inventory.csv")

print("\nSEMA survey names:")
print(sema["SURVEY_NAME"].value_counts().to_string())
print("\nSEMA triggers:")
print(sema["TRIGGER"].value_counts().to_string())
print("\nSEMA columns:", list(sema.columns))
print("\nSEMA MOOD values:")
print(sema["MOOD"].value_counts(dropna=False).to_string())
