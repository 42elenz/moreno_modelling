"""Phase 2 - feasibility metrics. No imputation anywhere in this file."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT, load_daily, load_sema, write
from panel import (DAILY_STAMPED, MOOD_COLS, PANEL_CHANNELS, PLACE_COLS,
                   TRUE_HOURLY, build_panel)

EVENT_COLS = {
    "exercise_any": "ev_exercise_any",
    "exercise_vigorous": "ev_exercise_vigorous",
    "walk_only": "ev_walk",
    "mindfulness": "ev_mindfulness",
}

p, meta = build_panel()
p = p.sort_values(["id", "ts"]).reset_index(drop=True)
print("panel:", meta)

# =====================================================================  1  ====
print("\n== 1. MISSINGNESS ==")
# Two distinct kinds of absence, kept separate on purpose.
rows = []
for ch in PANEL_CHANNELS:
    present = p[ch].notna()
    rows.append({
        "channel": ch,
        "granularity": "hourly" if ch in TRUE_HOURLY else "daily-stamped-onto-one-hour",
        "dense_slots": len(p),
        "slot_absent_no_csv_row": int((~p["row_present"]).sum()),
        "row_present_but_channel_nan": int((p["row_present"] & ~present).sum()),
        "channel_present": int(present.sum()),
        "pct_present_of_dense": round(100 * present.mean(), 2),
        "pct_present_given_row": round(100 * present[p["row_present"]].mean(), 2),
    })
miss_ch = write(pd.DataFrame(rows), "02_missingness_by_channel.csv")
print(miss_ch.to_string(index=False))

miss_pp = (p.groupby("id")[PANEL_CHANNELS].apply(lambda g: g.notna().mean())
           .round(4).reset_index())
miss_pp["dense_slots"] = p.groupby("id").size().values
miss_pp["row_present_frac"] = p.groupby("id")["row_present"].mean().round(4).values
write(miss_pp, "02_missingness_by_participant.csv")
print("\nper-participant bpm presence: "
      f"min={miss_pp['bpm'].min():.2f} p25={miss_pp['bpm'].quantile(.25):.2f} "
      f"med={miss_pp['bpm'].median():.2f} max={miss_pp['bpm'].max():.2f}")

miss_hod = (p[["hour"]].join(p[PANEL_CHANNELS].notna()).join(p[["row_present"]])
            .groupby("hour").mean().round(4).reset_index())
write(miss_hod, "02_missingness_by_hour_of_day.csv")
print("\npresence by hour-of-day (bpm / steps / row_present):")
print(miss_hod[["hour", "bpm", "steps", "row_present"]].to_string(index=False))

# =====================================================================  2  ====
print("\n== 2. GAP STRUCTURE (contiguous complete runs, hours) ==")
CHANNEL_SETS = {
    "bpm": ["bpm"],
    "bpm+steps": ["bpm", "steps"],
    "bpm+steps+calories": ["bpm", "steps", "calories"],
    "5x_true_hourly": TRUE_HOURLY,
    "5x_hourly+scl_avg": TRUE_HOURLY + ["scl_avg"],
}


def runs_of_true(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.array([], dtype=int)
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


gap_rows, runlen_rows = [], []
for name, cs in CHANNEL_SETS.items():
    all_runs = []
    for pid, g in p.groupby("id", sort=True):
        m = g[cs].notna().all(axis=1).to_numpy()
        r = runs_of_true(m)
        all_runs.append(r)
        gap_rows.append({
            "channel_set": name, "id": pid, "n_runs": len(r),
            "max_run_h": int(r.max()) if len(r) else 0,
            "median_run_h": float(np.median(r)) if len(r) else 0.0,
            "total_complete_h": int(r.sum()),
            "frac_complete": round(float(m.mean()), 4),
            "n_runs_ge_4h": int((r >= 4).sum()), "n_runs_ge_8h": int((r >= 8).sum()),
            "n_runs_ge_12h": int((r >= 12).sum()), "n_runs_ge_24h": int((r >= 24).sum()),
        })
    R = np.concatenate(all_runs) if all_runs else np.array([])
    for L in [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]:
        runlen_rows.append({"channel_set": name, "min_run_h": L,
                            "n_runs": int((R >= L).sum()),
                            "n_hours_in_such_runs": int(R[R >= L].sum())})
    print(f"{name:24s} runs={len(R):6d} median={np.median(R):5.1f}h "
          f"p90={np.percentile(R,90):6.1f}h max={R.max():5.0f}h "
          f"frac_hours_in_runs>=8h={R[R>=8].sum()/max(R.sum(),1):.2f}")
write(pd.DataFrame(gap_rows), "02_gap_runs_by_participant.csv")
write(pd.DataFrame(runlen_rows), "02_gap_run_length_distribution.csv")

# raw run-length histogram for the dashboard
hist_rows = []
for name, cs in CHANNEL_SETS.items():
    R = np.concatenate([runs_of_true(g[cs].notna().all(axis=1).to_numpy())
                        for _, g in p.groupby("id", sort=True)])
    vc = pd.Series(R).value_counts().sort_index()
    for k, v in vc.items():
        hist_rows.append({"channel_set": name, "run_len_h": int(k), "n": int(v)})
write(pd.DataFrame(hist_rows), "02_gap_run_histogram.csv")

# =====================================================================  3  ====
print("\n== 3. EVENT INVENTORY ==")
ev_rows = []
for name, col in EVENT_COLS.items():
    s = p[col].astype(bool)
    per_pp = p.assign(_e=s).groupby("id")["_e"].sum()
    per_pd = p.assign(_e=s).groupby(["id", "date"])["_e"].sum()
    ev_rows.append({
        "event_type": name, "n_events_participant_hours": int(s.sum()),
        "n_participants_with_any": int((per_pp > 0).sum()),
        "median_per_participant": float(per_pp.median()),
        "max_per_participant": int(per_pp.max()),
        "participant_days_with_any": int((per_pd > 0).sum()),
        "total_participant_days": int(per_pd.size),
        "rate_per_participant_day": round(float(s.sum() / per_pd.size), 4),
    })
ev_inv = write(pd.DataFrame(ev_rows), "03_event_inventory.csv")
print(ev_inv.to_string(index=False))

write(p.groupby("id")[list(EVENT_COLS.values())].sum().reset_index(),
      "03_events_per_participant.csv")
write(p.groupby(["id", "date"])[list(EVENT_COLS.values())].sum().reset_index()
      .query(" or ".join(f"{c} > 0" for c in EVENT_COLS.values())),
      "03_events_per_participant_day.csv")

# Sleep onset: explicitly absent. Quantify what does exist instead.
daily = load_daily()
sleep_rows = [{
    "field": "sleep onset timestamp (data.sleep_start in the absent fitbit collection)",
    "available": "NO", "n": 0,
    "note": "no timestamped sleep event anywhere in the released files",
}]
for c in ["sleep_duration", "minutesAsleep", "sleep_efficiency", "sleep_deep_ratio"]:
    sleep_rows.append({"field": f"daily.{c}", "available": "daily aggregate only",
                       "n": int(daily[c].notna().sum()),
                       "note": "one value per participant-night, no onset time"})
write(pd.DataFrame(sleep_rows), "03_sleep_event_availability.csv")
print("\nSLEEP ONSET: not present as an event. Daily sleep aggregates only:")
print(pd.DataFrame(sleep_rows).to_string(index=False))

# =====================================================================  4  ====
print("\n== 4. EMA INVENTORY ==")
sema = load_sema()
sema["is_mood"] = sema["SURVEY_NAME"] == "Context and Mood Survey"
sema["answered"] = ~sema["MOOD"].isin(["<no-response>", "<not-shown>"]) & sema["MOOD"].notna()
ema_rows = [{
    "metric": "sema.bson documents", "value": len(sema)},
    {"metric": "Context and Mood Survey prompts", "value": int(sema["is_mood"].sum())},
    {"metric": "  ...with an actual MOOD answer", "value": int((sema["is_mood"] & sema["answered"]).sum())},
    {"metric": "  ...response rate", "value": round(float(sema.loc[sema["is_mood"], "answered"].mean()), 3)},
    {"metric": "Step Goal Survey prompts", "value": int((~sema["is_mood"]).sum())},
    {"metric": "participants with any SEMA", "value": int(sema["user_id"].nunique())},
    {"metric": "answered mood EMA landing on the hourly panel", "value": int(p["ema_valid"].sum())},
]
write(pd.DataFrame(ema_rows), "04_ema_summary.csv")
print(pd.DataFrame(ema_rows).to_string(index=False))

ema_pp = (p.groupby("id")
          .agg(n_ema=("ema_valid", "sum"),
               n_days=("date", "nunique"))
          .assign(ema_per_day=lambda d: (d.n_ema / d.n_days).round(3))
          .reset_index().sort_values("n_ema", ascending=False))
write(ema_pp, "04_ema_per_participant.csv")
print(f"\nparticipants with >=1 answered mood EMA: {(ema_pp.n_ema>0).sum()}/71; "
      f"median EMA/participant={ema_pp.n_ema.median():.0f}, median EMA/day={ema_pp.ema_per_day.median():.2f}")

write(p[p.ema_valid].groupby("hour").size().rename("n").reset_index(), "04_ema_by_hour_of_day.csv")
write(p[p.ema_valid]["ema_mood"].value_counts().rename("n").reset_index(), "04_ema_mood_distribution.csv")
write(p[p.ema_valid]["ema_place"].value_counts(dropna=False).rename("n").reset_index(), "04_ema_place_distribution.csv")
print("\nEMA mood distribution:")
print(p[p.ema_valid]["ema_mood"].value_counts().to_string())

# time since previous event, for each answered EMA
tsp_rows = []
for name, col in EVENT_COLS.items():
    for pid, g in p.groupby("id", sort=True):
        ev_idx = np.flatnonzero(g[col].to_numpy().astype(bool))
        em_idx = np.flatnonzero(g["ema_valid"].to_numpy())
        if len(ev_idx) == 0 or len(em_idx) == 0:
            continue
        j = np.searchsorted(ev_idx, em_idx, side="right") - 1
        ok = j >= 0
        for dh in (em_idx[ok] - ev_idx[j[ok]]):
            tsp_rows.append({"event_type": name, "id": pid, "hours_since_prev_event": int(dh)})
tsp = pd.DataFrame(tsp_rows)
write(tsp, "04_ema_time_since_prev_event.csv")
print("\ntime since previous event, per EMA (hours):")
print(tsp.groupby("event_type")["hours_since_prev_event"]
      .describe(percentiles=[.1, .25, .5, .75]).round(1).to_string())
for name in EVENT_COLS:
    s = tsp[tsp.event_type == name]["hours_since_prev_event"]
    if len(s):
        print(f"  {name:20s} EMA within 4h of a prior event: {int((s<=4).sum()):5d} "
              f"({100*(s<=4).mean():.1f}% of {len(s)})")

# =====================================================================  5  ====
print("\n== 5. TRIPLE COUNT ==")


def triple_count(panel, W_h, D_h, channels, pmin, event_col, require_ema):
    """W_h hours before the event hour, D_h hours after. No imputation."""
    per_pp, total, total_ema = {}, 0, 0
    for pid, g in panel.groupby("id", sort=True):
        M = g[channels].notna().to_numpy()
        ok = M.sum(axis=1)                       # channels present per hour
        cs = np.concatenate(([0], np.cumsum(ok)))
        ema = np.concatenate(([0], np.cumsum(g["ema_valid"].to_numpy().astype(int))))
        idx = np.flatnonzero(g[event_col].to_numpy().astype(bool))
        idx = idx[(idx - W_h >= 0) & (idx + D_h < len(g))]
        if len(idx) == 0:
            continue
        pre = (cs[idx] - cs[idx - W_h]) / (W_h * len(channels))
        post = (cs[idx + 1 + D_h] - cs[idx + 1]) / (D_h * len(channels))
        good = (pre >= pmin) & (post >= pmin)
        has_ema = (ema[idx + 1 + D_h] - ema[idx + 1]) > 0
        n = int((good & has_ema).sum()) if require_ema else int(good.sum())
        if n:
            per_pp[pid] = n
        total += int(good.sum())
        total_ema += int((good & has_ema).sum())
    return (total_ema if require_ema else total), per_pp, total, total_ema


grid = []
for ev in EVENT_COLS:
    for W_h in [1, 2, 4, 8, 12, 24]:
        for D_h in [1, 2, 4, 8, 24]:
            for cname, cs in CHANNEL_SETS.items():
                for pmin in [0.5, 0.75, 0.9, 1.0]:
                    n, per_pp, n_noema, n_ema = triple_count(
                        p, W_h, D_h, cs, pmin, EVENT_COLS[ev], False)
                    grid.append({
                        "event_type": ev, "W_hours": W_h, "delta_hours": D_h,
                        "channel_set": cname, "n_channels": len(cs), "p_min": pmin,
                        "n_triples": n_noema, "n_triples_with_ema": n_ema,
                        "n_participants": len(per_pp),
                        "n_participants_ge10": sum(v >= 10 for v in per_pp.values()),
                        "n_participants_ge50": sum(v >= 50 for v in per_pp.values()),
                    })
gridf = write(pd.DataFrame(grid), "05_triple_count_grid.csv")
print(f"grid rows: {len(gridf)}")
print("\nBEST settings by raw triple count:")
print(gridf.sort_values("n_triples", ascending=False).head(15).to_string(index=False))
print("\nBEST settings by triples WITH a mood EMA in the post-window:")
print(gridf.sort_values("n_triples_with_ema", ascending=False).head(15).to_string(index=False))
print("\nHeadline reference setting (W=4h, D=4h, bpm+steps, p=0.9):")
print(gridf.query("W_hours==4 and delta_hours==4 and channel_set=='bpm+steps' and p_min==0.9").to_string(index=False))

# per-participant breakdown at a reference setting
ref_rows = []
for ev in EVENT_COLS:
    _, per_pp, _, _ = triple_count(p, 4, 4, CHANNEL_SETS["bpm+steps"], 0.9, EVENT_COLS[ev], False)
    for pid, n in per_pp.items():
        ref_rows.append({"event_type": ev, "id": pid, "n_triples": n})
write(pd.DataFrame(ref_rows), "05_triples_per_participant_ref.csv")

# =====================================================================  7  ====
print("\n== 7. PER-PARTICIPANT VIABILITY CURVE ==")
via = []
for ev in EVENT_COLS:
    for cname in ["bpm", "bpm+steps", "5x_true_hourly"]:
        for pmin in [0.75, 0.9]:
            _, per_pp, _, _ = triple_count(p, 4, 4, CHANNEL_SETS[cname], pmin, EVENT_COLS[ev], False)
            v = np.array(sorted(per_pp.values(), reverse=True)) if per_pp else np.array([])
            for thr in [1, 5, 10, 20, 30, 50, 100, 200]:
                via.append({"event_type": ev, "channel_set": cname, "p_min": pmin,
                            "min_triples_per_participant": thr,
                            "n_participants": int((v >= thr).sum())})
viadf = write(pd.DataFrame(via), "07_viability_curve.csv")
print(viadf.query("channel_set=='bpm+steps' and p_min==0.9").pivot(
    index="min_triples_per_participant", columns="event_type", values="n_participants").to_string())

json.dump({"panel": meta, "channel_sets": CHANNEL_SETS, "event_cols": EVENT_COLS},
          open(OUT / "02_config.json", "w"), indent=1, default=str)
print("\ndone")
