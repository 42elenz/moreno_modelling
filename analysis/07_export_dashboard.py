"""Phase 3 - export a compact payload for the self-contained HTML dashboard.

Design note (why one HTML file and not Streamlit): the triple count is a
prefix-sum query over a per-participant presence bitmap. Packed, the whole
71-participant x 180k-hour x 10-channel presence tensor is ~225 KB, so it fits
inside a single HTML file and every control can recompute the headline number
in the browser with no server, no environment, and no re-run. Streamlit would
need a running process to do the same job.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT
from panel import (DAILY_STAMPED, PANEL_CHANNELS, TRUE_HOURLY, build_panel)

EVENT_COLS = {
    "exercise_any": "ev_exercise_any",
    "exercise_vigorous": "ev_exercise_vigorous",
    "walk_only": "ev_walk",
    "mindfulness": "ev_mindfulness",
}


def packbits_b64(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(np.uint8)).tobytes()).decode()


def main():
    p, meta = build_panel()
    p = p.sort_values(["id", "ts"]).reset_index(drop=True)

    parts = []
    for i, (pid, g) in enumerate(p.groupby("id", sort=True)):
        g = g.sort_values("ts")
        rec = {
            "id": pid,
            "short": f"P{i+1:02d}",
            "t0": g["ts"].iloc[0].isoformat(),
            "n": int(len(g)),
            "ch": {c: packbits_b64(g[c].notna().to_numpy()) for c in PANEL_CHANNELS},
            "ema": np.flatnonzero(g["ema_valid"].to_numpy()).astype(int).tolist(),
        }
        for name, col in EVENT_COLS.items():
            rec[name] = np.flatnonzero(g[col].to_numpy().astype(bool)).astype(int).tolist()
        parts.append(rec)

    gran = []
    for c in PANEL_CHANNELS:
        num = pd.to_numeric(p[c], errors="coerce")
        nun = p.assign(_v=num).groupby(["id", "date"])["_v"].nunique()
        gran.append({
            "channel": c,
            "claimed": "hourly column",
            "actual": "hourly (varies within participant-day)" if c in TRUE_HOURLY
                      else "DAILY aggregate stamped onto one hour",
            "frac_pdays_varying": round(float((nun > 1).mean()), 3),
            "pct_present_of_dense_hours": round(100 * float(num.notna().mean()), 2),
            "min": None if num.notna().sum() == 0 else round(float(num.min()), 3),
            "max": None if num.notna().sum() == 0 else round(float(num.max()), 3),
        })

    prop = pd.read_csv(OUT / "06_propensity_histograms.csv")
    prop_sum = pd.read_csv(OUT / "06_positivity_summary.csv")

    payload = {
        "meta": {
            "n_participants": int(p["id"].nunique()),
            "n_dense_hours": int(len(p)),
            "n_rows_present": int(p["row_present"].sum()),
            "grid": "hourly",
            "date_min": str(p["ts"].min()), "date_max": str(p["ts"].max()),
            "true_hourly": TRUE_HOURLY, "daily_stamped": DAILY_STAMPED,
        },
        "channels": PANEL_CHANNELS,
        "event_types": list(EVENT_COLS),
        "participants": parts,
        "granularity": gran,
        "propensity": prop.to_dict("records"),
        "propensity_summary": prop_sum.fillna("").to_dict("records"),
        "event_counts": {k: int(p[v].astype(bool).sum()) for k, v in EVENT_COLS.items()},
        "ema_total": int(p["ema_valid"].sum()),
    }
    out = OUT / "dashboard_payload.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    return payload


if __name__ == "__main__":
    main()
