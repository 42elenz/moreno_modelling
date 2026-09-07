"""Phase 4 inputs - the numbers FEASIBILITY.md quotes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, load_daily, load_sema, write
from panel import build_panel

p, meta = build_panel()
p = p.sort_values(["id", "ts"]).reset_index(drop=True)
grid = pd.read_csv(OUT / "05_triple_count_grid.csv")

print("=" * 70)
print("A. CEILINGS")
EV = {"exercise_any": "ev_exercise_any", "exercise_vigorous": "ev_exercise_vigorous",
      "walk_only": "ev_walk", "mindfulness": "ev_mindfulness"}
rows = []
for name, col in EV.items():
    m = p[col].astype(bool)
    # merge consecutive event hours into one session
    sess = 0
    for _, g in p.groupby("id", sort=True):
        v = g[col].to_numpy().astype(bool)
        sess += int(np.sum(v & ~np.concatenate(([False], v[:-1]))))
    rows.append({"event_type": name, "event_hours": int(m.sum()), "distinct_sessions": sess,
                 "max_triples_any_setting": int(grid.query("event_type == @name").n_triples.max()),
                 "max_triples_with_ema": int(grid.query("event_type == @name").n_triples_with_ema.max())})
ceil = write(pd.DataFrame(rows), "09_ceilings.csv")
print(ceil.to_string(index=False))
print(f"\nUnion ceiling (exercise_any OR mindfulness): "
      f"{int((p.ev_exercise_any | p.ev_mindfulness).sum())} event hours")

print("\n" + "=" * 70)
print("B. >1,000 and >5,000 THRESHOLDS")
for thr in [1000, 5000]:
    ok = grid[grid.n_triples > thr]
    oke = grid[grid.n_triples_with_ema > thr]
    print(f"settings yielding >{thr} triples: {len(ok)}/{len(grid)}"
          f"   (event types: {sorted(ok.event_type.unique())})")
    print(f"settings yielding >{thr} triples WITH a mood EMA in the post-window: {len(oke)}"
          f"   (event types: {sorted(oke.event_type.unique())})")

print("\nStrictest settings still clearing 1,000 triples (p=1.0, all 5 hourly channels):")
print(grid.query("p_min == 1.0 and channel_set == '5x_true_hourly' and n_triples > 1000")
      .sort_values(["W_hours", "delta_hours"]).to_string(index=False))

print("\nWhat a 4h/4h window costs, by channel set (exercise_any):")
print(grid.query("event_type=='exercise_any' and W_hours==4 and delta_hours==4")
      .pivot(index="channel_set", columns="p_min", values="n_triples").to_string())

print("\n" + "=" * 70)
print("C. WHO CONTRIBUTES")
ref = pd.read_csv(OUT / "05_triples_per_participant_ref.csv")
for ev in EV:
    v = np.sort(ref.query("event_type == @ev").n_triples.to_numpy())[::-1]
    v = np.concatenate([v, np.zeros(71 - len(v), int)])
    tot = v.sum()
    k = int(np.searchsorted(np.cumsum(v), 0.8 * tot) + 1) if tot else 0
    print(f"{ev:20s} total={tot:5d}  top-{k} participants hold 80% of them; "
          f"median={np.median(v):.0f}  n>=20={int((v>=20).sum())}  n==0={int((v==0).sum())}")

print("\n" + "=" * 70)
print("D. EMA COUPLING")
n_ema = int(p.ema_valid.sum())
print(f"answered mood EMAs on the hourly panel: {n_ema}")
for ev, col in EV.items():
    for D in [1, 2, 4]:
        n = 0
        for _, g in p.groupby("id", sort=True):
            e = np.concatenate(([0], np.cumsum(g.ema_valid.to_numpy().astype(int))))
            idx = np.flatnonzero(g[col].to_numpy().astype(bool))
            idx = idx[idx + 1 + D <= len(g)]
            if len(idx):
                n += int(((e[idx + 1 + D] - e[idx + 1]) > 0).sum())
        print(f"  {ev:20s} events with >=1 answered EMA within {D}h after: {n}")

print("\n" + "=" * 70)
print("E. WHAT THE EVENT HOUR ITSELF LOOKS LIKE")
for ev, col in EV.items():
    sub = p[p[col].astype(bool)]
    print(f"  {ev:20s} n={len(sub):5d}  bpm observed in the event hour: "
          f"{100*sub.bpm.notna().mean():5.1f}%   steps: {100*sub.steps.notna().mean():5.1f}%")

print("\n" + "=" * 70)
print("F. SIGNAL DIMENSIONALITY OF THE LATENT STATE")
X = p[["bpm", "steps", "calories", "distance", "temperature"]].dropna()
print(f"complete 5-channel hours: {len(X)}")
print("correlation matrix:")
print(X.corr().round(3).to_string())
ev_, _ = np.linalg.eigh(np.corrcoef(((X - X.mean()) / X.std()).to_numpy().T))
ev_ = np.sort(ev_)[::-1]
print("PCA explained variance ratio:", np.round(ev_ / ev_.sum(), 3))
print(f"components for 95% variance: {int(np.searchsorted(np.cumsum(ev_ / ev_.sum()), 0.95) + 1)} of 5")

print("\n" + "=" * 70)
print("G. COHORT SPAN")
cov = p.groupby("id").agg(first=("ts", "min"), last=("ts", "max"), n=("ts", "size"))
cov["days"] = (cov["last"] - cov["first"]).dt.days
print(cov["days"].describe().round(1).to_string())
print(f"study window: {p.ts.min()} .. {p.ts.max()}")
