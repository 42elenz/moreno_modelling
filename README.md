# LifeSnaps feasibility assessment

Can the LifeSnaps release in this repository support an action-conditioned latent
transition model — learn `z_t` from multivariate wearable state, then
`P(z_{t+Δ} | z_t, a_t)` for a timestamped behavioural action `a_t`, at Δ of 1–4 h?

**Short answer: no, not as specified.** Read [`FEASIBILITY.md`](FEASIBILITY.md).

| document | what it holds |
|---|---|
| [`FEASIBILITY.md`](FEASIBILITY.md) | the verdict, with numbers |
| [`INVENTORY.md`](INVENTORY.md) | Phase 1 — every file, channel, granularity, coverage and timezone finding |
| `dashboard/lifesnaps_triple_audit.html` | Phase 3 — the interactive audit, open it in a browser |
| `outputs/*.csv` | Phase 2 — tidy tables behind every claim |

## Reproducing

```bash
pip install pandas numpy pymongo scikit-learn matplotlib pyarrow
python3 analysis/01_inventory.py          # discovery inventory
python3 analysis/panel.py                 # dense hourly panel  -> outputs/panel.parquet
python3 analysis/02_feasibility.py        # missingness, gaps, events, EMA, triple grid, viability
python3 analysis/06_positivity.py         # propensity / overlap
python3 analysis/07_export_dashboard.py   # payload
python3 analysis/08_build_dashboard.py    # standalone HTML
python3 analysis/09_headline_numbers.py   # the numbers FEASIBILITY.md quotes
```

Runtime is a few minutes end to end. **No imputation happens anywhere** in the
Phase-2 pipeline; missing is missing, and the distinction between "no CSV row for
this hour" and "row exists but this channel is NaN" is preserved throughout.

## The dashboard

`dashboard/lifesnaps_triple_audit.html` is **one self-contained file** — open it
from disk, no server, no environment. It carries a 0.41 MB embedded payload: the
71 × 180,305 × 10 participant/hour/channel presence tensor, bit-packed, plus event
and EMA index arrays.

*Why one HTML file rather than Streamlit.* The triple count is a prefix-sum query
over that presence bitmap. Packed, the whole tensor is small enough to sit inside
the file, so every control recomputes the headline number in the browser in
milliseconds — no running process, nothing to install, and the artifact stays
valid when you mail it to someone. Streamlit would need a live server to do the
same job and would recompute the same prefix sums on every widget change.

Controls, all of which recompute the count live: pre-window `W` (5 min – 24 h),
horizon `Δ` (30 min – 24 h), channel set `C` (multi-select over the channels that
actually exist, daily-aggregate ones flagged), completeness threshold `p`
(0.5–1.0), event type, and a "require a mood EMA inside Δ" toggle. Displays: the
live triple count with per-participant breakdown, the participants-vs-required-
triples curve, a participant × study-day coverage heatmap for the selected channel
set, the contiguous-run length distribution, the precomputed propensity overlap
plot, and the channel granularity table.

`W` and `Δ` are entered in minutes as specified and rounded up to whole hours,
because the underlying grid is hourly. The dashboard says so on screen rather than
pretending otherwise.

The browser-side triple count was verified against the Python implementation on
six independent settings (including the EMA-required variants) and matches exactly,
participant by participant.

## Layout

```
analysis/       numbered pipeline scripts + common.py (loaders) + panel.py (dense hourly panel)
outputs/        tidy CSVs, the propensity plot, the dashboard payload
dashboard/      template.html (source) and lifesnaps_triple_audit.html (built, standalone)
*.csv *.bson    the LifeSnaps release as received
```
