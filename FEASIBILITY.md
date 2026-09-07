# Feasibility verdict: action-conditioned latent transition model on LifeSnaps

**Not as specified. The version of LifeSnaps in this repository has no
minute-level data at all — `fitbit.bson`, the collection holding minute heart
rate, sleep-onset timestamps and exercise session start times, is absent, and
only its index metadata was shipped. The finest grid that exists is one hour, so
your Δ of 1–4 hours is 1–4 samples and a 5-minute pre-window is one sample. Of
the three actions you named, sleep onset does not exist as a timestamped event
anywhere in the release, mindfulness sessions number 131 event-hours across the
whole cohort, and exercise sessions are hour-bucket labels with ±60 min timing
uncertainty rather than timestamps. What remains — an exercise arm on ~2,900
usable hourly triples across ~45 participants — is enough to build and debug the
pipeline and to run a narrow pilot, and not enough to be the study you
described.**

Every number below is reproducible: `analysis/01`–`09`, tidy tables in `outputs/`,
live recomputation in `dashboard/lifesnaps_triple_audit.html`. No imputation
anywhere.

---

## 1. Is there any (W, Δ, C) setting yielding >1,000 usable triples? >5,000?

**>1,000: yes, in 1,061 of the 2,400 grid settings tested — but only for
`exercise_any` and `walk_only`.**
**>5,000: no, and it is structurally impossible.**

The ceiling is not missingness, it is events. There are only this many
action-like hours in the entire dataset:

| action | event-hours | distinct sessions | best triple count over all settings | best *with* a mood EMA in Δ |
|---|---:|---:|---:|---:|
| exercise (any label) | 3,909 | 3,458 | 3,764 | 1,807 |
| walk only | 2,871 | 2,606 | 2,796 | 1,324 |
| exercise excluding walk | 1,039 | 953 | 973 | 489 |
| mindfulness | 131 | 129 | 123 | 63 |

Union of all action types: **4,033 event-hours**. No windowing choice can produce
more triples than there are events, so 5,000 is unreachable by a margin of 20%
even at W = Δ = 1 h and p = 0.5.

Reference settings, `exercise_any`, all five true-hourly channels:

| W | Δ | p | triples | with EMA | participants ≥1 | ≥50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 h | 1 h | 1.0 | 2,845 | 215 | 59 | 23 |
| 4 h | 4 h | 0.9 | 2,544 | 770 | 59 | 20 |
| 4 h | 4 h | 1.0 | 2,344 | 722 | 59 | 19 |
| 8 h | 4 h | 1.0 | 2,091 | 634 | 57 | 14 |
| 12 h | 24 h | 1.0 | 1,006 | 563 | 51 | 5 |

Cost of adding channels at W = Δ = 4 h, `exercise_any`:

| channel set | p=0.5 | p=0.75 | p=0.9 | p=1.0 |
|---|---:|---:|---:|---:|
| `bpm` | 3,589 | 3,388 | 3,096 | 3,096 |
| `bpm+steps` | 3,573 | 3,356 | 2,893 | 2,893 |
| `bpm+steps+calories` | 3,762 | 3,384 | 3,066 | 2,893 |
| all 5 true-hourly | 3,570 | 3,228 | 2,544 | 2,344 |
| the 5 + `scl_avg` | 3,467 | 2,544 | **0** | **0** |

The last row is the trap. `scl_avg` looks like a sixth channel and is a daily
aggregate written into one hour of the day; requiring it at p ≥ 0.9 takes the
count to zero.

Gap structure is *not* the binding constraint, and this is the one pleasant
surprise in the audit. For `bpm+steps` there are 4,278 contiguous fully-observed
runs totalling 92,278 hours; 2,747 runs are ≥8 h and 1,071 are ≥24 h. For all
five hourly channels: 2,994 runs, 72,142 hours, 935 runs ≥24 h. Windows of
W + Δ + 1 ≤ 24 h fit comfortably. The data is missing in *blocks* (device off for
days), not shredded — 95% of hours in complete runs sit in runs of ≥8 h.

**Requiring a mood EMA inside Δ is what actually breaks the outcome arm.** Only
1,024 of 3,909 exercise-hours have an answered mood EMA within 4 h after; 545
within 2 h; 263 within 1 h. For mindfulness it is 14, 8 and 6. To clear 1,800
EMA-coupled triples you must stretch Δ to 24 h, which is no longer the
1–4 h mechanism you set out to model.

---

## 2. How many participants contribute meaningfully?

At W = Δ = 4 h, `bpm+steps`, p = 0.9, `exercise_any` (2,893 triples):

| threshold | participants of 71 |
|---|---:|
| ≥1 triple | 68 |
| ≥5 | 62 |
| ≥10 | 53 |
| ≥20 | 45 |
| ≥50 | 25 |
| ≥100 | 4 |
| ≥200 | 0 |

**32 participants hold 80% of all triples. Median contribution is 33.** For
exercise excluding walk it collapses: median 5 per participant, 15 participants
with ≥20, **exactly one** with ≥50. For mindfulness, 49 of 71 participants have
zero events and no participant reaches 20 triples.

Wear time is the driver, and it is bimodal. Per-participant `bpm` presence over
the participant's own span: median 0.64, but p25 = 0.35 and 21 of 71 sit below
0.40 — with 7 below 0.20, i.e. functionally absent. Overall wear is ~53% of
dense hours for `bpm`, with a shallow diurnal profile (48% at 04:00, 57% at
12:00) — no hour of day is well covered.

Practical read: budget for **~45 participants** in a hierarchical or
partially-pooled model, and about **25** if you want ≥50 triples each for
anything per-person. The remaining ~26 are noise you will spend effort excluding.

---

## 3. Which channels are genuinely minute-resolution, and which are daily aggregates dressed up as time series?

**None are minute-resolution. Zero.** That column of the answer is empty because
the collection that held minute data is not in the release.

Genuinely **hourly**, in the hourly table (test: does the column ever take two
values inside one participant-day?):

| channel | varies within participant-day | observed |
|---|---:|---:|
| `bpm` (hourly mean HR) | 67.9% | 53.3% |
| `calories` | 68.8% | 88.2% |
| `steps` | 66.7% | 51.6% |
| `distance` | 66.7% | 51.6% |
| `temperature` (skin-temp deviation) | 47.1% | 41.5% |

**Daily aggregates sitting in the hourly table**, all at 0.0% within-day
variation: `scl_avg`, `minutes_in_default_zone_1`, `minutes_below_default_zone_1`,
`minutes_in_default_zone_2`, `minutes_in_default_zone_3` (plus `step_goal`,
`min_goal`, `max_goal`). `minutes_below_default_zone_1` maxing at 1,440 —
minutes in a day — settles it.

**Daily-only, no hourly counterpart at all**, so unusable at a 1–4 h horizon:
`rmssd`, `nremhr`, `spo2`, `resting_hr`, `stress_score`, `nightly_temperature`,
`full_sleep_breathing_rate`, `filteredDemographicVO2Max`, every sleep variable,
and the three Fitbit "readiness" percentages. This is where HRV, SpO2 and sleep
live — everything you would want a physiological latent state to contain.

And the five hourly channels are not five independent dimensions. On the 72,142
fully-observed hours: `steps`↔`distance` r = 0.989 (one channel), `calories`
correlates 0.75 with both, `bpm` 0.51–0.57 with the activity block, `temperature`
−0.31 to −0.38 with everything. PCA: 65.7% / 16.9% / 11.3% / 5.9% / 0.2% — four
components for 95% variance, and honestly ~3 interpretable ones (activity volume,
cardiac load, skin temperature).

**So z_t at W = 4 h is a compression of roughly 12 informative numbers.** A latent
representation learner is not the right tool for that, and it is not where the
scientific content of your project was supposed to be.

---

## 4. Does positivity hold for any action type?

**Yes — this is the one binding constraint you listed that does *not* fail.**

Gradient-boosted classifier predicting "event in this hour" from a 4 h fully
observed pre-window (mean/std/last/max/min of bpm, steps, calories; bpm delta;
step sum; hour-of-day sin/cos; day of week; hours since last event of that type),
scored out-of-fold with participant-grouped 5-fold CV over 80,929 eligible
decision hours:

| action | positives | base rate | AUC logistic | AUC GBM | median within-participant AUC | verdict |
|---|---:|---:|---:|---:|---:|---|
| exercise (any) | 3,311 | 4.09% | 0.727 | **0.763** | 0.742 | overlap holds |
| walk only | 2,489 | 3.08% | 0.739 | **0.772** | 0.740 | overlap holds |
| exercise excl. walk | 836 | 1.03% | 0.809 | **0.828** | 0.737 | holds, but close to your 0.85 line |
| mindfulness | 100 | 0.12% | 0.807 | 0.401 | 0.610 | too few positives to conclude anything |

96% of acted hours fall inside the 1st–99th percentile propensity range of
not-acted hours for `exercise_any` (91% for vigorous exercise). Common support is
real. The within-participant AUCs sit at ~0.74 for all three exercise variants,
so the separability is genuine within-person timing structure (time of day,
recent activity), not just between-person differences in who exercises.

The mindfulness AUC of 0.401 is not evidence of good overlap — it is what a model
does with 100 positives in 81,000 rows. Ignore it.

---

## 5. Recommendation

**Viable for pipeline debugging and for one narrow pilot; not viable as the study
you described.**

What you can actually run today, honestly stated:

> *Given the hourly wearable state (mean HR, steps, calories, skin-temp
> deviation) over the previous W hours, does an hour containing a
> Fitbit-detected exercise bout shift the next 1–4 hourly samples, relative to
> matched non-exercise hours?*

That has ~2,900 triples, ~45 contributing participants, and adequate propensity
overlap. It will exercise every part of your machinery: window extraction,
completeness gating, the transition model, the propensity/overlap diagnostics,
participant-level pooling. Build it. Just do not call the resulting z_t a latent
representation of multivariate physiological state — it is a smoothing of three
correlated hourly summaries.

What is **not** viable here, at all:

- **Sleep onset as an action.** No timestamp exists. Nothing recovers it.
- **Mindfulness as an action.** 131 event-hours, 74 triples, 22 participants with
  any, one participant with ≥10. This is not an underpowered arm, it is an empty one.
- **Any outcome involving HRV, SpO2, resting HR, respiratory rate, stress score or
  sleep quality.** All daily-only.
- **Mood EMA as the outcome at a 1–4 h horizon.** 263 / 545 / 1,024 exercise events
  have an answered EMA within 1 / 2 / 4 h, and that outcome is a single 7-way
  categorical item, not a scale. You would be fitting a 7-class transition on
  ~1,000 observations that are 43% response-rate-selected and heavily clustered at
  10:00, 15:00 and 20:00.
- **Sub-hourly anything.** The dashboard accepts W down to 5 minutes because you
  asked for the control; it rounds to one hour and says so, which is the honest
  behaviour.

### What would have to be true instead

1. **Get `fitbit.bson`.** This is the single fix, and it is probably cheap. The
   full LifeSnaps RAIS MongoDB dump (Zenodo, ~10 GB) contains the collection whose
   metadata is sitting in this repository unaccompanied. With it you get
   minute-level `bpm`, sleep stage series with `sleep_start`, and exercise sessions
   with real `startTime` values — which converts every "absent" and "hour-bucket"
   caveat above into a normal analysis problem. **Do this before doing anything
   else.** Nothing in the rest of the list matters if this works, and everything in
   this repository will run unchanged on a finer grid.
2. If it cannot be obtained: **the mindfulness and sleep arms need a different
   dataset.** No amount of modelling recovers 131 events or a timestamp that was
   never released.
3. If you need mood as an outcome at 1–4 h: **you need a study with event-contingent
   EMA prompts.** LifeSnaps' prompts are all `TRIGGER = "scheduled"` — they were
   never fired after actions, which is exactly why so few land in a post-window.
4. If you need ≥5,000 triples: **you need more participants or a denser action
   definition**, not better windowing. At 3,458 exercise sessions across 71 people
   over a median 86 days of data, the event rate is 0.52 exercise-hours per
   participant-day. Even perfect wear compliance leaves you under 4,000.

### One caveat on anything you do build

There is no timezone information in the release, and the study window crosses both
the EU (2021-10-31) and US (2021-11-07) DST transitions. The SEMA and Fitbit clocks
agree with each other (15,292 of 15,380 EMA documents land on an hour that exists in
the hourly table), so within-participant alignment is fine — but absolute
time-of-day comparisons across participants after late October 2021 carry an
unresolvable ±1 h ambiguity. Since hour-of-day is one of the stronger propensity
features, that is worth stating in any writeup.
