# Phase 1 — Data inventory (LifeSnaps, as released in this repository)

Everything below was read off the files, not off the paper. Reproduce with
`python3 analysis/01_inventory.py`; tidy outputs land in `outputs/01_*.csv`.

---

## 0. Read this before anything else

### The minute-level Fitbit collection is not in this release

`fitbit.metadata.json` is present. `fitbit.bson` is not, and never was — it does
not appear anywhere in this repository's git history.

The orphaned metadata file describes the collection that is missing, and its
index list names exactly the fields the project depends on:

| indexed field in the absent collection | what it would have given you |
|---|---|
| `data.value.bpm` + `data.dateTime` | minute-resolution heart rate |
| `data.sleep_start` | timestamped sleep onset |
| `data.startTime` | timestamped exercise session start |
| `data.reading_time` | timestamped SpO2 / sensor readings |
| `data.timestamp` | generic per-reading timestamp |

**The finest temporal grid that actually exists here is one hour.** A 1–4 hour
horizon is therefore 1–4 samples, and a "pre-window" of 5–60 minutes is a single
sample. Every number in Phases 2–4 is computed on that hourly grid.

### Named channels that do not exist as you described them

| you asked for | status |
|---|---|
| **sleep onset** as a timestamped event | **ABSENT.** No onset time anywhere. Only per-night daily aggregates survive (`sleep_duration`, `minutesAsleep`, `minutesAwake`, `minutesToFallAsleep`, `sleep_efficiency`, the four stage ratios) — 3,551 participant-nights. You cannot place sleep onset on a clock. |
| **mindfulness sessions** | Exists, as an hourly boolean. **131 `True` hours in the entire cohort**, across 28 of 71 participants. 49 participants have none. |
| **exercise sessions** | Exists, but **not timestamped**: `activityType` is a list of labels attached to an *hour bucket*. 3,909 event-hours → 3,458 distinct sessions after merging consecutive hours. Session start time is unknown to ±60 min. |
| **EMA** | Exists and is genuinely minute-stamped (in `sema.bson`). See §5. |

### Columns in the hourly table that are daily aggregates in disguise

`scl_avg`, `minutes_in_default_zone_1`, `minutes_below_default_zone_1`,
`minutes_in_default_zone_2`, `minutes_in_default_zone_3`.

Each takes **exactly one value per participant-day** (0.0% within-day variation)
and is written into a single hourly row; the rest of that day's 23 rows are NaN.
`minutes_below_default_zone_1` reaching 1,440 — minutes in a day — is the giveaway,
and `outputs/02_missingness_by_hour_of_day.csv` closes the case: all five are
observed **only at hour 0** (5.8% of midnight hours for `scl_avg`, 63.7% for the
four zone columns) and are 0.0% observed at every one of the other 23 hours.
Treating any of these as an hourly channel silently destroys 96–99% of your
windows.

### One data-quality artifact

70 exact-duplicate `(id, timestamp)` row pairs, all at hour 0, across 2
participants. Dropped (`keep="first"`) in every downstream script.

---

## 1. Files

| file | MB | records | what it is |
|---|---:|---:|---|
| `hourly_fitbit_sema_df_unprocessed.csv` | 21.14 | 159,508 rows | **the only sub-daily source.** 38 columns, 71 participants |
| `daily_fitbit_sema_df_unprocessed.csv` | 2.07 | 7,410 rows | 62 columns, 71 participants. Sleep, HRV, SpO2, stress score live only here |
| `sema.bson` | 10.71 | 15,380 docs | SEMA3 EMA prompts, minute-stamped |
| `surveys.bson` | 0.64 | 935 docs | raw LimeSurvey item responses |
| `breq.csv` | 0.01 | 92 | scored BREQ (motivation) |
| `panas.csv` | 0.01 | 268 | scored PANAS |
| `personality.csv` | 0.01 | 50 | scored IPIP + gender |
| `stai.csv` | 0.02 | 279 | scored STAI |
| `ttm.csv` | 0.02 | 94 | scored TTM stage-of-change |
| `fitbit.metadata.json` | 0.00 | — | **index metadata for a collection that is not here** |
| `sema.metadata.json`, `surveys.metadata.json` | 0.00 | — | index metadata |

Participant-ID spaces are consistent: `hourly.id`, `daily.id`, `sema.user_id`,
`surveys.user_id` and the scored-survey `user_id` columns are all the same 24-hex
Mongo ObjectId strings. 71 participants in the wearable tables; 63 of them appear
in `sema.bson`; 67 in `surveys.bson`.

---

## 2. Channels, with actual granularity

Full table: `outputs/01_channel_profile.csv` (100 rows). The decisive column is
`frac_pdays_varying` — the fraction of participant-days on which the column takes
more than one value. Anything at 0.000 is not a time series.

### Hourly table — genuinely hourly

| channel | dtype | units / range | % of dense hours observed | varies within a participant-day |
|---|---|---|---:|---:|
| `bpm` | float64 | mean HR over the hour, 33.0 – 192.8 | 53.3% | 67.9% |
| `steps` | float64 | steps in the hour, 0 – 9,350 | 51.6% | 66.7% |
| `calories` | float64 | kcal in the hour, 0.69 – 1,139 | 88.2% | 68.8% |
| `distance` | float64 | metres in the hour, 0 – 10,460 | 51.6% | 66.7% |
| `temperature` | float64 | skin-temp **deviation**, −10.01 – +4.97 | 41.5% | 47.1% |

`calories` is present far more often than the rest because Fitbit estimates a
basal rate whether or not the device is worn — it is a poor wear indicator.
`steps` and `distance` are near-collinear (r = 0.989) and are one channel in practice.

### Hourly table — daily aggregates stamped onto one hour

| channel | % of dense hours | varies within a participant-day | range |
|---|---:|---:|---|
| `scl_avg` | 0.24% | **0.0%** | 0 – 33.47 |
| `minutes_in_default_zone_1` | 2.67% | **0.0%** | 0 – 1,023 |
| `minutes_below_default_zone_1` | 2.67% | **0.0%** | 0 – 1,440 |
| `minutes_in_default_zone_2` | 2.67% | **0.0%** | 0 – 313 |
| `minutes_in_default_zone_3` | 2.67% | **0.0%** | 0 – 247 |

### Hourly table — event-like, not numeric

| column | non-null hours | notes |
|---|---:|---|
| `activityType` | 3,912 (2.45%) | stringified python list, e.g. `['Walk']`. 18 distinct labels; Walk 2,903, Bike 255, Workout 178, Sport 170, Aerobic Workout 141, Run 127, Swim 65, Yoga/Pilates 52, Weights 44, Treadmill 37, Circuit Training 37, Elliptical 14, Hike 13, Martial Arts 6, Interval Workout 5, Spinning 3, Bootcamp 2, Tennis 2 |
| `mindfulness_session` | 159,112 (99.75%) — but **131 `True`** | hourly boolean |
| `badgeType` | 521 (0.33%) | Fitbit achievement notifications, not behaviour |
| `step_goal`, `min_goal`, `max_goal`, `step_goal_label` | ~1,919 (1.2%) | daily, 0.0% within-day variation |

### Hourly table — static per participant

`age` (binned `<30` / `>=30`), `gender` (MALE/FEMALE), `bmi` (binned at the tails:
`<19`, `>=25`, `>=30`, integers between). All k-anonymised.

### Hourly table — EMA one-hots merged in upstream

`ALERT, HAPPY, NEUTRAL, RESTED/RELAXED, SAD, TENSE/ANXIOUS, TIRED` and
`ENTERTAINMENT, GYM, HOME, HOME_OFFICE, OTHER, OUTDOORS, TRANSIT, WORK/SCHOOL` —
each non-null on the same 5,029 hours. This is the join of `sema.bson` onto the
hourly grid, already done for you. 5,029 of the 5,036 answered mood EMAs land on
an hourly row.

### Daily table — exists ONLY at daily granularity

These have no hourly counterpart anywhere, so nothing in them can enter a
1–4 hour transition model:

`nightly_temperature` (45.3%), `nremhr` (33.4%), `rmssd` (33.4%), `spo2` (17.1%),
`full_sleep_breathing_rate` (33.7%), `stress_score` (25.3%),
`sleep_points_percentage` / `exertion_points_percentage` /
`responsiveness_points_percentage` (25.3%), `daily_temperature_variation` (44.6%),
`filteredDemographicVO2Max` (64.7%), `resting_hr` (59.7%),
`lightly_/moderately_/very_active_minutes`, `sedentary_minutes` (95.6%),
`sleep_duration` (ms, 3.6e6 – 7.4e7), `minutesToFallAsleep`, `minutesAsleep`,
`minutesAwake`, `minutesAfterWakeup`, `sleep_efficiency`, and the four sleep stage
ratios (44.5–47.9%).

---

## 3. Per-participant coverage

`outputs/01_participant_coverage.csv`. 71 participants.

| | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| hourly rows | 1,392 | 1,625 | 1,994 | 2,466 | 5,901 |
| calendar days present | 63 | 68.5 | 86 | 106 | 244 |
| observation span (days) | 62.4 | 68.5 | 85.7 | 114.4 | 243 |
| daily rows | 64 | 73.5 | 88 | 114 | 244 |
| SEMA docs (63 participants) | 48 | 228 | 249 | 251 | 498 |

Study window: **2021-04-08 13:00 → 2022-01-22 00:00**. Cohort start dates are
staggered in three visible waves (late May, mid-October, mid-November 2021), so
calendar time is not comparable across participants — use time-since-enrolment.

Densified over each participant's own span the panel is **180,305 participant-hours**,
of which 159,438 (88.4%) have a CSV row at all.

Note the ~4-month figure from the paper describes the *span*; median calendar days
with any data is 86.

---

## 4. Timezone handling

**There is no timezone information anywhere in the release.**

| field | form |
|---|---|
| `hourly.date` | `YYYY-MM-DD` string, no tz; `hour` is a separate float column, 0–23 |
| `daily.date` | `YYYY-MM-DD` string, no tz |
| `sema.CREATED_TS` / `SCHEDULED_TS` / `STARTED_TS` / `COMPLETED_TS` / `UPLOADED_TS` | naive ISO local time, minute resolution |

Consistency across sources is good in practice: 15,292 of 15,380 SEMA documents
land on an hour that exists in the hourly table, and 5,029 of 5,036 answered mood
EMAs do. So the two clocks agree.

What cannot be verified: (a) which timezone that is, per participant — the cohort
was multi-site; (b) DST handling. The study window crosses the EU DST transition
of 2021-10-31 and the US one of 2021-11-07, and there is no way to tell whether
the repeated/skipped local hour was collapsed, duplicated, or shifted. If you
build anything that depends on absolute time-of-day alignment across participants,
treat a ±1 h ambiguity after late October 2021 as unresolvable from these files.

---

## 5. EMA (`sema.bson`)

| | |
|---|---:|
| documents | 15,380 |
| "Context and Mood Survey" prompts | 11,526 |
| …with an actual `MOOD` answer | **5,036** (43.7% response rate) |
| "Step Goal Survey" prompts | 3,854 |
| participants with any SEMA | 63 of 71 |
| answered mood EMAs landing on the hourly panel | 5,029 |

All 15,380 have `TRIGGER = "scheduled"` — there are no event-contingent or
participant-initiated prompts.

Per document: `PLACE` (8 categories), `MOOD` (7 categories), `STEPS`, each with a
reaction time, plus `CREATED_TS` / `SCHEDULED_TS` / `STARTED_TS` / `COMPLETED_TS`
/ `UPLOADED_TS`. Unanswered items are the string `<no-response>`; unshown items
`<not-shown>`.

**The mood outcome is a single 7-way categorical item**, not a scale:
RESTED/RELAXED 1,178 · TIRED 1,126 · NEUTRAL 822 · HAPPY 790 · TENSE/ANXIOUS 620 ·
ALERT 344 · SAD 149. (A handful of documents — 8 total — carry stray labels
SURPRISE/ANGER/SADNESS/FEAR/JOY from an earlier study version.)

Per participant: median 55 answered mood EMAs, IQR 9–141, max 200; **8 participants
have zero**. Median 0.44 answered EMAs per calendar day.

Prompt times are clustered, not uniform — 10:00 (626), 15:00 (620), 20:00 (572),
11:00 (423) — with essentially nothing between 01:00 and 09:00.

---

## 6. Surveys

`surveys.bson` holds raw item responses; the five CSVs hold the scored versions.

| instrument | docs | participants |
|---|---:|---:|
| `bfpt` (IPIP personality) | 52 | 52 |
| `breq` (motivation) | 99 | 54 |
| `dq` (demographics) | 66 | 66 |
| `panas` | 302 | 53 |
| `stai` | 314 | 55 |
| `ttmspbf` (stage of change) | 102 | 55 |

These are baseline/periodic person-level covariates. Useful for conditioning or
stratification; irrelevant to the 1–4 hour transition dynamics.
