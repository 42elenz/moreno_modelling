# Are fatal falls from windows overrepresented among Russian elites?

An epidemiological study designed so that it **could return a null result**, and
an honest account of how far it can be executed in this environment.

> **Read [`FEASIBILITY.md`](FEASIBILITY.md) first.** Short version: the protocol
> is written and frozen, the statistics are built and tested, and the power
> analysis is done and changes which hypothesis should be primary. No cohort was
> built and no death was classified, because every data source the protocol needs
> is blocked here, and the one available tool — web search — was measured and
> shown to manufacture the study's own conclusion.

| document | what it is |
|---|---|
| [`protocol/PROTOCOL.md`](protocol/PROTOCOL.md) | the frozen prespecified protocol (v1.0, 2026-09-11) |
| [`FEASIBILITY.md`](FEASIBILITY.md) | what can and cannot be run, the power analysis, the recommendation |
| [`search/ASCERTAINMENT_PROBE.md`](search/ASCERTAINMENT_PROBE.md) | a measurement of the media-ascertainment bias |
| [`schema/codebook.md`](schema/codebook.md) | the four-table data schema |
| `figures/` | power and flow figures |
| `outputs/` | power-analysis tables |

## The headline numbers

- Under the null, a realistic 16-year cohort (~1,950 elites per country) expects
  **0.5 Russian** and **2.0 comparator** fatal falls. The international comparison
  can only detect a **~10×** difference.
- The effect implied by the press narrative is **20–40×**, so power is *not* the
  problem — **evenhanded counting is**.
- A neutral, native-language, politics-free query about *Polish* executives
  falling from windows returned **7 of 9 results about Russians and 0 about
  Poles**. Across Polish, Czech and Hungarian: **58% Russian, 4% target-country
  elite**.
- The **within-Russia matched case-control** design detects an odds ratio of 5
  with 84% power at 20 cases, needs one country instead of six, and is immune to
  that bias. It should be the primary analysis.

## Layout

```
protocol/PROTOCOL.md          frozen protocol - countries, period, cohorts,
                              outcomes, exposures, bias register, stop/go rule
schema/                       codebook, empty CSV templates, validator + fixture
analysis/rates.py             exact Poisson CIs, exact conditional IRR, person-time
analysis/matched.py           conditional MLE odds ratio + exact conditional CI
analysis/power.py             the prespecified power analysis
analysis/figures.py           feasibility figures
search/build_queries.py       multilingual query grid + §8.3 blocklist enforcement
search/ASCERTAINMENT_PROBE.md the bias measurement
```

## Reproducing

```bash
pip install pandas numpy scipy matplotlib

python3 window_falls/analysis/rates.py          # self-tests
python3 window_falls/analysis/matched.py        # self-tests
python3 window_falls/search/build_queries.py    # 192 queries + blocklist check
python3 window_falls/analysis/power.py          # ~3 min
python3 window_falls/analysis/figures.py

# once data exists, before any analysis:
python3 window_falls/schema/validate.py <data_dir>
```

## Notes on the statistics

Everything is exact or mid-p. Garwood limits for single rates; exact conditional
(binomial) limits for incidence-rate ratios, which stay valid at zero events;
conditional MLE with exact Poisson-binomial limits for the matched design, which
reduces to McNemar at 1:1 matching. The self-tests check each of those against
published values. Asymptotic normal approximations are never used for inference,
because at these expected counts they are wrong in the direction that makes
findings look stronger.

The validator refuses to repair data. It reports violations and exits non-zero —
a silently corrected dataset is how coding decisions stop being traceable. Its
fixture plants one violation of every rule and checks all of them fire.

## What this study does not do

It does not adjudicate whether any individual death was a homicide. Official
manner of death and journalistic characterisation are stored as separate
variables and never reconciled. A finding of suicide is not recoded because a
journalist called the circumstances suspicious. Political views are never
inferred from nationality, occupation or association.
