# Study protocol — Fatal falls from windows, balconies and buildings among national elites

**Version 1.0 — frozen 2026-09-11, before any cohort member was enumerated and
before any death was identified.**

Any change after this date must be recorded in `AMENDMENTS.md` with a date and a
reason, and analyses run under an amended protocol must be labelled as such.

---

## 1. Objective

Estimate whether people affiliated with the Russian political and business elite
experience fatal falls from windows, balconies or buildings more frequently than
comparable elites from other countries, and whether such events are associated
with documented pre-existing political, legal or state-business conflict.

The study is designed so that it **can return a null result**. The workflow is

> define population → observe deaths → classify mechanism → test exposure

and explicitly **not**

> search for suspicious deaths → count interesting examples.

### What this study does not do

It does not adjudicate whether any individual death was a homicide. Official
manner of death and journalistic characterisation are recorded as separate
variables and are never reconciled into a single investigator judgement.

---

## 2. Hypotheses

| ID | Statement | Primary estimand |
|---|---|---|
| **H1** | Russian-affiliated elites have a higher incidence of fatal window/balcony/building falls than comparable elites from control countries. | IRR, Russia vs pooled PL+CZ+HU+RO |
| **H2** | The incidence among Russian-affiliated elites increased after 2022-02-24. | IRR, post vs pre period within Russia |
| **H3** | Among Russian elites, fatal-fall cases have a higher prevalence of documented pre-existing political/legal/state-business conflict than matched Russian controls. | Conditional OR, 1:4 matched |
| **H4** | Any Russian excess persists when restricted to deaths occurring outside Russia. | IRR, deaths-abroad only |
| **H5** | Russia does not show a general-population excess in building-fall deaths large enough to explain any elite-specific excess. | Age/sex-specific population rates, ICD-10 W13/X80/Y01/Y30 |

H1 is primary. H2–H5 are secondary and are not corrected for multiplicity; they
are interpreted as supporting or undermining the H1 result, not as independent
confirmatory tests.

---

## 3. Study period

**2010-01-01 to 2025-12-31.**

The window deliberately begins twelve years before the phenomenon became
newsworthy. Starting in 2022 would select the observation window on the outcome.

Prespecified breakpoint for H2: **2022-02-24**.
- Pre-period: 2010-01-01 – 2022-02-23
- Post-period: 2022-02-24 – 2025-12-31

The breakpoint is a geopolitical marker, not a claimed cause. No language in any
output may describe February 2022 as *causing* a change in incidence.

---

## 4. Countries

| Role | Countries | Rationale |
|---|---|---|
| Exposure | Russia | — |
| Primary comparator (pooled **and** separate) | Poland, Czech Republic, Hungary, Romania | post-socialist housing stock, broadly comparable historical/economic background, adequate pooled denominator, materially different contemporary political environment |
| Secondary comparator (never pooled with the above) | Kazakhstan | shares Soviet institutional history, Soviet-era housing, resource-intensive industry, state-connected business elite, post-Soviet political structure |

**Ukraine is excluded as a comparator for any period after 2022-02-24**: the war
creates severe mortality and ascertainment confounding.

Western European comparators may be added later **only** as a labelled
sensitivity analysis.

---

## 5. Cohorts — defined before any death is sought

This is the single most important rule in the protocol. Cohort membership is
determined by role and year, never by how a person died or whether their death
was reported internationally.

### Cohort A — corporate elite (primary)

For each country and each year 2010–2025:

- Take the **largest 100 companies by revenue** from a pre-registered national
  ranking (§5.4).
- Eligible roles: CEO or national equivalent; member of the executive board or
  management board; chair of the board of directors/supervisory board;
  controlling owner or founder where they hold an operational role.
- The threshold is **identical across all countries**. No country may use top-50
  while another uses top-100.

De-duplicate individuals appearing in several years or several companies; retain
every (person, year, company, role) tuple in `cohort_membership.csv`.

### Cohort B — wealth elite (sensitivity)

National rich lists, where a reproducible one exists for the year: Forbes Russia
Top 200, Forbes Poland Top 100, and the comparable Czech, Hungarian and Romanian
lists. Less comprehensive than Cohort A but highly reproducible. **Analysed
separately, never merged into Cohort A.**

### Cohort C — political/state elite (optional)

National ministers; deputy ministers; regional governors; senior federal/state
officials; senior judges and prosecutors; directors of major state-owned
enterprises. Eligibility rules must be written down per country before
enumeration. Miscellaneous famous people are **not** admissible.

### 5.4 Ranking sources — registered in advance

The ranking source for each country-year is fixed before enumeration and
recorded in `sources.csv`. Where a year is unavailable, the gap is reported as
missing; it is never filled by substituting a different ranking methodology.

### 5.5 Entry, exit and person-time

- **Cohort entry**: the later of 2010-01-01 and the date of first eligible
  appointment.
- **Cohort exit**: the earliest of death, 2025-12-31, and — under the *in-role*
  definition only — the date of leaving the last eligible role.

Two person-time definitions are computed and reported side by side:

| Definition | At risk | Interpretation |
|---|---|---|
| `in_role` | only while holding an eligible position | exposure is current elite status |
| `ever_elite` | from first eligible appointment to death or 2025-12-31 | exposure is having been an elite |

`ever_elite` is **primary**, because several widely discussed cases involve
former executives. Both are reported for every estimate.

Annual cohort membership is retained rather than a single static list, so that
survivors are not preferentially enumerated (§12.4).

---

## 6. Country affiliation

Affiliation is not citizenship and not place of death.

> **Rule.** Assign a person to country X if their principal senior political,
> governmental or business role during the five years preceding cohort entry (or
> preceding death, whichever is relevant) was located within, controlled from, or
> directly associated with country X.

A Russian-affiliated executive who dies falling from a hotel window in India
remains a Russian-affiliated case with `death_country = IN`.

Recorded as **separate** fields: `citizenship`, `residence_country`,
`birth_country`, `elite_affiliation_country`, `death_country`.

Genuinely ambiguous cases are coded `ambiguous` and are excluded from the primary
analysis and included in a sensitivity analysis. They are never forced.

---

## 7. Ascertainment of deaths

For every cohort member, determine vital status at 2025-12-31; if deceased,
determine date, place and reported circumstances.

Sources, in descending order of preference: company filings and press releases;
government, police or prosecutor statements; national news archives in the
national language; obituaries; GDELT; Media Cloud; Wikipedia/Wikidata **as
discovery tools only, never as sole evidence**; general web search.

**Ascertainment is person-driven.** Each cohort member is searched individually
by name. Case-finding by searching for the outcome is prohibited (§8.3).

Evidentiary threshold for a fatal-fall case:

- **required**: at least two independent sources;
- **preferred**: at least one primary, local or official source.

Every source is recorded in `sources.csv` with its URL and publication date.

---

## 8. Search strategy

### 8.1 Concept groups

Neutral mechanism terms only:

| Group | English seed terms |
|---|---|
| Window | fell from a window, fell out of a window, fell through a window, window fall |
| Balcony | fell from balcony, fell off balcony, balcony fall |
| Building / height | fell from building, fell from roof, fell from height, plunged from building, high-rise fall |

### 8.2 Languages

Every term is translated into Russian, Polish, Czech, Hungarian, Romanian and
Kazakh/Russian for Kazakhstan, and the same grid is run for every country. The
generated grid lives in `search/queries.csv`.

### 8.3 Prohibited qualifiers during case ascertainment

The following may **not** appear in any primary ascertainment query:

> Putin critic · opposition · suspicious · oligarch · murdered · Kremlin enemy ·
> mysterious · assassination

These may only be used *after* a person has already been identified as a case or
control through the person-driven route, when coding §10 exposures.

---

## 9. Outcome definitions

Three nested outcomes, kept **separate**. They are never combined for the
primary analysis.

| Variable | Includes |
|---|---|
| `window_balcony_fatal_fall` **(primary)** | fall from a window, through a window, or from a balcony |
| `building_fatal_fall` | the above plus roof, building, or comparable elevated structural location |
| `high_place_fatal_fall` | the above plus cliffs, scaffolding, bridges, other clearly elevated structures |

A death counts only if the fall is the reported mechanism of death, whatever the
manner.

---

## 10. Circumstances and official classification

Recorded separately and never reconciled:

- `official_manner_of_death` ∈ {accident, suicide, homicide, undetermined,
  investigation ongoing, unclear/not available}
- `media_description` — free text, attributed
- `evidence_of_foul_play` — what was documented, with source

> An official finding of suicide is **not** recoded as homicide because a
> journalist called the circumstances suspicious.

---

## 11. Political / state exposure variables

Coded as documented events occurring **before** the death date. Labels such as
"pro-Putin" or "anti-Putin" are prohibited.

`public_government_criticism` · `public_antiwar_statement` ·
`formal_opposition_activity` · `recent_criminal_investigation` ·
`corruption_investigation` · `recent_interrogation_or_arrest` ·
`recent_dismissal_or_demotion` · `state_owned_company_role` ·
`major_state_contracts` · `sanctions_exposure` ·
`major_asset_or_ownership_dispute` · `documented_conflict_with_state` ·
`documented_conflict_with_employer` · `documented_conflict_with_business_partner`

Every positive code requires a row in `exposure_evidence.csv` carrying source
URL, publication date, a short evidence excerpt or paraphrase, and a confidence
level.

**Only information published before the death date counts in the main analysis.**
Information first appearing after the death is stored with
`published_after_death = 1` and analysed separately.

Political views are never inferred from nationality, occupation, social group or
vague association.

---

## 12. Analysis

### 12.1 Primary — international incidence comparison (H1)

Person-years per country per §5.5. Incidence per 10,000 elite person-years.
Report events, person-years, incidence, IRR and 95% CI.

**Exact methods are primary**: Garwood limits for single rates; exact
conditional (binomial) limits for the IRR. Mid-p limits are reported alongside.
Asymptotic normal approximations must not carry the primary inference.

Comparisons: Russia vs pooled PL+CZ+HU+RO (primary); Russia vs each separately;
Russia vs Kazakhstan.

### 12.2 Before/after 2022 (H2)

Incidence in each period with exact IRR. Yearly events and person-years plotted
with 2022-02-24 marked.

### 12.3 Within-Russia matched case-control (H3)

For every Russian fatal-fall case, select ~4 Russian elite controls alive at the
case's death date, matched on sex; age ±10 years; calendar year; occupational
category; industry; seniority; state-owned vs private sector.

Analysis: conditional logistic regression if numbers permit; otherwise exact
conditional inference on the matched sets, reporting the conditional MLE odds
ratio with exact limits. With sparse data, effect sizes and intervals are
emphasised over p-values.

This comparison is internal to Russia and therefore automatically controls for
country-level ascertainment, architecture and reporting factors. **Given §21 it
is expected to be the most defensible analysis in the study.**

### 12.4 Deaths abroad (H4)

`death_abroad = death_country != elite_affiliation_country`. Report events split
by domestic and foreign death; repeat the primary analysis restricted to deaths
abroad, as a quasi-negative-control for architecture explanations.

### 12.5 Population background (H5)

WHO Mortality Database, ICD-10 **W13** (fall from/out of/through building or
structure), **X80** (intentional self-harm by jumping from a high place),
**Y01** (assault by pushing from a high place), **Y30** (fall/jump/push from a
high place, undetermined intent).

Restricted by sex and to age bands approximating the elite cohort;
age-standardised or age-specific; multi-year averages. Missing years are
reported prominently and **never interpolated and presented as observations**.
WHO data is never used to classify an individual elite death.

### 12.6 Architecture (secondary, explanatory)

Comparable national information on minimum window sill height, guardrail
requirements, high-rise housing prevalence, apartment-living prevalence,
building age, and window restrictor requirements. The question is only whether
Russia's built environment is unusual enough to plausibly account for an excess.
§12.5 is the stronger test of that explanation.

---

## 13. Prespecified sensitivity analyses

1. window only · 2. window + balcony · 3. all building falls · 4. high-confidence
cases only · 5. ≥2 independent sources only · 6. corporate elite only ·
7. wealth-list elite only · 8. excluding official accidents · 9. excluding
official suicides · 10. all official classifications · 11. domestic deaths only ·
12. deaths abroad only · 13. pre-2022 · 14. post-2022 · 15. `in_role` vs
`ever_elite` person-time · 16. excluding ambiguous affiliation.

Presented as a forest plot (Figure 8). A finding that survives only one
definition is reported as such.

---

## 14. Bias register

| Bias | Mechanism | Countermeasure | Residual risk |
|---|---|---|---|
| **Media ascertainment** | Russian window deaths receive disproportionate international coverage | person-driven ascertainment from a pre-defined cohort; never outcome-driven search | **High — see §21 of the feasibility report** |
| **Language** | English-language search under-detects control-country deaths | native-language search grid, mechanically equivalent across countries | High if the search engine is not locale-aware |
| **Fame** | prominent individuals generate more articles | count people, not articles; vital status sought for every cohort member regardless of prominence | Moderate |
| **Political confirmation** | coders already know the famous Russian cases | structured coding rules; blind the §11 exposure coding to outcome status where possible | Moderate |
| **Survival / selection** | a static "current elite" list omits people who already died | annual cohort membership and person-years | Low once §5.5 is implemented |
| **Differential death registration** | manner-of-death classification practice differs by country | official manner recorded but never used to define the outcome; mechanism alone defines the outcome | Moderate |

---

## 15. Quality control

All potential fall cases receive manual review. Double coding for: fall
mechanism; country affiliation; official manner of death; §11 exposure
variables. Inter-rater agreement (Cohen's κ) reported for each.

Confidence levels `high` / `medium` / `low` on every case. Main analyses repeated
using high-confidence cases only. Ambiguity is retained, never forced.

---

## 16. Stop/go checkpoint

**After Phase 5 (manual verification of candidate falls) and before any §11
exposure coding**, report: cohort members by country; total person-years; deaths;
window deaths; balcony deaths; broader building falls; data completeness.

Decision rule, fixed in advance:

| Observed | Action |
|---|---|
| Russia ≈ 2 eligible events, controls ≈ 1 | **Underpowered.** The study becomes primarily descriptive. No formal comparative inference. |
| Russia ≈ 10–20+ events, controls several across a large denominator | Proceed to formal comparative analysis. |

Statistical complexity is not manufactured when the event count is tiny.

---

## 17. Interpretation rules, fixed in advance

| Result | Permitted conclusion |
|---|---|
| Russia ≫ controls | An unusual epidemiological pattern exists in the defined Russian elite. **This does not demonstrate homicide or state involvement.** |
| Russia ≈ controls | The popular "Russian window" phenomenon is substantially attributable to selective media attention and denominator neglect. |
| Russia > controls, political criticism not enriched | If real, the phenomenon relates more to elite/state/business vulnerability than to explicit political opposition. |
| Russia > controls, cases enriched for investigations, ownership disputes, state-company exposure, dismissals — but not for anti-government criticism | Report exactly that. It must **not** be rewritten into a "Putin critics" narrative. |

The primary analytical unit is the **person**. The exposure is their **elite-system
affiliation**. The country in which they physically died is a **separate
variable**.

---

## 18. Outputs

Tables 1–4 and Figures 1–8 as enumerated in the brief, plus a supplement
containing every search query, inclusion decision, excluded candidate case,
evidence source and coding rule.
