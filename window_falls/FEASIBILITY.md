# Feasibility report — can this study be run here?

**No, not past Phase 1. Every external data source the protocol depends on is
blocked by this environment's egress policy: Wikidata, Wikipedia, the WHO
Mortality Database, Reuters, BBC, CNBC — all of them. The only working research
tool is a US-locale web search that returns snippets, and a measurement reported
below shows that using it for case-finding would manufacture the study's own
conclusion. So: the protocol is written and frozen, the analysis machinery is
built and tested, and the power analysis is done — and it changes which
hypothesis should be primary. No cohort was enumerated and no death was
classified, because doing either with the tools available would produce a
number that looks like evidence and is not.**

---

## 1. What was delivered

| Phase | Status |
|---|---|
| 1 — write and freeze the protocol | **Done.** `protocol/PROTOCOL.md`, frozen 2026-09-11 |
| 2 — construct elite cohorts | **Blocked.** No access to any company ranking, rich list or registry |
| 3 — vital status and deaths | **Blocked.** No access to filings, archives, obituaries, Wikidata |
| 4–5 — classify and verify fall mechanisms | **Blocked.** Depends on 2–3 |
| 6–7 — exposure coding, matched controls | **Blocked.** Depends on 2–5 |
| 8–9 — person-years, IRR, pre/post-2022, abroad | **Machinery built and tested; no data to run it on** |
| 10 — WHO mortality background | **Blocked.** `who.int` and `apps.who.int` refused |
| 11 — architecture evidence | **Blocked.** No access to building-code sources |
| 12 — sensitivity analyses and figures | **Machinery built; two feasibility figures produced** |
| — | **Added: a prespecified power analysis, and a measurement of the ascertainment bias** |

---

## 2. Evidence that the sources are blocked

Every host below returned a proxy-level refusal, not a 404 and not a paywall.
These are organisation egress-policy denials and cannot be retried around.

| host | purpose in the protocol | result |
|---|---|---|
| `query.wikidata.org` | reproducible cohort + vital status (§5, §7) | `403 CONNECT` / `EGRESS_BLOCKED` |
| `en.wikipedia.org` | discovery tool (§7) | `EGRESS_BLOCKED` |
| `www.who.int`, `apps.who.int` | ICD-10 W13/X80/Y01/Y30 rates (§12.5) | `403 CONNECT` / `EGRESS_BLOCKED` |
| `www.reuters.com` | primary source for case verification (§7) | fetch refused |
| `www.bbc.com` | primary source | fetch refused |
| `www.cnbc.com` | primary source | `EGRESS_BLOCKED` |

The proxy's own diagnostic endpoint records the refusals as
`connect_rejected — gateway answered 403 to CONNECT (policy denial)`.

Web **search** works and returns titles, URLs and a synthesised summary. Web
**fetch** does not. That distinction matters: the protocol requires recording a
source URL *and its publication date* and reading enough of the document to code
§10 and §11 variables. Snippets cannot support that, and §7's two-independent-
source rule cannot be verified from them.

---

## 3. Why "just use web search" would produce a false positive

See `search/ASCERTAINMENT_PROBE.md` for the full measurement.

Three politically neutral, native-language queries — Polish, Czech, Hungarian —
each naming that country's own companies and executives, and each containing no
prohibited qualifier under §8.3:

| probe | language | targeted | results | about **Russian**-affiliated people | about a **target-country elite** |
|---|---|---|---:|---:|---:|
| P1 | Polish | Poland | 9 | **7** | **0** |
| P2 | Czech | Czechia | 8 | **4** | 1 (from 1995, outside the window) |
| P3 | Hungarian | Hungary | 9 | **4** | **0** |
| | | total | 26 | **15 (58%)** | **1 (4%)** |

A search asking, in Polish, about Polish executives falling from windows returns
Lukoil, Yukos and Transneft.

This is differential outcome ascertainment acting only on the comparator
countries. It biases the incidence-rate ratio **upward, toward the hypothesis**,
by an unknown and unbounded amount. Case-finding built this way returns
"Russia ≫ controls" whether or not that is true, which is the exact failure the
protocol was designed to exclude.

The protocol's countermeasure — enumerate the cohort first, then check each named
person individually — is the right fix and is not available: it needs per-person
retrieval from hosts that are blocked, for roughly 10,000 people. And a search
that returns nothing for an obscure Romanian board member is not evidence that
they are alive.

---

## 4. The power analysis changes the study design

Run before any data, on stated assumptions. Background rate for fatal building
falls (W13 + X80 + Y30) in men aged ~45–70 taken from published aggregates —
European suicide ~12/100k/yr overall with substantially higher male rates in this
band, jumping from height roughly 3–8% of suicides, plus accidental and
undetermined-intent building falls. Central value **3 per 100,000 person-years**,
deliberately set high for an elite population, which makes every figure below
*optimistic*.

Constructible cohort: top-100 companies × ~6 eligible senior roles × 16 years
≈ **1,950 unique people per country**, ≈ **16,600 ever-elite person-years**.

### H1 — the international comparison is the weakest design in the protocol

| background rate /100k py | expected RU events | expected comparator events (4 pooled) | smallest IRR detectable at 80% power | power at IRR = 3 |
|---:|---:|---:|---:|---:|
| 1 | 0.17 | 0.66 | 23.7 | 5% |
| **3** | **0.50** | **1.99** | **10.5** | **14%** |
| 6 | 1.00 | 3.98 | 6.6 | 26% |
| 10 | 1.66 | 6.64 | 4.8 | 41% |

**Under the null, the entire 16-year study expects one Russian event and two
comparator events.** The design can only detect a roughly **tenfold** difference.
A real threefold excess would be missed 86% of the time.

Scaling up does not rescue it within the protocol's own definition:

| cohort multiple | unique RU elites | smallest detectable IRR | power at IRR = 3 |
|---:|---:|---:|---:|
| ×1 | 1,950 | 10.5 | 14% |
| ×3 | 5,850 | 5.1 | 38% |
| ×8 | 15,600 | 3.1 | 76% |
| ×20 | 39,000 | 2.2 | 98% |

Detecting a threefold excess needs ~15,600 Russian elites and four times that
pooled across the comparators. That is no longer "senior leadership of the top
100 companies"; the comparator countries do not contain that many people who
meet the eligibility rule.

### The one thing H1 *is* powered for

| observed Russian events | expected under background | implied rate ratio |
|---:|---:|---:|
| 3 | 0.50 | 6× |
| 10 | 0.50 | **20×** |
| 20 | 0.50 | **40×** |

The effect size implied by the press narrative — on the order of a dozen named
Russian elite fall deaths since 2022 — is a 20–40× rate ratio, comfortably above
the 10.5× detection threshold. **So the study's difficulty is not statistical
power. It is entirely whether the numerator and the denominator were ascertained
the same way in every country** — and §3 above shows that, here, they cannot be.

### H3 — the within-Russia matched design is the salvageable one

4 controls per case, exact conditional inference:

| Russian cases available | control exposure 10% | 20% | 35% |
|---:|---:|---:|---:|
| 5 | OR 30 | OR 30 | never reaches 80% |
| 10 | OR 10 | OR 10 | OR 10 |
| 20 | OR 7 | OR 5 | OR 5 |
| 30 | OR 5 | OR 4 | OR 4 |
| 50 | OR 4 | OR 3 | OR 3 |

With ~20 cases this detects an odds ratio of 5 with 84% power. More importantly,
**cases and controls are both drawn from the Russian elite**, so the comparison is
immune to the cross-country ascertainment asymmetry measured in §3, and to
architecture, housing stock and death-registration differences. It needs no
comparator country and no WHO data.

At 5 cases the design is not merely weak — with 4:1 matching and 35% control
exposure, *no* odds ratio reaches 80% power, because the exact test cannot
produce a p-value below 0.05 from five matched sets. That is a discreteness
floor, not a sample-size problem, and no amount of modelling removes it.

---

## 5. Recommendation

**Reorder the protocol. Make H3 primary and H1 secondary.**

1. **H3 (within-Russia matched case-control) should be the primary analysis.**
   It is better powered at realistic case counts, it needs a cohort in one country
   rather than six, and it is structurally immune to the bias that makes H1
   untrustworthy. It also addresses the more interesting question: *given* that a
   Russian elite died in a fall, was documented state/legal/business conflict more
   common than among matched peers who did not?

2. **H1 should be reported as a descriptive rate comparison with exact intervals
   and an explicit ascertainment caveat**, not as the headline. If it is run, the
   comparator case-finding must be done by a native-speaker researcher working
   from national archives person-by-person — not by search engine.

3. **H5 (WHO population background) stays worth doing** and is cheap once
   `who.int` is reachable. If Russia shows no general-population excess in
   W13/X80/Y30 while the elite cohort does, that is the strongest available
   argument against the architecture explanation — stronger than §12.6's building
   codes.

4. **Do not run Phases 2–5 with search-engine ascertainment.** The probe in §3 is
   the reason. A case list built that way is not a weak dataset; it is a dataset
   whose bias points at the conclusion.

### What would have to be true instead

| Need | Unblocks |
|---|---|
| `query.wikidata.org` | reproducible, language-neutral cohort + vital status for all six countries — the single highest-value unblock |
| `who.int` / `apps.who.int` | H5 entirely |
| A commercial registry (Orbis, Bureau van Dijk, Refinitiv) | Cohort A properly, with consistent revenue rankings across countries |
| Native-language archive access (PAP, ČTK, MTI, Agerpres, Interfax) | control-country case ascertainment, which is the thing search cannot do |
| A second coder | §15 double coding and κ — currently impossible with one agent |

None of these are research problems. They are access problems, and until they are
solved the honest output of this project is the protocol, the machinery, and the
power analysis that says which hypothesis to spend the effort on.

---

## 6. One thing worth saying plainly about the original question

The power analysis shows the claimed effect is large enough to measure —
20–40× — *if* it is real. That is an unusually favourable situation: most
epidemiological questions fail because the effect is small. This one fails, here,
only because the counting cannot be done evenhandedly. It is a tractable study
for anyone with Wikidata, a registry, and four native-speaker research assistants.
It is not a tractable study for a search engine, and the 58% figure in §3 is what
happens when you try.
