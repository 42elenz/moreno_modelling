# Data schema and codebook

Four tables. One row per person in `persons.csv`; everything repeatable lives in
its own table so that evidence is not crushed into one wide CSV.

| file | grain |
|---|---|
| `persons.csv` | one row per person |
| `cohort_membership.csv` | one row per person × cohort × year |
| `sources.csv` | one row per source document |
| `exposure_evidence.csv` | one row per exposure claim, each linked to a source |

Country codes are ISO 3166-1 alpha-2. Dates are ISO `YYYY-MM-DD`; partial dates
are `YYYY` or `YYYY-MM` and are flagged, never silently completed.

---

## persons.csv

| column | type | allowed values / notes |
|---|---|---|
| `person_id` | str | stable key, `{ISO2}-{5 digits}` |
| `name` | str | as rendered in the primary source |
| `name_native` | str | native-script name where applicable |
| `sex` | enum | `M`, `F`, `unknown` |
| `birth_year` | int | blank if unknown |
| `citizenship` | str | ISO2, `;`-separated if multiple |
| `residence_country` | str | ISO2 |
| `birth_country` | str | ISO2 |
| `elite_affiliation_country` | enum | `RU`,`PL`,`CZ`,`HU`,`RO`,`KZ`,`ambiguous` — protocol §6 |
| `elite_category` | enum | `corporate`,`wealth`,`political_state` |
| `occupation` | str | free text |
| `industry` | str | NACE-like top level |
| `organization` | str | |
| `organization_state_owned` | enum | `1`,`0`,`unknown` |
| `role` | enum | `ceo`,`exec_board`,`board_chair`,`controlling_owner`,`minister`,`dep_minister`,`governor`,`senior_official`,`judge_prosecutor`,`soe_director`,`other` |
| `cohort_entry_date` | date | protocol §5.5 |
| `cohort_exit_date` | date | protocol §5.5 |
| `exit_reason` | enum | `death`,`left_role`,`admin_censor_2025_12_31` |
| `vital_status_2025_12_31` | enum | `alive`,`dead`,`unknown` |
| `death_date` | date | blank if alive |
| `death_country` | str | ISO2 |
| `death_city` | str | |
| `death_abroad` | enum | `1`,`0`,`na` — derived: `death_country != elite_affiliation_country` |
| `death_setting` | enum | `home`,`hotel`,`hospital`,`office`,`other`,`unknown` |
| `death_mechanism` | enum | `fall`,`gunshot`,`illness`,`vehicle`,`poisoning`,`other`,`unknown` |
| `window_fall` | enum | `1`,`0`,`unknown` |
| `balcony_fall` | enum | `1`,`0`,`unknown` |
| `building_fall` | enum | `1`,`0`,`unknown` |
| `high_place_fall` | enum | `1`,`0`,`unknown` |
| `official_manner_of_death` | enum | `accident`,`suicide`,`homicide`,`undetermined`,`investigation_ongoing`,`unclear_na` |
| `investigation_status` | enum | `none`,`open`,`closed`,`unknown` |
| `media_description` | str | attributed free text |
| `evidence_of_foul_play` | str | what was documented, not what was insinuated |
| *(14 exposure flags)* | enum | `1`,`0`,`unknown` — see protocol §11; every `1` needs an `exposure_evidence` row |
| `source_1`,`source_2`,`source_3` | str | `source_id` references |
| `coding_confidence` | enum | `high`,`medium`,`low` |
| `double_coded` | enum | `1`,`0` |
| `notes` | str | |

### Derivation rules, applied by the validator, never by hand

- `window_fall = 1` or `balcony_fall = 1` ⟹ `building_fall = 1` ⟹ `high_place_fall = 1`
- `window_balcony_fatal_fall = window_fall OR balcony_fall` (protocol §9, primary)
- `death_abroad` is derived, never entered
- a fatal-fall case requires ≥2 distinct `source_id`s

---

## cohort_membership.csv

`person_id` · `cohort` (`A_corporate`/`B_wealth`/`C_political`) · `year` ·
`country` · `organization` · `role` · `ranking_source_id` · `rank_of_organization`

Annual rows, so that person-time is computed from membership rather than from a
static list of survivors (protocol §5.5, §14).

## sources.csv

`source_id` · `url` · `publication_date` · `accessed_date` · `outlet` ·
`language` · `source_type` (`official`/`company`/`national_news`/
`international_news`/`obituary`/`encyclopaedia`/`other`) · `country_of_outlet` ·
`title` · `archived_url`

## exposure_evidence.csv

`evidence_id` · `person_id` · `exposure_variable` · `value` · `event_date` ·
`source_id` · `published_after_death` (`1`/`0`) · `excerpt` · `confidence`

`published_after_death = 1` rows are excluded from the main analysis (protocol §11).
