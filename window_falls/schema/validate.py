"""Schema validator. Enforces the protocol's derivation and evidence rules.

Run it against a populated data directory before any analysis:
    python3 window_falls/schema/validate.py <data_dir>

It refuses to "fix" anything. Every violation is reported and the exit code is
non-zero, because a silently repaired dataset is how coding decisions stop being
traceable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXPOSURES = [
    "public_government_criticism", "public_antiwar_statement", "formal_opposition_activity",
    "recent_criminal_investigation", "corruption_investigation", "recent_interrogation_or_arrest",
    "recent_dismissal", "state_company_role", "major_state_contract", "sanctions_exposure",
    "asset_dispute", "state_conflict", "business_conflict", "employer_conflict",
]
COUNTRIES = {"RU", "PL", "CZ", "HU", "RO", "KZ", "ambiguous"}
MANNER = {"accident", "suicide", "homicide", "undetermined", "investigation_ongoing", "unclear_na"}
TRI = {"1", "0", "unknown"}
STUDY_START, STUDY_END = pd.Timestamp("2010-01-01"), pd.Timestamp("2025-12-31")


def _d(s):
    return pd.to_datetime(s, errors="coerce")


def validate(data_dir: Path) -> list[str]:
    err: list[str] = []
    P = pd.read_csv(data_dir / "persons.csv", dtype=str).fillna("")
    S = pd.read_csv(data_dir / "sources.csv", dtype=str).fillna("")
    E = pd.read_csv(data_dir / "exposure_evidence.csv", dtype=str).fillna("")
    M = pd.read_csv(data_dir / "cohort_membership.csv", dtype=str).fillna("")

    def bad(mask, msg):
        for pid in P.loc[mask, "person_id"]:
            err.append(f"{pid}: {msg}")

    if P.empty:
        return ["persons.csv is empty - nothing to validate"]

    # identity
    dup = P["person_id"][P["person_id"].duplicated()].tolist()
    err += [f"duplicate person_id: {d}" for d in dup]

    # controlled vocabularies
    bad(~P["elite_affiliation_country"].isin(COUNTRIES), "invalid elite_affiliation_country")
    bad(~P["official_manner_of_death"].isin(MANNER | {""}), "invalid official_manner_of_death")
    for c in ["window_fall", "balcony_fall", "building_fall", "high_place_fall"]:
        bad(~P[c].isin(TRI | {""}), f"invalid {c}")

    # outcome hierarchy (protocol §9)
    wb = (P["window_fall"] == "1") | (P["balcony_fall"] == "1")
    bad(wb & (P["building_fall"] != "1"), "window/balcony fall must imply building_fall=1")
    bad((P["building_fall"] == "1") & (P["high_place_fall"] != "1"),
        "building_fall must imply high_place_fall=1")
    bad((P["high_place_fall"] == "1") & (P["death_mechanism"] != "fall"),
        "fall outcome set but death_mechanism is not 'fall'")

    # death_abroad is derived, never entered
    dead = P["vital_status_2025_12_31"] == "dead"
    want = (P["death_country"] != P["elite_affiliation_country"]).map({True: "1", False: "0"})
    bad(dead & (P["death_country"] != "") & (P["death_abroad"] != want),
        "death_abroad inconsistent with death_country vs elite_affiliation_country")

    # vital status vs dates
    bad(dead & (P["death_date"] == ""), "dead but no death_date")
    bad((P["vital_status_2025_12_31"] == "alive") & (P["death_date"] != ""),
        "alive but death_date present")
    bad(dead & (P["exit_reason"] != "death"), "dead but exit_reason is not 'death'")

    entry, exit_, dod = _d(P["cohort_entry_date"]), _d(P["cohort_exit_date"]), _d(P["death_date"])
    bad(entry.notna() & exit_.notna() & (exit_ < entry), "cohort_exit_date before cohort_entry_date")
    bad(entry.notna() & (entry < STUDY_START), "cohort_entry_date before study start")
    bad(exit_.notna() & (exit_ > STUDY_END), "cohort_exit_date after study end")
    bad(dod.notna() & exit_.notna() & (dod != exit_) & (P["exit_reason"] == "death"),
        "exit_reason='death' but cohort_exit_date != death_date")
    bad(dod.notna() & ((dod < STUDY_START) | (dod > STUDY_END)), "death_date outside study period")

    # evidence thresholds (protocol §7)
    fall = P["high_place_fall"] == "1"
    n_src = P[["source_1", "source_2", "source_3"]].ne("").sum(axis=1)
    bad(fall & (n_src < 2), "fatal-fall case with fewer than 2 sources")
    known = set(S["source_id"])
    for col in ["source_1", "source_2", "source_3"]:
        bad(P[col].ne("") & ~P[col].isin(known), f"{col} not present in sources.csv")

    # every positive exposure needs evidence (protocol §11)
    ev_keys = set(zip(E["person_id"], E["exposure_variable"]))
    for x in EXPOSURES:
        if x not in P.columns:
            err.append(f"persons.csv missing exposure column {x}")
            continue
        bad((P[x] == "1") & ~P.apply(lambda r: (r["person_id"], x) in ev_keys, axis=1),
            f"{x}=1 without a row in exposure_evidence.csv")

    # evidence integrity
    if not E.empty:
        orphan = E.loc[~E["person_id"].isin(P["person_id"]), "evidence_id"]
        err += [f"evidence {e}: unknown person_id" for e in orphan]
        orphan = E.loc[~E["source_id"].isin(known), "evidence_id"]
        err += [f"evidence {e}: unknown source_id" for e in orphan]
        bad_x = E.loc[~E["exposure_variable"].isin(EXPOSURES), "evidence_id"]
        err += [f"evidence {e}: unknown exposure_variable" for e in bad_x]
        # exposure must predate death unless explicitly flagged
        ed = _d(E["event_date"])
        dmap = dict(zip(P["person_id"], dod))
        for i, r in E.iterrows():
            d = dmap.get(r["person_id"])
            if pd.notna(ed[i]) and pd.notna(d) and ed[i] > d and r["published_after_death"] != "1":
                err.append(f"evidence {r['evidence_id']}: event_date after death "
                           f"but published_after_death != 1")

    # sources
    if not S.empty:
        err += [f"source {s}: missing publication_date"
                for s in S.loc[S["publication_date"] == "", "source_id"]]
        err += [f"duplicate source_id: {s}" for s in S["source_id"][S["source_id"].duplicated()]]

    # annual membership must exist for every cohort member (protocol §5.5)
    if not M.empty:
        missing = set(P["person_id"]) - set(M["person_id"])
        err += [f"{p}: no cohort_membership rows" for p in sorted(missing)]
        yrs = pd.to_numeric(M["year"], errors="coerce")
        err += [f"cohort_membership: year out of range ({int(y)})"
                for y in yrs[(yrs < 2010) | (yrs > 2025)].dropna().unique()]

    return err


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
    errs = validate(d)
    if not errs:
        print(f"OK - {d} passes every schema and protocol rule")
        return 0
    print(f"{len(errs)} violation(s) in {d}:")
    for e in errs[:200]:
        print("  -", e)
    if len(errs) > 200:
        print(f"  ... and {len(errs)-200} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
