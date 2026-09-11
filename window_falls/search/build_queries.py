# -*- coding: utf-8 -*-
"""Generate the mechanically-equivalent multilingual search grid (protocol §8).

Two query families:

  person_probe  - the ONLY family admissible for primary ascertainment.
                  Runs per cohort member, by name, to establish vital status.
  mechanism     - neutral mechanism terms, used to confirm circumstances for a
                  person ALREADY identified through the person-driven route.

The prohibited-qualifier blocklist (protocol §8.3) is enforced here rather than
left to discipline: `check_query` rejects any string containing a political
qualifier, and the generator asserts its own output is clean.
"""
from __future__ import annotations

import itertools
import re
import unicodedata
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent

# Terms are grouped by concept so the same grid can be run for every country.
# Kazakhstan is searched in BOTH Kazakh and Russian, because its business press
# is substantially Russian-language; that is a property of the country, not a
# relaxation of the protocol.
TERMS: dict[str, dict[str, list[str]]] = {
    "ru": {
        "window":   ["выпал из окна", "выпала из окна", "выпал через окно", "падение из окна"],
        "balcony":  ["выпал с балкона", "упал с балкона", "падение с балкона"],
        "building": ["упал с крыши", "выпал из здания", "упал с высоты", "разбился упав с здания"],
    },
    "pl": {
        "window":   ["wypadł z okna", "wypadła z okna", "wypadł przez okno", "upadek z okna"],
        "balcony":  ["wypadł z balkonu", "spadł z balkonu", "upadek z balkonu"],
        "building": ["spadł z dachu", "runął z budynku", "spadł z wysokości", "wypadł z wieżowca"],
    },
    "cs": {
        "window":   ["vypadl z okna", "vypadla z okna", "vypadl oknem", "pád z okna"],
        "balcony":  ["vypadl z balkonu", "spadl z balkonu", "pád z balkonu"],
        "building": ["spadl ze střechy", "zřítil se z budovy", "pád z výšky", "spadl z výškové budovy"],
    },
    "hu": {
        "window":   ["kizuhant az ablakon", "kiesett az ablakon", "ablakból zuhant ki"],
        "balcony":  ["leesett az erkélyről", "lezuhant az erkélyről", "erkélyről zuhant"],
        "building": ["lezuhant a tetőről", "magasból zuhant", "épületről zuhant le", "kizuhant az emeletről"],
    },
    "ro": {
        "window":   ["a căzut de la fereastră", "a căzut pe fereastră", "cădere de la fereastră"],
        "balcony":  ["a căzut de pe balcon", "cădere de pe balcon"],
        "building": ["a căzut de pe acoperiș", "a căzut de la înălțime", "a căzut de la etaj", "s-a prăbușit de pe clădire"],
    },
    "kk": {
        "window":   ["терезеден құлап түсті", "терезеден құлады"],
        "balcony":  ["балконнан құлады", "балконнан құлап түсті"],
        "building": ["шатырдан құлады", "биіктен құлап түсті", "ғимараттан құлады"],
    },
    "en": {
        "window":   ["fell from a window", "fell out of a window", "fell through a window", "window fall"],
        "balcony":  ["fell from balcony", "fell off balcony", "balcony fall"],
        "building": ["fell from building", "fell from roof", "fell from height", "plunged from building", "high-rise fall"],
    },
}

# Every country is searched in its own language AND English, identically.
COUNTRY_LANGS = {
    "RU": ["ru", "en"], "PL": ["pl", "en"], "CZ": ["cs", "en"],
    "HU": ["hu", "en"], "RO": ["ro", "en"], "KZ": ["kk", "ru", "en"],
}

# Neutral vital-status probes. No mechanism, no politics.
PERSON_PROBE = {
    "ru": ["{name} умер", "{name} скончался", "{name} некролог", "{name} биография"],
    "pl": ["{name} zmarł", "{name} nie żyje", "{name} nekrolog", "{name} biografia"],
    "cs": ["{name} zemřel", "{name} úmrtí", "{name} nekrolog", "{name} životopis"],
    "hu": ["{name} meghalt", "{name} elhunyt", "{name} gyászjelentés", "{name} életrajz"],
    "ro": ["{name} a murit", "{name} deces", "{name} necrolog", "{name} biografie"],
    "kk": ["{name} қайтыс болды", "{name} өмірбаяны"],
    "en": ["{name} died", "{name} obituary", "{name} death", "{name} biography"],
}

# Protocol §8.3 - inadmissible during primary ascertainment, any language.
PROHIBITED = [
    "putin", "critic", "opposition", "suspicious", "oligarch", "murder", "kremlin",
    "mysterious", "assassinat", "enemy", "regime", "dissident",
    "критик", "оппозиц", "олигарх", "подозрительн", "убийств", "кремл", "загадочн",
    "krytyk", "opozycj", "oligarch", "podejrzan", "zabójstw", "tajemnicz",
    "kritik", "opozic", "oligarch", "podezřel", "vražd", "záhadn",
    "kritikus", "ellenzék", "oligarcha", "gyanús", "gyilkos", "rejtélyes",
    "critic", "opoziți", "oligarh", "suspect", "crimă", "misterio",
    "сын", "оппозиция", "олигарх", "күдікті", "кісі өлтіру",
]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold()


def check_query(q: str) -> list[str]:
    """Return the prohibited qualifiers present in a query. Empty means clean."""
    n = _norm(q)
    return sorted({p for p in PROHIBITED if _norm(p) in n})


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    mech = []
    for country, langs in COUNTRY_LANGS.items():
        for lang in langs:
            for concept, terms in TERMS[lang].items():
                for t in terms:
                    mech.append({"family": "mechanism", "country": country, "language": lang,
                                 "concept": concept, "query": t})
    mech_df = pd.DataFrame(mech)

    probe = []
    for country, langs in COUNTRY_LANGS.items():
        for lang in langs:
            for tpl in PERSON_PROBE[lang]:
                probe.append({"family": "person_probe", "country": country,
                              "language": lang, "concept": "vital_status", "query": tpl})
    probe_df = pd.DataFrame(probe)

    all_q = pd.concat([probe_df, mech_df], ignore_index=True)
    dirty = {q: bad for q in all_q["query"] if (bad := check_query(q.replace("{name}", "")))}
    assert not dirty, f"generated queries contain prohibited qualifiers: {dirty}"
    return all_q, mech_df


def coverage_report(mech_df: pd.DataFrame) -> pd.DataFrame:
    """Every country must get the same concepts - that is the whole point."""
    piv = (mech_df.pivot_table(index="country", columns="concept", values="query",
                               aggfunc="count", fill_value=0))
    piv["total"] = piv.sum(axis=1)
    return piv.reset_index()


if __name__ == "__main__":
    all_q, mech_df = build()
    all_q.to_csv(OUT / "queries.csv", index=False)
    cov = coverage_report(mech_df)
    cov.to_csv(OUT / "query_coverage.csv", index=False)

    print(f"{len(all_q)} queries written to {OUT/'queries.csv'}")
    print(f"  person_probe (primary ascertainment): {(all_q.family=='person_probe').sum()}")
    print(f"  mechanism (confirmation only):        {(all_q.family=='mechanism').sum()}")
    print("\nMechanism-term coverage per country (must be non-zero in every concept):")
    print(cov.to_string(index=False))
    assert (cov[["balcony", "building", "window"]] > 0).all().all(), "a country lacks a concept"

    # Demonstrate the guard actually bites.
    for bad in ["Putin critic fell from window", "podejrzana śmierć oligarchy",
                "выпал из окна загадочная смерть"]:
        print(f"\nblocked: {bad!r} -> {check_query(bad)}")
        assert check_query(bad), "blocklist failed to fire"
    print("\nprohibited-qualifier guard verified")
