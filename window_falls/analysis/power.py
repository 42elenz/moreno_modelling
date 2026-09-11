"""Prespecified power analysis (protocol §25).

This runs BEFORE any cohort is built, on purpose. The protocol's stop/go rule
asks what to do if the eligible event count turns out to be tiny; this module
answers the prior question of whether the count could ever be otherwise, given
cohort sizes that are actually constructible and a background rate taken from
the published literature.

Nothing here uses study data. Every input is a stated assumption.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from functools import lru_cache

from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).parent))
from matched import MatchedSet, exact_test

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- assumptions
# Background rate of fatal falls from a building (ICD-10 W13 + X80 + Y30) in
# men aged roughly 45-70. Derived from published aggregates, NOT from data in
# this repository: European all-cause suicide ~12/100k/yr overall with male
# rates several-fold higher in this age band, jumping from height accounting
# for roughly 3-8% of suicides in Europe, plus accidental (W13) and
# undetermined-intent (Y30) building falls of broadly similar magnitude.
# The central value is deliberately conservative-high for an elite population,
# which biases every power estimate below in the OPTIMISTIC direction.
BACKGROUND_RATES = [1e-5, 3e-5, 6e-5, 1e-4]      # per person-year
CENTRAL_RATE = 3e-5

# Constructible cohort sizes, per country, 2010-2025 (16 years).
# "in_role"   : at risk only while holding an eligible senior position
#               ~100 companies x ~6 eligible roles x 16 years
# "ever_elite": followed from first eligible appointment to 2025-12-31
#               ~1,950 unique people x ~8.5 years mean follow-up
COHORT_PY = {"in_role": 9_600.0, "ever_elite": 16_600.0}
N_COMPARATOR_COUNTRIES = 4                        # PL + CZ + HU + RO pooled

ALPHA = 0.05
# Enumeration range is chosen per call from the Poisson means; a fixed cap
# silently truncates the distribution and makes power fall as the effect grows.
KMAX_FLOOR = 60
ENUMERATION_CAP = 300                             # beyond this, Monte Carlo


# --------------------------------------------------------------- exact power
@lru_cache(maxsize=None)
def _reject(k1: int, T: int, p_null: float, alpha: float = ALPHA) -> bool:
    """Two-sided exact conditional (binomial) test of IRR = 1.

    The decision depends only on (k1, T, p_null), so it is cached; the power
    enumeration revisits the same cells many thousands of times.
    """
    if T == 0:
        return False
    return bool(stats.binomtest(k1, T, p_null).pvalue < alpha)


def _irr_power_mc(lam1: float, lam0: float, p_null: float, alpha: float,
                  n_sim: int = 40_000, seed: int = 0) -> float:
    """Monte-Carlo power, for grids too large to enumerate exactly."""
    rng = np.random.default_rng(seed)
    k1 = rng.poisson(lam1, n_sim)
    T = k1 + rng.poisson(lam0, n_sim)
    return sum(_reject(int(a), int(t), p_null, alpha) for a, t in zip(k1, T)) / n_sim


def irr_power(irr: float, py1: float, py0: float, base_rate: float,
              alpha: float = ALPHA) -> float:
    """Power of the exact conditional test.

    Enumerates the joint Poisson exactly where the grid is small enough. A
    fixed truncation would silently make power FALL as the effect grows, so
    the range is chosen from the Poisson means and asserted to capture the
    whole mass; past ENUMERATION_CAP the cost is O(kmax^2) and Monte Carlo
    takes over.
    """
    lam1, lam0 = irr * base_rate * py1, base_rate * py0
    p_null = py1 / (py1 + py0)
    kmax = int(max(KMAX_FLOOR,
                   np.ceil(lam1 + 12 * np.sqrt(lam1) + 20),
                   np.ceil(lam0 + 12 * np.sqrt(lam0) + 20)))
    if kmax > ENUMERATION_CAP:
        return _irr_power_mc(lam1, lam0, p_null, alpha)
    ks = np.arange(kmax + 1)
    p1 = stats.poisson.pmf(ks, lam1)
    p0 = stats.poisson.pmf(ks, lam0)
    assert p1.sum() > 1 - 1e-9 and p0.sum() > 1 - 1e-9, "enumeration range too small"
    power = 0.0
    for k1 in range(kmax + 1):
        if p1[k1] < 1e-14:
            continue
        for k0 in range(kmax + 1):
            if p0[k0] < 1e-14:
                continue
            if _reject(k1, k1 + k0, p_null, alpha):
                power += p1[k1] * p0[k0]
    return float(power)


def detectable_irr(py1: float, py0: float, base_rate: float,
                   target: float = 0.80) -> float:
    """Smallest IRR reaching `target` power. inf if unreachable below IRR=500."""
    f = lambda x: irr_power(x, py1, py0, base_rate) - target
    if f(500.0) < 0:
        return float("inf")
    if f(1.001) > 0:
        return 1.0
    return float(optimize.brentq(f, 1.001, 500.0, xtol=1e-3))


# ------------------------------------------------- matched design (protocol §12)
def matched_power(n_cases: int, n_controls: int, p_control: float, odds_ratio: float,
                  n_sim: int = 3000, alpha: float = ALPHA, seed: int = 0) -> float:
    """Simulated power of the exact conditional test for a 1:M matched design."""
    rng = np.random.default_rng(seed)
    odds = odds_ratio * p_control / (1 - p_control)
    p_case = odds / (1 + odds)
    case_x = rng.random((n_sim, n_cases)) < p_case
    ctrl_x = rng.binomial(n_controls, p_control, size=(n_sim, n_cases))
    hits = 0
    for i in range(n_sim):
        sets = [MatchedSet(bool(case_x[i, j]), int(ctrl_x[i, j]), n_controls)
                for j in range(n_cases)]
        if exact_test(sets)["p_value"] < alpha:
            hits += 1
    return hits / n_sim


def detectable_or(n_cases: int, n_controls: int, p_control: float,
                  target: float = 0.80, n_sim: int = 1500) -> float:
    """Smallest OR reaching `target` power, on a coarse grid (simulation-limited)."""
    for orr in [1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]:
        if matched_power(n_cases, n_controls, p_control, orr, n_sim=n_sim) >= target:
            return orr
    return float("inf")


# ---------------------------------------------------------------------- main
def main() -> None:
    rows = []
    for defn, py_ru in COHORT_PY.items():
        py_cmp = py_ru * N_COMPARATOR_COUNTRIES
        for rate in BACKGROUND_RATES:
            rows.append({
                "cohort_definition": defn,
                "background_rate_per_100k_py": rate * 1e5,
                "py_russia": py_ru,
                "py_comparator_pooled": py_cmp,
                "expected_events_russia_null": py_ru * rate,
                "expected_events_comparator_null": py_cmp * rate,
                "expected_total_null": (py_ru + py_cmp) * rate,
                "detectable_irr_80pct": detectable_irr(py_ru, py_cmp, rate),
                "power_at_irr_3": irr_power(3, py_ru, py_cmp, rate),
                "power_at_irr_5": irr_power(5, py_ru, py_cmp, rate),
                "power_at_irr_10": irr_power(10, py_ru, py_cmp, rate),
                "power_at_irr_30": irr_power(30, py_ru, py_cmp, rate),
            })
    irr_tab = pd.DataFrame(rows)
    irr_tab.to_csv(OUT / "power_irr.csv", index=False)
    print("== INTERNATIONAL INCIDENCE COMPARISON (H1) ==")
    print(irr_tab.round(3).to_string(index=False))

    # How large would the cohort have to be?
    rows = []
    for mult in [1, 2, 3, 5, 8, 12, 20]:
        py_ru = COHORT_PY["ever_elite"] * mult
        py_cmp = py_ru * N_COMPARATOR_COUNTRIES
        rows.append({
            "cohort_multiple": mult,
            "py_russia": py_ru,
            "unique_people_russia_approx": round(1950 * mult),
            "expected_events_russia_null": py_ru * CENTRAL_RATE,
            "detectable_irr_80pct": detectable_irr(py_ru, py_cmp, CENTRAL_RATE),
            "power_at_irr_3": irr_power(3, py_ru, py_cmp, CENTRAL_RATE),
        })
    size_tab = pd.DataFrame(rows)
    size_tab.to_csv(OUT / "power_cohort_size.csv", index=False)
    print("\n== COHORT SIZE REQUIRED (background rate 3 per 100,000 py) ==")
    print(size_tab.round(3).to_string(index=False))

    # Matched within-Russia design
    rows = []
    for n_cases in [5, 10, 15, 20, 30, 50]:
        for p0 in [0.10, 0.20, 0.35]:
            rows.append({
                "n_cases": n_cases, "controls_per_case": 4,
                "control_exposure_prevalence": p0,
                "detectable_or_80pct": detectable_or(n_cases, 4, p0),
                "power_at_or_3": matched_power(n_cases, 4, p0, 3),
                "power_at_or_5": matched_power(n_cases, 4, p0, 5),
                "power_at_or_10": matched_power(n_cases, 4, p0, 10),
            })
    m_tab = pd.DataFrame(rows)
    m_tab.to_csv(OUT / "power_matched.csv", index=False)
    print("\n== WITHIN-RUSSIA MATCHED CASE-CONTROL (H3), 4 controls per case ==")
    print(m_tab.round(3).to_string(index=False))

    # What would an observed count of N Russian events imply against background?
    rows = []
    py_ru = COHORT_PY["ever_elite"]
    for k in [1, 2, 3, 5, 10, 15, 20]:
        exp = py_ru * CENTRAL_RATE
        rows.append({"observed_russian_events": k,
                     "expected_under_background": round(exp, 3),
                     "implied_rate_ratio_vs_background": round(k / exp, 1)})
    smr = pd.DataFrame(rows)
    smr.to_csv(OUT / "power_implied_smr.csv", index=False)
    print("\n== IMPLIED RATE RATIO vs BACKGROUND (ever-elite cohort, 3/100k py) ==")
    print(smr.to_string(index=False))
    print(f"\n(expected Russian events under the null over 2010-2025: {py_ru*CENTRAL_RATE:.2f})")


if __name__ == "__main__":
    main()
