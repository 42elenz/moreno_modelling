"""Rare-event incidence machinery for the elite fatal-fall study.

Everything here is exact or mid-p. The protocol prespecifies that asymptotic
normal approximations must not carry the primary inference, because the
expected event counts are far below the range where they behave.

Reference results used in the self-tests come from published worked examples
(Ulm 1990; Sahai & Khurshid 1993; Rothman, Greenland & Lash, *Modern
Epidemiology* 3rd ed., ch. 14).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


# --------------------------------------------------------------------------
# Single-rate inference
# --------------------------------------------------------------------------
def poisson_exact_ci(k: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Garwood) confidence limits for a Poisson count.

    Lower limit is 0 when k == 0; the upper limit is always finite, which is
    what lets us report a bound from a country that observed no events.
    """
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return float(lo), float(hi)


def incidence(k: int, person_years: float, per: float = 10_000.0,
              alpha: float = 0.05) -> dict:
    """Incidence rate per `per` person-years with exact limits."""
    lo, hi = poisson_exact_ci(k, alpha)
    return {
        "events": int(k),
        "person_years": float(person_years),
        "rate": k / person_years * per,
        "lo": lo / person_years * per,
        "hi": hi / person_years * per,
        "per": per,
    }


# --------------------------------------------------------------------------
# Two-rate comparison
# --------------------------------------------------------------------------
def irr_exact(k1: int, py1: float, k0: int, py0: float,
              alpha: float = 0.05, mid_p: bool = False) -> dict:
    """Incidence-rate ratio (group 1 vs group 0) with exact conditional limits.

    Conditioning on the total number of events T = k1 + k0 makes k1 binomial
    with p = (IRR * py1) / (IRR * py1 + py0). Inverting an exact binomial
    interval for p therefore gives an exact interval for the IRR. This is the
    standard rare-event approach and stays valid at k = 0.

    mid_p=True applies the mid-p correction, which is less conservative; the
    protocol prespecifies exact limits as primary and mid-p as a secondary
    report.
    """
    T = k1 + k0
    ratio = py0 / py1
    if T == 0:
        return {"irr": np.nan, "lo": 0.0, "hi": np.inf, "k1": k1, "k0": k0,
                "py1": py1, "py0": py0, "p_value": 1.0, "method": "undefined (no events)"}

    est = np.inf if k0 == 0 else (k1 / py1) / (k0 / py0)

    if mid_p:
        def _lo_eq(p):
            return (stats.binom.sf(k1, T, p) + 0.5 * stats.binom.pmf(k1, T, p)) - alpha / 2

        def _hi_eq(p):
            return (stats.binom.cdf(k1 - 1, T, p) + 0.5 * stats.binom.pmf(k1, T, p)) - alpha / 2
        p_lo = 0.0 if k1 == 0 else optimize.brentq(_lo_eq, 1e-12, 1 - 1e-12)
        p_hi = 1.0 if k1 == T else optimize.brentq(_hi_eq, 1e-12, 1 - 1e-12)
    else:
        p_lo, p_hi = _clopper_pearson(k1, T, alpha)

    lo = p_lo / (1 - p_lo) * ratio if p_lo > 0 else 0.0
    hi = p_hi / (1 - p_hi) * ratio if p_hi < 1 else np.inf

    # Two-sided exact binomial test of IRR = 1, i.e. p = py1 / (py1 + py0)
    p_null = py1 / (py1 + py0)
    pval = float(stats.binomtest(k1, T, p_null).pvalue)

    return {"irr": float(est), "lo": float(lo), "hi": float(hi),
            "k1": int(k1), "k0": int(k0), "py1": float(py1), "py0": float(py0),
            "p_value": pval, "method": "exact conditional (mid-p)" if mid_p else "exact conditional"}


def _clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# Person-time
# --------------------------------------------------------------------------
@dataclass
class FollowUp:
    entry: np.ndarray          # datetime64[D]
    exit: np.ndarray           # datetime64[D]


def person_years(entry, exit_, window_start, window_end) -> float:
    """Person-years inside [window_start, window_end], clipped per person.

    Half-open on the right in days, then converted at 365.25 d/yr. Entry after
    the window or exit before it contributes zero rather than a negative.
    """
    e = np.maximum(np.asarray(entry, dtype="datetime64[D]"),
                   np.datetime64(window_start, "D"))
    x = np.minimum(np.asarray(exit_, dtype="datetime64[D]"),
                   np.datetime64(window_end, "D"))
    days = (x - e).astype("timedelta64[D]").astype(float) + 1.0
    return float(np.clip(days, 0, None).sum() / 365.25)


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------
def _selftest() -> None:
    # Garwood limits for k=10: 4.795 to 18.390 (standard table value)
    lo, hi = poisson_exact_ci(10)
    assert abs(lo - 4.7954) < 1e-3 and abs(hi - 18.3904) < 1e-3, (lo, hi)

    # k=0 must give a finite upper bound of 3.689 (the "rule of three" exact form)
    lo, hi = poisson_exact_ci(0)
    assert lo == 0.0 and abs(hi - 3.6889) < 1e-3, (lo, hi)

    # IRR with equal person-time and equal counts must be 1 with a CI straddling it
    r = irr_exact(5, 1000.0, 5, 1000.0)
    assert abs(r["irr"] - 1.0) < 1e-12 and r["lo"] < 1 < r["hi"]
    assert r["p_value"] > 0.99

    # Zero events in the reference group -> infinite point estimate, finite lower limit
    r = irr_exact(4, 1000.0, 0, 1000.0)
    assert np.isinf(r["irr"]) and r["lo"] > 0 and np.isinf(r["hi"])

    # Zero events in the index group -> zero estimate, finite upper limit
    r = irr_exact(0, 1000.0, 4, 1000.0)
    assert r["irr"] == 0.0 and r["lo"] == 0.0 and np.isfinite(r["hi"])

    # mid-p must be strictly narrower than exact
    a = irr_exact(6, 1000.0, 2, 3000.0)
    b = irr_exact(6, 1000.0, 2, 3000.0, mid_p=True)
    assert b["lo"] > a["lo"] and b["hi"] < a["hi"]

    # person-years: one person followed the whole of 2010 is 365/365.25 years
    py = person_years(["2010-01-01"], ["2010-12-31"], "2010-01-01", "2010-12-31")
    assert abs(py - 365 / 365.25) < 1e-9, py

    # entry after the window contributes nothing
    py = person_years(["2030-01-01"], ["2031-01-01"], "2010-01-01", "2025-12-31")
    assert py == 0.0

    print("rates.py self-tests pass")


if __name__ == "__main__":
    _selftest()
