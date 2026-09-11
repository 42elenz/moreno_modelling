"""Matched case-control inference for the within-Russia analysis (protocol §12).

Design: 1:M matched sets, one fatal-fall case per set, M living elite controls
matched on sex, age, calendar year, sector, seniority and ownership type.
Exposure is a documented pre-death political / legal / state-business conflict.

For a binary exposure in a 1:M set, conditioning on the number exposed in the
set (m) reduces the conditional likelihood to

    P(case is the exposed one | m, psi) = m*psi / (m*psi + (M + 1 - m))

so the number of exposed cases is Poisson-binomial. That gives exact p-values
and exact confidence limits with no asymptotics, which is what the protocol
requires when events are sparse. At M = 1 this reduces to McNemar's exact test
and psi_hat = b/c, which the self-tests check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class MatchedSet:
    """One case plus its matched controls."""
    case_exposed: bool
    n_exposed_controls: int
    n_controls: int

    @property
    def m(self) -> int:
        """Total exposed in the set, case included."""
        return int(self.case_exposed) + self.n_exposed_controls

    @property
    def size(self) -> int:
        return 1 + self.n_controls


def _p_case_exposed(m: int, size: int, psi: float) -> float:
    """P(the case is an exposed member | m exposed in the set)."""
    n_un = size - m
    if m == 0:
        return 0.0
    if n_un == 0:
        return 1.0
    return m * psi / (m * psi + n_un)


def _informative(sets: list[MatchedSet]) -> list[MatchedSet]:
    """Sets where every member or no member is exposed carry no information."""
    return [s for s in sets if 0 < s.m < s.size]


def conditional_loglik(sets: list[MatchedSet], psi: float) -> float:
    ll = 0.0
    for s in _informative(sets):
        p = _p_case_exposed(s.m, s.size, psi)
        ll += np.log(p if s.case_exposed else 1 - p)
    return ll


def conditional_mle(sets: list[MatchedSet]) -> float:
    """Conditional maximum-likelihood odds ratio.

    Returns 0 or inf when the data are degenerate (no exposed case, or every
    informative case exposed); report the exact interval in that situation
    rather than the point estimate.
    """
    inf = _informative(sets)
    if not inf:
        return float("nan")
    n_exposed_cases = sum(s.case_exposed for s in inf)
    if n_exposed_cases == 0:
        return 0.0
    if n_exposed_cases == len(inf):
        return float("inf")
    r = optimize.minimize_scalar(
        lambda lb: -conditional_loglik(sets, float(np.exp(lb))),
        bounds=(-25, 25), method="bounded", options={"xatol": 1e-10})
    return float(np.exp(r.x))


def _poisson_binomial_pmf(ps: np.ndarray) -> np.ndarray:
    """Exact distribution of the number of successes, by convolution."""
    pmf = np.array([1.0])
    for p in ps:
        pmf = np.convolve(pmf, [1 - p, p])
    return pmf


def exact_test(sets: list[MatchedSet], psi_null: float = 1.0) -> dict:
    """Two-sided exact conditional test of OR = psi_null.

    The p-value is the total probability of every outcome no more likely than
    the observed one, which is the standard exact two-sided convention and
    avoids doubling a one-sided tail.
    """
    inf = _informative(sets)
    if not inf:
        return {"p_value": 1.0, "n_informative": 0, "t_obs": 0}
    ps = np.array([_p_case_exposed(s.m, s.size, psi_null) for s in inf])
    pmf = _poisson_binomial_pmf(ps)
    t_obs = int(sum(s.case_exposed for s in inf))
    tol = 1e-10
    p = float(pmf[pmf <= pmf[t_obs] + tol].sum())
    return {"p_value": min(p, 1.0), "n_informative": len(inf), "t_obs": t_obs,
            "expected_t_under_null": float((np.arange(len(pmf)) * pmf).sum())}


def exact_ci(sets: list[MatchedSet], alpha: float = 0.05) -> tuple[float, float]:
    """Exact conditional limits, by inverting the one-sided tails."""
    inf = _informative(sets)
    if not inf:
        return (0.0, float("inf"))
    t_obs = int(sum(s.case_exposed for s in inf))
    n = len(inf)

    def upper_tail(psi):                       # P(T >= t_obs)
        ps = np.array([_p_case_exposed(s.m, s.size, psi) for s in inf])
        return float(_poisson_binomial_pmf(ps)[t_obs:].sum())

    def lower_tail(psi):                       # P(T <= t_obs)
        ps = np.array([_p_case_exposed(s.m, s.size, psi) for s in inf])
        return float(_poisson_binomial_pmf(ps)[:t_obs + 1].sum())

    lo = 0.0 if t_obs == 0 else float(np.exp(
        optimize.brentq(lambda lb: upper_tail(np.exp(lb)) - alpha / 2, -30, 30)))
    hi = float("inf") if t_obs == n else float(np.exp(
        optimize.brentq(lambda lb: lower_tail(np.exp(lb)) - alpha / 2, -30, 30)))
    return lo, hi


def analyse(sets: list[MatchedSet], alpha: float = 0.05) -> dict:
    lo, hi = exact_ci(sets, alpha)
    t = exact_test(sets)
    inf = _informative(sets)
    return {
        "n_sets": len(sets),
        "n_informative_sets": len(inf),
        "n_exposed_cases": int(sum(s.case_exposed for s in sets)),
        "n_cases": len(sets),
        "or_conditional_mle": conditional_mle(sets),
        "ci_lo": lo, "ci_hi": hi,
        "p_exact": t["p_value"],
        "expected_exposed_cases_under_null": t.get("expected_t_under_null", float("nan")),
    }


def _selftest() -> None:
    # 1:1 matching reduces to McNemar: b discordant with case exposed, c the other way.
    b, c = 9, 2
    sets = ([MatchedSet(True, 0, 1)] * b) + ([MatchedSet(False, 1, 1)] * c)
    assert abs(conditional_mle(sets) - b / c) < 1e-6, conditional_mle(sets)
    from scipy import stats as st
    assert abs(exact_test(sets)["p_value"] - st.binomtest(b, b + c, 0.5).pvalue) < 1e-12

    # Concordant sets must not shift anything.
    padded = sets + [MatchedSet(True, 1, 1)] * 7 + [MatchedSet(False, 0, 1)] * 5
    assert abs(conditional_mle(padded) - b / c) < 1e-6
    assert abs(exact_test(padded)["p_value"] - exact_test(sets)["p_value"]) < 1e-12

    # No association: OR near 1, p near 1, CI straddling 1.
    null = ([MatchedSet(True, 1, 4)] * 6) + ([MatchedSet(False, 2, 4)] * 6)
    r = analyse(null)
    assert r["ci_lo"] < 1 < r["ci_hi"]

    # Every case exposed while controls mostly are not: infinite MLE, finite lower limit.
    strong = [MatchedSet(True, 0, 4)] * 8
    r = analyse(strong)
    assert np.isinf(r["or_conditional_mle"]) and r["ci_lo"] > 1 and np.isinf(r["ci_hi"])
    assert r["p_exact"] < 0.01

    # The exact CI must cover the conditional MLE.
    mid = ([MatchedSet(True, 1, 4)] * 7) + ([MatchedSet(False, 1, 4)] * 3)
    r = analyse(mid)
    assert r["ci_lo"] <= r["or_conditional_mle"] <= r["ci_hi"]

    print("matched.py self-tests pass")


if __name__ == "__main__":
    _selftest()
