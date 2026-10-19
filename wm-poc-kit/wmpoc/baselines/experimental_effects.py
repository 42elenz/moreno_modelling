"""Effects of the bread action estimated from the randomized experiment only.

Uses windows whose first week is an experiment slot (`experiment_slot == 1`: the bread discount
was drawn at random), inside the given weeks and stores. For each window the outcome is the log
of a 4-week margin; it is regressed on discount-level dummies plus store and week fixed effects:

    log margin_4w[s, t] = α_s + γ_t + Σ_d β_d · 1[level = d] + ε

`exp(β_d) − 1` is the relative change caused by level d (vs 0 %). Store and week effects take out
the large differences between stores and the calendar, so only the random assignment remains.
Training code may call this: it reads sales data only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wmpoc.data.features import Panel


@dataclass
class RelativeEffects:
    levels: list[float]  # discount levels, first one is the reference (0 %)
    rel: np.ndarray  # [L] relative change vs the reference level (rel[0] = 0)
    se: np.ndarray  # [L] standard error of β_d (log scale)
    n: np.ndarray  # [L] slot windows per level

    def at(self, d: np.ndarray) -> np.ndarray:
        """Relative effect for arbitrary levels (linear interpolation between tested levels)."""
        return np.interp(d, self.levels, self.rel)


def slot_windows(p: Panel, weeks: tuple[int, int], stores: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    """(store, origin) of experiment-slot windows whose whole horizon lies in [weeks[0], weeks[1])."""
    keep = np.zeros(p.n_stores, dtype=bool)
    keep[stores] = True
    s, t = np.nonzero(p.slot & keep[:, None])
    ok = (t >= weeks[0]) & (t + H <= weeks[1])
    return s[ok], t[ok]


def relative_effects(p: Panel, weeks: tuple[int, int], stores: np.ndarray, levels: list[float], H: int,
                     categories: str = "other") -> RelativeEffects:
    """categories: "other" (all but the action categories), "bread" (action categories) or "total"."""
    s, t = slot_windows(p, weeks, stores, H)
    act = p.action_idx
    cat_mask = np.ones(len(p.categories), dtype=bool)
    if categories == "other":
        cat_mask[act] = False
    elif categories == "bread":
        cat_mask[:] = False
        cat_mask[act] = True
    elif categories != "total":
        raise ValueError(categories)
    y = sum(p.margin[s, t + h][:, cat_mask].sum(-1) for h in range(H))
    ok = y > 0
    s, t, y = s[ok], t[ok], np.log(y[ok])
    d = p.discount[s, t, act[0]]
    lvl = np.array([int(np.argmin(np.abs(np.asarray(levels) - x))) for x in d])
    # design: level dummies (without the reference), store dummies, week dummies (minus one)
    us, si = np.unique(s, return_inverse=True)
    ut, ti = np.unique(t, return_inverse=True)
    L = len(levels)
    X = np.zeros((len(y), (L - 1) + len(us) + len(ut) - 1))
    for j in range(1, L):
        X[lvl == j, j - 1] = 1.0
    X[np.arange(len(y)), (L - 1) + si] = 1.0
    wk = ti > 0
    X[np.nonzero(wk)[0], (L - 1) + len(us) + ti[wk] - 1] = 1.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    b = np.concatenate([[0.0], beta[: L - 1]])
    se = np.concatenate([[0.0], np.sqrt(np.diag(cov)[: L - 1])])
    n = np.array([(lvl == j).sum() for j in range(L)])
    return RelativeEffects(list(levels), np.exp(b) - 1, se, n)


@dataclass
class PullForward:
    """Bread units in horizon weeks h = 0..H-1 after a randomized slot at level d, relative to 0 %."""

    levels: list[float]
    beta: np.ndarray  # [L,H] unit effect of level d in week h (beta[0] = 0)
    se: np.ndarray  # [L,H]
    ratio: np.ndarray  # [L,H] beta[d,h] / beta[d,0]: share of the week-0 uplift gained (+) or lost (−) in week h
    slope: np.ndarray  # [H] units per unit of discount fraction in week h (FE regression on the continuous level)
    slope_se: np.ndarray  # [H]
    share: np.ndarray  # [H] pooled share slope[h] / slope[0]; 0 where the slope is not significant (|t| < t_min)

    def summary(self) -> str:
        return ", ".join(f"w{h} {r:+.3f} (t {sl / se:+.1f})" if se > 0 else f"w{h} {r:+.3f}"
                         for h, (r, sl, se) in enumerate(zip(self.share, self.slope, self.slope_se)))


def _fe_regression(y: np.ndarray, lvl: np.ndarray, s: np.ndarray, t: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    us, si = np.unique(s, return_inverse=True)
    ut, ti = np.unique(t, return_inverse=True)
    X = np.zeros((len(y), (L - 1) + len(us) + len(ut) - 1))
    for j in range(1, L):
        X[lvl == j, j - 1] = 1.0
    X[np.arange(len(y)), (L - 1) + si] = 1.0
    wk = ti > 0
    X[np.nonzero(wk)[0], (L - 1) + len(us) + ti[wk] - 1] = 1.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / max(len(y) - np.linalg.matrix_rank(X), 1)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X))[: L - 1])
    return np.concatenate([[0.0], beta[: L - 1]]), np.concatenate([[0.0], se])


def _fe_slope(y: np.ndarray, x: np.ndarray, s: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Slope of y on x with store and week fixed effects (Frisch–Waugh: demean x and y on both FEs)."""
    us, si = np.unique(s, return_inverse=True)
    ut, ti = np.unique(t, return_inverse=True)
    F = np.zeros((len(y), len(us) + len(ut) - 1))
    F[np.arange(len(y)), si] = 1.0
    wk = ti > 0
    F[np.nonzero(wk)[0], len(us) + ti[wk] - 1] = 1.0
    proj = lambda v: v - F @ np.linalg.lstsq(F, v, rcond=None)[0]
    xr, yr = proj(x), proj(y)
    b = float(xr @ yr / (xr @ xr))
    resid = yr - b * xr
    dof = max(len(y) - F.shape[1] - 1, 1)
    return b, float(np.sqrt(resid @ resid / dof / (xr @ xr)))


def pull_forward(p: Panel, weeks: tuple[int, int], stores: np.ndarray, levels: list[float], H: int,
                 t_min: float = 2.0) -> PullForward:
    """Unit effects of a randomized bread slot on bread sales in each of the next H weeks.

    Per-level effects share one reference group (the 0 % slots), so their noise is correlated; the
    pooled `share` uses the slope on the continuous discount instead and keeps only weeks whose
    slope is significant, so noise in late weeks is not mistaken for a pull-forward effect.
    """
    s, t = slot_windows(p, weeks, stores, H)
    act = p.action_idx[0]
    d = p.discount[s, t, act]
    lvl = np.array([int(np.argmin(np.abs(np.asarray(levels) - x))) for x in d])
    L = len(levels)
    beta, se = np.zeros((L, H)), np.zeros((L, H))
    for h in range(H):
        beta[:, h], se[:, h] = _fe_regression(p.units[s, t + h, act], lvl, s, t, L)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(beta[:, :1] != 0, beta / beta[:, :1], 0.0)
    slope, slope_se = np.zeros(H), np.zeros(H)
    for h in range(H):
        slope[h], slope_se[h] = _fe_slope(p.units[s, t + h, act], d, s, t)
    share = slope / slope[0]
    share[1:] = np.where(np.abs(slope[1:] / slope_se[1:]) >= t_min, share[1:], 0.0)
    return PullForward(list(levels), beta, se, ratio, slope, slope_se, share)


def level_effects(p: Panel, weeks: tuple[int, int], stores: np.ndarray, levels: list[float], y: np.ndarray,
                  H: int) -> tuple[np.ndarray, np.ndarray]:
    """FE effect of each level on y(s, t) (an outcome per slot window, e.g. 4-week bread waste).

    y: array [S, W] indexed by (store, origin). Returns (beta [L], se [L]); beta[0] = 0.
    """
    s, t = slot_windows(p, weeks, stores, H)
    d = p.discount[s, t, p.action_idx[0]]
    lvl = np.array([int(np.argmin(np.abs(np.asarray(levels) - x))) for x in d])
    return _fe_regression(y[s, t], lvl, s, t, len(levels))
