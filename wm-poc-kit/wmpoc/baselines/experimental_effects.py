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
