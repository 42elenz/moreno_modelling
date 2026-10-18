"""Randomized difference in means: the experimental effect estimate per discount level."""
from __future__ import annotations

import numpy as np

from wmpoc.data.features import Panel


def diff_in_means(panel: Panel, weeks: tuple[int, int], levels=(0.1, 0.2, 0.3),
                  store_idx: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """Bread rows with experiment_slot == 1 and weeks[0] <= week <= weeks[1] (inclusive).

    Returns per level: mean units at level d minus mean units at 0 %, its standard error and n.
    """
    tgt = panel.categories.index("bread") if "bread" in panel.categories else 0
    w = np.arange(panel.n_weeks)[None, :]
    sel = panel.slot & (w >= weeks[0]) & (w <= weeks[1])
    if store_idx is not None:
        keep = np.zeros(panel.n_stores, dtype=bool)
        keep[store_idx] = True
        sel &= keep[:, None]
    units = panel.units[:, :, tgt]
    disc = panel.discount[:, :, tgt]
    g0 = units[sel & np.isclose(disc, 0.0)]
    out = {}
    for d in levels:
        g1 = units[sel & np.isclose(disc, d)]
        if len(g1) < 2 or len(g0) < 2:
            out[f"{int(round(d * 100))}%"] = {"estimate": float("nan"), "se": float("nan"), "n": int(len(g1))}
            continue
        est = g1.mean() - g0.mean()
        se = np.sqrt(g1.var(ddof=1) / len(g1) + g0.var(ddof=1) / len(g0))
        out[f"{int(round(d * 100))}%"] = {"estimate": float(est), "se": float(se), "n": int(len(g1))}
    return out
