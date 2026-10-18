"""Week-based splits with a horizon gap, plus cold-start stores.

A window has origin `t` (first predicted week), context weeks [t-T, t) and horizon [t, t+H).
For split [a, b) the origins are a ≤ t ≤ b-H, so the horizon never leaves the split: the
last H weeks before each boundary are a gap in which no window is anchored. Context weeks
may reach back into the previous split; that is observed history, not leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPLITS = ("train", "val", "test")


@dataclass
class Splits:
    ranges: dict[str, tuple[int, int]]
    train_stores: np.ndarray  # panel indices
    holdout_stores: np.ndarray  # cold start: never in any training split
    context_weeks: int
    horizon_weeks: int
    stride: int = 1

    def origins(self, split: str) -> np.ndarray:
        a, b = self.ranges[split]
        lo = max(a, self.context_weeks)
        return np.arange(lo, b - self.horizon_weeks + 1, self.stride)

    def stores(self, which: str) -> np.ndarray:
        if which == "train":
            return self.train_stores
        if which == "holdout":
            return self.holdout_stores
        if which == "all":
            return np.sort(np.concatenate([self.train_stores, self.holdout_stores]))
        raise ValueError(which)

    def windows(self, split: str, stores: str = "train") -> tuple[np.ndarray, np.ndarray]:
        """(store_idx, origin) pairs. Training always uses stores='train'."""
        if split == "train" and stores != "train":
            raise ValueError("training windows only come from training stores")
        s = self.stores(stores)
        o = self.origins(split)
        ss, oo = np.meshgrid(s, o, indexing="ij")
        return ss.ravel(), oo.ravel()

    def target_weeks(self, split: str) -> set[int]:
        o = self.origins(split)
        return set((o[:, None] + np.arange(self.horizon_weeks)[None, :]).ravel().tolist())


def make_splits(n_stores: int, cfg) -> Splits:
    sc = cfg.data.split
    rng = np.random.default_rng(int(sc.seed))
    n_hold = int(round(float(sc.holdout_store_frac) * n_stores))
    perm = rng.permutation(n_stores)
    hold = np.sort(perm[:n_hold])
    train = np.sort(perm[n_hold:])
    ranges = {k: (int(sc[k][0]), int(sc[k][1])) for k in SPLITS}
    w = cfg.data.window
    s = Splits(ranges, train, hold, int(w.context_weeks), int(w.horizon_weeks), int(w.stride_weeks))
    for a, b in [("train", "val"), ("val", "test")]:
        if s.ranges[a][1] > s.ranges[b][0]:
            raise ValueError(f"split {a} overlaps {b}")
    return s
