"""Pretraining masks shared by `jepa` and `masked_ae`. One mask type is sampled per batch."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from wmpoc.data.windows import Dims

KINDS = ("future", "category", "price", "random_block")


@dataclass
class MaskSpec:
    kind: str
    cell_mask: Tensor  # [B,T,C] True = cell hidden from the context encoder
    loss_mask: Tensor  # [B,T,C] True = cell contributes to the loss
    drop_idx: list[int] = field(default_factory=list)  # features blanked everywhere


class MaskSampler:
    def __init__(self, masks_cfg, dims: Dims, seed: int = 0) -> None:
        self.cfg = masks_cfg
        self.dims = dims
        self.kinds = [k for k in KINDS if k in masks_cfg and float(masks_cfg[k].p) > 0]
        p = np.array([float(masks_cfg[k].p) for k in self.kinds])
        self.p = p / p.sum()
        self.rng = np.random.default_rng(seed)

    def sample(self, B: int, device=None, kind: str | None = None) -> MaskSpec:
        T, C = self.dims.T, self.dims.C
        kind = kind or str(self.rng.choice(self.kinds, p=self.p))
        m = np.zeros((B, T, C), dtype=bool)
        drop: list[int] = []
        c = self.cfg[kind]
        if kind == "future":
            lo, hi = (int(x) for x in c.n_weeks)
            n = self.rng.integers(lo, min(hi, T - 1) + 1, B)
            for i in range(B):
                m[i, T - n[i]:] = True
            loss = m
        elif kind == "category":
            lo, hi = (int(x) for x in c.n_categories)
            n = self.rng.integers(lo, min(hi, C - 1) + 1, B)
            for i in range(B):
                m[i, :, self.rng.choice(C, n[i], replace=False)] = True
            loss = m
        elif kind == "price":
            drop = list(self.dims.price_idx)
            loss = np.ones_like(m)
        elif kind == "random_block":
            ratio = float(c.ratio)
            for i in range(B):
                while m[i].mean() < ratio:
                    tl, cl = self.rng.integers(1, max(2, T // 2) + 1), self.rng.integers(1, 4)
                    t0, c0 = self.rng.integers(0, T - tl + 1), self.rng.integers(0, C - cl + 1)
                    m[i, t0:t0 + tl, c0:c0 + cl] = True
            loss = m
        else:
            raise ValueError(kind)
        return MaskSpec(kind, torch.as_tensor(m, device=device), torch.as_tensor(loss, device=device), drop)
