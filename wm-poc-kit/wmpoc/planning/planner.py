"""Planner: roll out every tested discount level, score by risk-adjusted 4-week total margin.

A predictor maps windows (store_idx, origin) and a bread action sequence to per-member
predictions: `bread_units [K,N,H]` and `total_margin_4w [K,N]` (K = ensemble members).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from wmpoc.baselines.lightgbm_direct import LightGBMDirect
from wmpoc.data.features import Panel


class Predictor(Protocol):
    name: str

    def predict(self, s: np.ndarray, t: np.ndarray, actions: np.ndarray | None = None) -> dict[str, np.ndarray]:
        ...


class LightGBMPredictor:
    name = "lightgbm"

    def __init__(self, model: LightGBMDirect, panel: Panel) -> None:
        self.model, self.panel = model, panel

    def predict(self, s, t, actions=None):
        out = self.model.predict(self.panel, s, t, actions)
        return {k: v[None] for k, v in out.items()}


def plan_sequences(level: float, n: int, H: int) -> np.ndarray:
    """The ground-truth counterfactual policy: [d, 0, 0, 0]."""
    a = np.zeros((n, H))
    a[:, 0] = level
    return a


def candidate_levels(cfg) -> list[float]:
    levels = [float(x) for x in cfg.planning.discount_levels]
    if cfg.planning.allow_untested_levels:
        step = float(cfg.eval.untested_grid_step)
        levels = sorted(set(levels) | set(np.round(np.arange(0, max(levels) + 1e-9, step), 4).tolist()))
    return levels


def rollout_levels(pred: Predictor, s: np.ndarray, t: np.ndarray, levels: list[float], H: int) -> dict[str, np.ndarray]:
    """Per-level predictions: bread_units [K,L,N,H], total_margin_4w [K,L,N]."""
    outs = [pred.predict(s, t, plan_sequences(d, len(s), H)) for d in levels]
    return {k: np.stack([o[k] for o in outs], 1) for k in outs[0]}


def choose(margin: np.ndarray, risk_aversion: float) -> tuple[np.ndarray, np.ndarray]:
    """margin [K,L,N] -> (chosen level index [N], score [L,N]). score = mean - λ·std over members."""
    mean = margin.mean(0)
    std = margin.std(0) if margin.shape[0] > 1 else np.zeros_like(mean)
    score = mean - risk_aversion * std
    return score.argmax(0), score
