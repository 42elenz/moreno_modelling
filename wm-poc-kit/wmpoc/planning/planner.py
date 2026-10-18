"""Planner: roll out every tested discount level, score by risk-adjusted 4-week total margin.

A predictor maps windows (store_idx, origin) and a bread action sequence to per-member
predictions: `bread_units [K,N,H]` and `total_margin_4w [K,N]` (K = ensemble members).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from wmpoc.baselines.lightgbm_direct import LightGBMDirect
from wmpoc.data.features import ACTION_SCALE, Panel


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


def choose(margin: np.ndarray, risk_aversion: float, risk_on: str = "uplift") -> tuple[np.ndarray, np.ndarray]:
    """margin [K,L,N] -> (chosen level index [N], score [L,N]). score = mean - λ·std over members.

    risk_on="uplift" scores each member's margin relative to its own first level (0 %), so the
    penalty measures disagreement about the *effect* of the discount; "level" uses the raw margin,
    whose ensemble spread is mostly shared baseline error that does not change the ranking.
    """
    if risk_on == "uplift":
        margin = margin - margin[:, :1]
    elif risk_on != "level":
        raise ValueError(f"unknown risk_on {risk_on!r}")
    mean = margin.mean(0)
    std = margin.std(0) if margin.shape[0] > 1 else np.zeros_like(mean)
    score = mean - risk_aversion * std
    return score.argmax(0), score


class WorldModelPredictor:
    """Encoder → ensemble rollouts → heads → margin, for arbitrary bread action sequences."""

    name = "world_model"

    def __init__(self, system, data, device: str | object = "cpu", batch_size: int = 512) -> None:
        self.system, self.data, self.device, self.bs = system, data, device, batch_size
        self.tgt = data.dims.target_idx

    def _batches(self, s: np.ndarray, t: np.ndarray, actions: np.ndarray | None):
        for i in range(0, len(s), self.bs):
            b = self.data.ds.batch(s[i:i + self.bs], t[i:i + self.bs])
            if actions is not None:
                a = torch.as_tensor(actions[i:i + self.bs], dtype=torch.float32)[:, :, None] * ACTION_SCALE
                b = b.with_actions(a.expand(-1, -1, b.actions.shape[-1]).contiguous())
            yield b.to(self.device)

    def predict(self, s, t, actions=None):
        self.system.eval()
        units, margin = [], []
        for b in self._batches(s, t, actions):
            out = self.system.predict(b)
            units.append(out["sales"][..., self.tgt].cpu().numpy())
            margin.append(out["margin"].sum((2, 3)).cpu().numpy())
        return {"bread_units": np.concatenate(units, 1), "total_margin_4w": np.concatenate(margin, 1)}

    def real_embedding_units(self, s, t):
        self.system.eval()
        out = [self.system.decode_real(b)["sales"][..., self.tgt].cpu().numpy() for b in self._batches(s, t, None)]
        return np.concatenate(out, 0)

    def embeddings(self, s, t):
        """z(t): bread token and category mean of the last context week, concatenated."""
        self.system.eval()
        out = []
        with torch.no_grad():
            for b in self._batches(s, t, None):
                z = self.system.encoder.encode(b.ctx_window())[:, -1]
                out.append(torch.cat([z[:, self.tgt], z.mean(1)], -1).cpu().numpy())
        return np.concatenate(out, 0)
