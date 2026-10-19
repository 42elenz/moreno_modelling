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
        units, waste, margin, bmargin = [], [], [], []
        for b in self._batches(s, t, actions):
            out = self.system.predict(b)
            units.append(out["sales"][..., self.tgt].cpu().numpy())
            waste.append(out["waste_units"][..., self.tgt].cpu().numpy())
            margin.append(out["margin"].sum((2, 3)).cpu().numpy())
            bmargin.append(out["margin"][..., self.tgt].sum(2).cpu().numpy())
        return {"bread_units": np.concatenate(units, 1), "total_margin_4w": np.concatenate(margin, 1),
                "bread_margin_4w": np.concatenate(bmargin, 1), "bread_waste_units": np.concatenate(waste, 1)}

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


class HybridPredictor:
    """World model for bread, randomized experiment for the other categories (option 1).

    For a plan [d, 0, 0, 0]: total margin = world-model bread margin under the plan + the world
    model's *no-discount* margin of the other categories × (1 + r(d)), where r(d) is the relative
    effect of level d on the other categories' 4-week margin, estimated only from randomized slot
    weeks of the training period (`baselines.experimental_effects`). The world model's own
    (confounded) cross-category response is not used. Forecasts under observed actions are the
    world model's.
    """

    name = "world_model_hybrid"

    def __init__(self, base: WorldModelPredictor, other_effects) -> None:
        self.base, self.eff = base, other_effects
        self._zero_key, self._zero = None, None

    def _no_discount(self, s, t):
        key = (s.tobytes(), t.tobytes())
        if key != self._zero_key:
            self._zero = self.base.predict(s, t, np.zeros((len(s), self.base.data.dims.H)))
            self._zero_key = key
        return self._zero

    def predict(self, s, t, actions=None):
        out = self.base.predict(s, t, actions)
        if actions is None:
            return out
        z = self._no_discount(s, t)
        other0 = z["total_margin_4w"] - z["bread_margin_4w"]  # [K,N]
        other = other0 * (1 + self.eff.at(actions[:, 0]))[None]
        return {"bread_units": out["bread_units"], "total_margin_4w": out["bread_margin_4w"] + other,
                "bread_margin_4w": out["bread_margin_4w"]}


class StructuredPredictor(HybridPredictor):
    """Hybrid plus a structured bread margin (the fix for the pull-forward error).

    For a plan [d, 0, 0, 0]:
      week 0      bread sales and waste from the world model under the plan;
      weeks 1..3  bread sales = world-model no-discount sales + share_h × (week-0 uplift), where
                  share_h is the pull-forward pattern estimated from the randomized slots
                  (`baselines.experimental_effects.pull_forward`); waste from the world model;
      bread margin = Σ_h (net price − unit cost) · sales − unit cost · waste, with known prices;
      other categories as in `HybridPredictor`.
    The world model supplies what it predicts well (store-specific week-0 uplift, waste, the
    no-discount baseline); the experiment supplies the two effects it gets wrong.
    """

    name = "world_model_structured"

    def __init__(self, base: WorldModelPredictor, other_effects, pull_forward) -> None:
        super().__init__(base, other_effects)
        self.pf = pull_forward

    def predict(self, s, t, actions=None):
        out = self.base.predict(s, t, actions)
        if actions is None or np.any(actions[:, 1:] != 0):
            return out  # the structure is defined for single-week plans only
        return self._structured(s, t, actions, out, self._no_discount(s, t))

    def _structured(self, s, t, actions, out, z):
        p, b, H = self.base.data.panel, self.base.tgt, self.base.data.dims.H
        wk = t[:, None] + np.arange(H)[None]
        base_price, cost = p.base_price[s[:, None], wk, b], p.unit_cost[s[:, None], wk, b]
        net = base_price * (1 - actions)
        u0 = out["bread_units"][..., 0] - z["bread_units"][..., 0]  # [K,N] week-0 uplift
        units = z["bread_units"].copy()
        units[..., 0] = out["bread_units"][..., 0]
        units[..., 1:] += self.pf.share[None, None, 1:H] * u0[..., None]
        bread = ((net - cost)[None] * units - cost[None] * out["bread_waste_units"]).sum(-1)
        other = (z["total_margin_4w"] - z["bread_margin_4w"]) * (1 + self.eff.at(actions[:, 0]))[None]
        return {"bread_units": units, "total_margin_4w": bread + other, "bread_margin_4w": bread,
                "bread_waste_units": out["bread_waste_units"]}


class CalibratedPredictor(StructuredPredictor):
    """Structured planner whose average bread effects are calibrated to the experiment.

    For each tested level d, the world model's week-0 bread uplift and its 4-week bread waste
    change are rescaled so that their means over the training-period slot windows equal the
    experimental estimates (store and week fixed effects) on the same windows:
        k_u(d) = β_units,w0(d) / mean WM uplift(d),   k_w(d) = β_waste,4w(d) / mean WM Δwaste(d).
    The world model keeps the *differences between stores*; the experiment fixes the *average*.
    Factors are clipped to [0.5, 2] and interpolated between tested levels.
    """

    name = "world_model_calibrated"

    def __init__(self, base, other_effects, pull_forward, calib_s, calib_t, units_beta, waste_beta, levels) -> None:
        super().__init__(base, other_effects, pull_forward)
        self.levels = np.asarray(levels, dtype=float)
        H = base.data.dims.H
        z = base.predict(calib_s, calib_t, np.zeros((len(calib_s), H)))
        ku, kw = np.ones(len(levels)), np.ones(len(levels))
        for i, d in enumerate(levels[1:], start=1):
            o = base.predict(calib_s, calib_t, plan_sequences(d, len(calib_s), H))
            du = (o["bread_units"][..., 0] - z["bread_units"][..., 0]).mean()
            dw = (o["bread_waste_units"] - z["bread_waste_units"]).sum(-1).mean()
            ku[i] = units_beta[i] / du if abs(du) > 1e-6 else 1.0
            kw[i] = waste_beta[i] / dw if abs(dw) > 1e-6 else 1.0
        self.ku, self.kw = np.clip(ku, 0.5, 2.0), np.clip(kw, 0.5, 2.0)

    def summary(self) -> str:
        return ", ".join(f"{d:.0%}: units ×{u:.2f} waste ×{w:.2f}" for d, u, w in zip(self.levels[1:], self.ku[1:], self.kw[1:]))

    def predict(self, s, t, actions=None):
        if actions is None or np.any(actions[:, 1:] != 0):
            return self.base.predict(s, t, actions)
        out = self.base.predict(s, t, actions)
        z = self._no_discount(s, t)
        d = actions[:, 0]
        ku, kw = np.interp(d, self.levels, self.ku), np.interp(d, self.levels, self.kw)
        units0 = z["bread_units"][..., 0] + ku[None] * (out["bread_units"][..., 0] - z["bread_units"][..., 0])
        waste = z["bread_waste_units"] + kw[None, :, None] * (out["bread_waste_units"] - z["bread_waste_units"])
        cal = dict(out, bread_waste_units=waste)
        cal["bread_units"] = out["bread_units"].copy()
        cal["bread_units"][..., 0] = units0
        return self._structured(s, t, actions, cal, z)
