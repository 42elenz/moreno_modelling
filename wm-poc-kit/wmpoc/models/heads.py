"""Outcome heads: decode embeddings into units, waste and stock per category.

Margin is computed from the decoded outcomes with prices and costs (`compute_margin`);
it is never learned.

Censoring (`stockout_hours > 0` means sales are a lower bound on demand):
  none    plain loss on log sales; the units output is sales
  exclude censored cells are dropped from the units loss; the output is demand
  tobit   censored cells are only penalised when the prediction is below the observed sales;
          the output is demand
With `exclude`/`tobit`, sales = min(demand, stock). Stock is always an extra head output
(`log_stock_units`) because waste in units = waste_rate × stock.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor, nn

from wmpoc.data.features import ACTION_SCALE, HEAD_TARGETS, Normalizer
from wmpoc.data.windows import Dims, TokenBatch
from wmpoc.registry import HEADS
from wmpoc.training.losses import outcome_elementwise, weighted_mean


class OutcomeHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """z [B,H,C,D] -> {target: [B,H,C]} in normalised target space."""

    @abstractmethod
    def loss(self, preds: dict[str, Tensor], batch: TokenBatch) -> Tensor:
        """Handles censoring."""

    @abstractmethod
    def decode(self, preds: dict[str, Tensor]) -> dict[str, Tensor]:
        """Raw units: sales, demand, stock, waste_units, waste_rate [B,H,C]."""


class PerCategoryLinear(nn.Module):
    def __init__(self, C: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(C, d_in, d_out) / np.sqrt(d_in))
        self.b = nn.Parameter(torch.zeros(C, d_out))

    def forward(self, x: Tensor) -> Tensor:  # [...,C,d_in]
        return torch.einsum("...ci,cio->...co", x, self.w) + self.b


@HEADS.register("mlp")
class MLPHead(OutcomeHead):
    def __init__(self, dims: Dims, cfg, norm: Normalizer) -> None:
        super().__init__()
        hc = cfg.model.outcome_heads
        self.targets = list(hc.targets) + [t for t in ("log_stock_units",) if t not in hc.targets]
        for t in self.targets:
            if t not in HEAD_TARGETS:
                raise KeyError(f"unknown head target {t!r}")
        self.censoring = str(hc.censoring)
        if self.censoring not in ("none", "exclude", "tobit"):
            raise ValueError(f"unknown censoring {self.censoring!r}")
        self.loss_kind = str(hc.loss)
        sizes = [int(cfg.model.d_model)] + [int(h) for h in hc.hidden]
        layers: list[nn.Module] = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [PerCategoryLinear(dims.C, a, b), nn.GELU()]
        layers.append(PerCategoryLinear(dims.C, sizes[-1], len(self.targets)))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mean", torch.tensor(np.stack([norm.target_mean[t] for t in self.targets]), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.stack([norm.target_std[t] for t in self.targets]), dtype=torch.float32))

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        out = self.net(z)  # [B,H,C,K]
        return {t: out[..., i] for i, t in enumerate(self.targets)}

    def loss(self, preds: dict[str, Tensor], batch: TokenBatch, weights: Tensor | None = None) -> Tensor:
        H = next(iter(preds.values())).shape[1]
        w = (batch.weights if weights is None else weights)[:, :H, None]
        cens = batch.censored[:, :H]
        total = 0.0
        for t, p in preds.items():
            y = batch.target(t)[:, :H]
            if t == "log_units_sold" and self.censoring == "tobit":
                e = torch.where(cens, outcome_elementwise(torch.clamp(p - y, max=0.0), torch.zeros_like(p), self.loss_kind),
                                outcome_elementwise(p, y, self.loss_kind))
                total = total + weighted_mean(e, w)
            elif t == "log_units_sold" and self.censoring == "exclude":
                total = total + weighted_mean(outcome_elementwise(p, y, self.loss_kind), w * (~cens).float())
            else:
                total = total + weighted_mean(outcome_elementwise(p, y, self.loss_kind), w)
        return total / len(preds)

    def raw(self, preds: dict[str, Tensor], t: str) -> Tensor:
        i = self.targets.index(t)
        return preds[t] * self.std[i] + self.mean[i]

    def decode(self, preds: dict[str, Tensor]) -> dict[str, Tensor]:
        units = torch.expm1(self.raw(preds, "log_units_sold")).clamp_min(0)
        stock = torch.expm1(self.raw(preds, "log_stock_units")).clamp_min(0)
        sales = units if self.censoring == "none" else torch.minimum(units, stock)
        rate = self.raw(preds, "waste_rate").clamp(0, 0.95) if "waste_rate" in preds else torch.zeros_like(units)
        return {"sales": sales, "demand": units, "stock": stock, "waste_rate": rate, "waste_units": rate * stock}


def compute_margin(dec: dict[str, Tensor], batch: TokenBatch, dims: Dims) -> Tensor:
    """margin [B,H,C] = (net_price − unit_cost)·sales − unit_cost·waste.

    Action categories get net_price = base_price·(1 − action); other categories keep their
    planned (observed) net price.
    """
    H = dec["sales"].shape[-2]
    net = batch.net_price[:, :H].clone()
    for j, c in enumerate(dims.action_idx):
        net[:, :, c] = batch.base_price[:, :H, c] * (1 - batch.actions[:, :H, j] / ACTION_SCALE)
    cost = batch.unit_cost[:, :H]
    return (net - cost) * dec["sales"] - cost * dec["waste_units"]
