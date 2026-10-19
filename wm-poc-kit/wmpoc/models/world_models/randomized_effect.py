"""Baseline world model + action-effect module learned from the randomized experiment only.

    z(t+h) = baseline(z, known future)  +  g(z, a_≤h, known future) − g(z, 0, known future)

The baseline is the block-causal transformer with actions zeroed, trained on all windows. The
effect module g is a small per-category MLP. It only receives gradients from windows whose whole
horizon lies inside the experiment (`randomized`), where the bread discount was assigned at random
(slot weeks) or fixed by design (washout weeks). In observational windows the effect is applied
but detached, so confounded promo weeks can teach the baseline but not the effect of the action.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wmpoc.models.world_models.base import mlp
from wmpoc.models.world_models.transformer import TransformerWorldModel
from wmpoc.registry import WORLD_MODELS


@WORLD_MODELS.register("randomized_effect")
class RandomizedEffectWorldModel(TransformerWorldModel):
    zero_actions = True  # the baseline never sees the action

    def __init__(self, dims, cfg) -> None:
        super().__init__(dims, cfg)
        D, C, H, A = self.d, dims.C, dims.H, dims.n_actions
        self.eff_cat = nn.Parameter(torch.randn(C, D) * 0.02)
        self.eff_step = nn.Parameter(torch.randn(H, D) * 0.02)
        self.eff_static = nn.Linear(dims.n_static, D)
        self.effect = mlp(2 * D + H * A + dims.n_known_future, 2 * D, D, zero_last=True)
        self._learn_effect: Tensor | None = None

    def observe_batch(self, batch) -> None:
        self._learn_effect = batch.randomized.all(1) if self.training else None

    def _g(self, z_last: Tensor, acts: Tensor, kf: Tensor, s: Tensor, Hh: int) -> Tensor:
        B, C, D = z_last.shape
        x = torch.cat([
            z_last[:, None].expand(B, Hh, C, D),
            (s[:, None, None] + self.eff_cat + self.eff_step[:Hh, None]).expand(B, Hh, C, D),
            acts[:, :, None].expand(B, Hh, C, acts.shape[-1]),
            kf[:, :, None].expand(B, Hh, C, kf.shape[-1]),
        ], -1)
        return self.effect(x)

    def action_effect(self, z_ctx: Tensor, actions: Tensor, known_future: Tensor, static: Tensor, store_emb: Tensor) -> Tensor:
        """[B,Hh,C,D] latent effect of the plan; zero when every action is zero."""
        B, Hh, A = actions.shape
        H = self.dims.H
        # step h sees the actions of steps 0..h (causal), zero-padded to H steps
        causal = torch.tril(torch.ones(Hh, Hh, device=actions.device))
        seen = actions[:, None] * causal[None, :, :, None]  # [B,Hh,Hh,A]
        seen = torch.cat([seen, actions.new_zeros(B, Hh, H - Hh, A)], 2).reshape(B, Hh, H * A)
        s = self.eff_static(static) + store_emb
        z_last = z_ctx[:, -1]
        return self._g(z_last, seen, known_future, s, Hh) - self._g(z_last, torch.zeros_like(seen), known_future, s, Hh)

    def forward(self, z_ctx, actions, known_future, static, store_emb, z_in=None):
        base = super().forward(z_ctx, actions, known_future, static, store_emb, z_in)
        eff = self.action_effect(z_ctx, actions, known_future, static, store_emb)
        if self._learn_effect is not None:
            eff = torch.where(self._learn_effect[:, None, None, None], eff, eff.detach())
        return base + eff
