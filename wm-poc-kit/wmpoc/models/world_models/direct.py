"""No latent dynamics: map z(t), the action and known future of step h straight to a
representation the outcome heads read. Trained through the heads (outcome loss)."""
from __future__ import annotations

import torch
from torch import nn

from wmpoc.models.world_models.base import WorldModel, mlp
from wmpoc.registry import WORLD_MODELS


@WORLD_MODELS.register("direct")
class DirectWorldModel(WorldModel):
    trained_on = "outcome"

    def __init__(self, dims, cfg) -> None:
        super().__init__(dims, cfg)
        D, C, H = self.d, dims.C, dims.H
        self.cat = nn.Parameter(torch.randn(C, D) * 0.02)
        self.step = nn.Parameter(torch.randn(H, D) * 0.02)
        self.ctx = nn.Linear(D, D)
        self.f = mlp(2 * D + D + dims.n_actions + dims.n_known_future, 2 * D, D)

    def forward(self, z_ctx, actions, known_future, static, store_emb, z_in=None):
        B, T, C, D = z_ctx.shape
        Hh = actions.shape[1]
        zl = z_ctx[:, -1][:, None].expand(B, Hh, C, D)
        hist = self.ctx(z_ctx.mean(1))[:, None].expand(B, Hh, C, D)
        s = (self.static_proj(static) + store_emb)[:, None, None]
        step = (self.step[:Hh, None] + self.cat)[None] + s  # [B,Hh,C,D]
        ak = torch.cat([actions, known_future], -1)[:, :, None].expand(B, Hh, C, -1)
        return zl + self.f(torch.cat([zl, hist, step, ak], -1))

    def rollout(self, z_ctx, actions, known_future, static, store_emb, steps, z0=None):
        return self.forward(z_ctx, actions[:, :steps], known_future[:, :steps], static, store_emb)
