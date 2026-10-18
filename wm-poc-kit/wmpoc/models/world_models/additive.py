"""z(t+h) = z(t+h-1) + MLP(action, known future): the simplest 'add an action' model."""
from __future__ import annotations

import torch

from wmpoc.models.world_models.base import WorldModel, mlp
from wmpoc.registry import WORLD_MODELS


@WORLD_MODELS.register("additive")
class AdditiveWorldModel(WorldModel):
    def __init__(self, dims, cfg) -> None:
        super().__init__(dims, cfg)
        D, C = self.d, dims.C
        self.f = mlp(dims.n_actions + dims.n_known_future + D, 2 * D, C * D, zero_last=True)

    def forward(self, z_ctx, actions, known_future, static, store_emb, z_in=None):
        if z_in is None:
            raise ValueError("additive world model needs z_in or rollout()")
        B, Hh, C, D = z_in.shape
        s = (self.static_proj(static) + store_emb)[:, None].expand(B, Hh, D)
        return z_in + self.f(torch.cat([actions, known_future, s], -1)).view(B, Hh, C, D)
