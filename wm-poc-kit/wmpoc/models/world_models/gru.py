"""Mean-pooled z (over categories), GRU over the horizon, per-category residual read-out."""
from __future__ import annotations

import torch
from torch import nn

from wmpoc.models.world_models.base import WorldModel
from wmpoc.registry import WORLD_MODELS


@WORLD_MODELS.register("gru")
class GRUWorldModel(WorldModel):
    def __init__(self, dims, cfg) -> None:
        super().__init__(dims, cfg)
        D, C = self.d, dims.C
        self.hd = 2 * D
        self.h0 = nn.Linear(2 * D, self.hd)
        self.cell = nn.GRUCell(dims.n_actions + dims.n_known_future + D, self.hd)
        self.out = nn.Linear(self.hd, C * D)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_ctx, actions, known_future, static, store_emb, z_in=None):
        if z_in is None:
            raise ValueError("gru world model needs z_in or rollout()")
        B, _, C, D = z_ctx.shape
        h = torch.tanh(self.h0(torch.cat([z_ctx.mean((1, 2)), self.static_proj(static) + store_emb], -1)))
        outs = []
        for k in range(z_in.shape[1]):
            h = self.cell(torch.cat([actions[:, k], known_future[:, k], z_in[:, k].mean(1)], -1), h)
            outs.append(z_in[:, k] + self.out(h).view(B, C, D))
        return torch.stack(outs, 1)
