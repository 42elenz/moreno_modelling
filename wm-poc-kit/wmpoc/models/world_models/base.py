"""World-model interface: predict future (week, category) embeddings from the encoded past,
the planned action sequence and known-future features."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from wmpoc.data.windows import Dims
from wmpoc.models.backbone import StoreEmbedding


class WorldModel(nn.Module, ABC):
    trained_on: str = "latent"  # "outcome": no latent targets, trained through the heads (direct)

    def __init__(self, dims: Dims, cfg) -> None:
        super().__init__()
        self.dims = dims
        self.d = int(cfg.model.d_model)
        self.wc = cfg.model.world_model
        self.store_embedding = StoreEmbedding(dims.n_stores, self.d, cfg.model.store_id_embedding)
        self.static_proj = nn.Linear(dims.n_static, self.d)

    def embed_store(self, store_id: Tensor, known: Tensor) -> Tensor:
        return self.store_embedding(store_id, known)

    @abstractmethod
    def forward(self, z_ctx: Tensor, actions: Tensor, known_future: Tensor, static: Tensor, store_emb: Tensor,
                z_in: Tensor | None = None) -> Tensor:
        """Teacher-forced. z_ctx [B,T,C,D]; actions [B,H',A]; known_future [B,H',Fk];
        z_in [B,H',C,D] = latent fed into each horizon step (z0, then the real targets).
        Returns predicted embeddings of the H' horizon weeks [B,H',C,D]."""

    def rollout(self, z_ctx: Tensor, actions: Tensor, known_future: Tensor, static: Tensor, store_emb: Tensor,
                steps: int, z0: Tensor | None = None) -> Tensor:
        """Autoregressive: every step is fed the model's own previous prediction."""
        inputs = [z_ctx[:, -1] if z0 is None else z0]
        preds = []
        for h in range(steps):
            out = self.forward(z_ctx, actions[:, : h + 1], known_future[:, : h + 1], static, store_emb,
                               torch.stack(inputs, 1))
            preds.append(out[:, h])
            inputs.append(out[:, h])
        return torch.stack(preds, 1)


def mlp(d_in: int, d_hidden: int, d_out: int, zero_last: bool = False) -> nn.Sequential:
    m = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))
    if zero_last:
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
    return m
