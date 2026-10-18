"""Representation baseline: per-token linear projection plus ID embeddings, no attention."""
from __future__ import annotations

from torch import nn

from wmpoc.models.backbone import TokenEmbedding
from wmpoc.models.encoders.base import Encoder, as_window
from wmpoc.registry import ENCODERS


@ENCODERS.register("linear")
class LinearEncoder(Encoder):
    requires_pretraining = False

    def __init__(self, dims, cfg) -> None:
        super().__init__(dims, cfg)
        self.embed = TokenEmbedding(dims, self.d, cfg.model.store_id_embedding)
        self.norm = nn.LayerNorm(self.d)

    def encode(self, batch, drop_features=()):
        return self.norm(self.embed(as_window(batch), self.drop_idx(drop_features)))[:, :, : self.dims.C]
