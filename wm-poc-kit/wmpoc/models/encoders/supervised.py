"""Backbone without pretraining; trained end to end in stage 2 (freeze_encoder: false)."""
from __future__ import annotations

from wmpoc.models.backbone import Backbone
from wmpoc.models.encoders.base import Encoder, as_window
from wmpoc.registry import ENCODERS


@ENCODERS.register("supervised")
class SupervisedEncoder(Encoder):
    requires_pretraining = False

    def __init__(self, dims, cfg) -> None:
        super().__init__(dims, cfg)
        self.backbone = Backbone(dims, self.d, cfg.model.encoder, cfg.model.store_id_embedding)

    def encode(self, batch, drop_features=()):
        return self.backbone(as_window(batch), self.drop_idx(drop_features))[:, :, : self.dims.C]
