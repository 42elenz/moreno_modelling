"""Control: the shared backbone with random weights, frozen everywhere (also in joint finetune)."""
from __future__ import annotations

from wmpoc.models.encoders.supervised import SupervisedEncoder
from wmpoc.registry import ENCODERS


@ENCODERS.register("random_frozen")
class RandomFrozenEncoder(SupervisedEncoder):
    requires_pretraining = False
    trainable = False

    def __init__(self, dims, cfg) -> None:
        super().__init__(dims, cfg)
        for p in self.parameters():
            p.requires_grad_(False)
