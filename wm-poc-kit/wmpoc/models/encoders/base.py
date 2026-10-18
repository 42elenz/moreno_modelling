"""Encoder interface. Encoders map a T-week window to one embedding per (week, category)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor, nn

from wmpoc.data.windows import Dims, TokenBatch, Window


def as_window(x: Window | TokenBatch) -> Window:
    return x.ctx_window() if isinstance(x, TokenBatch) else x


class Encoder(nn.Module, ABC):
    requires_pretraining: bool = False  # False for supervised, linear, random_frozen
    trainable: bool = True  # random_frozen never trains

    def __init__(self, dims: Dims, cfg) -> None:
        super().__init__()
        self.dims = dims
        self.cfg = cfg
        self.d = int(cfg.model.d_model)

    def drop_idx(self, drop_features: Sequence[str]) -> list[int]:
        return [self.dims.token_names.index(f) for f in drop_features if f in self.dims.token_names]

    @abstractmethod
    def encode(self, batch: Window | TokenBatch, drop_features: Sequence[str] = ()) -> Tensor:
        """[B,T,C,D]"""

    def target_encode(self, batch: Window | TokenBatch, drop_features: Sequence[str] = ()) -> Tensor:
        """Embeddings used as world-model targets. JEPA: EMA target encoder. Others: self, detached."""
        with torch.no_grad():
            return self.encode(batch, drop_features).detach()

    def pretrain_step(self, batch: Window | TokenBatch) -> dict[str, Tensor]:
        raise NotImplementedError(f"{type(self).__name__} has no pretraining objective")

    def after_optimizer_step(self, step: int, total_steps: int) -> None:
        pass
