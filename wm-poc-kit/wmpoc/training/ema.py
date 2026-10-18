"""EMA helpers for the JEPA target encoder."""
from __future__ import annotations

import math

import torch
from torch import nn


def ema_momentum(step: int, total: int, start: float, end: float) -> float:
    """Cosine schedule from `start` (step 0) to `end` (last step)."""
    if total <= 1:
        return end
    frac = min(max(step / (total - 1), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * frac) + 1) / 2


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, m: float) -> None:
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(m).add_(po.detach(), alpha=1 - m)
    for bt, bo in zip(target.buffers(), online.buffers()):
        bt.copy_(bo)
