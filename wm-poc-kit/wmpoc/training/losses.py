"""Loss functions shared by the stages."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def elementwise_latent(pred: Tensor, target: Tensor, kind: str = "smooth_l1") -> Tensor:
    """Per-token loss, mean over the embedding dim. [...,D] -> [...]."""
    if kind == "l1":
        return (pred - target).abs().mean(-1)
    if kind == "l2":
        return ((pred - target) ** 2).mean(-1)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="none").mean(-1)
    if kind == "cosine":
        return 1 - F.cosine_similarity(pred, target, dim=-1)
    raise ValueError(f"unknown latent loss {kind!r}")


def weighted_mean(x: Tensor, w: Tensor | None) -> Tensor:
    if w is None:
        return x.mean()
    w = w.expand_as(x).to(x.dtype)
    return (x * w).sum() / w.sum().clamp_min(1e-8)


def latent_loss(pred: Tensor, target: Tensor, weight: Tensor | None = None, kind: str = "smooth_l1") -> Tensor:
    return weighted_mean(elementwise_latent(pred, target, kind), weight)


def vicreg(z: Tensor, var_weight: float, cov_weight: float) -> tuple[Tensor, Tensor, Tensor]:
    """VICReg variance + covariance terms on a set of embeddings z [N,D]."""
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var = F.relu(1 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_l = (off ** 2).sum() / d
    return var_weight * var + cov_weight * cov_l, var, cov_l


def huber(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    return F.huber_loss(pred, target, reduction="none", delta=delta)


def outcome_elementwise(pred: Tensor, target: Tensor, kind: str = "huber") -> Tensor:
    if kind == "huber":
        return huber(pred, target)
    if kind in ("l2", "mse"):
        return (pred - target) ** 2
    if kind == "l1":
        return (pred - target).abs()
    raise ValueError(f"unknown outcome loss {kind!r}")
