"""Shared backbone: token embedding + factorized time/category attention.

Token embedding = Linear([features · keep, keep] → D) + category embedding + week-position
embedding + store embedding + static projection. `keep` is 0 for dropped features (price
blanking, `target_drops_price`) and for masked cells (pretraining masks), so a single
mechanism covers all of them. One extra context token per week carries the calendar.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from wmpoc.data.windows import Dims, Window


class StoreEmbedding(nn.Module):
    """Store-ID vector, zeroed for unknown (cold-start) stores and, in training, with prob. `dropout`."""

    def __init__(self, n_stores: int, d: int, cfg) -> None:
        super().__init__()
        self.enabled = bool(cfg.enabled)
        self.p = float(cfg.dropout)
        self.emb = nn.Embedding(n_stores, d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, store_id: Tensor, known: Tensor) -> Tensor:
        e = self.emb(store_id)
        if not self.enabled:
            return torch.zeros_like(e)
        keep = known.to(e.dtype)
        if self.training and self.p > 0:
            keep = keep * (torch.rand_like(keep) >= self.p).to(e.dtype)
        return e * keep[:, None]


def keep_mask(tokens: Tensor, drop_idx: Sequence[int] = (), cell_mask: Tensor | None = None) -> Tensor:
    keep = torch.ones_like(tokens)
    if len(drop_idx):
        keep[..., list(drop_idx)] = 0.0
    if cell_mask is not None:
        keep = keep * (~cell_mask)[..., None].to(tokens.dtype)
    return keep


class TokenEmbedding(nn.Module):
    def __init__(self, dims: Dims, d: int, store_cfg) -> None:
        super().__init__()
        self.value = nn.Linear(2 * dims.F, d)
        self.ctx = nn.Linear(dims.n_context, d)
        self.cat = nn.Embedding(dims.C + 1, d)  # last slot = per-week context token
        self.pos = nn.Embedding(dims.T, d)
        self.static = nn.Linear(dims.n_static, d)
        self.store = StoreEmbedding(dims.n_stores, d, store_cfg)
        for e in (self.cat, self.pos):
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, w: Window, drop_idx: Sequence[int] = (), cell_mask: Tensor | None = None) -> Tensor:
        T = w.tokens.shape[1]
        keep = keep_mask(w.tokens, drop_idx, cell_mask)
        x = self.value(torch.cat([w.tokens * keep, keep], -1))  # [B,T,C,D]
        x = torch.cat([x, self.ctx(w.context)[:, :, None]], 2)  # [B,T,C+1,D]
        x = x + self.cat.weight[None, None] + self.pos.weight[:T][None, :, None]
        return x + (self.static(w.static) + self.store(w.store_id, w.store_known))[:, None, None]


def causal_mask(n: int, device) -> Tensor:
    return torch.ones(n, n, dtype=torch.bool, device=device).triu(1)  # True = blocked


class FactorizedBlock(nn.Module):
    """Attention over weeks (per slot), then over categories (per week), then an MLP."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: float, dropout: float, causal_time: bool,
                 full: bool = False) -> None:
        super().__init__()
        self.causal, self.full = causal_time, full
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.time_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.cat_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, d))

    def forward(self, x: Tensor) -> Tensor:
        B, T, S, D = x.shape
        if self.full:  # joint attention over all (week, slot) tokens, block-causal in time
            h = self.n1(x).reshape(B, T * S, D)
            m = None
            if self.causal:
                wk = torch.arange(T, device=x.device).repeat_interleave(S)
                m = wk[None, :] > wk[:, None]
            x = x + self.time_attn(h, h, h, attn_mask=m, need_weights=False)[0].reshape(B, T, S, D)
        else:
            h = self.n1(x).transpose(1, 2).reshape(B * S, T, D)
            m = causal_mask(T, x.device) if self.causal else None
            a = self.time_attn(h, h, h, attn_mask=m, need_weights=False)[0]
            x = x + a.reshape(B, S, T, D).transpose(1, 2)
            h = self.n2(x).reshape(B * T, S, D)
            x = x + self.cat_attn(h, h, h, need_weights=False)[0].reshape(B, T, S, D)
        return x + self.mlp(self.n3(x))


class Backbone(nn.Module):
    def __init__(self, dims: Dims, d: int, enc_cfg, store_cfg, n_layers: int | None = None) -> None:
        super().__init__()
        self.embed = TokenEmbedding(dims, d, store_cfg)
        full = str(enc_cfg.get("attention", "factorized")) == "full"
        n = int(enc_cfg.n_layers if n_layers is None else n_layers)
        self.blocks = nn.ModuleList([
            FactorizedBlock(d, int(enc_cfg.n_heads), float(enc_cfg.mlp_ratio), float(enc_cfg.dropout),
                            bool(enc_cfg.causal_time), full) for _ in range(n)])
        self.norm = nn.LayerNorm(d)

    def forward(self, w: Window, drop_idx: Sequence[int] = (), cell_mask: Tensor | None = None) -> Tensor:
        """Returns [B,T,C+1,D] (last slot = context token)."""
        x = self.embed(w, drop_idx, cell_mask)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
