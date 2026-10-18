"""Block-causal transformer world model.

Sequence = prefix (2·C context tokens: last context week and the mean over the context weeks,
plus one static/store token) followed by one block per horizon step: C latent tokens + an action token + a known-future token. Block h attends
to the prefix and blocks ≤ h only. Actions enter as tokens and via AdaLN on the tokens of the
action (target) categories.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wmpoc.models.world_models.base import WorldModel, mlp
from wmpoc.registry import WORLD_MODELS


class AdaLNBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        # no dropout on attention weights: it forces the slow attention kernel; the MLP keeps dropout
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, d))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 4 * d))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x: Tensor, mask: Tensor, cond: Tensor | None, mod_idx: Tensor | None) -> Tensor:
        """cond [B,M,D]: one conditioning vector per modulated token, at sequence positions mod_idx [M]."""
        g1 = b1 = g2 = b2 = None
        if cond is not None:
            g1, b1, g2, b2 = (self._scatter(m, x, mod_idx) for m in self.mod(cond).chunk(4, -1))
        h = self.n1(x)
        if g1 is not None:
            h = h * (1 + g1) + b1
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = self.n2(x)
        if g2 is not None:
            h = h * (1 + g2) + b2
        return x + self.mlp(h)

    @staticmethod
    def _scatter(m: Tensor, x: Tensor, idx: Tensor) -> Tensor:
        out = x.new_zeros(x.shape)
        out[:, idx] = m
        return out


@WORLD_MODELS.register("transformer")
class TransformerWorldModel(WorldModel):
    zero_actions = False

    def __init__(self, dims, cfg) -> None:
        super().__init__(dims, cfg)
        D, C, T, H = self.d, dims.C, dims.T, dims.H
        cond = str(self.wc.action_conditioning)
        if cond not in ("token", "adaln", "adaln_and_token"):
            raise ValueError(f"unknown action_conditioning {cond!r}")
        self.use_token = cond in ("token", "adaln_and_token")
        self.use_adaln = cond in ("adaln", "adaln_and_token")
        p = lambda *s: nn.Parameter(torch.randn(*s) * 0.02)
        self.cat, self.prefix_pos, self.step, self.tok_type = p(C, D), p(2, D), p(H, D), p(4, D)
        self.prefix_in, self.block_in = nn.Linear(D, D), nn.Linear(D, D)
        self.act_in = mlp(dims.n_actions, D, D)
        self.kf_in = nn.Linear(dims.n_known_future, D)
        self.layers = nn.ModuleList([AdaLNBlock(D, int(self.wc.n_heads), float(self.wc.dropout))
                                     for _ in range(int(self.wc.n_layers))])
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, D)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        am = torch.zeros(C, dtype=torch.bool)
        am[dims.action_idx] = True
        self.register_buffer("action_cat_mask", am, persistent=False)

    def forward(self, z_ctx, actions, known_future, static, store_emb, z_in=None):
        if z_in is None:
            raise ValueError("the transformer world model needs z_in (teacher forcing) or rollout()")
        if self.zero_actions:
            actions = torch.zeros_like(actions)
        B, T, C, D = z_ctx.shape
        Hh = z_in.shape[1]
        dev = z_ctx.device
        # the encoder is causal in time, so the last week already summarises the history
        summary = torch.stack([z_ctx[:, -1], z_ctx.mean(1)], 1)  # [B,2,C,D]
        prefix = (self.prefix_in(summary) + self.cat + self.prefix_pos[:, None] + self.tok_type[0]).reshape(B, 2 * C, D)
        stat = (self.static_proj(static) + store_emb + self.tok_type[0])[:, None]
        a = self.act_in(actions)  # [B,Hh,D]
        lat = self.block_in(z_in) + self.cat + self.step[:Hh, None] + self.tok_type[1]  # [B,Hh,C,D]
        parts = [lat]
        if self.use_token:
            parts.append((a + self.step[:Hh] + self.tok_type[2])[:, :, None])
        parts.append((self.kf_in(known_future) + self.step[:Hh] + self.tok_type[3])[:, :, None])
        blocks = torch.cat(parts, 2)  # [B,Hh,C+e,D]
        nb = blocks.shape[2]
        x = torch.cat([prefix, stat, blocks.reshape(B, Hh * nb, D)], 1)
        n_pre = 2 * C + 1
        block_id = torch.cat([torch.full((n_pre,), -1, device=dev), torch.arange(Hh, device=dev).repeat_interleave(nb)])
        mask = block_id[None, :] > block_id[:, None]  # True = blocked: no attention to later blocks
        cond = mod_idx = None
        if self.use_adaln:  # AdaLN only on the tokens of the action categories in each horizon block
            cats = torch.nonzero(self.action_cat_mask).flatten()  # [nA]
            mod_idx = (n_pre + torch.arange(Hh, device=dev)[:, None] * nb + cats[None, :]).flatten()
            cond = a[:, :, None].expand(B, Hh, len(cats), D).reshape(B, -1, D)
        for layer in self.layers:
            x = layer(x, mask, cond, mod_idx)
        out = self.norm(x[:, n_pre:]).reshape(B, Hh, nb, D)[:, :, :C]
        return z_in + self.out(out)
