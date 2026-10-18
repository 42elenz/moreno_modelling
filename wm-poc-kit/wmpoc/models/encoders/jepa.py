"""JEPA encoder: context encoder + EMA target encoder + small predictor.

Latent loss on masked tokens against the stop-gradient target encoder, plus a VICReg-style
variance/covariance term on the context-encoder outputs against collapse.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from wmpoc.models.backbone import Backbone, FactorizedBlock
from wmpoc.models.encoders.base import Encoder, as_window
from wmpoc.registry import ENCODERS
from wmpoc.training.ema import ema_momentum, ema_update
from wmpoc.training.losses import latent_loss, vicreg
from wmpoc.training.masks import MaskSampler


class Predictor(nn.Module):
    def __init__(self, dims, d_in: int, pcfg, enc_cfg) -> None:
        super().__init__()
        dp = int(pcfg.d_model)
        self.inp = nn.Linear(d_in, dp)
        self.mask_token = nn.Parameter(torch.zeros(dp))
        self.pos = nn.Parameter(torch.randn(dims.T, dp) * 0.02)
        self.cat = nn.Parameter(torch.randn(dims.C + 1, dp) * 0.02)
        self.blocks = nn.ModuleList([FactorizedBlock(dp, int(pcfg.n_heads), float(enc_cfg.mlp_ratio),
                                                     float(enc_cfg.dropout), causal_time=False)
                                     for _ in range(int(pcfg.n_layers))])
        self.norm = nn.LayerNorm(dp)
        self.out = nn.Linear(dp, d_in)

    def forward(self, ctx: Tensor, cell_mask: Tensor) -> Tensor:
        B, T, S, _ = ctx.shape
        m = torch.cat([cell_mask, torch.zeros(B, T, S - cell_mask.shape[2], dtype=torch.bool, device=ctx.device)], 2)
        x = torch.where(m[..., None], self.mask_token, self.inp(ctx))
        x = x + self.pos[None, :, None] + self.cat[None, None]
        for b in self.blocks:
            x = b(x)
        return self.out(self.norm(x))


@ENCODERS.register("jepa")
class JEPAEncoder(Encoder):
    requires_pretraining = True

    def __init__(self, dims, cfg) -> None:
        super().__init__(dims, cfg)
        ec = cfg.model.encoder
        self.context_encoder = Backbone(dims, self.d, ec, cfg.model.store_id_embedding)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(dims, self.d, ec.jepa.predictor, ec)
        self.masks = MaskSampler(ec.masks, dims, seed=int(cfg.experiment.seed))
        self.jc = ec.jepa

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()  # never dropout / store-ID dropout in the target
        return self

    def encode(self, batch, drop_features=()):
        return self.context_encoder(as_window(batch), self.drop_idx(drop_features))[:, :, : self.dims.C]

    def target_encode(self, batch, drop_features=()):
        with torch.no_grad():
            return self.target_encoder(as_window(batch), self.drop_idx(drop_features))[:, :, : self.dims.C]

    def pretrain_step(self, batch):
        w = as_window(batch)
        C = self.dims.C
        spec = self.masks.sample(w.tokens.shape[0], w.tokens.device)
        ctx = self.context_encoder(w, spec.drop_idx, spec.cell_mask)
        with torch.no_grad():
            tgt = self.target_encoder(w)[:, :, :C]
        pred = self.predictor(ctx, spec.cell_mask)[:, :, :C]
        lat = latent_loss(pred, tgt, spec.loss_mask.float(), str(self.jc.loss))
        out = {"latent": lat.detach()}
        loss = lat
        ac = self.jc.anticollapse
        if str(ac.type) == "vicreg":
            vis = ctx[:, :, :C][~spec.cell_mask]
            if vis.shape[0] > 4096:
                vis = vis[torch.randperm(vis.shape[0], device=vis.device)[:4096]]
            anti, var, cov = vicreg(vis, float(ac.var_weight), float(ac.cov_weight))
            loss = loss + anti
            out.update(var=var.detach(), cov=cov.detach())
        out["loss"] = loss
        return out

    def after_optimizer_step(self, step: int, total_steps: int) -> None:
        e = self.jc.ema
        ema_update(self.target_encoder, self.context_encoder, ema_momentum(step, total_steps, float(e.start), float(e.end)))
