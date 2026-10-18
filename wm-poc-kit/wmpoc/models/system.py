"""The full stack: encoder + ensemble of world models + outcome heads, built from the registries."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wmpoc.data.features import Normalizer
from wmpoc.data.windows import Dims, TokenBatch
from wmpoc.models.encoders.base import Encoder
from wmpoc.models.heads import OutcomeHead, compute_margin
from wmpoc.models.world_models.base import WorldModel
from wmpoc.registry import ENCODERS, HEADS, WORLD_MODELS


class System(nn.Module):
    def __init__(self, dims: Dims, cfg, norm: Normalizer) -> None:
        super().__init__()
        self.dims, self.cfg = dims, cfg
        seed = int(cfg.experiment.seed)
        torch.manual_seed(seed)
        self.encoder: Encoder = ENCODERS.build(str(cfg.model.encoder.type), dims, cfg)
        wc = cfg.model.world_model
        wms = []
        for k in range(int(wc.ensemble_size)):
            torch.manual_seed(seed * 1000 + 17 * (k + 1))  # independently seeded members
            wms.append(WORLD_MODELS.build(str(wc.type), dims, cfg))
        self.world_models = nn.ModuleList(wms)
        n_heads = len(wms) if wms[0].trained_on == "outcome" else 1
        torch.manual_seed(seed + 99)
        self.heads = nn.ModuleList([HEADS.build(str(cfg.model.outcome_heads.type), dims, cfg, norm) for _ in range(n_heads)])
        self.target_drop = list(cfg.data.price_features) if bool(wc.target_drops_price) else []

    @property
    def K(self) -> int:
        return len(self.world_models)

    @property
    def latent_wm(self) -> bool:
        return self.world_models[0].trained_on == "latent"

    def head(self, k: int) -> OutcomeHead:
        return self.heads[k if len(self.heads) > 1 else 0]

    def latents(self, batch: TokenBatch, enc_grad: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """z_ctx [B,T,C,D] (context encoder, with price), z0 [B,C,D] and targets [B,H,C,D] (target space)."""
        H = self.dims.H
        if batch.cached is not None and not enc_grad:
            return batch.cached
        with torch.set_grad_enabled(enc_grad and torch.is_grad_enabled()):
            z_ctx = self.encoder.encode(batch.ctx_window())
        z0 = self.encoder.target_encode(batch.ctx_window(), self.target_drop)[:, -1]
        tgt = self.encoder.target_encode(batch.target_window(), self.target_drop)[:, -H:]
        return z_ctx, z0, tgt

    def wm_inputs(self, k: int, batch: TokenBatch) -> tuple:
        wm = self.world_models[k]
        return wm, wm.embed_store(batch.store_id, batch.store_known)

    def rollout(self, k: int, batch: TokenBatch, z_ctx: Tensor, z0: Tensor, steps: int | None = None) -> Tensor:
        wm, se = self.wm_inputs(k, batch)
        return wm.rollout(z_ctx, batch.actions, batch.known_future, batch.static, se, steps or self.dims.H, z0)

    def teacher_forced(self, k: int, batch: TokenBatch, z_ctx: Tensor, z0: Tensor, tgt: Tensor) -> Tensor:
        wm, se = self.wm_inputs(k, batch)
        z_in = torch.cat([z0[:, None], tgt[:, :-1]], 1)
        return wm(z_ctx, batch.actions, batch.known_future, batch.static, se, z_in)

    @torch.no_grad()
    def predict(self, batch: TokenBatch) -> dict[str, Tensor]:
        """Per-member rollouts decoded to outcomes and margin: each value [K,B,H,C]."""
        z_ctx, z0, _ = self.latents(batch)
        outs: dict[str, list[Tensor]] = {}
        for k in range(self.K):
            dec = self.head(k).decode(self.head(k)(self.rollout(k, batch, z_ctx, z0)))
            dec["margin"] = compute_margin(dec, batch, self.dims)
            for n, v in dec.items():
                outs.setdefault(n, []).append(v)
        return {n: torch.stack(v) for n, v in outs.items()}

    @torch.no_grad()
    def decode_real(self, batch: TokenBatch) -> dict[str, Tensor]:
        """Heads applied to the real future embeddings (no world model)."""
        _, _, tgt = self.latents(batch)
        if not self.latent_wm:
            raise ValueError("direct world models have no real-embedding read-out")
        return self.heads[0].decode(self.heads[0](tgt))
