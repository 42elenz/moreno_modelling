"""Masked autoencoder: same backbone and masks, reconstructs raw scaled features of masked cells."""
from __future__ import annotations

from torch import nn

from wmpoc.models.backbone import Backbone, FactorizedBlock
from wmpoc.models.encoders.base import Encoder, as_window
from wmpoc.registry import ENCODERS
from wmpoc.training.losses import outcome_elementwise, weighted_mean
from wmpoc.training.masks import MaskSampler


@ENCODERS.register("masked_ae")
class MaskedAEEncoder(Encoder):
    requires_pretraining = True

    def __init__(self, dims, cfg) -> None:
        super().__init__(dims, cfg)
        ec = cfg.model.encoder
        self.backbone = Backbone(dims, self.d, ec, cfg.model.store_id_embedding)
        self.decoder = nn.ModuleList([FactorizedBlock(self.d, int(ec.n_heads), float(ec.mlp_ratio), float(ec.dropout),
                                                      causal_time=False) for _ in range(int(ec.masked_ae.decoder.n_layers))])
        self.head = nn.Linear(self.d, dims.F)
        self.masks = MaskSampler(ec.masks, dims, seed=int(cfg.experiment.seed))
        self.loss_kind = str(ec.masked_ae.loss)

    def encode(self, batch, drop_features=()):
        return self.backbone(as_window(batch), self.drop_idx(drop_features))[:, :, : self.dims.C]

    def pretrain_step(self, batch):
        w = as_window(batch)
        spec = self.masks.sample(w.tokens.shape[0], w.tokens.device)
        x = self.backbone(w, spec.drop_idx, spec.cell_mask)
        for b in self.decoder:
            x = b(x)
        rec = self.head(x)[:, :, : self.dims.C]  # [B,T,C,F]
        err = outcome_elementwise(rec, w.tokens, self.loss_kind)
        wt = spec.loss_mask[..., None].float().expand_as(err).clone()
        if spec.kind == "price":  # every cell visible: reconstruct only the blanked price features
            feat = err.new_zeros(self.dims.F)
            feat[spec.drop_idx] = 1.0
            wt = wt * feat
        loss = weighted_mean(err, wt)
        return {"loss": loss, "recon": loss.detach()}
