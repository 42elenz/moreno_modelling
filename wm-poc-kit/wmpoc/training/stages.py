"""Training stages: pretrain → world_model → outcome_heads → joint_finetune.

Every stage uses the same `fit` loop: AdamW, warmup + cosine LR, grad clipping, early
stopping on the val loss (val weeks × training stores; cold-start stores are never used for
model selection), best-weights restore and a CSV log.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import math
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor, nn

from wmpoc.data.windows import DataBundle, TokenBatch
from wmpoc.models.system import System
from wmpoc.training.losses import latent_loss

STAGES = ("pretrain", "world_model", "outcome_heads", "joint_finetune")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class StageContext:
    def __init__(self, system: System, data: DataBundle, cfg, run_dir: Path, device: torch.device) -> None:
        self.system, self.data, self.cfg, self.run_dir, self.device = system, data, cfg, run_dir, device
        self.train = data.splits.windows("train")
        self.val = data.splits.windows("val", stores="train")
        self.bs = int(cfg.training.batch_size)
        self.seed = int(cfg.experiment.seed)
        wc = cfg.model.world_model
        w = wc.category_loss_weights
        default = float(w.get("default", 1.0))
        self.cat_w = torch.tensor([float(w.get(c, default)) for c in data.dims.categories], device=device)
        self.latent_kind = str(cfg.model.encoder.jepa.loss)
        self.cache: dict[str, tuple[Tensor, ...]] | None = None

    def build_cache(self) -> None:
        """Precompute (z_ctx, z0, targets) of the frozen encoder for every train / val window.

        Exactly what `System.latents` returns in eval mode, so cached and uncached training match.
        """
        self.system.encoder.eval()
        self.cache = {}
        with torch.no_grad():
            for which, (s, o) in (("train", self.train), ("val", self.val)):
                parts = [self.system.latents(b.to(self.device))
                         for b in self.data.ds.iterate(s, o, 512, shuffle=False)]
                self.cache[which] = tuple(torch.cat(p) for p in zip(*parts))

    def clear_cache(self) -> None:
        self.cache = None

    def batches(self, which: str, shuffle: bool, epoch: int = 0) -> Iterable[TokenBatch]:
        s, o = self.train if which == "train" else self.val
        rng = np.random.default_rng(self.seed * 7919 + epoch)
        order = rng.permutation(len(s)) if shuffle else np.arange(len(s))
        for i in range(0, len(s), self.bs):
            j = order[i:i + self.bs]
            b = self.data.ds.batch(s[j], o[j])
            if self.cache is not None:
                b.cached = tuple(c[torch.as_tensor(j)] for c in self.cache[which])
            yield b.to(self.device)

    def n_steps(self) -> int:
        return math.ceil(len(self.train[0]) / self.bs)


def fit(ctx: StageContext, name: str, train_modules: list[nn.Module], frozen_modules: list[nn.Module],
        step_fn: Callable[[TokenBatch], dict[str, Tensor]], epochs: int, lr: float, warmup_frac: float = 0.0,
        after_step: Callable[[int, int], None] | None = None, val_context=None) -> list[dict]:
    cfg = ctx.cfg
    params = [p for m in train_modules for p in m.parameters() if p.requires_grad]
    if not params or epochs <= 0:
        _log(f"{name}: nothing to train")
        return []
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(cfg.training.weight_decay))
    total = epochs * ctx.n_steps()
    warm = int(warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    es = cfg.training.early_stopping
    patience = int(es.patience)
    clip = float(cfg.training.grad_clip)
    log_every = int(cfg.training.log_every_steps)
    best, best_state, bad, step = float("inf"), None, 0, 0
    history: list[dict] = []
    for epoch in range(epochs):
        for m in train_modules:
            m.train()
        for m in frozen_modules:
            m.eval()
        sums: dict[str, float] = {}
        n = 0
        for batch in ctx.batches("train", shuffle=True, epoch=epoch):
            out = step_fn(batch)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            sched.step()
            if after_step is not None:
                after_step(step, total)
            step += 1
            n += 1
            for k, v in out.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            if log_every > 0 and step % log_every == 0:
                _log(f"{name} step {step}/{total} loss {float(out['loss'].detach()):.4f}")
        for m in train_modules + frozen_modules:
            m.eval()
        vs: dict[str, float] = {}
        nv = 0
        with torch.no_grad(), (val_context() if val_context else contextlib.nullcontext()):
            for batch in ctx.batches("val", shuffle=False):
                out = step_fn(batch)
                for k, v in out.items():
                    vs[k] = vs.get(k, 0.0) + float(v)
                nv += 1
        row = {"stage": name, "epoch": epoch + 1, "step": step, "lr": sched.get_last_lr()[0]}
        row.update({f"train_{k}": v / max(n, 1) for k, v in sums.items()})
        row.update({f"val_{k}": v / max(nv, 1) for k, v in vs.items()})
        history.append(row)
        val = row.get("val_loss", float("nan"))
        _log(f"{name} epoch {epoch + 1}/{epochs} train {row.get('train_loss', float('nan')):.4f} val {val:.4f}")
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = [copy.deepcopy(m.state_dict()) for m in train_modules]
        else:
            bad += 1
            if bad >= patience:
                _log(f"{name}: early stop at epoch {epoch + 1}")
                break
    if best_state is not None:
        for m, sd in zip(train_modules, best_state):
            m.load_state_dict(sd)
    _write_log(ctx.run_dir / "logs" / f"{name}.csv", history)
    return history


def _write_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("stage", "epoch", "step", "lr"), k))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------- stage 1
def run_pretrain(ctx: StageContext) -> list[dict]:
    sc = ctx.cfg.training.stages.pretrain
    enc = ctx.system.encoder
    if not enc.requires_pretraining:
        _log(f"pretrain: encoder {ctx.cfg.model.encoder.type} needs no pretraining, skipped")
        return []
    return fit(ctx, "pretrain", [enc], [], lambda b: enc.pretrain_step(b.ctx_window()), int(sc.epochs),
               float(sc.lr), float(sc.get("warmup_frac", 0.0)), after_step=enc.after_optimizer_step,
               val_context=getattr(getattr(enc, "masks", None), "fixed", None))


# ---------------------------------------------------------------------------- stage 2
def latent_terms(ctx: StageContext, k: int, batch: TokenBatch, z_ctx: Tensor, z0: Tensor, tgt: Tensor) -> dict[str, Tensor]:
    wc = ctx.cfg.model.world_model
    sys_ = ctx.system
    w = batch.weights[:, :, None] * ctx.cat_w[None, None]
    out: dict[str, Tensor] = {}
    pred_tf = sys_.teacher_forced(k, batch, z_ctx, z0, tgt)
    out["latent_tf"] = latent_loss(pred_tf, tgt, w, ctx.latent_kind)
    loss = float(wc.teacher_forcing_weight) * out["latent_tf"]
    R = min(int(wc.rollout_steps), ctx.data.dims.H)
    if float(wc.rollout_weight) > 0 and R > 0:
        pred_ro = sys_.rollout(k, batch, z_ctx, z0, R)
        out["latent_rollout"] = latent_loss(pred_ro, tgt[:, :R], w[:, :R], ctx.latent_kind)
        loss = loss + float(wc.rollout_weight) * out["latent_rollout"]
    out["latent"] = loss
    out["_pred_tf"] = pred_tf
    return out


def wm_step(ctx: StageContext, k: int, enc_trainable: bool) -> Callable[[TokenBatch], dict[str, Tensor]]:
    sys_ = ctx.system
    aux = float(ctx.cfg.model.world_model.outcome_aux_weight)

    def step(batch: TokenBatch) -> dict[str, Tensor]:
        z_ctx, z0, tgt = sys_.latents(batch, enc_grad=enc_trainable)
        if not sys_.latent_wm:  # direct: trained through the heads
            wm, se = sys_.wm_inputs(k, batch)
            head = sys_.head(k)
            loss = head.loss(head(wm(z_ctx, batch.actions, batch.known_future, batch.static, se)), batch)
            return {"loss": loss, "outcome": loss.detach()}
        out = latent_terms(ctx, k, batch, z_ctx, z0, tgt)
        pred_tf = out.pop("_pred_tf")
        loss = out["latent"]
        if enc_trainable:
            # A trainable encoder whose targets are its own detached outputs would collapse under a
            # pure latent loss; the outcome loss anchors it to the data.
            head = sys_.heads[0]
            real = sys_.encoder.encode(batch.target_window(), sys_.target_drop)[:, -ctx.data.dims.H:]
            o = head.loss(head(real), batch) + head.loss(head(pred_tf), batch)
            out["outcome"] = o.detach()
            loss = loss + aux * o
        out = {k_: v.detach() for k_, v in out.items()}
        out["loss"] = loss
        return out

    return step


def run_world_model(ctx: StageContext) -> list[dict]:
    sc = ctx.cfg.training.stages.world_model
    sys_ = ctx.system
    enc = sys_.encoder
    enc_trainable = (not bool(sc.freeze_encoder)) and enc.trainable
    hist: list[dict] = []
    for k in range(sys_.K):
        torch.manual_seed(int(ctx.cfg.experiment.seed) * 1000 + 31 * (k + 1))
        train = [sys_.world_models[k]]
        if not sys_.latent_wm:
            train.append(sys_.head(k))
        # the encoder trains end to end with the first member only; later members see it frozen
        et = enc_trainable and k == 0
        if et:
            train += [enc, sys_.heads[0]]
        frozen = [] if et else [enc]
        if et:
            ctx.clear_cache()
        elif ctx.cache is None:
            ctx.build_cache()
        h = fit(ctx, f"world_model_m{k}", train, frozen, wm_step(ctx, k, et), int(sc.epochs), float(sc.lr))
        for r in h:
            r["member"] = k
        hist += h
    ctx.clear_cache()
    return hist


# ---------------------------------------------------------------------------- stage 3
def run_outcome_heads(ctx: StageContext) -> list[dict]:
    sc = ctx.cfg.training.stages.outcome_heads
    sys_ = ctx.system
    if not sys_.latent_wm:
        _log("outcome_heads: direct world model heads were trained with it in stage 2, skipped")
        return []
    mode = str(ctx.cfg.model.outcome_heads.train_on)
    if mode not in ("real_embeddings", "predicted_embeddings", "both"):
        raise ValueError(f"unknown train_on {mode!r}")
    head = sys_.heads[0]
    rng = np.random.default_rng(int(ctx.cfg.experiment.seed))

    def step(batch: TokenBatch) -> dict[str, Tensor]:
        z_ctx, z0, tgt = sys_.latents(batch)
        out: dict[str, Tensor] = {}
        l_real = head.loss(head(tgt), batch)
        with torch.no_grad():
            k = int(rng.integers(sys_.K)) if head.training else 0
            pred = sys_.rollout(k, batch, z_ctx, z0)
        l_pred = head.loss(head(pred), batch)
        out["real"], out["predicted"] = l_real.detach(), l_pred.detach()
        out["loss"] = {"real_embeddings": l_real, "predicted_embeddings": l_pred, "both": l_real + l_pred}[mode]
        return out

    ctx.build_cache()
    h = fit(ctx, "outcome_heads", [head], [sys_.encoder, sys_.world_models], step, int(sc.epochs), float(sc.lr))
    ctx.clear_cache()
    return h


# ---------------------------------------------------------------------------- stage 4
def run_joint_finetune(ctx: StageContext) -> list[dict]:
    sc = ctx.cfg.training.stages.joint_finetune
    sys_ = ctx.system
    lw = sc.loss_weights
    enc_t = sys_.encoder.trainable

    def step(batch: TokenBatch) -> dict[str, Tensor]:
        z_ctx, z0, tgt = sys_.latents(batch, enc_grad=enc_t)
        lat = z_ctx.new_zeros(())
        outc = z_ctx.new_zeros(())
        for k in range(sys_.K):
            wm, se = sys_.wm_inputs(k, batch)
            head = sys_.head(k)
            if sys_.latent_wm:
                lat = lat + latent_terms(ctx, k, batch, z_ctx, z0, tgt)["latent"]
                pred = sys_.rollout(k, batch, z_ctx, z0)
            else:
                pred = wm(z_ctx, batch.actions, batch.known_future, batch.static, se)
            outc = outc + head.loss(head(pred), batch)
        loss = (float(lw.latent) * lat + float(lw.outcome) * outc) / sys_.K
        return {"loss": loss, "latent": lat.detach() / sys_.K, "outcome": outc.detach() / sys_.K}

    train = [sys_.world_models, sys_.heads] + ([sys_.encoder] if enc_t else [])
    ctx.clear_cache() if enc_t else ctx.build_cache()
    return fit(ctx, "joint_finetune", train, [] if enc_t else [sys_.encoder], step, int(sc.epochs), float(sc.lr))


RUNNERS = {"pretrain": run_pretrain, "world_model": run_world_model, "outcome_heads": run_outcome_heads,
           "joint_finetune": run_joint_finetune}
