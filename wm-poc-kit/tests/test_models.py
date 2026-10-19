import copy

import numpy as np
import pytest
import torch

from wmpoc.models.system import System
from wmpoc.registry import ENCODERS, HEADS, WORLD_MODELS
from wmpoc.training.stages import StageContext, latent_terms, wm_step
from tests.conftest import tiny_cfg


def _batch(data, n=6):
    s, o = data.splits.windows("train")
    idx = np.linspace(0, len(s) - 1, n).astype(int)
    return data.ds.batch(s[idx], o[idx])


def test_registries_complete():
    assert set(ENCODERS.keys()) >= {"jepa", "masked_ae", "supervised", "linear", "random_frozen"}
    assert set(WORLD_MODELS.keys()) >= {"transformer", "gru", "additive", "direct", "no_action"}
    assert "mlp" in HEADS


@pytest.mark.parametrize("enc", ["jepa", "masked_ae", "supervised", "linear", "random_frozen"])
@pytest.mark.parametrize("wm", ["transformer", "gru", "additive", "direct", "no_action", "randomized_effect"])
def test_shapes_every_combination(data, enc, wm):
    cfg = tiny_cfg(**{"model.encoder.type": enc, "model.world_model.type": wm, "model.world_model.ensemble_size": 2})
    sys_ = System(data.dims, cfg, data.norm)
    b = _batch(data)
    B, T, C, D, H = b.B, data.dims.T, data.dims.C, int(cfg.model.d_model), data.dims.H
    z_ctx, z0, tgt = sys_.latents(b)
    assert z_ctx.shape == (B, T, C, D) and z0.shape == (B, C, D) and tgt.shape == (B, H, C, D)
    for k in range(sys_.K):
        ro = sys_.rollout(k, b, z_ctx, z0)
        assert ro.shape == (B, H, C, D)
        preds = sys_.head(k)(ro)
        assert all(v.shape == (B, H, C) for v in preds.values())
        assert torch.isfinite(sys_.head(k).loss(preds, b))
    out = sys_.predict(b)
    assert out["sales"].shape == (sys_.K, B, H, C) and out["margin"].shape == (sys_.K, B, H, C)
    if sys_.encoder.requires_pretraining:
        r = sys_.encoder.pretrain_step(b.ctx_window())
        assert torch.isfinite(r["loss"])
    # one stage-2 training step runs and produces finite gradients
    ctx = StageContext(sys_, data, cfg, run_dir=None, device="cpu")
    loss = wm_step(ctx, 0, enc_trainable=(enc == "supervised"))(b)["loss"]
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("wm", ["transformer", "gru", "additive", "no_action", "randomized_effect"])
def test_block_causal(data, wm):
    """Perturbing step-h inputs (latent, action, known future) leaves earlier steps unchanged."""
    cfg = tiny_cfg(**{"model.world_model.type": wm, "model.encoder.type": "linear"})
    sys_ = System(data.dims, cfg, data.norm).eval()
    b = _batch(data)
    z_ctx, z0, tgt = sys_.latents(b)
    m, se = sys_.wm_inputs(0, b)
    # non-zero output projections so later steps are actually influenced
    for p in m.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)
    z_in = torch.cat([z0[:, None], tgt[:, :-1]], 1)
    base = m(z_ctx, b.actions, b.known_future, b.static, se, z_in)
    H = data.dims.H
    for h in range(1, H):
        z2, a2, k2 = z_in.clone(), b.actions.clone(), b.known_future.clone()
        z2[:, h:] += torch.randn_like(z2[:, h:])
        a2[:, h:] += 3.0
        k2[:, h:] += torch.randn_like(k2[:, h:])
        out = m(z_ctx, a2, k2, b.static, se, z2)
        torch.testing.assert_close(out[:, :h], base[:, :h], rtol=1e-5, atol=1e-5)
        assert not torch.allclose(out[:, h:], base[:, h:])
    ro = m.rollout(z_ctx, b.actions, b.known_future, b.static, se, H, z0)
    a3 = b.actions.clone()
    a3[:, 2:] += 3.0
    ro2 = m.rollout(z_ctx, a3, b.known_future, b.static, se, H, z0)
    torch.testing.assert_close(ro2[:, :2], ro[:, :2], rtol=1e-5, atol=1e-5)


def test_actions_matter_except_no_action(data):
    b = _batch(data)
    for wm, should_change in (("transformer", True), ("no_action", False)):
        cfg = tiny_cfg(**{"model.world_model.type": wm, "model.encoder.type": "linear"})
        sys_ = System(data.dims, cfg, data.norm).eval()
        m, se = sys_.wm_inputs(0, b)
        for p in m.parameters():
            if p.abs().sum() == 0:
                torch.nn.init.normal_(p, std=0.1)
        z_ctx, z0, _ = sys_.latents(b)
        a0 = m.rollout(z_ctx, b.actions * 0, b.known_future, b.static, se, 4, z0)
        a1 = m.rollout(z_ctx, b.actions * 0 + 3, b.known_future, b.static, se, 4, z0)
        assert (not torch.allclose(a0, a1)) == should_change


def test_jepa_target_encoder_no_grad_and_ema(data):
    cfg = tiny_cfg(**{"model.encoder.type": "jepa"})
    sys_ = System(data.dims, cfg, data.norm)
    enc = sys_.encoder.train()
    b = _batch(data, 16)
    before = copy.deepcopy(enc.target_encoder.state_dict())
    opt = torch.optim.SGD([p for p in enc.parameters() if p.requires_grad], lr=0.5)
    out = enc.pretrain_step(b.ctx_window())
    out["loss"].backward()
    assert all(p.grad is None for p in enc.target_encoder.parameters())
    assert all(not p.requires_grad for p in enc.target_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.context_encoder.parameters())
    opt.step()
    # the optimizer alone never touches the target encoder
    for k, v in enc.target_encoder.state_dict().items():
        torch.testing.assert_close(v, before[k])
    enc.after_optimizer_step(0, 10)
    changed = [k for k, v in enc.target_encoder.state_dict().items() if not torch.allclose(v, before[k])]
    assert changed, "EMA update must move the target encoder towards the context encoder"
    # target_encode is gradient-free
    assert not enc.target_encode(b.ctx_window()).requires_grad


@pytest.mark.parametrize("enc", ["jepa", "linear", "masked_ae"])
def test_target_drops_price(data, enc):
    """With target_drops_price, the price features never reach world-model targets."""
    b = _batch(data)
    price = data.dims.price_idx
    assert price, "config must define price features"
    for drop, invariant in ((True, True), (False, False)):
        cfg = tiny_cfg(**{"model.encoder.type": enc, "model.world_model.target_drops_price": drop})
        sys_ = System(data.dims, cfg, data.norm).eval()
        _, z0, tgt = sys_.latents(b)
        b2 = copy.copy(b)
        b2.future_tokens = b.future_tokens.clone()
        b2.tokens = b.tokens.clone()
        b2.future_tokens[..., price] += torch.randn_like(b2.future_tokens[..., price]) * 3
        b2.tokens[..., price] += torch.randn_like(b2.tokens[..., price]) * 3
        _, z0b, tgt2 = sys_.latents(b2)
        assert torch.allclose(tgt, tgt2, atol=1e-6) == invariant
        assert torch.allclose(z0, z0b, atol=1e-6) == invariant
        assert (sys_.target_drop == list(cfg.data.price_features)) == drop


def test_store_id_zeroed_for_cold_start(data):
    cfg = tiny_cfg()
    sys_ = System(data.dims, cfg, data.norm).eval()
    se = sys_.world_models[0].store_embedding
    ids = torch.arange(data.dims.n_stores)
    known = torch.zeros(data.dims.n_stores, dtype=torch.bool)
    known[data.splits.train_stores] = True
    e = se(ids, known)
    assert (e[~known] == 0).all() and (e[known].abs().sum(1) > 0).all()


def test_randomized_effect_learns_only_from_randomized_windows(data):
    cfg = tiny_cfg(**{"model.world_model.type": "randomized_effect", "model.encoder.type": "linear"})
    sys_ = System(data.dims, cfg, data.norm)
    s, o = data.splits.windows("train")
    b = data.ds.batch(s[:64], o[:64])
    wm, se = sys_.wm_inputs(0, b)
    for p in wm.effect.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)
    z_ctx, z0, tgt = sys_.latents(b)
    wm.eval()
    # zero plan -> zero effect; the baseline ignores the action
    zero = torch.zeros_like(b.actions)
    assert wm.action_effect(z_ctx, zero, b.known_future, b.static, se).abs().max() == 0
    assert wm.action_effect(z_ctx, zero + 2, b.known_future, b.static, se).abs().max() > 0
    # effect gradients only from windows whose horizon is fully randomized
    wm.train()
    for rnd in (False, True):
        b.randomized = torch.full_like(b.randomized, rnd)
        wm.zero_grad()
        sys_.wm_inputs(0, b)
        out = sys_.teacher_forced(0, b, z_ctx, z0, tgt)
        out.pow(2).mean().backward()
        g = sum(p.grad.abs().sum() for p in wm.effect.parameters() if p.grad is not None)
        assert (g > 0) == rnd
