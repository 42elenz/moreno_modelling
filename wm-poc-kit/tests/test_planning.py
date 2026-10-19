import numpy as np
import torch

from wmpoc.baselines.experimental_effects import pull_forward, relative_effects
from wmpoc.data.windows import build_data
from wmpoc.models.system import System
from wmpoc.planning.planner import HybridPredictor, StructuredPredictor, WorldModelPredictor, choose, plan_sequences
from tests.conftest import tiny_cfg


def test_choose_uplift_risk_ignores_shared_baseline_noise():
    rng = np.random.default_rng(0)
    effect = np.array([0.0, 5.0, -3.0])[:, None]  # level 10 % pays off
    shared = rng.normal(0, 1000, (5, 1, 7))  # large member disagreement on the baseline only
    margin = 1000 + shared + effect[None]
    idx, _ = choose(margin, 1.0, "uplift")
    assert (idx == 1).all()


def test_structured_and_hybrid_match_world_model_without_discount(data, cfg):
    sys_ = System(data.dims, cfg, data.norm).eval()
    wm = WorldModelPredictor(sys_, data)
    p, sp, H = data.panel, data.splits, data.dims.H
    lv = [0.0, 0.1, 0.2, 0.3]
    eff = relative_effects(p, sp.ranges["train"], sp.train_stores, lv, H, "other")
    pf = pull_forward(p, sp.ranges["train"], sp.train_stores, lv, H)
    s, t = sp.windows("val", stores="all")
    s, t = s[:20], t[:20]
    zero = plan_sequences(0.0, len(s), H)
    ref = wm.predict(s, t, zero)
    for pred in (HybridPredictor(wm, eff), StructuredPredictor(wm, eff, pf)):
        out = pred.predict(s, t, zero)
        np.testing.assert_allclose(out["total_margin_4w"], ref["total_margin_4w"], rtol=1e-4, atol=1e-2)
    # a discount changes later bread weeks by exactly share_h × week-0 uplift
    st = StructuredPredictor(wm, eff, pf)
    a = st.predict(s, t, plan_sequences(0.2, len(s), H))
    u0 = a["bread_units"][..., 0] - ref["bread_units"][..., 0]
    np.testing.assert_allclose(a["bread_units"][..., 1:] - ref["bread_units"][..., 1:],
                               pf.share[None, None, 1:] * u0[..., None], rtol=1e-4, atol=1e-3)


def test_experiment_finds_pull_forward_on_full_data():
    cfg = tiny_cfg(**{"data.max_stores": None})
    d = build_data(cfg)
    pf = pull_forward(d.panel, d.splits.ranges["train"], d.splits.train_stores, [0.0, 0.1, 0.2, 0.3], 4)
    assert pf.share[0] == 1.0
    assert -0.45 < pf.share[1] < -0.2 and -0.2 < pf.share[2] < -0.03  # generator: 30 % and 10 % of the extra volume
    eff = relative_effects(d.panel, d.splits.ranges["train"], d.splits.train_stores, [0.0, 0.1, 0.2, 0.3], 4, "bread")
    assert (np.diff(eff.rel) < 0).all()  # deeper discounts cost more bread margin
