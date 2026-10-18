import numpy as np

from wmpoc.data.windows import build_data
from tests.conftest import tiny_cfg


def test_weeks_disjoint_with_gap(data):
    sp = data.splits
    H, T = sp.horizon_weeks, sp.context_weeks
    tr, va, te = (sp.target_weeks(k) for k in ("train", "val", "test"))
    assert not (tr & va) and not (va & te) and not (tr & te)
    # every window's horizon stays inside its split, so the last H weeks before a boundary
    # are never used as an origin
    for k in ("train", "val", "test"):
        a, b = sp.ranges[k]
        o = sp.origins(k)
        assert o.min() >= a and o.max() + H <= b
    assert max(tr) < sp.ranges["val"][0] and max(va) < sp.ranges["test"][0]
    # training inputs (context + horizon) never touch val/test weeks
    o = sp.origins("train")
    assert o.max() + H - 1 < sp.ranges["val"][0] and o.min() - T >= 0


def test_holdout_stores_never_in_training_batches():
    cfg = tiny_cfg(**{"data.max_stores": 40})
    d = build_data(cfg)
    hold = set(d.splits.holdout_stores.tolist())
    assert len(hold) == round(0.15 * 40)
    s, o = d.splits.windows("train")
    seen = set()
    for b in d.ds.iterate(s, o, 64, shuffle=True, rng=np.random.default_rng(0)):
        seen |= set(b.store_id.tolist())
        assert b.store_known.all()
    assert not (seen & hold)
    assert seen == set(d.splits.train_stores.tolist())
    try:
        d.splits.windows("train", stores="all")
        raise AssertionError("training windows from all stores must be refused")
    except ValueError:
        pass


def test_normalisation_fit_on_train_only(data):
    p, sp = data.panel, data.splits
    x = p.tokens[sp.train_stores, slice(*sp.ranges["train"])]
    np.testing.assert_allclose(data.norm.tok_mean, x.mean((0, 1)), rtol=1e-5)
