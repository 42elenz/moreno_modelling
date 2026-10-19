"""How much does the structured planner's regret depend on the cross-category effect estimate?

python -m wmpoc.eval.cross_effect_sensitivity runs/poc_tiny
Re-plans with the per-level estimate, the pooled slope, zero and the true effect (diagnostic only).
"""
import sys, numpy as np, torch
from wmpoc.config import load_config
from wmpoc.data.windows import build_data
from wmpoc.models.system import System
from wmpoc.eval.ground_truth import load_ground_truth
from wmpoc.baselines.experimental_effects import relative_effects, pull_forward, RelativeEffects, slot_windows, _fe_slope
from wmpoc.planning.planner import WorldModelPredictor, StructuredPredictor, rollout_levels, choose
torch.set_num_threads(2)
run = sys.argv[1]
cfg = load_config(f"{run}/config.yaml"); d = build_data(cfg); p, sp, H = d.panel, d.splits, d.dims.H
sys_ = System(d.dims, cfg, d.norm); sys_.load_state_dict(torch.load(f"{run}/checkpoints/outcome_heads.pt", weights_only=False)["system"]); sys_.eval()
lv = [0.0, 0.1, 0.2, 0.3]
eff = relative_effects(p, sp.ranges["train"], sp.train_stores, lv, H, "other")
pf = pull_forward(p, sp.ranges["train"], sp.train_stores, lv, H)
s_, t_ = slot_windows(p, sp.ranges["train"], sp.train_stores, H)
mask = np.ones(len(p.categories), bool); mask[p.action_idx] = False
g, _ = _fe_slope(np.log(sum(p.margin[s_, t_ + h][:, mask].sum(-1) for h in range(H))), p.discount[s_, t_, p.action_idx[0]], s_, t_)
gt = load_ground_truth(cfg.data.dir)
wm = WorldModelPredictor(sys_, d)
variants = {"per-level estimate (used)": eff.rel, "pooled slope": np.exp(g * np.array(lv)) - 1,
            "zero cross effect": np.zeros(4), "TRUE cross effect (diagnostic)": np.array([0, .0056, .0115, .0176])}
for split in ["val", "test"]:
    s, t = sp.windows(split, stores="all")
    cf = gt.cf_arrays(p.store_ids[s], t); s, t = s[cf["valid"]], t[cf["valid"]]; tm = cf["total_margin_4w"][cf["valid"]]
    oracle = tm.max(1).mean()
    out = []
    for name, rel in variants.items():
        st = StructuredPredictor(wm, RelativeEffects(lv, rel, np.zeros(4), np.zeros(4)), pf)
        roll = rollout_levels(st, s, t, lv, H)
        idx, _ = choose(roll["total_margin_4w"], 1.0, "uplift")
        out.append(f"{name}: {oracle - tm[np.arange(len(idx)), idx].mean():.1f} (disc {np.mean(idx > 0):.0%})")
    print(f"{run} {split}: never {oracle - tm[:, 0].mean():.1f} | " + " | ".join(out), flush=True)
