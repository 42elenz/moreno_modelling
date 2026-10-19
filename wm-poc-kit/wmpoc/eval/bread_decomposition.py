"""Decompose the predicted vs true 4-week bread margin effect into price loss, volume and waste.

python -m wmpoc.eval.bread_decomposition runs/poc_tiny [val|test]
"""
import sys, numpy as np, torch
from wmpoc.config import load_config
from wmpoc.data.windows import build_data
from wmpoc.models.system import System
from wmpoc.eval.ground_truth import load_ground_truth
from wmpoc.planning.planner import plan_sequences
from wmpoc.data.features import ACTION_SCALE
torch.set_num_threads(2)
run, split = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "test")
cfg = load_config(f"{run}/config.yaml")
d = build_data(cfg); sys_ = System(d.dims, cfg, d.norm)
sys_.load_state_dict(torch.load(f"{run}/checkpoints/outcome_heads.pt", weights_only=False)["system"]); sys_.eval()
gt = load_ground_truth(cfg.data.dir)
s, t = d.splits.windows(split, stores="all")
idx = np.random.default_rng(0).choice(len(s), 800, replace=False); s, t = s[idx], t[idx]
cf = gt.cf_arrays(d.panel.store_ids[s], t)
b0 = 0
res = {}
for lv in [0.0, 0.1, 0.3]:
    b = d.ds.batch(s, t).with_actions(torch.as_tensor(plan_sequences(lv, len(s), 4), dtype=torch.float32)[:, :, None] * ACTION_SCALE)
    out = sys_.predict(b)
    res[lv] = {k: v.mean(0)[..., b0].numpy() for k, v in out.items()}  # [N,H] bread
bp = d.panel.base_price[s[:, None], t[:, None] + np.arange(4), b0]; c = d.panel.unit_cost[s[:, None], t[:, None] + np.arange(4), b0]
for lv, li in [(0.1, 1), (0.3, 3)]:
    m1, m0 = res[lv], res[0.0]
    price = -(lv * bp[:, 0] * m1["sales"][:, 0])
    vol = ((bp - c) * (m1["sales"] - m0["sales"])).sum(1)
    waste = -(c * (m1["waste_units"] - m0["waste_units"])).sum(1)
    tot = (m1["margin"] - m0["margin"]).sum(1)
    tu1, tu0 = cf["bread_units"][:, li], cf["bread_units"][:, 0]
    tprice = -(lv * bp[:, 0] * tu1[:, 0]); tvol = ((bp - c) * (tu1 - tu0)).sum(1)
    ttot = cf["bread_margin_4w"][:, li] - cf["bread_margin_4w"][:, 0]
    print(f"level {lv}: model total {tot.mean():7.1f} = price {price.mean():7.1f} + volume {vol.mean():6.1f} + waste {waste.mean():6.1f}"
          f" | truth total {ttot.mean():7.1f} = price {tprice.mean():7.1f} + volume {tvol.mean():6.1f} + waste(rest) {(ttot - tprice - tvol).mean():6.1f}")
    print("   per-week Δunits model", np.round((m1["sales"] - m0["sales"]).mean(0), 1), " truth", np.round((tu1 - tu0).mean(0), 1),
          " Δwaste units model", np.round((m1["waste_units"] - m0["waste_units"]).mean(0), 1), " Δstock model", np.round((m1["stock"] - m0["stock"]).mean(0), 1))
