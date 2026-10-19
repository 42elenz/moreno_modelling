"""Does the synthetic data contain enough information to plan discounts?

python -m wmpoc.eval.data_audit --config configs/poc_tiny.yaml   ->  runs/data_audit/{report.md,audit.json}

Checks, using ground truth (so it lives in wmpoc/eval):
  1. experiment size: randomized slot windows per level and split
  2. identifiability: effects estimated from the randomized slots of the training period (store and
     week fixed effects) against the true counterfactual effects, with standard errors
  3. confounding: the same regression on observational (non-experiment) weeks, plus how the hidden
     traffic shock differs between discounted and non-discounted observational weeks
  4. headroom: how much margin the best policies gain over "never discount", and what precision a
     planner needs to capture it
  5. sample size needed for the experiment to resolve the margin effect at that precision
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wmpoc.baselines.experimental_effects import relative_effects, slot_windows
from wmpoc.config import load_config
from wmpoc.data.windows import build_data
from wmpoc.eval.ground_truth import load_ground_truth
from wmpoc.eval.report import table, to_jsonable


def _lvl(d: float) -> str:
    return f"{int(round(d * 100))}%"


def _observational_effects(p, weeks, stores, levels, H, cats):
    """FE regression on observational origin weeks (no experiment, no national flyer): confounded."""
    keep = np.zeros(p.n_stores, dtype=bool)
    keep[stores] = True
    flyer = p.known_future[:, :, p.known_future_names.index("national_bread_flyer")] > 0
    sel = (~p.randomized) & (~flyer) & keep[:, None]
    # the horizon must also be free of experiment weeks
    W = p.n_weeks
    for h in range(1, H):
        sel[:, : W - h] &= ~p.randomized[:, h:]
    s, t = np.nonzero(sel)
    ok = (t >= weeks[0]) & (t + H <= weeks[1])
    s, t = s[ok], t[ok]
    act = p.action_idx[0]
    d = p.discount[s, t, act]
    # keep only windows where later horizon weeks are not discounted (policy [d, 0, 0, 0])
    later = sum(p.discount[s, t + h, act] for h in range(1, H))
    s, t, d = s[later == 0], t[later == 0], d[later == 0]
    cat_mask = np.ones(len(p.categories), dtype=bool)
    if cats == "other":
        cat_mask[p.action_idx] = False
    elif cats == "bread":
        cat_mask[:] = False
        cat_mask[p.action_idx] = True
    y = np.log(sum(p.margin[s, t + h][:, cat_mask].sum(-1) for h in range(H)))
    lvl = np.array([int(np.argmin(np.abs(np.asarray(levels) - x))) for x in d])
    us, si = np.unique(s, return_inverse=True)
    ut, ti = np.unique(t, return_inverse=True)
    L = len(levels)
    X = np.zeros((len(y), (L - 1) + len(us) + len(ut) - 1))
    for j in range(1, L):
        X[lvl == j, j - 1] = 1.0
    X[np.arange(len(y)), (L - 1) + si] = 1.0
    wk = ti > 0
    X[np.nonzero(wk)[0], (L - 1) + len(us) + ti[wk] - 1] = 1.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rel = np.exp(np.concatenate([[0.0], beta[: L - 1]])) - 1
    return rel, np.array([(lvl == j).sum() for j in range(L)]), s, t, d


def audit(cfg) -> dict:
    data = build_data(cfg)
    gt = load_ground_truth(cfg.data.dir)
    p, sp, H = data.panel, data.splits, data.dims.H
    levels = [float(x) for x in cfg.planning.discount_levels]
    out: dict = {}

    # 1. experiment size
    size = {}
    for split in ("train", "val", "test"):
        a, b = sp.ranges[split]
        stores = sp.train_stores if split == "train" else np.arange(p.n_stores)
        s, t = slot_windows(p, (a, b), stores, H)
        d = p.discount[s, t, p.action_idx[0]]
        size[split] = {_lvl(x): int(np.isclose(d, x).sum()) for x in levels}
        size[split]["share_of_windows"] = float(len(s) / max(len(sp.windows(split, "train" if split == "train" else "all")[0]), 1))
    out["experiment_size"] = size

    # true effects (relative, averaged over all counterfactual store-weeks, val + test)
    cf = gt.counterfactuals
    piv = cf.pivot_table(index=["store_id", "week"], columns="discount", values=["total_margin_4w", "bread_margin_4w"])
    true = {}
    for x in levels[1:]:
        tot0, tot = piv[("total_margin_4w", 0.0)], piv[("total_margin_4w", x)]
        br0, br = piv[("bread_margin_4w", 0.0)], piv[("bread_margin_4w", x)]
        true[_lvl(x)] = {"bread": float((br / br0 - 1).mean()), "other": float(((tot - br) / (tot0 - br0) - 1).mean()),
                         "total": float((tot / tot0 - 1).mean()), "total_eur": float((tot - tot0).mean())}
    out["true_relative_effects"] = true

    # 2. identifiability from the training-period experiment
    ident = {}
    for cats in ("bread", "other", "total"):
        e = relative_effects(p, sp.ranges["train"], sp.train_stores, levels, H, cats)
        ident[cats] = {_lvl(x): {"estimate": float(e.rel[i]), "se": float(e.se[i]), "true": true[_lvl(x)][cats],
                                 "t_stat_true_effect": float(true[_lvl(x)][cats] / e.se[i]) if e.se[i] > 0 else None}
                       for i, x in enumerate(levels) if i > 0}
    out["experimental_estimates_train"] = ident

    # 3. confounding in observational weeks
    conf = {}
    for cats in ("bread", "other"):
        rel, n, s, t, d = _observational_effects(p, sp.ranges["train"], sp.train_stores, levels, H, cats)
        conf[cats] = {_lvl(x): {"observational_estimate": float(rel[i]), "true": true[_lvl(x)][cats], "n": int(n[i])}
                      for i, x in enumerate(levels) if i > 0}
    shock = gt.traffic_shock[p.store_ids[s], t]
    conf["traffic_shock_mean"] = {"discounted_weeks": float(shock[d > 0].mean()), "undiscounted_weeks": float(shock[d == 0].mean()),
                                  "difference_in_traffic_pct": float((np.exp(shock[d > 0].mean() - shock[d == 0].mean()) - 1) * 100)}
    out["confounding_train"] = conf

    # 4. headroom on test
    s, t = sp.windows("test", stores="all")
    c = gt.cf_arrays(p.store_ids[s], t)
    s, t, tm = s[c["valid"]], t[c["valid"]], c["total_margin_4w"][c["valid"]]
    oracle = tm.max(1).mean()
    best_const = max(range(len(levels)), key=lambda i: tm[:, i].mean())
    per_store = np.zeros(len(s))
    for u in np.unique(s):
        m = s == u
        per_store[m] = tm[m][:, tm[m].mean(0).argmax()]
    mean_margin = float(tm[:, 0].mean())
    out["headroom_test"] = {
        "mean_4w_total_margin": mean_margin,
        "regret_never_discount": float(oracle - tm[:, 0].mean()),
        "regret_best_constant_level": float(oracle - tm[:, best_const].mean()), "best_constant_level": _lvl(levels[best_const]),
        "regret_best_fixed_level_per_store": float(oracle - per_store.mean()),
        "gain_of_oracle_over_never_pct": float((oracle - tm[:, 0].mean()) / mean_margin * 100),
        "spread_of_true_10pct_effect_eur": {"p10": float(np.percentile(tm[:, 1] - tm[:, 0], 10)),
                                            "p50": float(np.median(tm[:, 1] - tm[:, 0])),
                                            "p90": float(np.percentile(tm[:, 1] - tm[:, 0], 90))},
        "share_store_weeks_where_10pct_pays": float((tm[:, 1] > tm[:, 0]).mean()),
    }

    # 5. sample size: slots needed to resolve the 10 % total-margin effect to ± (oracle gain) at 1 s.e.
    se_tot = ident["total"]["10%"]["se"] * mean_margin
    n_now = size["train"]["10%"] + size["train"]["0%"]
    target = out["headroom_test"]["regret_never_discount"]
    out["sample_size"] = {"se_total_effect_10pct_eur": float(se_tot), "target_precision_eur": float(target),
                          "slot_windows_now_0_and_10pct": int(n_now),
                          "multiplier_needed": float((se_tot / target) ** 2)}
    return out


def write(out: dict, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "audit.json").write_text(json.dumps(to_jsonable(out), indent=2))
    L = ["# Data audit", "", "Does the synthetic panel contain enough information to plan bread discounts?", ""]
    sz = out["experiment_size"]
    L += ["## 1. Experiment size (randomized slot windows, policy start week drawn at random)", "",
          table(["split"] + [k for k in sz["train"] if k != "share_of_windows"] + ["share of windows"],
                [[k] + [v[x] for x in v if x != "share_of_windows"] + [v["share_of_windows"]] for k, v in sz.items()]), ""]
    L += ["## 2. Can the training-period experiment identify the effects?", "",
          "Relative change of the 4-week margin vs 0 %, from slot windows of training weeks × training stores "
          "(store and week fixed effects), against the truth (mean over all counterfactual store-weeks).", ""]
    rows = []
    for cats, v in out["experimental_estimates_train"].items():
        for lvl, r in v.items():
            rows.append([cats, lvl, f"{r['estimate'] * 100:+.2f} %", f"± {r['se'] * 100:.2f} %", f"{r['true'] * 100:+.2f} %",
                         r["t_stat_true_effect"]])
    L += [table(["margin of", "level", "experimental estimate", "s.e.", "true", "true effect / s.e."], rows), ""]
    L += ["## 3. Confounding in the observational weeks", "",
          "Same regression on non-experiment, non-flyer weeks with policy [d, 0, 0, 0] (training period).", ""]
    rows = []
    for cats in ("bread", "other"):
        for lvl, r in out["confounding_train"][cats].items():
            rows.append([cats, lvl, f"{r['observational_estimate'] * 100:+.2f} %", f"{r['true'] * 100:+.2f} %", r["n"]])
    ts = out["confounding_train"]["traffic_shock_mean"]
    L += [table(["margin of", "level", "observational estimate", "true", "n"], rows), "",
          f"Hidden traffic in discounted vs undiscounted observational weeks: {ts['difference_in_traffic_pct']:+.2f} % "
          f"(mean AR(1) shock {ts['discounted_weeks']:+.4f} vs {ts['undiscounted_weeks']:+.4f}).", ""]
    h = out["headroom_test"]
    L += ["## 4. Headroom (test)", "",
          table(["policy", "regret € per store-week"],
                [["oracle (per store-week)", 0.0], ["best fixed level per store (needs per-store truth)", h["regret_best_fixed_level_per_store"]],
                 [f"best single level for everyone ({h['best_constant_level']})", h["regret_best_constant_level"]],
                 ["never discount", h["regret_never_discount"]]]), "",
          f"Mean 4-week total margin {h['mean_4w_total_margin']:,.0f} €; the oracle gains {h['gain_of_oracle_over_never_pct']:.2f} % over never discounting. "
          f"True effect of 10 % on 4-week total margin: p10 {h['spread_of_true_10pct_effect_eur']['p10']:+.0f} €, median "
          f"{h['spread_of_true_10pct_effect_eur']['p50']:+.0f} €, p90 {h['spread_of_true_10pct_effect_eur']['p90']:+.0f} €; "
          f"10 % pays off in {h['share_store_weeks_where_10pct_pays']:.0%} of store-weeks.", ""]
    ss = out["sample_size"]
    L += ["## 5. How much experiment would be enough?", "",
          f"The training-period experiment pins down the 10 % effect on 4-week total margin to ± {ss['se_total_effect_10pct_eur']:.0f} € (1 s.e.). "
          f"Capturing the headroom needs about ± {ss['target_precision_eur']:.0f} €, i.e. roughly {ss['multiplier_needed']:.0f}× as many "
          f"slot windows as the {ss['slot_windows_now_0_and_10pct']} at 0 % and 10 % today, for the *average* effect alone; "
          "store-level decisions need more.", ""]
    p = run_dir / "report.md"
    p.write_text("\n".join(L))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc_tiny.yaml")
    ap.add_argument("--out", default="runs/data_audit")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    print(write(audit(load_config(a.config, a.overrides)), Path(a.out)).read_text())


if __name__ == "__main__":
    main()
