"""Evaluation against ground truth: forecast, effect recovery, heterogeneity, action
sensitivity, planning, representation probes and censoring. The only consumer of
`wmpoc.eval.ground_truth`.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from scipy.stats import spearmanr

from wmpoc.baselines.diff_in_means import diff_in_means
from wmpoc.baselines.probes import calendar_store_features, knn_forecast, ridge_probe, static_calendar_features
from wmpoc.data.windows import DataBundle
from wmpoc.eval.ground_truth import GroundTruth, load_ground_truth
from wmpoc.planning.planner import Predictor, choose, rollout_levels


class ModelExtras(Protocol):
    def real_embedding_units(self, s: np.ndarray, t: np.ndarray) -> np.ndarray: ...  # [N,H] bread units
    def embeddings(self, s: np.ndarray, t: np.ndarray) -> np.ndarray: ...  # [N,Dz]


def wape(pred: np.ndarray, true: np.ndarray) -> float:
    den = np.abs(true).sum()
    return float(np.abs(pred - true).sum() / den) if den > 0 else float("nan")


def _lvl(d: float) -> str:
    return f"{int(round(d * 100))}%"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _forecast(pu: np.ndarray, pm: np.ndarray, au: np.ndarray, am: np.ndarray, H: int) -> dict:
    out = {f"wape_bread_h{h + 1}": wape(pu[:, h], au[:, h]) for h in range(H)}
    out["wape_bread_4w"] = wape(pu.sum(1), au.sum(1))
    out["margin_4w_wape"] = wape(pm, am)
    out["margin_4w_mae"] = float(np.abs(pm - am).mean())
    out["margin_4w_bias"] = float((pm - am).mean())
    return out


def evaluate(data: DataBundle, cfg, predictors: dict[str, Predictor], extras: ModelExtras | None = None,
             primary: str = "world_model", gt: GroundTruth | None = None) -> tuple[dict, dict]:
    """Returns (metrics, artefacts for plotting)."""
    gt = gt or load_ground_truth(cfg.data.dir)
    p, sp, H = data.panel, data.splits, data.dims.H
    b = data.dims.target_idx
    levels = [float(x) for x in cfg.planning.discount_levels]  # only tested levels have ground truth
    li = {d: i for i, d in enumerate(levels)}
    lam = float(cfg.planning.risk_aversion)
    hold = np.zeros(p.n_stores, dtype=bool)
    hold[sp.holdout_stores] = True
    metrics: dict = {}
    art: dict = {}

    for split in cfg.eval.splits:
        s, t = sp.windows(split, stores="all")
        cf = gt.cf_arrays(p.store_ids[s], t)
        s, t = s[cf["valid"]], t[cf["valid"]]
        cf = {k: v[cf["valid"]] for k, v in cf.items() if k != "valid"}
        au = np.stack([p.units[s, t + h, b] for h in range(H)], 1)
        am = sum(p.margin[s, t + h].sum(-1) for h in range(H))
        obs = {n: pr.predict(s, t, None) for n, pr in predictors.items()}
        roll = {n: rollout_levels(pr, s, t, levels, H) for n, pr in predictors.items()}
        real_u = extras.real_embedding_units(s, t) if extras is not None else None
        a, e = sp.ranges[split]
        dim_weeks = (a, e - H)
        subsets = {"all": np.ones(len(s), bool), "randomized": p.slot[s, t], "cold_start_stores": hold[s]}
        msplit: dict = {}
        art[split] = {}
        for sub in cfg.eval.subsets:
            m = subsets[sub]
            if m.sum() == 0:
                continue
            r: dict = {"n_windows": int(m.sum())}
            # 1. forecast
            r["forecast"] = {n: _forecast(o["bread_units"].mean(0)[m], o["total_margin_4w"].mean(0)[m], au[m], am[m], H)
                             for n, o in obs.items()}
            if real_u is not None and not np.isnan(real_u).all():
                r["forecast"][f"{primary}_heads_on_real_embeddings"] = {
                    f"wape_bread_h{h + 1}": wape(real_u[m, h], au[m, h]) for h in range(H)}
            # 2. effect recovery, 3. heterogeneity, 4. action sensitivity
            true_up = {d: cf["bread_units"][m, li[d], 0] - cf["bread_units"][m, 0, 0] for d in levels[1:]}
            true_mup = {d: cf["total_margin_4w"][m, li[d]] - cf["total_margin_4w"][m, 0] for d in levels[1:]}
            dim_stores = sp.holdout_stores if sub == "cold_start_stores" else None
            dim = diff_in_means(p, dim_weeks, levels[1:], dim_stores)
            naive = gt.manifest["sanity_checks_test_period"] if split == "test" else {}
            eff: dict = {}
            for d in levels[1:]:
                row = {"true_uplift_bread_w0": float(true_up[d].mean()),
                       "true_uplift_total_margin_4w": float(true_mup[d].mean()),
                       "diff_in_means": dim[_lvl(d)]["estimate"], "diff_in_means_se": dim[_lvl(d)]["se"]}
                if _lvl(d) in naive and sub != "cold_start_stores":
                    row["naive_observational"] = naive[_lvl(d)]["naive_observational_estimate"]
                for n, ro in roll.items():
                    pu = ro["bread_units"].mean(0)  # [L,N,H]
                    up = pu[li[d], m, 0] - pu[0, m, 0]
                    mup = ro["total_margin_4w"].mean(0)[li[d], m] - ro["total_margin_4w"].mean(0)[0, m]
                    row[f"{n}_uplift_bread_w0"] = float(up.mean())
                    row[f"{n}_bias"] = float(up.mean() - true_up[d].mean())
                    row[f"{n}_rmse"] = float(np.sqrt(((up - true_up[d]) ** 2).mean()))
                    row[f"{n}_uplift_total_margin_4w"] = float(mup.mean())
                    row[f"{n}_margin_bias"] = float(mup.mean() - true_mup[d].mean())
                eff[_lvl(d)] = row
            r["effect"] = eff
            het, sens = {}, {}
            d10, dmax = levels[1], levels[-1]
            stores_m = s[m]
            ust = np.unique(stores_m)
            true_store = np.array([true_up[d10][stores_m == u].mean() for u in ust])
            beta = gt.beta(p.store_ids[ust])
            for n, ro in roll.items():
                pu = ro["bread_units"].mean(0)
                up = pu[li[d10], m, 0] - pu[0, m, 0]
                pred_store = np.array([up[stores_m == u].mean() for u in ust])
                het[n] = {"spearman_vs_true_uplift": _spearman(pred_store, true_store),
                          "spearman_vs_bread_beta": _spearman(pred_store, beta), "n_stores": int(len(ust))}
                upmax = np.abs(pu[li[dmax], m, 0] - pu[0, m, 0]).mean()
                tmax = np.abs(cf["bread_units"][m, li[dmax], 0] - cf["bread_units"][m, 0, 0]).mean()
                ratio = float(upmax / tmax) if tmax > 0 else float("nan")
                sens[n] = {"ratio": ratio, "pred_abs_diff": float(upmax), "true_abs_diff": float(tmax),
                           "ignores_action": bool(ratio < float(cfg.eval.action_ignore_threshold))}
            het["true_uplift_vs_bread_beta"] = _spearman(true_store, beta)
            r["heterogeneity"], r["action_sensitivity"] = het, sens
            # 5. planning
            tm = cf["total_margin_4w"][m]
            oracle = tm.max(1)
            best = tm.argmax(1)
            pl = {"oracle": float(oracle.mean()), "never_discount": float(tm[:, 0].mean())}
            if 0.1 in li:
                pl["always_10"] = float(tm[:, li[0.1]].mean())
            pl["true_best_level_share"] = {_lvl(d): float((best == i).mean()) for i, d in enumerate(levels)}
            for n, ro in roll.items():
                idx, _ = choose(ro["total_margin_4w"][:, :, m], lam)
                val = tm[np.arange(len(idx)), idx]
                pl[n] = {"value": float(val.mean()), "regret": float((oracle - val).mean()),
                         "oracle_match_share": float((idx == best).mean()),
                         "chosen_share": {_lvl(d): float((idx == i).mean()) for i, d in enumerate(levels)}}
            for base in ("never_discount", "always_10"):
                if base in pl:
                    pl[f"{base}_regret"] = float(pl["oracle"] - pl[base])
            r["planning"] = pl
            # 7. censoring
            cens = p.stockout_hours[s, t, b][m] > 0
            r["censoring"] = {n: {"wape_bread_h1_censored": wape(o["bread_units"].mean(0)[m][cens, 0], au[m][cens, 0]),
                                  "wape_bread_h1_uncensored": wape(o["bread_units"].mean(0)[m][~cens, 0], au[m][~cens, 0]),
                                  "share_censored": float(cens.mean())} for n, o in obs.items()}
            msplit[sub] = r
            art[split][sub] = {
                "levels": levels, "true_up": {d: float(v.mean()) for d, v in true_up.items()},
                "pred_up": {n: {d: msplit[sub]["effect"][_lvl(d)][f"{n}_uplift_bread_w0"] for d in levels[1:]} for n in roll},
                "dim": {d: dim[_lvl(d)]["estimate"] for d in levels[1:]},
                "naive": {d: naive[_lvl(d)]["naive_observational_estimate"] for d in levels[1:] if _lvl(d) in naive},
                "store_true_up": true_store, "store_beta": beta,
                "store_pred_up": {n: np.array([(ro["bread_units"].mean(0)[li[d10], m, 0] - ro["bread_units"].mean(0)[0, m, 0])[stores_m == u].mean() for u in ust]) for n, ro in roll.items()},
                "planning": {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in pl.items()
                             if k in ("oracle", "never_discount", "always_10") or k in roll},
            }
        # 6. representation
        if extras is not None:
            msplit["representation"], art[split]["embedding"] = _representation(data, cfg, gt, extras, s, t)
        metrics[split] = msplit
    return metrics, art


def _representation(data: DataBundle, cfg, gt: GroundTruth, extras: ModelExtras, s: np.ndarray, t: np.ndarray):
    p, b = data.panel, data.dims.target_idx
    z = extras.embeddings(s, t)
    prev = t - 1
    sid = p.store_ids[s]
    targets = {
        "bread_beta": gt.beta(sid),
        "size_sqm": p.static[s, p.static_names.index("size_sqm")] if "size_sqm" in p.static_names else None,
        "holiday": p.holiday[s, prev].astype(float),
        "traffic_shock": gt.traffic_shock[sid, prev],
    }
    floors = {"calendar_store": calendar_store_features(p, s, prev), "static_calendar": static_calendar_features(p, s, prev)}
    probes = {}
    for name in cfg.eval.probes.targets:
        y = targets.get(name)
        if y is None:
            continue
        row = {"z": ridge_probe(z, y, s)}
        for fn, X in floors.items():
            row[fn] = ridge_probe(X, y, s)
        probes[name] = row
    # kNN: do neighbours in z-space predict next-week bread units better than static neighbours?
    k = int(cfg.eval.knn_k)
    ts, tt = data.splits.windows("train")
    zt = extras.embeddings(ts, tt)
    yt = p.units[ts, tt, b]
    true = p.units[s, t, b]
    knn = {"z": wape(knn_forecast(zt, yt, z, k), true),
           "static": wape(knn_forecast(p.static[ts], yt, p.static[s], k), true)}
    return {"probes_r2": probes, "knn_wape_next_week_bread": knn}, {"z": z, "beta": targets["bread_beta"]}
