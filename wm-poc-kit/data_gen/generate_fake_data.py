#!/usr/bin/env python3
"""
Synthetic retail panel for the discount world-model proof of concept.

Store x week x category data with a KNOWN bread-discount effect, so a model can be
checked against ground truth. Everything here is invented; no real Lidl data.

Mechanisms built in (the model only sees their consequences):
  * store-specific bread price sensitivity (beta), partly explained by observable
    store features (income, competitor distance, Lidl Plus share), partly hidden
  * halo: a bread discount brings extra visitors -> all categories rise a little
  * cannibalisation (rolls down) and complement (spreads up)
  * pull-forward: the weeks after a bread discount lose part of the extra volume
  * stock-outs: deep discounts in sensitive stores sell out -> sales are censored
  * confounded observational promos (more promos in holiday weeks, weak-traffic weeks
    and in price-sensitive stores) next to a randomized switchback experiment
  * unobserved local shocks: AR(1) traffic noise, competitor openings, road closures

Usage:
  python generate_fake_data.py --out data --n-stores 200 --n-weeks 156 --seed 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATS = ["bread", "rolls", "dairy", "spreads", "produce", "drinks", "snacks", "household"]
BREAD, ROLLS, SPREADS = 0, 1, 3
SHARE = np.array([0.30, 0.22, 0.55, 0.12, 0.70, 0.45, 0.25, 0.15])  # units per visitor-week
PRICE = np.array([1.29, 0.99, 0.89, 2.19, 1.49, 0.79, 1.59, 2.49])  # EUR, start of panel
COST = np.array([0.55, 0.40, 0.62, 1.30, 0.95, 0.45, 0.85, 1.60])
SPOIL = np.array([0.90, 0.95, 0.50, 0.00, 0.60, 0.00, 0.00, 0.00])  # leftover stock wasted
BUFFER = np.array([1.10, 1.10, 1.12, 1.15, 1.12, 1.15, 1.15, 1.20])  # stock = forecast * buffer
SEAS_AMP = np.array([0.03, 0.04, 0.02, 0.05, 0.10, 0.15, 0.04, 0.02])
SEAS_PEAK = np.array([45, 30, 10, 50, 28, 29, 50, 15])  # week of year with peak demand
XMAS = np.array([0.15, 0.25, 0.20, 0.45, 0.05, 0.30, 0.50, 0.10])
EASTER = np.array([0.20, 0.25, 0.15, 0.30, 0.10, 0.10, 0.15, 0.00])
TEMP_SENS = np.array([0, 0, 0, 0, 0.006, 0.025, -0.004, 0])
OWN_ELAST = np.array([0.0, 1.1, 1.0, 1.2, 0.9, 1.4, 1.5, 0.8])  # non-bread promo response
URBAN_MIX = np.array([0.05, 0.10, 0.00, -0.05, 0.05, 0.10, 0.15, -0.10])
LEVELS = np.array([0.0, 0.1, 0.2, 0.3])
PF1, PF2 = 0.30, 0.10  # pull-forward share of extra volume lost in the next 2 weeks
MEAN_BETA = 1.3
EASTER_WOY = {0: 14, 1: 13, 2: 16, 3: 15, 4: 14}  # ISO week of Easter, 2023..2027
EXPERIMENT_WINDOWS = [(30, 50), (80, 100), (110, 126), (132, 152)]  # [start, end) weeks
OPEN_HOURS = 78.0


def make_world(n_stores: int, n_weeks: int, rng: np.random.Generator, cohort_frac: float = 0.5,
               windows: list[tuple[int, int]] | None = None) -> dict:
    windows = EXPERIMENT_WINDOWS if windows is None else windows
    S, W, C = n_stores, n_weeks, len(CATS)
    n_regions = 6
    w = np.arange(W)
    year = w // 52
    woy = w % 52 + 1

    # ---------------- stores
    region = rng.integers(0, n_regions, S)
    size = rng.uniform(800, 1600, S).round(-1)
    urban = (rng.random(S) < 0.45).astype(int)
    income = rng.normal(0, 1, S).round(2)
    comp_km = np.clip(rng.lognormal(0.7, 0.6, S), 0.2, 15).round(2)
    lplus = rng.beta(3, 5, S).round(3)
    beta = (MEAN_BETA - 0.35 * income + 0.6 * (np.exp(-comp_km / 2) - 0.45)
            + 1.6 * (lplus - 0.375) + rng.normal(0, 0.35, S))
    beta = np.clip(beta, 0.3, 3.0)
    halo = np.clip(rng.normal(0.25, 0.08, S), 0.05, 0.5)
    base_ff = 6000 * (size / 1200) ** 0.8 * np.exp(0.15 * urban) * rng.lognormal(0, 0.2, S)
    affinity = np.exp(rng.normal(0, 0.12, (S, C)) + urban[:, None] * URBAN_MIX[None, :])
    cohort = rng.random(S) < cohort_frac

    # ---------------- calendar per region
    xmas = np.isin(woy, [51, 52]).astype(float) * np.where(woy == 51, 1.0, 0.6)
    easter = np.array([float(woy[i] == EASTER_WOY.get(year[i], 14)) for i in range(W)])
    newyear = (woy == 1).astype(float)
    reg_hol = np.zeros((n_regions, W))
    for r in range(n_regions):
        for y in range(int(year.max()) + 1):
            cand = [i for i in range(W) if year[i] == y and woy[i] in range(18, 45)]
            if cand:
                reg_hol[r, rng.choice(cand)] = 1.0
    temp = np.zeros((n_regions, W))
    rain = np.zeros((n_regions, W))
    for r in range(n_regions):
        off = rng.normal(0, 1.5)
        ar = 0.0
        for i in range(W):
            ar = 0.6 * ar + rng.normal(0, 2.0)
            temp[r, i] = 10 + off + 9 * np.sin(2 * np.pi * (woy[i] - 16) / 52) + ar
        rain[r] = np.where(rng.random(W) < 0.45, rng.gamma(2.0, 6.0, W), 0.0)
    temp_fc = temp + rng.normal(0, 1.5, temp.shape)
    rain_fc = np.clip(rain + rng.normal(0, 4.0, rain.shape), 0, None)

    # ---------------- unobserved local shocks
    eps = np.zeros((S, W))
    for i in range(1, W):
        eps[:, i] = 0.75 * eps[:, i - 1] + rng.normal(0, 0.04, S)
    comp_open = np.where(rng.random(S) < 0.15, rng.integers(20, max(21, W - 10), S), -1)
    comp_factor = np.ones((S, W))
    for s in np.where(comp_open >= 0)[0]:
        comp_factor[s, comp_open[s]:] = 0.88
    road_start = np.where(rng.random(S) < 0.06, rng.integers(10, max(11, W - 10), S), -1)
    road_len = rng.integers(3, 7, S)
    road_factor = np.ones((S, W))
    for s in np.where(road_start >= 0)[0]:
        road_factor[s, road_start[s]:road_start[s] + road_len[s]] = 0.75

    noise = np.exp(rng.normal(0, 0.07, (S, W, C)))
    stock_noise = np.exp(rng.normal(0, 0.05, (S, W, C)))
    small_so = np.where(rng.random((S, W, C)) < 0.10, rng.uniform(0.5, 4, (S, W, C)), 0.0)

    # ---------------- discount policy
    disc = np.zeros((S, W, C))
    randomized = np.zeros((S, W), dtype=int)
    slot = np.zeros((S, W), dtype=int)  # 1 = this week's bread discount was drawn at random
    in_window = np.zeros(W, dtype=bool)
    for a, b in windows:
        in_window[a:min(b, W)] = True
    flyer = np.zeros(W, dtype=bool)  # national bread flyer weeks, chosen near holidays
    for y in range(int(year.max()) + 1):
        yw = [i for i in range(W) if year[i] == y]
        hol = [i for i in yw if xmas[i] > 0 or easter[i] > 0]
        rest = [i for i in yw if i not in hol]
        pick = hol[:3] + list(rng.choice(rest, size=min(3, len(rest)), replace=False))
        flyer[pick] = True
    sw_off = rng.integers(0, 3, S)
    for s in range(S):
        for i in range(W):
            if cohort[s] and in_window[i]:
                randomized[s, i] = 1
                start = max(a for a, b in windows if a <= i)
                if (i - start + sw_off[s]) % 3 == 0:
                    slot[s, i] = 1
                    disc[s, i, BREAD] = rng.choice(LEVELS)
                continue
            if flyer[i]:
                disc[s, i, BREAD] = 0.2
                continue
            p = 0.07
            p *= 2.5 if (xmas[i] > 0 or easter[i] > 0) else 1.0
            p *= 1.8 if eps[s, i] < -0.03 else 1.0
            p *= np.exp(0.6 * (beta[s] - MEAN_BETA))
            if rng.random() < min(p, 0.9):
                disc[s, i, BREAD] = rng.choice([0.1, 0.2, 0.3], p=[0.6, 0.3, 0.1])
    other = rng.random((S, W, C)) < 0.05
    other[:, :, BREAD] = False
    disc = np.where(other, rng.choice([0.1, 0.2], size=(S, W, C)), disc)

    return dict(
        S=S, W=W, C=C, w=w, year=year, woy=woy, region=region, size=size, urban=urban,
        income=income, comp_km=comp_km, lplus=lplus, beta=beta, halo=halo, base_ff=base_ff,
        affinity=affinity, cohort=cohort, xmas=xmas, easter=easter, newyear=newyear,
        reg_hol=reg_hol, temp=temp, rain=rain, temp_fc=temp_fc, rain_fc=rain_fc, eps=eps,
        comp_open=comp_open, comp_factor=comp_factor, road_start=road_start,
        road_len=road_len, road_factor=road_factor, noise=noise, stock_noise=stock_noise,
        small_so=small_so, disc=disc, randomized=randomized, slot=slot, flyer=flyer, sw_off=sw_off,
    )


def simulate(m: dict, disc: np.ndarray) -> dict:
    """Deterministic given the world m. disc: [S, W, C] discount fractions."""
    S, W, C = m["S"], m["W"], m["C"]
    reg = m["region"]
    temp = m["temp"][reg]  # [S, W]
    rain = m["rain"][reg]
    rh = m["reg_hol"][reg]
    xm, ea, ny = m["xmas"][None, :], m["easter"][None, :], m["newyear"][None, :]
    db = disc[:, :, BREAD]

    hol_ff = (1 + 0.12 * xm) * (1 + 0.06 * ea) * (1 - 0.15 * ny) * (1 - 0.03 * rh)
    season_ff = np.exp(0.03 * np.cos(2 * np.pi * (m["woy"][None, :] - 50) / 52))
    known_ff = m["base_ff"][:, None] * season_ff * hol_ff * m["comp_factor"] * m["road_factor"]
    ff_no_halo = known_ff * np.exp(m["eps"]) * np.exp(-0.04 * rain / 5)
    ff = ff_no_halo * (1 + m["halo"][:, None] * db)

    woy = m["woy"][None, :, None]
    seas = np.exp(SEAS_AMP * np.cos(2 * np.pi * (woy - SEAS_PEAK) / 52))
    hol = 1 + XMAS * xm[..., None] + EASTER * ea[..., None]
    wthr = np.exp(TEMP_SENS * (temp[..., None] - 10))
    cat_base = SHARE * m["affinity"][:, None, :] * seas * hol * wthr  # [S, W, C]

    own = np.exp(OWN_ELAST * disc)
    own[:, :, BREAD] = np.exp(m["beta"][:, None] * db)
    cross = np.ones((S, W, C))
    cross[:, :, ROLLS] = np.exp(-0.9 * db)
    cross[:, :, SPREADS] = np.exp(0.7 * db)

    demand = ff[..., None] * cat_base * own * cross * m["noise"]
    bread_no_promo = ff_no_halo * cat_base[:, :, BREAD] * m["noise"][:, :, BREAD]
    extra = demand[:, :, BREAD] - bread_no_promo
    pf = np.zeros((S, W))
    pf[:, 1:] += PF1 * extra[:, :-1]
    pf[:, 2:] += PF2 * extra[:, :-2]
    demand[:, :, BREAD] = np.maximum(demand[:, :, BREAD] - pf, 0.3 * demand[:, :, BREAD])
    demand = np.round(demand)

    # store managers stock for expected demand: known calendar, average promo response
    exp_own = np.exp(OWN_ELAST * disc)
    exp_own[:, :, BREAD] = np.exp(MEAN_BETA * db)
    exp_ff = known_ff * (1 + 0.25 * db)
    forecast = exp_ff[..., None] * cat_base * exp_own * cross
    stock = np.round(forecast * BUFFER * m["stock_noise"])

    sales = np.minimum(demand, stock)
    short = np.where(demand > stock, 1 - stock / np.maximum(demand, 1), 0.0)
    so_hours = np.clip(np.where(short > 0, OPEN_HOURS * short * 1.6, m["small_so"]), 0, OPEN_HOURS)
    so_hours = np.round(so_hours * 2) / 2
    waste = np.round(SPOIL * np.maximum(stock - sales, 0))

    infl = (1 + 0.03 * m["w"] / 52)[None, :, None]
    regp = (1 + 0.02 * (m["region"] - 2.5) / 2.5)[:, None, None]
    base_price = PRICE * infl * regp
    net_price = base_price * (1 - disc)
    unit_cost = COST * infl * np.ones_like(regp)
    margin = (net_price - unit_cost) * sales - unit_cost * waste
    return dict(demand=demand, stock=stock, sales=sales, so_hours=so_hours, waste=waste,
                base_price=base_price * np.ones((S, 1, 1)), net_price=net_price,
                unit_cost=unit_cost * np.ones((1, W, 1)), margin=margin)


def counterfactuals(m: dict, start_weeks: list[int], horizon: int = 4) -> pd.DataFrame:
    """True outcomes for action sequence [d, 0, 0, 0] on bread from week t, all stores."""
    rows = []
    obs = m["disc"]
    for t in start_weeks:
        for d in LEVELS:
            dc = obs.copy()
            dc[:, t, BREAD] = d
            dc[:, t + 1:t + horizon, BREAD] = 0.0
            r = simulate(m, dc)
            sl = slice(t, t + horizon)
            bu = r["sales"][:, sl, BREAD]
            rec = {
                "store_id": np.arange(m["S"]), "week": t, "discount": d,
                "bread_units_4w": bu.sum(1), "bread_margin_4w": r["margin"][:, sl, BREAD].sum(1),
                "total_margin_4w": r["margin"][:, sl, :].sum((1, 2)),
                "bread_demand_4w": r["demand"][:, sl, BREAD].sum(1),
            }
            for k in range(horizon):
                rec[f"bread_units_w{k}"] = bu[:, k]
            rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-stores", type=int, default=200)
    ap.add_argument("--n-weeks", type=int, default=156)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cf-from-week", type=int, default=104, help="first week with ground-truth counterfactuals")
    ap.add_argument("--cohort-frac", type=float, default=0.5, help="share of stores in the switchback experiment")
    ap.add_argument("--experiment-windows", default="30-50,80-100,110-126,132-152",
                    help="comma-separated [start-end) week ranges of the experiment")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    windows = [tuple(int(x) for x in w.split("-")) for w in a.experiment_windows.split(",")]
    m = make_world(a.n_stores, a.n_weeks, rng, a.cohort_frac, windows)
    r = simulate(m, m["disc"])
    S, W, C = m["S"], m["W"], m["C"]

    # ---------------- observable tables
    pd.DataFrame({
        "store_id": np.arange(S), "region": m["region"], "size_sqm": m["size"], "urban": m["urban"],
        "income_index": m["income"], "competitor_dist_km": m["comp_km"],
        "lidl_plus_share": m["lplus"], "experiment_cohort": m["cohort"].astype(int),
    }).to_csv(out / "stores.csv", index=False)

    n_reg = m["reg_hol"].shape[0]
    hol_name = np.where(m["xmas"] > 0, "christmas", np.where(m["easter"] > 0, "easter",
                        np.where(m["newyear"] > 0, "new_year", "")))
    pd.DataFrame({
        "region": np.repeat(np.arange(n_reg), W), "week": np.tile(m["w"], n_reg),
        "iso_year": np.tile(2023 + m["year"], n_reg), "week_of_year": np.tile(m["woy"], n_reg),
        "national_holiday": np.tile(hol_name, n_reg),
        "regional_holiday": m["reg_hol"].ravel().astype(int),
        "temp_c": m["temp"].ravel().round(1), "rain_mm": m["rain"].ravel().round(1),
        "temp_forecast_c": m["temp_fc"].ravel().round(1), "rain_forecast_mm": m["rain_fc"].ravel().round(1),
        "national_bread_flyer": np.tile(m["flyer"].astype(int), n_reg),
    }).to_csv(out / "calendar.csv", index=False)

    sid, wk, cat = np.meshgrid(np.arange(S), m["w"], np.arange(C), indexing="ij")
    rnd = np.repeat(m["randomized"][:, :, None], C, axis=2)
    rnd[:, :, 1:] = 0  # only the bread action is randomized
    slt = np.repeat(m["slot"][:, :, None], C, axis=2)
    slt[:, :, 1:] = 0
    sales = pd.DataFrame({
        "store_id": sid.ravel(), "week": wk.ravel(), "category": np.array(CATS)[cat.ravel()],
        "units_sold": r["sales"].ravel().astype(int), "base_price": r["base_price"].ravel().round(3),
        "discount": m["disc"].ravel().round(2), "net_price": r["net_price"].ravel().round(3),
        "randomized": rnd.ravel(), "experiment_slot": slt.ravel(), "stock_units": r["stock"].ravel().astype(int),
        "stockout_hours": r["so_hours"].ravel(), "waste_units": r["waste"].ravel().astype(int),
        "unit_cost": r["unit_cost"].ravel().round(3), "margin_eur": r["margin"].ravel().round(2),
    })
    sales.to_csv(out / "sales.csv.gz", index=False)

    # ---------------- ground truth (evaluation only, never a model input)
    pd.DataFrame({
        "store_id": np.arange(S), "bread_beta": m["beta"].round(4), "halo": m["halo"].round(4),
        "competitor_open_week": m["comp_open"], "road_closure_start": m["road_start"],
        "road_closure_weeks": np.where(m["road_start"] >= 0, m["road_len"], 0),
    }).to_csv(out / "ground_truth" / "store_traits.csv", index=False)
    pd.DataFrame({"store_id": sid.ravel(), "week": wk.ravel(),
                  "category": np.array(CATS)[cat.ravel()],
                  "true_demand": r["demand"].ravel().astype(int),
                  "traffic_shock": np.repeat(m["eps"][:, :, None], C, 2).ravel().round(4)}
                 ).to_csv(out / "ground_truth" / "demand.csv.gz", index=False)
    starts = list(range(a.cf_from_week, W - 3))
    cf = counterfactuals(m, starts)
    cf.round(3).to_csv(out / "ground_truth" / "counterfactuals.csv.gz", index=False)

    # ---------------- sanity checks: can the effect be recovered?
    test0 = max(a.cf_from_week, W - 26)
    slot = m["slot"].astype(bool)
    bu = r["sales"][:, :, BREAD]
    db = m["disc"][:, :, BREAD]
    cfw = cf.set_index(["store_id", "week", "discount"])
    checks = {}
    for d in LEVELS[1:]:
        sel = slot & (np.arange(W)[None, :] >= test0) & (np.arange(W)[None, :] <= W - 4)
        ss, ww = np.where(sel)
        true = np.mean([cfw.loc[(s_, w_, d), "bread_units_w0"] - cfw.loc[(s_, w_, 0.0), "bread_units_w0"]
                        for s_, w_ in zip(ss, ww)])
        true4 = np.mean([cfw.loc[(s_, w_, d), "total_margin_4w"] - cfw.loc[(s_, w_, 0.0), "total_margin_4w"]
                         for s_, w_ in zip(ss, ww)])
        g1, g0 = bu[sel & np.isclose(db, d)], bu[sel & np.isclose(db, 0)]
        dim = g1.mean() - g0.mean()
        se = np.sqrt(g1.var(ddof=1) / len(g1) + g0.var(ddof=1) / len(g0))
        obs_sel = (m["randomized"] == 0) & ~m["flyer"][None, :]
        naive = bu[obs_sel & np.isclose(db, d)].mean() - bu[obs_sel & np.isclose(db, 0)].mean()
        checks[f"{int(d * 100)}%"] = {
            "true_ate_bread_units_week0": round(float(true), 1),
            "diff_in_means_randomized_test": round(float(dim), 1),
            "diff_in_means_se": round(float(se), 1),
            "naive_observational_estimate": round(float(naive), 1),
            "true_ate_total_margin_4w_eur": round(float(true4), 1),
            "n_randomized_test_slots": int((sel & np.isclose(db, d)).sum()),
        }
    share_censored = float((r["demand"][:, :, BREAD] > r["stock"][:, :, BREAD]).mean())
    manifest = {
        "seed": a.seed, "n_stores": S, "n_weeks": W, "categories": CATS,
        "discount_levels": LEVELS.tolist(), "experiment_windows": [list(w) for w in windows],
        "cohort_frac": a.cohort_frac,
        "suggested_split": {"train": [0, 104], "val": [104, 130], "test": [130, W]},
        "counterfactual_policy": "bread discount d in week t, 0 in weeks t+1..t+3, all else as observed",
        "bread_weeks_censored_share": round(share_censored, 3),
        "sanity_checks_test_period": checks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
