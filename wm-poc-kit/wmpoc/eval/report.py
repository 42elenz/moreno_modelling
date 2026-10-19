"""metrics.json → report.md with tables and PNG plots."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# reference categorical palette, slots 1-3 (validated all-pairs), plus neutral ink for truth
BLUE, ORANGE, AQUA, VIOLET, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#8a8984"
SERIES = {"world_model": BLUE, "lightgbm": ORANGE, "world_model_hybrid": VIOLET, "world_model_structured": AQUA,
          "world_model_calibrated": "#e87ba4", "world_model_structured_pooled": "#008300"}
MODELS = ("world_model", "world_model_hybrid", "world_model_structured", "world_model_structured_pooled",
          "world_model_calibrated", "lightgbm")
SAME_BREAD = {"world_model_hybrid", "world_model_structured", "world_model_structured_pooled"}  # bread predictions identical to world_model: left out of bread-only plots


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return "–"
        if abs(x) >= 100:
            return f"{x:,.0f}"
        return f"{x:.{nd}f}"
    return str(x)


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------- plots
def plot_training_curves(run_dir: Path) -> str | None:
    logs = sorted((run_dir / "logs").glob("*.csv"))
    if not logs:
        return None
    fig, axes = plt.subplots(1, len(logs), figsize=(3.2 * len(logs), 2.6), squeeze=False)
    for ax, f in zip(axes[0], logs):
        rows = list(csv.DictReader(f.open()))
        ep = [int(r["epoch"]) for r in rows]
        for key, col, lab in (("train_loss", BLUE, "train"), ("val_loss", ORANGE, "val")):
            ys = [float(r[key]) for r in rows if r.get(key)]
            if ys:
                ax.plot(ep[: len(ys)], ys, color=col, linewidth=2, marker="o", markersize=3, label=lab)
        ax.set_title(f.stem, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        _style(ax)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p = run_dir / "plots" / "training_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p.name


def plot_uplift(run_dir: Path, a: dict, title: str) -> str:
    lv = [0.0] + list(a["true_up"].keys())
    x = [100 * d for d in lv]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(x, [0] + list(a["true_up"].values()), color=INK, linewidth=2, marker="o", markersize=6, label="truth")
    for n, up in a["pred_up"].items():
        if n in SAME_BREAD:
            continue
        ax.plot(x, [0] + [up[d] for d in lv[1:]], color=SERIES.get(n, AQUA), linewidth=2, marker="o", markersize=6, label=n)
    ax.plot(x, [0] + [a["dim"][d] for d in lv[1:]], color=AQUA, linewidth=2, marker="s", markersize=6,
            linestyle="--", label="diff in means")
    if a.get("naive"):
        ax.plot(x, [0] + [a["naive"][d] for d in lv[1:]], color=MUTED, linewidth=1.5, marker="^", markersize=6,
                linestyle=":", label="naive observational")
    ax.set_xlabel("bread discount in week 0 (%)", fontsize=8)
    ax.set_ylabel("uplift in bread units, week 0", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    p = run_dir / "plots" / "uplift_vs_truth.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p.name


def plot_store_uplift(run_dir: Path, a: dict) -> str:
    names = [n for n in a["store_pred_up"] if n not in SAME_BREAD]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.3), squeeze=False)
    t = a["store_true_up"]
    for ax, n in zip(axes[0], names):
        pr = a["store_pred_up"][n]
        ax.scatter(t, pr, s=14, color=SERIES.get(n, AQUA), edgecolor="white", linewidth=0.5)
        lo, hi = float(min(t.min(), pr.min())), float(max(t.max(), pr.max()))
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(n, fontsize=9)
        ax.set_xlabel("true per-store 10% uplift", fontsize=8)
        ax.set_ylabel("predicted", fontsize=8)
        _style(ax)
    fig.tight_layout()
    p = run_dir / "plots" / "store_uplift_pred_vs_true.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p.name


def plot_planning(run_dir: Path, pl: dict) -> str:
    keys = [k for k in ("oracle",) + MODELS + ("never_discount", "always_10") if k in pl]
    oracle = pl["oracle"]
    vals = [oracle - pl[k] for k in keys]  # regret: lower is better, one common axis
    fig, ax = plt.subplots(figsize=(5, 2.8))
    cols = [MUTED if k in ("oracle", "never_discount", "always_10") else SERIES.get(k, AQUA) for k in keys]
    bars = ax.barh(keys, vals, color=cols, height=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:,.1f}", va="center", fontsize=8, color="#52514e")
    ax.invert_yaxis()
    ax.set_xlabel("regret vs oracle, € per store-week (4-week total margin, lower is better)", fontsize=8)
    _style(ax)
    fig.tight_layout()
    p = run_dir / "plots" / "planning_regret.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p.name


def plot_embedding_pca(run_dir: Path, emb: dict) -> str:
    z = emb["z"] - emb["z"].mean(0)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    pc = z @ vt[:2].T
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=emb["beta"], cmap="Blues", s=6, linewidth=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("bread_beta (true)", fontsize=8)
    ev = (s[:2] ** 2) / (s ** 2).sum()
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=8)
    _style(ax)
    fig.tight_layout()
    p = run_dir / "plots" / "embedding_pca.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p.name


# ---------------------------------------------------------------------------- markdown
def _forecast_rows(r: dict, H: int) -> list[list]:
    rows = []
    for n, f in r["forecast"].items():
        rows.append([n] + [f.get(f"wape_bread_h{h + 1}") for h in range(H)] + [f.get("margin_4w_wape"), f.get("margin_4w_mae")])
    return rows


def write_report(run_dir: Path, metrics: dict, art: dict, cfg, plots: bool = True) -> Path:
    run_dir = Path(run_dir)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    H = int(cfg.data.window.horizon_weeks)
    meta = metrics.get("meta", {})
    L: list[str] = [f"# Run report: {cfg.experiment.name}", ""]
    L.append(f"encoder `{cfg.model.encoder.type}` · world model `{cfg.model.world_model.type}` × "
             f"{cfg.model.world_model.ensemble_size} · target_drops_price {cfg.model.world_model.target_drops_price} · "
             f"censoring `{cfg.model.outcome_heads.censoring}` · seed {cfg.experiment.seed} · "
             f"stores {meta.get('n_stores', '?')} · runtime {fmt(meta.get('runtime_s'))} s")
    L.append("")
    t = metrics.get("test", {}).get("all")
    if t:
        pl = t["planning"]
        label = {"world_model": "world model", "world_model_hybrid": "hybrid (WM bread + experimental cross effects)",
                 "world_model_structured": "structured (hybrid + experimental pull-forward)",
                 "world_model_structured_pooled": "structured, pooled cross-category slope",
                 "world_model_calibrated": "calibrated (structured, pooled slope, bread effects calibrated)", "lightgbm": "LightGBM"}
        cands = {label[n]: pl[n]["regret"] for n in MODELS if n in pl}
        cands.update({"never discount": pl.get("never_discount_regret"), "always 10%": pl.get("always_10_regret")})
        cands = {k: v for k, v in cands.items() if v is not None}
        best = min(cands, key=cands.get)
        L += ["## Headline (test, all stores)", "",
              f"Lowest planning regret: **{best}** ({fmt(cands[best], 1)} € per store-week). "
              + ", ".join(f"{k} {fmt(v, 1)}" for k, v in cands.items()) + ".", ""]
        sens = t["action_sensitivity"].get("world_model", {})
        if sens.get("ignores_action"):
            L += [f"**WARNING: the world model ignores the action** (|pred(30%)−pred(0%)| is {fmt(sens['ratio'])}× the true difference).", ""]
        e20 = metrics["test"].get("randomized", t)["effect"].get("20%", {})
        rows = []
        for n in MODELS:
            if n not in t["forecast"]:
                continue
            rows.append([n, t["forecast"][n]["wape_bread_h1"], t["forecast"][n]["margin_4w_wape"], e20.get(f"{n}_bias"),
                         t["heterogeneity"][n]["spearman_vs_true_uplift"], t["action_sensitivity"][n]["ratio"],
                         pl[n]["regret"], pl[n]["oracle_match_share"]])
        L += [table(["model", "WAPE bread h1", "WAPE margin 4w", "ATE bias 20% (randomized)", "CATE Spearman",
                     "action sensitivity", "planning regret €", "oracle match"], rows), ""]
    for split in cfg.eval.splits:
        ms = metrics.get(split, {})
        if not ms:
            continue
        L += [f"## {split}", ""]
        for sub, r in ms.items():
            if sub == "representation":
                continue
            L += [f"### {split} / {sub} ({r['n_windows']} windows)", "", "**Forecast** (WAPE of bread units per horizon step; 4-week total margin)", "",
                  table(["model"] + [f"h{h + 1}" for h in range(H)] + ["margin WAPE", "margin MAE €"], _forecast_rows(r, H)), ""]
            erows = []
            names = [n for n in MODELS if f"{n}_bias" in next(iter(r["effect"].values()))]
            for lvl, e in r["effect"].items():
                erows.append([lvl, e["true_uplift_bread_w0"]] + [e[f"{n}_uplift_bread_w0"] for n in names]
                             + [e[f"{n}_bias"] for n in names] + [e[f"{n}_rmse"] for n in names]
                             + [e["diff_in_means"], e.get("naive_observational"), e["true_uplift_total_margin_4w"]]
                             + [e[f"{n}_uplift_total_margin_4w"] for n in names])
            L += ["**Effect recovery** (bread units in week 0, action [d,0,0,0], vs ground-truth counterfactuals)", "",
                  table(["level", "true"] + [f"{n} pred" for n in names] + [f"{n} bias" for n in names]
                        + [f"{n} RMSE" for n in names] + ["diff in means", "naive obs.", "true Δ margin 4w"]
                        + [f"{n} Δ margin 4w" for n in names], erows), ""]
            if any("_uplift_other_margin_4w" in k for k in next(iter(r["effect"].values()))):
                mrows = []
                for lvl, e in r["effect"].items():
                    for n in names:
                        if f"{n}_uplift_bread_margin_4w" in e:
                            mrows.append([lvl, n, e[f"{n}_uplift_bread_margin_4w"], e["true_uplift_bread_margin_4w"],
                                          e[f"{n}_uplift_other_margin_4w"], e["true_uplift_other_margin_4w"],
                                          e[f"{n}_uplift_total_margin_4w"], e["true_uplift_total_margin_4w"]])
                L += ["**Margin effect split** (Δ 4-week margin vs 0 %, € per store-week: bread vs all other categories)", "",
                      table(["level", "model", "bread pred", "bread true", "other pred", "other true", "total pred", "total true"], mrows), ""]
            het = r["heterogeneity"]
            L += ["**Heterogeneity and action sensitivity**", "",
                  table(["model", "Spearman vs true per-store uplift", "Spearman vs bread_beta", "sensitivity ratio", "ignores action"],
                        [[n, het[n]["spearman_vs_true_uplift"], het[n]["spearman_vs_bread_beta"],
                          r["action_sensitivity"][n]["ratio"], r["action_sensitivity"][n]["ignores_action"]] for n in names]),
                  f"(true per-store uplift vs bread_beta: Spearman {fmt(het['true_uplift_vs_bread_beta'])})", ""]
            pl = r["planning"]
            prow = [["oracle", pl["oracle"], 0.0, 1.0, ""], ["never discount", pl["never_discount"], pl["never_discount_regret"], pl["true_best_level_share"].get("0%"), ""]]
            if "always_10" in pl:
                prow.append(["always 10%", pl["always_10"], pl["always_10_regret"], pl["true_best_level_share"].get("10%"), ""])
            for n in names:
                prow.append([n, pl[n]["value"], pl[n]["regret"], pl[n]["oracle_match_share"],
                             ", ".join(f"{k} {v:.0%}" for k, v in pl[n]["chosen_share"].items())])
            L += ["**Planning** (true 4-week total margin of the chosen level, € per store-week)", "",
                  table(["policy", "value €", "regret €", "oracle match", "levels chosen"], prow),
                  "true best level shares: " + ", ".join(f"{k} {v:.0%}" for k, v in pl["true_best_level_share"].items()), ""]
            L += ["**Censoring** (WAPE bread h1)", "",
                  table(["model", "censored weeks", "uncensored weeks", "share censored"],
                        [[n, c["wape_bread_h1_censored"], c["wape_bread_h1_uncensored"], c["share_censored"]]
                         for n, c in r["censoring"].items()]), ""]
        rep = ms.get("representation")
        if rep:
            L += [f"### {split} / representation", "", "Ridge probes on frozen z(t), out-of-fold R², folds grouped by store", "",
                  table(["target", "z(t)", "calendar + store one-hot", "calendar + static features"],
                        [[k, v["z"], v["calendar_store"], v["static_calendar"]] for k, v in rep["probes_r2"].items()]), "",
                  f"kNN (k={cfg.eval.knn_k}) next-week bread units, WAPE: neighbours in z-space {fmt(rep['knn_wape_next_week_bread']['z'])}, "
                  f"neighbours on static features {fmt(rep['knn_wape_next_week_bread']['static'])}", ""]
    if plots:
        imgs = []
        tc = plot_training_curves(run_dir)
        if tc:
            imgs.append(("Training curves", tc))
        ta = art.get("test", {})
        sub = ta.get("randomized") or ta.get("all")
        if sub:
            imgs.append(("Uplift vs truth (test, randomized slot weeks)", plot_uplift(run_dir, sub, "test / randomized slot weeks")))
        if ta.get("all"):
            imgs.append(("Per-store 10% uplift, predicted vs true (test)", plot_store_uplift(run_dir, ta["all"])))
            imgs.append(("Planning regret (test, all)", plot_planning(run_dir, metrics["test"]["all"]["planning"] | {
                k: v["value"] for k, v in metrics["test"]["all"]["planning"].items() if isinstance(v, dict) and "value" in v})))
        if ta.get("embedding"):
            imgs.append(("Embedding PCA coloured by bread_beta (test)", plot_embedding_pca(run_dir, ta["embedding"])))
        L += ["## Plots", ""] + [f"**{t_}**\n\n![{t_}](plots/{f})\n" for t_, f in imgs]
    out = run_dir / "report.md"
    out.write_text("\n".join(L))
    return out


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    return x


def write_metrics(run_dir: Path, metrics: dict) -> Path:
    p = Path(run_dir) / "metrics.json"
    p.write_text(json.dumps(to_jsonable(metrics), indent=2))
    return p
