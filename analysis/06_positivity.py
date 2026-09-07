"""Phase 2, item 6 - positivity / overlap check.

Question: given the pre-window state, how separable are hours where the person
acts from hours where they don't? If a model can call it with high AUC, there
is no common support and no causal contrast to learn.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, write
from panel import build_panel

EVENT_COLS = {
    "exercise_any": "ev_exercise_any",
    "exercise_vigorous": "ev_exercise_vigorous",
    "walk_only": "ev_walk",
    "mindfulness": "ev_mindfulness",
}
FEAT_CH = ["bpm", "steps", "calories"]
W = 4  # hours of pre-window used to build features


def build_xy(p, event_col, W=W):
    """One row per eligible decision hour. Pre-window must be 100% complete on
    FEAT_CH so that no imputation is needed anywhere."""
    X, y, g, ts = [], [], [], []
    for pid, gg in p.groupby("id", sort=True):
        gg = gg.sort_values("ts")
        A = gg[FEAT_CH].to_numpy(dtype=float)
        ok = ~np.isnan(A).any(axis=1)
        ev = gg[event_col].to_numpy().astype(bool)
        hod = gg["hour"].to_numpy(dtype=float)
        dow = gg["dow"].to_numpy(dtype=float)
        n = len(gg)
        cok = np.concatenate(([0], np.cumsum(ok)))
        # index of most recent prior event, for a "time since last action" feature
        last_ev = np.full(n, -10 ** 6)
        cur = -10 ** 6
        for i in range(n):
            last_ev[i] = cur
            if ev[i]:
                cur = i
        for t in range(W, n):
            if cok[t] - cok[t - W] != W:      # pre-window not fully observed
                continue
            win = A[t - W:t]
            f = np.concatenate([
                win.mean(axis=0), win.std(axis=0), win[-1], win.max(axis=0), win.min(axis=0),
                [win[-1, 0] - win[0, 0], win[:, 1].sum()],
                [np.sin(2 * np.pi * hod[t] / 24), np.cos(2 * np.pi * hod[t] / 24), dow[t]],
                [min(t - last_ev[t], 24 * 14)],
            ])
            X.append(f); y.append(int(ev[t])); g.append(pid); ts.append(gg["ts"].iloc[t])
    names = ([f"{c}_mean" for c in FEAT_CH] + [f"{c}_std" for c in FEAT_CH]
             + [f"{c}_last" for c in FEAT_CH] + [f"{c}_max" for c in FEAT_CH]
             + [f"{c}_min" for c in FEAT_CH]
             + ["bpm_delta", "steps_sum", "hod_sin", "hod_cos", "dow", "hours_since_last_event"])
    return np.array(X), np.array(y), np.array(g), np.array(ts), names


def grouped_auc(X, y, g, model):
    oof = np.full(len(y), np.nan)
    n_groups = len(np.unique(g))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    for tr, te in gkf.split(X, y, g):
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        m = model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    auc = roc_auc_score(y[ok], oof[ok]) if len(np.unique(y[ok])) > 1 else np.nan
    return auc, oof


def within_participant_auc(y, g, oof):
    """AUC computed inside each participant, then pooled by weight. Between-person
    differences cannot inflate this one."""
    out = []
    for pid in np.unique(g):
        m = (g == pid) & ~np.isnan(oof)
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        out.append((pid, roc_auc_score(y[m], oof[m]), int(m.sum()), int(y[m].sum())))
    return pd.DataFrame(out, columns=["id", "auc", "n", "n_pos"])


def main():
    p, _ = build_panel()
    p = p.sort_values(["id", "ts"]).reset_index(drop=True)

    summary, hists, wp_all = [], [], []
    for name, col in EVENT_COLS.items():
        X, y, g, ts, names = build_xy(p, col)
        if y.sum() < 20:
            summary.append({"event_type": name, "n_rows": len(y), "n_pos": int(y.sum()),
                            "base_rate": round(float(y.mean()), 5),
                            "auc_logreg": np.nan, "auc_gbm": np.nan,
                            "note": "too few positives to fit"})
            continue
        auc_lr, oof_lr = grouped_auc(X, y, g, lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)))
        auc_gb, oof_gb = grouped_auc(X, y, g, lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=0))
        wp = within_participant_auc(y, g, oof_gb)
        wp.insert(0, "event_type", name)
        wp_all.append(wp)

        # overlap of the propensity distributions
        pos, neg = oof_gb[(y == 1) & ~np.isnan(oof_gb)], oof_gb[(y == 0) & ~np.isnan(oof_gb)]
        edges = np.linspace(0, max(1e-9, float(np.nanmax(oof_gb))), 41)
        hp, _ = np.histogram(pos, bins=edges)
        hn, _ = np.histogram(neg, bins=edges)
        for i in range(len(edges) - 1):
            hists.append({"event_type": name, "bin_lo": edges[i], "bin_hi": edges[i + 1],
                          "n_acted": int(hp[i]), "n_not_acted": int(hn[i])})
        # common support: fraction of acted rows whose propensity is inside the
        # 1st-99th percentile range of the not-acted rows, and vice versa
        lo, hi = np.percentile(neg, [1, 99])
        overlap_pos = float(((pos >= lo) & (pos <= hi)).mean())
        lo2, hi2 = np.percentile(pos, [1, 99])
        overlap_neg = float(((neg >= lo2) & (neg <= hi2)).mean())

        summary.append({
            "event_type": name, "n_rows": len(y), "n_pos": int(y.sum()),
            "base_rate": round(float(y.mean()), 5),
            "auc_logreg": round(float(auc_lr), 4), "auc_gbm": round(float(auc_gb), 4),
            "auc_within_participant_median": round(float(wp.auc.median()), 4) if len(wp) else np.nan,
            "n_participants_scored": len(wp),
            "frac_acted_in_notacted_support": round(overlap_pos, 4),
            "frac_notacted_in_acted_support": round(overlap_neg, 4),
            "note": "",
        })
        print(f"{name:20s} n={len(y):7d} pos={int(y.sum()):5d} base={y.mean():.4f} "
              f"AUC_lr={auc_lr:.3f} AUC_gbm={auc_gb:.3f} "
              f"within-person median AUC={wp.auc.median() if len(wp) else float('nan'):.3f}")

    s = write(pd.DataFrame(summary), "06_positivity_summary.csv")
    write(pd.DataFrame(hists), "06_propensity_histograms.csv")
    if wp_all:
        write(pd.concat(wp_all, ignore_index=True), "06_positivity_within_participant.csv")
    print("\n" + s.to_string(index=False))

    # static plot for the repo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = pd.DataFrame(hists)
    evs = H.event_type.unique()
    fig, axes = plt.subplots(1, len(evs), figsize=(4.2 * len(evs), 3.4), squeeze=False)
    for ax, ev in zip(axes[0], evs):
        h = H[H.event_type == ev]
        c = (h.bin_lo + h.bin_hi) / 2
        w = (h.bin_hi - h.bin_lo).iloc[0]
        ax.bar(c, h.n_not_acted / max(h.n_not_acted.sum(), 1), width=w, alpha=.55, label="not acted")
        ax.bar(c, h.n_acted / max(h.n_acted.sum(), 1), width=w, alpha=.55, label="acted")
        a = s.loc[s.event_type == ev, "auc_gbm"].iloc[0]
        ax.set_title(f"{ev}\nGBM AUC={a}")
        ax.set_xlabel("estimated propensity"); ax.set_yscale("log")
    axes[0][0].set_ylabel("density (log)"); axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "06_propensity_overlap.png", dpi=130)
    print("wrote", OUT / "06_propensity_overlap.png")


if __name__ == "__main__":
    main()
