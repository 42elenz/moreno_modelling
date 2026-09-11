"""Feasibility figures. These are the only figures the study can produce before
a cohort exists; protocol Figures 1-8 all require data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from power import (BACKGROUND_RATES, CENTRAL_RATE, COHORT_PY,
                   N_COMPARATOR_COUNTRIES, irr_power)

FIG = Path(__file__).resolve().parent.parent / "figures"
OUT = Path(__file__).resolve().parent.parent / "outputs"
FIG.mkdir(exist_ok=True)

# Validated categorical slots 1-3 (all-pairs safe), plus chart chrome.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False,
})


def fig_power() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))

    # --- left: expected events under the null -----------------------------
    rates = np.array(BACKGROUND_RATES) * 1e5
    py_ru = COHORT_PY["ever_elite"]
    py_cmp = py_ru * N_COMPARATOR_COUNTRIES
    x = np.arange(len(rates))
    w = 0.38
    ru = py_ru * np.array(BACKGROUND_RATES)
    cm = py_cmp * np.array(BACKGROUND_RATES)
    ax1.bar(x - w / 2, ru, w, color=S1, label="Russia", zorder=3)
    ax1.bar(x + w / 2, cm, w, color=S2, label="comparator (4 pooled)", zorder=3)
    for xi, v in zip(x - w / 2, ru):
        ax1.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
    for xi, v in zip(x + w / 2, cm):
        ax1.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
    ax1.axhline(1, color=MUTED, ls="--", lw=1, zorder=2)
    ax1.text(-0.45, 1.12, "one event", ha="left", fontsize=8, color=MUTED)
    ax1.set_xticks(x, [f"{r:g}" for r in rates])
    ax1.set_xlabel("assumed background rate per 100,000 person-years")
    ax1.set_ylabel("expected fatal falls, 2010-2025")
    ax1.set_title("Expected events under the null\n(~1,950 elites per country, ever-elite follow-up)",
                  fontsize=10, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=8)

    # --- right: power vs true IRR -----------------------------------------
    irrs = np.array([1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50])
    for (name, py), col in zip(COHORT_PY.items(), [S3, S1]):
        p = [irr_power(i, py, py * N_COMPARATOR_COUNTRIES, CENTRAL_RATE) for i in irrs]
        ax2.plot(irrs, p, "-o", color=col, lw=2, ms=4.5,
                 markeredgecolor="#fcfcfb", markeredgewidth=1.2,
                 label=f"{name} ({py:,.0f} py)")
    ax2.axhline(0.8, color=MUTED, ls="--", lw=1)
    ax2.text(1.1, 0.82, "80% power", fontsize=8, color=MUTED)
    ax2.set_xscale("log")
    ax2.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("true incidence-rate ratio, Russia vs pooled comparator")
    ax2.set_ylabel("power (exact conditional test)")
    ax2.set_title("Power of the primary comparison (H1)\nbackground rate 3 per 100,000 py",
                  fontsize=10, color=INK, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG / "F1_power_h1.png", dpi=150)
    print("wrote", FIG / "F1_power_h1.png")


def fig_matched() -> None:
    m = pd.read_csv(OUT / "power_matched.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))

    for p0, col in zip(sorted(m.control_exposure_prevalence.unique()), [S1, S2, S3]):
        s = m[m.control_exposure_prevalence == p0].sort_values("n_cases")
        d = s.detectable_or_80pct.replace(np.inf, np.nan)
        ax1.plot(s.n_cases, d, "-o", color=col, lw=2, ms=4.5,
                 markeredgecolor="#fcfcfb", markeredgewidth=1.2,
                 label=f"control exposure {p0:.0%}")
        ax2.plot(s.n_cases, s.power_at_or_5, "-o", color=col, lw=2, ms=4.5,
                 markeredgecolor="#fcfcfb", markeredgewidth=1.2,
                 label=f"control exposure {p0:.0%}")
    ax1.set_xlabel("Russian fatal-fall cases available")
    ax1.set_ylabel("smallest odds ratio detectable at 80% power")
    ax1.set_title("Within-Russia matched design (H3), 4 controls per case\n"
                  "gaps = no OR reaches 80% power at that case count",
                  fontsize=10, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=8)

    ax2.axhline(0.8, color=MUTED, ls="--", lw=1)
    ax2.text(5.5, 0.82, "80% power", fontsize=8, color=MUTED)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Russian fatal-fall cases available")
    ax2.set_ylabel("power at a true odds ratio of 5")
    ax2.set_title("Power to detect OR = 5\n(a large, policy-relevant association)",
                  fontsize=10, color=INK, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG / "F2_power_h3_matched.png", dpi=150)
    print("wrote", FIG / "F2_power_h3_matched.png")


def fig_flowchart() -> None:
    """Protocol Figure 1 skeleton, with the counts that must be filled in."""
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.axis("off")
    ax.grid(False)
    steps = [
        ("Eligible cohort\ntop-100 companies x senior roles x 2010-2025", "N = ____"),
        ("Person-time accrued\nin-role and ever-elite", "PY = ____"),
        ("Deaths during follow-up", "n = ____"),
        ("Deaths with a fall mechanism", "n = ____"),
        ("Building falls (roof / building / elevated)", "n = ____"),
        ("Window or balcony falls  [PRIMARY]", "n = ____"),
    ]
    y = 0.94
    for i, (label, count) in enumerate(steps):
        col = S2 if i == len(steps) - 1 else S1
        ax.add_patch(plt.Rectangle((0.06, y - 0.105), 0.88, 0.105, transform=ax.transAxes,
                                   facecolor="#fcfcfb", edgecolor=col, lw=1.6, zorder=2))
        ax.text(0.10, y - 0.037, label, transform=ax.transAxes, fontsize=9,
                va="center", color=INK, zorder=3)
        ax.text(0.90, y - 0.037, count, transform=ax.transAxes, fontsize=9,
                va="center", ha="right", color=col, weight="bold", zorder=3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.145), xytext=(0.5, y - 0.105),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
        y -= 0.155
    ax.set_title("Figure 1 - ascertainment flow (to be completed per country)",
                 fontsize=10, color=INK, loc="left")
    ax.text(0.06, 0.02, "Counts are reported separately for RU, PL, CZ, HU, RO, KZ.\n"
                        "The stop/go checkpoint (protocol §16) is evaluated at the last two rows.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "F0_flowchart_template.png", dpi=150)
    print("wrote", FIG / "F0_flowchart_template.png")


if __name__ == "__main__":
    fig_power()
    fig_matched()
    fig_flowchart()
