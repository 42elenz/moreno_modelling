"""Ablation runner.

python -m wmpoc.ablate --config configs/poc_tiny.yaml --grid configs/ablations.yaml
    [--runs reference,enc_linear] [--seeds 0,1] [--jobs 4] [dot.overrides ...]

Every grid entry runs for every seed (`runs/ablations/<run>/seed<k>/`). Runs whose pretraining
configuration is identical share one pretraining checkpoint: the first such run pretrains, the
others start at the world-model stage from a copy of its encoder (bit-identical to pretraining
again, because every stage reseeds). Finished runs (metrics.json present) are skipped, so the
command can be re-run to resume. Writes `runs/ablations/summary.md` (`summary_<grid>.md` for other grids),
covering every entry of the grid that has finished runs, whatever `--runs` selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from wmpoc.config import apply_overrides, load_config, parse_overrides, to_dict

KEY_METRICS = [
    ("test WAPE bread h1", ("test", "all", "forecast", "world_model", "wape_bread_h1")),
    ("ATE bias 20% (test randomized)", ("test", "randomized", "effect", "20%", "world_model_bias")),
    ("CATE Spearman (test)", ("test", "all", "heterogeneity", "world_model", "spearman_vs_true_uplift")),
    ("planning regret € (test)", ("test", "all", "planning", "world_model", "regret")),
    ("planning regret € (val)", ("val", "all", "planning", "world_model", "regret")),
    ("oracle match (test)", ("test", "all", "planning", "world_model", "oracle_match_share")),
    ("action sensitivity (test)", ("test", "all", "action_sensitivity", "world_model", "ratio")),
    ("Δ margin 4w bias 10% € (test)", ("test", "all", "effect", "10%", "world_model_margin_bias")),
    ("hybrid regret € (test)", ("test", "all", "planning", "world_model_hybrid", "regret")),
    ("hybrid regret € (val)", ("val", "all", "planning", "world_model_hybrid", "regret")),
    ("structured regret € (test)", ("test", "all", "planning", "world_model_structured", "regret")),
    ("structured regret € (val)", ("val", "all", "planning", "world_model_structured", "regret")),
]


def _get(d: dict, path: tuple) -> float:
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return float("nan")
        d = d[k]
    return float("nan") if d is None or isinstance(d, dict) else float(d)


def pretrain_key(cfg) -> str | None:
    from wmpoc.registry import ENCODERS

    if not bool(cfg.training.stages.pretrain.enabled) or not ENCODERS.get(str(cfg.model.encoder.type)).requires_pretraining:
        return None
    c = to_dict(cfg)
    data = {k: v for k, v in c["data"].items() if k != "randomized_sample_weight"}
    enc = {k: v for k, v in c["model"]["encoder"].items() if k != "pooling"}  # pooling is read by world models only
    blob = {"data": data, "d_model": c["model"]["d_model"], "encoder": enc,
            "store": c["model"]["store_id_embedding"], "pretrain": c["training"]["stages"]["pretrain"],
            "batch": c["training"]["batch_size"], "opt": [c["training"]["weight_decay"], c["training"]["grad_clip"]],
            "es": c["training"]["early_stopping"], "seed": c["experiment"]["seed"]}
    return hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]


def _cli_overrides(ov: dict) -> list[str]:
    return [f"{k}={json.dumps(v)}" for k, v in ov.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--runs", default=None, help="comma-separated subset of grid entries")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds (default: grid seeds)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per job")
    ap.add_argument("--out", default="runs/ablations")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()

    grid = OmegaConf.to_container(OmegaConf.load(a.grid))
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(grid.get("seeds", [0]))
    names = a.runs.split(",") if a.runs else list(grid["runs"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    glob_ov = parse_overrides(a.overrides)

    jobs = []
    for name in names:
        for seed in seeds:
            ov = dict(glob_ov)
            ov.update(grid["runs"][name] or {})
            ov.update({"experiment.seed": seed, "experiment.name": f"{out.name}/{name}/seed{seed}",
                       "experiment.output_dir": str(out / name / f"seed{seed}"), "experiment.threads": a.threads})
            cfg = apply_overrides(load_config(a.config), ov)
            jobs.append({"name": name, "seed": seed, "ov": ov, "dir": out / name / f"seed{seed}", "key": pretrain_key(cfg)})

    if not a.summary_only:
        pending = [j for j in jobs if not (j["dir"] / "metrics.json").exists()]
        producers: dict[str, dict] = {}
        # any earlier run under --out with a pretraining checkpoint can provide it (keyed by its saved config)
        for ck in sorted(out.glob("*/seed*/checkpoints/pretrain.pt")):
            run_dir = ck.parent.parent
            if (run_dir / "config.yaml").exists() and (run_dir / "metrics.json").exists():
                key = pretrain_key(OmegaConf.load(run_dir / "config.yaml"))
                if key and key not in producers:
                    producers[key] = {"name": run_dir.parent.name, "seed": None, "dir": run_dir, "key": key}
        phase1, phase2 = [], []
        for j in pending:
            if j["key"] is None:
                phase1.append(j)
            elif j["key"] not in producers:
                producers[j["key"]] = j
                phase1.append(j)
            elif producers[j["key"]] is not j:
                phase2.append(j)
        print(f"{len(jobs)} runs, {len(pending)} pending: {len(phase1)} first, {len(phase2)} reuse pretraining", flush=True)

        def launch(j: dict) -> int:
            cmd = [sys.executable, "-m", "wmpoc.run", "--config", a.config] + _cli_overrides(j["ov"])
            prod = producers.get(j["key"]) if j["key"] else None
            if prod is not None and prod is not j:
                ck = j["dir"] / "checkpoints"
                ck.mkdir(parents=True, exist_ok=True)
                shutil.copy(prod["dir"] / "checkpoints" / "pretrain.pt", ck / "pretrain.pt")
                (j["dir"] / "logs").mkdir(parents=True, exist_ok=True)
                if (prod["dir"] / "logs" / "pretrain.csv").exists():
                    shutil.copy(prod["dir"] / "logs" / "pretrain.csv", j["dir"] / "logs" / "pretrain.csv")
                cmd.insert(5, "--from-stage")
                cmd.insert(6, "world_model")
            j["dir"].mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with (j["dir"] / "stdout.log").open("w") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
            print(f"{'ok ' if rc == 0 else 'FAIL'} {j['name']} seed {j['seed']} ({time.time() - t0:.0f}s)", flush=True)
            return rc

        with ThreadPoolExecutor(a.jobs) as ex:
            rcs = list(ex.map(launch, phase1))
        with ThreadPoolExecutor(a.jobs) as ex:
            rcs += list(ex.map(launch, phase2))
        if any(rcs):
            print(f"{sum(r != 0 for r in rcs)} runs failed; see their stdout.log", flush=True)

    stem = Path(a.grid).stem
    write_summary(out, list(grid["runs"]), "summary" if stem == "ablations" else f"summary_{stem.removeprefix('ablations_')}")


def _ms(vals: list[float]) -> tuple[float, float, int]:
    v = np.array([x for x in vals if not np.isnan(x)])
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def _cell(m: float, s: float) -> str:
    if np.isnan(m):
        return "–"
    nd = 1 if abs(m) >= 10 else 3
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def write_summary(out: Path, names: list[str], fname: str = "summary") -> None:
    metrics = {}
    for name in names:
        for p in sorted((out / name).glob("seed*/metrics.json")):
            metrics[(name, p.parent.name)] = json.loads(p.read_text())
    rows = []
    for name in names:
        ms = [m for (n, _), m in metrics.items() if n == name]
        if not ms:
            continue
        cells = [_ms([_get(m, path) for m in ms]) for _, path in KEY_METRICS]
        ignores = sum(bool(m["test"]["all"]["action_sensitivity"]["world_model"]["ignores_action"]) for m in ms)
        rows.append((name, len(ms), cells, ignores))
    base = [m for (n, _), m in metrics.items() if n == "reference"] or list(metrics.values())
    lg = {"test WAPE bread h1": ("test", "all", "forecast", "lightgbm", "wape_bread_h1"),
          "ATE bias 20% (test randomized)": ("test", "randomized", "effect", "20%", "lightgbm_bias"),
          "CATE Spearman (test)": ("test", "all", "heterogeneity", "lightgbm", "spearman_vs_true_uplift"),
          "planning regret € (test)": ("test", "all", "planning", "lightgbm", "regret"),
          "planning regret € (val)": ("val", "all", "planning", "lightgbm", "regret"),
          "oracle match (test)": ("test", "all", "planning", "lightgbm", "oracle_match_share"),
          "action sensitivity (test)": ("test", "all", "action_sensitivity", "lightgbm", "ratio"),
          "Δ margin 4w bias 10% € (test)": ("test", "all", "effect", "10%", "lightgbm_margin_bias"),
          "hybrid regret € (test)": (), "hybrid regret € (val)": (),
          "structured regret € (test)": (), "structured regret € (val)": ()}
    if base and "lightgbm" in base[0]["test"]["all"]["forecast"]:
        rows.append(("**lightgbm** (baseline)", len(base), [_ms([_get(m, lg[k]) for m in base]) for k, _ in KEY_METRICS], 0))
    if base:
        nan = (float("nan"), float("nan"), 0)
        pl = lambda m, sp, k: _get(m, (sp, "all", "planning", k))
        for label, key in (("**never discount**", "never_discount_regret"), ("**always 10%**", "always_10_regret")):
            cells = [nan] * len(KEY_METRICS)
            cells[3] = _ms([pl(m, "test", key) for m in base])
            cells[4] = _ms([pl(m, "val", key) for m in base])
            rows.append((label, len(base), cells, 0))
        cells = [nan] * len(KEY_METRICS)
        cells[3] = cells[4] = (0.0, 0.0, len(base))
        cells[5] = (1.0, 0.0, len(base))
        rows.append(("**oracle**", len(base), cells, 0))
    rows.sort(key=lambda r: (np.inf if np.isnan(r[2][3][0]) else r[2][3][0]))
    hdr = ["run", "seeds"] + [k for k, _ in KEY_METRICS] + ["runs ignoring action"]
    L = ["# Ablation summary", "",
         "Mean ± std over seeds, sorted by test planning regret (lower is better). Planning regret = true "
         "4-week total margin of the oracle level minus that of the chosen level, € per store-week. "
         "Baseline rows (LightGBM, fixed policies) use the reference runs' evaluation windows. "
         "Model choices should be made on the val column, not the test column.", "",
         "| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    for name, n, cells, ign in rows:
        L.append("| " + " | ".join([name, str(n)] + [_cell(m, s) for m, s, _ in cells] + [str(ign) if not name.startswith("**") else "–"]) + " |")
    (out / f"{fname}.md").write_text("\n".join(L) + "\n")
    (out / f"{fname}.json").write_text(json.dumps(
        [{"run": r[0], "seeds": r[1], **{k: {"mean": c[0], "std": c[1]} for (k, _), c in zip(KEY_METRICS, r[2])}} for r in rows],
        indent=2, default=lambda x: None))
    print(f"wrote {out / (fname + '.md')}", flush=True)


if __name__ == "__main__":
    main()
