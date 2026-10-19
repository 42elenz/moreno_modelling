"""Entry point: python -m wmpoc.run --config <yaml> [dot.overrides ...]

Runs the enabled stages in order, the baselines, evaluation and the report, writing
everything to `runs/<name>/`. `--resume` continues after the last saved stage checkpoint;
`--from-stage <stage>` restarts at that stage from the checkpoint of the stage before it.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from wmpoc.baselines.lightgbm_direct import LightGBMDirect
from wmpoc.config import load_config, to_yaml
from wmpoc.data.windows import build_data
from wmpoc.eval.metrics import evaluate
from wmpoc.eval.report import write_metrics, write_report
from wmpoc.models.system import System
from wmpoc.baselines.experimental_effects import level_effects, pull_forward, relative_effects, slot_windows
from wmpoc.planning.planner import (CalibratedPredictor, HybridPredictor, LightGBMPredictor, StructuredPredictor,
                                    WorldModelPredictor)
from wmpoc.training.stages import RUNNERS, STAGES, StageContext, _log


def set_seed(seed: int, threads: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if threads > 0:
        torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_of(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def enabled_stages(cfg, system: System) -> list[str]:
    st = cfg.training.stages
    out = []
    for s in STAGES:
        if not bool(st[s].enabled):
            continue
        if s == "pretrain" and not system.encoder.requires_pretraining:
            continue
        out.append(s)
    return out


def run(cfg, resume: bool = False, from_stage: str | None = None, eval_only: bool = False) -> dict:
    t0 = time.time()
    set_seed(int(cfg.experiment.seed), int(cfg.experiment.threads))
    out = Path(cfg.experiment.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(to_yaml(cfg))
    device = device_of(str(cfg.experiment.device))
    data = build_data(cfg)
    system = System(data.dims, cfg, data.norm).to(device)
    ctx = StageContext(system, data, cfg, out, device)
    stages = enabled_stages(cfg, system)
    _log(f"run {cfg.experiment.name}: encoder={cfg.model.encoder.type} wm={cfg.model.world_model.type} "
         f"stages={stages} train windows={len(ctx.train[0])} device={device}")

    start = 0
    if from_stage is not None:
        if from_stage not in stages:
            raise ValueError(f"--from-stage {from_stage} is not an enabled stage {stages}")
        start = stages.index(from_stage)
    elif resume or eval_only:
        done = [i for i, s in enumerate(stages) if (out / "checkpoints" / f"{s}.pt").exists()]
        start = done[-1] + 1 if done else 0
    if start > 0:
        prev = stages[start - 1]
        ck = torch.load(out / "checkpoints" / f"{prev}.pt", map_location=device, weights_only=False)
        if prev == "pretrain":  # only the encoder exists after pretraining (portable across world models)
            system.encoder.load_state_dict(ck["encoder"])
        else:
            system.load_state_dict(ck["system"])
        _log(f"resumed from checkpoint {prev}")

    if eval_only:
        if start < len(stages):
            raise FileNotFoundError(f"--eval-only needs checkpoints of all stages {stages} in {out / 'checkpoints'}")
    for s in ([] if eval_only else stages[start:]):
        set_seed(int(cfg.experiment.seed) + STAGES.index(s), int(cfg.experiment.threads))
        hist = RUNNERS[s](ctx)
        torch.save({"system": system.state_dict(), "encoder": system.encoder.state_dict(), "stage": s, "history": hist},
                   out / "checkpoints" / f"{s}.pt")
    train_time = time.time() - t0
    if eval_only and (out / "metrics.json").exists():  # keep the training time of the original run
        train_time = json.loads((out / "metrics.json").read_text()).get("meta", {}).get("train_time_s", train_time)

    t_eval = time.time()
    predictors = {"world_model": WorldModelPredictor(system, data, device)}
    if bool(cfg.planning.hybrid_other_effects):
        eff = relative_effects(data.panel, data.splits.ranges["train"], data.splits.train_stores,
                               [float(x) for x in cfg.planning.discount_levels], data.dims.H, "other")
        _log("experimental other-category effects (train slots): "
             + ", ".join(f"{d:.0%} {r:+.2%} ± {e:.2%}" for d, r, e in zip(eff.levels, eff.rel, eff.se)))
        predictors["world_model_hybrid"] = HybridPredictor(predictors["world_model"], eff)
        if bool(cfg.planning.structured_bread):
            pf = pull_forward(data.panel, data.splits.ranges["train"], data.splits.train_stores,
                              [float(x) for x in cfg.planning.discount_levels], data.dims.H)
            _log(f"experimental bread pull-forward shares (train slots): {pf.summary()}")
            predictors["world_model_structured"] = StructuredPredictor(predictors["world_model"], eff, pf)
            if bool(cfg.planning.calibrate_bread):
                lv = [float(x) for x in cfg.planning.discount_levels]
                tr, st, H = data.splits.ranges["train"], data.splits.train_stores, data.dims.H
                p, a = data.panel, data.panel.action_idx[0]
                waste4 = np.zeros(p.units.shape[:2])
                waste4[:, : p.n_weeks - H + 1] = sum(p.waste[:, h: p.n_weeks - H + 1 + h, a] for h in range(H))
                wb, _ = level_effects(p, tr, st, lv, waste4, H)
                cs, ct = slot_windows(p, tr, st, H)
                cal = CalibratedPredictor(predictors["world_model"], eff, pf, cs, ct, pf.beta[:, 0], wb, lv)
                _log(f"bread calibration to the experiment (train slots): {cal.summary()}")
                predictors["world_model_calibrated"] = cal
    if bool(cfg.baselines.lightgbm.enabled):
        _log("fitting LightGBM baseline")
        predictors["lightgbm"] = LightGBMPredictor(LightGBMDirect(cfg).fit(data.panel, data.splits), data.panel)
    _log("evaluating")
    extras = predictors["world_model"] if system.latent_wm else _NoRealEmbeddings(predictors["world_model"])
    metrics, art = evaluate(data, cfg, predictors, extras)
    metrics["meta"] = {"name": str(cfg.experiment.name), "n_stores": int(data.panel.n_stores), "stages": stages,
                       "train_time_s": train_time, "runtime_s": train_time + (time.time() - t_eval),
                       "evaluated_from_checkpoint": bool(eval_only),
                       "n_params": int(sum(p.numel() for p in system.parameters())),
                       "train_windows": int(len(ctx.train[0]))}
    write_metrics(out, metrics)
    write_report(out, metrics, art, cfg, plots=bool(cfg.eval.plots))
    _log(f"done in {time.time() - t0:.0f}s -> {out / 'report.md'}")
    return metrics


class _NoRealEmbeddings:
    """Direct world models have no target embedding space: no heads-on-real read-out."""

    def __init__(self, p: WorldModelPredictor) -> None:
        self.p = p

    def real_embedding_units(self, s, t):
        return np.full((len(s), self.p.data.dims.H), np.nan)

    def embeddings(self, s, t):
        return self.p.embeddings(s, t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--from-stage", choices=STAGES)
    ap.add_argument("--eval-only", action="store_true", help="evaluate the saved final checkpoint, no training")
    ap.add_argument("overrides", nargs="*", help="dot.key=value overrides")
    a = ap.parse_args()
    run(load_config(a.config, a.overrides), resume=a.resume, from_stage=a.from_stage, eval_only=a.eval_only)


if __name__ == "__main__":
    main()
