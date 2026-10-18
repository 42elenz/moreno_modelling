"""Config loading: OmegaConf, `defaults:` merging and dot-list overrides.

Overrides *replace* the value at a key instead of merging into it, so
`model.world_model.category_loss_weights={default: 1.0}` really drops the bread weight.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from omegaconf import DictConfig, OmegaConf

# Keys the brief's configs do not set. Everything here can be overridden from the CLI.
CODE_DEFAULTS: dict[str, Any] = {
    "experiment": {"threads": 0},  # 0 = torch default
    "data": {"split": {"seed": 0}},  # seed of the cold-start store draw (fixed across model seeds)
    "model": {
        "encoder": {"causal_time": True},
        "world_model": {"outcome_aux_weight": 1.0},  # outcome loss when the encoder trains in stage 2
    },
    "baselines": {"lightgbm": {"num_threads": 1}},
    "planning": {"risk_on": "uplift"},  # uplift | level: what the ensemble-std penalty is computed on
    "eval": {"action_ignore_threshold": 0.25, "untested_grid_step": 0.05},
}


def _load_with_defaults(path: Path) -> DictConfig:
    raw = OmegaConf.load(path)
    if "defaults" in raw:
        parents = raw.pop("defaults")
        parents = [parents] if isinstance(parents, str) else list(parents)
        merged = OmegaConf.create()
        for p in parents:
            merged = OmegaConf.merge(merged, _load_with_defaults(path.parent / str(p)))
        raw = OmegaConf.merge(merged, raw)
    return raw


def parse_overrides(overrides: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value) if value.strip() else None
    return out


def apply_overrides(cfg: DictConfig, overrides: Mapping[str, Any] | Iterable[str]) -> DictConfig:
    if not isinstance(overrides, Mapping):
        overrides = parse_overrides(overrides)
    for key, value in overrides.items():
        if isinstance(value, (dict, list)):
            value = OmegaConf.create(value)
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    return cfg


def load_config(path: str | Path, overrides: Mapping[str, Any] | Iterable[str] = ()) -> DictConfig:
    cfg = OmegaConf.merge(OmegaConf.create(CODE_DEFAULTS), _load_with_defaults(Path(path)))
    apply_overrides(cfg, overrides)
    return cfg


def to_yaml(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)


def to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


__all__ = ["load_config", "apply_overrides", "parse_overrides", "to_yaml", "to_dict", "yaml"]
