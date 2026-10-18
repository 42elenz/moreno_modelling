"""The only module that reads `data/ground_truth/` (and the manifest's true effects).

Training and model-selection code never imports this. `wmpoc.run` calls it after all
stages are finished, through `wmpoc.eval.metrics.evaluate`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GT_DIR = "ground_truth"


@dataclass
class GroundTruth:
    traits: pd.DataFrame  # store_id, bread_beta, halo, ...
    traffic_shock: np.ndarray  # [S_all, W] by original store id
    counterfactuals: pd.DataFrame
    manifest: dict
    levels: np.ndarray

    def cf_arrays(self, store_ids: np.ndarray, origins: np.ndarray) -> dict[str, np.ndarray]:
        """True outcomes of [d,0,0,0] for each window and level. NaN where no counterfactual exists."""
        cf = self.counterfactuals
        L, N = len(self.levels), len(store_ids)
        H = 4
        lvl_idx = np.searchsorted(self.levels, cf["discount"].round(3).to_numpy())
        key = pd.MultiIndex.from_arrays([cf["store_id"], cf["week"], lvl_idx])
        units = np.full((N, L, H), np.nan)
        tm = np.full((N, L), np.nan)
        bm = np.full((N, L), np.nan)
        cols = [f"bread_units_w{k}" for k in range(H)]
        vals = cf[cols].to_numpy()
        tmv, bmv = cf["total_margin_4w"].to_numpy(), cf["bread_margin_4w"].to_numpy()
        for li in range(L):
            q = pd.MultiIndex.from_arrays([store_ids, origins, np.full(N, li)])
            pos = key.get_indexer(q)
            ok = pos >= 0
            units[ok, li] = vals[pos[ok]]
            tm[ok, li] = tmv[pos[ok]]
            bm[ok, li] = bmv[pos[ok]]
        return {"bread_units": units, "total_margin_4w": tm, "bread_margin_4w": bm,
                "valid": ~np.isnan(tm).any(1)}

    def beta(self, store_ids: np.ndarray) -> np.ndarray:
        return self.traits.set_index("store_id").loc[store_ids, "bread_beta"].to_numpy()


def load_ground_truth(data_dir: str | Path) -> GroundTruth:
    d = Path(data_dir) / GT_DIR
    traits = pd.read_csv(d / "store_traits.csv")
    dem = pd.read_csv(d / "demand.csv.gz", usecols=["store_id", "week", "category", "traffic_shock"])
    dem = dem[dem["category"] == dem["category"].iloc[0]]
    S, W = int(dem["store_id"].max()) + 1, int(dem["week"].max()) + 1
    shock = np.full((S, W), np.nan)
    shock[dem["store_id"].to_numpy(), dem["week"].to_numpy()] = dem["traffic_shock"].to_numpy()
    cf = pd.read_csv(d / "counterfactuals.csv.gz")
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    return GroundTruth(traits, shock, cf, manifest, np.array(manifest["discount_levels"], dtype=float))
