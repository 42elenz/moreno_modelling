"""Raw table loading for training code, with the ground-truth guard.

Nothing under a `ground_truth/` directory may be read through this module. Only
`wmpoc/eval/ground_truth.py` reads those files, with its own reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FORBIDDEN_DIR = "ground_truth"


class GroundTruthAccessError(PermissionError):
    """Raised when training or model-selection code tries to read ground-truth data."""


def guard_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    if FORBIDDEN_DIR in p.parts or FORBIDDEN_DIR in Path(path).parts:
        raise GroundTruthAccessError(f"refusing to read {path}: ground truth is evaluation-only")
    return p


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(guard_path(path), **kwargs)


@dataclass
class RawData:
    sales: pd.DataFrame
    calendar: pd.DataFrame
    stores: pd.DataFrame


def load_raw(data_dir: str | Path, max_stores: int | None = None) -> RawData:
    d = guard_path(data_dir)
    stores = read_csv(d / "stores.csv")
    sales = read_csv(d / "sales.csv.gz")
    calendar = read_csv(d / "calendar.csv", keep_default_na=False)
    if max_stores is not None:
        keep = sorted(stores["store_id"])[: int(max_stores)]
        stores = stores[stores["store_id"].isin(keep)].reset_index(drop=True)
        sales = sales[sales["store_id"].isin(keep)].reset_index(drop=True)
    return RawData(sales=sales, calendar=calendar, stores=stores)
