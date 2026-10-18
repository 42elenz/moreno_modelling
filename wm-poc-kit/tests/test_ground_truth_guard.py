import re
from pathlib import Path

import pytest

from wmpoc.data.loading import GroundTruthAccessError, load_raw, read_csv
from tests.conftest import ROOT

PKG = ROOT / "wmpoc"


def test_loader_refuses_ground_truth():
    for p in [ROOT / "data/ground_truth/store_traits.csv", ROOT / "data/ground_truth/counterfactuals.csv.gz",
              "data/ground_truth/demand.csv.gz", ROOT / "data/ground_truth/../ground_truth/store_traits.csv"]:
        with pytest.raises(GroundTruthAccessError):
            read_csv(p)
    with pytest.raises(GroundTruthAccessError):
        load_raw(ROOT / "data/ground_truth")


def test_only_eval_mentions_ground_truth():
    """Training / model-selection code must not reference ground-truth files at all."""
    offenders = []
    for f in PKG.rglob("*.py"):
        rel = f.relative_to(PKG)
        if rel.parts[0] == "eval" or rel == Path("data/loading.py"):
            continue
        txt = f.read_text()
        if re.search(r"ground_truth|counterfactuals\.csv|store_traits|demand\.csv|manifest\.json", txt):
            offenders.append(str(rel))
    assert not offenders, f"ground truth referenced outside wmpoc/eval: {offenders}"
