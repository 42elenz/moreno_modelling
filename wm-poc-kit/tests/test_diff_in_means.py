import json

from wmpoc.baselines.diff_in_means import diff_in_means
from wmpoc.data.features import build_panel
from wmpoc.data.loading import load_raw
from tests.conftest import ROOT, tiny_cfg


def test_diff_in_means_matches_manifest():
    cfg = tiny_cfg(**{"data.max_stores": None})
    panel = build_panel(load_raw(cfg.data.dir), cfg)
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())["sanity_checks_test_period"]
    est = diff_in_means(panel, weeks=(130, 152))
    for lvl, ref in manifest.items():
        got = est[lvl]
        assert abs(got["estimate"] - ref["diff_in_means_randomized_test"]) <= 0.01 * abs(ref["diff_in_means_randomized_test"])
        assert abs(got["se"] - ref["diff_in_means_se"]) <= 0.01 * ref["diff_in_means_se"] + 0.1
        assert got["n"] == ref["n_randomized_test_slots"]
