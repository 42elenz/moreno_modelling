import json

from wmpoc.run import run
from tests.conftest import ROOT, tiny_cfg


def _smoke_cfg(name):
    return tiny_cfg(**{
        "experiment.name": name, "experiment.output_dir": str(ROOT / "runs" / name), "experiment.threads": 1,
        "data.max_stores": 14, "training.stages.pretrain.epochs": 1, "training.stages.world_model.epochs": 1,
        "training.stages.outcome_heads.epochs": 1, "training.stages.joint_finetune.enabled": True,
        "training.stages.joint_finetune.epochs": 1, "baselines.lightgbm.n_estimators": 20, "eval.plots": False,
        "eval.splits": ["test"], "training.log_every_steps": 0,
    })


def _strip(m):
    m = dict(m)
    m.pop("meta")
    return m


def test_same_seed_same_metrics():
    a = run(_smoke_cfg("_test_det_a"))
    b = run(_smoke_cfg("_test_det_b"))
    ja = json.loads((ROOT / "runs/_test_det_a/metrics.json").read_text())
    jb = json.loads((ROOT / "runs/_test_det_b/metrics.json").read_text())
    assert _strip(ja) == _strip(jb)
    assert a["test"]["all"]["planning"]["world_model"]["regret"] == b["test"]["all"]["planning"]["world_model"]["regret"]
