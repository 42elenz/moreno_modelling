from pathlib import Path

import pytest

from wmpoc.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def tiny_cfg(**overrides):
    base = {
        "data.dir": str(ROOT / "data"), "data.max_stores": 12, "model.d_model": 16,
        "model.encoder.n_layers": 1, "model.encoder.n_heads": 2, "model.world_model.n_layers": 1,
        "model.world_model.n_heads": 2, "model.world_model.ensemble_size": 1,
        "model.encoder.jepa.predictor": {"n_layers": 1, "n_heads": 2, "d_model": 8},
        "model.outcome_heads.hidden": [8], "training.batch_size": 32,
    }
    base.update(overrides)
    return load_config(ROOT / "configs" / "poc_tiny.yaml", base)


@pytest.fixture(scope="session")
def cfg():
    return tiny_cfg()


@pytest.fixture(scope="session")
def data(cfg):
    from wmpoc.data.windows import build_data
    return build_data(cfg)
