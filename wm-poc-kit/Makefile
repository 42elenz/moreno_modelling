PY ?= .venv/bin/python
SMOKE = experiment.name=smoke data.max_stores=30 \
	training.stages.pretrain.epochs=2 training.stages.world_model.epochs=2 \
	training.stages.outcome_heads.epochs=2 training.stages.joint_finetune.epochs=2 \
	training.stages.joint_finetune.enabled=true baselines.lightgbm.n_estimators=50

.PHONY: setup test smoke tiny ablate data data_rich audit

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

data:
	$(PY) data_gen/generate_fake_data.py --out data --n-stores 200 --n-weeks 156 --seed 7

data_rich:
	$(PY) data_gen/generate_fake_data.py --out data_rich --cohort-frac 0.9 --experiment-windows 10-154

audit:
	$(PY) -m wmpoc.eval.data_audit --config configs/poc_tiny.yaml

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m wmpoc.run --config configs/poc_tiny.yaml $(SMOKE)

tiny:
	$(PY) -m wmpoc.run --config configs/poc_tiny.yaml

ablate:
	$(PY) -m wmpoc.ablate --config configs/poc_tiny.yaml --grid configs/ablations.yaml
