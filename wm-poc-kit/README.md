# wmpoc: action-conditioned latent world model for retail discounts (proof of concept)

Learns, from a synthetic store × week × category panel (`data/`), how a bread discount changes a
store's next four weeks, and uses that to pick the discount that maximises 4-week total margin.
The stack is encoder → world model (ensemble) → outcome heads → planner, and every result is
compared against simple baselines on held-out weeks with known ground truth.

Brief: [`PROMPT.md`](PROMPT.md) · plan: [`PLAN.md`](PLAN.md) · findings: [`RESULTS.md`](RESULTS.md)

## Setup

```bash
make setup          # uv venv --python 3.11 .venv && uv pip install -e ".[dev]"
make test           # pytest
make smoke          # 30 stores, 2 epochs per stage, all stages + eval + report (< 5 min CPU)
make tiny           # python -m wmpoc.run --config configs/poc_tiny.yaml
make ablate         # python -m wmpoc.ablate --config configs/poc_tiny.yaml --grid configs/ablations.yaml
```

## Running

```bash
python -m wmpoc.run --config configs/poc_tiny.yaml [dot.overrides ...]
python -m wmpoc.run --config configs/poc_tiny.yaml model.encoder.type=masked_ae experiment.name=mae
python -m wmpoc.run --config configs/poc_tiny.yaml --resume            # continue after the last saved stage
python -m wmpoc.run --config configs/poc_tiny.yaml --from-stage outcome_heads
python -m wmpoc.ablate --config configs/poc_tiny.yaml --grid configs/ablations.yaml \
    --runs reference,enc_linear --seeds 0,1 --jobs 4
```

Overrides replace the value at a key (a dict override does not merge into the old dict).
Everything lands in `runs/<name>/`: `config.yaml`, `checkpoints/<stage>.pt`, `logs/<stage>.csv`,
`metrics.json`, `report.md`, `plots/*.png`. Ablations write `runs/ablations/summary.md`.

## Layout

```
wmpoc/
  config.py registry.py run.py ablate.py
  data/        loading.py (ground-truth guard)  features.py  windows.py (TokenBatch)  splits.py
  models/      backbone.py  system.py  heads.py
               encoders/{jepa,masked_ae,supervised,linear,random_frozen}.py
               world_models/{transformer,gru,additive,direct,no_action}.py
  training/    stages.py  losses.py  ema.py  masks.py
  baselines/   lightgbm_direct.py  diff_in_means.py  probes.py
  planning/    planner.py
  eval/        ground_truth.py  metrics.py  report.py      (the only code that reads data/ground_truth/)
tests/
```

## Adding a component

```python
from wmpoc.models.encoders.base import Encoder
from wmpoc.registry import ENCODERS

@ENCODERS.register("my_encoder")
class MyEncoder(Encoder):
    requires_pretraining = False
    def encode(self, batch, drop_features=()): ...   # [B,T,C,D]
```

Drop the file into `wmpoc/models/encoders/` (every module there is imported automatically) and
select it with `model.encoder.type=my_encoder`. World models and heads work the same way.

## Data rules

- `data/ground_truth/` is evaluation-only. `wmpoc.data.loading` refuses any path containing a
  `ground_truth` component, and a test checks that no code outside `wmpoc/eval/` mentions it.
- Splits are by week (`train [0,104)`, `val [104,130)`, `test [130,156)`); a window's horizon never
  leaves its split. 15 % of stores are cold-start stores, never used for training or model selection.
- Normalisation is fit on train weeks × train stores.
