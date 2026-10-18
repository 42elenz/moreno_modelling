# Task: build a proof of concept for an action-conditioned latent world model for retail discounts

You are implementing a research proof of concept in Python/PyTorch. Read this whole brief,
then `data/README.md`, `data/manifest.json` and the three files in `configs/` before writing
code. Start by writing a short plan (`PLAN.md`) with the milestones below, then work through
them in order. Commit after each milestone.

## 1. Goal

Learn, from a store × week × category sales panel, how a bread discount changes a store's
future, and use that to pick the discount that maximises 4-week total margin.

Architecture, in four stages:

1. **Encoder** turns a window of the panel (categories × weeks, plus context) into one
   embedding per (category, week) token. Default: JEPA-style self-supervised pretraining.
2. **World model** takes the encoded past z(t), the planned action (bread discount per
   future week) and known-future features, and predicts the embeddings of the next
   1–4 weeks. Trained with a latent loss against the real future embeddings.
3. **Outcome heads** decode embeddings into units sold and waste per category.
   Margin is computed from these with prices and costs; it is never learned.
4. **Planner** rolls out every tested discount level with an ensemble of world models and
   picks the best risk-adjusted 4-week total margin.

The central research question is whether this stack beats simple baselines on held-out
randomized weeks. The code must make that comparison easy and honest. If a simpler variant
wins, the report must say so.

## 2. What is provided

```
data_gen/generate_fake_data.py   synthetic data generator with known ground truth
data/                            generated data (200 stores, 156 weeks, 8 categories)
data/ground_truth/               TRUE effects and counterfactuals, evaluation only
configs/poc_tiny.yaml            default config (CPU)
configs/poc_small.yaml           larger config (GPU), merges onto poc_tiny
configs/ablations.yaml           ablation grid
```

Hard rule: nothing under `data/ground_truth/` may be read by any training or model-selection
code. Only `wmpoc/eval/` may read it. Enforce this in the data loader (refuse those paths)
and with a test.

## 3. Swappable components (most important design requirement)

Each component is an abstract base class plus a registry, selected by the `type:` key in the
config. Adding a new encoder must not require touching any training loop.

```python
# wmpoc/registry.py
ENCODERS, WORLD_MODELS, HEADS = Registry("encoder"), Registry("world_model"), Registry("head")

@ENCODERS.register("jepa")
class JEPAEncoder(Encoder): ...
```

Tensor convention: `B` batch, `T` context weeks, `H` horizon weeks, `C` categories,
`D` = `model.d_model`. A `TokenBatch` dataclass carries `tokens [B,T,C,F]`,
`context [B,T,Fc]`, `known_future [B,H,Fk]`, `actions [B,H,A]`, `static [B,Fs]`,
`store_id [B]`, `randomized [B,H]`, plus targets and masks.

```python
class Encoder(nn.Module, ABC):
    requires_pretraining: bool                # False for supervised, linear, random_frozen
    def encode(self, batch, drop_features: Sequence[str] = ()) -> Tensor:        # [B,T,C,D]
    def target_encode(self, batch, drop_features: Sequence[str] = ()) -> Tensor: # no grad
        """Embeddings used as world-model targets. JEPA: EMA target encoder. Others: self, detached."""
    def pretrain_step(self, batch) -> dict[str, Tensor]:   # must contain "loss"
    def after_optimizer_step(self, step: int, total_steps: int) -> None:  # EMA update hook

class WorldModel(nn.Module, ABC):
    def forward(self, z_ctx, actions, known_future, static, store_emb) -> Tensor:   # [B,H,C,D], teacher-forced
    def rollout(self, z_ctx, actions, known_future, static, store_emb, steps) -> Tensor  # autoregressive

class OutcomeHead(nn.Module, ABC):
    def forward(self, z) -> dict[str, Tensor]:     # e.g. {"log_units_sold": [B,H,C], "waste_rate": [B,H,C]}
    def loss(self, preds, batch) -> Tensor        # handles censoring
```

Implementations required:

| Registry | Key | What it is |
|---|---|---|
| encoder | `jepa` | Context encoder + EMA target encoder + small predictor. Latent loss on masked tokens, stop-gradient on the target, VICReg-style anti-collapse term (configurable). |
| encoder | `masked_ae` | Same backbone and masks, but reconstructs raw scaled features of masked cells. |
| encoder | `supervised` | Same backbone, no pretraining, trained end-to-end in stage 2. |
| encoder | `linear` | Per-token linear projection plus ID embeddings, no attention. Representation-learning baseline. |
| encoder | `random_frozen` | Same backbone, random weights, frozen. Control: does pretraining matter at all? |
| world_model | `transformer` | Block-causal transformer over horizon steps, per-category tokens, actions as tokens and via AdaLN on the target-category tokens. |
| world_model | `gru` | Mean-pooled z, GRU over the horizon. |
| world_model | `additive` | z(t+h) = z(t+h−1) + MLP(action, known future). The simplest "add an action" version. |
| world_model | `direct` | No latent dynamics: the heads read z(t) + actions + known future directly for each horizon step. |
| world_model | `no_action` | Transformer with actions zeroed. The gap to `transformer` is the learned action effect. |
| head | `mlp` | Per-category MLP. Tobit-style loss when `stockout_hours > 0`, because sales are a lower bound on demand. |

The backbone (token embedding: Linear(F→D) + category embedding + week-position embedding +
store embedding, then factorized time/category attention) is shared code used by `jepa`,
`masked_ae`, `supervised` and `random_frozen`.

## 4. Data pipeline

- Build features exactly as listed in the config (`token_features`, `context_features`,
  `known_future_features`, `static_store_features`). Log transforms use `log1p`.
  Fit normalisation on the train split only.
- A sample = one store, `context_weeks` of history, `horizon_weeks` of future.
- Splits are by week, with a gap of `horizon_weeks` so no window crosses a boundary.
  `holdout_store_frac` of stores are never used in any training split (cold start).
- Store-ID embedding dropout: with probability `store_id_embedding.dropout`, the ID vector
  is zeroed during training, so the model must infer the store type from its data.
- `target_drops_price: true`: world-model targets are encoded with the price features
  blanked (the same blanking the `price` pretraining mask uses). This prevents the model
  from lowering the loss by copying the action into the target.
- Randomized bread weeks get `randomized_sample_weight` in stages 2 and 3.

## 5. Training stages

One entry point `python -m wmpoc.run --config <yaml> [dot.overrides ...]` runs the enabled
stages in order and writes everything to `runs/<name>/`.
Use OmegaConf for config loading, `defaults:` merging and dot-list CLI overrides.

1. **pretrain** (if `encoder.requires_pretraining`): masked latent prediction (JEPA) or
   reconstruction (MAE), with one mask type sampled per batch from `encoder.masks`.
2. **world_model**: encoder frozen (unless `supervised`). Loss =
   `teacher_forcing_weight · L_tf + rollout_weight · L_rollout`, both latent losses against
   `target_encode(future, drop price)`, weighted per category by `category_loss_weights`.
   Train `ensemble_size` independently seeded world models.
3. **outcome_heads**: encoder and world model frozen. `train_on: real_embeddings` trains heads
   on `target_encode` of the real future weeks. Always also report head accuracy on
   predicted embeddings, so the real-vs-predicted gap is visible.
4. **joint_finetune** (optional): unfreeze everything, small LR, latent + outcome loss.

Save checkpoints per stage and allow resuming from any stage.

## 6. Baselines

- `lightgbm`: direct model per horizon step predicting bread units (and total margin) from
  lagged features, static features, known future and the action. Same splits and sample
  weights. This is the benchmark the world model must beat.
- `diff_in_means`: effect per discount level from bread rows with `experiment_slot == 1`
  in weeks 130–152 only (mean units at level d minus mean units at 0%).
- `calendar_store_probe`: ridge on calendar + one-hot store, the floor for the probes.

## 7. Evaluation (`wmpoc/eval/`, the only place that reads ground truth)

Compute for val and test, on subsets `all`, `randomized` and `cold_start_stores`:

1. **Forecast**: WAPE of bread units for h = 1..4 and error of 4-week total margin.
2. **Effect recovery**: for each level d, predicted uplift `pred(d) − pred(0)` against the true
   uplift from `counterfactuals.csv.gz` (bias, RMSE), next to `diff_in_means` and the naive
   observational estimate from `manifest.json`.
3. **Heterogeneity**: Spearman correlation between per-store predicted 10% uplift and the
   true per-store uplift; also against `bread_beta`.
4. **Action sensitivity**: mean |pred(30%) − pred(0%)| relative to the true difference.
   Flag a run where the model ignores the action.
5. **Planning**: for every test store-week with counterfactuals, look up the true
   `total_margin_4w` of the chosen level. Report policy value against the oracle,
   "never discount", "always 10%" and the LightGBM planner; regret; share of oracle-matching
   choices.
6. **Representation**: ridge probes on frozen z(t) (grouped CV by store) for `bread_beta`,
   `size_sqm`, holiday and `traffic_shock`, compared with the calendar+store probe. kNN test:
   do the 10 nearest neighbours in z-space predict next-week bread units better than
   neighbours chosen on static features?
7. **Censoring**: forecast error split by censored vs uncensored weeks.

Write `runs/<name>/metrics.json` and `runs/<name>/report.md` (tables, plus PNG plots:
training curves, uplift curve vs truth, predicted vs true per-store uplift, planning value
bars, embedding PCA coloured by `bread_beta`).

## 8. Ablations

`python -m wmpoc.ablate --config configs/poc_tiny.yaml --grid configs/ablations.yaml`
runs every entry in the grid for every seed and writes `runs/ablations/summary.md`:
one row per run, mean ± std over seeds for the key metrics (test WAPE bread h1,
ATE bias at 20%, CATE Spearman, planning regret), sorted by planning regret, with the
LightGBM and oracle rows included.

## 9. Tests and acceptance criteria

`pytest` must cover at least:

- shapes for every encoder × world model combination with a tiny config
- block-causal mask: perturbing future-step inputs leaves earlier-step outputs unchanged
- JEPA: target encoder parameters get no gradient; EMA update changes them
- `target_drops_price`: price features are absent from world-model target inputs
- splits: weeks disjoint with the gap, held-out stores never in training batches
- ground-truth guard: training code cannot load `data/ground_truth/`
- determinism: same seed, same metrics on a smoke run
- `diff_in_means` reproduces the manifest values to within 1%

Done means:

- `make smoke` (30 stores, 2 epochs per stage, all stages, eval, report) passes in under
  5 minutes on CPU
- `python -m wmpoc.run --config configs/poc_tiny.yaml` completes and writes the report
- `python -m wmpoc.ablate ...` completes for at least the `reference`, `enc_masked_ae`,
  `enc_linear`, `wm_direct`, `wm_no_action` and `keep_price_in_targets` runs
- `RESULTS.md`: a short, plain write-up of what the ablations show, including where the
  simpler model wins, and a list of anything you changed from this brief and why

## 10. Repo layout and tooling

```
wmpoc/
  config.py  registry.py  run.py  ablate.py
  data/       loading.py  features.py  windows.py  splits.py
  models/     backbone.py  encoders/{jepa,masked_ae,supervised,linear,random_frozen}.py
              world_models/{transformer,gru,additive,direct,no_action}.py  heads.py
  training/   stages.py  losses.py  ema.py  masks.py
  baselines/  lightgbm_direct.py  diff_in_means.py  probes.py
  planning/   planner.py
  eval/       ground_truth.py  metrics.py  report.py
tests/
Makefile  pyproject.toml  README.md
```

Python 3.11, PyTorch ≥ 2.2, numpy, pandas, omegaconf, lightgbm, scikit-learn, matplotlib,
pytest. Use `uv` for the environment. No experiment-tracking service; log to CSV/JSON
in the run folder. Type-hint public functions. Keep models small enough that
`poc_tiny` trains on CPU.

## 11. Milestones

- **M0** Environment, config loading with overrides, registry, `make smoke` stub.
- **M1** Data pipeline, features, windows, splits, ground-truth guard, tests.
- **M2** Baselines (`diff_in_means`, `lightgbm`, probe floor) and the eval module. Check
  that `diff_in_means` matches the manifest before building any neural model.
- **M3** Backbone, `linear` and `supervised` encoders, `direct` world model, heads, end to
  end. This is the simplest neural pipeline.
- **M4** `jepa` and `masked_ae` pretraining with all masks, plus the `transformer`, `gru`,
  `additive` and `no_action` world models, ensemble, rollout loss.
- **M5** Planner, full report, ablation runner.
- **M6** Run `poc_tiny` and the ablations, write `RESULTS.md`.

If something in this brief is ambiguous, pick the simplest reasonable option, note it in
`RESULTS.md`, and keep going. Do not tune on the test split; use val for all choices.
