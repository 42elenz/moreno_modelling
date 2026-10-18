# Plan: action-conditioned latent world model PoC

Brief: `PROMPT.md`. Data: `data/` (200 stores × 156 weeks × 8 categories, synthetic, known truth).

## Design decisions up front

- **One window = one store + one forecast origin `t`.** The loader slices weeks `[t-T, t+H)`.
  The encoder sees `[t-T, t)`. World-model targets are the last `H` tokens of the encoder run on
  the shifted window `[t+H-T, t+H)`, so every target sees the same number of weeks as the context.
- **Encoder time attention is causal** (`model.encoder.causal_time: true`): the embedding of week
  `w` only depends on weeks `≤ w`, so a target embedding never contains information from later weeks.
- **Masking = feature dropping.** Every token gets `[features · keep, keep]`. A masked cell has
  `keep = 0` everywhere; the price mask and `target_drops_price` set `keep = 0` on the price columns.
  One mechanism covers JEPA/MAE masks and the leakage fix.
- **Heads also predict `log_stock_units`.** Margin needs waste in units (waste_rate × stock) and the
  Tobit head predicts demand, so sales = min(demand, stock). Stock is therefore a third head output.
- **Prices in the horizon are treated as known** (base price and unit cost are planned by the
  retailer); bread net price = base price × (1 − action). They are never model inputs, only used to
  compute margin.
- Ground truth is only read by `wmpoc/eval/ground_truth.py`. The shared loader refuses any path
  with a `ground_truth` component.

## Milestones (commit after each)

- **M0** `pyproject.toml`, `Makefile`, `wmpoc/config.py` (OmegaConf, `defaults:` merging, dot-list
  overrides that replace rather than merge), `wmpoc/registry.py`, `make smoke` stub.
- **M1** `wmpoc/data/{loading,features,windows,splits}.py`: features exactly as named in the config,
  normalisation fit on train weeks × train stores, week splits with origins restricted so horizons
  stay inside the split, 15 % cold-start stores, `TokenBatch`. Tests: splits, ground-truth guard.
- **M2** Baselines: `diff_in_means` (check against manifest first), LightGBM direct per horizon step +
  4-week total margin model, calendar+store ridge probe. `wmpoc/eval/` with forecast, effect
  recovery, heterogeneity, action sensitivity, planning, representation, censoring metrics.
- **M3** Backbone (token embedding + factorized time/category attention), `linear` and `supervised`
  encoders, `direct` world model, `mlp` heads with Tobit loss, stages 2–3, eval end to end.
- **M4** `jepa` (EMA target, predictor, VICReg) and `masked_ae` pretraining with the four mask
  types; `transformer` (block-causal, action tokens + AdaLN), `gru`, `additive`, `no_action`;
  ensemble; rollout loss; joint finetune; checkpoints + resume.
- **M5** Planner (ensemble mean − λ·std of 4-week total margin over tested levels), full report with
  plots, `wmpoc.ablate` with summary table.
- **M6** Run `poc_tiny` and the ablation grid, write `RESULTS.md`.
