# Results

*Status: draft. Reference run and the brief's six required ablation entries are complete; the remaining
grid entries (phase B) and the follow-up runs are still running and will be added.*

Data: synthetic panel, 200 stores × 156 weeks × 8 categories, known ground truth. Test = weeks 130–155
(4 600 store-weeks with counterfactuals), val = weeks 104–129. Everything below is on **test, all stores**
unless stated; model choices were made on val. Planning regret = true 4-week total margin of the oracle level
minus that of the chosen level (€ per store-week, lower is better).

## Short answer

**The world-model stack does not beat the simplest policy.** "Never discount" has a test regret of 38 €; the
reference world model 49 € (full budget) and 66 ± 10 € (reduced ablation budget, 3 seeds). It does beat the
LightGBM planner by a wide margin (103–107 €), and it recovers the bread discount effect about as well as LightGBM.
The headroom is small: the oracle beats "never discount" by only 38 € on a 37 245 € 4-week margin (0.1 %).

What the stack gets right, and where it fails:

1. **Bread effect: yes.** Mean predicted 10 % uplift in week-0 bread units 323 vs true 322; per-store uplift
   ranks correlate 0.91 with the truth (LightGBM 0.90). Diff-in-means on the randomized slots (252) and the naive
   observational estimate (340) are both further from the truth (322).
2. **Margin effect: no.** The planner needs Δ 4-week total margin. At 10 % the world model's bread margin effect
   is right (−217 € vs −223 € on val) but its effect on the **other seven categories has the wrong sign**
   (−118 € predicted vs +165 € true: the halo brings extra visitors). The likely cause is confounding in the
   observational weeks, where bread promos are more frequent in weak-traffic weeks, so "bread discount" co-occurs
   with "other categories down". The model therefore over-states the cost of discounting and picks 0 % in 91 %
   of store-weeks (truth: 68 %).
3. **Representation: yes, modestly.** Frozen z(t) predicts the hidden AR(1) traffic shock (R² 0.66, vs ≈ 0 for
   calendar + store and calendar + static features) and the store's true price sensitivity better than the
   observable store features (R² 0.59–0.62 vs 0.54). kNN in z-space forecasts next-week bread units slightly
   better than kNN on static features (WAPE 0.171 vs 0.187).
4. **Forecasting: LightGBM wins for the JEPA reference** (bread WAPE h1 0.103 vs 0.081), but the masked-AE
   variant beats LightGBM (0.079).

## Reference run (`configs/poc_tiny.yaml`, JEPA + transformer × 3, full budget, 55 min on 4 CPU cores)

Full report: [`runs/poc_tiny/report.md`](runs/poc_tiny/report.md).

| test subset | world model regret € | LightGBM regret € | never discount € | always 10 % € | WAPE bread h1 WM / LightGBM |
|---|---|---|---|---|---|
| all | 48.7 | 107.5 | **38.3** | 98.7 | 0.103 / 0.081 |
| randomized slot weeks | 39.4 | 115.4 | **29.3** | 109.1 | 0.098 / 0.082 |
| cold-start stores | 41.8 | 114.9 | **24.7** | 121.3 | 0.107 / 0.082 |

- Effect recovery, test randomized slots (comparable to the manifest): bias at 20 % −39 units (LightGBM +40,
  diff-in-means −76, naive +152).
- Heads on *real* future embeddings reach bread WAPE 0.048; on *predicted* embeddings 0.103. Half of the forecast
  error is the world model's, half the decoder's.
- Censored and uncensored weeks have the same error (0.101 vs 0.103), so the Tobit head does not leave a bias.
- Cold-start stores (no store ID): CATE Spearman 0.80 vs LightGBM 0.82; forecast WAPE 0.107 vs 0.082.

## Ablations

Reduced budget so the grid fits on a 4-core CPU: pretraining and world-model stages 8 epochs (instead of 15/20),
ensemble of 2 (instead of 3), one thread per run, 3 seeds. Table: [`runs/ablations/summary.md`](runs/ablations/summary.md).

| run | test regret € | val regret € | WAPE bread h1 | CATE Spearman | Δ margin bias 10 % € |
|---|---|---|---|---|---|
| never discount | **38.3** | **37.7** | – | – | – |
| wm_no_action | 38.3 ± 0.0 | 37.7 ± 0.0 | 0.136 | – (constant) | −218 |
| enc_masked_ae | 63.4 ± 15.7 | 42.3 ± 1.7 | **0.079** | 0.895 | −172 ± 166 |
| enc_linear | 64.3 ± 15.3 | 63.5 ± 19.3 | 0.092 | 0.885 | −226 ± 122 |
| reference (jepa) | 66.2 ± 10.3 | 60.3 ± 11.9 | 0.099 | 0.882 | −56 ± 82 |
| always 10 % | 98.7 | 95.6 | – | – | – |
| LightGBM | 102.9 ± 5.9 | 76.2 ± 10.9 | 0.081 | 0.893 | −16 ± 7 |
| keep_price_in_targets | 112.1 ± 47.4 | 117.1 ± 81.3 | 0.095 | 0.877 | +2 ± 203 |
| wm_direct | 198.7 ± 14.9 | 186.1 ± 23.1 | 0.085 | 0.895 | +167 ± 7 |

What this shows:

- **The simpler model wins.** `wm_no_action` (the world model with actions zeroed) never discounts, so it ties
  "never discount" and has the lowest regret of any model. Correctly flagged as ignoring the action.
- **JEPA pretraining is not worth it here.** Masked-AE is better on forecasting (0.079 vs 0.099) and on val regret
  (42 vs 60); a frozen random *linear* projection plans as well as JEPA (64 vs 66). Differences in test regret
  between the three encoders are within one standard deviation.
- **Latent dynamics matter for planning.** `wm_direct` (heads read z(t) + action directly) forecasts bread well
  (0.085) and ranks stores well (0.895) but has the worst regret (199 €): it over-predicts the margin gain of a
  discount (+167 € bias) and discounts 75 % of store-weeks.
- **Dropping prices from the targets matters.** Keeping them roughly doubles regret and makes it unstable across
  seeds (112 ± 47 €), without changing the bread effect much.
- **LightGBM** has the smallest *average* margin bias but the worst *ranking* of store-weeks among the planners,
  so its choices are poor (oracle match 51 % vs 60 % for the reference).

## Changes from the brief, and why

Data and targets
- **Project location.** The kit lives in `wm-poc-kit/` inside this repository (the repo root holds an unrelated
  earlier analysis). All commands run from `wm-poc-kit/`.
- **Encoder time attention is causal** (`model.encoder.causal_time: true`, new key). An embedding of week w only
  sees weeks ≤ w, so a target embedding cannot leak later weeks, and the last context week is a summary of z(t).
- **World-model targets** are the last H tokens of `target_encode` run on the window `[t+H−T, t+H)` (same length T as
  the context), with the price features dropped. The step-0 input of the world model is the last context week encoded
  the same way, so the dynamics live entirely in the target space; z(t) from `encode` (with prices) is the prefix.
- **Masking is feature dropping.** Each token is `Linear([x·keep, keep])`; masks set `keep = 0` for whole cells, the
  price mask and `target_drops_price` set it for the price columns. JEPA's predictor gets a mask token at hidden cells.
- **Heads also predict `log_stock_units`.** Margin needs waste in units (= waste_rate × stock), and with Tobit/exclude
  the units head predicts demand, so sales = min(demand, stock). Stock is therefore a third head output.
- **Censoring flag.** `stockout_hours > 0` as specified. This flags 19.7 % of bread weeks, not the 10.9 % that are
  truly censored, because the generator adds small random stock-out hours to 10 % of cells.
- **Prices in the horizon are treated as known** (base price and unit cost are planned). Bread net price =
  base price × (1 − action); other categories keep their observed (planned) net price. Used only for margin.
- **"randomized" evaluation subset** = windows whose first horizon week has `experiment_slot == 1` (the bread
  discount was drawn at random that week), which matches the manifest's sanity-check population.
- `diff_in_means` uses weeks 130–152 for test (as specified) and 104–126 for val.

Models
- **Transformer prefix is compressed** to 2·C tokens (last context week and the mean over context weeks) plus one
  static/store token, instead of all T·C context tokens, and attention weights have no dropout. Both are for CPU speed
  (the uncompressed model was ~3× slower and ~50× slower than the GRU); with a causal encoder the last week already
  summarises the window.
- **`direct` is trained through the heads** (outcome loss, one head per ensemble member) because it has no latent
  target to regress; stage 3 is skipped for it. It therefore has no "heads on real embeddings" row.
- **`supervised` encoder:** a latent loss against its own detached outputs collapses, so stage 2 adds the outcome loss
  on real and predicted embeddings (`model.world_model.outcome_aux_weight`, new key). The encoder trains with ensemble
  member 0; members 1..K−1 are then trained on the frozen result.
- **`gru`:** always mean-pools z over categories for its state and reads out per-category residuals.
  `encoder.pooling` is accepted but only the GRU uses pooling.
- **`additive`:** f(action, known future, static + store embedding), no z input, as literally specified.
- **Joint finetune:** the JEPA target encoder stays fixed (no EMA updates); outcome loss is on rollouts.
- **JEPA pretraining has no early stopping** (fixed `epochs`): the EMA target moves, so the latent loss is not
  comparable across epochs. Masked-AE pretraining and all other stages early-stop on val (val weeks × training stores,
  never cold-start stores) and restore the best weights. Validation masks use a fixed seed.
- **Frozen-encoder latents are cached** for stages 2 and 3 (exactly equal to recomputing them in eval mode).

Planning and evaluation
- **Risk adjustment on the uplift.** score = mean_k(M_k(d) − M_k(0)) − λ·std_k(M_k(d) − M_k(0)) instead of the
  mean − λ·std of the raw level margin (`planning.risk_on: uplift | level`, new key). The spread of the raw margin is
  mostly baseline error shared by all levels; it says nothing about whether the discount pays off.
- **Probes** use z(t) = [bread token, category mean] of the last context week. Besides the calendar + store one-hot
  floor (useless for store-level traits under store-grouped CV by construction) there is a calendar + static-features
  floor, the honest comparison for `bread_beta` and `size_sqm`.
- **LightGBM** has no store ID (grouped by nothing; static features stand in) and predicts total margin directly, as
  specified. Its planner uses that margin model.

Runs
- **Ablation budget reduced** (8 + 8 epochs, ensemble of 2, see above) to fit the 17 × 3 grid on 4 CPU cores. Runs
  with identical pretraining settings share one pretraining checkpoint (bit-identical to re-running it).
- **Holiday probe is undefined on test**: no national holiday falls on a test origin week; it is reported for val.
