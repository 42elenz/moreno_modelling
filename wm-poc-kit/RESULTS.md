# Results

Data: synthetic panel, 200 stores × 156 weeks × 8 categories, known ground truth. Test = origins in weeks 130–152
(4 600 store-weeks with counterfactuals), val = weeks 104–126. All numbers are on **test, all stores** unless stated.
Planning regret = true 4-week total margin of the oracle discount level minus that of the chosen level, € per
store-week, lower is better. Model choices should be read off the val column; test is reported for completeness.

## Short answer

**On its own, the world model does not beat "never discount".** With two pieces taken from the randomized
experiment, it can, by a little, and by more when the experiment is bigger.

- The plain world model loses to "never discount" (test regret 47 ± 4 € vs 38 €, full budget, 3 seeds), though it
  beats the LightGBM planner (103 €). Two errors cause this: a confounded cross-category effect and an
  under-predicted bread pull-forward.
- The **structured planner** keeps what the world model predicts well (store-specific week-0 bread uplift, waste,
  the no-discount baseline) and takes the pull-forward pattern and the cross-category effect from the experiment.
  With the *true* cross-category effect it beats "never discount" on both datasets (test 34.6 vs 38.3 €; 35.3 vs
  40.8 € on the richer data), so the world-model part is good enough.
- Estimated from data, the cross-category effect is too noisy on the original data (± 0.8 % of a ~30 k€ margin) to
  act on safely: the per-level estimate wins there by luck (test 35.6 ± 1.5 €) and loses on the richer data
  (56 €). The pooled-slope estimate is robust: never worse than "never discount" in any run or subset, ≈ equal on
  the original data (37.9 vs 38.3 €; it hardly ever discounts) and **clearly better on the 4× richer experiment
  (34.9 ± 0.4 vs 40.8 €, val 32.7 vs 37.0 €)**.
- The headroom is small throughout: the oracle beats "never discount" by ~38–41 € on a ~37 k€ 4-week margin (0.1 %).

Findings of the first round (plain world model):

1. **The bread effect is learned well.** Mean predicted 10 % uplift in week-0 bread units 323 vs true 322; per-store
   uplift ranks correlate 0.88–0.91 with the truth, as good as LightGBM (0.89–0.90). Diff-in-means on the randomized
   slots (252) and the naive observational estimate (340) are both further from the truth.
2. **The margin effect is wrong, and that is why planning fails.** The model gets the *bread* margin effect roughly
   right, but predicts that the *other seven categories* lose money under a bread discount, when they actually gain
   ~170 € per store-week at 10 % (the halo brings extra visitors). It therefore over-states the cost of discounting
   and picks 0 % in ~80–90 % of store-weeks (truth: 68 %).
3. **The cause is confounding in the observational weeks** (bread promos are more frequent in weak-traffic weeks).
   Evidence: removing the extra weight on randomized weeks makes the other-category error much worse; up-weighting
   them moves it towards the truth (table below). Up-weighting alone does not fix planning, though: the bread effect
   then becomes too optimistic and val regret rises.
4. **JEPA pretraining is not worth it here.** Masked-AE, linear and even random frozen encoders plan as well as JEPA
   (differences within seed noise); masked-AE and supervised encoders forecast better.
5. **Better forecasts do not mean better plans.** The supervised encoder and LightGBM have the best forecasts
   (bread WAPE 0.078 / 0.081) and among the worst planning regrets (82 € / 103 €).
6. **The representation is the clear positive.** Frozen z(t) predicts the hidden traffic shock (R² 0.63 vs ≈ 0 for
   calendar + store and calendar + static features) and the true store price sensitivity better than observable
   store features (R² 0.59 vs 0.54).

7. **Fixes and data check.** See "Round 2" below: data audit, hybrid, structured, calibrated and pooled planners,
   full-budget seeds and the richer dataset.

## Reference run (`configs/poc_tiny.yaml`, JEPA + transformer × 3, full budget, 55 min on 4 CPU cores)

Full report: [`runs/poc_tiny/report.md`](runs/poc_tiny/report.md).

| test subset | world model regret € | LightGBM regret € | never discount € | always 10 % € | WAPE bread h1 WM / LightGBM |
|---|---|---|---|---|---|
| all | 48.7 | 107.5 | **38.3** | 98.7 | 0.103 / 0.081 |
| randomized slot weeks | 39.4 | 115.4 | **29.3** | 109.1 | 0.098 / 0.082 |
| cold-start stores | 41.8 | 114.9 | **24.7** | 121.3 | 0.107 / 0.082 |

- Effect recovery on the test randomized slots (same population as the manifest): bias at 20 % −39 units
  (LightGBM +40, diff-in-means −76, naive observational +152).
- Margin effect split at 10 % (val): bread −217 € predicted vs −223 € true; other categories −118 € vs +165 €;
  total −335 € vs −58 €.
- Heads on *real* future embeddings reach bread WAPE 0.048; on *predicted* embeddings 0.103. About half of the
  forecast error is the world model's.
- Censored and uncensored weeks have the same error (0.101 vs 0.103): the Tobit head leaves no visible bias.
- Cold-start stores (no store ID): CATE Spearman 0.80 vs LightGBM 0.82; forecast WAPE 0.107 vs 0.082.
- Representation (test): traffic-shock probe R² 0.66 (floors −0.00 / −0.02), `bread_beta` 0.59 (static floor
  0.54), kNN next-week bread WAPE 0.171 in z-space vs 0.187 on static features.

## Ablations (17 entries × 3 seeds)

Reduced budget so the grid fits on a 4-core CPU: pretraining and world-model stages 8 epochs (instead of 15/20),
ensemble of 2 (instead of 3), one thread per run. Full table: [`runs/ablations/summary.md`](runs/ablations/summary.md).
Differences below ~15 € regret are within seed noise.

| run | test regret € | val regret € | WAPE bread h1 | CATE Spearman | Δ margin bias 10 % € |
|---|---|---|---|---|---|
| never discount | **38.3** | **37.7** | – | – | – |
| wm_no_action | 38.3 ± 0.0 | 37.7 ± 0.0 | 0.136 | – (constant) | −218 |
| observational_only | 42.5 ± 3.7 | 49.4 ± 9.4 | 0.106 | 0.865 | −342 |
| no_rollout_loss | 59 ± 14 | 65 | 0.099 | 0.882 | −217 |
| enc_masked_ae | 63 ± 16 | 42 ± 2 | 0.079 | 0.895 | −172 |
| enc_linear | 64 ± 15 | 64 ± 19 | 0.092 | 0.885 | −226 |
| **reference** (jepa + transformer) | **66 ± 10** | 60 ± 12 | 0.099 | 0.882 | −56 |
| enc_random | 67 ± 17 | 48 | 0.084 | 0.904 | −162 |
| no_store_id | 67 ± 21 | 56 | 0.098 | 0.882 | −33 |
| store_id_no_dropout | 69 ± 2 | 65 | 0.098 | 0.889 | +15 |
| no_censoring | 71 ± 12 | 64 | 0.100 | 0.864 | −31 |
| no_bread_weighting | 73 ± 6 | 65 | 0.102 | 0.856 | +52 |
| joint_finetune | 78 ± 16 | 75 ± 23 | 0.093 | 0.883 | −28 |
| enc_supervised | 82 ± 29 | 60 | **0.078** | **0.925** | −129 |
| wm_gru | 86 ± 14 | 123 | 0.150 | 0.781 | +104 |
| always 10 % | 98.7 | 95.6 | – | – | – |
| LightGBM (baseline) | 103 ± 6 | 76 ± 11 | 0.081 | 0.893 | −16 |
| keep_price_in_targets | 112 ± 47 | 117 ± 81 | 0.095 | 0.877 | +2 |
| wm_direct | 199 ± 15 | 186 ± 23 | 0.085 | 0.895 | +167 |
| wm_additive | 561 ± 0.5 | 528 | 0.174 | 0.779 | +1 060 |

What this shows:

- **The simpler model wins.** `wm_no_action` (actions zeroed) never discounts and ties "never discount", the lowest
  regret of any model. It is correctly flagged as ignoring the action. `observational_only` comes next for the same
  reason: confounding makes it so pessimistic that it almost never discounts.
- **Encoder choice does not matter for planning.** JEPA, masked-AE, linear and random frozen encoders are within
  one standard deviation of each other on test regret. Masked-AE forecasts better (0.079 vs 0.099) and has the
  lowest val regret among models that act (42 €). The supervised encoder has the best forecast and effect ranking but
  a noisy, poor planning regret.
- **The world-model architecture does matter.** The block-causal transformer beats the GRU (86 €), the direct
  model without latent dynamics (199 €) and the additive model (561 €). Additive fails because its action delta
  cannot depend on the store's state, so it over-predicts the margin gain of discounting by ~1 000 €. Direct
  over-predicts it by ~170 € and discounts 75 % of store-weeks.
- **Dropping prices from the world-model targets matters.** Keeping them nearly doubles regret and makes it unstable
  across seeds (112 ± 47 €).
- **Rollout loss, bread weighting, store-ID choices, censoring handling and joint finetuning** all change regret by
  less than the seed noise.
- **LightGBM** has the smallest *average* margin bias but the worst *ranking* of store-weeks among the reasonable
  planners (oracle match 51 % vs 60 % for the reference), so its choices are poor.

## Follow-up: is the margin error confounding? (`configs/ablations_extra.yaml`, 3 seeds)

Margin effect of a 10 % discount on the **other seven categories**, € per store-week (truth: +165 on val,
+177 on test), and the bread margin effect (truth: −223 val / −237 test):

| randomized-week weight | val other-cat effect | val bread effect | val regret € | test regret € | picks 0 % (test) |
|---|---|---|---|---|---|
| 0 (`observational_only`) | −122 ± 160 | −176 | 49 ± 9 | 43 ± 4 | 100 % |
| 3 (reference) | −49 ± 177 | −102 | 60 ± 12 | 66 ± 10 | 80 % |
| 10 | +20 ± 207 | −65 | 68 ± 16 | 71 ± 14 | 70 % |
| 30 | +146 ± 189 | −42 | 88 ± 18 | 75 ± 8 | 60 % |

The other-category effect moves towards the truth as the randomized weeks get more weight, which supports the
confounding explanation. But the bread margin effect drifts the other way (too optimistic), the seed spread is
large, and val regret gets worse. Re-weighting one shared model is not the fix.

## Is the data informative enough? (`python -m wmpoc.eval.data_audit`)

Full audit: [`runs/data_audit/report.md`](runs/data_audit/report.md). The generator looks correct (no bugs found);
the problem is how little the data says about the quantity the planner needs.

- **The prize is tiny and mostly store-level.** The oracle gains 38 € (0.10 %) over never discounting. A fixed
  level per store, chosen with hindsight, already captures 29 € of it (regret 9 €); week-to-week timing adds ~9 €.
  A 10 % discount pays off in 32 % of store-weeks; its true effect ranges from −246 € (p10) to +116 € (p90).
- **Confounding is real and flips the sign of the cross-category effect.** In observational weeks, discounted weeks
  have 1.6 % less hidden traffic. A store + week fixed-effects regression on those weeks gives −1.1 % for the other
  categories' margin at 10 % (true +0.56 %): the same error the world model makes.
- **The randomized experiment is unbiased but small.** From the training period (309–322 slot windows per level):
  bread margin effect −3.9 % ± 1.3 % at 10 % (true −4.4 %), well identified; other categories +0.65 % ± 0.83 %
  (true +0.56 %), not significant; total margin +0.02 % ± 0.84 % (true −0.18 %).
- **Required sample size.** The experiment pins down the average 10 % effect on 4-week total margin to ± 314 € per
  store-week. Capturing a 38 € headroom needs about ± 38 €: roughly **67× more randomized slot windows** for the
  average effect alone, more for store-level decisions. The planner can only win by leaning on structure (the
  precisely estimated bread effect), not by learning the total effect from the data.

## Fixes tried

**Option 1 — hybrid planner** (`world_model_hybrid`, evaluated in every run). Bread margin from the world model;
the other categories' margin = the world model's no-discount prediction × (1 + r(d)), with r(d) estimated only from
the randomized slots of the training period (+0.65 %, +1.12 %, +0.67 % at 10/20/30 %).

| run | regret € val: WM → hybrid | regret € test: WM → hybrid | Δ margin bias 10 % € (test): WM → hybrid |
|---|---|---|---|
| full-budget reference (`runs/poc_tiny`) | 46.9 → **39.2** | 48.7 → 45.1 | −188 → +85 |
| reduced-budget reference, 3 seeds | 60 ± 12 → 61 ± 14 | 66 ± 10 → 65 ± 11 | −56 → +162 |
| never discount | 37.7 | 38.3 | – |

The hybrid removes the confounded cross-category error, and the full-budget model comes within 1.5 € of "never
discount" on val. It still does not beat it. The next error appears: the world model's **bread margin** effect is
too optimistic (full budget −189 € vs −237 € true on test; reduced budget −112 €). Before, the two errors partly
cancelled; now the hybrid over-discounts (0 % chosen in 71 % / 39 % of store-weeks vs 68 % optimal).

**Option 2 — effect learned only from randomized windows** (`model.world_model.type: randomized_effect`, 3 seeds,
reduced budget). Baseline transformer with actions zeroed, trained on all windows, plus a per-category effect
module that only receives gradients from windows whose horizon lies inside the experiment.

| | test regret € | val regret € | CATE Spearman | other-cat effect 10 % (test, true +177) | bread margin effect 10 % (test, true −237) |
|---|---|---|---|---|---|
| reference | 66 ± 10 | 60 ± 12 | 0.88 | −5 | −112 |
| randomized_effect | 102 ± 24 | 144 ± 18 | 0.77 | +86 | −54 |
| randomized_effect + hybrid | 94 ± 8 | 95 ± 11 | – | (experimental) | −54 |

It moves the cross-category effect the right way, but ~3 000 overlapping randomized windows are not enough to learn
the full effect in latent space: the bread margin effect is badly under-estimated and store ranking gets worse.
This is the same information limit the audit shows; the representation does not get around it.

## Round 2: fixing the margin effect, full budget, richer data

### Where the bread margin error comes from

Decomposing the 10 % bread margin effect (full-budget reference, test): price loss −309 € (true −310), waste
−86 € (true −92), volume +210 € (true +163). The volume error is the **pull-forward**: the model predicts
weeks 1–3 lose [−28, −19, −13] units against a true [−91, −29, 0]. The randomized slots measure the pull-forward
precisely (pooled share of the week-0 uplift: week 1 −0.325, t = −19; week 2 −0.110, t = −6; week 3 not
significant, set to 0), close to the generator's 30 % / 10 % without being told.

### Planners compared (all computed from the same trained world models; regret € per store-week, mean ± sd)

- **world model**: plain rollouts.
- **hybrid**: world-model bread + experimental cross-category effect (per level).
- **structured**: hybrid + experimental pull-forward; bread margin from week-0 uplift, waste and known prices.
- **structured pooled**: structured with a pooled log-linear cross-category slope (one parameter, less noise).
- **calibrated**: structured pooled + week-0 uplift and waste rescaled per level to the experiment's averages.

| models (seeds) | split | world model | hybrid | structured | structured pooled | calibrated | LightGBM | never discount |
|---|---|---|---|---|---|---|---|---|
| full budget, original data (3) | val | 44.2 ± 4.3 | 53.6 ± 16 | **32.4 ± 1.5** | 37.0 ± 0.5 | 37.0 ± 0.5 | 76 | 37.7 |
| | test | 46.6 ± 3.8 | 56.3 ± 11 | **35.6 ± 1.5** | 37.9 ± 0.2 | 37.8 ± 0.2 | 103 | 38.3 |
| | test, cold-start | 41.1 | 59.7 | 31.6 | **23.7** | **23.7** | 101 | 24.7 |
| full budget, rich data (2) | val | 64.7 ± 18 | 71.0 ± 9 | 51.0 ± 0.4 | **32.7 ± 0.5** | **32.6 ± 0.2** | 116 | 37.0 |
| | test | 72.2 ± 23 | 78.5 ± 14 | 56.2 ± 3.5 | **34.9 ± 0.4** | **34.9 ± 0.4** | 119 | 40.8 |
| | test, cold-start | 75.1 | 94.9 | 68.7 | 25.7 | **24.1** | 134 | 25.7 |
| reduced budget, reference (3) | test | 66.2 | 65.0 | 37.9 ± 3.0 | 37.8 ± 0.2 | 37.8 | 103 | 38.3 |
| reduced, enc_masked_ae (3) | test | 63.4 | 53.1 | 35.7 ± 1.2 | 36.5 ± 0.5 | 36.4 | 103 | 38.3 |
| reduced, enc_linear (3) | test | 64.3 | 59.4 | 33.1 ± 0.8 | 37.1 ± 0.4 | 37.1 | 103 | 38.3 |
| reduced, enc_supervised (3) | test | 82.2 | 53.1 | **32.7 ± 1.2** | 37.1 ± 0.4 | 37.1 | 103 | 38.3 |
| reduced, enc_random (3) | test | 66.8 | 51.1 | 34.5 ± 0.6 | 36.7 ± 0.4 | 36.8 | 103 | 38.3 |
| reduced, randomized_effect (3) | test | 102.0 | 94.0 | 39.7 ± 1.3 | 38.3 ± 0.0 | 38.3 | 103 | 38.3 |

### Is the remaining error the world model or the data? (sensitivity, full budget)

Same world-model predictions, structured planner, only the cross-category effect r(d) changed:

| r(d) used | original data val / test | rich data val / test |
|---|---|---|
| never discount | 37.7 / 38.3 | 37.0 / 40.8 |
| per-level estimate (10 %: +0.65 % ± 0.83 / rich +0.81 % ± 0.44) | 31.3 / 34.6 | 50.8 / 53.7 |
| pooled slope (10 %: +0.25 % ± 0.25 / rich +0.43 % ± 0.14) | 37.5 / 38.1 | 33.0 / 35.1 |
| zero | 37.7 / 38.3 | 34.3 / 37.5 |
| **true** (+0.56 %, diagnostic only) | 33.6 / 34.6 | 34.4 / 35.3 |

With the true cross effect the stack beats "never discount" on both datasets by 3–5 €, so the world model's own
part (store-specific bread uplift, waste, baseline) is good enough. What decides the result is one number the
experiment measures with ± 130–250 € of noise per store-week. On the original data the per-level estimate landed
close to the truth by chance; on the rich data it did not. Choosing between estimators by their test result would
be tuning on test, so both are reported; the pooled slope is the one that never makes things worse.

### What the three steps showed

1. **Bread margin fix (structured planner):** removes the pull-forward error. Combined with any reasonable
   cross-category estimate it turns the world model from the worst sensible planner into one that matches or beats
   "never discount". Encoders matter again here: with the structured planner the supervised encoder is best
   (test 32.7 €), JEPA the worst (37.9 €), but differences are within ~2 sd. Calibrating the average bread effects
   to the experiment changes little (calibration factors 0.9–1.2) and only helps on cold-start stores.
2. **Full budget, 3 seeds:** more training helps the plain world model (test 47 vs 66 € reduced) and gives the
   best structured result (35.6 ± 1.5 €). Seeds agree closely once the structured planner is used (sd ≤ 1.5 €).
3. **Richer data (4× randomized slots, `data_rich/`):** the experiment-based parts become precise enough to act
   on: the pooled planner discounts 16 % of store-weeks (all at 10 %), matches the oracle in 66 % and beats "never
   discount" by ~6 € (≈ 14 % of the headroom) on val and test, both seeds. The plain world model gets *worse* on
   the richer data (72 € test): with less confounding its two errors no longer cancel.

## What to try next

1. **Shrink the cross-category effect towards a structural prior** (e.g. one halo parameter for all categories,
   or a hierarchical prior across stores) instead of choosing between per-level and pooled estimates.
2. **Put the pull-forward into the world model** (e.g. an auxiliary loss on randomized slots for weeks 1–3), so
   the structured planner's experimental piece is learned rather than bolted on.
3. **Store-level decisions:** the audit shows a fixed level per store captures ~75 % of the headroom; a planner
   that pools a store's weeks before deciding may be more robust than per store-week decisions.
4. **More experimentation** remains the lasting fix: the gain grew from ~0 to ~6 € with 4× the randomized weeks.

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
- **LightGBM** has no store ID (static store features stand in) and predicts total margin directly, as
  specified. Its planner uses that margin model.

Runs
- **Ablation budget reduced** (8 + 8 epochs, ensemble of 2, see above) to fit the 17 × 3 grid on 4 CPU cores. Runs
  with identical pretraining settings share one pretraining checkpoint (bit-identical to re-running it).
- **Holiday probe is undefined on test**: no national holiday falls on a test origin week; it is reported for val.
- **Planner variants** (`planning.hybrid_other_effects`, `structured_bread`, `calibrate_bread`, new keys) are
  evaluated next to the plain world model in every run; they use experiment estimates from training-period slot
  weeks only (`wmpoc/baselines/experimental_effects.py`). Earlier runs were re-evaluated from their checkpoints.
- **`data_rich/`** (`--cohort-frac 0.9 --experiment-windows 10-154`, new generator options whose defaults reproduce
  `data/` byte for byte) and `configs/poc_tiny_rich.yaml` are additions to the brief.
- **Follow-up grid** `configs/ablations_extra.yaml` (randomized-week weight 10 and 30) is an addition to the brief,
  run with the same budget; its summary is `runs/ablations/summary_extra.md`.
- **The full-budget reference** was evaluated a second time from its checkpoints (`--eval-only`) after the
  bread / other-category margin split was added to the evaluation; the training was not repeated.
