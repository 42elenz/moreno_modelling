# Results

Data: synthetic panel, 200 stores × 156 weeks × 8 categories, known ground truth. Test = origins in weeks 130–152
(4 600 store-weeks with counterfactuals), val = weeks 104–126. All numbers are on **test, all stores** unless stated.
Planning regret = true 4-week total margin of the oracle discount level minus that of the chosen level, € per
store-week, lower is better. Model choices should be read off the val column; test is reported for completeness.

## Short answer

**The world-model stack does not beat the simplest policy.** "Never discount" has a test regret of 38 €. The
reference world model reaches 49 € at full budget and 66 ± 10 € in the ablation budget; no variant goes below 38 €.
The stack does beat the LightGBM planner by a wide margin (103–107 €). The headroom is small: the oracle beats
"never discount" by only 38 € on a 37 245 € 4-week margin (0.1 %).

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

7. **Fixes and data check (follow-up).** Taking the cross-category effect from the experiment (hybrid planner)
   brings the full-budget model to 39 € val regret, level with "never discount" but not better; learning the
   whole effect from randomized windows only does worse (102 €). The data audit explains why: the experiment
   identifies the average margin effect only to ± 314 € per store-week while the decision needs ~± 38 €.

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

## What to try next

1. **Get the bread margin effect right before anything else.** Bread units are right but the 4-week bread margin
   effect is too optimistic, most likely through pull-forward and waste in weeks 1–3. A structured decoder (margin
   from predicted units, stock and the known pull-forward window, or the bread effect fitted to the randomized slots
   where it is well identified, t = 3.5–17) is the obvious next step.
2. **Use structure for the cross-category effect** instead of learning it: the hybrid's experimental r(d), or a
   pooled "halo" parameter shared by all categories, since the experiment cannot resolve each category alone.
3. **Train longer / bigger ensembles** — the full-budget run is clearly better than the reduced ablation budget
   (hybrid val regret 39 vs 61 €), so the reduced-budget ablations understate what the stack can do.
4. **More experimentation is the lasting fix.** In the simulator: more stores in the cohort or more slot weeks
   (`data_gen/generate_fake_data.py`), then re-run the audit; the audit says ~67× for the average effect.

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
- **Follow-up grid** `configs/ablations_extra.yaml` (randomized-week weight 10 and 30) is an addition to the brief,
  run with the same budget; its summary is `runs/ablations/summary_extra.md`.
- **The full-budget reference** was evaluated a second time from its checkpoints (`--eval-only`) after the
  bread / other-category margin split was added to the evaluation; the training was not repeated.
