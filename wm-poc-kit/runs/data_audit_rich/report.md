# Data audit

Does the synthetic panel contain enough information to plan bread discounts?

## 1. Experiment size (randomized slot windows, policy start week drawn at random)

| split | 0% | 10% | 20% | 30% | share of windows |
|---|---|---|---|---|---|
| train | 1217 | 1173 | 1227 | 1180 | 0.303 |
| val | 383 | 361 | 325 | 352 | 0.309 |
| test | 317 | 360 | 379 | 369 | 0.310 |

## 2. Can the training-period experiment identify the effects?

Relative change of the 4-week margin vs 0 %, from slot windows of training weeks × training stores (store and week fixed effects), against the truth (mean over all counterfactual store-weeks).

| margin of | level | experimental estimate | s.e. | true | true effect / s.e. |
|---|---|---|---|---|---|
| bread | 10% | -3.98 % | ± 0.64 % | -4.41 % | -6.911 |
| bread | 20% | -11.65 % | ± 0.63 % | -10.88 % | -17.245 |
| bread | 30% | -20.12 % | ± 0.64 % | -19.94 % | -31.129 |
| other | 10% | +0.81 % | ± 0.44 % | +0.56 % | 1.286 |
| other | 20% | +0.72 % | ± 0.43 % | +1.15 % | 2.659 |
| other | 30% | +1.48 % | ± 0.44 % | +1.76 % | 4.022 |
| total | 10% | +0.13 % | ± 0.44 % | -0.17 % | -0.395 |
| total | 20% | -1.02 % | ± 0.44 % | -0.63 % | -1.445 |
| total | 30% | -1.57 % | ± 0.44 % | -1.44 % | -3.255 |

## 3. Confounding in the observational weeks

Same regression on non-experiment, non-flyer weeks with policy [d, 0, 0, 0] (training period).

| margin of | level | observational estimate | true | n |
|---|---|---|---|---|
| bread | 10% | -3.86 % | -4.41 % | 63 |
| bread | 20% | -8.53 % | -10.88 % | 34 |
| bread | 30% | -15.25 % | -19.94 % | 13 |
| other | 10% | +0.66 % | +0.56 % | 63 |
| other | 20% | -0.53 % | +1.15 % | 34 |
| other | 30% | +5.42 % | +1.76 % | 13 |

Hidden traffic in discounted vs undiscounted observational weeks: -1.58 % (mean AR(1) shock -0.0136 vs +0.0023).

## 4. Headroom (test)

| policy | regret € per store-week |
|---|---|
| oracle (per store-week) | 0.000 |
| best fixed level per store (needs per-store truth) | 9.860 |
| best single level for everyone (0%) | 40.850 |
| never discount | 40.850 |

Mean 4-week total margin 37,128 €; the oracle gains 0.11 % over never discounting. True effect of 10 % on 4-week total margin: p10 -242 €, median -58 €, p90 +120 €; 10 % pays off in 32% of store-weeks.

## 5. How much experiment would be enough?

The training-period experiment pins down the 10 % effect on 4-week total margin to ± 163 € (1 s.e.). Capturing the headroom needs about ± 41 €, i.e. roughly 16× as many slot windows as the 2390 at 0 % and 10 % today, for the *average* effect alone; store-level decisions need more.
