# Data audit

Does the synthetic panel contain enough information to plan bread discounts?

## 1. Experiment size (randomized slot windows, policy start week drawn at random)

| split | 0% | 10% | 20% | 30% | share of windows |
|---|---|---|---|---|---|
| train | 309 | 275 | 312 | 322 | 0.077 |
| val | 144 | 130 | 146 | 158 | 0.126 |
| test | 177 | 184 | 185 | 171 | 0.156 |

## 2. Can the training-period experiment identify the effects?

Relative change of the 4-week margin vs 0 %, from slot windows of training weeks × training stores (store and week fixed effects), against the truth (mean over all counterfactual store-weeks).

| margin of | level | experimental estimate | s.e. | true | true effect / s.e. |
|---|---|---|---|---|---|
| bread | 10% | -3.94 % | ± 1.26 % | -4.41 % | -3.500 |
| bread | 20% | -10.91 % | ± 1.21 % | -10.90 % | -9.012 |
| bread | 30% | -20.27 % | ± 1.21 % | -19.97 % | -16.565 |
| other | 10% | +0.65 % | ± 0.83 % | +0.56 % | 0.671 |
| other | 20% | +1.12 % | ± 0.80 % | +1.15 % | 1.433 |
| other | 30% | +0.67 % | ± 0.80 % | +1.76 % | 2.208 |
| total | 10% | +0.02 % | ± 0.84 % | -0.18 % | -0.208 |
| total | 20% | -0.58 % | ± 0.81 % | -0.63 % | -0.782 |
| total | 30% | -2.28 % | ± 0.81 % | -1.45 % | -1.793 |

## 3. Confounding in the observational weeks

Same regression on non-experiment, non-flyer weeks with policy [d, 0, 0, 0] (training period).

| margin of | level | observational estimate | true | n |
|---|---|---|---|---|
| bread | 10% | -5.85 % | -4.41 % | 335 |
| bread | 20% | -10.16 % | -10.90 % | 163 |
| bread | 30% | -19.78 % | -19.97 % | 65 |
| other | 10% | -1.12 % | +0.56 % | 335 |
| other | 20% | +0.78 % | +1.15 % | 163 |
| other | 30% | -0.12 % | +1.76 % | 65 |

Hidden traffic in discounted vs undiscounted observational weeks: -1.61 % (mean AR(1) shock -0.0133 vs +0.0029).

## 4. Headroom (test)

| policy | regret € per store-week |
|---|---|
| oracle (per store-week) | 0.000 |
| best fixed level per store (needs per-store truth) | 9.387 |
| best single level for everyone (0%) | 38.324 |
| never discount | 38.324 |

Mean 4-week total margin 37,207 €; the oracle gains 0.10 % over never discounting. True effect of 10 % on 4-week total margin: p10 -246 €, median -60 €, p90 +116 €; 10 % pays off in 32% of store-weeks.

## 5. How much experiment would be enough?

The training-period experiment pins down the 10 % effect on 4-week total margin to ± 314 € (1 s.e.). Capturing the headroom needs about ± 38 €, i.e. roughly 67× as many slot windows as the 584 at 0 % and 10 % today, for the *average* effect alone; store-level decisions need more.
