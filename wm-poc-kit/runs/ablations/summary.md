# Ablation summary

Mean ± std over seeds, sorted by test planning regret (lower is better). Planning regret = true 4-week total margin of the oracle level minus that of the chosen level, € per store-week. Baseline rows (LightGBM, fixed policies) use the reference runs' evaluation windows. Model choices should be made on the val column, not the test column.

| run | seeds | test WAPE bread h1 | ATE bias 20% (test randomized) | CATE Spearman (test) | planning regret € (test) | planning regret € (val) | oracle match (test) | action sensitivity (test) | Δ margin 4w bias 10% € (test) | runs ignoring action |
|---|---|---|---|---|---|---|---|---|---|---|
| **oracle** | 3 | – | – | – | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | – | – | – |
| **never discount** | 3 | – | – | – | 38.3 ± 0.0 | 37.7 ± 0.0 | – | – | – | – |
| wm_no_action | 3 | 0.136 ± 0.002 | -656.9 ± 0.0 | – | 38.3 ± 0.0 | 37.7 ± 0.0 | 0.678 ± 0.000 | 0.000 ± 0.000 | -217.7 ± 4.4 | 3 |
| enc_masked_ae | 3 | 0.079 ± 0.001 | 53.5 ± 33.0 | 0.895 ± 0.010 | 63.4 ± 15.7 | 42.3 ± 1.7 | 0.614 ± 0.032 | 1.048 ± 0.057 | -172.0 ± 165.5 | 0 |
| enc_linear | 3 | 0.092 ± 0.001 | 89.5 ± 19.6 | 0.885 ± 0.013 | 64.3 ± 15.3 | 63.5 ± 19.3 | 0.625 ± 0.032 | 1.105 ± 0.008 | -226.1 ± 122.4 | 0 |
| reference | 3 | 0.099 ± 0.001 | -7.221 ± 25.704 | 0.882 ± 0.009 | 66.2 ± 10.3 | 60.3 ± 11.9 | 0.596 ± 0.033 | 0.956 ± 0.053 | -56.4 ± 81.5 | 0 |
| **always 10%** | 3 | – | – | – | 98.7 ± 0.0 | 95.6 ± 0.0 | – | – | – | – |
| **lightgbm** (baseline) | 3 | 0.081 ± 0.000 | 40.7 ± 2.9 | 0.893 ± 0.006 | 102.9 ± 5.9 | 76.2 ± 10.9 | 0.512 ± 0.019 | 0.944 ± 0.008 | -16.0 ± 6.6 | – |
| keep_price_in_targets | 3 | 0.095 ± 0.002 | 36.3 ± 38.6 | 0.877 ± 0.028 | 112.1 ± 47.4 | 117.1 ± 81.3 | 0.503 ± 0.127 | 0.932 ± 0.062 | 2.357 ± 203.134 | 0 |
| wm_direct | 3 | 0.085 ± 0.000 | 42.8 ± 7.4 | 0.895 ± 0.016 | 198.7 ± 14.9 | 186.1 ± 23.1 | 0.291 ± 0.015 | 1.066 ± 0.026 | 166.5 ± 7.1 | 0 |
