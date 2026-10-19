# Ablation summary

Mean ± std over seeds, sorted by test planning regret (lower is better). Planning regret = true 4-week total margin of the oracle level minus that of the chosen level, € per store-week. Baseline rows (LightGBM, fixed policies) use the reference runs' evaluation windows. Model choices should be made on the val column, not the test column.

| run | seeds | test WAPE bread h1 | ATE bias 20% (test randomized) | CATE Spearman (test) | planning regret € (test) | planning regret € (val) | oracle match (test) | action sensitivity (test) | Δ margin 4w bias 10% € (test) | hybrid regret € (test) | hybrid regret € (val) | structured regret € (test) | structured regret € (val) | structured pooled regret € (test) | structured pooled regret € (val) | calibrated regret € (test) | calibrated regret € (val) | runs ignoring action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **oracle** | 3 | – | – | – | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | – | – | – | – | – | – | – | – | – | – | – |
| **never discount** | 3 | – | – | – | 38.3 ± 0.0 | 37.7 ± 0.0 | – | – | – | – | – | – | – | – | – | – | – | – |
| reference | 3 | 0.099 ± 0.001 | -7.221 ± 25.704 | 0.882 ± 0.009 | 66.2 ± 10.3 | 60.3 ± 11.9 | 0.596 ± 0.033 | 0.956 ± 0.053 | -56.4 ± 81.5 | 65.0 ± 10.8 | 60.8 ± 14.3 | 37.9 ± 3.0 | 33.7 ± 0.1 | 37.8 ± 0.2 | 36.9 ± 0.3 | 37.8 ± 0.3 | 36.9 ± 0.2 | 0 |
| rand_weight_10 | 3 | 0.101 ± 0.001 | 10.4 ± 21.0 | 0.864 ± 0.002 | 71.0 ± 14.2 | 67.9 ± 16.3 | 0.546 ± 0.081 | 0.953 ± 0.022 | 28.3 ± 154.7 | – | – | – | – | – | – | – | – | 0 |
| rand_weight_30 | 3 | 0.104 ± 0.001 | 17.6 ± 15.7 | 0.853 ± 0.007 | 74.9 ± 7.9 | 88.0 ± 18.0 | 0.506 ± 0.075 | 0.949 ± 0.018 | 119.6 ± 124.3 | – | – | – | – | – | – | – | – | 0 |
| **always 10%** | 3 | – | – | – | 98.7 ± 0.0 | 95.6 ± 0.0 | – | – | – | – | – | – | – | – | – | – | – | – |
| wm_randomized_effect | 3 | 0.099 ± 0.001 | -24.2 ± 13.6 | 0.765 ± 0.035 | 102.0 ± 24.4 | 143.6 ± 18.0 | 0.536 ± 0.044 | 0.929 ± 0.046 | 92.7 ± 133.3 | 94.0 ± 8.2 | 94.9 ± 11.3 | 39.7 ± 1.3 | 37.6 ± 0.1 | 38.3 ± 0.0 | 37.7 ± 0.0 | 38.3 ± 0.0 | 37.7 ± 0.0 | 0 |
| **lightgbm** (baseline) | 3 | 0.081 ± 0.000 | 40.7 ± 2.9 | 0.893 ± 0.006 | 102.9 ± 5.9 | 76.2 ± 10.9 | 0.512 ± 0.019 | 0.944 ± 0.008 | -16.0 ± 6.6 | – | – | – | – | – | – | – | – | – |
