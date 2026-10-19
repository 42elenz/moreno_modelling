# Ablation summary

Mean ± std over seeds, sorted by test planning regret (lower is better). Planning regret = true 4-week total margin of the oracle level minus that of the chosen level, € per store-week. Baseline rows (LightGBM, fixed policies) use the reference runs' evaluation windows. Model choices should be made on the val column, not the test column.

| run | seeds | test WAPE bread h1 | ATE bias 20% (test randomized) | CATE Spearman (test) | planning regret € (test) | planning regret € (val) | oracle match (test) | action sensitivity (test) | Δ margin 4w bias 10% € (test) | runs ignoring action |
|---|---|---|---|---|---|---|---|---|---|---|
| **oracle** | 3 | – | – | – | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | – | – | – |
| **never discount** | 3 | – | – | – | 38.3 ± 0.0 | 37.7 ± 0.0 | – | – | – | – |
| reference | 3 | 0.099 ± 0.001 | -7.221 ± 25.704 | 0.882 ± 0.009 | 66.2 ± 10.3 | 60.3 ± 11.9 | 0.596 ± 0.033 | 0.956 ± 0.053 | -56.4 ± 81.5 | 0 |
| rand_weight_30 | 1 | 0.103 ± 0.000 | 22.7 ± 0.0 | 0.852 ± 0.000 | 79.6 ± 0.0 | 69.6 ± 0.0 | 0.476 ± 0.000 | 0.946 ± 0.000 | 129.7 ± 0.0 | 0 |
| rand_weight_10 | 1 | 0.100 ± 0.000 | 17.0 ± 0.0 | 0.863 ± 0.000 | 79.8 ± 0.0 | 62.7 ± 0.0 | 0.497 ± 0.000 | 0.957 ± 0.000 | 94.6 ± 0.0 | 0 |
| **always 10%** | 3 | – | – | – | 98.7 ± 0.0 | 95.6 ± 0.0 | – | – | – | – |
| **lightgbm** (baseline) | 3 | 0.081 ± 0.000 | 40.7 ± 2.9 | 0.893 ± 0.006 | 102.9 ± 5.9 | 76.2 ± 10.9 | 0.512 ± 0.019 | 0.944 ± 0.008 | -16.0 ± 6.6 | – |
