# Run report: poc_tiny

encoder `jepa` · world model `transformer` × 3 · target_drops_price True · censoring `tobit` · seed 0 · stores 200 · runtime 5,239 s

## Headline (test, all stores)

Lowest planning regret: **structured (hybrid + experimental pull-forward)** (34.9 € per store-week). world model 48.7, hybrid (WM bread + experimental cross effects) 45.1, structured (hybrid + experimental pull-forward) 34.9, LightGBM 107, never discount 38.3, always 10% 98.7.

| model | WAPE bread h1 | WAPE margin 4w | ATE bias 20% (randomized) | CATE Spearman | action sensitivity | planning regret € | oracle match |
|---|---|---|---|---|---|---|---|
| world_model | 0.103 | 0.078 | -38.548 | 0.910 | 0.874 | 48.696 | 0.644 |
| world_model_hybrid | 0.103 | 0.078 | -38.548 | 0.910 | 0.874 | 45.081 | 0.577 |
| world_model_structured | 0.103 | 0.078 | -38.548 | 0.910 | 0.874 | 34.881 | 0.653 |
| lightgbm | 0.081 | 0.079 | 39.672 | 0.899 | 0.953 | 107 | 0.492 |

## val

### val / all (4600 windows)

**Forecast** (WAPE of bread units per horizon step; 4-week total margin)

| model | h1 | h2 | h3 | h4 | margin WAPE | margin MAE € |
|---|---|---|---|---|---|---|
| world_model | 0.110 | 0.109 | 0.111 | 0.113 | 0.079 | 2,751 |
| world_model_hybrid | 0.110 | 0.109 | 0.111 | 0.113 | 0.079 | 2,751 |
| world_model_structured | 0.110 | 0.109 | 0.111 | 0.113 | 0.079 | 2,751 |
| lightgbm | 0.087 | 0.089 | 0.090 | 0.092 | 0.080 | 2,781 |
| world_model_heads_on_real_embeddings | 0.055 | 0.054 | 0.054 | 0.055 | – | – |

**Effect recovery** (bread units in week 0, action [d,0,0,0], vs ground-truth counterfactuals)

| level | true | world_model pred | world_model_hybrid pred | world_model_structured pred | lightgbm pred | world_model bias | world_model_hybrid bias | world_model_structured bias | lightgbm bias | world_model RMSE | world_model_hybrid RMSE | world_model_structured RMSE | lightgbm RMSE | diff in means | naive obs. | true Δ margin 4w | world_model Δ margin 4w | world_model_hybrid Δ margin 4w | world_model_structured Δ margin 4w | lightgbm Δ margin 4w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 316 | 282 | 282 | 282 | 306 | -33.660 | -33.660 | -33.660 | -9.134 | 108 | 108 | 108 | 116 | 135 | – | -57.903 | -335 | -21.806 | -57.174 | -87.143 |
| 20% | 676 | 615 | 615 | 615 | 717 | -61.379 | -61.379 | -61.379 | 40.100 | 216 | 216 | 216 | 238 | 645 | – | -215 | -587 | -285 | -243 | -91.172 |
| 30% | 1,085 | 948 | 948 | 948 | 1,009 | -136 | -136 | -136 | -75.646 | 362 | 362 | 362 | 347 | 926 | – | -498 | -947 | -939 | -816 | -184 |

**Margin effect split** (Δ 4-week margin vs 0 %, € per store-week: bread vs all other categories)

| level | model | bread pred | bread true | other pred | other true | total pred | total true |
|---|---|---|---|---|---|---|---|
| 10% | world_model | -217 | -223 | -118 | 165 | -335 | -57.903 |
| 10% | world_model_hybrid | -217 | -223 | 195 | 165 | -21.806 | -57.903 |
| 10% | world_model_structured | -252 | -223 | 195 | 165 | -57.174 | -57.903 |
| 20% | world_model | -619 | -553 | 31.916 | 338 | -587 | -215 |
| 20% | world_model_hybrid | -619 | -553 | 334 | 338 | -285 | -215 |
| 20% | world_model_structured | -577 | -553 | 334 | 338 | -243 | -215 |
| 30% | world_model | -1,139 | -1,017 | 191 | 519 | -947 | -498 |
| 30% | world_model_hybrid | -1,139 | -1,017 | 200 | 519 | -939 | -498 |
| 30% | world_model_structured | -1,015 | -1,017 | 200 | 519 | -816 | -498 |

**Heterogeneity and action sensitivity**

| model | Spearman vs true per-store uplift | Spearman vs bread_beta | sensitivity ratio | ignores action |
|---|---|---|---|---|
| world_model | 0.921 | 0.589 | 0.874 | no |
| world_model_hybrid | 0.921 | 0.589 | 0.874 | no |
| world_model_structured | 0.921 | 0.589 | 0.874 | no |
| lightgbm | 0.896 | 0.453 | 0.930 | no |
(true per-store uplift vs bread_beta: Spearman 0.722)

**Planning** (true 4-week total margin of the chosen level, € per store-week)

| policy | value € | regret € | oracle match | levels chosen |
|---|---|---|---|---|
| oracle | 34,848 | 0.000 | 1.000 |  |
| never discount | 34,810 | 37.661 | 0.668 |  |
| always 10% | 34,752 | 95.564 | 0.217 |  |
| world_model | 34,801 | 46.900 | 0.651 | 0% 92%, 10% 3%, 20% 3%, 30% 2% |
| world_model_hybrid | 34,809 | 39.231 | 0.627 | 0% 78%, 10% 21%, 20% 2%, 30% 0% |
| world_model_structured | 34,816 | 31.660 | 0.674 | 0% 88%, 10% 12%, 20% 0%, 30% 0% |
| lightgbm | 34,782 | 66.126 | 0.560 | 0% 73%, 10% 18%, 20% 5%, 30% 4% |
true best level shares: 0% 67%, 10% 22%, 20% 11%, 30% 1%

**Censoring** (WAPE bread h1)

| model | censored weeks | uncensored weeks | share censored |
|---|---|---|---|
| world_model | 0.107 | 0.111 | 0.189 |
| world_model_hybrid | 0.107 | 0.111 | 0.189 |
| world_model_structured | 0.107 | 0.111 | 0.189 |
| lightgbm | 0.088 | 0.087 | 0.189 |

### val / randomized (578 windows)

**Forecast** (WAPE of bread units per horizon step; 4-week total margin)

| model | h1 | h2 | h3 | h4 | margin WAPE | margin MAE € |
|---|---|---|---|---|---|---|
| world_model | 0.108 | 0.104 | 0.099 | 0.111 | 0.073 | 2,536 |
| world_model_hybrid | 0.108 | 0.104 | 0.099 | 0.111 | 0.073 | 2,536 |
| world_model_structured | 0.108 | 0.104 | 0.099 | 0.111 | 0.073 | 2,536 |
| lightgbm | 0.090 | 0.095 | 0.087 | 0.097 | 0.081 | 2,811 |
| world_model_heads_on_real_embeddings | 0.055 | 0.053 | 0.052 | 0.055 | – | – |

**Effect recovery** (bread units in week 0, action [d,0,0,0], vs ground-truth counterfactuals)

| level | true | world_model pred | world_model_hybrid pred | world_model_structured pred | lightgbm pred | world_model bias | world_model_hybrid bias | world_model_structured bias | lightgbm bias | world_model RMSE | world_model_hybrid RMSE | world_model_structured RMSE | lightgbm RMSE | diff in means | naive obs. | true Δ margin 4w | world_model Δ margin 4w | world_model_hybrid Δ margin 4w | world_model_structured Δ margin 4w | lightgbm Δ margin 4w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 303 | 259 | 259 | 259 | 283 | -43.962 | -43.962 | -43.962 | -19.224 | 99.348 | 99.348 | 99.348 | 89.291 | 135 | – | -73.699 | -399 | -65.521 | -65.633 | -109 |
| 20% | 649 | 563 | 563 | 563 | 667 | -86.061 | -86.061 | -86.061 | 18.661 | 196 | 196 | 196 | 164 | 645 | – | -243 | -685 | -345 | -254 | -113 |
| 30% | 1,039 | 856 | 856 | 856 | 957 | -183 | -183 | -183 | -82.398 | 346 | 346 | 346 | 285 | 926 | – | -532 | -1,050 | -984 | -813 | -203 |

**Margin effect split** (Δ 4-week margin vs 0 %, € per store-week: bread vs all other categories)

| level | model | bread pred | bread true | other pred | other true | total pred | total true |
|---|---|---|---|---|---|---|---|
| 10% | world_model | -258 | -231 | -140 | 158 | -399 | -73.699 |
| 10% | world_model_hybrid | -258 | -231 | 193 | 158 | -65.521 | -73.699 |
| 10% | world_model_structured | -258 | -231 | 193 | 158 | -65.633 | -73.699 |
| 20% | world_model | -675 | -566 | -9.868 | 323 | -685 | -243 |
| 20% | world_model_hybrid | -675 | -566 | 331 | 323 | -345 | -243 |
| 20% | world_model_structured | -585 | -566 | 331 | 323 | -254 | -243 |
| 30% | world_model | -1,181 | -1,029 | 132 | 497 | -1,050 | -532 |
| 30% | world_model_hybrid | -1,181 | -1,029 | 198 | 497 | -984 | -532 |
| 30% | world_model_structured | -1,011 | -1,029 | 198 | 497 | -813 | -532 |

**Heterogeneity and action sensitivity**

| model | Spearman vs true per-store uplift | Spearman vs bread_beta | sensitivity ratio | ignores action |
|---|---|---|---|---|
| world_model | 0.941 | 0.669 | 0.824 | no |
| world_model_hybrid | 0.941 | 0.669 | 0.824 | no |
| world_model_structured | 0.941 | 0.669 | 0.824 | no |
| lightgbm | 0.891 | 0.505 | 0.921 | no |
(true per-store uplift vs bread_beta: Spearman 0.739)

**Planning** (true 4-week total margin of the chosen level, € per store-week)

| policy | value € | regret € | oracle match | levels chosen |
|---|---|---|---|---|
| oracle | 34,856 | 0.000 | 1.000 |  |
| never discount | 34,828 | 27.554 | 0.758 |  |
| always 10% | 34,754 | 101 | 0.149 |  |
| world_model | 34,818 | 37.871 | 0.744 | 0% 97%, 10% 1%, 20% 1%, 30% 1% |
| world_model_hybrid | 34,824 | 31.915 | 0.708 | 0% 86%, 10% 13%, 20% 0%, 30% 0% |
| world_model_structured | 34,832 | 23.582 | 0.761 | 0% 91%, 10% 9%, 20% 0%, 30% 0% |
| lightgbm | 34,800 | 56.094 | 0.644 | 0% 80%, 10% 10%, 20% 4%, 30% 6% |
true best level shares: 0% 76%, 10% 15%, 20% 9%, 30% 0%

**Censoring** (WAPE bread h1)

| model | censored weeks | uncensored weeks | share censored |
|---|---|---|---|
| world_model | 0.118 | 0.105 | 0.227 |
| world_model_hybrid | 0.118 | 0.105 | 0.227 |
| world_model_structured | 0.118 | 0.105 | 0.227 |
| lightgbm | 0.100 | 0.087 | 0.227 |

### val / cold_start_stores (690 windows)

**Forecast** (WAPE of bread units per horizon step; 4-week total margin)

| model | h1 | h2 | h3 | h4 | margin WAPE | margin MAE € |
|---|---|---|---|---|---|---|
| world_model | 0.117 | 0.116 | 0.118 | 0.121 | 0.086 | 2,925 |
| world_model_hybrid | 0.117 | 0.116 | 0.118 | 0.121 | 0.086 | 2,925 |
| world_model_structured | 0.117 | 0.116 | 0.118 | 0.121 | 0.086 | 2,925 |
| lightgbm | 0.091 | 0.094 | 0.098 | 0.100 | 0.086 | 2,937 |
| world_model_heads_on_real_embeddings | 0.056 | 0.060 | 0.060 | 0.060 | – | – |

**Effect recovery** (bread units in week 0, action [d,0,0,0], vs ground-truth counterfactuals)

| level | true | world_model pred | world_model_hybrid pred | world_model_structured pred | lightgbm pred | world_model bias | world_model_hybrid bias | world_model_structured bias | lightgbm bias | world_model RMSE | world_model_hybrid RMSE | world_model_structured RMSE | lightgbm RMSE | diff in means | naive obs. | true Δ margin 4w | world_model Δ margin 4w | world_model_hybrid Δ margin 4w | world_model_structured Δ margin 4w | lightgbm Δ margin 4w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 282 | 263 | 263 | 263 | 290 | -19.043 | -19.043 | -19.043 | 7.902 | 105 | 105 | 105 | 120 | 122 | – | -94.959 | -307 | -23.768 | -57.294 | -91.980 |
| 20% | 603 | 577 | 577 | 577 | 673 | -26.329 | -26.329 | -26.329 | 69.256 | 219 | 219 | 219 | 263 | 510 | – | -279 | -501 | -271 | -243 | -95.221 |
| 30% | 966 | 879 | 879 | 879 | 944 | -87.307 | -87.307 | -87.307 | -22.375 | 375 | 375 | 375 | 388 | 809 | – | -575 | -769 | -903 | -813 | -179 |

**Margin effect split** (Δ 4-week margin vs 0 %, € per store-week: bread vs all other categories)

| level | model | bread pred | bread true | other pred | other true | total pred | total true |
|---|---|---|---|---|---|---|---|
| 10% | world_model | -218 | -226 | -88.684 | 131 | -307 | -94.959 |
| 10% | world_model_hybrid | -218 | -226 | 195 | 131 | -23.768 | -94.959 |
| 10% | world_model_structured | -252 | -226 | 195 | 131 | -57.294 | -94.959 |
| 20% | world_model | -605 | -550 | 104 | 271 | -501 | -279 |
| 20% | world_model_hybrid | -605 | -550 | 334 | 271 | -271 | -279 |
| 20% | world_model_structured | -577 | -550 | 334 | 271 | -243 | -279 |
| 30% | world_model | -1,103 | -994 | 334 | 419 | -769 | -575 |
| 30% | world_model_hybrid | -1,103 | -994 | 200 | 419 | -903 | -575 |
| 30% | world_model_structured | -1,012 | -994 | 200 | 419 | -813 | -575 |

**Heterogeneity and action sensitivity**

| model | Spearman vs true per-store uplift | Spearman vs bread_beta | sensitivity ratio | ignores action |
|---|---|---|---|---|
| world_model | 0.833 | 0.615 | 0.910 | no |
| world_model_hybrid | 0.833 | 0.615 | 0.910 | no |
| world_model_structured | 0.833 | 0.615 | 0.910 | no |
| lightgbm | 0.771 | 0.499 | 0.977 | no |
(true per-store uplift vs bread_beta: Spearman 0.875)

**Planning** (true 4-week total margin of the chosen level, € per store-week)

| policy | value € | regret € | oracle match | levels chosen |
|---|---|---|---|---|
| oracle | 34,205 | 0.000 | 1.000 |  |
| never discount | 34,184 | 20.474 | 0.745 |  |
| always 10% | 34,089 | 115 | 0.188 |  |
| world_model | 34,159 | 45.433 | 0.712 | 0% 92%, 10% 2%, 20% 3%, 30% 2% |
| world_model_hybrid | 34,168 | 36.272 | 0.675 | 0% 79%, 10% 19%, 20% 1%, 30% 0% |
| world_model_structured | 34,183 | 21.546 | 0.735 | 0% 86%, 10% 13%, 20% 0%, 30% 0% |
| lightgbm | 34,149 | 55.477 | 0.613 | 0% 77%, 10% 14%, 20% 5%, 30% 3% |
true best level shares: 0% 74%, 10% 19%, 20% 6%, 30% 0%

**Censoring** (WAPE bread h1)

| model | censored weeks | uncensored weeks | share censored |
|---|---|---|---|
| world_model | 0.106 | 0.120 | 0.174 |
| world_model_hybrid | 0.106 | 0.120 | 0.174 |
| world_model_structured | 0.106 | 0.120 | 0.174 |
| lightgbm | 0.086 | 0.092 | 0.174 |

### val / representation

Ridge probes on frozen z(t), out-of-fold R², folds grouped by store

| target | z(t) | calendar + store one-hot | calendar + static features |
|---|---|---|---|
| bread_beta | 0.615 | -0.005 | 0.542 |
| size_sqm | 0.959 | -0.005 | 1.000 |
| holiday | 0.733 | 1.000 | 1.000 |
| traffic_shock | 0.603 | -0.007 | -0.033 |

kNN (k=10) next-week bread units, WAPE: neighbours in z-space 0.212, neighbours on static features 0.224

## test

### test / all (4600 windows)

**Forecast** (WAPE of bread units per horizon step; 4-week total margin)

| model | h1 | h2 | h3 | h4 | margin WAPE | margin MAE € |
|---|---|---|---|---|---|---|
| world_model | 0.103 | 0.100 | 0.102 | 0.107 | 0.078 | 2,890 |
| world_model_hybrid | 0.103 | 0.100 | 0.102 | 0.107 | 0.078 | 2,890 |
| world_model_structured | 0.103 | 0.100 | 0.102 | 0.107 | 0.078 | 2,890 |
| lightgbm | 0.081 | 0.084 | 0.087 | 0.088 | 0.079 | 2,937 |
| world_model_heads_on_real_embeddings | 0.048 | 0.048 | 0.050 | 0.051 | – | – |

**Effect recovery** (bread units in week 0, action [d,0,0,0], vs ground-truth counterfactuals)

| level | true | world_model pred | world_model_hybrid pred | world_model_structured pred | lightgbm pred | world_model bias | world_model_hybrid bias | world_model_structured bias | lightgbm bias | world_model RMSE | world_model_hybrid RMSE | world_model_structured RMSE | lightgbm RMSE | diff in means | naive obs. | true Δ margin 4w | world_model Δ margin 4w | world_model_hybrid Δ margin 4w | world_model_structured Δ margin 4w | lightgbm Δ margin 4w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 322 | 323 | 323 | 323 | 309 | 0.243 | 0.243 | 0.243 | -13.412 | 102 | 102 | 102 | 100 | 252 | 340 | -60.351 | -248 | 24.884 | -35.283 | -71.053 |
| 20% | 690 | 659 | 659 | 659 | 726 | -31.520 | -31.520 | -31.520 | 35.763 | 208 | 208 | 208 | 199 | 581 | 809 | -227 | -546 | -295 | -238 | -77.769 |
| 30% | 1,105 | 966 | 966 | 966 | 1,053 | -139 | -139 | -139 | -51.847 | 362 | 362 | 362 | 309 | 965 | 1,184 | -523 | -972 | -1,026 | -855 | -148 |

**Margin effect split** (Δ 4-week margin vs 0 %, € per store-week: bread vs all other categories)

| level | model | bread pred | bread true | other pred | other true | total pred | total true |
|---|---|---|---|---|---|---|---|
| 10% | world_model | -189 | -237 | -58.890 | 177 | -248 | -60.351 |
| 10% | world_model_hybrid | -189 | -237 | 214 | 177 | 24.884 | -60.351 |
| 10% | world_model_structured | -249 | -237 | 214 | 177 | -35.283 | -60.351 |
| 20% | world_model | -662 | -588 | 116 | 361 | -546 | -227 |
| 20% | world_model_hybrid | -662 | -588 | 367 | 361 | -295 | -227 |
| 20% | world_model_structured | -605 | -588 | 367 | 361 | -238 | -227 |
| 30% | world_model | -1,245 | -1,078 | 274 | 555 | -972 | -523 |
| 30% | world_model_hybrid | -1,245 | -1,078 | 220 | 555 | -1,026 | -523 |
| 30% | world_model_structured | -1,075 | -1,078 | 220 | 555 | -855 | -523 |

**Heterogeneity and action sensitivity**

| model | Spearman vs true per-store uplift | Spearman vs bread_beta | sensitivity ratio | ignores action |
|---|---|---|---|---|
| world_model | 0.910 | 0.580 | 0.874 | no |
| world_model_hybrid | 0.910 | 0.580 | 0.874 | no |
| world_model_structured | 0.910 | 0.580 | 0.874 | no |
| lightgbm | 0.899 | 0.474 | 0.953 | no |
(true per-store uplift vs bread_beta: Spearman 0.723)

**Planning** (true 4-week total margin of the chosen level, € per store-week)

| policy | value € | regret € | oracle match | levels chosen |
|---|---|---|---|---|
| oracle | 37,245 | 0.000 | 1.000 |  |
| never discount | 37,207 | 38.324 | 0.678 |  |
| always 10% | 37,146 | 98.674 | 0.217 |  |
| world_model | 37,196 | 48.696 | 0.644 | 0% 91%, 10% 4%, 20% 3%, 30% 2% |
| world_model_hybrid | 37,200 | 45.081 | 0.577 | 0% 71%, 10% 29%, 20% 0%, 30% 0% |
| world_model_structured | 37,210 | 34.881 | 0.653 | 0% 82%, 10% 18%, 20% 0%, 30% 0% |
| lightgbm | 37,138 | 107 | 0.492 | 0% 62%, 10% 25%, 20% 2%, 30% 11% |
true best level shares: 0% 68%, 10% 22%, 20% 10%, 30% 1%

**Censoring** (WAPE bread h1)

| model | censored weeks | uncensored weeks | share censored |
|---|---|---|---|
| world_model | 0.101 | 0.103 | 0.193 |
| world_model_hybrid | 0.101 | 0.103 | 0.193 |
| world_model_structured | 0.101 | 0.103 | 0.193 |
| lightgbm | 0.078 | 0.081 | 0.193 |

### test / randomized (717 windows)

**Forecast** (WAPE of bread units per horizon step; 4-week total margin)

| model | h1 | h2 | h3 | h4 | margin WAPE | margin MAE € |
|---|---|---|---|---|---|---|
| world_model | 0.098 | 0.102 | 0.093 | 0.103 | 0.078 | 2,822 |
| world_model_hybrid | 0.098 | 0.102 | 0.093 | 0.103 | 0.078 | 2,822 |
| world_model_structured | 0.098 | 0.102 | 0.093 | 0.103 | 0.078 | 2,822 |
| lightgbm | 0.082 | 0.088 | 0.080 | 0.088 | 0.077 | 2,794 |
| world_model_heads_on_real_embeddings | 0.049 | 0.047 | 0.046 | 0.049 | – | – |

**Effect recovery** (bread units in week 0, action [d,0,0,0], vs ground-truth counterfactuals)

| level | true | world_model pred | world_model_hybrid pred | world_model_structured pred | lightgbm pred | world_model bias | world_model_hybrid bias | world_model_structured bias | lightgbm bias | world_model RMSE | world_model_hybrid RMSE | world_model_structured RMSE | lightgbm RMSE | diff in means | naive obs. | true Δ margin 4w | world_model Δ margin 4w | world_model_hybrid Δ margin 4w | world_model_structured Δ margin 4w | lightgbm Δ margin 4w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 308 | 310 | 310 | 310 | 293 | 2.274 | 2.274 | 2.274 | -15.471 | 87.176 | 87.176 | 87.176 | 84.994 | 252 | 340 | -79.780 | -270 | -3.448 | -43.447 | -65.709 |
| 20% | 657 | 618 | 618 | 618 | 697 | -38.548 | -38.548 | -38.548 | 39.672 | 175 | 175 | 175 | 163 | 581 | 809 | -264 | -597 | -349 | -256 | -73.713 |
| 30% | 1,053 | 887 | 887 | 887 | 1,028 | -166 | -166 | -166 | -24.575 | 327 | 327 | 327 | 248 | 965 | 1,184 | -569 | -1,006 | -1,072 | -864 | -146 |

**Margin effect split** (Δ 4-week margin vs 0 %, € per store-week: bread vs all other categories)

| level | model | bread pred | bread true | other pred | other true | total pred | total true |
|---|---|---|---|---|---|---|---|
| 10% | world_model | -213 | -245 | -56.483 | 165 | -270 | -79.780 |
| 10% | world_model_hybrid | -213 | -245 | 210 | 165 | -3.448 | -79.780 |
| 10% | world_model_structured | -253 | -245 | 210 | 165 | -43.447 | -79.780 |
| 20% | world_model | -709 | -602 | 112 | 339 | -597 | -264 |
| 20% | world_model_hybrid | -709 | -602 | 360 | 339 | -349 | -264 |
| 20% | world_model_structured | -616 | -602 | 360 | 339 | -256 | -264 |
| 30% | world_model | -1,287 | -1,090 | 281 | 521 | -1,006 | -569 |
| 30% | world_model_hybrid | -1,287 | -1,090 | 215 | 521 | -1,072 | -569 |
| 30% | world_model_structured | -1,079 | -1,090 | 215 | 521 | -864 | -569 |

**Heterogeneity and action sensitivity**

| model | Spearman vs true per-store uplift | Spearman vs bread_beta | sensitivity ratio | ignores action |
|---|---|---|---|---|
| world_model | 0.925 | 0.660 | 0.842 | no |
| world_model_hybrid | 0.925 | 0.660 | 0.842 | no |
| world_model_structured | 0.925 | 0.660 | 0.842 | no |
| lightgbm | 0.914 | 0.549 | 0.977 | no |
(true per-store uplift vs bread_beta: Spearman 0.735)

**Planning** (true 4-week total margin of the chosen level, € per store-week)

| policy | value € | regret € | oracle match | levels chosen |
|---|---|---|---|---|
| oracle | 36,287 | 0.000 | 1.000 |  |
| never discount | 36,257 | 29.338 | 0.764 |  |
| always 10% | 36,178 | 109 | 0.165 |  |
| world_model | 36,247 | 39.363 | 0.731 | 0% 95%, 10% 2%, 20% 1%, 30% 2% |
| world_model_hybrid | 36,247 | 39.145 | 0.656 | 0% 77%, 10% 23%, 20% 0%, 30% 0% |
| world_model_structured | 36,259 | 28.040 | 0.738 | 0% 84%, 10% 16%, 20% 0%, 30% 0% |
| lightgbm | 36,171 | 115 | 0.497 | 0% 60%, 10% 25%, 20% 2%, 30% 14% |
true best level shares: 0% 76%, 10% 16%, 20% 6%, 30% 1%

**Censoring** (WAPE bread h1)

| model | censored weeks | uncensored weeks | share censored |
|---|---|---|---|
| world_model | 0.097 | 0.099 | 0.252 |
| world_model_hybrid | 0.097 | 0.099 | 0.252 |
| world_model_structured | 0.097 | 0.099 | 0.252 |
| lightgbm | 0.076 | 0.084 | 0.252 |

### test / cold_start_stores (690 windows)

**Forecast** (WAPE of bread units per horizon step; 4-week total margin)

| model | h1 | h2 | h3 | h4 | margin WAPE | margin MAE € |
|---|---|---|---|---|---|---|
| world_model | 0.107 | 0.105 | 0.107 | 0.109 | 0.080 | 2,954 |
| world_model_hybrid | 0.107 | 0.105 | 0.107 | 0.109 | 0.080 | 2,954 |
| world_model_structured | 0.107 | 0.105 | 0.107 | 0.109 | 0.080 | 2,954 |
| lightgbm | 0.082 | 0.088 | 0.088 | 0.092 | 0.081 | 3,008 |
| world_model_heads_on_real_embeddings | 0.050 | 0.051 | 0.054 | 0.055 | – | – |

**Effect recovery** (bread units in week 0, action [d,0,0,0], vs ground-truth counterfactuals)

| level | true | world_model pred | world_model_hybrid pred | world_model_structured pred | lightgbm pred | world_model bias | world_model_hybrid bias | world_model_structured bias | lightgbm bias | world_model RMSE | world_model_hybrid RMSE | world_model_structured RMSE | lightgbm RMSE | diff in means | naive obs. | true Δ margin 4w | world_model Δ margin 4w | world_model_hybrid Δ margin 4w | world_model_structured Δ margin 4w | lightgbm Δ margin 4w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 288 | 295 | 295 | 295 | 293 | 7.223 | 7.223 | 7.223 | 4.746 | 106 | 106 | 106 | 104 | 115 | – | -96.661 | -247 | 12.692 | -39.053 | -65.478 |
| 20% | 614 | 603 | 603 | 603 | 679 | -11.164 | -11.164 | -11.164 | 65.025 | 223 | 223 | 223 | 230 | 309 | – | -294 | -497 | -293 | -243 | -70.106 |
| 30% | 979 | 875 | 875 | 875 | 979 | -104 | -104 | -104 | 0.161 | 380 | 380 | 380 | 350 | 774 | – | -610 | -833 | -1,000 | -857 | -131 |

**Margin effect split** (Δ 4-week margin vs 0 %, € per store-week: bread vs all other categories)

| level | model | bread pred | bread true | other pred | other true | total pred | total true |
|---|---|---|---|---|---|---|---|
| 10% | world_model | -202 | -240 | -44.581 | 144 | -247 | -96.661 |
| 10% | world_model_hybrid | -202 | -240 | 215 | 144 | 12.692 | -96.661 |
| 10% | world_model_structured | -254 | -240 | 215 | 144 | -39.053 | -96.661 |
| 20% | world_model | -662 | -589 | 165 | 296 | -497 | -294 |
| 20% | world_model_hybrid | -662 | -589 | 368 | 296 | -293 | -294 |
| 20% | world_model_structured | -612 | -589 | 368 | 296 | -243 | -294 |
| 30% | world_model | -1,220 | -1,067 | 387 | 457 | -833 | -610 |
| 30% | world_model_hybrid | -1,220 | -1,067 | 220 | 457 | -1,000 | -610 |
| 30% | world_model_structured | -1,077 | -1,067 | 220 | 457 | -857 | -610 |

**Heterogeneity and action sensitivity**

| model | Spearman vs true per-store uplift | Spearman vs bread_beta | sensitivity ratio | ignores action |
|---|---|---|---|---|
| world_model | 0.804 | 0.634 | 0.894 | no |
| world_model_hybrid | 0.804 | 0.634 | 0.894 | no |
| world_model_structured | 0.804 | 0.634 | 0.894 | no |
| lightgbm | 0.819 | 0.610 | 1.000 | no |
(true per-store uplift vs bread_beta: Spearman 0.900)

**Planning** (true 4-week total margin of the chosen level, € per store-week)

| policy | value € | regret € | oracle match | levels chosen |
|---|---|---|---|---|
| oracle | 37,239 | 0.000 | 1.000 |  |
| never discount | 37,215 | 24.664 | 0.749 |  |
| always 10% | 37,118 | 121 | 0.190 |  |
| world_model | 37,198 | 41.801 | 0.704 | 0% 93%, 10% 3%, 20% 3%, 30% 1% |
| world_model_hybrid | 37,187 | 51.907 | 0.593 | 0% 73%, 10% 27%, 20% 0%, 30% 0% |
| world_model_structured | 37,206 | 33.531 | 0.664 | 0% 82%, 10% 18%, 20% 0%, 30% 0% |
| lightgbm | 37,124 | 115 | 0.509 | 0% 64%, 10% 22%, 20% 3%, 30% 12% |
true best level shares: 0% 75%, 10% 19%, 20% 6%, 30% 0%

**Censoring** (WAPE bread h1)

| model | censored weeks | uncensored weeks | share censored |
|---|---|---|---|
| world_model | 0.106 | 0.108 | 0.186 |
| world_model_hybrid | 0.106 | 0.108 | 0.186 |
| world_model_structured | 0.106 | 0.108 | 0.186 |
| lightgbm | 0.080 | 0.083 | 0.186 |

### test / representation

Ridge probes on frozen z(t), out-of-fold R², folds grouped by store

| target | z(t) | calendar + store one-hot | calendar + static features |
|---|---|---|---|
| bread_beta | 0.585 | -0.005 | 0.542 |
| size_sqm | 0.965 | -0.005 | 1.000 |
| holiday | – | – | – |
| traffic_shock | 0.657 | -0.003 | -0.021 |

kNN (k=10) next-week bread units, WAPE: neighbours in z-space 0.171, neighbours on static features 0.187

## Plots

**Training curves**

![Training curves](plots/training_curves.png)

**Uplift vs truth (test, randomized slot weeks)**

![Uplift vs truth (test, randomized slot weeks)](plots/uplift_vs_truth.png)

**Per-store 10% uplift, predicted vs true (test)**

![Per-store 10% uplift, predicted vs true (test)](plots/store_uplift_pred_vs_true.png)

**Planning regret (test, all)**

![Planning regret (test, all)](plots/planning_regret.png)

**Embedding PCA coloured by bread_beta (test)**

![Embedding PCA coloured by bread_beta (test)](plots/embedding_pca.png)
