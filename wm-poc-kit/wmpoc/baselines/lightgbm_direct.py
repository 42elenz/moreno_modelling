"""LightGBM direct baseline: one model per horizon step for bread units, one for 4-week total margin.

Inputs: lagged sales features, static store features, known-future features and the planned
bread action sequence. Same windows, splits and sample weights as the neural models.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from wmpoc.data.features import Panel
from wmpoc.data.splits import Splits


class LightGBMDirect:
    def __init__(self, cfg) -> None:
        b = cfg.baselines.lightgbm
        self.lags = [int(x) for x in b.lags]
        self.params = dict(
            num_leaves=int(b.num_leaves), n_estimators=int(b.n_estimators), learning_rate=float(b.learning_rate),
            random_state=int(cfg.experiment.seed), n_jobs=int(b.num_threads), deterministic=True,
            force_row_wise=True, verbose=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        )
        self.H = int(cfg.data.window.horizon_weeks)
        self.rw = float(cfg.data.randomized_sample_weight)
        self.tgt = cfg.data.categories.index(cfg.data.target_category)
        self.unit_models: list[lgb.LGBMRegressor] = []
        self.margin_model: lgb.LGBMRegressor | None = None

    # ------------------------------------------------------------------ features
    def _base(self, p: Panel, s: np.ndarray, t: np.ndarray, actions: np.ndarray) -> np.ndarray:
        b = self.tgt
        lw = lambda l: np.clip(t - l, 0, None)
        logu = np.log1p(p.units)
        tot_margin = p.margin.sum(-1)
        cols = [p.static[s]]
        cols += [logu[s, lw(l), b][:, None] for l in self.lags]
        cols += [p.discount[s, lw(l), b][:, None] for l in (1, 2, 3, 4)]
        cols += [(p.stockout_hours[s, lw(1), b] / 78.0)[:, None], (p.waste[s, lw(1), b] / np.maximum(p.stock[s, lw(1), b], 1))[:, None]]
        cols += [logu[s, lw(1)], np.log1p(p.units[s, lw(1)].sum(-1))[:, None]]
        cols += [np.mean([logu[s, lw(l), b] for l in (1, 2, 3, 4)], 0)[:, None]]
        cols += [tot_margin[s, lw(l)][:, None] for l in (1, 2, 4)]
        cols += [np.log(p.base_price[s, t, b])[:, None], actions]
        return np.concatenate(cols, 1)

    def features(self, p: Panel, s: np.ndarray, t: np.ndarray, actions: np.ndarray) -> list[np.ndarray]:
        """One matrix per horizon step plus one for the 4-week margin model (last)."""
        base = self._base(p, s, t, actions)
        kf = [p.known_future[s, t + h] for h in range(self.H)]
        per_h = [np.concatenate([base, kf[h], actions[:, h:h + 1]], 1) for h in range(self.H)]
        return per_h + [np.concatenate([base] + kf, 1)]

    def _observed_actions(self, p: Panel, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.stack([p.discount[s, t + h, self.tgt] for h in range(self.H)], 1)

    # ------------------------------------------------------------------ fit / predict
    def fit(self, p: Panel, splits: Splits) -> "LightGBMDirect":
        s, t = splits.windows("train")
        X = self.features(p, s, t, self._observed_actions(p, s, t))
        self.unit_models = []
        for h in range(self.H):
            y = np.log1p(p.units[s, t + h, self.tgt])
            w = np.where(p.randomized[s, t + h], self.rw, 1.0)
            self.unit_models.append(lgb.LGBMRegressor(**self.params).fit(X[h], y, sample_weight=w))
        ym = sum(p.margin[s, t + h].sum(-1) for h in range(self.H))
        wm = np.where(p.randomized[s, t], self.rw, 1.0)
        self.margin_model = lgb.LGBMRegressor(**self.params).fit(X[-1], ym, sample_weight=wm)
        return self

    def predict(self, p: Panel, s: np.ndarray, t: np.ndarray, actions: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """actions: [N,H] bread discount fractions; None = as observed."""
        if actions is None:
            actions = self._observed_actions(p, s, t)
        X = self.features(p, s, t, actions)
        units = np.stack([np.expm1(m.predict(X[h])) for h, m in enumerate(self.unit_models)], 1)
        return {"bread_units": units, "total_margin_4w": self.margin_model.predict(X[-1])}
