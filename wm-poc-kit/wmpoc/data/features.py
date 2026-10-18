"""Feature construction. Every feature name in the config maps to a builder below."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from wmpoc.data.loading import RawData

OPEN_HOURS = 78.0
HOLIDAYS = ["christmas", "easter", "new_year"]
N_REGIONS = 6
ACTION_SCALE = 10.0  # actions are discount fractions × 10, so 0..3 for 0..30 %
HEAD_TARGETS = ["log_units_sold", "waste_rate", "log_stock_units"]


@dataclass
class Panel:
    """Dense store × week × category arrays (raw, not normalised)."""

    categories: list[str]
    store_ids: np.ndarray  # [S] original store ids
    n_weeks: int
    tokens: np.ndarray  # [S,W,C,F]
    token_names: list[str]
    context: np.ndarray  # [S,W,Fc]
    context_names: list[str]
    known_future: np.ndarray  # [S,W,Fk]
    known_future_names: list[str]
    static: np.ndarray  # [S,Fs]
    static_names: list[str]
    units: np.ndarray  # [S,W,C] sales (censored)
    stock: np.ndarray
    waste: np.ndarray
    stockout_hours: np.ndarray
    discount: np.ndarray
    base_price: np.ndarray
    net_price: np.ndarray
    unit_cost: np.ndarray
    margin: np.ndarray
    randomized: np.ndarray  # [S,W] bread week is in the experiment
    slot: np.ndarray  # [S,W] bread discount drawn at random this week
    cohort: np.ndarray  # [S]
    holiday: np.ndarray  # [S,W] any national holiday
    week_of_year: np.ndarray  # [W]
    action_categories: list[str]
    head_targets: dict[str, np.ndarray] = field(default_factory=dict)  # name -> [S,W,C]

    @property
    def n_stores(self) -> int:
        return len(self.store_ids)

    @property
    def action_idx(self) -> list[int]:
        return [self.categories.index(c) for c in self.action_categories]

    @property
    def actions(self) -> np.ndarray:  # [S,W,A], scaled
        return self.discount[:, :, self.action_idx] * ACTION_SCALE


def _cube(sales: pd.DataFrame, col: str, stores: np.ndarray, W: int, cats: list[str]) -> np.ndarray:
    s_idx = pd.Index(stores).get_indexer(sales["store_id"])
    c_idx = pd.Index(cats).get_indexer(sales["category"])
    out = np.full((len(stores), W, len(cats)), np.nan, dtype=np.float64)
    out[s_idx, sales["week"].to_numpy(), c_idx] = sales[col].to_numpy(dtype=np.float64)
    if np.isnan(out).any():
        raise ValueError(f"panel has missing cells for {col}")
    return out


TOKEN_FEATURES: dict[str, Callable[[dict], np.ndarray]] = {
    "log_units_sold": lambda a: np.log1p(a["units"]),
    "log_net_price": lambda a: np.log1p(a["net_price"]),
    "discount": lambda a: a["discount"],
    "stockout_frac": lambda a: a["stockout_hours"] / OPEN_HOURS,
    "waste_rate": lambda a: a["waste"] / np.maximum(a["stock"], 1.0),
    "log_stock_units": lambda a: np.log1p(a["stock"]),
}


def _calendar_features(cal: dict[str, np.ndarray], names: Sequence[str], extra: dict) -> tuple[list[np.ndarray], list[str]]:
    cols, out_names = [], []
    for n in names:
        if n == "week_of_year_sin":
            cols.append(np.sin(2 * np.pi * cal["week_of_year"] / 52)); out_names.append(n)
        elif n == "week_of_year_cos":
            cols.append(np.cos(2 * np.pi * cal["week_of_year"] / 52)); out_names.append(n)
        elif n == "national_holiday_onehot":
            for h in HOLIDAYS:
                cols.append((cal["national_holiday"] == h).astype(float)); out_names.append(f"holiday_{h}")
        elif n in ("regional_holiday", "temp_c", "rain_mm", "temp_forecast_c", "rain_forecast_mm",
                   "national_bread_flyer"):
            cols.append(cal[n].astype(float)); out_names.append(n)
        elif n == "planned_discount_non_action_categories":
            for c, arr in extra["non_action_discounts"].items():
                cols.append(arr); out_names.append(f"planned_discount_{c}")
        else:
            raise KeyError(f"unknown calendar / known-future feature {n!r}")
    return cols, out_names


def _static_features(st: pd.DataFrame, names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    cols, out_names = [], []
    for n in names:
        if n == "region_onehot":
            for r in range(N_REGIONS):
                cols.append((st["region"].to_numpy() == r).astype(float)); out_names.append(f"region_{r}")
        elif n in st.columns:
            cols.append(st[n].to_numpy(dtype=float)); out_names.append(n)
        else:
            raise KeyError(f"unknown static feature {n!r}")
    return np.stack(cols, -1), out_names


def build_panel(raw: RawData, cfg) -> Panel:
    dcfg = cfg.data
    cats = list(dcfg.categories)
    stores = np.sort(raw.stores["store_id"].to_numpy())
    st = raw.stores.set_index("store_id").loc[stores].reset_index()
    W = int(raw.sales["week"].max()) + 1
    sales = raw.sales[raw.sales["category"].isin(cats)]

    a = {k: _cube(sales, col, stores, W, cats) for k, col in [
        ("units", "units_sold"), ("stock", "stock_units"), ("waste", "waste_units"),
        ("stockout_hours", "stockout_hours"), ("discount", "discount"), ("base_price", "base_price"),
        ("net_price", "net_price"), ("unit_cost", "unit_cost"), ("margin", "margin_eur"),
        ("randomized", "randomized"), ("slot", "experiment_slot")]}

    token_names = list(dcfg.token_features)
    for n in token_names:
        if n not in TOKEN_FEATURES:
            raise KeyError(f"unknown token feature {n!r}; available {sorted(TOKEN_FEATURES)}")
    tokens = np.stack([TOKEN_FEATURES[n](a) for n in token_names], -1)
    head_targets = {n: TOKEN_FEATURES[n](a) for n in HEAD_TARGETS}

    # calendar is per region × week; broadcast to store × week
    cal = raw.calendar.sort_values(["region", "week"])
    region = st["region"].to_numpy()
    cal_arr: dict[str, np.ndarray] = {}
    for col in cal.columns:
        if col in ("region", "week"):
            continue
        grid = cal[col].to_numpy().reshape(N_REGIONS, -1)[:, :W]
        cal_arr[col] = grid[region]  # [S,W]
    action_cats = list(dcfg.action_categories)
    non_action = {c: a["discount"][:, :, cats.index(c)] for c in cats if c not in action_cats}
    ctx_cols, ctx_names = _calendar_features(cal_arr, dcfg.context_features, {"non_action_discounts": non_action})
    kf_cols, kf_names = _calendar_features(cal_arr, dcfg.known_future_features, {"non_action_discounts": non_action})
    static, static_names = _static_features(st, dcfg.static_store_features)

    tgt = cats.index(dcfg.target_category)
    return Panel(
        categories=cats, store_ids=stores, n_weeks=W,
        tokens=tokens.astype(np.float32), token_names=token_names,
        context=np.stack(ctx_cols, -1).astype(np.float32), context_names=ctx_names,
        known_future=np.stack(kf_cols, -1).astype(np.float32), known_future_names=kf_names,
        static=static.astype(np.float32), static_names=static_names,
        units=a["units"], stock=a["stock"], waste=a["waste"], stockout_hours=a["stockout_hours"],
        discount=a["discount"], base_price=a["base_price"], net_price=a["net_price"],
        unit_cost=a["unit_cost"], margin=a["margin"],
        randomized=a["randomized"][:, :, tgt].astype(bool), slot=a["slot"][:, :, tgt].astype(bool),
        cohort=st["experiment_cohort"].to_numpy().astype(bool),
        holiday=np.isin(cal_arr["national_holiday"], HOLIDAYS),
        week_of_year=cal_arr["week_of_year"][0].astype(int),
        action_categories=action_cats, head_targets=head_targets,
    )


@dataclass
class Normalizer:
    """z-scores fit on train weeks × train stores only."""

    tok_mean: np.ndarray  # [C,F]
    tok_std: np.ndarray
    ctx_mean: np.ndarray
    ctx_std: np.ndarray
    kf_mean: np.ndarray
    kf_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    target_mean: dict[str, np.ndarray]  # name -> [C]
    target_std: dict[str, np.ndarray]

    @staticmethod
    def fit(panel: Panel, train_stores: np.ndarray, train_weeks: tuple[int, int]) -> "Normalizer":
        sl = (train_stores, slice(*train_weeks))

        def ms(x: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
            return x.mean(axes), np.maximum(x.std(axes), 1e-6)

        tm, ts = ms(panel.tokens[sl], (0, 1))
        cm, cs = ms(panel.context[sl], (0, 1))
        km, ks = ms(panel.known_future[sl], (0, 1))
        sm, ss = ms(panel.static[train_stores], (0,))
        tgm, tgs = {}, {}
        for n, arr in panel.head_targets.items():
            tgm[n], tgs[n] = ms(arr[sl], (0, 1))
        return Normalizer(tm, ts, cm, cs, km, ks, sm, ss, tgm, tgs)

    def tokens(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.tok_mean) / self.tok_std).astype(np.float32)

    def context(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.ctx_mean) / self.ctx_std).astype(np.float32)

    def known_future(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.kf_mean) / self.kf_std).astype(np.float32)

    def static(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.static_mean) / self.static_std).astype(np.float32)

    def target(self, name: str, x: np.ndarray) -> np.ndarray:
        return ((x - self.target_mean[name]) / self.target_std[name]).astype(np.float32)

    def state_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @staticmethod
    def from_state_dict(d: dict) -> "Normalizer":
        return Normalizer(**d)
