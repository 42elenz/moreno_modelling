"""Windows and batches. `TokenBatch` is the single input type every model component reads."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from wmpoc.data.features import HEAD_TARGETS, Normalizer, Panel
from wmpoc.data.loading import load_raw
from wmpoc.data.features import build_panel
from wmpoc.data.splits import Splits, make_splits


@dataclass
class Window:
    """What an encoder reads: T weeks of tokens for one store."""

    tokens: Tensor  # [B,T,C,F] normalised
    context: Tensor  # [B,T,Fc]
    static: Tensor  # [B,Fs]
    store_id: Tensor  # [B] panel index
    store_known: Tensor  # [B] bool, False for cold-start stores (ID vector is zeroed)


@dataclass
class TokenBatch:
    tokens: Tensor  # [B,T,C,F] context weeks
    context: Tensor  # [B,T,Fc]
    known_future: Tensor  # [B,H,Fk]
    actions: Tensor  # [B,H,A]
    static: Tensor  # [B,Fs]
    store_id: Tensor  # [B]
    randomized: Tensor  # [B,H] bool
    future_tokens: Tensor  # [B,H,C,F] real future weeks (targets only, never WM input)
    future_context: Tensor  # [B,H,Fc]
    store_known: Tensor  # [B] bool
    slot: Tensor  # [B,H] bool
    targets: Tensor  # [B,H,C,K] normalised head targets, K = len(HEAD_TARGETS)
    censored: Tensor  # [B,H,C] bool, stockout_hours > 0
    weights: Tensor  # [B,H] sample weights
    base_price: Tensor  # [B,H,C] raw EUR, for margin only
    net_price: Tensor
    unit_cost: Tensor
    units: Tensor  # [B,H,C] observed sales
    margin: Tensor  # [B,H,C] observed margin
    origin: Tensor  # [B] first horizon week

    @property
    def B(self) -> int:
        return self.tokens.shape[0]

    def ctx_window(self) -> Window:
        return Window(self.tokens, self.context, self.static, self.store_id, self.store_known)

    def target_window(self) -> Window:
        """The T-week window that ends at the last horizon week."""
        T = self.tokens.shape[1]
        tok = torch.cat([self.tokens, self.future_tokens], 1)[:, -T:]
        ctx = torch.cat([self.context, self.future_context], 1)[:, -T:]
        return Window(tok, ctx, self.static, self.store_id, self.store_known)

    def target(self, name: str) -> Tensor:
        return self.targets[..., HEAD_TARGETS.index(name)]

    def to(self, device: torch.device | str) -> "TokenBatch":
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def with_actions(self, actions: Tensor) -> "TokenBatch":
        return replace(self, actions=actions)


class WindowDataset:
    """Holds the normalised panel as tensors and slices windows by (store, origin)."""

    def __init__(self, panel: Panel, norm: Normalizer, splits: Splits, randomized_weight: float) -> None:
        self.panel, self.norm, self.splits = panel, norm, splits
        self.T, self.H = splits.context_weeks, splits.horizon_weeks
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x))
        self.tok = t(norm.tokens(panel.tokens))
        self.ctx = t(norm.context(panel.context))
        self.kf = t(norm.known_future(panel.known_future))
        self.act = t(panel.actions.astype(np.float32))
        self.static = t(norm.static(panel.static))
        self.tgt = t(np.stack([norm.target(n, panel.head_targets[n]) for n in HEAD_TARGETS], -1))
        self.censored = t(panel.stockout_hours > 0)
        self.randomized = t(panel.randomized)
        self.slot = t(panel.slot)
        self.weight = t(np.where(panel.randomized, float(randomized_weight), 1.0).astype(np.float32))
        f32 = lambda x: t(x.astype(np.float32))
        self.base_price, self.net_price, self.unit_cost = f32(panel.base_price), f32(panel.net_price), f32(panel.unit_cost)
        self.units, self.margin = f32(panel.units), f32(panel.margin)
        known = np.zeros(panel.n_stores, dtype=bool)
        known[splits.train_stores] = True
        self.known = t(known)

    def batch(self, store_idx: np.ndarray, origin: np.ndarray) -> TokenBatch:
        s = torch.as_tensor(store_idx, dtype=torch.long)
        o = torch.as_tensor(origin, dtype=torch.long)
        T, H = self.T, self.H
        wk_c = o[:, None] + torch.arange(-T, 0)[None, :]
        wk_f = o[:, None] + torch.arange(0, H)[None, :]
        sc, sf = s[:, None], s[:, None]
        return TokenBatch(
            tokens=self.tok[sc, wk_c], context=self.ctx[sc, wk_c], known_future=self.kf[sf, wk_f],
            actions=self.act[sf, wk_f], static=self.static[s], store_id=s,
            randomized=self.randomized[sf, wk_f], future_tokens=self.tok[sf, wk_f],
            future_context=self.ctx[sf, wk_f], store_known=self.known[s], slot=self.slot[sf, wk_f],
            targets=self.tgt[sf, wk_f], censored=self.censored[sf, wk_f], weights=self.weight[sf, wk_f],
            base_price=self.base_price[sf, wk_f], net_price=self.net_price[sf, wk_f],
            unit_cost=self.unit_cost[sf, wk_f], units=self.units[sf, wk_f], margin=self.margin[sf, wk_f],
            origin=o,
        )

    def iterate(self, store_idx: np.ndarray, origin: np.ndarray, batch_size: int, shuffle: bool = False,
                rng: np.random.Generator | None = None) -> Iterator[TokenBatch]:
        n = len(store_idx)
        order = (rng or np.random.default_rng(0)).permutation(n) if shuffle else np.arange(n)
        for i in range(0, n, batch_size):
            j = order[i:i + batch_size]
            yield self.batch(store_idx[j], origin[j])


@dataclass
class Dims:
    categories: list[str]
    token_names: list[str]
    price_idx: list[int]
    action_idx: list[int]
    target_idx: int
    n_context: int
    n_known_future: int
    n_static: int
    n_actions: int
    n_stores: int
    T: int
    H: int

    @property
    def C(self) -> int:
        return len(self.categories)

    @property
    def F(self) -> int:
        return len(self.token_names)

    def feature_idx(self, names) -> list[int]:
        return [self.token_names.index(n) for n in names]


@dataclass
class DataBundle:
    panel: Panel
    norm: Normalizer
    splits: Splits
    ds: WindowDataset
    dims: Dims


def build_data(cfg) -> DataBundle:
    raw = load_raw(cfg.data.dir, cfg.data.max_stores)
    panel = build_panel(raw, cfg)
    splits = make_splits(panel.n_stores, cfg)
    norm = Normalizer.fit(panel, splits.train_stores, splits.ranges["train"])
    ds = WindowDataset(panel, norm, splits, float(cfg.data.randomized_sample_weight))
    price = [panel.token_names.index(n) for n in cfg.data.price_features if n in panel.token_names]
    dims = Dims(
        categories=panel.categories, token_names=panel.token_names, price_idx=price,
        action_idx=panel.action_idx, target_idx=panel.categories.index(cfg.data.target_category),
        n_context=panel.context.shape[-1], n_known_future=panel.known_future.shape[-1],
        n_static=panel.static.shape[-1], n_actions=len(panel.action_idx), n_stores=panel.n_stores,
        T=splits.context_weeks, H=splits.horizon_weeks,
    )
    return DataBundle(panel, norm, splits, ds, dims)
