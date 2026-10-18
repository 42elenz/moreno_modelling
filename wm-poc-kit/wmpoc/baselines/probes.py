"""Linear probes with grouped CV, plus the calendar + store floor and a kNN forecast test."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wmpoc.data.features import Panel

ALPHAS = np.logspace(-2, 4, 13)


def ridge_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> float:
    """Out-of-fold R² of a ridge probe; folds are grouped by store (no store in train and test)."""
    n_splits = min(n_splits, len(np.unique(groups)))
    pred = np.zeros_like(y, dtype=float)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def calendar_store_features(p: Panel, s: np.ndarray, week: np.ndarray) -> np.ndarray:
    """Calendar of the week plus one-hot store: the floor every probe on z(t) must beat."""
    cal = p.context[s, week]
    onehot = np.zeros((len(s), p.n_stores))
    onehot[np.arange(len(s)), s] = 1.0
    return np.concatenate([cal, onehot], 1)


def static_calendar_features(p: Panel, s: np.ndarray, week: np.ndarray) -> np.ndarray:
    """Calendar plus observable store features (a stronger floor for store-level traits)."""
    return np.concatenate([p.context[s, week], p.static[s]], 1)


def knn_forecast(train_X: np.ndarray, train_y: np.ndarray, query_X: np.ndarray, k: int = 10) -> np.ndarray:
    sc = StandardScaler().fit(train_X)
    nn = NearestNeighbors(n_neighbors=k).fit(sc.transform(train_X))
    _, idx = nn.kneighbors(sc.transform(query_X))
    return train_y[idx].mean(1)
