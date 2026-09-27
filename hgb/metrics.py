"""
hgb/metrics.py
==============

The five performance measures requested:

  * accuracy, macro_f1  -- from a downstream classifier's test predictions
  * ari, nmi            -- KMeans clustering quality of the embedding itself
  * embed_time_s        -- wall-clock time to GENERATE the embedding

Clustering is done directly on the raw embedding (k = number of classes),
averaged over several KMeans seeds, so ARI/NMI measure the geometry of the
representation, independent of any classifier.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score,
    normalized_mutual_info_score,
)

from . import config as C


@contextmanager
def timer():
    """with timer() as t: ... ; elapsed seconds available as t()."""
    start = time.perf_counter()
    box = {}
    yield lambda: box.get("elapsed", time.perf_counter() - start)
    box["elapsed"] = time.perf_counter() - start


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def clustering_metrics(emb: np.ndarray, y_true: np.ndarray,
                       n_runs: int = C.N_CLUSTER_RUNS,
                       base_seed: int = 0) -> dict:
    """KMeans (k = #classes) on the embedding; mean ARI / NMI over n_runs."""
    emb = np.asarray(emb, dtype=np.float64)
    k = int(y_true.max()) + 1
    aris, nmis = [], []
    for r in range(n_runs):
        km = KMeans(n_clusters=k, n_init=10, random_state=base_seed + r)
        pred = km.fit_predict(emb)
        aris.append(adjusted_rand_score(y_true, pred))
        nmis.append(normalized_mutual_info_score(y_true, pred))
    return {"ari": float(np.mean(aris)), "nmi": float(np.mean(nmis))}


def aggregate(values: list[float]) -> tuple[float, float]:
    """mean, std (population) of a list of scalar metric values."""
    if len(values) == 0:
        return float("nan"), float("nan")
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std())
