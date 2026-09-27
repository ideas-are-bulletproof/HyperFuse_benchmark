"""
fuse_core/base.py  ->  imported by every FUSE version as ``B``
==============================================================

This is the small "standard library" every FUSE version leans on. A version
file does::

    from fuse_core import base as B

and then uses ``B.scipy_adj(d)``, ``B.standardize(X)``, ``B.CONFIG``,
``B.logger``, ``B.LOG_EVERY``, ``B.DEVICE``, ``B.set_seed(seed)``,
``B.scalable_struct_features(d, A)``, ``B.rownorm_P(G)``.

A "dataset dict" ``d`` (the thing FUSE versions receive) is::

    {
      "name": str,
      "num_nodes": int,
      "num_features": int,
      "num_classes": int,
      "edge_index": LongTensor [2, E]   (undirected, canonical order),
      "x": FloatTensor [N, F]  or None,
      "y": LongTensor [N],
    }

``make_dataset_dict(canonical)`` builds one from a ``load_canonical`` result,
so the FUSE stack and the baseline runners consume the exact same graph.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import scipy.sparse as sp

import config as _cfg


# --------------------------------------------------------------------------- #
# Globals the FUSE versions read
# --------------------------------------------------------------------------- #
CONFIG = _cfg.CONFIG
LOG_EVERY = _cfg.LOG_EVERY
DEVICE = _cfg.DEVICE

logger = logging.getLogger("fuse")
if not logger.handlers:
    logger.setLevel(logging.INFO)


def set_logger(lg: logging.Logger) -> None:
    """Let the harness swap in its own configured logger (file + console)."""
    global logger
    logger = lg


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Dataset-dict adapter
# --------------------------------------------------------------------------- #
def make_dataset_dict(canonical) -> dict:
    """Turn a ``load_canonical`` SimpleNamespace into the dict FUSE versions eat."""
    return {
        "name": canonical.name,
        "num_nodes": int(canonical.num_nodes),
        "num_features": int(canonical.num_features),
        "num_classes": int(canonical.num_classes),
        "edge_index": canonical.edge_index,
        "x": canonical.x,
        "y": canonical.y,
    }


# --------------------------------------------------------------------------- #
# Numpy / scipy helpers
# --------------------------------------------------------------------------- #
def standardize(X) -> np.ndarray:
    """Zero-mean, unit-variance per column -> float32. Constant columns pass
    through unscaled (sd forced to 1)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype(np.float32)


def l2_normalize_rows(X) -> np.ndarray:
    """Row-normalize to unit L2 norm -> float32. Zero rows pass through as zero.

    Used (instead of column z-score standardization) wherever a feature Gram
    x_i . x_j feeds a modularity/similarity weight. Column standardization
    inflates rare/sparse columns (e.g. rare words in a bag-of-words vector) to
    large magnitudes; a single shared rare feature between two nodes can then
    dominate the Gram product with an outlier value. Row L2-normalization
    instead bounds every dot product to the [-1, 1] cosine-similarity range
    regardless of feature sparsity or rarity, which keeps the resulting
    modularity operator's entries on a comparable, bounded scale.
    """
    X = np.asarray(X, dtype=np.float64)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (X / norm).astype(np.float32)


def scipy_adj(d) -> sp.csr_matrix:
    """Symmetric, binary, zero-diagonal scipy CSR adjacency from ``d['edge_index']``."""
    ei = d["edge_index"]
    try:
        ei = ei.detach().cpu().numpy()
    except AttributeError:
        ei = np.asarray(ei)
    n = int(d["num_nodes"])
    A = sp.csr_matrix((np.ones(ei.shape[1], dtype=np.float64), (ei[0], ei[1])),
                      shape=(n, n))
    A = A.maximum(A.T)               # symmetrize
    A.setdiag(0.0)
    A.eliminate_zeros()
    A.data[:] = 1.0                  # binary
    return A.tocsr()


def rownorm_P(G) -> sp.csr_matrix:
    """Row-normalize a (sparse) matrix to a right-stochastic operator D^{-1} G."""
    G = sp.csr_matrix(G)
    rs = np.asarray(G.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    D_inv = sp.diags(1.0 / rs)
    return (D_inv @ G).tocsr()


def scalable_struct_features(d, A=None) -> np.ndarray:
    """Local Degree Profile (LDP): cheap, feature-free structural node features.

    For every node: its degree, log-degree, and the min / max / mean / std of
    its neighbours' degrees.  One sparse pass -> O(E), works on CPU for every
    dataset in the suite. Used as the GCN encoder input for the deep FUSE
    versions (so structure, not the raw features, drives message passing).
    """
    if A is None:
        A = scipy_adj(d)
    A = sp.csr_matrix(A)
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)

    Aco = A.tocoo()
    nbr_deg = deg[Aco.col]                              # neighbour degree per edge
    row = Aco.row

    def _seg(reduce_fn, fill):
        out = np.full(n, fill, dtype=np.float64)
        # group edges by row; rows are not guaranteed sorted, so use np.add.at style
        order = np.argsort(row, kind="stable")
        r = row[order]
        v = nbr_deg[order]
        if len(r) == 0:
            return out
        # boundaries between distinct rows
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(r)) + 1, [len(r)]))
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            out[r[b0]] = reduce_fn(v[b0:b1])
        return out

    mean_nd = np.divide(A @ deg, np.maximum(deg, 1.0))
    min_nd = _seg(np.min, 0.0)
    max_nd = _seg(np.max, 0.0)
    std_nd = _seg(np.std, 0.0)

    feats = np.stack(
        [deg, np.log1p(deg), mean_nd, min_nd, max_nd, std_nd], axis=1
    ).astype(np.float32)
    return feats
