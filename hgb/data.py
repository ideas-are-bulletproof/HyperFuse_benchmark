"""
hgb/data.py
===========

Unified loader that turns the on-disk ``pickle_dir`` format shared by
TriCL / SE-HSSL / CHGNN --

    features.pickle    scipy.sparse matrix  [N, F]
    hypergraph.pickle  dict {edge_key: [node_id, ...]}
    labels.pickle      1-D array/list of length N

-- into a single canonical ``HGSample`` that every embedder and classifier in
the harness consumes.  Keeping ONE loader means all methods see byte-for-byte
the same nodes, features, hyperedges and labels, which is what makes the
comparison fair.

The bipartite incidence ``hyperedge_index`` (row 0 = node ids, row 1 =
hyperedge ids in ``0..E-1``) is built exactly the way the upstream repos build
it (enumerate the hyperedges, emit one [node, edge] pair per membership).  We
enumerate edges in sorted-key order purely for run-to-run determinism; edge
ids are just labels, so this changes nothing about any method's math.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from . import config as C


@dataclass
class HGSample:
    name: str
    num_nodes: int
    num_features: int
    num_classes: int
    x: sp.csr_matrix                 # [N, F] sparse features (may be all-zero if none)
    y: np.ndarray                    # [N] int64, contiguous 0..C-1
    hyperedges: list                 # list[list[int]] node-id lists (deduped, size>=1)
    hyperedge_index: np.ndarray      # [2, nnz] int64 bipartite incidence
    num_edges: int
    has_features: bool = True

    # convenience views built on demand
    def x_dense(self) -> np.ndarray:
        return np.asarray(self.x.todense(), dtype=np.float32)

    def to_fuse_dict(self) -> dict:
        """The dataset-dict shape expected by fuse_versions.fuse_hyper_r."""
        return {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "x": self.x_dense(),
            "y": self.y,
            "hyperedges": self.hyperedges,
        }


def _relabel_contiguous(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    # some raw label arrays are one-hot / 2-D; collapse to class index if so
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    return np.asarray([remap[v] for v in y], dtype=np.int64)


def dataset_available(name: str) -> tuple[bool, str]:
    if name not in C.DATASETS:
        return False, f"unknown dataset {name!r}"
    kind, rel, registered_available = C.DATASETS[name]
    path = os.path.join(C.DATA_ROOT, rel)
    on_disk = os.path.isdir(path) and all(
        os.path.exists(os.path.join(path, f))
        for f in ("features.pickle", "hypergraph.pickle", "labels.pickle")
    )
    if not registered_available and not on_disk:
        return False, (
            f"{C.DATASET_DISPLAY.get(name, name)}: no faithful data source was "
            "supplied with the provided repos; not fabricated. Drop a "
            f"pickle_dir at {path} to enable it."
        )
    if not on_disk:
        return False, (
            f"{C.DATASET_DISPLAY.get(name, name)}: expected data at {path} "
            "(run setup_benchmark.py to stage it from the provided archives)."
        )
    return True, path


def load(name: str) -> HGSample:
    ok, info = dataset_available(name)
    if not ok:
        raise FileNotFoundError(info)
    path = info

    with open(os.path.join(path, "features.pickle"), "rb") as f:
        X = pickle.load(f)
    with open(os.path.join(path, "hypergraph.pickle"), "rb") as f:
        hg = pickle.load(f)
    with open(os.path.join(path, "labels.pickle"), "rb") as f:
        y_raw = pickle.load(f)

    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X))
    X = X.tocsr().astype(np.float32)

    y = _relabel_contiguous(np.asarray(y_raw))
    N = X.shape[0]

    # Build deduped hyperedges (sorted key order for determinism) + incidence.
    rows, cols, hyperedges = [], [], []
    e = 0
    for key in sorted(hg.keys(), key=lambda k: (str(type(k)), k)):
        nodes = [int(v) for v in hg[key] if 0 <= int(v) < N]
        nodes = sorted(set(nodes))
        if len(nodes) == 0:
            continue
        hyperedges.append(nodes)
        for v in nodes:
            rows.append(v)
            cols.append(e)
        e += 1

    hyperedge_index = np.asarray([rows, cols], dtype=np.int64)
    num_edges = e

    has_features = bool(X.nnz > 0) and X.shape[1] > 0
    return HGSample(
        name=name,
        num_nodes=N,
        num_features=int(X.shape[1]),
        num_classes=int(y.max()) + 1,
        x=X,
        y=y,
        hyperedges=hyperedges,
        hyperedge_index=hyperedge_index,
        num_edges=num_edges,
        has_features=has_features,
    )


def make_label_splits(y: np.ndarray, seed: int, n_splits: int,
                      train_prop: float, valid_prop: float) -> list[dict]:
    """Deterministic per-seed family of random train/val/test index splits,
    stratification-free to mirror AllSet's ``rand_train_test_idx``.  Shared by
    every classifier so only the classifier differs between rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    splits = []
    for s in range(n_splits):
        perm = rng.permutation(n)
        n_tr = int(n * train_prop)
        n_va = int(n * valid_prop)
        tr = np.sort(perm[:n_tr])
        va = np.sort(perm[n_tr:n_tr + n_va])
        te = np.sort(perm[n_tr + n_va:])
        splits.append({"train": tr, "valid": va, "test": te})
    return splits


def _tricl_split(num_nodes: int, split_id: int,
                 train_ratio: float = 0.1, val_ratio: float = 0.1) -> dict:
    """Verbatim logic of TriCL's dataset/preprocess.py::generate_random_split
    (fresh default_rng(split_id) per split), returned as index arrays."""
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)
    perm = np.random.default_rng(split_id).permutation(num_nodes)
    return {
        "train": np.sort(perm[:num_train]),
        "valid": np.sort(perm[num_train:num_train + num_val]),
        "test": np.sort(perm[num_train + num_val:]),
    }


def make_tricl_splits(name: str, num_nodes: int, n_splits: int = 20,
                      train_ratio: float = 0.1, val_ratio: float = 0.1) -> list[dict]:
    """TriCL node-classification splits (10/10/80, split ids 0..n-1).

    Uses TriCL's stored ``splits/{i}.pickle`` next to the dataset when present
    (cross-checked against the regenerated split); otherwise regenerates the
    split with TriCL's exact code, which reproduces those files bit-for-bit.
    Independent of the benchmark seed, as in TriCL.
    """
    ok, path = dataset_available(name)
    split_dir = os.path.join(path, "splits") if ok else None
    splits = []
    for i in range(n_splits):
        gen = _tricl_split(num_nodes, i, train_ratio, val_ratio)
        f = os.path.join(split_dir, f"{i}.pickle") if split_dir else None
        if f and os.path.exists(f):
            with open(f, "rb") as fh:
                m = pickle.load(fh)
            if len(np.asarray(m["train_mask"])) != num_nodes:
                raise ValueError(f"{f}: mask length != num_nodes ({num_nodes})")
            stored = {
                "train": np.flatnonzero(np.asarray(m["train_mask"])),
                "valid": np.flatnonzero(np.asarray(m["val_mask"])),
                "test": np.flatnonzero(np.asarray(m["test_mask"])),
            }
            if not all(np.array_equal(stored[k], gen[k]) for k in stored):
                raise ValueError(f"{f}: stored TriCL split disagrees with regenerated split")
            splits.append(stored)
        else:
            splits.append(gen)
    return splits


def get_label_splits(sample: "HGSample") -> list[dict]:
    """The split family every classifier is evaluated on (config: TriCL)."""
    return make_tricl_splits(sample.name, sample.num_nodes, C.N_CLF_SPLITS,
                             C.TRAIN_PROP, C.VALID_PROP)
