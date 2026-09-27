"""
HyperFuse embedder adapter (our method).

Runs the vendored ``vendor/fuse/fuse_versions/HyperFuse.py`` (a single,
self-contained file: numpy/scipy for stages A/B, torch for the hypergraph-CCA
self-supervised stage) on the canonical sample.  We only put ``vendor/fuse``
on sys.path so ``fuse_versions.HyperFuse`` can be imported.

The method's hyperparameters live in that file's ``DEFAULTS``; the only thing
the harness sets is the output width (``EMB_DIM``, 128 by default), from which
the internal block sizes are derived (S/F/G = EMB_DIM/4, /2, /4).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from .. import config as C

_FUSE_DIR = os.path.join(C.VENDOR_ROOT, "fuse")


def _ensure_on_path():
    if _FUSE_DIR not in sys.path:
        sys.path.insert(0, _FUSE_DIR)


def _method():
    _ensure_on_path()
    from fuse_versions import HyperFuse as HF
    return HF


# recorded for transparency (the values actually used are the vendored ones)
try:
    DEFAULTS = dict(_method().DEFAULTS, emb_dim=C.EMB_DIM)
except Exception:          # torch-free import of the harness must never fail here
    DEFAULTS = dict(emb_dim=C.EMB_DIM)


def embed(sample, seed: int, device: str, logger) -> np.ndarray:
    HF = _method()
    logger.info(f"    [hyperfuse] N={sample.num_nodes} F={sample.num_features} "
                f"E={sample.num_edges} classes={sample.num_classes} emb_dim={C.EMB_DIM}")
    emb = HF.embed(sample.hyperedges, sample.x, num_nodes=sample.num_nodes,
                   seed=seed, device=device, logger=logger, emb_dim=C.EMB_DIM)
    return np.asarray(emb, dtype=np.float32)
