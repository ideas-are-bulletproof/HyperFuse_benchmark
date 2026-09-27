"""
hgb/embedders
=============

Each embedder module exposes:

    DEFAULTS : dict          # the method's official default hyperparameters
                             # (verbatim from the upstream repo, for the record)
    def embed(sample, seed, device, logger) -> np.ndarray  # [N, d] float32

``get_embedder`` imports the adapter lazily so the pure-numpy method (and the
whole harness) still works in an environment without torch.
"""

from __future__ import annotations

import importlib

from .. import config as C


def get_embedder(name: str):
    if name not in C.EMBEDDERS:
        raise KeyError(f"unknown embedder {name!r}; known: {list(C.EMBEDDERS)}")
    spec = C.EMBEDDERS[name]
    mod = importlib.import_module(spec["module"])
    return mod, spec
