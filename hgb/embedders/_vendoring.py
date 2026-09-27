"""
Isolated import of vendored upstream code.

Several upstream repos use bare, generic module names (``import utils``,
``from layers import ...``, ``import model``).  If we simply added them all to
sys.path at once, one repo's ``utils`` would shadow another's.  ``vendor_path``
prepends a single vendor directory to sys.path, and afterwards evicts any
module cached under a supplied set of generic names, so each adapter imports
its OWN upstream files and leaves no residue that could poison the next.

The upstream .py files are never edited -- this only manipulates the import
machinery around them, so the code that runs is byte-for-byte theirs.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys

from .. import config as C

# generic names upstream repos import bare; evicted after each isolated import
_GENERIC = ("utils", "layers", "model", "models", "tricl_encoder",
            "contrast_loss", "config", "aug")


@contextlib.contextmanager
def vendor_path(subdir: str, evict=_GENERIC):
    path = os.path.join(C.VENDOR_ROOT, subdir)
    saved = {n: sys.modules.get(n) for n in evict}
    sys.path.insert(0, path)
    try:
        yield path
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass
        # restore/evict generic names so the next adapter imports fresh
        for n in evict:
            if saved[n] is not None:
                sys.modules[n] = saved[n]
            else:
                sys.modules.pop(n, None)


def require_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This embedder/classifier needs PyTorch (and, for the deep "
            "methods, torch_geometric + torch_scatter [+ torch_sparse for the "
            "AllSet classifiers]). Install them in the environment described in "
            "requirements.txt, then re-run -- the harness will resume from the "
            "last completed cell."
        ) from e
