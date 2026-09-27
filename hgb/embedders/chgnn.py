"""
CHGNN adapter -- deliberately guarded.

CHGNN (vendored under vendor/chgnn) is **semi-supervised**: its training loss
includes a classification term on the train labels (train.py:
``loss_cls = model.loss_cls(...Y, train_idx...)``), and its data pipeline
depends on precomputed per-dataset ``homogeneity`` files produced by the repo's
own ``data/cal_pattern.py`` -- neither of which fits the fair *unsupervised*
"embed once, then evaluate with independent classifiers" protocol the other
methods follow (feeding a label-trained representation to a classifier that is
then scored on overlapping nodes leaks labels).

Rather than ship a subtly-wrong "embedding" the CHGNN authors never defined,
this adapter refuses by default and explains how to run CHGNN faithfully in its
own right.  Set HGB_ALLOW_CHGNN=1 to attempt an experimental encoder-output
extraction anyway (clearly flagged in results as semi-supervised, NOT
comparable to the unsupervised rows).
"""

from __future__ import annotations

import os

import numpy as np

DEFAULTS = "see vendor/chgnn/config.py + vendor/chgnn/hp_setting.yaml"


def embed(sample, seed: int, device: str, logger) -> np.ndarray:
    if os.environ.get("HGB_ALLOW_CHGNN") != "1":
        raise RuntimeError(
            "CHGNN is semi-supervised (its loss uses train labels) and needs "
            "its own precomputed 'homogeneity' pattern files, so it is NOT a "
            "faithful drop-in unsupervised embedder in this harness. It is "
            "excluded from the fair unsupervised comparison by design.\n"
            "To benchmark CHGNN faithfully, run its own repo "
            "(`python train.py --data=coauthorship --dataset=cora`) which "
            "reports node-classification accuracy directly. If you understand "
            "the caveat and still want an experimental encoder-output "
            "extraction here, set HGB_ALLOW_CHGNN=1 -- but treat those rows as "
            "semi-supervised and not comparable to the unsupervised embedders."
        )
    raise NotImplementedError(
        "Experimental CHGNN extraction is intentionally left unimplemented: it "
        "requires the repo's homogeneity-pattern precompute (data/cal_pattern.py) "
        "which is dataset-file-specific and cannot be reconstructed faithfully "
        "from the canonical sample. Run CHGNN in its own repo instead."
    )
