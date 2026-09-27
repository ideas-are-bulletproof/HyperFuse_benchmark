"""hgb/classifiers -- downstream classifiers on top of frozen embeddings.

  flat   : logreg, mlp                 (flat_clf.py, structure-free)
  allset : allset, hnhn, hgnn, hypergcn, unigcn   (allset_clf.py, official AllSet)
"""
from __future__ import annotations

from .. import config as C


def get_classifier_method(name: str) -> str:
    if name not in C.CLASSIFIERS:
        raise KeyError(f"unknown classifier {name!r}; known: {list(C.CLASSIFIERS)}")
    spec = C.CLASSIFIERS[name]
    return spec["allset_method"] or name


def run_downstream(name: str, emb, sample, splits, seed: int, device: str, logger,
                   done=None, on_split=None) -> dict:
    """Run classifier `name` over all splits.

    done     : {split_id: (acc, f1)} already finished -> those splits are skipped
    on_split : callback(split_id, acc, f1) called right after each new split
    """
    if name not in C.CLASSIFIERS:
        raise KeyError(f"unknown classifier {name!r}; known: {list(C.CLASSIFIERS)}")
    spec = C.CLASSIFIERS[name]
    if spec["backend"] == "flat":
        from .flat_clf import run_flat
        return run_flat(name, emb, sample, splits, seed, logger,
                        done=done, on_split=on_split)
    from .allset_clf import run_classifier
    return run_classifier(spec["allset_method"], emb, sample, splits, seed,
                          device, logger, done=done, on_split=on_split)
