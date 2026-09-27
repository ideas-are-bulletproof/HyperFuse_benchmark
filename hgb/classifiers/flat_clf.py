"""
Structure-free classifiers: Logistic Regression and MLP (scikit-learn) on the
node-embedding vectors alone -- the hypergraph is ignored.

Moved here from benchmark_flat.py so they run inside benchmark.py with the same
splits, the same result files and the same resume logic as the AllSet
classifiers.  Settings are the ones from benchmark_flat.py:

  * features z-scored with a StandardScaler fitted on the training nodes
  * trained on the train nodes only (the valid nodes are not used)
  * logreg : LogisticRegression(max_iter=2000)
  * mlp    : MLPClassifier(hidden=(256,), relu, adam, max_iter=500,
             early_stopping=True, n_iter_no_change=20)
  * random_state = seed*100 + split_id

One robustness fix: sklearn's early stopping holds out 10% of the TRAIN nodes,
stratified by class.  With 10% training labels that is impossible on tiny
datasets (Zoo has 10 training nodes over 7 classes) and sklearn raises.  In
that case early stopping is switched off for that split and it is logged.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..seeding import set_global_seed

VALIDATION_FRACTION = 0.1   # sklearn default used by early_stopping


def _early_stopping_possible(y_train: np.ndarray) -> bool:
    n = len(y_train)
    classes, counts = np.unique(y_train, return_counts=True)
    n_val = math.ceil(VALIDATION_FRACTION * n)
    return (counts.min() >= 2 and n_val >= len(classes)
            and (n - n_val) >= len(classes))


def make_classifier(name: str, random_state: int, early_stopping: bool = True):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256,), activation="relu",
                             solver="adam", max_iter=500,
                             early_stopping=early_stopping,
                             validation_fraction=VALIDATION_FRACTION,
                             n_iter_no_change=20, random_state=random_state)
    raise ValueError(f"unknown flat classifier {name!r} (logreg, mlp)")


def run_flat(name: str, emb: np.ndarray, sample, splits, seed: int, logger,
             done: dict | None = None, on_split=None, standardize: bool = True) -> dict:
    """Same contract as allset_clf.run_classifier (done / on_split = resume)."""
    X = np.asarray(emb, dtype=np.float64)
    y = np.asarray(sample.y).ravel()
    done = done or {}
    accs, f1s = [], []
    for si, split in enumerate(splits):
        if si in done:
            a, f = done[si]
            accs.append(a); f1s.append(f)
            continue
        tr, te = np.asarray(split["train"]), np.asarray(split["test"])
        Xtr, Xte = X[tr], X[te]
        if standardize:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        rs = int(seed) * 100 + si
        set_global_seed(rs)
        es = True
        if name == "mlp" and not _early_stopping_possible(y[tr]):
            es = False
            if si == 0:
                logger.info(f"    [{name}] only {len(tr)} training nodes -> "
                            "early_stopping disabled (cannot hold out a "
                            "stratified validation set)")
        model = make_classifier(name, rs, early_stopping=es)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        a = float(accuracy_score(y[te], pred))
        f = float(f1_score(y[te], pred, average="macro", zero_division=0))
        accs.append(a); f1s.append(f)
        if on_split is not None:
            on_split(si, a, f)
        if (si + 1) % 5 == 0 or si == len(splits) - 1:
            logger.info(f"    [{name}] split {si+1}/{len(splits)} acc={a:.4f} macroF1={f:.4f}")

    return {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s)),
            "split_accuracy": accs, "split_macro_f1": f1s}
