"""
hgb/config.py
=============

Single source of truth for the hypergraph node-representation benchmark:
which datasets exist, which embedders/classifiers are registered, the seeds,
the evaluation protocol and all on-disk paths.

Nothing in here is method-specific hyperparameters -- each embedder keeps its
OWN official defaults inside its adapter (loaded from the upstream repo's own
config file where one exists), and each classifier keeps the official AllSet
defaults inside the AllSet adapter.  This file only holds harness-level knobs
that are shared by *every* method, which is what keeps the comparison fair.
"""

from __future__ import annotations

import os


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def resolve_device() -> str:
    forced = os.environ.get("HGB_DEVICE")
    if forced:
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = resolve_device()

# --------------------------------------------------------------------------- #
# Reproducibility / protocol
# --------------------------------------------------------------------------- #
SEEDS = [0, 1, 2]                # each (dataset, embedder) is run once per seed
EMB_DIM = 128                    # target embedding width for methods that expose it
LOG_EVERY = 10                   # deep encoders / classifiers log every N epochs

# Downstream classifier split protocol = TriCL (Lee & Shin, AAAI'23, Sec. 4.2):
# random 10% / 10% / 80% train / valid / test, 20 splits.  Split i is exactly
# TriCL's  dataset/<name>/splits/{i}.pickle  (np.random.default_rng(i)
# permutation).  The splits are FIXED -- the same for every embedder,
# classifier and seed -- so only the method differs between rows.
#   HGB_N_SPLITS=k   use only the first k splits (quick runs)
TRAIN_PROP = 0.10
VALID_PROP = 0.10
N_CLF_SPLITS = int(os.environ.get("HGB_N_SPLITS", 3))
# stored with every classifier result so results from a different split setup
# are never resumed or mixed into the summary tables
SPLIT_TAG = f"tricl_{int(TRAIN_PROP*100)}-{int(VALID_PROP*100)}_x{N_CLF_SPLITS}"

# KMeans clustering (ARI / NMI) is run this many times per embedding with
# different KMeans seeds; we report the mean.
N_CLUSTER_RUNS = 5

# --------------------------------------------------------------------------- #
# Paths (relative to the repo root = parent of this file's package)
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_ROOT = os.path.join(ROOT, "vendor")
DATA_ROOT = os.environ.get("HGB_DATA_ROOT", os.path.join(ROOT, "data"))
RESULTS_ROOT = os.path.join(ROOT, "results")
EMB_CACHE_ROOT = os.path.join(ROOT, "embeddings")
LOGS_ROOT = os.path.join(ROOT, "logs")

# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
# Canonical benchmark name  ->  (loader_kind, on-disk relative path under DATA_ROOT)
#
# loader_kind == "pickle_dir": a directory with features.pickle (scipy sparse),
# hypergraph.pickle (dict {edge_key: [node_ids]}) and labels.pickle (1-D array),
# which is the format shared by TriCL / SE-HSSL / CHGNN.  This is the one
# faithful format we standardize on.
#
# available=False means we have NO faithful data source for it in the material
# supplied, so it is registered (for transparency in the tables) but skipped
# with a clear message rather than being silently invented.
DATASETS = {
    # name          kind          relative path                       available
    "cora_c":     ("pickle_dir", "cocitation/cora",                   True),
    "citeseer":   ("pickle_dir", "cocitation/citeseer",              True),
    "pubmed":     ("pickle_dir", "cocitation/pubmed",                True),
    "cora_a":     ("pickle_dir", "coauthorship/cora",                True),
    "dblp":       ("pickle_dir", "coauthorship/dblp",                True),
    "modelnet40": ("pickle_dir", "ModelNet40",                        True),
    "zoo":        ("pickle_dir", "zoo",                               True),
    "20news":     ("pickle_dir", "20newsW100",                        True),
    "mushroom":   ("pickle_dir", "Mushroom",                          True),
    "ntu2012":    ("pickle_dir", "NTU2012",                           True),
    # ---- requested but NOT present in any supplied repo (do not fabricate) ----
    "imdb":       ("pickle_dir", "IMDB",                              False),
    "aminer":     ("pickle_dir", "AMiner",                            False),
    "dblp_a":     ("pickle_dir", "DBLP-A",                            False),
    "dblp_p":     ("pickle_dir", "DBLP-P",                            False),
    "house":      ("pickle_dir", "House",                             False),
}

# Human-readable labels for the result tables.
DATASET_DISPLAY = {
    "cora_c": "Cora-C", "citeseer": "Citeseer", "pubmed": "Pubmed",
    "cora_a": "Cora-A", "dblp": "DBLP", "modelnet40": "ModelNet40",
    "zoo": "Zoo", "20news": "20News", "mushroom": "Mushroom",
    "ntu2012": "NTU2012", "imdb": "IMDB", "aminer": "AMiner",
    "dblp_a": "DBLP-A", "dblp_p": "DBLP-P", "house": "House",
}

AVAILABLE_DATASETS = [k for k, v in DATASETS.items() if v[2]]
ALL_DATASETS = list(DATASETS.keys())

# --------------------------------------------------------------------------- #
# Embedder registry.  entry = (module_path, requires_torch, requires_features,
#                              is_unsupervised)
# The adapters are imported lazily so the pure-numpy method (and the whole
# harness) works even in an environment without torch installed.
# --------------------------------------------------------------------------- #
EMBEDDERS = {
    "HyperFuse": dict(module="hgb.embedders.HyperFuse", torch=False,
                         feats=True, unsupervised=True,
                         display="HyperFuse (ours)"),
    "tricl":        dict(module="hgb.embedders.tricl", torch=True,
                         feats=True, unsupervised=True, display="TriCL"),
    "sehssl":       dict(module="hgb.embedders.sehssl", torch=True,
                         feats=True, unsupervised=True, display="SE-HSSL"),
    "villain":      dict(module="hgb.embedders.villain", torch=True,
                         feats=False, unsupervised=True, display="VilLain"),
    "hypeboy":      dict(module="hgb.embedders.hypeboy", torch=True,
                         feats=True, unsupervised=True, display="HypeBoy"),
    # semi-supervised: uses train labels while learning the representation.
    # NOT part of the fair unsupervised comparison; opt in explicitly.
    "chgnn":        dict(module="hgb.embedders.chgnn", torch=True,
                         feats=True, unsupervised=False, display="CHGNN (semi-sup)"),
}

# The default embedder set for a "fair unsupervised" run (excludes CHGNN).
DEFAULT_EMBEDDERS = ["HyperFuse", "tricl", "sehssl", "villain", "hypeboy"]

# --------------------------------------------------------------------------- #
# Classifier registry.  AllSet ones are faithful to the official repo (vendored).
# key -> AllSet `--method` string.  "UniGCN" is served by AllSet's UniGNN-family
# model UniGCNII (the only Uni model AllSet wires into its method dispatch); the
# display name records that mapping honestly.
# --------------------------------------------------------------------------- #
CLASSIFIERS = {
    # structure-free baselines (scikit-learn, embedding vectors only;
    # hgb/classifiers/flat_clf.py)
    "logreg":   dict(backend="flat",   allset_method=None,  display="Logistic Regression (no structure)"),
    "mlp":      dict(backend="flat",   allset_method=None,  display="MLP (no structure)"),
    # hypergraph-aware classifiers from the official AllSet repo
    "allset":   dict(backend="allset", allset_method="AllSetTransformer", display="AllSet (AllSetTransformer)"),
    "hnhn":     dict(backend="allset", allset_method="HNHN",              display="HNHN"),
    "hgnn":     dict(backend="allset", allset_method="HGNN",              display="HGNN"),
    "hypergcn": dict(backend="allset", allset_method="HyperGCN",          display="HyperGCN"),
    "unigcn":   dict(backend="allset", allset_method="UniGCNII",          display="UniGCN (AllSet UniGCNII)"),
}
DEFAULT_CLASSIFIERS = list(CLASSIFIERS.keys())

METRICS = ["accuracy", "macro_f1", "ari", "nmi", "embed_time_s"]
