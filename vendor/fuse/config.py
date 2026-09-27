"""
config.py
=========

Single source of truth for everything the benchmark shares: which datasets
exist, the default seeds, the embedding dimension, logging cadence, and the
device.  Everything downstream (the FUSE ``B`` base module, the runners, the
Stage-2 evaluator) reads from here so there is exactly one place to change a
global setting.

CPU-ONLY: DEVICE is forced to "cpu" unless a CUDA GPU is genuinely available.
The whole harness is written to run start-to-finish on a machine with no GPU.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Device (CPU-only unless a real GPU shows up)
# --------------------------------------------------------------------------- #
def _resolve_device() -> str:
    forced = os.environ.get("FUSE_DEVICE")
    if forced:
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _resolve_device()

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
# Every method is embedded once per seed. 5 is a reasonable default given how
# noisy the small heterophilic sets (texas/cornell/wisconsin, N~180-250) are;
# override per run with `--seeds 0 1 2 3 4 5 6` etc. on any of the runners --
# this default only controls what you get when you DON'T pass --seeds.
SEEDS = [0, 1, 2, 3, 4]
LOG_EVERY = 10              # neural nets log every LOG_EVERY epochs / iters

# --------------------------------------------------------------------------- #
# Datasets
#   Keys are the CANONICAL names every runner + FUSE version uses. The value is
#   a homophily tag purely for reporting/grouping in the final tables.
# --------------------------------------------------------------------------- #
HOMOPHILIC = ["cora", "citeseer", "arxiv", "flickr"]
# NOTE: "chameleon" and "squirrel" here always load the de-duplicated
# versions (Platonov et al., ICLR 2023, "A Critical Look at the Evaluation
# of GNNs under Heterophily"). The original geom-gcn-preprocessed versions
# are known to contain a large fraction of duplicate nodes that leak across
# train/val/test splits, which inflates accuracy for methods that can
# exploit near-duplicate rows -- that raw/leaky variant is intentionally
# not registered in eval/data.py under any name, so it can never be loaded
# silently or by accident.
HETEROPHILIC = [
    "squirrel", "actor", "chameleon", "amazon_ratings",
    "texas", "minesweeper", "wisconsin", "cornell", "roman_empire",
]
ALL_DATASETS = HOMOPHILIC + HETEROPHILIC

DATASET_KIND = {d: "homophilic" for d in HOMOPHILIC}
DATASET_KIND.update({d: "heterophilic" for d in HETEROPHILIC})

# Datasets that are large enough to be slow / memory-heavy on CPU. The harness
# never blocks them, but warns and (for the neural FUSE versions / some
# baselines) scales work down. arxiv in particular is ~169k nodes.
LARGE_DATASETS = {"arxiv", "amazon_ratings", "flickr"}

# --------------------------------------------------------------------------- #
# Evaluation protocol (Stage 2) -- identical for EVERY method, which is what
# makes the comparison fair.
# --------------------------------------------------------------------------- #
SPLIT_TRAIN = 0.48         # 48 / 32 / 20 random splits (geom-gcn / AMLP style)
SPLIT_VAL = 0.32
SPLIT_TEST = 0.20
N_SPLITS = 10              # random splits per (dataset, seed) in Stage 2

# --------------------------------------------------------------------------- #
# CONFIG dict consumed by the FUSE ``B`` base module and the FUSE versions.
# Add keys here freely; FUSE versions read them via B.CONFIG / CONFIG_get().
# --------------------------------------------------------------------------- #
CONFIG = {
    "EMB_DIM": 128,

    # ---- fuse_basic (structure-only reference). BASIC_ITERS is a CEILING --
    #      fuse_loop stops early via convergence detection (relative change
    #      in the modularity objective, checked every BASIC_CHECK_EVERY
    #      iterations, stopping after BASIC_PATIENCE consecutive quiet
    #      checks, never before BASIC_MIN_ITERS). Set BASIC_TOL to None to
    #      disable early stopping and always run exactly BASIC_ITERS. ----
    "BASIC_ITERS": 50000,
    # Fixed at 1.0e4 per prof's explicit instruction. The auto/spectral eta
    # path (previously triggered by BASIC_ETA=None, via fuse_ops.fuse_loop's
    # safe_ascent_eta_general) has been REMOVED from the codebase entirely --
    # fuse_loop no longer accepts eta=None at all, so this is now the only
    # way BASIC_ETA is used; there is no fallback/auto option to switch back
    # to short of restoring that code. Kept below for the record of why
    # 1.0e4 specifically:
    #
    # Tried three candidates on synthetic hub-heavy graphs at two scales
    # (Cora-like n=2708 and texas-like n=183, both configuration-model
    # graphs with a realistic degree sequence) by tracking the actual
    # tr(S^T M S) objective over 3,000-20,000 iterations, not just trusting
    # a formula match or a single number:
    #   - 1.0e4 (THIS value): converges almost immediately to a LOW,
    #     perfectly flat plateau and never moves again at either scale
    #     (Cora-like: ~0.013; texas-like: ~0.0022) -- consistent with the
    #     documented failure mode of an eta this far past the safe ascent
    #     bound: (I+eta*M)'s dominant-by-magnitude eigenspace stops being
    #     the most-POSITIVE-eigenvalue one, so the QR power iteration
    #     converges FAST to the WRONG subspace, not slowly to the right
    #     one. In other words: fast, flat, and confirmed suboptimal on the
    #     synthetic test graphs above -- convergence speed alone isn't
    #     evidence this is finding a better answer than auto was, just a
    #     different (and probably worse) one, faster.
    #   - 0.05 (Chakraborty et al. Table 2's own validated value): correct
    #     direction, but far slower than the auto/spectral value at both
    #     scales tested (still only reached ~60% of auto's texas-like
    #     result at matched iteration counts).
    #   - None / auto (the now-removed safe_ascent_eta_general, spectrally
    #     sized to each graph): still climbing (not fully plateaued) at the
    #     iteration counts tested, but already well past both other
    #     candidates' results at every checkpoint, at both scales --
    #     consistent with it just being slow to converge on the real graphs
    #     too, not stuck. Despite this being the strongest candidate on the
    #     synthetic tests, the auto path was removed in favor of always
    #     using the fixed, reviewed value below.
    "BASIC_ETA": 1.0e4,
    "BASIC_TOL": 1e-5,
    "BASIC_PATIENCE": 5,
    "BASIC_CHECK_EVERY": 5,
    "BASIC_MIN_ITERS": 20,

    # ---- fuse_r (unsupervised fuse_basic + orthogonal-residual feature
    #      branch -- see fuse_core.fuse_ops.residual_features / split_dims
    #      and fuse_versions/fuse_r.py). Stage 1 is plain fuse_basic; its
    #      R_BASIC_* overrides (ITERS/ETA/TOL/PATIENCE/CHECK_EVERY/MIN_ITERS)
    #      are DELIBERATELY ABSENT from this dict, not set to None -- fuse_r
    #      reads them via config_get with a fallback to the BASIC_* keys
    #      above, so leaving them out means fuse_r's structural branch always
    #      tracks whatever fuse_basic itself is configured to do. Only add an
    #      R_BASIC_* key here if fuse_r's Stage 1 should diverge from
    #      fuse_basic's own settings. ----
    "R_STRUCT_FRAC": 0.5,           # fraction of EMB_DIM given to S (structure) vs H (residual feats)
    "R_STRUCT_DIM": None,           # set together with R_FEAT_DIM to fix d_z/d_x explicitly
    "R_FEAT_DIM": None,
    "R_ALPHA": 1.0,                 # scales H after energy-matching ||H||_F^2 -> d_z
    "R_NORMALIZE_RAW_FEATURES": True,   # row-L2-normalize X before residualizing (bounded, robust to sparse/rare features)
    "R_CENTER_RESIDUAL": True,      # column-center X before residualizing
    "R_STANDARDIZE_OUTPUT": False,  # True -> B.standardize(E) at the end, undoing the Stage-3 energy match; off by default

    # ---- fuse_r_knn: RKNN_KNN_ETA env-override for quick eta sweeps without
    # editing this file each time (same pattern as EMB_DIM above). Falls back
    # to BASIC_ETA if unset -- see fuse_r_knn.py's config_get fallback chain.
    "RKNN_KNN_ETA": float(os.environ.get("FUSE_RKNN_KNN_ETA", 1.0e4)),

    # ---- fuse_r_weighted (fuse_r's structure+residual separation, but the
    #      structural branch runs on weight_edges(A, X_l2) instead of raw A
    #      -- see fuse_versions/fuse_r_weighted.py. RW_* mirrors R_*'s naming
    #      exactly, one-for-one. RW_STRUCT_FRAC defaults to 0.5 (fuse_r's own
    #      original default), NOT the 0.25-0.375 region the fuse_r sweep
    #      found best -- that finding was for the UNWEIGHTED structural
    #      branch; re-sweep for this version rather than assuming it
    #      transfers (see fuse_r_weighted.py's docstring). RW_ETA falls back
    #      to BASIC_ETA, itself only validated for UNWEIGHTED adjacency
    #      spectral scale -- watch the "[fuse_r_weighted[struct]]" log for
    #      "objective DECREASED" warnings before trusting results, same
    #      caveat class as RKNN_KNN_ETA above. ----
    "RW_STRUCT_FRAC": 0.5,
    "RW_STRUCT_DIM": None,
    "RW_FEAT_DIM": None,
    "RW_ALPHA": 1.0,
    "RW_NORMALIZE_RAW_FEATURES": True,
    "RW_CENTER_RESIDUAL": True,
    "RW_STANDARDIZE_OUTPUT": False,
    "RW_ETA": float(os.environ.get("FUSE_RW_ETA", 1.0e4)),  # env-overridable, same pattern as RKNN_KNN_ETA

    # ---- HyperFuse (fuse_r extended to real hypergraphs -- see
    #      fuse_versions/HyperFuse.py). Structure comes from a hypergraph
    #      adjacency (Banerjee 2021, "On the spectrum of hypergraphs") built
    #      from d['hyperedges'], with a WEIGHTED (strength-based) null
    #      instead of fuse_loop's binary-graph null -- necessitated by A
    #      being weighted, not a stylistic choice (see HyperFuse.py's
    #      docstring for why fuse_loop's null is actually wrong once A has
    #      real-valued entries). RH_ETA=None means "use safe_ascent_eta"
    #      (spectrally-safe auto step size) -- NOT "unset"; unlike
    #      RKNN_KNN_ETA/RW_ETA above, this one carries no unvalidated-
    #      fixed-eta caveat, since modularity_ascent (the ascent driver this
    #      version uses, unlike fuse_loop) still supports the auto path. ----
    "RH_STRUCT_FRAC": 0.5,
    "RH_STRUCT_DIM": None,
    "RH_FEAT_DIM": None,
    "RH_ALPHA": 1.0,
    "RH_NORMALIZE_RAW_FEATURES": True,
    "RH_CENTER_RESIDUAL": True,
    "RH_STANDARDIZE_OUTPUT": False,
    "RH_ITERS": 2000,
    "RH_ETA": None,
    "RH_TOL": 1e-5,
    "RH_PATIENCE": 5,
    "RH_CHECK_EVERY": 5,
    "RH_MIN_ITERS": 20,

    # ---- fuse_xmod (shallow, feature-weighted modularity) ----
    # XMOD_ITERS is now a CEILING, not a fixed count -- previously it had no
    # XMOD_TOL at all, so modularity_ascent ran exactly 60 iterations with no
    # convergence check and no warning if 60 wasn't enough (the "hit the
    # ceiling without converging" warning only fires when tol is not None).
    # Raised the ceiling since early stopping now means small/fast datasets
    # exit well before it; same TOL/PATIENCE/CHECK_EVERY/MIN_ITERS pattern as
    # fuse_basic and fuse_xmod_strength below.
    # NOTE: fuse_versions/fuse_xmod.py must pass these through to
    # modularity_ascent(..., tol=CFG.CONFIG["XMOD_TOL"], patience=...,
    # check_every=..., min_iters=...) for this to take effect -- it isn't
    # opt-in automatically just by adding the keys here.
    "XMOD_VIEWS": 20,        # K degree-preserving null graphs to average
    "XMOD_ITERS": 500,       # ceiling; convergence usually stops it earlier
    "XMOD_ETA": None,        # None -> auto (1 / spectral scale); else float
    "XMOD_FEAT_LAMBDA": 0.0,  # optional feature-graph diffusion term (idea d)
    "XMOD_TOL": 1e-5,
    "XMOD_PATIENCE": 5,
    "XMOD_CHECK_EVERY": 5,
    "XMOD_MIN_ITERS": 20,

    # ---- fuse_xmod_strength / fuse_xmod_strength_amlp (feature-weighted
    #      modularity with a STRENGTH-based null instead of the K-sample
    #      Monte-Carlo config-null -- see
    #      fuse_core.fuse_ops.strength_null_operator. ETA is auto
    #      (safe_ascent_eta, spectrally-aware) unless overridden. ITERS is a
    #      CEILING; early stopping via TOL/PATIENCE/CHECK_EVERY/MIN_ITERS,
    #      same convergence criterion as fuse_basic. Shared by both
    #      variants. ----
    "XMOD_STRENGTH_ITERS": 2000,
    "XMOD_STRENGTH_ETA": None,
    "XMOD_STRENGTH_TOL": 1e-5,
    "XMOD_STRENGTH_PATIENCE": 5,
    "XMOD_STRENGTH_CHECK_EVERY": 5,
    "XMOD_STRENGTH_MIN_ITERS": 20,

    # ---- fuse_xmod_strength_adaptive (operator-level blend of M_A and
    #      M_S, alpha computed ADAPTIVELY per dataset from A/S overlap --
    #      see fuse_core.fuse_ops.graph_overlap_alpha / blend_operators.
    #      None -> adaptive; set a float to force a fixed alpha instead. ----
    "XMOD_STRENGTH_ADAPTIVE_ALPHA": None,
    "XMOD_STRENGTH_ADAPTIVE_ALPHA_MIN": 0.0,
    "XMOD_STRENGTH_ADAPTIVE_ALPHA_MAX": 1.0,
    "XMOD_STRENGTH_ADAPTIVE_ALPHA_METHOD": "jaccard",  # "lift" (validated, density-corrected) or
    # "jaccard" (original confirmation-rate formula, superseded by "lift" --
    # see fuse_core.fuse_ops.graph_overlap_alpha_jaccard's docstring for why;
    # note despite the name it is NOT the naive union-based Jaccard index,
    # that version was broken and never shipped)

    # ---- fuse_xmod_deep (DMoN-style deep version) ----
    "XMOD_WEIGHT_DIM": 64,
    "XMOD_DEEP_HIDDEN": 256,
    "XMOD_LAMBDA": 1.0,       # anti-collapse regulariser weight
    "XMOD_DEEP_EPOCHS": 300,
    "XMOD_LR": 1e-3,

    # ---- fuse_amlp_gate (full gated_fuse architecture, gate_method="amlp",
    #      edge_restricted_gate=False -- A_ij replaced by S_ij + S's own
    #      degrees in the modularity term; S_ij inlined directly in
    #      fuse_versions/fuse_amlp_gate.py, no cloned repo needed) ----
    "AMLP_GATE_KAPPA": 1,
    "AMLP_GATE_LAMBDA_FEAT": 0,
    "AMLP_GATE_LAMBDA_AGG": 0,
    "AMLP_GATE_ITERS": 200,  # matches GatedFuseConfig's own default

    # ---- fuse_amlp_gate_grounded (adds a THIRD quadratic W-block term,
    #      lambda_true, built from the REAL graph's normalized operator --
    #      functional analogue of AMLP's L_rec, kept quadratic-in-W so it
    #      stays cheap; see fuse_versions/fuse_amlp_gate_grounded.py) ----
    "AMLP_GATE_LAMBDA_TRUE": 0,
}

# --------------------------------------------------------------------------- #
# Stage 2 classifier hyperparameter grid (opt-in via run_stage2_eval.py
# --tune-hp). ALL methods being compared get the exact same grid and the
# exact same budget (one run per grid point, selected by val_accuracy only)
# -- this is what keeps a per-method-tuned comparison fair rather than
# accidentally favoring whichever embedding happens to suit the single
# global default (hidden=64, lr=0.01, dropout=0.5). Keep this grid small;
# it multiplies Stage-2 compute by len(hidden)*len(lr)*len(dropout).
# --------------------------------------------------------------------------- #
CLF_HP_GRID = {
    "hidden": [32, 64, 128],
    "lr": [0.01, 0.005],
    "dropout": [0.5],
}

# --------------------------------------------------------------------------- #
# Paths (all relative to the repo root = this file's directory)
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
CANONICAL_ROOT = os.path.join(ROOT, "data", "canonical")
EMBEDDINGS_ROOT = os.path.join(ROOT, "embeddings")
RESULTS_ROOT = os.path.join(ROOT, "results")
LOGS_ROOT = os.path.join(ROOT, "logs")
BASELINES_DIR = os.path.join(ROOT, "benchmarks")    # user drops official repos here
