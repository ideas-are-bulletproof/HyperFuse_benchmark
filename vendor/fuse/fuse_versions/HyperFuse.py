"""
HyperFuse  (Eureka utility + hypergraph CCA self-supervision)  -- single file.

Self-contained: numpy + scipy for stages A/B, torch for stage C.
No imports from the benchmark repo (fuse_core, hgb).

    from fuse_versions.HyperFuse import embed
    Z = embed(hyperedges, X, num_nodes=N, seed=0, device="cuda")

    hyperedges : list of node-id lists (ids in 0..N-1)
    X          : [N, F] node features, dense array or scipy sparse
    returns    : [N, emb_dim] float32 (default 128), rows L2-normalised

Pipeline (no labels anywhere)
-----------------------------
A  Structure.  Banerjee's hypergraph adjacency A = H W H^T - diag(H W 1) and
   its strength-null modularity operator, applied matrix-free; projected
   gradient ascent + QR gives S (emb_dim/4 columns, orthonormal).
B  Features.  X residualised against S, two corrupted views (feature masking /
   membership masking), three propagation scales, agreement-gated and
   SVD-reduced to F (emb_dim/2); G (emb_dim/4) = top singular vectors of the
   normalised incidence.  Eureka utility: 13 descriptors per hyperedge ->
   ridge -> hyperedge weights w_e in [0.65, 1.35], mean 1.
C  Self-supervision (replaces SAGE).  Hypergraph CCA:
     * two views per epoch, built from the SAME two corruptions as stage B:
       feature masking of the raw features (only when they are sparse; dense
       continuous descriptors are left intact) and membership masking of the
       incidence (each node-hyperedge entry dropped independently);
     * a 2-layer hypergraph encoder whose hyperedge->node step is the
       utility-weighted mean (so the Eureka weights shape every message),
       reading the raw (sparse) features plus the stage-B summary [F | G],
       followed by `hops` parameter-free lazy propagation steps (SSGC-style
       average);
     * loss = || Z1 - Z2 ||^2  +  lam * ( ||Z1^T Z1 - I||^2 + ||Z2^T Z2 - I||^2 )
       on column-standardised outputs (Zhang et al., NeurIPS 2021): view
       invariance plus decorrelation, no negatives, no projector, no target
       network, O(N d^2) per epoch.
   The encoder's output layer produces exactly emb_dim columns.

Dimensions -- nothing is ever zero-padded
------------------------------------------
S, F, G default to emb_dim/4, emb_dim/2, emb_dim/4 (32/64/32 for 128).  They
are internal signals (residualisation, utility descriptors, encoder input);
if the data cannot fill a block (e.g. fewer features than the block width) the
block is simply narrower.  The output width is set by a learned linear layer,
so it is always exactly emb_dim.

Why not SAGE: the skip-gram/negative-sampling objective could not use the raw
features (feeding them in made it worse), so it had to work from the 88-d
compressed [F | G], zero-padded to 128.  The CCA objective uses the raw
features directly, and in our runs this is where the accuracy gain comes from.
"""
from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["embed", "DEFAULTS", "stage_ab", "feats_hyperfuse"]

LOG = logging.getLogger("hyperfuse")

DEFAULTS = dict(
    # dimensions (None -> derived from emb_dim: /4, /2, /4)
    emb_dim=128, struct_dim=None, feat_dim=None, group_dim=None,
    # stage A
    ascent_iters=1000, ascent_tol=1e-5, ascent_patience=5, ascent_check_every=5,
    ascent_min_iters=20, ascent_eta=None,
    # stage B
    mask_rate=0.10, temperature=0.25, min_scale_weight=0.05, beta=0.5,
    lazy_alpha=0.5, ssgc_hops=3, normalize_raw_features=True, center_residual=True,
    # eureka utility
    utility="ridge",                      # ridge | target | uniform
    utility_epochs=160, utility_lr=0.03, utility_l2=5e-2, utility_temp=0.8,
    weight_min=0.65, weight_max=1.35,
    # stage C (hypergraph CCA)
    hidden=512, epochs=100, lr=1e-3, wd=0.0, lam=1e-3,
    feature_drop="auto",                  # auto: 0.3, or 0 for dense real-valued descriptors
    membership_drop=0.3,
    hops=2, hop_alpha=0.5,
    encoder_input="x_fg",                 # x_fg | x | fg
    sparse_threshold=0.25,                # features count as sparse if nonzero density < this
    l2_normalise_output=True,
)


def _dims(c):
    d = int(c["emb_dim"])
    ds = int(c["struct_dim"]) if c["struct_dim"] is not None else d // 4
    dg = int(c["group_dim"]) if c["group_dim"] is not None else d // 4
    df = int(c["feat_dim"]) if c["feat_dim"] is not None else d - ds - dg
    return d, ds, df, dg

# =========================================================================== #
# stage A/B helpers
# =========================================================================== #
def _incidence(hyperedges, n):
    rows, cols = [], []
    for j, e in enumerate(hyperedges):
        nodes = np.unique(np.asarray(tuple(e), dtype=np.int64))
        nodes = nodes[(nodes >= 0) & (nodes < n)]
        if len(nodes) < 2:
            continue
        rows.extend(nodes.tolist()); cols.extend([j] * len(nodes))
    if not rows:
        return sp.csr_matrix((n, 0), dtype=np.float64), np.zeros(0), np.zeros(n)
    H = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(hyperedges)))
    H.eliminate_zeros()
    return H.tocsr(), np.asarray(H.sum(0)).ravel(), np.asarray(H.sum(1)).ravel()


def _csc(H):
    Hc = sp.csc_matrix(H, copy=True)
    Hc.sort_indices()
    return Hc


def _R(H, es, nd):
    if H.shape[1] == 0:
        return sp.csr_matrix((H.shape[0], 0), dtype=np.float64)
    return (sp.diags(1 / np.sqrt(np.maximum(nd, 1))) @ H
            @ sp.diags(1 / np.sqrt(np.maximum(es, 1)))).tocsr()


def _P(R, X):
    if R.shape[1] == 0 or X.shape[1] == 0:
        return np.zeros_like(X, dtype=np.float64)
    return np.asarray(R @ (R.T @ X), dtype=np.float64)


def _lazy_P(R, X, alpha):
    return alpha * X + (1.0 - alpha) * _P(R, X)


def _scales(R, X, alpha, K):
    """raw, lazy 1-hop, SSGC-averaged K hops."""
    X = np.asarray(X, dtype=np.float64)
    s1 = _lazy_P(R, X, alpha)
    acc = np.zeros_like(X); cur = X
    for _ in range(max(K, 1)):
        cur = _lazy_P(R, cur, alpha); acc += cur
    return (X, s1, acc / float(max(K, 1)))


def _cos(A, B):
    return np.sum(A * B, axis=1) / np.maximum(
        np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-12)


def _mask_features(X, seed, rate):
    if rate <= 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    keep = rng.random(X.shape) >= rate
    return X * keep / max(1 - rate, 1e-8)


def _mask_memberships(H, seed, rate):
    """Drop a fraction of each hyperedge's members (never below 2)."""
    if H.shape[1] == 0 or rate <= 0:
        return H.copy()
    rng = np.random.default_rng(seed)
    Hc = _csc(H)
    indptr, indices = Hc.indptr, Hc.indices
    data = Hc.data.copy()
    for e in range(Hc.shape[1]):
        lo, hi = indptr[e], indptr[e + 1]
        n_mem = hi - lo
        if n_mem <= 2:
            continue
        k = min(int(np.floor(rate * n_mem)), n_mem - 2)
        if k > 0:
            members = indices[lo:hi]
            drop = rng.choice(members, size=k, replace=False)
            data[lo + np.searchsorted(members, drop)] = 0
    out = sp.csc_matrix((data, indices, indptr), shape=Hc.shape).tocsr()
    out.eliminate_zeros()
    return out


def _weights(sa, sb, T, minw):
    C = np.stack([np.nan_to_num(_cos(a, b), nan=0, posinf=1, neginf=-1)
                  for a, b in zip(sa, sb)], axis=1)
    z = C / max(T, 1e-4); z -= z.max(1, keepdims=True)
    W = np.exp(np.clip(z, -40, 40)); W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    if minw > 0:
        W = (1 - 3 * minw) * W + minw
    W /= W.sum(1, keepdims=True)
    return W, C


def _topk_svd(X, q, seed):
    n, f = X.shape
    q = max(1, min(q, min(n, f) - 1))
    Xf = X.asfptype() if sp.issparse(X) else X
    U, S, _ = spla.svds(Xf, k=q, random_state=seed)
    order = np.argsort(-S)
    return U[:, order], S[order]


def _sign_fix(U):
    """Deterministic column signs (largest-|entry| positive); an isometry."""
    idx = np.argmax(np.abs(U), axis=0)
    sgn = np.sign(U[idx, np.arange(U.shape[1])]); sgn[sgn == 0] = 1.0
    return U * sgn


def _randomized_svd(R, k, seed, oversample=10, iters=60):
    """Block subspace iteration; robust to repeated singular values."""
    rng = np.random.default_rng(seed)
    l = min(k + oversample, min(R.shape))
    Q, _ = np.linalg.qr(np.asarray(R @ rng.standard_normal((R.shape[1], l))))
    for _ in range(iters):
        Q, _ = np.linalg.qr(np.asarray(R.T @ Q))
        Q, _ = np.linalg.qr(np.asarray(R @ Q))
    B = np.asarray((R.T @ Q).T)                       # l x E, small
    Ub, s, _ = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k]


def _incidence_svd(R, k, seed, logger=LOG):
    """Top-k left singular vectors of the normalised incidence R.

    sigma_1 = 1 with multiplicity = #connected components, so the spectrum
    is clustered;
    Order: exact dense SVD (small R) -> LOBPCG (block solver, handles
    clusters) -> randomized subspace iteration."""
    n = R.shape[0]
    kk = min(int(k), R.shape[0] - 1, R.shape[1] - 1)
    if kk <= 0:
        return np.zeros((n, 0), np.float32)
    R = R.astype(np.float64)
    U = s = None

    if min(R.shape) <= 3000:
        Uf, sf, _ = np.linalg.svd(R.toarray() if sp.issparse(R) else R,
                                  full_matrices=False)
        U, s = Uf[:, :kk], sf[:kk]
    else:
        try:
            with warnings.catch_warnings():
                # LOBPCG warns when it misses its (very strict) internal tol;
                # the residual check below is the real acceptance test.
                warnings.filterwarnings("ignore", message="Exited",
                                        category=UserWarning)
                U, s, _ = spla.svds(R, k=kk, solver="lobpcg",
                                    random_state=seed, maxiter=500)
            resid = np.linalg.norm(R @ (R.T @ U) - U * s ** 2, axis=0).max()
            if not np.isfinite(resid) or resid > 1e-2:
                raise RuntimeError(f"LOBPCG residual {resid:.2e}")
        except Exception as exc:
            logger.info(f"  [hyperfuse] incidence SVD: LOBPCG failed ({exc}); "
                        f"using randomized subspace iteration")
            U, s = _randomized_svd(R, kk, seed)

    order = np.argsort(-s); U, s = U[:, order], s[order]
    Z = _sign_fix(np.asarray(U)) * s[None, :]
    Z *= np.sqrt(kk) / (np.linalg.norm(Z) + 1e-12)
    return Z.astype(np.float32)

# ---- FUSE modularity on Banerjee's hypergraph adjacency -------------------- #
def banerjee_modularity_operator(H, edge_weights=None):
    """M = A - s s^T / 2W as a LinearOperator; A never materialised."""
    H = sp.csr_matrix(H, dtype=np.float64)
    size = np.asarray(H.sum(0)).ravel()
    W = np.zeros_like(size)
    ok = size >= 2
    W[ok] = 1.0 / (size[ok] - 1.0)
    if edge_weights is not None:
        W = W * np.asarray(edge_weights, dtype=np.float64)
    Ht = H.T.tocsr()
    self_w = np.asarray(H @ W).ravel()
    s = np.asarray(H @ (W * np.maximum(size - 1.0, 0.0))).ravel()
    two_w = max(float(s.sum()), 1e-8)

    def _apply(Z):
        Z = np.asarray(Z, dtype=np.float64)
        one_d = Z.ndim == 1
        if one_d:
            Z = Z[:, None]
        AZ = H @ (W[:, None] * (Ht @ Z)) - self_w[:, None] * Z
        out = AZ - s[:, None] * (s @ Z)[None, :] / two_w
        return out[:, 0] if one_d else out

    return spla.LinearOperator(shape=(H.shape[0],) * 2, dtype=np.float64,
                               matvec=_apply, rmatvec=_apply, matmat=_apply)


def safe_ascent_eta(M, safety=0.9, max_dim_for_exact=20_000):
    """Step size keeping ascent on the most-positive eigenspace of M."""
    N = M.shape[0]
    try:
        if N > max_dim_for_exact:
            raise RuntimeError("large graph: use the power-iteration bound")
        lam_max = float(spla.eigsh(M, k=1, which="LA", return_eigenvectors=False)[0])
        lam_min = float(spla.eigsh(M, k=1, which="SA", return_eigenvectors=False)[0])
    except Exception:
        rng = np.random.default_rng(0)
        v = rng.normal(size=N); v /= np.linalg.norm(v) + 1e-12
        rho = 0.0
        for _ in range(50):
            v = M @ v
            nrm = np.linalg.norm(v)
            if nrm < 1e-12:
                break
            v /= nrm; rho = nrm
        lam_max, lam_min = rho, -rho
    denom = lam_max + abs(min(lam_min, 0.0))
    return 1.0 if denom <= 1e-9 else safety * (2.0 / denom)


def modularity_ascent(M, dim, iters, seed, eta=None, tol=1e-5, patience=5,
                      check_every=5, min_iters=20, logger=LOG):
    """max tr(Z^T M Z) s.t. Z^T Z = I, by projected gradient ascent + QR."""
    rng = np.random.default_rng(seed)
    Z, _ = np.linalg.qr(rng.normal(size=(M.shape[0], dim)))
    if eta is None:
        eta = safe_ascent_eta(M)
    prev_obj, quiet = None, 0
    for t in range(iters):
        grad = M @ Z
        if tol is not None and (t + 1) % check_every == 0:
            obj = float(np.trace(Z.T @ grad))
            if prev_obj is not None:
                rel = abs(obj - prev_obj) / (abs(prev_obj) + 1e-12)
                quiet = quiet + 1 if rel < tol else 0
                if quiet >= patience and (t + 1) >= min_iters:
                    logger.info(f"  [modularity] converged at iter {t + 1}/{iters} (rel={rel:.2e})")
                    Z, _ = np.linalg.qr(Z + eta * grad)
                    return Z
            prev_obj = obj
        Z, _ = np.linalg.qr(Z + eta * grad)
    return Z


# ---- Eureka: hyperedge descriptors, targets, utility weights --------------- #
def _rownorm(Z):
    Z = np.asarray(Z, dtype=np.float64)
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def _edge_moments(B_, cnt, v):
    s1 = np.asarray(B_.T @ v).ravel()
    s2 = np.asarray(B_.T @ (v * v)).ravel()
    c = np.maximum(cnt, 1.0)
    mu = s1 / c
    return mu, np.sqrt(np.maximum(s2 / c - mu * mu, 0.0))


def _pair_cos_mean(B_, cnt, Z):
    """Mean pairwise cosine inside each hyperedge, via
    sum_{i<j} q_i.q_j = (||sum q||^2 - sum ||q||^2) / 2."""
    if Z.shape[1] == 0:
        return np.zeros(B_.shape[1])
    Q = _rownorm(Z)
    sums = np.asarray(B_.T @ Q)
    sq = np.asarray(B_.T @ np.sum(Q * Q, axis=1)).ravel()
    pairs = cnt * (cnt - 1.0)
    out = np.zeros(B_.shape[1])
    ok = pairs > 0
    out[ok] = (np.sum(sums[ok] ** 2, axis=1) - sq[ok]) / pairs[ok]
    return np.clip(out, -1.0, 1.0)


def _centroid_absmean(B_, cnt, Z):
    if Z.shape[1] == 0:
        return np.zeros(B_.shape[1])
    cent = np.asarray(B_.T @ np.asarray(Z, dtype=np.float64)) / np.maximum(cnt, 1.0)[:, None]
    return np.mean(np.abs(_rownorm(cent)), axis=1)


def hyperedge_descriptors(H, S, F, G, C, nd):
    """13 descriptors per hyperedge with >= 2 members."""
    Hc = _csc(H)
    cnt = np.diff(Hc.indptr).astype(np.float64)
    valid = np.flatnonzero(cnt >= 2)
    if len(valid) == 0:
        return np.zeros((0, 13)), valid.astype(np.int64)
    B_ = sp.csc_matrix((np.ones_like(Hc.data, dtype=np.float64), Hc.indices, Hc.indptr),
                       shape=Hc.shape)
    C = np.asarray(C, dtype=np.float64)
    denom = np.maximum(cnt * C.shape[1], 1.0)
    cmean = np.asarray(B_.T @ C.sum(1)).ravel() / denom
    cstd = np.sqrt(np.maximum(np.asarray(B_.T @ (C * C).sum(1)).ravel() / denom - cmean ** 2, 0.0))
    cmin = np.zeros(len(cnt))
    nz = cnt > 0
    cmin[nz] = np.minimum.reduceat(C.min(1)[Hc.indices], Hc.indptr[:-1][nz])
    degm, degsd = _edge_moments(B_, cnt, np.asarray(nd, dtype=np.float64))
    desc = np.column_stack([
        _pair_cos_mean(B_, cnt, S), _pair_cos_mean(B_, cnt, F), _pair_cos_mean(B_, cnt, G),
        cmean, cmin, cstd, cnt, np.log1p(cnt), degm, degsd / (degm + 1e-8),
        _centroid_absmean(B_, cnt, S), _centroid_absmean(B_, cnt, F),
        _centroid_absmean(B_, cnt, G),
    ])
    return desc[valid], valid.astype(np.int64)


def _rank01(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(x); r[order] = np.arange(x.size, dtype=np.float64)
    return np.ones_like(x) * 0.5 if x.size == 1 else r / (x.size - 1)


def edge_aug_targets(H, X, R, seed, rate, alpha, K, HB=None):
    """How stable each hyperedge's members are under feature and membership
    masking, minus a size-matched random baseline, rank-normalised to [0,1]."""
    X = np.asarray(X, dtype=np.float64)
    XA = _mask_features(X, seed + 104729, rate)
    if HB is None:
        HB = _mask_memberships(H, seed + 13007, rate)
    Rb = _R(HB, np.asarray(HB.sum(0)).ravel(), np.asarray(HB.sum(1)).ravel())
    A0, AA, AB = _scales(R, X, alpha, K), _scales(R, XA, alpha, K), _scales(Rb, X, alpha, K)
    Cfa = np.stack([_cos(A0[k], AA[k]) for k in range(3)], axis=1)
    Cmb = np.stack([_cos(A0[k], AB[k]) for k in range(3)], axis=1)
    node_stab = np.clip((Cfa.mean(1) + Cmb.mean(1)) * 0.25, -1, 1)
    node_cross = np.clip(np.stack([_cos(AA[k], AB[k]) for k in range(3)], axis=1).mean(1), -1, 1)

    Hc = _csc(H)
    cnt = np.diff(Hc.indptr)
    valid = np.flatnonzero(cnt >= 2)
    B_ = sp.csc_matrix((np.ones_like(Hc.data, dtype=np.float64), Hc.indices, Hc.indptr),
                       shape=Hc.shape)
    local, disp = _edge_moments(B_, cnt.astype(float), node_stab)
    cross, _ = _edge_moments(B_, cnt.astype(float), node_cross)
    raw = (0.65 * local + 0.35 * cross - 0.20 * disp)[valid]

    n = X.shape[0]
    rng = np.random.default_rng(seed + 271828)
    null = np.empty(len(valid))
    for t, size in enumerate(cnt[valid]):
        vals = []
        for _ in range(3):
            idx = rng.choice(n, size=min(int(size), n), replace=False)
            vals.append(0.65 * float(np.mean(node_stab[idx]))
                        + 0.35 * float(np.mean(node_cross[idx])))
        null[t] = float(np.mean(vals))
    return np.clip(_rank01(raw - null), 0.0, 1.0)


def _fit_ridge(Z, y, epochs=160, lr=0.03, l2=5e-2, seed=0):
    if len(y) == 0:
        return np.zeros(0, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    X = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-8)
    X = np.c_[np.ones((len(X), 1)), X]
    w = np.zeros(X.shape[1], dtype=np.float64)
    rng = np.random.default_rng(seed)
    w[1:] = 0.005 * rng.standard_normal(X.shape[1] - 1)
    for _ in range(max(1, epochs)):
        grad = X.T @ (X @ w - y) / max(len(y), 1)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return np.clip(X @ w, 0.0, 1.0)


def project_weights(raw, wmin, wmax, iters=50):
    """w in [wmin, wmax] with mean(w) = 1."""
    w = np.asarray(raw, dtype=np.float64)
    w = w / max(float(w.mean()), 1e-12)
    for _ in range(iters):
        w = np.clip(w, wmin, wmax)
        free = (w > wmin) & (w < wmax)
        gap = len(w) - float(w.sum())
        if abs(gap) < 1e-9 * len(w) or not free.any():
            break
        w[free] += gap / free.sum()
    return np.clip(w, wmin, wmax)


def utility_weights(desc, targets, cfg, seed):
    info = {}
    n = len(targets)
    src = str(cfg["utility"])
    if n == 0 or src == "uniform":
        return np.ones(n), info
    if src == "target":
        pred = np.asarray(targets, dtype=np.float64)
    elif src == "ridge":
        pred = _fit_ridge(desc, targets, epochs=int(cfg["utility_epochs"]),
                          lr=float(cfg["utility_lr"]), l2=float(cfg["utility_l2"]), seed=seed)
        info["ridge_r2"] = 1.0 - float(np.mean((pred - targets) ** 2)) / max(float(np.var(targets)), 1e-12)
    else:
        raise ValueError(f"unknown utility {src!r}")
    z = (pred - pred.mean()) / (pred.std() + 1e-8)
    raw = np.exp(np.clip(float(cfg["utility_temp"]) * z, -2.0, 2.0))
    wmin = float(np.clip(cfg["weight_min"], 0.1, 0.99))
    wmax = float(np.clip(cfg["weight_max"], 1.01, 3.0))
    w = project_weights(raw, wmin, wmax)
    info["frac_at_bounds"] = float(np.mean((w <= wmin + 1e-9) | (w >= wmax - 1e-9)))
    return w, info

# =========================================================================== #
# stages A + B
# =========================================================================== #
def stage_ab(hyperedges, X, num_nodes=None, seed=0, cfg=None, logger=LOG):
    """Returns S, F, G, the utility weights w, the incidence H and the
    row-normalised centred features X (dense)."""
    c = dict(DEFAULTS); c.update(cfg or {})
    sd = int(seed)
    N = int(num_nodes if num_nodes is not None else (max(max(e) for e in hyperedges if len(e)) + 1))
    dim, ds, df, dg = _dims(c)
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = np.array(X, dtype=np.float64, copy=True)      # never mutate the caller's array
    rate = float(np.clip(c["mask_rate"], 0, 0.9))
    alpha = float(np.clip(c["lazy_alpha"], 0, 1))
    K = int(c["ssgc_hops"])
    tm = {}

    # ---- A: structure ----
    t0 = time.perf_counter()
    H, es, nd = _incidence(hyperedges, N)
    S = modularity_ascent(banerjee_modularity_operator(H), dim=max(1, min(ds, N - 1)),
                          iters=int(c["ascent_iters"]), seed=sd, eta=c["ascent_eta"],
                          tol=c["ascent_tol"], patience=int(c["ascent_patience"]),
                          check_every=int(c["ascent_check_every"]),
                          min_iters=int(c["ascent_min_iters"]), logger=logger)
    tm["structure_s"] = time.perf_counter() - t0

    # ---- B: features ----
    t0 = time.perf_counter()
    R = _R(H, es, nd)
    if bool(c["normalize_raw_features"]):
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    if bool(c["center_residual"]):
        X -= X.mean(0, keepdims=True)
    beta = float(np.clip(c["beta"], 0, 1))
    Xres = X - beta * (S @ (S.T @ X)) if (S.shape[1] and beta > 0) else X
    XA = _mask_features(Xres, sd + 104729, rate)
    HB = _mask_memberships(H, sd + 13007, rate)
    RB = _R(HB, np.asarray(HB.sum(0)).ravel(), np.asarray(HB.sum(1)).ravel())
    sa = _scales(R, XA, alpha, K)
    sb = _scales(RB, Xres, alpha, K)
    Wg, Cg = _weights(sa, sb, float(c["temperature"]),
                      float(np.clip(c["min_scale_weight"], 0, 1 / 3)))
    Fcat = np.concatenate([Wg[:, k:k + 1] * sa[k] for k in range(3)], axis=1)
    q = min(df, N - 1, Fcat.shape[1] - 1)             # rank-limited -> narrower, never padded
    if q > 0:
        U, s = _topk_svd(Fcat, q, sd + 7919)
        F = _sign_fix(np.asarray(U)) * s[None, :]
        F *= np.sqrt(F.shape[1]) / (np.linalg.norm(F) + 1e-12)
    else:
        F = np.zeros((N, 0))
    F = F.astype(np.float32)
    G = _incidence_svd(R, dg, sd + 15485863, logger=logger)
    tm["features_s"] = time.perf_counter() - t0

    # ---- B: Eureka utility ----
    t0 = time.perf_counter()
    desc, valid = hyperedge_descriptors(H, S, F, G, Cg, nd)
    targets = (np.zeros(len(valid)) if str(c["utility"]) == "uniform"
               else edge_aug_targets(H, X, R, sd, rate, alpha, K, HB=HB))
    u_valid, uinfo = utility_weights(desc, targets, c, sd + 314159)
    w = np.ones(H.shape[1]); w[valid] = u_valid
    tm["utility_s"] = time.perf_counter() - t0
    logger.info(f"  [hyperfuse] S={S.shape[1]} F={F.shape[1]} G={G.shape[1]} | utility {c['utility']} "
                f"min={w.min():.3f} max={w.max():.3f} "
                + " ".join(f"{k}={v:.3f}" for k, v in uinfo.items()))
    return dict(S=S, F=F, G=G, w=w, H=H, X=X, nd=nd, timings=tm, cfg=c, N=N)

# =========================================================================== #
# stage C: hypergraph CCA self-supervision
# =========================================================================== #
def _torch():
    import torch
    return torch


def _csr(crow, col, val, shape):
    torch = _torch()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # "sparse CSR support is in beta"
        return torch.sparse_csr_tensor(crow, col, val, shape)


def _make_spmm():
    torch = _torch()

    class SpMM(torch.autograd.Function):
        """y = A @ x for a constant sparse A; the backward uses the stored A^T,
        so only sparse-times-dense forward kernels are ever needed."""

        @staticmethod
        def forward(ctx, x, A, At):
            ctx.At = At
            return A @ x

        @staticmethod
        def backward(ctx, g):
            return ctx.At @ g, None, None

    return SpMM.apply


class _Incidence:
    """Node-hyperedge structure on the training device.  A view changes only
    the operator *values*, so the two CSR index sets are built once."""

    def __init__(self, H, w, device):
        torch = _torch()
        Hc = sp.csc_matrix(H, copy=True); Hc.sort_indices()
        self.N, self.E = Hc.shape
        node = Hc.indices.astype(np.int64)
        edge = np.repeat(np.arange(self.E, dtype=np.int64), np.diff(Hc.indptr))
        perm = np.argsort(node, kind="stable")
        t = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        self.dev, self.nnz = device, len(node)
        self.node, self.edge, self.perm = t(node), t(edge), t(perm)
        self.crowE, self.colE = t(Hc.indptr.astype(np.int64)), t(node)          # E x N
        self.crowN = t(np.r_[0, np.cumsum(np.bincount(node, minlength=self.N))])
        self.colN = t(edge[perm])                                              # N x E
        self.w = t(w, torch.float32)

    def operator(self, keep=None):
        """m = E2V(V2E x): V2E = mean over (kept) members, E2V = utility-
        weighted mean over (kept) incident hyperedges."""
        torch = _torch()
        k = torch.ones(self.nnz, device=self.dev) if keep is None else keep.float()
        size = torch.zeros(self.E, device=self.dev).index_add(0, self.edge, k)
        v2e = k / size.clamp_min(1.0)[self.edge]
        num = k * self.w[self.edge]
        den = torch.zeros(self.N, device=self.dev).index_add(0, self.node, num)
        e2v = num / den.clamp_min(1e-12)[self.node]
        return (_csr(self.crowE, self.colE, v2e, (self.E, self.N)),
                _csr(self.crowN, self.colN, v2e[self.perm], (self.N, self.E)),
                _csr(self.crowN, self.colN, e2v[self.perm], (self.N, self.E)),
                _csr(self.crowE, self.colE, e2v, (self.E, self.N)))

    def masked_operator(self, rate, gen):
        torch = _torch()
        return self.operator(torch.rand(self.nnz, generator=gen, device=self.dev) >= rate)


class _Features:
    """Encoder input: sparse CSR (with its transpose, for the weight gradient)
    when the features are sparse, else a dense tensor."""

    def __init__(self, X, device, threshold, spmm):
        torch = _torch()
        X = sp.csr_matrix(X, dtype=np.float32); X.eliminate_zeros(); X.sort_indices()
        self.shape, self.spmm = X.shape, spmm
        self.sparse = X.nnz < float(threshold) * max(X.shape[0] * X.shape[1], 1)
        t = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        if self.sparse:
            order = sp.csr_matrix((np.arange(1, X.nnz + 1, dtype=np.float64), X.indices, X.indptr),
                                  shape=X.shape).T.tocsr()
            order.sort_indices()
            self.perm = t(order.data.astype(np.int64) - 1)        # X order -> X^T order
            self.crow, self.col = t(X.indptr.astype(np.int64)), t(X.indices.astype(np.int64))
            self.crowT, self.colT = t(order.indptr.astype(np.int64)), t(order.indices.astype(np.int64))
            self.val = t(X.data, torch.float32)
        else:
            self.X = t(X.toarray(), torch.float32)

    def matmul(self, W, drop=0.0, gen=None):
        torch = _torch()
        if self.sparse:
            v = self.val
            if drop > 0:
                v = v * (torch.rand(v.shape, generator=gen, device=v.device) >= drop).float()
            A = _csr(self.crow, self.col, v, self.shape)
            At = _csr(self.crowT, self.colT, v[self.perm], (self.shape[1], self.shape[0]))
            return self.spmm(W, A, At)
        X = self.X
        if drop > 0:
            X = X * (torch.rand(X.shape, generator=gen, device=X.device) >= drop).float()
        return X @ W


def _build_encoder(f_in, d_side, hidden, out_dim, spmm):
    torch = _torch()
    nn = torch.nn

    def agg(x, op):
        V2E, V2Et, E2V, E2Vt = op
        return spmm(spmm(x, V2E, V2Et), E2V, E2Vt)

    class HypergraphEncoder(nn.Module):
        """Two layers of  self(x) + E2V(V2E(neighbour(x))).  Each layer's
        self/neighbour weights are fused into one matrix; the hyperedge
        aggregation is applied after the linear map (the two commute)."""

        def __init__(self):
            super().__init__()
            self.W1 = nn.Parameter(torch.empty(f_in, 2 * hidden)); nn.init.xavier_uniform_(self.W1)
            self.b1 = nn.Parameter(torch.zeros(2 * hidden))
            self.side = nn.Linear(d_side, 2 * hidden, bias=False) if d_side > 0 else None
            self.ln = nn.LayerNorm(hidden)
            self.W2 = nn.Linear(hidden, 2 * out_dim)
            self.h, self.o = hidden, out_dim

        def forward(self, feats, side, op, drop=0.0, gen=None):
            a = feats.matmul(self.W1, drop, gen) + self.b1
            if self.side is not None:
                s = side if drop <= 0 else side * (
                    torch.rand(side.shape, generator=gen, device=side.device) >= drop).float()
                a = a + self.side(s)
            h = torch.nn.functional.gelu(self.ln(a[:, :self.h] + agg(a[:, self.h:], op)))
            b = self.W2(h)
            return b[:, :self.o] + agg(b[:, self.o:], op)

    return HypergraphEncoder(), agg


def train_cca(X_in, side, H, w, cfg, device_name, seed, logger=LOG, timings=None):
    """Stage C.  Returns (Z [N, emb_dim] numpy, device_used)."""
    torch = _torch()
    tm = {} if timings is None else timings
    t0 = time.perf_counter()
    want_cuda = str(device_name).lower().startswith("cuda")
    if want_cuda and not torch.cuda.is_available():
        logger.warning("  [hyperfuse-cca] CUDA requested but unavailable; using CPU")
    device = torch.device(device_name if want_cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    gen = torch.Generator(device=device); gen.manual_seed(int(seed) + 112358)

    spmm = _make_spmm()
    inc = _Incidence(H, w, device)
    feats = _Features(X_in, device, cfg["sparse_threshold"], spmm)
    side_t = None if side is None or side.shape[1] == 0 else \
        torch.as_tensor(np.asarray(side, np.float32), device=device)
    dim = int(cfg["emb_dim"])
    model, agg = _build_encoder(feats.shape[1], 0 if side_t is None else side_t.shape[1],
                                int(cfg["hidden"]), dim, spmm)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    eye = torch.eye(dim, device=device)
    lam, fdrop, mdrop = float(cfg["lam"]), float(cfg["feature_drop"]), float(cfg["membership_drop"])
    hops, ha = int(cfg["hops"]), float(cfg["hop_alpha"])
    epochs = max(1, int(cfg["epochs"]))

    def encode(op, drop):
        h = model(feats, side_t, op, drop, gen)
        if hops > 0:                                  # parameter-free SSGC-style averaging
            acc, cur = h, h
            for _ in range(hops):
                cur = ha * cur + (1.0 - ha) * agg(cur, op); acc = acc + cur
            h = acc / (hops + 1)
        return h

    def standardise(z):
        z = z - z.mean(0)
        return z / (z.std(0) + 1e-6) / np.sqrt(z.shape[0])

    tm["cca_setup_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    log_every = max(1, epochs // 5)
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        z1 = standardise(encode(inc.masked_operator(mdrop, gen), fdrop))
        z2 = standardise(encode(inc.masked_operator(mdrop, gen), fdrop))
        inv = ((z1 - z2) ** 2).sum()
        dec = ((z1.T @ z1 - eye) ** 2).sum() + ((z2.T @ z2 - eye) ** 2).sum()
        loss = inv + lam * dec
        loss.backward()
        opt.step()
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            logger.info(f"  [hyperfuse-cca] epoch {ep}/{epochs} loss={float(loss):.4f} "
                        f"inv={float(inv):.4f} dec={float(dec):.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    tm["cca_train_s"] = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        Z = encode(inc.operator(), 0.0)
    return Z.cpu().numpy().astype(np.float64), device.type


# =========================================================================== #
# public entry points
# =========================================================================== #
def _row_normalise_sparse(X):
    X = sp.csr_matrix(X, dtype=np.float64)
    nr = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    return sp.diags(1.0 / np.maximum(nr, 1e-12)) @ X


def _resolve_feature_drop(X, cfg, logger):
    """feature_drop='auto'.  Masking entries of sparse or discrete features
    (bag-of-words, binary / categorical attributes) is a mild corruption and
    is what makes the two views differ.  Masking coordinates of a dense
    real-valued descriptor (e.g. CNN / MVCNN features) destroys information,
    so those are left intact -- the same choice TriCL's own configs make for
    NTU2012 and ModelNet40."""
    fd = cfg["feature_drop"]
    if fd != "auto":
        return float(fd)
    Xs = sp.csr_matrix(X); Xs.eliminate_zeros()
    density = Xs.nnz / max(Xs.shape[0] * Xs.shape[1], 1)
    discrete = bool(np.all(np.mod(Xs.data, 1.0) == 0.0))
    dense_real = density >= float(cfg["sparse_threshold"]) and not discrete
    fd = 0.0 if dense_real else 0.3
    logger.info(f"  [hyperfuse] features density={density:.3f} discrete={discrete} "
                f"-> feature_drop={fd:g}")
    return fd


def embed(hyperedges, X, num_nodes=None, seed=0, device="cpu", logger=None,
          timings=None, **overrides):
    """HyperFuse node embeddings: [N, emb_dim] float32."""
    lg = logger or LOG
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    cfg = dict(DEFAULTS); cfg.update(overrides)
    tm = {} if timings is None else timings
    t_all = time.perf_counter()

    base = stage_ab(hyperedges, X, num_nodes=num_nodes, seed=seed, cfg=cfg, logger=lg)
    tm.update(base["timings"])
    N = base["N"]
    if N < 2:
        raise ValueError("HyperFuse needs at least two nodes")

    inp = str(cfg["encoder_input"])
    FG = np.c_[base["F"], base["G"]].astype(np.float64)
    if inp == "x_fg":
        X_in, side = _row_normalise_sparse(X), FG
    elif inp == "x":
        X_in, side = _row_normalise_sparse(X), None
    elif inp == "fg":
        X_in, side = FG, None
    else:
        raise ValueError(f"unknown encoder_input {inp!r}")

    cfg["feature_drop"] = _resolve_feature_drop(X, cfg, lg)
    Z, dev = train_cca(X_in, side, base["H"], base["w"], cfg, device, int(seed), lg, timings=tm)
    Z -= Z.mean(0, keepdims=True)
    if bool(cfg["l2_normalise_output"]):
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    tm["total_s"] = time.perf_counter() - t_all
    lg.info(f"  [hyperfuse] {Z.shape} input={inp} device={dev} | "
            + " ".join(f"{k}={v:.2f}" for k, v in tm.items()))
    return np.asarray(Z, dtype=np.float32)


def feats_hyperfuse(d, meta=None, seed=None, device="cpu", logger=None, **overrides):
    """Old fuse_versions interface: d = {'hyperedges', 'x', 'num_nodes', ...}."""
    return embed(d["hyperedges"], d["x"], num_nodes=int(d["num_nodes"]),
                 seed=0 if seed is None else int(seed), device=device, logger=logger, **overrides)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(0)
    N, C, Fd = 600, 5, 120
    y = rng.integers(0, C, N)
    Xd = (rng.random((N, Fd)) < 0.05) * 1.0 + 0.4 * (rng.normal(size=(C, Fd))[y] > 1.0)
    hes = []
    for _ in range(400):
        c = rng.integers(C)
        pool = np.where(y == c)[0] if rng.random() < 0.8 else np.arange(N)
        hes.append(sorted(rng.choice(pool, int(rng.integers(2, 8)), replace=False).tolist()))
    Z = embed(hes, Xd, num_nodes=N, seed=0, device="cpu", epochs=30)
    print("embedding:", Z.shape, "row norms ~", float(np.linalg.norm(Z, axis=1).mean()))
