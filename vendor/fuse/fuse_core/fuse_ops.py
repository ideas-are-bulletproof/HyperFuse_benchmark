"""
fuse_core/fuse_ops.py
=====================

Shared numpy/scipy building blocks for the feature-weighted-modularity FUSE
family (fuse_xmod, fuse_xmod_deep, ...).

FIXES vs the original version (see accompanying discussion):

  1. ACCURACY BUG (weight_edges / xmod_operator):
     The Gram term x_i . x_j was being computed on column-standardized
     (z-scored) features. For sparse binary features (bag-of-words style),
     z-scoring inflates rare columns to huge magnitudes, so two nodes that
     happen to share one rare feature get an outlier-dominated edge weight
     that swamps genuine structural/feature signal. Fix: use L2 row-
     normalized features (bounded cosine-similarity-style dot products in
     [-1, 1]) for the Gram term instead of column-standardized features.

  2. MEMORY/PERFORMANCE BUG (xmod_operator):
     The Gram term for the null-model matrix P was computed via a single
     dense advanced-index gather `X[Pco.row], X[Pco.col]` of shape
     (nnz(P), F). On graphs with heavy-tailed/hub degree distributions,
     nnz(P) can be tens of times larger than the real edge count, and
     combined with high feature dimensionality this allocates multi-GB
     dense arrays in one shot (observed to OOM / thrash badly). Fix: compute
     both Gram terms in bounded-size chunks so peak memory is independent of
     nnz(P) and nnz(A).

  3. STABILITY BUG (modularity_ascent):
     The auto step size eta = 1 / (mean |M_ij|) has no relationship to M's
     actual spectral extremes. Because M = Aw - Pw is indefinite, the QR
     power iteration Z <- QR(Z + eta*M*Z) only converges to the top-k
     *algebraically largest* eigenspace of M if eta is small enough that
     |1 + eta*lambda_max| stays larger than |1 + eta*lambda_min|. The old
     heuristic gave no such guarantee (verified: it was frequently a near
     miss). Fix: estimate lambda_max and lambda_min of M cheaply (via a
     few-iteration Lanczos / eigsh call, capped for large graphs) and pick a
     step size that provably preserves the correct eigenvalue ordering.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fuse_core import base as B


# --------------------------------------------------------------------------- #
# tiny helpers
# --------------------------------------------------------------------------- #
def config_get(key, default):
    """B.CONFIG.get(key, default) when CONFIG is a dict, else default."""
    return B.CONFIG.get(key, default) if isinstance(B.CONFIG, dict) else default


def require_feats(d) -> np.ndarray:
    """Return node features as a float numpy array, or raise if the dataset has
    none (every dataset in this suite ships features, so this is a guard)."""
    x = d.get("x", None)
    if x is None:
        raise ValueError(
            f"Dataset {d.get('name','?')!r} has no node features, but this FUSE "
            "version requires them."
        )
    try:
        return x.detach().cpu().numpy().astype(np.float64)
    except AttributeError:
        return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# chunked Gram helper (fix #2: bounded memory regardless of nnz)
# --------------------------------------------------------------------------- #
def _chunked_row_gram(X: np.ndarray, row: np.ndarray, col: np.ndarray,
                       max_cells: int = 5_000_000) -> np.ndarray:
    """Compute [x_row[k] . x_col[k] for k in range(len(row))] without ever
    materializing a full (len(row), F) dense gather.

    max_cells bounds peak memory: chunk size = max(1, max_cells // F), so a
    graph with a huge nnz(P) and/or a high-dimensional feature matrix costs
    the same *peak* memory as a small one -- only wall-clock time grows.
    """
    n = len(row)
    F = X.shape[1] if X.ndim == 2 else 1
    chunk = max(1, max_cells // max(F, 1))
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        Xr = X[row[start:end]]
        Xc = X[col[start:end]]
        out[start:end] = np.einsum("ij,ij->i", Xr, Xc)
    return out


# --------------------------------------------------------------------------- #
# (b) feature-weighted edges that KEEP the adjacency signal
# --------------------------------------------------------------------------- #
def weight_edges(A: sp.spmatrix, X: np.ndarray) -> sp.csr_matrix:
    """Sparse copy of A whose value on edge (i, j) is  x_i . x_j + A_ij.

    A_ij is 1 on the support (A is binary), so the weight is (x_i . x_j) + 1 --
    the feature Gram PLUS the adjacency unit that was previously thrown away.

    X should be a *bounded* per-node feature representation (e.g. L2
    row-normalized -> cosine-similarity-style dot products in [-1, 1]); see
    xmod_operator, which is responsible for handing this function features
    prepared that way rather than raw z-scored features.
    """
    Aco = A.tocoo()
    gram = _chunked_row_gram(X, Aco.row, Aco.col)
    w = gram + Aco.data            # + A_ij   (adjacency signal retained)
    return sp.csr_matrix((w, (Aco.row, Aco.col)), shape=A.shape)


def config_null_P(A: sp.spmatrix, K: int, seed) -> sp.csr_matrix:
    """Average binary adjacency over K degree-preserving random graphs (stub
    matching). Sparse (<= K*m nonzeros); values in (0, 1] = fraction of the K
    random views that contained each edge, i.e. the config-model edge
    probability estimate."""
    N = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel().astype(np.int64)
    stubs = np.repeat(np.arange(N), deg)          # 2m stubs
    rng = np.random.default_rng(seed)
    P = sp.csr_matrix((N, N))
    for _ in range(K):
        s = stubs.copy()
        rng.shuffle(s)
        src, dst = s[0::2], s[1::2]
        keep = src != dst                          # drop self-loops
        src, dst = src[keep], dst[keep]
        r = np.concatenate([src, dst])
        c = np.concatenate([dst, src])
        Rk = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(N, N))
        Rk.data[:] = 1.0                           # binary (dedup multi-edges)
        P = P + Rk
    P = P.multiply(1.0 / K).tocsr()
    P.setdiag(0)
    P.eliminate_zeros()
    return P


def xmod_operator(d, X: np.ndarray, seed, K: int | None = None) -> sp.csr_matrix:
    """Feature-weighted, adjacency-preserving modularity operator

        M = Aw - Pw

    where, on real edges,   Aw_ij = x_i . x_j + A_ij,
    and on config-null edges Pw_ij = P_ij * (x_i . x_j + 1)   (same weight, in
    expectation under the degree-preserving null). Symmetric sparse.

    NOTE (fix #1): X is expected to already be a *bounded* per-node
    representation. Callers should pass L2 row-normalized features
    (B.l2_normalize_rows), not column-standardized (z-scored) features --
    z-scoring sparse/rare features before a Gram product produces outlier
    dot products that dominate M and destabilize the whole operator.
    """
    if K is None:
        K = config_get("XMOD_VIEWS", 20)
    A = B.scipy_adj(d)
    N = A.shape[0]

    Aw = weight_edges(A, X)                          # x-weighted real edges (+A_ij)
    P = config_null_P(A, K, seed)                    # degree-preserving null (avg of K)

    Pco = P.tocoo()
    # expected weight on a null edge = P_ij * (x_i.x_j + 1); the "+1" mirrors the
    # adjacency unit added to the real edges above, so M stays a clean
    # actual-minus-expected modularity of the SAME weighting.
    # Computed in bounded-size chunks (fix #2) so peak memory doesn't scale
    # with nnz(P) * F, which can otherwise reach many GB on hub-heavy graphs
    # with high-dimensional features.
    gram_null = _chunked_row_gram(X, Pco.row, Pco.col)
    pw = Pco.data * (gram_null + 1.0)
    Pw = sp.csr_matrix((pw, (Pco.row, Pco.col)), shape=(N, N))

    return (Aw - Pw).tocsr()


# --------------------------------------------------------------------------- #
# strength-based (weighted-degree) modularity null -- generalises the
# classical d_i d_j / 2m null to a WEIGHTED graph using Aw's own strengths,
# implemented as an implicit LinearOperator (no N x N matrix, no Monte-Carlo
# sampling step at all -- replaces config_null_P + xmod_operator's approach
# for callers that want this instead).
# --------------------------------------------------------------------------- #
def strength_null_operator(Aw: sp.spmatrix) -> spla.LinearOperator:
    """Implicit (never-materialized) weighted-modularity operator

        M = Aw - s s^T / (2W)

    where s_i = Aw's weighted degree ("strength", row sum) and W = sum(s)/2
    is the total edge weight -- the direct weighted generalisation of the
    classical modularity null d_i d_j / 2m (Newman), computed from Aw's OWN
    weights. (Mixing a probability-scaled classical null d_i d_j/2m with a
    weight-scaled Aw is not dimensionally consistent; this avoids that.)

    Implemented as a scipy LinearOperator computing M @ Z directly as
    Aw @ Z - s[:, None] * (s @ Z) / (2W), WITHOUT ever forming the null term
    as an N x N matrix (dense or sparse). Unlike config_null_P (which draws
    K random degree-preserving rewirings and averages them), there is no
    sampling step here at all -- this is both more principled (it's the
    exact weighted-modularity null, not a Monte-Carlo estimate of one) and
    strictly cheaper (O(nnz(Aw)*dim + N*dim) per multiply vs O(nnz(P)*dim)
    with nnz(P) up to K*m, plus no K-rewiring construction cost up front).

    Drop-in wherever xmod_operator's sparse M was used: safe_ascent_eta and
    modularity_ascent both only ever call M's matvec (M @ v / M @ Z), which
    LinearOperator provides identically to a materialized sparse matrix.
    """
    N = Aw.shape[0]
    s = np.asarray(Aw.sum(axis=1)).ravel()
    Wtot = max(float(s.sum()) / 2.0, 1e-8)

    def _apply(Z):
        Z = np.asarray(Z)
        one_d = (Z.ndim == 1)
        if one_d:
            Z = Z[:, None]
        out = Aw @ Z - s[:, None] * (s @ Z)[None, :] / (2.0 * Wtot)
        return out[:, 0] if one_d else out

    return spla.LinearOperator(shape=(N, N), dtype=np.float64,
                                matvec=_apply, rmatvec=_apply, matmat=_apply)


def blend_operators(M_A: spla.LinearOperator, M_S: spla.LinearOperator,
                     alpha: float) -> spla.LinearOperator:
    """Combine two modularity operators as (1-alpha)*M_A + alpha*M_S,
    evaluated by calling each operator's own matvec/matmat and summing the
    results -- NOT by blending the underlying adjacency matrices first.

    This matters: M_A and M_S are each already internally-consistent
    "signal minus its own correctly-computed null" objects. Blending the
    raw adjacency matrices before computing a null (e.g.
    A_blend = (1-a)*A + a*S) produces fractional "half-edges" with no
    structural meaning wherever A and S disagree, AND corrupts the null
    model itself (the resulting degree/strength sequence doesn't
    correspond to any real graph). Blending the two OPERATORS instead
    keeps each null model coherent and combines two valid modularity
    signals, not two incoherent graphs.

    Generic over any two same-shape LinearOperators (or sparse matrices,
    which support @ the same way), not specific to strength_null_operator.
    """
    if M_A.shape != M_S.shape:
        raise ValueError(f"shape mismatch: M_A {M_A.shape} vs M_S {M_S.shape}")
    N = M_A.shape[0]

    def _apply(Z):
        return (1.0 - alpha) * (M_A @ Z) + alpha * (M_S @ Z)

    return spla.LinearOperator(shape=(N, N), dtype=np.float64,
                                matvec=_apply, rmatvec=_apply, matmat=_apply)


def graph_overlap_alpha(A: sp.spmatrix, S: sp.spmatrix,
                         alpha_min: float = 0.0, alpha_max: float = 1.0) -> float:
    """Unsupervised, label-free blend weight for combining a real graph A
    and a reconstructed graph S:

        confirmation = |A ∩ S| / |A|            (fraction of A's edges S confirms)
        p_S          = |S| / (N*(N-1))           (S's own base density)
        lift         = confirmation / p_S        (enrichment over chance)
        alpha        = 1 / (1 + lift)

    SECOND CORRECTION -- the previous version (alpha = 1 - confirmation,
    with no density normalization) was empirically STILL confounded, just
    in the opposite direction from the original Jaccard version: it
    depends on S's own absolute density, which varies enormously by
    dataset. Verified on a real run: texas/cornell/wisconsin have TINY
    real graphs (N~183-251) but S built at ~26-35% density -- so dense
    that even a random, meaningless S would "confirm" roughly a third of
    A's edges by pure chance, and that's almost exactly what was observed
    (confirmation ~0.32-0.33, chance-level density ~0.26-0.35 --
    essentially NO signal above chance). Meanwhile chameleon/squirrel have
    SPARSE S (~0.5-0.7% density), so their much smaller-looking raw
    confirmation (~6%) is actually 9-12x what chance would predict at that
    density -- a far stronger genuine signal than texas's apparent 33%.
    Raw confirmation rate conflated "S happens to be very dense" with
    "structure and features genuinely agree," exactly inverting the
    intended alpha ordering on a real run (chameleon/squirrel got the
    HIGHEST alpha of all nine datasets, precisely where a LOW alpha was
    needed).

    Normalizing by S's own density (a lift / enrichment score) fixes this:
    a completely random S produces lift ~= 1 regardless of density or
    graph size (alpha ~= 0.5, verified on synthetic tests spanning both
    the texas-like dense-S and chameleon-like sparse-S regimes), so alpha
    only moves away from 0.5 when there's a genuine, non-chance
    relationship between A's edges and S's edges. lift > 1 (structure
    confirmed well above chance -> homophilic-leaning) pulls alpha below
    0.5, toward A. lift <= 1 (no better than chance, or S disagrees with
    A) pushes alpha toward/above 0.5, toward S.

    Still entirely label-free, still never selected from downstream
    accuracy.

    Degenerate cases: an empty S confirms nothing by construction, but
    that means "no reconstruction signal available" rather than "totally
    untrustworthy structure" -- forced to alpha=0 (trust A only). An empty
    A (pathological) forces alpha=1 by the same logic in reverse.
    """
    A_bin = (A != 0)
    S_bin = (S != 0)
    a_nnz = A_bin.nnz
    s_nnz = S_bin.nnz
    n = A.shape[0]
    if a_nnz == 0 or s_nnz == 0:
        alpha = 0.0 if s_nnz == 0 else 1.0
        return float(np.clip(alpha, alpha_min, alpha_max))

    inter = A_bin.multiply(S_bin).nnz
    confirmation = inter / a_nnz          # fraction of A's OWN edges also present in S
    p_S = max(s_nnz / (n * (n - 1)), 1e-12)  # S's own base density (chance-level confirmation rate)
    lift = confirmation / p_S             # enrichment over chance
    alpha = 1.0 / (1.0 + lift)
    return float(np.clip(alpha, alpha_min, alpha_max))


def graph_overlap_alpha_jaccard(A: sp.spmatrix, S: sp.spmatrix,
                                 alpha_min: float = 0.0, alpha_max: float = 1.0) -> float:
    """ORIGINAL overlap-based alpha, superseded by graph_overlap_alpha (the
    "lift" method) above -- kept here, still working, for deliberately
    re-running the ORIGINAL adaptive-alpha behaviour against the current
    fixed pipeline (chunked Gram / spectrally-safe eta / de-duplicated
    chameleon_filtered & squirrel_filtered), which it was never actually
    tested against before being superseded.

        confirmation = |A ∩ S| / |A|      (fraction of A's own edges S confirms)
        alpha        = 1 - confirmation

    NOTE ON THE NAME: despite the "_jaccard" suffix (inherited from the
    config key that has always selected this path, XMOD_STRENGTH_ADAPTIVE_
    ALPHA_METHOD="jaccard"), this is NOT the naive union-normalized Jaccard
    index 1 - |A∩S|/|A∪S|. That truly-Jaccard version was tried FIRST and
    found broken: because S (AMLP's cosine-threshold reconstruction) can be
    far denser than the real graph A, |A∪S| is dominated by S's own edge
    count regardless of whether A's edges are actually confirmed, crushing
    alpha toward the same narrow ~0.8-1.0 band on every dataset (verified:
    cora at homophily~0.81 and cornell at homophily~0.12 came out at 0.925
    and 0.982 -- almost no daylight). This function is the fix that
    followed: normalizing by |A| instead of |A∪S| removes S's own density
    from the computation entirely, so alpha responds to whether A's edges
    are confirmed, not to how many edges S happens to have.

    THIS in turn is what graph_overlap_alpha's "lift" method above later
    superseded: this formula is still confounded by S's ABSOLUTE density
    (a dense-but-random S can "confirm" a large fraction of A purely by
    chance -- observed on texas/cornell/wisconsin, where S sits at
    26-35% density and produced confirmation~0.32-0.33 despite carrying
    no real signal above chance). "lift" additionally normalizes
    confirmation by S's own base density (p_S) to correct for that. This
    function does NOT include that correction -- that is precisely the
    difference being tested by comparing the two methods.

    Degenerate cases identical to graph_overlap_alpha: empty S -> alpha=0
    (trust A only); empty A -> alpha=1 (trust S only).
    """
    A_bin = (A != 0)
    S_bin = (S != 0)
    a_nnz = A_bin.nnz
    s_nnz = S_bin.nnz
    if a_nnz == 0 or s_nnz == 0:
        alpha = 0.0 if s_nnz == 0 else 1.0
        return float(np.clip(alpha, alpha_min, alpha_max))

    inter = A_bin.multiply(S_bin).nnz
    confirmation = inter / a_nnz          # fraction of A's OWN edges also present in S
    alpha = 1.0 - confirmation
    return float(np.clip(alpha, alpha_min, alpha_max))


# --------------------------------------------------------------------------- #
# spectrally-aware step size (fix #3)
# --------------------------------------------------------------------------- #
def safe_ascent_eta(M, safety: float = 0.9,
                     max_dim_for_exact: int = 20_000) -> float:
    """Pick a step size for Z <- QR(Z + eta*M*Z) that provably keeps the
    algebraically-largest eigenspace of (I + eta*M) the dominant one in
    magnitude -- i.e. gradient ascent actually converges toward maximizing
    tr(Zt M Z), not toward the most-negative eigenspace.

    M is indefinite (M = actual - expected). Orthogonal/subspace iteration on
    (I + eta*M) converges to the top-k eigenspace ranked by |1 + eta*lambda|.
    That only matches "top-k most positive lambda" if

        (1 + eta*lambda_max) > |1 + eta*lambda_min|
        <=>  eta < 2 / (lambda_max + |lambda_min|)      (when lambda_min < 0)

    We estimate lambda_max and lambda_min via a cheap sparse eigensolve
    (ARPACK, k=1 each) and pick eta at `safety` fraction of that bound. For
    very large graphs where even a k=1 eigsh call is unwelcome, fall back to
    a bounded power-iteration estimate of the spectral radius, which gives a
    (looser but still safe) bound via eta < 1 / ||M||_2.

    M may be a scipy sparse matrix OR a scipy.sparse.linalg.LinearOperator
    (e.g. a rank-1-corrected operator that's never materialized as a dense
    or sparse N x N matrix -- see strength_modularity_ascent). eigsh and the
    power-iteration fallback both only ever call M's matvec (`M @ v`), so
    either input type works identically; the only type-specific step is
    ``.tocsr()``, which is skipped for a LinearOperator.
    """
    N = M.shape[0]
    if sp.issparse(M):
        M = M.tocsr()

    try:
        if N <= max_dim_for_exact:
            lam_max = float(spla.eigsh(M, k=1, which="LA", return_eigenvectors=False)[0])
            lam_min = float(spla.eigsh(M, k=1, which="SA", return_eigenvectors=False)[0])
        else:
            raise RuntimeError("graph too large for exact extremal eigsh; use power-iteration bound")
    except Exception:
        # Fallback: bounded power iteration to estimate the spectral radius
        # ||M||_2, which safely bounds both |lambda_max| and |lambda_min|.
        rng = np.random.default_rng(0)
        v = rng.normal(size=N)
        v /= np.linalg.norm(v) + 1e-12
        spectral_radius = 0.0
        for _ in range(50):
            v = M @ v
            nrm = np.linalg.norm(v)
            if nrm < 1e-12:
                break
            v /= nrm
            spectral_radius = nrm
        lam_max = spectral_radius
        lam_min = -spectral_radius

    denom = lam_max + abs(min(lam_min, 0.0))
    if denom <= 1e-9:
        return 1.0  # M ~ 0; step size is irrelevant
    eta_bound = 2.0 / denom
    return safety * eta_bound


# --------------------------------------------------------------------------- #
# (a) gradient-ascent modularity maximiser (replaces the spectral solve)
# --------------------------------------------------------------------------- #
def modularity_ascent(M: sp.spmatrix, dim: int, iters: int, seed,
                      eta: float | None = None,
                      feat_graph: sp.spmatrix | None = None,
                      feat_lambda: float = 0.0,
                      tag: str = "fuse_xmod",
                      tol: float | None = None, patience: int = 5,
                      check_every: int = 5, min_iters: int = 20) -> np.ndarray:
    """max tr(Zt M Z) s.t. Zt Z = I via projected gradient ascent + QR.

        Z <- QR( Z + eta * ( M Z  [+ feat_lambda * Pf Z] ) )

    M may be indefinite (it is a modularity operator = actual - null), and
    may be a scipy sparse matrix OR a LinearOperator (e.g.
    strength_null_operator's implicit form).

    eta: if None, uses safe_ascent_eta(M), a spectrally-aware step size that
    is provably small enough to keep gradient ascent converging toward the
    algebraically-largest eigenspace of M (fix #3) -- the previous
    mean-|M|-based heuristic gave no such guarantee.
    feat_graph / feat_lambda: optional feature-graph diffusion term (idea d);
    Pf is row-normalised inside.

    CONVERGENCE-BASED STOPPING (opt-in via tol; default None preserves the
    exact original fixed-iteration-count behaviour for existing callers):
    mirrors fuse_loop's criterion exactly -- track tr(Z^T M Z), read off for
    free as tr(Z^T @ grad) since grad = M @ Z is already computed every
    iteration, stop once its relative change has stayed under tol for
    patience consecutive checks (checked every check_every iterations),
    never before min_iters. M may be indefinite, but tr(Z^T M Z) is still
    exactly the quantity gradient ascent is maximizing here, so the same
    objective-based (not raw-distance-based; see fuse_loop's docstring for
    why) convergence argument applies unchanged.
    """
    N = M.shape[0]
    rng = np.random.default_rng(seed)
    Z, _ = np.linalg.qr(rng.normal(size=(N, dim)))

    if eta is None:
        eta = safe_ascent_eta(M)

    Pf = B.rownorm_P(feat_graph) if (feat_graph is not None and feat_lambda) else None

    prev_obj = None
    quiet = 0

    for t in range(iters):
        grad = M @ Z
        if Pf is not None:
            grad = grad + feat_lambda * (Pf @ Z)

        if tol is not None and (t + 1) % check_every == 0:
            obj = float(np.trace(Z.T @ grad))
            if prev_obj is not None:
                rel_change = abs(obj - prev_obj) / (abs(prev_obj) + 1e-12)
                if rel_change < tol:
                    quiet += 1
                else:
                    quiet = 0
                if quiet >= patience and (t + 1) >= min_iters:
                    B.logger.info(
                        f"  [{tag}] converged at iter {t + 1}/{iters} "
                        f"(rel_change={rel_change:.2e} < tol={tol:g} for "
                        f"{patience} consecutive checks)"
                    )
                    Z, _ = np.linalg.qr(Z + eta * grad)
                    return Z
            prev_obj = obj

        Z, _ = np.linalg.qr(Z + eta * grad)
        if (t + 1) % B.LOG_EVERY == 0 or t == iters - 1:
            B.logger.info(f"  [{tag}] iter {t + 1}/{iters}")

    if tol is not None:
        B.logger.warning(
            f"  [{tag}] hit the {iters}-iteration ceiling without meeting "
            f"the convergence criterion (tol={tol:g}, patience={patience}) "
            "-- the objective may still be improving; consider raising the "
            "ceiling if this method's downstream accuracy looks off."
        )
    return Z

def fuse_loop(A: sp.spmatrix, S0: np.ndarray, eta: float, iters: int,
              feat_graph: sp.spmatrix | None = None, feat_lambda: float = 0.0,
              tag: str = "fuse_fa",
              tol: float | None = 1e-5, patience: int = 5,
              check_every: int = 5, min_iters: int = 20) -> np.ndarray:
    """Newman-modularity power iteration on a PLAIN adjacency A (used by the
    structure-only reference version). Kept for parity with the original FUSE
    loop; the feature-weighted versions use ``modularity_ascent`` on M instead.

    STEP SIZE: ``eta`` is now always a fixed, caller-supplied value -- there
    is no auto/spectral fallback here anymore. config.py's BASIC_ETA is
    1.0e4, chosen after comparing it against both the paper's own Table 2
    value (0.05) and the auto/spectral bound on synthetic graphs at two
    scales (see config.py's CONFIG["BASIC_ETA"] comment for the actual
    numbers and the reasoning); 1e4 is the one kept. Pass a different float
    to override if you have a specific graph-appropriate value and reason
    to; ``eta=None`` is no longer accepted.

    CONVERGENCE-BASED STOPPING (in addition to the ``iters`` ceiling):
    ``iters`` is a safety-net MAXIMUM, not a fixed count. Every
    ``check_every`` iterations we track the modularity objective this power
    iteration is actually maximizing, tr(S^T B S) -- read off for free from
    the already-computed ``grad = B @ S`` as tr(S^T @ grad), no extra
    matmul. We deliberately track the *objective*, not raw
    ||S_t - S_{t-1}||: QR has a column-sign / near-degenerate-eigenvalue
    ambiguity, so two consecutive iterates can differ noticeably
    elementwise even once the SUBSPACE has genuinely converged, which would
    make a raw distance-based check unreliable (too late, or never
    triggering). The objective has no such ambiguity -- it's invariant to
    any orthogonal transformation within the converged subspace.

    Stops once the objective's relative change has been below ``tol`` for
    ``patience`` consecutive checks (guards a single noisy plateau from
    triggering a premature stop) -- but never before ``min_iters``. Pass
    ``tol=None`` to disable early stopping entirely and always run exactly
    ``iters`` iterations (the old, fixed-count behaviour).
    """
    S, _ = np.linalg.qr(S0)
    deg = np.asarray(A.sum(1)).ravel()
    m = max(A.nnz / 2.0, 1.0)
    Pf = B.rownorm_P(feat_graph) if (feat_graph is not None and feat_lambda) else None

    eta = float(eta)

    prev_obj = None
    quiet = 0

    for t in range(iters):
        grad = (A @ S - (deg[:, None] / (2 * m)) * S.sum(0, keepdims=True)) / (2 * m)
        if Pf is not None:
            grad = grad + feat_lambda * (Pf @ S)

        if tol is not None and (t + 1) % check_every == 0:
            obj = float(np.trace(S.T @ grad))
            if prev_obj is not None:
                # DIVERGENCE CHECK: a decrease is a much stronger signal
                # that eta is too large than a merely-slow rel_change --
                # ascent should be monotonically non-decreasing in the
                # objective (up to QR-retraction noise); a real decrease
                # means the step overshot the ascent direction. Distinct
                # from the plateau check below, which looks for the
                # opposite failure mode (step too small / already
                # converged).
                if obj < prev_obj - abs(prev_obj) * tol:
                    B.logger.warning(
                        f"  [{tag}] objective DECREASED at iter {t + 1} "
                        f"({prev_obj:.6g} -> {obj:.6g}) -- eta={eta:.4g} is "
                        "likely too large for this graph; ascent may be "
                        "unstable. Consider a smaller fixed eta."
                    )
                rel_change = abs(obj - prev_obj) / (abs(prev_obj) + 1e-12)
                if rel_change < tol:
                    quiet += 1
                else:
                    quiet = 0
                if quiet >= patience and (t + 1) >= min_iters:
                    B.logger.info(
                        f"  [{tag}] converged at iter {t + 1}/{iters} "
                        f"(rel_change={rel_change:.2e} < tol={tol:g} for "
                        f"{patience} consecutive checks)"
                    )
                    S, _ = np.linalg.qr(S + eta * grad)
                    return S
            prev_obj = obj

        S = S + eta * grad
        S, _ = np.linalg.qr(S)
        if (t + 1) % B.LOG_EVERY == 0 or t == iters - 1:
            B.logger.info(f"  [{tag}] iter {t + 1}/{iters}")

    if tol is not None:
        B.logger.warning(
            f"  [{tag}] hit the {iters}-iteration ceiling without meeting "
            f"the convergence criterion (tol={tol:g}, patience={patience}) "
            "-- the objective may still be improving; consider raising the "
            "ceiling if this method's downstream accuracy looks off."
        )
    return S


# --------------------------------------------------------------------------- #
# Residual feature branch (shared Stage 2-3 for the *-R family, e.g. fuse_r)
#
# This is a SEPARATION-based answer to the same homophily/heterophily
# problem strength_null_operator / blend_operators / graph_overlap_alpha
# above solve by MIXING: those blend two operators (M_A, M_S) so features
# still compete with structure for the same eigendirections, just with a
# smarter (adaptive, label-free) blend weight. residual_features instead
# never lets that competition happen at all -- it solves structure alone
# first (via fuse_loop, whose S already has S^T S = I from its own QR
# retraction), and only keeps the part of X that S provably could not have
# captured. The two approaches are worth comparing empirically rather than
# assuming one dominates; see fuse_versions/fuse_r.py.
# --------------------------------------------------------------------------- #
def split_dims(total: int, struct_dim: int | None, feat_dim: int | None,
               struct_frac: float) -> tuple[int, int]:
    """Resolve (d_z, d_x) so that d_z + d_x == total (EMB_DIM)."""
    if struct_dim is not None and feat_dim is not None:
        return int(struct_dim), int(feat_dim)
    dz = int(round(total * float(struct_frac)))
    dz = max(1, min(dz, total - 1)) if total > 1 else total
    dx = total - dz
    return dz, dx


def _topk_svd(X: np.ndarray, q: int, seed) -> tuple[np.ndarray, np.ndarray]:
    """Top-q left singular vectors * singular values of X, via ARPACK
    (scipy.sparse.linalg.svds). Works on dense or sparse X and stays cheap
    even when X is [N, F] with N in the hundreds of thousands, since q is
    typically << min(N, F) (q = d_x, a slice of EMB_DIM)."""
    n, f = X.shape
    q = max(1, min(q, min(n, f) - 1))
    Xf = X.asfptype() if sp.issparse(X) else X
    U, S, _ = spla.svds(Xf, k=q, random_state=seed)
    order = np.argsort(-S)
    return U[:, order], S[order]


def residual_features(S: np.ndarray, X: np.ndarray, d_x: int, *,
                      alpha: float = 1.0, normalize_raw: bool = True,
                      center: bool = True, seed=0,
                      tag: str = "residual") -> np.ndarray:
    """Stages 2-3 of the *-R family: project X onto S's orthogonal
    complement, PCA-compress the residual, energy-match it to S's
    Frobenius norm, and scale by alpha.

    S must already have (numerically) orthonormal columns, S^T S = I_{d_z}
    -- ``fuse_loop`` above already guarantees this via its final QR
    retraction (whether it exits by hitting ``iters`` or by early
    convergence -- both code paths end in a QR step), so no extra
    orthonormalization happens here.

        X_res = X - S (S^T X)             exact, matrix-free (no N x N matrix);
                                           S^T X_res = 0 EXACTLY because S^T S = I
        H     = PCA_{d_x}(X_res)          top singular directions of the residual
        H    <- H * sqrt(d_z) / ||H||_F   energy-matched to ||S||_F^2 = d_z
        H    <- alpha * H

    Returns H as float32 [N, d_x] (zero-padded if the residual's rank was
    below d_x).
    """
    n, d_z = S.shape
    Xd = np.asarray(X, dtype=np.float64)
    if normalize_raw:
        norm = np.linalg.norm(Xd, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        Xd = Xd / norm
    if center:
        Xd = Xd - Xd.mean(axis=0, keepdims=True)

    X_res = Xd - S @ (S.T @ Xd)              # exact because S^T S = I

    max_rank = max(1, min(X_res.shape) - 1)
    q = int(min(d_x, max_rank))
    U, sv = _topk_svd(X_res, q, seed)
    H = U * sv[None, :]                      # [N, q], variance-aware PCA scores

    h_energy = np.linalg.norm(H) + 1e-12
    H = H * (float(d_z) ** 0.5 / h_energy)   # ||H||_F^2 -> d_z, matching ||S||_F^2
    H = alpha * H

    if q < d_x:
        pad = np.zeros((n, d_x - q), dtype=np.float64)
        H = np.concatenate([H, pad], axis=1)
        B.logger.info(f"    [{tag}] residual rank {q} < d_x {d_x}; zero-padded")

    ortho = float(np.abs(S.T @ X_res).max())
    B.logger.info(f"    [{tag}] feature branch: PCA(X_res) -> H (d_x={d_x}), "
                 f"max|S^T X_res|={ortho:.2e}, alpha={alpha}")

    return H.astype(np.float32)
