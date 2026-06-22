r"""Diagnostic metrics for VFE_3.0 runs (publication-oriented, registry-backed).

A registry of named metrics over beliefs / transport / attention. Each metric reads
what it needs from a keyword context and returns a plain float (or a small dict), so
new probes slot in by ``register_metric`` without editing call sites; ``compute_metrics``
emits a CSV/JSON-friendly record. The metrics are pure measurements (no gradients, no
side effects) suitable for logging a training run or a figure.

Provided: effective_rank (spectral participation ratio), attention_entropy (row entropy
of beta), holonomy_deviation (cocycle/curvature departure of the transport from flat),
gauge_trace_spread (spread of log|det Omega| = tr embed(phi)), and free_energy_terms
(the per-term F decomposition).
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch


def effective_rank(
    spectrum: torch.Tensor,              # (..., K) non-negative spectrum (diagonal variances or eigenvalues)

    *,
    eps:      float = 1e-12,
) -> torch.Tensor:                       # (...) participation-ratio effective rank
    r"""Spectral effective rank (sum lam)^2 / sum lam^2 over the last axis.

    Equals K when the spectrum is flat (all equal) and -> 1 when one mode dominates. For a
    full covariance, pass its eigenvalues (``torch.linalg.eigvalsh(Sigma)``); for a diagonal
    belief, the variances ARE the spectrum.
    """
    lam = spectrum.float().clamp(min=0.0)
    s1 = lam.sum(dim=-1)
    s2 = (lam ** 2).sum(dim=-1).clamp(min=eps)
    return (s1 * s1 / s2).to(spectrum.dtype)


def attention_entropy(
    beta: torch.Tensor,                  # (..., N, N) row-stochastic attention weights

    *,
    eps:  float = 1e-12,
) -> torch.Tensor:                       # () mean row entropy
    r"""Mean Shannon entropy -Σ_j β_ij log β_ij over query rows (log N for uniform β)."""
    h = -(beta.clamp(min=eps) * torch.log(beta.clamp(min=eps))).sum(dim=-1)
    return h.mean()


def holonomy_deviation(
    omega: torch.Tensor,                 # (N, N, K, K) pairwise transport Omega_ij

    *,
    max_triangles: int = 512,
) -> torch.Tensor:                       # () mean ||Omega_ij Omega_jk Omega_ki - I||_F
    r"""Curvature proxy: mean Frobenius departure of the triangle holonomy from identity.

    For a flat (Regime I) cocycle Omega_ij = exp(phi_i)exp(-phi_j) every triangle closes
    (H_ijk = I) so the deviation is ~0; a non-flat / non-cocycle transport gives > 0.
    """
    N, K = omega.shape[0], omega.shape[-1]
    eye = torch.eye(K, device=omega.device, dtype=omega.dtype)

    # Enumerate the first ``max_triangles`` distinct (i, j, k) triples in the same
    # row-major order the nested-loop form used, then evaluate them as ONE batched
    # (T, K, K) matmul rather than T Python-dispatched (K, K) matmuls -- same triangles,
    # same value, one kernel launch instead of T.
    triples: List[Tuple[int, int, int]] = []
    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            for k in range(N):
                if k == i or k == j:
                    continue
                triples.append((i, j, k))
                if len(triples) >= max_triangles:
                    break
            if len(triples) >= max_triangles:
                break
        if len(triples) >= max_triangles:
            break
    if not triples:
        return torch.tensor(0.0, device=omega.device, dtype=omega.dtype)

    idx = torch.tensor(triples, device=omega.device)                      # (T, 3)
    o_ij = omega[idx[:, 0], idx[:, 1]]                                     # (T, K, K)
    o_jk = omega[idx[:, 1], idx[:, 2]]                                     # (T, K, K)
    o_ki = omega[idx[:, 2], idx[:, 0]]                                     # (T, K, K)
    H = o_ij @ o_jk @ o_ki                                                # (T, K, K) holonomy
    return torch.linalg.norm(H - eye, dim=(-2, -1)).mean()


def gauge_trace_spread(
    phi:        torch.Tensor,            # (..., n_gen) gauge-frame coordinates
    generators: torch.Tensor,           # (n_gen, K, K)
) -> torch.Tensor:                       # () std of tr(embed(phi)) = std of log|det exp(embed phi)|
    r"""Spread (std) of log|det Omega| across tokens; 0 at phi = 0 (Omega = I)."""
    traces = generators.diagonal(dim1=-2, dim2=-1).sum(-1)        # (n_gen,) tr(G_a)
    logdet = torch.einsum("...a,a->...", phi, traces)            # tr(embed(phi)) = sum_a phi^a tr(G_a)
    return logdet.flatten().std(unbiased=False)


def free_energy_terms(
    self_div: torch.Tensor,              # (..., N) D(q_i||p_i)
    energy:   torch.Tensor,              # (..., N, N) E_ij
    beta:     torch.Tensor,              # (..., N, N) attention weights
    alpha:    torch.Tensor,              # (..., N) self-coupling

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    lambda_beta:               'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure F)
    eps:                       float = 1e-12,
    include_attention_entropy: bool  = True,                   # gate the entropy term in ``total``

    log_prior:                 Optional[torch.Tensor] = None,  # (..., N, N) attention log-prior
    alpha_reg:                 Optional[torch.Tensor] = None,  # (..., N[,K]) R(alpha_i) if state-dependent
) -> Dict[str, float]:
    r"""Per-term free-energy decomposition: self-coupling, belief-coupling, attention entropy.

    Mirrors ``vfe3.free_energy.free_energy`` so the reported ``total`` equals the scalar that
    function returns for the SAME inputs (within float32 rounding):

        F = sum_i [ alpha_i D(q_i||p_i) + R(alpha_i)
                  + lambda_beta ( sum_j beta_ij E_ij
                                  + tau sum_j beta_ij log(beta_ij/pi_ij) ) ]   (entropy: canonical only)

    ``belief_coupling`` and ``attention_entropy`` are the RAW (unweighted) block energies, so each
    stays individually interpretable; ``total`` is the runtime-realised SCALED free energy
    self_coupling + lambda_beta (belief_coupling + attention_entropy), matching what the E-step
    actually minimizes (VFE_2.0 parity). At lambda_beta = 1.0 total is byte-identical to the
    unscaled sum.

    ``alpha_reg`` is the per-position regularizer R(alpha_i) (b0 alpha - c0 log alpha under
    ``state_dependent_per_coord``); when supplied it is added elementwise into the self-coupling
    term (and so into ``total``), matching ``free_energy``. Default None leaves the result
    byte-identical to the unregularized path. ``include_attention_entropy`` gates the entropy
    contribution to ``total`` exactly as ``free_energy`` does -- when False the surrogate objective
    the E-step descends omits the entropy term, but its value is still reported under
    ``attention_entropy`` for diagnostics.
    """
    self_term = alpha * self_div
    if alpha_reg is not None:
        self_term = self_term + alpha_reg
    self_coupling = float(self_term.sum())
    belief_coupling = float((beta * energy).sum())
    # log pi WITHOUT materializing a full (H,N,N) uniform-pi tensor: with log_prior=None, pi=1/N is
    # uniform so log(pi)=-log(N) is a scalar (audit 2026-06-17 r2 id9; the scalar new_tensor(1/N) keeps
    # the value byte-identical to the old full_like alloc). The log_prior branch is unchanged.
    if log_prior is not None:
        log_pi = torch.log(torch.softmax(log_prior, dim=-1).clamp(min=eps))
    else:
        log_pi = torch.log(beta.new_tensor(1.0 / beta.shape[-1]).clamp(min=eps))
    from vfe3.free_energy import _broadcast_tau          # align a per-head (H,) tau to the head axis
    _tau_e = _broadcast_tau(tau, energy)
    entropy = float((_tau_e * (beta * (torch.log(beta.clamp(min=eps)) - log_pi))).sum())
    total = self_coupling + float(lambda_beta) * belief_coupling
    if include_attention_entropy:
        total = total + float(lambda_beta) * entropy
    return {
        "self_coupling":   self_coupling,
        "belief_coupling": belief_coupling,
        "attention_entropy": entropy,
        "total":           total,
    }


# ===========================================================================
# Publication-figure metrics (pure measurements; no gradients, no side effects).
# These return per-token / per-head VECTORS or small dicts for figures, rather
# than the single CSV scalar the registry path logs. Runners that reload a
# checkpoint or drive the model live (belief_bank, e_step_belief_trace,
# per_unit_eval_nats, ...) live in vfe3/viz/extract.py, NOT here, so this
# module keeps its "pure measurement" contract.
# ===========================================================================

def _is_full_cov(
    sigma:    torch.Tensor,              # (..., K) diagonal OR (..., K, K) full
    diagonal: Optional[bool] = None,
) -> bool:
    r"""Whether ``sigma`` is a full (..., K, K) covariance vs a diagonal (..., K) variance vector.

    Explicit ``diagonal`` wins; otherwise inferred from a trailing SQUARE pair (last two axes
    equal and ndim >= 2). The diagonal default of this codebase is (..., K), so the only
    ambiguous case is a coincidental K == leading-dim; pass ``diagonal=`` to disambiguate.
    """
    if diagonal is not None:
        return not diagonal
    return sigma.dim() >= 2 and sigma.shape[-1] == sigma.shape[-2]


def _spectrum(
    sigma:    torch.Tensor,              # (..., K) diagonal variances OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
) -> torch.Tensor:                       # (..., K) eigenvalue spectrum
    r"""The covariance spectrum: the variances themselves (diagonal) or ``eigvalsh`` (full)."""
    if _is_full_cov(sigma, diagonal):
        return torch.linalg.eigvalsh(0.5 * (sigma + sigma.transpose(-1, -2)))
    return sigma


def effective_rank_per_token(
    sigma:    torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
    eps:      float = 1e-12,
) -> torch.Tensor:                       # (...) per-token effective rank (NOT mean-reduced)
    r"""Per-token spectral effective rank (sum lam)^2 / sum lam^2, the (...) VECTOR.

    The registered ``effective_rank`` metric reduces this to a single ``.mean()`` per eval; this
    keeps the per-token distribution a single-seed run needs for a ridgeline/violin. Full
    covariances are passed through ``eigvalsh`` first.
    """
    return effective_rank(_spectrum(sigma, diagonal=diagonal), eps=eps)


def belief_spectrum(
    sigma:    torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
    eps:      float = 1e-12,
) -> Dict[str, torch.Tensor]:
    r"""Per-token spectral picture of the belief covariances (all PER-TOKEN, not mean-reduced).

    Returns ``eigenvalues`` (..., K) sorted DESCENDING, the spectral condition number
    ``condition`` = lam_max / lam_min (...), and the ``effective_rank`` (...). The sole producer
    of the guarded eigenvalue scree (figure F9 Panel B); the numerical-trust panel references it.
    """
    lam = _spectrum(sigma, diagonal=diagonal).clamp(min=0.0)
    lam_desc = torch.sort(lam, dim=-1, descending=True).values
    lam_max = lam_desc[..., 0]
    lam_min = lam_desc[..., -1].clamp(min=eps)
    return {
        "eigenvalues":    lam_desc,
        "condition":      lam_max / lam_min,
        "effective_rank": effective_rank(lam, eps=eps),
    }


def fisher_trace(
    sigma:    torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
    eps:      float = 1e-12,
) -> torch.Tensor:                       # (...) per-token Fisher-information trace in the mean block
    r"""Per-token Fisher-information trace of the Gaussian mean block, tr(Sigma^{-1}) / 2.

    For a Gaussian N(mu, Sigma) the Fisher information of the mean is Sigma^{-1}; its trace
    measures total precision (confidence). Diagonal: sum_k 1 / (2 sigma_k). Full:
    (1/2) tr(Sigma^{-1}). Used as the marker-size glyph in the belief-UMAP figure (confident
    beliefs render solid).
    """
    if _is_full_cov(sigma, diagonal):
        sym = 0.5 * (sigma + sigma.transpose(-1, -2))
        # eps ridge floors the inverse like the diagonal branch's clamp (audit 2026-06-09 IG1):
        # a singular belief covariance must not crash the figure path.
        eye = torch.eye(sym.shape[-1], device=sym.device, dtype=sym.dtype)
        inv = torch.linalg.inv(sym + eps * eye)
        return 0.5 * torch.diagonal(inv, dim1=-2, dim2=-1).sum(dim=-1)
    return (0.5 / sigma.clamp(min=eps)).sum(dim=-1)


def sigma_trace(
    sigma:    torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
) -> torch.Tensor:                       # (...) per-token tr(Sigma_q) = sum_k Var_k
    r"""Per-token covariance trace tr(Sigma_q) = sum_k Var_k -- the total belief UNCERTAINTY.

    The complement of ``fisher_trace`` (which returns the PRECISION trace tr(Sigma^{-1})/2): for the
    Sigma_q-as-calibrated-uncertainty probe (B1/EXP-3) the load-bearing quantity is the variance
    trace itself, whose across-token spread (see ``cv``) gates whether the channel carries any
    decode-time signal. Diagonal: sum_k sigma_k. Full: sum_k Sigma_kk.
    """
    if _is_full_cov(sigma, diagonal):
        return torch.diagonal(sigma, dim1=-2, dim2=-1).sum(dim=-1)
    return sigma.sum(dim=-1)


def rank_one_residual(
    mu:  torch.Tensor,                   # (..., N, K) per-token belief means of ONE layer

    *,
    eps: float = 1e-12,
) -> torch.Tensor:                       # (...) relative distance of the token cloud from rank-one
    r"""Dong rank-one residual r(X) = ||X - 1 xbar^T||_F / ||X||_F over the (N, K) mean matrix.

    Measures how far the per-token mean cloud X (rows = tokens) sits from the rank-one matrix
    1 xbar^T (every token collapsed to the mean token xbar = mean_n X[n,:]): r -> 0 is total rank
    collapse (all tokens identical), r near 1 is full spread. This is the anti-rank-collapse /
    FFN-brake object of F2/EXP-7, computed on the MEAN stream -- DISTINCT from ``effective_rank``,
    which is the spectral rank of the per-token COVARIANCE Sigma (a different object). Reference:
    Dong et al. 2021, "Attention is not all you need: pure attention loses rank doubly exponentially".
    """
    xbar = mu.mean(dim=-2, keepdim=True)                                  # (..., 1, K) mean token
    num = torch.linalg.norm((mu - xbar).flatten(-2, -1), dim=-1)          # ||X - 1 xbar^T||_F
    den = torch.linalg.norm(mu.flatten(-2, -1), dim=-1).clamp(min=eps)    # ||X||_F
    return num / den


def depth_decay_rate(
    curve: torch.Tensor,                 # (L,) a per-layer scalar (e.g. r(X) by depth)

    *,
    eps:   float = 1e-12,
) -> float:                              # log-linear slope d log(curve) / d layer
    r"""Log-linear decay rate (slope of log(curve) vs layer index) of a per-depth scalar.

    For F2/EXP-7 the per-arm rank-residual curves are compared by their DECAY RATE, not absolute
    level (the no-anchor control plateaus rather than collapsing to rank-one). Fits
    log(curve) ~ a + b * layer by least squares and returns b (negative = decaying with depth).
    """
    y = torch.log(curve.flatten().clamp(min=eps).to(torch.float64))
    n = y.shape[0]
    if n < 2:
        raise ValueError(f"depth_decay_rate needs >= 2 layers, got {n}")
    x = torch.arange(n, dtype=torch.float64, device=y.device)
    xm, ym = x.mean(), y.mean()
    return float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum().clamp(min=eps))


def spearman_rho(
    x: torch.Tensor,                     # (M,) sample
    y: torch.Tensor,                     # (M,) sample

    *,
    eps: float = 1e-12,
) -> float:                              # Spearman rank correlation in [-1, 1]
    r"""Spearman rank correlation: the Pearson correlation of the RANKS of x and y.

    The headline statistic of the Sigma_q-calibration probe (B1/EXP-3): rho(tr Sigma_q, per-token
    CE). Ranks via double argsort (ties broken by position, adequate for continuous diagnostics).
    Returns 0.0 for a degenerate zero-variance input rather than NaN.
    """
    x, y = x.flatten().to(torch.float64), y.flatten().to(torch.float64)
    if x.numel() != y.numel():
        raise ValueError(f"spearman_rho needs equal-length inputs, got {x.numel()} vs {y.numel()}")
    rx = x.argsort().argsort().to(torch.float64)
    ry = y.argsort().argsort().to(torch.float64)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp(min=eps))


def cv(
    x: torch.Tensor,                     # (M,) sample

    *,
    eps: float = 1e-12,
) -> float:                              # coefficient of variation std / |mean|
    r"""Coefficient of variation std(x) / |mean(x)| -- the pre-registered spread gate for B1/EXP-3.

    The calibration experiment requires CV(tr Sigma_q) > 0.10: below it the covariance carries no
    across-token signal and the result is reported "covariance inert", not miscoded as "decode
    does not matter". Uses the unbiased std (ddof=1).
    """
    x = x.flatten().to(torch.float64)
    if x.numel() < 2:
        raise ValueError(f"cv needs >= 2 samples, got {x.numel()}")
    return float(x.std(unbiased=True) / x.mean().abs().clamp(min=eps))


def spd_geodesic_distance(
    sigma_a:  torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance
    sigma_b:  torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
    eps:      float = 1e-12,
) -> torch.Tensor:                       # (...) affine-invariant SPD distance d_AI(Sigma_a, Sigma_b)
    r"""Affine-invariant (Rao/Fisher) geodesic distance on the SPD cone.

    d_AI(Sigma_a, Sigma_b) = ||log(Sigma_a^{-1/2} Sigma_b Sigma_a^{-1/2})||_F
                           = sqrt(sum_k (log lambda_k)^2),
    where lambda_k are the generalized eigenvalues of (Sigma_b, Sigma_a). For diagonal beliefs
    this reduces to sqrt(sum_k log^2(sigma_b,k / sigma_a,k)). This is the metric the SPD
    retraction itself uses, so belief-trajectory / E-step-residual lengths are measured in the
    geometry the inference actually moves in. Symmetric, zero iff Sigma_a == Sigma_b.
    """
    if not _is_full_cov(sigma_a, diagonal):
        ratio = sigma_b.clamp(min=eps) / sigma_a.clamp(min=eps)
        return torch.sqrt((torch.log(ratio) ** 2).sum(dim=-1).clamp(min=0.0))
    a = 0.5 * (sigma_a + sigma_a.transpose(-1, -2))
    b = 0.5 * (sigma_b + sigma_b.transpose(-1, -2))
    wa, qa = torch.linalg.eigh(a)
    inv_sqrt = (qa * wa.clamp(min=eps).rsqrt().unsqueeze(-2)) @ qa.transpose(-1, -2)
    whitened = inv_sqrt @ b @ inv_sqrt
    lam = torch.linalg.eigvalsh(0.5 * (whitened + whitened.transpose(-1, -2))).clamp(min=eps)
    return torch.sqrt((torch.log(lam) ** 2).sum(dim=-1).clamp(min=0.0))


def attention_entropy_rows(
    beta: torch.Tensor,                  # (..., N, N) row-stochastic attention weights

    *,
    eps:  float = 1e-12,
) -> torch.Tensor:                       # (..., N) per-row (per query i) Shannon entropy
    r"""Per-row attention entropy H_i = -sum_j beta_ij log beta_ij, keeping every leading axis.

    The registered ``attention_entropy`` collapses heads, layers, and query rows into one float;
    this keeps the per-(layer, head, query) distribution a single-seed run must lean on. By
    construction ``attention_entropy_rows(beta).mean() == attention_entropy(beta)``. Shared by the
    attention-structure and gauge-specialization figures.
    """
    b = beta.clamp(min=eps)
    return -(b * torch.log(b)).sum(dim=-1)


def _ols_slope(
    y: torch.Tensor,                     # (..., M) response sampled at x = 0..M-1
) -> torch.Tensor:                       # (...) least-squares slope over the last axis
    r"""Ordinary-least-squares slope of ``y`` against the integer index x = 0..M-1 (last axis)."""
    m = y.shape[-1]
    x = torch.arange(m, device=y.device, dtype=y.dtype)
    x = x - x.mean()
    y = y - y.mean(dim=-1, keepdim=True)
    denom = (x * x).sum().clamp(min=1e-12)
    return (y * x).sum(dim=-1) / denom


def causal_sanity(
    beta:   torch.Tensor,                # (..., N, N) attention weights (rows = query i, cols = key j)

    *,
    active: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    r"""Causal-mask and row-stochastic correctness scalars, keeping leading (layer/head) axes.

    Returns ``future_leakage`` = max_{j>i} beta_ij (must be ~0 under a causal mask),
    ``row_sum_error`` = max_i |sum_j beta_ij - 1| (row-stochasticity), and ``active_set_slope``
    = OLS slope of the per-row active-key count (beta_ij > ``active``) against query index i
    (expected ~1 as the causal active set grows by one key per step). Each reduces the (N, N)
    pair block and keeps any leading axes, so a (..., H, N, N) stack yields per-head (...,) values.
    """
    n = beta.shape[-1]
    future = torch.triu(torch.ones(n, n, device=beta.device, dtype=torch.bool), diagonal=1)
    leak = torch.where(future, beta, torch.zeros_like(beta)).amax(dim=(-2, -1))
    row_sum_err = (beta.sum(dim=-1) - 1.0).abs().amax(dim=-1)
    counts = (beta > active).sum(dim=-1).to(beta.dtype)              # (..., N) active keys per row
    return {
        "future_leakage":  leak,
        "row_sum_error":   row_sum_err,
        "active_set_slope": _ols_slope(counts),
    }


def guard_saturation(
    sigma:     torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariance
    energy:    torch.Tensor,             # (..., N, N) pairwise energies E_ij
    self_div:  torch.Tensor,             # (..., N) self-divergences D(q_i || p_i)

    *,
    diagonal:  Optional[bool] = None,
    eps:       float = 1e-6,
    sigma_max: float = 5.0,
    kl_max:    float = 100.0,
    rtol:      float = 1e-3,
) -> Dict[str, float]:
    r"""Fraction of converged-belief entries pinned at each numerical guard boundary.

    Certifies whether the SPD variance floor/ceiling (``eps``, ``sigma_max``) and the KL clamps
    (``kl_max`` on E_ij and on D(q||p)) are INERT on the pure path or load-bearing. sat_frac =
    mean(|x - boundary| < rtol * |boundary|) over the relevant entries; the variance spectrum is
    the eigenvalues for a full covariance. A near-zero map means the clamps never bind.
    """
    spec = _spectrum(sigma, diagonal=diagonal)
    # eigvalsh error scales with the largest eigenvalue (~eps_machine * lam_max), so on a FULL
    # covariance the tight relative window rtol*eps at the SPD FLOOR (eps ~ 1e-6) sits far below the
    # eigensolver noise -- which would report sigma_floor as NEVER binding even when it binds on every
    # token. Widen the spectrum-boundary tolerance to that noise floor on the full-cov path only; the
    # diagonal path (_spectrum returns the variances bit-exact) keeps the tight relative window.
    spec_atol = (64.0 * 2.0 ** -23 * float(spec.detach().abs().max())
                 if (_is_full_cov(sigma, diagonal) and spec.numel()) else 0.0)

    def _frac(x: torch.Tensor, boundary: float, *, atol: float = 0.0) -> float:
        if x.numel() == 0:
            return 0.0
        scale = max(abs(boundary), eps)
        return float(((x - boundary).abs() < max(rtol * scale, atol)).float().mean())

    return {
        "sigma_floor_frac":   _frac(spec, eps, atol=spec_atol),
        "sigma_ceil_frac":    _frac(spec, sigma_max, atol=spec_atol),
        "energy_klmax_frac":  _frac(energy, kl_max),
        "selfdiv_klmax_frac": _frac(self_div, kl_max),
    }


# --- gauge invariants (group-dispatched; fixes gauge_trace_spread's det-blindness) ---

def group_gauge_invariant(
    exp_phi: torch.Tensor,               # (..., K, K) per-token vertex factor exp(embed(phi_i))
    group,                               # GaugeGroup (dispatches the right invariant)

    *,
    eps:     float = 1e-12,
) -> torch.Tensor:                       # (...) per-token scalar gauge invariant
    r"""Per-token gauge invariant, dispatched on the group so it is non-vacuous for every group.

    The logged ``gauge_trace_spread`` (= std of log|det Omega| = tr embed(phi)) is identically 0
    for the unimodular groups SO(K) and Sp(2m) (traceless generators, det == 1), so it is blind
    off block_glk. This dispatches the correct invariant: GL volume log|det exp(phi)| for
    glk / block_glk / tied_block_glk, total rotation angle (1/2) sum_k |arg(eig)| for so_k and
    the orthogonal so_n irrep towers, and the eigenvalue-modulus squeeze max_k log|lambda_k| -
    min_k log|lambda_k| for sp / sp_n (whose Sym^p images are traceless, so logdet is blind there
    too; the eigenvalue moduli are conjugation-invariant, unlike the singular values).
    """
    name = getattr(group, "name", "glk")
    if name in ("so_k", "so_n"):
        ang = torch.angle(torch.linalg.eigvals(exp_phi))         # (..., K) eigenphases
        return 0.5 * ang.abs().sum(dim=-1)
    if name in ("sp", "sp_n"):
        # The gauge action is GL(K) CONGRUENCE, so exp(phi) transforms by conjugation g exp(phi) g^{-1};
        # the singular-value squeeze log(s_max/s_min) is invariant only under ORTHOGONAL conjugation, and
        # Sp(2m,R) is not orthogonal. Use the eigenvalue-MODULUS squeeze instead -- eigenvalues are
        # conjugation-invariant and symplectic spectra come in reciprocal pairs, so this is the natural
        # Ad-invariant squeeze (was frame-dependent under svdvals; audit 2026-06-13 review).
        logmod = torch.log(torch.linalg.eigvals(exp_phi).abs().clamp(min=eps))   # (..., K) log|eig|
        return logmod.amax(dim=-1) - logmod.amin(dim=-1)
    return torch.linalg.slogdet(exp_phi).logabsdet              # GL volume log|det|


def per_head_gauge_invariants(
    exp_phi:    torch.Tensor,            # (..., K, K) per-token vertex factor exp(embed(phi_i))
    irrep_dims: List[int],               # gauge-irrep block sizes; sum == K

    *,
    eps:        float = 1e-12,
) -> Dict[str, torch.Tensor]:
    r"""Per-head, per-token GL(d_head) invariants from the converged vertex factor.

    For a block-diagonal group the head-h frame is the (d_head, d_head) diagonal block of
    ``exp_phi``; returns its log-volume ``logdet`` (..., H) and its shear/anisotropy
    ``anisotropy`` = s_max / s_min (..., H). A single-block group (``irrep_dims = [K]``) yields
    H = 1. Feeds the per-head gauge-specialization ridgelines.
    """
    logdets, anisos = [], []
    start = 0
    for d in irrep_dims:
        blk = exp_phi[..., start:start + d, start:start + d]      # (..., d, d) head block
        logdets.append(torch.linalg.slogdet(blk).logabsdet)
        s = torch.linalg.svdvals(blk)
        anisos.append(s[..., 0] / s[..., -1].clamp(min=eps))
        start += d
    return {
        "logdet":     torch.stack(logdets, dim=-1),              # (..., H)
        "anisotropy": torch.stack(anisos, dim=-1),               # (..., H)
    }


# --- transport / energy directedness (the ln(3) symmetry-breaking signal) ---

def transport_asymmetry(
    omega: torch.Tensor,                 # (..., N, N, K, K) pairwise transport Omega_ij
) -> torch.Tensor:                       # (..., N, N) A_ij = ||Omega_ij - Omega_ji||_F
    r"""Directedness of the transport: A_ij = ||Omega_ij - Omega_ji||_F.

    Zero when the gauge is off (phi = 0 so every Omega_ij = I, hence symmetric); structured once
    the learned directed transport Omega_ij = exp(phi_i) exp(-phi_j) breaks the i<->j averaging
    symmetry. Omega_ji is the swap of the two token axes (not a matrix transpose).
    """
    omega_ji = omega.transpose(-4, -3)                           # swap the i and j token axes
    return torch.linalg.norm(omega - omega_ji, dim=(-2, -1))


def energy_directedness(
    energy: torch.Tensor,                # (..., N, N) pairwise energies E_ij

    *,
    eps:    float = 1e-12,
) -> Dict[str, torch.Tensor]:
    r"""Asymmetry of the pre-softmax pairwise energy E_ij over off-diagonal pairs.

    Returns the mean absolute asymmetry mean|E_ij - E_ji| and its scale-free normalization
    mean(|E_ij - E_ji| / (E_ij + E_ji)); both are 0 for symmetric (frozen-gauge) transport and
    grow once Omega(phi) is directional.
    """
    n = energy.shape[-1]
    off = ~torch.eye(n, dtype=torch.bool, device=energy.device)
    e_t = energy.transpose(-1, -2)
    abs_asym = (energy - e_t).abs()
    rel_asym = abs_asym / (energy + e_t).abs().clamp(min=eps)
    mask = off.expand_as(abs_asym)
    return {
        "abs_asymmetry": abs_asym[mask].mean() if mask.any() else abs_asym.new_zeros(()),
        "rel_asymmetry": rel_asym[mask].mean() if mask.any() else rel_asym.new_zeros(()),
    }


def structured_head_scores(
    beta:        torch.Tensor,           # (..., N, N) attention weights (rows = query i)

    *,
    period:      int = 3,
    band_width:  int = 1,
) -> Dict[str, torch.Tensor]:
    r"""Per-head induction/copy structure: prev-token, period-match, and diagonal-band mass.

    ``prev_token`` = mean_i beta_{i,i-1}; ``period_match`` = mean over causal pairs with i > j and
    (i - j) mod ``period`` == 0 of beta_ij (the period-3 copy structure on the synthetic stream);
    ``diagonal_band`` = mean_i sum_{0 < i-j <= band_width} beta_ij. Each keeps leading (head) axes.
    """
    n = beta.shape[-1]
    ii = torch.arange(n, device=beta.device).unsqueeze(-1)
    jj = torch.arange(n, device=beta.device).unsqueeze(0)
    diff = ii - jj
    causal = diff > 0
    period_mask = causal & (diff % period == 0)
    band_mask = causal & (diff <= band_width)
    prev = torch.diagonal(beta, offset=-1, dim1=-2, dim2=-1).mean(dim=-1)

    def _masked_mean(mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(beta.dtype)
        denom = m.sum().clamp(min=1.0)
        return (beta * m).sum(dim=(-2, -1)) / denom

    return {
        "prev_token":    prev,
        "period_match":  _masked_mean(period_mask),
        "diagonal_band": _masked_mean(band_mask),
    }


# --- attention structure (entropy, head redundancy, distance decay) ---

def head_redundancy_js(
    beta: torch.Tensor,                  # (H, N, N) per-head attention for one layer

    *,
    eps:  float = 1e-12,
) -> torch.Tensor:                       # (H, H) mean row-wise Jensen-Shannon divergence (nats)
    r"""Pairwise Jensen-Shannon divergence between heads' row distributions (head specialization).

    JS(beta^h || beta^h') = mean_i [ (1/2) KL(beta_i^h || m_i) + (1/2) KL(beta_i^h' || m_i) ],
    m_i = (1/2)(beta_i^h + beta_i^h'). High off-diagonal -> specialized heads, low -> redundant.
    """
    b = beta.clamp(min=eps)                                       # (H, N, N)
    bh = b.unsqueeze(1)                                           # (H, 1, N, N)
    bh2 = b.unsqueeze(0)                                          # (1, H, N, N)
    m = 0.5 * (bh + bh2)
    kl1 = (bh * (torch.log(bh) - torch.log(m))).sum(dim=-1)       # (H, H, N)
    kl2 = (bh2 * (torch.log(bh2) - torch.log(m))).sum(dim=-1)
    return (0.5 * (kl1 + kl2)).mean(dim=-1)                       # (H, H)


def attention_distance_decay(
    beta: torch.Tensor,                  # (..., N, N) attention weights
) -> Dict[str, torch.Tensor]:
    r"""Per-head attention-vs-offset profile beta_bar(d) = mean over causal pairs i - j = d.

    Returns ``offsets`` (D,) = 0..N-1 and ``profile`` (..., D), the mean attention at each
    query-key offset d (averaged over the offset-d diagonal). The positional-decay diagnostic;
    figures may bootstrap over rows for a band.
    """
    n = beta.shape[-1]
    profiles = [torch.diagonal(beta, offset=-d, dim1=-2, dim2=-1).mean(dim=-1) for d in range(n)]
    return {
        "offsets": torch.arange(n, device=beta.device),
        "profile": torch.stack(profiles, dim=-1),                # (..., N)
    }


def positional_content_score(
    beta: torch.Tensor,                  # (..., N, N) attention weights

    *,
    eps:  float = 1e-12,
) -> torch.Tensor:                       # (...) per-head R^2 of log beta vs offset |i - j|
    r"""Per-head positional<->content score: R^2 of the OLS fit log(beta_ij) ~ a + b |i - j|.

    Over the causal entries (i >= j). R^2 near 1 means attention is explained by token DISTANCE
    (positional, the gauge/RoPE machinery); near 0 means it is content-driven (the divergence
    energy E_ij). Places each head on a positional<->content axis.
    """
    n = beta.shape[-1]
    ii = torch.arange(n, device=beta.device).unsqueeze(-1)
    jj = torch.arange(n, device=beta.device).unsqueeze(0)
    mask = ii >= jj                                              # causal incl. diagonal (N, N)
    x = (ii - jj).to(beta.dtype)[mask]                           # (P,)
    y = torch.log(beta.clamp(min=eps))[..., mask]               # (..., P)
    xc = x - x.mean()
    yc = y - y.mean(dim=-1, keepdim=True)
    slope = (yc * xc).sum(dim=-1) / (xc * xc).sum().clamp(min=eps)
    ss_res = ((yc - slope.unsqueeze(-1) * xc) ** 2).sum(dim=-1)
    ss_tot = (yc ** 2).sum(dim=-1).clamp(min=eps)
    return 1.0 - ss_res / ss_tot


# --- holonomy / curvature (corrected sampling; Regime-II quantity) ---

def holonomy_deviation_sampled(
    omega:            torch.Tensor,      # (N, N, K, K) pairwise transport Omega_ij

    *,
    n_triples:        int  = 512,
    n_boot:           int  = 200,
    seed:             int  = 0,
) -> Dict[str, torch.Tensor]:
    r"""Triangle holonomy ||Omega_ij Omega_jk Omega_ki - I||_F over RANDOM distinct triples.

    Replaces ``holonomy_deviation``'s deterministic first-512 row-major enumeration (which always
    samples the same low-index-token triangles -- a systematically biased curvature estimate)
    with seeded random distinct (i, j, k), returning the full per-triple distribution, a
    bootstrap-over-triples mean CI, and each triple's index span max|.-.|. On the flat phi-cocycle
    every triangle closes (H = I) so this is ~0 (a flatness certificate); genuine curvature
    appears only under the opt-in regime_ii connection.
    """
    n, k = omega.shape[0], omega.shape[-1]
    eye = torch.eye(k, device=omega.device, dtype=omega.dtype)
    gen = torch.Generator(device=omega.device).manual_seed(int(seed))
    draw = torch.randint(0, n, (max(n_triples * 3, 12), 3), generator=gen, device=omega.device)
    keep = (draw[:, 0] != draw[:, 1]) & (draw[:, 1] != draw[:, 2]) & (draw[:, 0] != draw[:, 2])
    idx = draw[keep][:n_triples]
    if idx.numel() == 0:
        z = omega.new_zeros(())
        return {"mean": z, "ci_lo": z, "ci_hi": z,
                "per_triple": omega.new_zeros(0), "span": omega.new_zeros(0)}
    h = omega[idx[:, 0], idx[:, 1]] @ omega[idx[:, 1], idx[:, 2]] @ omega[idx[:, 2], idx[:, 0]]
    per = torch.linalg.norm(h - eye, dim=(-2, -1))               # (T,)
    span = (idx.amax(dim=1) - idx.amin(dim=1)).to(omega.dtype)
    ridx = torch.randint(0, per.shape[0], (n_boot, per.shape[0]), generator=gen, device=omega.device)
    boot = per[ridx].mean(dim=1)
    return {
        "mean":       per.mean(),
        "ci_lo":      torch.quantile(boot, 0.025),
        "ci_hi":      torch.quantile(boot, 0.975),
        "per_triple": per,
        "span":       span,
    }


def holonomy_wilson_sampled(
    omega:      torch.Tensor,            # (N, N, K, K) pairwise transport Omega_ij

    *,
    n_heads:    int  = 1,
    n_triples:  int  = 512,
    n_boot:     int  = 200,
    seed:       int  = 0,
) -> Dict[str, torch.Tensor]:
    r"""Wilson holonomy observable W_ijk = Re Tr(H_ijk) over random distinct triples.

    The gauge-INVARIANT triangle observable (PIFB:862-869): H_ijk = Omega_ij Omega_jk Omega_ki,
    W_ijk = Re Tr(H_ijk), reported normalized as W/K so a closed triangle H = I gives W/K = 1,
    with the Wilson-action density 1 - W/K (flat -> 0) -- the trace complement of
    ``holonomy_deviation_sampled``'s Frobenius ||H - I||_F. When ``n_heads > 1`` the gauge is
    block-diagonal H = oplus_h H^(h), so Tr(H) = sum_h Tr(H^(h)) and the per-head observable is
    W^(h)/d_k with d_k = K / n_heads; the per-head normalized values average back to the full W/K.
    The real part is exact (= Tr) for the real groups built today; it would need ``.real`` only for
    a complex GL(K, C) gauge. On the flat phi-cocycle every triangle closes so W/K = 1 (deviation 0);
    genuine holonomy appears only under the opt-in regime_ii connection. Seeded random triples match
    ``holonomy_deviation_sampled`` so the two observables are directly comparable.
    """
    n, k = omega.shape[0], omega.shape[-1]
    if n_heads > 1 and k % n_heads != 0:
        raise ValueError(f"n_heads={n_heads} must divide K={k} for the per-head Wilson decomposition")
    gen = torch.Generator(device=omega.device).manual_seed(int(seed))
    draw = torch.randint(0, n, (max(n_triples * 3, 12), 3), generator=gen, device=omega.device)
    keep = (draw[:, 0] != draw[:, 1]) & (draw[:, 1] != draw[:, 2]) & (draw[:, 0] != draw[:, 2])
    idx = draw[keep][:n_triples]
    if idx.numel() == 0:
        z = omega.new_zeros(())
        return {"wilson_mean": omega.new_ones(()), "deviation_mean": z, "ci_lo": z, "ci_hi": z,
                "per_triple": omega.new_zeros(0), "span": omega.new_zeros(0),
                "per_head": omega.new_ones(n_heads)}
    h = omega[idx[:, 0], idx[:, 1]] @ omega[idx[:, 1], idx[:, 2]] @ omega[idx[:, 2], idx[:, 0]]   # (T, K, K)
    diag = torch.diagonal(h, dim1=-2, dim2=-1)                   # (T, K) Re diagonal of H
    per = diag.sum(dim=-1) / k                                   # (T,) W_ijk / K
    span = (idx.amax(dim=1) - idx.amin(dim=1)).to(omega.dtype)
    dev = 1.0 - per                                              # (T,) Wilson-action density 1 - W/K
    ridx = torch.randint(0, dev.shape[0], (n_boot, dev.shape[0]), generator=gen, device=omega.device)
    boot = dev[ridx].mean(dim=1)
    d_k = k // n_heads
    per_head = diag.reshape(diag.shape[0], n_heads, d_k).sum(dim=-1).mean(dim=0) / d_k   # (n_heads,) W^(h)/d_k
    return {
        "wilson_mean":    per.mean(),
        "deviation_mean": dev.mean(),
        "ci_lo":          torch.quantile(boot, 0.025),
        "ci_hi":          torch.quantile(boot, 0.975),
        "per_triple":     per,
        "span":           span,
        "per_head":       per_head,
    }


def curvature_field(
    omega:  torch.Tensor,                # (N, N, K, K) pairwise transport Omega_ij
    anchor: int = 0,
) -> torch.Tensor:                       # (N, N) F_ij = ||Omega_ai Omega_ij Omega_ja - I||_F
    r"""Spatial curvature map for a fixed anchor a: F_ij = ||Omega_ai Omega_ij Omega_ja - I||_F.

    Shows where curvature concentrates (a Regime-II quantity; ~0 everywhere on the flat cocycle).
    """
    k = omega.shape[-1]
    eye = torch.eye(k, device=omega.device, dtype=omega.dtype)
    o_ai = omega[anchor].unsqueeze(1)                            # (N, 1, K, K) Omega_{a, i} (over i)
    o_ja = omega[:, anchor].unsqueeze(0)                         # (1, N, K, K) Omega_{j, a} (over j)
    h = o_ai @ omega @ o_ja                                      # (N, N, K, K)
    return torch.linalg.norm(h - eye, dim=(-2, -1))


def cocycle_residual_sampled(
    omega:      torch.Tensor,            # (N, N, K, K) pairwise transport Omega_ij

    *,
    n_triples:  int = 512,
    seed:       int = 0,
) -> torch.Tensor:                       # () mean ||Omega_ik - Omega_ij Omega_jk||_F over random triples
    r"""Cocycle defect mean ||Omega_ik - Omega_ij Omega_jk||_F over random distinct triples.

    The flatness certificate at the level of the COMPOSITION law itself: a flat Regime-I phi-cocycle
    Omega_ij = exp(phi_i) exp(-phi_j) satisfies Omega_ik = Omega_ij Omega_jk exactly, so this is ~0
    (cheaper and more fundamental than the triangle holonomy, which is the special case k -> i). A
    non-cocycle (regime_ii) transport gives > 0. Seeded random triples (matching
    ``holonomy_deviation_sampled``) avoid the row-major low-index sampling bias.
    """
    n, k = omega.shape[0], omega.shape[-1]
    gen = torch.Generator(device=omega.device).manual_seed(int(seed))
    draw = torch.randint(0, n, (max(n_triples * 3, 12), 3), generator=gen, device=omega.device)
    keep = (draw[:, 0] != draw[:, 1]) & (draw[:, 1] != draw[:, 2]) & (draw[:, 0] != draw[:, 2])
    idx = draw[keep][:n_triples]
    if idx.numel() == 0:
        return omega.new_zeros(())
    o_ik = omega[idx[:, 0], idx[:, 2]]                           # (T, K, K)
    o_ij = omega[idx[:, 0], idx[:, 1]]
    o_jk = omega[idx[:, 1], idx[:, 2]]
    return torch.linalg.norm(o_ik - o_ij @ o_jk, dim=(-2, -1)).mean()


# --- free-energy closure and per-token profile (the headline F figure) ---

def free_energy_full_decomposition(
    self_coupling:   'float | torch.Tensor',     # F self-term sum_i[alpha_i D(q_i||p_i) + R(alpha_i)] (nats)
    belief_coupling: 'float | torch.Tensor',     # raw sum_ij beta_ij E_ij (nats)
    attention_ent:   'float | torch.Tensor',     # raw tau sum_ij beta_ij log(beta_ij/pi_ij) (nats)
    data_term:       'float | torch.Tensor',     # -E_q[log p(o|x)] = val_ce in nats

    *,
    lambda_beta:     'float | torch.Tensor' = 1.0,
    lambda_h:        float = 0.0,
    lambda_gamma:    float = 0.0,
) -> Dict[str, 'float | torch.Tensor']:
    r"""Close the free-energy stack: add the data/likelihood term and the lambda_beta scaling.

    The runtime-realised total is self_coupling + lambda_beta (belief_coupling + attention_entropy)
    + data_term, so the stacked-area figure CLOSES to the F the E-step minimizes (the current
    one-bar snapshot omits the data term and the scaling). GUARDED: the closure holds only because
    the model-channel terms are inert at the defaults lambda_h = lambda_gamma = 0 (free_energy_terms
    carries no s-channel term); a nonzero value means the displayed total UNDERCOUNTS the
    hierarchical h->s->p->q F, so warn loudly. Accepts scalars or per-step arrays elementwise.
    """
    if lambda_h != 0.0 or lambda_gamma != 0.0:
        import warnings
        warnings.warn(
            f"free_energy_full_decomposition: lambda_h={lambda_h}, lambda_gamma={lambda_gamma} "
            f"are nonzero, but free_energy_terms carries NO model-channel (s) term, so the returned "
            f"'total' UNDERCOUNTS the hierarchical free energy. The stack closes to the runtime F "
            f"only at lambda_h = lambda_gamma = 0.",
            RuntimeWarning, stacklevel=2,
        )
    belief_scaled = lambda_beta * belief_coupling
    entropy_scaled = lambda_beta * attention_ent
    return {
        "self_coupling":   self_coupling,
        "belief_coupling": belief_scaled,
        "attention_entropy": entropy_scaled,
        "data_term":       data_term,
        "total":           self_coupling + belief_scaled + entropy_scaled + data_term,
    }


def self_coupling_profile(
    self_div: torch.Tensor,              # (N,) or (N, K) per-coordinate D(q_i||p_i)
    alpha:    torch.Tensor,              # (N,) or (N, K) self-coupling alpha_i

    *,
    eps:      float = 1e-12,
) -> Dict[str, torch.Tensor]:
    r"""Surface the per-token (D_i, alpha_i, alpha_i D_i) that diagnostics computes then collapses.

    ``self_divergence_for_alpha`` and ``self_coupling_alpha`` already produce these inside
    model.diagnostics before they are summed into the scalar ``self_coupling``; this returns the
    per-token vectors for the self-divergence violin. Per-coordinate inputs (N, K) are summed over
    the coordinate axis for the per-token totals while the raw arrays are kept. (alpha_i is a flat
    scalar on the default constant-alpha path; only informative under a state-dependent alpha.)
    """
    div_tok = self_div.sum(dim=-1) if self_div.dim() > 1 else self_div
    coupling_tok = (alpha * self_div)
    coupling_tok = coupling_tok.sum(dim=-1) if coupling_tok.dim() > 1 else coupling_tok
    return {
        "self_div":             div_tok,
        "alpha":                alpha,
        "self_coupling_per_token": coupling_tok,
    }


def estep_residuals(
    mu_traj:    torch.Tensor,            # (T+1, N, K) belief means over inner iterations
    sigma_traj: torch.Tensor,            # (T+1, N, K) diagonal OR (T+1, N, K, K) full covariances
    phi_traj:   torch.Tensor,            # (T+1, N, n_gen) gauge frames

    *,
    diagonal:   Optional[bool] = None,
    eps:        float = 1e-12,
) -> Dict[str, torch.Tensor]:
    r"""Per-iteration belief-change residuals (T, N), the covariance step in the SPD metric.

    r_mu(t) = ||mu_t - mu_{t-1}||, r_sigma(t) = spd_geodesic_distance(Sigma_{t-1}, Sigma_t)
    (the affine-invariant step, NOT a Euclidean one), r_phi(t) = ||phi_t - phi_{t-1}||. Shrinking
    residuals certify the E-step converges to a fixed point in the geometry it actually moves in.
    """
    r_mu = torch.linalg.norm(mu_traj[1:] - mu_traj[:-1], dim=-1)              # (T, N)
    r_sigma = spd_geodesic_distance(sigma_traj[:-1], sigma_traj[1:], diagonal=diagonal, eps=eps)
    r_phi = torch.linalg.norm(phi_traj[1:] - phi_traj[:-1], dim=-1)          # (T, N)
    return {"r_mu": r_mu, "r_sigma": r_sigma, "r_phi": r_phi}


# --- single-seed bootstrap bands (resample data, NOT seeds) ---

def bootstrap_ce_band(
    per_seq_nats:   torch.Tensor,        # (S,) summed nats per validation sequence
    per_seq_tokens: torch.Tensor,        # (S,) non-ignored token count per sequence

    *,
    n_boot:         int   = 1000,
    seed:           int   = 0,
    q_lo:           float = 0.025,
    q_hi:           float = 0.975,
) -> Dict[str, float]:
    r"""Token-weighted cross-entropy with a bootstrap-over-SEQUENCES band (single-seed-legitimate).

    Resamples validation sequences with replacement and reports the token-weighted CE
    sum(nats) / sum(tokens) percentiles. This is within-run uncertainty over the eval set, NOT a
    cross-seed confidence interval (the run protocol is single-seed); captions must say so.
    """
    gen = torch.Generator(device=per_seq_nats.device).manual_seed(int(seed))
    s = per_seq_nats.shape[0]
    idx = torch.randint(0, s, (n_boot, s), generator=gen, device=per_seq_nats.device)
    boot = per_seq_nats[idx].sum(dim=1) / per_seq_tokens[idx].sum(dim=1).clamp(min=1.0)
    point = float(per_seq_nats.sum() / per_seq_tokens.sum().clamp(min=1.0))
    return {"ce": point, "lo": float(torch.quantile(boot, q_lo)), "hi": float(torch.quantile(boot, q_hi))}


def bootstrap_token_ce_band(
    arm_token_nats:  torch.Tensor,       # (M,) per-token nats for the ablation arm
    full_token_nats: torch.Tensor,       # (M,) per-token nats for the full model (SAME tokens)

    *,
    n_boot:          int   = 1000,
    seed:            int   = 0,
    q_lo:            float = 0.025,
    q_hi:            float = 0.975,
) -> Dict[str, float]:
    r"""Paired bootstrap-over-TOKENS band for the ablation delta (arm minus full) cross-entropy.

    The SAME resample index is applied to both arms (paired), so the band reflects the per-token
    correlation between models, not independent noise. Single-seed-legitimate (within-run over the
    shared token set), not a cross-seed CI.
    """
    if arm_token_nats.shape != full_token_nats.shape:
        raise ValueError(f"paired bootstrap needs aligned tokens; got {tuple(arm_token_nats.shape)} "
                         f"vs {tuple(full_token_nats.shape)}")
    gen = torch.Generator(device=arm_token_nats.device).manual_seed(int(seed))
    m = arm_token_nats.shape[0]
    idx = torch.randint(0, m, (n_boot, m), generator=gen, device=arm_token_nats.device)
    delta = arm_token_nats[idx].mean(dim=1) - full_token_nats[idx].mean(dim=1)
    point = float(arm_token_nats.mean() - full_token_nats.mean())
    return {"delta": point, "lo": float(torch.quantile(delta, q_lo)), "hi": float(torch.quantile(delta, q_hi))}


# --- gauge-equivariance certificate (the symmetry the construction rests on) ---

def gauge_equivariance_residual(
    mu:                torch.Tensor,     # (N, K) converged belief means
    sigma:             torch.Tensor,     # (N, K) diagonal OR (N, K, K) full covariances
    omega:             torch.Tensor,     # (N, N, K, K) converged transport Omega_ij
    group,                               # GaugeGroup (generators, irrep_dims)

    *,
    kappa:             float = 1.0,
    renyi_order:       float = 1.0,
    kl_max:            float = 100.0,
    eps:               float = 1e-6,
    diagonal:          Optional[bool] = None,
    n_samples:         int   = 8,
    scale:             float = 0.5,
    seed:              int   = 0,
    divergence_family: str   = "renyi",
) -> Dict[str, torch.Tensor]:
    r"""Empirical gauge-equivariance certificate of the attention energy and weights.

    The construction's central symmetry is D(rho(g) q_i || rho(g) Omega_ij q_j) = D(q_i||Omega_ij q_j)
    under a GLOBAL structure-group element g (with the transport co-transforming Omega -> g Omega g^{-1}).
    Applies n_samples random IN-group g = exp(sum_a c_a G_a) and a matched OUT-of-group control
    g = exp(scale * randn(K, K)) (generic GL(K), which for a block/skew/symplectic group does NOT
    respect the structure), recomputes E_ij and beta_ij with the FULL Gaussian family (a general g
    makes a diagonal Sigma full), and returns the relative residuals. In-group residuals cluster at
    float32 eps; the out-of-group control sits far above. (For glk the group IS all of GL(K), so the
    control is also invariant -- correct, there is no 'outside'.)

    SCOPE (audit 2026-06-10 F12): this certifies the joint-congruence invariance of the GIVEN
    operator tensor -- (mu, Sigma, Omega) co-transformed together -- which holds for ANY omega,
    including a regime_ii one. It is therefore BLIND to a BUILDER-level equivariance break: under
    transport_mode='regime_ii' the built operator satisfies Omega(g mu) != g Omega(mu) g^{-1} for
    nonzero connection_W (the documented user-accepted break, pinned by
    tests/test_regime_ii.py::test_regime_ii_edge_factor_breaks_gauge_invariance_for_nonzero_W),
    and that deviation never enters this residual because omega is supplied, not rebuilt. Read
    the in-group panel as "congruence consistency of the converged operator", NOT as a
    certificate that the regime_ii construction is equivariant.
    """
    from vfe3.families.base import get_family
    from vfe3.free_energy import attention_tau, attention_weights, pairwise_energy
    from vfe3.geometry.transport import transport_covariance, transport_mean

    full = get_family("gaussian_full")
    is_full = _is_full_cov(sigma, diagonal)
    sigma0 = sigma if is_full else torch.diag_embed(sigma)                    # (N, K, K)
    k = mu.shape[-1]
    _kappa = torch.as_tensor(kappa, dtype=torch.float32, device=mu.device) if isinstance(kappa, (list, tuple)) else kappa
    tau = attention_tau(_kappa, group.irrep_dims)
    off = ~torch.eye(mu.shape[0], dtype=torch.bool, device=mu.device)
    gen = torch.Generator(device=mu.device).manual_seed(int(seed))

    def _energy(mu_q: torch.Tensor, sig_q: torch.Tensor, om: torch.Tensor):
        mu_t = transport_mean(om.unsqueeze(0), mu_q.unsqueeze(0))[0]          # (N, N, K)
        sig_t = transport_covariance(om.unsqueeze(0), sig_q.unsqueeze(0))[0]  # (N, N, K, K)
        e = pairwise_energy(full(mu_q, sig_q), full(mu_t, sig_t),
                            alpha=renyi_order, kl_max=kl_max, eps=eps,
                            divergence_family=divergence_family, irrep_dims=group.irrep_dims)
        return e, attention_weights(e, tau=tau)

    e0, beta0 = _energy(mu, sigma0, omega)

    def _residuals(g: torch.Tensor):
        g_inv = torch.linalg.inv(g)
        mu_g = mu @ g.transpose(-1, -2)                                       # (N, K) g mu_i
        sig_g = g @ sigma0 @ g.transpose(-1, -2)                              # (N, K, K) g Sigma g^T
        om_g = torch.einsum("kl,ijlm,mn->ijkn", g, omega, g_inv)             # g Omega g^{-1}
        e, beta = _energy(mu_g, sig_g, om_g)
        r_e = ((e - e0).abs() / e0.abs().clamp(min=eps))
        r_b = (beta - beta0).abs()
        return r_e[..., off].flatten(), r_b.flatten()         # off masks the trailing (N, N) axes

    in_e, in_b, out_e, out_b = [], [], [], []
    for _ in range(n_samples):
        c = scale * torch.randn(group.generators.shape[0], generator=gen, device=mu.device, dtype=mu.dtype)
        g_in = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", c, group.generators))
        re, rb = _residuals(g_in)
        in_e.append(re); in_b.append(rb)
        m = scale * torch.randn(k, k, generator=gen, device=mu.device, dtype=mu.dtype)
        re2, rb2 = _residuals(torch.linalg.matrix_exp(m))
        out_e.append(re2); out_b.append(rb2)
    return {
        "energy_in_group":  torch.cat(in_e),
        "energy_out_group": torch.cat(out_e),
        "beta_in_group":    torch.cat(in_b),
        "beta_out_group":   torch.cat(out_b),
    }


def head_mixer_gauge_residual(
    mu:        torch.Tensor,             # (N, K) converged belief means
    sigma:     torch.Tensor,             # (N, K) diagonal OR (N, K, K) full covariances
    head_mixer,                          # HeadMixer module (the trained Schur-commutant mixer)
    group,                               # GaugeGroup (generators, irrep_dims)

    *,
    n_samples: int            = 8,
    scale:     float          = 0.5,
    seed:      int            = 0,
    eps:       float          = 1e-6,
    diagonal:  Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    r"""BUILDER-level gauge-equivariance residual of the head mixer (tied vs untied).

    Certifies whether the mixer operation commutes with an IN-group gauge action, i.e.
    mix(g mu, g Sigma g^T) == g mix(mu, Sigma) g^T for g = exp(sum_a c_a G_a) drawn from the
    GROUP's OWN generators. The Schur-commutant mixer M = blockdiag_t(A_t kron I_d) commutes
    with a TIED gauge Omega = kron(I_n, h) (group ``tied_block_glk``, generators kron(I_n, gl(d)))
    -- residual at float32 eps -- but NOT with the UNTIED per-head gauge of ``block_glk`` (each
    head its own gl(d)), where the residual grows as A drifts from I during training.

    This is the complement of :func:`gauge_equivariance_residual`, which co-transforms a SUPPLIED
    Omega (mu, Sigma, Omega together) and is therefore BLIND to a builder-level break: it certifies
    the converged operator's congruence consistency, not whether the construction is equivariant.
    Here the operator is REBUILT under g, so the tied-vs-untied distinction is visible -- this is
    the instrument the A2/EXP-9 ablation needs. Uses the FULL-covariance mixer path (the diagonal
    closed form sigma'[m] = sum_n A[m,n]^2 sigma[n] is equivariant only under DIAGONAL gauges), so a
    diagonal sigma is promoted to full via ``diag_embed``. Returns relative residuals for the mean
    (per token) and the covariance (Frobenius, per token); in-group they cluster at eps for a tied
    gauge and rise with mixer drift for an untied one.
    """
    is_full = _is_full_cov(sigma, diagonal)
    sigma0 = sigma if is_full else torch.diag_embed(sigma)                    # (N, K, K)
    mu_m, sig_m = head_mixer(mu, sigma0)                                      # mix(mu, Sigma)
    gen = torch.Generator(device=mu.device).manual_seed(int(seed))
    r_mu, r_sig = [], []
    for _ in range(n_samples):
        c = scale * torch.randn(group.generators.shape[0], generator=gen, device=mu.device, dtype=mu.dtype)
        g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", c, group.generators))   # in-group g
        mu_g  = mu @ g.transpose(-1, -2)                                      # (N, K)    g mu_i
        sig_g = torch.einsum("kl,nlm,jm->nkj", g, sigma0, g)                  # (N, K, K) g Sigma g^T
        mu_L, sig_L = head_mixer(mu_g, sig_g)                                 # mix(g .)
        mu_R  = mu_m @ g.transpose(-1, -2)                                    # g . mix (mean)
        sig_R = torch.einsum("kl,nlm,jm->nkj", g, sig_m, g)                   # g . mix (cov)
        r_mu.append((mu_L - mu_R).norm(dim=-1) / mu_R.norm(dim=-1).clamp(min=eps))
        r_sig.append((sig_L - sig_R).flatten(1).norm(dim=-1) / sig_R.flatten(1).norm(dim=-1).clamp(min=eps))
    return {"mu_residual": torch.cat(r_mu), "sigma_residual": torch.cat(r_sig)}


# ---------------------------------------------------------------------------
# Registry: name -> metric(**context). New probes slot in by name.
# ---------------------------------------------------------------------------
_METRICS: Dict[str, Callable] = {}


def register_metric(name: str) -> Callable:
    """Decorator registering a metric that reads its inputs from the context kwargs."""
    def _wrap(fn: Callable) -> Callable:
        _METRICS[name] = fn
        return fn
    return _wrap


def get_metric(name: str) -> Callable:
    """Return the registered metric (KeyError if absent)."""
    if name not in _METRICS:
        raise KeyError(f"no metric {name!r}; available: {sorted(_METRICS)}")
    return _METRICS[name]


# Each metric's OWN context key is REQUIRED (no None default): a missing or mis-keyed
# context now raises TypeError at the call instead of an AttributeError deep inside the
# kernel (effective_rank(None) etc.). The trailing **kw stays only to absorb SIBLING
# metrics' context keys, since ``compute_metrics`` floods the full context to every metric.
@register_metric("effective_rank")
def _m_eff_rank(*, sigma: torch.Tensor, **kw) -> float:
    """Mean spectral effective rank of the belief covariances.

    Routes through ``_spectrum`` so a FULL covariance (..., K, K) is reduced to its eigenvalue
    spectrum before the participation ratio (audit 2026-06-13 L15); passing the matrix directly
    treated its rows as a spectrum. A diagonal (..., K) variance tensor is its own spectrum."""
    return float(effective_rank(_spectrum(sigma)).mean())


@register_metric("attention_entropy")
def _m_attn_entropy(*, beta: torch.Tensor, **kw) -> float:
    """Mean attention row entropy."""
    return float(attention_entropy(beta))


@register_metric("holonomy_deviation")
def _m_holonomy(*, omega: torch.Tensor, **kw) -> float:
    """Mean triangle-holonomy departure from identity (curvature proxy).

    Routes to the SAMPLED estimator (audit 2026-06-10 F16): the deterministic row-major
    ``holonomy_deviation`` always reads the same low-index-token triangles -- a systematically
    biased curvature estimate -- and ``diagnostics()`` already logs the sampled one, so the
    registry key now agrees with the training log."""
    return float(holonomy_deviation_sampled(omega)["mean"])


@register_metric("gauge_trace_spread")
def _m_gauge_spread(*, phi: torch.Tensor, generators: torch.Tensor, **kw) -> float:
    """Spread of log|det Omega| across tokens."""
    return float(gauge_trace_spread(phi, generators))


@register_metric("free_energy_terms")
def _m_free_energy_terms(*, self_div=None, energy=None, beta=None, alpha=None,
                         tau, log_prior=None, **kw) -> Dict[str, float]:
    """Per-term free-energy decomposition (self-coupling, belief-coupling, attention entropy).

    ``tau`` is REQUIRED (no default): the wrapper has no way to recover the group-aware softmax
    temperature tau = kappa*sqrt(d_head), and a silent tau=1.0 makes the attention-entropy term (and
    so ``total``) wrong for any K>1 (audit 2026-06-13 L16). Callers pass ``attention_tau(...)`` -- as
    the live diagnostics path already does -- matching this module's required-context-key convention."""
    return free_energy_terms(self_div, energy, beta, alpha, tau=tau, log_prior=log_prior)


def compute_metrics(
    names: List[str],

    **context,
) -> Dict[str, 'float | Dict[str, float]']:
    r"""Run the named metrics against the keyword ``context`` (sigma=, beta=, omega=, ...).

    Most metrics return a float; ``free_energy_terms`` returns a nested ``Dict[str, float]``,
    so the value type is widened accordingly."""
    return {n: get_metric(n)(**context) for n in names}
