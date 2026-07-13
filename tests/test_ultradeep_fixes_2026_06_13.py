r"""Regression tests for the 2026-06-13 ultra-deep multi-expert audit fixes.

Each test pins a finding from ``docs/audits/audit-2026-06-13-ultradeep.md``. Grouped by the
audit's finding id (M*/L*). Device-agnostic (CPU default; VFE3_TEST_DEVICE=cuda for GPU).
"""

import math
import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.families.base import renyi
from vfe3.families.gaussian import DiagonalGaussian, FullGaussian


# ---------------------------------------------------------------------------
# M1 — s_e_step=True now REFINES the model channel in the configured family (PB-11,
#      2026-07-12): the diagonal-only rejection is obsolete because _refine_s dispatches
#      through get_family(cfg.family) against the family-rank centroid r, so a full-covariance
#      s channel constructs and refines to a full (B,N,K,K) covariance instead of crashing.
# ---------------------------------------------------------------------------
def test_m1_s_e_step_accepts_and_refines_full_covariance_family():
    from vfe3.model.model import VFEModel

    cfg = VFE3Config(
        vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1, n_e_steps=1,
        s_e_step=True, prior_source="model_channel", family="gaussian_full", decode_mode="full",
        lambda_h=0.5, lambda_gamma=1.0, e_s_mu_lr=0.5, e_s_sigma_lr=0.5,
    )
    assert cfg.s_e_step and cfg.family == "gaussian_full"
    torch.manual_seed(0)
    m = VFEModel(cfg)
    tok = torch.randint(0, 6, (2, 4))
    phi0 = m._apply_pos_phi(m.prior_bank.encode(tok).phi)
    s1_mu, s1_sigma = m._refine_s(tok, phi0)
    assert s1_mu.shape == (2, 4, 4)
    assert s1_sigma.shape == (2, 4, 4, 4)                             # refined s stays full rank
    assert torch.isfinite(s1_mu).all() and torch.isfinite(s1_sigma).all()


def test_m1_s_e_step_accepts_diagonal_family():
    cfg = VFE3Config(s_e_step=True, prior_source="model_channel", family="gaussian_diagonal",
                     lambda_h=0.5)
    assert cfg.s_e_step and cfg.family == "gaussian_diagonal"


# ---------------------------------------------------------------------------
# float64 references (the exact divergence; the fp32 kernels must track these)
# ---------------------------------------------------------------------------
def _ref_diag_renyi(mu_q, sigma_q, mu_t, sigma_t, alpha, eps=1e-6):
    sq = sigma_q.double().clamp(min=eps)
    st = sigma_t.double().clamp(min=eps)
    delta = (mu_t - mu_q).double()
    blend = (1.0 - alpha) * sq + alpha * st
    mahal = (alpha * delta ** 2 / blend).sum(dim=-1)
    logdet = (
        (1.0 - alpha) * torch.log(sq) + alpha * torch.log(st) - torch.log(blend)
    ).sum(dim=-1) / (alpha - 1.0)
    return 0.5 * (mahal + logdet)


def _ref_full_renyi(mu_q, sigma_q, mu_t, sigma_t, alpha, eps=1e-6):
    K = mu_q.shape[-1]
    eye = torch.eye(K, dtype=torch.float64)
    sq = sigma_q.double() + eps * eye
    st = sigma_t.double() + eps * eye
    delta = (mu_t - mu_q).double()
    blend = (1.0 - alpha) * sq + alpha * st
    blend = 0.5 * (blend + blend.transpose(-1, -2))
    v = torch.linalg.solve(blend, delta.unsqueeze(-1)).squeeze(-1)
    mahal = alpha * (delta * v).sum(dim=-1)
    ldq = torch.linalg.slogdet(sq).logabsdet
    ldt = torch.linalg.slogdet(st).logabsdet
    ldb = torch.linalg.slogdet(blend).logabsdet
    logdet = ((1.0 - alpha) * ldq + alpha * ldt - ldb) / (alpha - 1.0)
    return 0.5 * (mahal + logdet)


_BAND_ALPHAS = [1.0 - 1e-3, 1.0 - 3e-4, 1.0 - 1e-4, 1.0 - 3e-5,
                1.0 + 3e-5, 1.0 + 1e-4, 1.0 + 3e-4, 1.0 + 1e-3]


# ---------------------------------------------------------------------------
# M2 — Renyi float32 catastrophic-cancellation band just outside the KL switch
# ---------------------------------------------------------------------------
def test_m2_renyi_diag_no_cancellation_in_kl_band():
    torch.manual_seed(8)
    K = 4
    mu_q = torch.randn(K)
    mu_t = torch.randn(K)
    sigma_q = torch.rand(K) * 2.0 + 0.1
    sigma_t = torch.rand(K) * 2.0 + 0.1
    q = DiagonalGaussian(mu_q, sigma_q)
    t = DiagonalGaussian(mu_t, sigma_t)
    for alpha in _BAND_ALPHAS:
        got = float(q.renyi_closed_form(t, alpha=alpha))
        ref = float(_ref_diag_renyi(mu_q, sigma_q, mu_t, sigma_t, alpha))
        assert abs(got - ref) <= 1e-4 * abs(ref) + 1e-6, (alpha, got, ref)


def test_m2_renyi_per_coord_no_cancellation_in_kl_band():
    torch.manual_seed(8)
    K = 4
    mu_q = torch.randn(K)
    mu_t = torch.randn(K)
    sigma_q = torch.rand(K) * 2.0 + 0.1
    sigma_t = torch.rand(K) * 2.0 + 0.1
    q = DiagonalGaussian(mu_q, sigma_q)
    t = DiagonalGaussian(mu_t, sigma_t)
    for alpha in _BAND_ALPHAS:
        got = float(q.renyi_per_coord(t, alpha=alpha).sum(dim=-1))
        ref = float(_ref_diag_renyi(mu_q, sigma_q, mu_t, sigma_t, alpha))
        assert abs(got - ref) <= 1e-4 * abs(ref) + 1e-6, (alpha, got, ref)


def test_m2_renyi_full_no_cancellation_in_kl_band():
    torch.manual_seed(8)
    K = 4
    mu_q = torch.randn(K)
    mu_t = torch.randn(K)
    A = torch.randn(K, K)
    B = torch.randn(K, K)
    sigma_q = A @ A.transpose(-1, -2) + torch.eye(K)
    sigma_t = B @ B.transpose(-1, -2) + torch.eye(K)
    q = FullGaussian(mu_q, sigma_q)
    t = FullGaussian(mu_t, sigma_t)
    for alpha in _BAND_ALPHAS:
        got = float(q.renyi_closed_form(t, alpha=alpha))
        ref = float(_ref_full_renyi(mu_q, sigma_q, mu_t, sigma_t, alpha))
        assert abs(got - ref) <= 1e-4 * abs(ref) + 1e-6, (alpha, got, ref)


def test_m2_renyi_diag_out_of_band_matches_ref():
    # away from the band the (unchanged) fp32 path must still match the exact divergence
    torch.manual_seed(3)
    K = 4
    mu_q = torch.randn(K)
    mu_t = torch.randn(K)
    # variances kept near 1 so the alpha>1 blend (1-alpha)*sigma_q + alpha*sigma_t stays PD
    sigma_q = torch.rand(K) * 0.5 + 0.75
    sigma_t = torch.rand(K) * 0.5 + 0.75
    q = DiagonalGaussian(mu_q, sigma_q)
    t = DiagonalGaussian(mu_t, sigma_t)
    for alpha in [0.5, 0.9, 1.1, 2.0]:
        got = float(q.renyi_closed_form(t, alpha=alpha))
        ref = float(_ref_diag_renyi(mu_q, sigma_q, mu_t, sigma_t, alpha))
        assert abs(got - ref) <= 1e-4 * abs(ref) + 1e-6, (alpha, got, ref)


# ---------------------------------------------------------------------------
# L7 — renyi() must not emit the non-PD-blend warning when the KL branch is taken
# ---------------------------------------------------------------------------
def test_l7_no_alpha_gt_one_warning_inside_kl_switch():
    K = 3
    q = DiagonalGaussian(torch.zeros(K), torch.ones(K))
    p = DiagonalGaussian(torch.ones(K), torch.ones(K))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        renyi(q, p, alpha=1.0 + 5e-7)            # > 1 but the closed form takes the plain KL branch
    assert not any(issubclass(w.category, RuntimeWarning) for w in rec), [str(w.message) for w in rec]


def test_l7_alpha_gt_one_warning_still_fires_outside_switch():
    K = 3
    q = DiagonalGaussian(torch.zeros(K), torch.ones(K))
    p = DiagonalGaussian(torch.ones(K), torch.ones(K))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        renyi(q, p, alpha=2.0)
    assert any(issubclass(w.category, RuntimeWarning) for w in rec)


# ---------------------------------------------------------------------------
# L9 — NOT LANDED. The audit recommended FullGaussian.entropy() return NaN on a non-PD
# Sigma, but that conflicts with the deliberate, separately-tested robustness contract
# (tests/test_full_covariance.py::test_full_entropy_survives_non_pd_covariance): entropy
# mirrors the KL path's jittered-Cholesky robustness (finite, never NaN to the caller, as KL
# clamps non-PD to kl_max). The deliberate tested design is kept; L9 is overruled.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# L10 — FullGaussian.log_partition_at must mask (not raise) on a non-PD natural param
# ---------------------------------------------------------------------------
def test_l10_full_log_partition_nan_on_non_pd():
    K = 3
    theta = (torch.zeros(K), 0.5 * torch.eye(K))     # neg2t2 = -2*t2 = -I (non-PD)
    out = FullGaussian.log_partition_at(theta)
    assert math.isnan(float(out))


def test_l10_full_log_partition_finite_on_pd():
    K = 3
    theta = (torch.zeros(K), -0.5 * torch.eye(K))    # neg2t2 = I (PD)
    out = FullGaussian.log_partition_at(theta)
    assert math.isfinite(float(out))


# ---------------------------------------------------------------------------
# M3 — regime_ii edge factor must be the EXACT exp for non-orthonormal bases
#      (cap the EMBEDDED matrix Frobenius norm, not the coordinate norm)
# ---------------------------------------------------------------------------
def test_m3_regime_ii_edge_factor_exact_for_non_orthonormal_son():
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.transport import get_transport

    torch.manual_seed(0)
    # equal-block (so the factored vertex path is happy) non-orthonormal so_n l2 tower:
    # two spin-2 blocks of dim 5 (Gram diag >> 1), K = 10, n_gen = N(N-1)/2 = 3 for N=3.
    group = get_group("so_n")(10, group_n=3, irrep_spec=[("l2", 2)])
    G = group.generators                                          # (3, 10, 10)
    n_gen, K = G.shape[0], G.shape[-1]
    B, N = 1, 3
    cap = 12.0
    phi = torch.zeros(B, N, n_gen)                                # vertex factors = I
    mu = torch.randn(B, N, K) * 5.0                               # large means -> raw delta >> cap
    W = torch.randn(n_gen, K, K) * 1.0

    out = get_transport("regime_ii")(phi, group, mu=mu, connection_W=W,
                                     cocycle_relaxation=1.0, delta_soft_cap=cap)
    omega = out["Omega"][0]                                       # (N, N, K, K)

    # reconstruct the algebra element the builder exponentiates, capped in the MATRIX Frobenius norm
    delta = torch.einsum("bik,akl,bjl->bija", mu, W, mu)[0]       # (N, N, n_gen)
    delta = delta.masked_fill(torch.eye(N, dtype=torch.bool).unsqueeze(-1), 0.0)
    delta_mat = torch.einsum("ija,akl->ijkl", delta, G)          # (N, N, K, K)

    # bug precondition: the OLD coordinate-norm cap leaves the embedded operator above max_norm=15
    coord_sq = delta.pow(2).sum(dim=-1, keepdim=True)
    delta_coord = delta * torch.rsqrt(1.0 + coord_sq / (cap * cap))
    embedded_coord = torch.einsum("ija,akl->ijkl", delta_coord, G)
    assert float(embedded_coord.norm(dim=(-2, -1)).max()) > 15.0

    # the NEW matrix-Frobenius cap -> exact exp (no stable_matrix_exp_pair clamp)
    fro = delta_mat.norm(dim=(-2, -1), keepdim=True)
    capped = delta_mat * torch.rsqrt(1.0 + (fro * fro) / (cap * cap))
    ref = torch.matrix_exp(capped)                               # exact operator
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.allclose(omega[off], ref[off], atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# M4 — full-covariance congruence sandwich is computed in a float64 island
#      (the sandwich squares cond(Omega); fp32 einsum accumulation corrupts it).
#      The fix makes the fp32-stored result the correctly-rounded float64 value;
#      the result-storage dtype is still the fundamental fp32 limit at extreme cond.
# ---------------------------------------------------------------------------
def test_m4_full_cov_sandwich_is_float64_island():
    from vfe3.geometry.transport import transport_covariance

    torch.manual_seed(0)
    K, N = 6, 2
    a = 4.0                                                       # ill-conditioned, non-orthogonal
    Q, _ = torch.linalg.qr(torch.randn(K, K))
    evals = torch.zeros(K)
    evals[0], evals[1] = a, -a
    A = (Q @ torch.diag(evals) @ Q.transpose(-1, -2)).float()
    Omega1 = torch.matrix_exp(A)
    omega = Omega1.expand(N, N, K, K).contiguous()
    S = torch.randn(N, K, K)
    sigma = (S @ S.transpose(-1, -2) + torch.eye(K)).contiguous()

    got = transport_covariance(omega, sigma, diagonal_out=False)
    # the fix evaluates the contraction in float64 then casts back: the result must be bit-identical
    # to the float64-computed-then-cast reference (the old fp32 einsum differed by accumulation)
    ref = torch.einsum("ijkl,jlm,ijnm->ijkn",
                       omega.double(), sigma.double(), omega.double()).to(torch.float32)
    assert torch.equal(got, ref)


def test_m4_full_cov_sandwich_diagonal_path_unchanged():
    # the diagonal default path (the hot path) must NOT be upcast / changed
    from vfe3.geometry.transport import transport_covariance

    torch.manual_seed(1)
    K, N = 5, 3
    omega = torch.randn(N, N, K, K)
    sigma = torch.rand(N, K) + 0.1                               # diagonal variances
    got = transport_covariance(omega, sigma, diagonal_out=True)
    ref = torch.einsum("ijkl,ijkl,jl->ijk", omega, omega, sigma)  # fp32, unchanged
    assert torch.equal(got, ref)


# ---------------------------------------------------------------------------
# L11 — the SPD full retraction's gap-regularized eigh backward must scale gap_eps
#       to the spectrum so small-but-MEANINGFUL eigenvalue gaps are not over-damped
# ---------------------------------------------------------------------------
def _eigh_loss(eigh_fn, A):
    w, V = eigh_fn(A)
    return V.sum() + w.sum()


def test_l11_rel_gap_eps_keeps_small_meaningful_gaps_accurate():
    from vfe3.geometry.retraction import _eigh_damped, _rel_gap_eps

    torch.manual_seed(0)
    K = 4
    Q, _ = torch.linalg.qr(torch.randn(K, K))
    base, gap = 2.0, 1e-4
    evals = torch.tensor([base, base + gap, base + 0.7, base + 1.5])
    A0 = (Q @ torch.diag(evals) @ Q.transpose(-1, -2)).float()
    A0 = 0.5 * (A0 + A0.transpose(-1, -2))

    A1 = A0.clone().requires_grad_(True)
    _eigh_loss(torch.linalg.eigh, A1).backward()
    g_exact = A1.grad

    A2 = A0.clone().requires_grad_(True)
    ge = _rel_gap_eps(A2)
    _eigh_loss(lambda X: _eigh_damped(X, ge), A2).backward()
    rel_new = float((A2.grad - g_exact).norm() / g_exact.norm())
    assert rel_new < 1e-2, (rel_new, ge)

    # contrast: the OLD fixed gap_eps=1e-8 over-damps this meaningful 1e-4 gap
    A3 = A0.clone().requires_grad_(True)
    _eigh_loss(lambda X: _eigh_damped(X, 1e-8), A3).backward()
    rel_old = float((A3.grad - g_exact).norm() / g_exact.norm())
    assert rel_old > 1e-1, rel_old


def test_l11_retract_spd_full_still_finite_at_isotropic():
    from vfe3.geometry.retraction import retract_spd_full

    K = 4
    sigma = torch.eye(K).reshape(1, K, K).clone().requires_grad_(True)
    delta = torch.zeros(1, K, K)
    out = retract_spd_full(sigma, delta)
    out.sum().backward()
    assert torch.isfinite(sigma.grad).all()


# ---------------------------------------------------------------------------
# L13 — the SPD condition_number monitor must handle a diagonal (...,K) spectrum
#       (the default gaussian_diagonal family), not raise on the rank mismatch
# ---------------------------------------------------------------------------
def test_l13_condition_number_diagonal_spectrum():
    from vfe3.numerics import condition_number

    sigma = torch.tensor([[1.0, 2.0, 4.0], [0.5, 0.5, 0.5]])     # (2, 3) diagonal variances, N!=K
    cond = condition_number(sigma)
    assert torch.allclose(cond, torch.tensor([4.0, 1.0]), atol=1e-5)


def test_l13_condition_number_monitor_diagonal_no_raise():
    from vfe3.numerics import get_monitor

    sigma = torch.rand(2, 3, 5) + 0.1                            # (B,N,K) diagonal, N=3 != K=5
    val = get_monitor("condition_number")(sigma)
    assert math.isfinite(val) and val >= 1.0


def test_l13_condition_number_full_matrix_unchanged():
    from vfe3.numerics import condition_number

    M = torch.diag(torch.tensor([1.0, 2.0, 4.0]))               # (3,3) full SPD, cond 4
    assert abs(float(condition_number(M)) - 4.0) < 1e-5


# ---------------------------------------------------------------------------
# L15 — the registered effective_rank metric must use the eigenvalue spectrum
#       for a full covariance (not treat matrix rows as a spectrum)
# ---------------------------------------------------------------------------
def test_l15_effective_rank_full_cov_uses_eigenvalues():
    from vfe3.metrics import get_metric, effective_rank, _spectrum

    torch.manual_seed(0)
    K = 4
    A = torch.randn(2, K, K)
    sigma = A @ A.transpose(-1, -2) + torch.eye(K)              # (2,K,K) full covariance
    got = get_metric("effective_rank")(sigma=sigma, diagonal=False)   # explicit flag (PB-07)
    ref = float(effective_rank(_spectrum(sigma, diagonal=False)).mean())
    assert abs(got - ref) < 1e-5


def test_l15_effective_rank_diagonal_unchanged():
    from vfe3.metrics import get_metric, effective_rank

    torch.manual_seed(0)
    sigma = torch.rand(2, 5, 4) + 0.1                            # (B,N,K) diagonal variances
    got = get_metric("effective_rank")(sigma=sigma, diagonal=True)    # explicit flag (PB-07)
    ref = float(effective_rank(sigma).mean())                    # diagonal: variances ARE the spectrum
    assert abs(got - ref) < 1e-5


# ---------------------------------------------------------------------------
# L16 — the free_energy_terms metric wrapper must require tau (no silent tau=1.0
#       that drops the group-aware softmax temperature)
# ---------------------------------------------------------------------------
def test_l16_free_energy_terms_metric_requires_tau():
    from vfe3.metrics import get_metric

    sd = torch.zeros(2)
    e = torch.zeros(2, 2)
    b = torch.full((2, 2), 0.5)
    a = torch.ones(2)
    with pytest.raises(TypeError, match="tau"):
        get_metric("free_energy_terms")(self_div=sd, energy=e, beta=b, alpha=a)
    out = get_metric("free_energy_terms")(self_div=sd, energy=e, beta=b, alpha=a, tau=2.0)
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# L20 — CGCoupling must thread a single atol through BOTH cg_selection (n_mult) and the
#       per-triple cg_intertwiners buffer build, so the prune and the multiplicities agree
# ---------------------------------------------------------------------------
def test_l20_cg_coupling_threads_atol_consistently():
    from vfe3.geometry.groups import get_group
    from vfe3.model.cg_coupling import CGCoupling

    # SO(3) tower with a self-coupling triple (l1 x l1 -> l2 exists), K = 2*3 + 5 = 11
    g = get_group("so_n")(11, group_n=3, irrep_spec=[("l1", 2), ("l2", 1)])
    base = CGCoupling(3, "so", g.irrep_dims, g.irrep_labels)          # default atol (param present)
    assert len(base.paths) > 0
    # an explicit default-equal atol must give an identical path set (the param is threaded, not
    # ignored); a non-default atol in the clean-gap regime must stay consistent (cg_selection's
    # n_mult and the buffer leading dim agree -- no mismatch crash, same count)
    same = CGCoupling(3, "so", g.irrep_dims, g.irrep_labels, atol=1e-8)
    other = CGCoupling(3, "so", g.irrep_dims, g.irrep_labels, atol=1e-6)
    assert len(same.paths) == len(base.paths)
    assert len(other.paths) == len(base.paths)
