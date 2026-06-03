import pytest
import torch

from vfe3.geometry.retraction import (
    get_retraction,
    natural_gradient,
    register_retraction,
    retract_logeuclidean_full,
    retract_spd_diagonal,
    retract_spd_full,
)


def test_sigma_max_caps_variance_consistently_across_diag_and_full():
    # sigma_max must denote ONE physical variance ceiling on both arms: the eigenvalues of a full
    # covariance ARE variances, so the full retraction must cap them at sigma_max -- NOT sigma_max**2,
    # which let the full family hold variances a factor sigma_max larger than the diagonal arm under
    # the same knob (the cov_kind-seam mismatch).
    sigma_max, K = 3.0, 4
    out_d = retract_spd_diagonal(torch.ones(2, K), torch.full((2, K), 1e3), sigma_max=sigma_max)
    assert out_d.max().item() <= sigma_max + 1e-5
    sig_f = torch.eye(K).expand(2, K, K).contiguous()
    huge  = torch.eye(K).expand(2, K, K).contiguous() * 1e3
    eig_affine = torch.linalg.eigvalsh(
        retract_spd_full(sig_f, huge, trust_region=0.0, sigma_max=sigma_max))
    assert eig_affine.max().item() <= sigma_max + 1e-4
    eig_le = torch.linalg.eigvalsh(
        retract_logeuclidean_full(sig_f, huge, trust_region=0.0, sigma_max=sigma_max))
    assert eig_le.max().item() <= sigma_max + 1e-4


def test_diagonal_retraction_positive_and_bounded():
    g = torch.Generator().manual_seed(0)
    sigma = torch.rand(4, 6, generator=g) + 0.1
    delta = 5.0 * torch.randn(4, 6, generator=g)
    out = retract_spd_diagonal(sigma, delta, sigma_max=5.0)
    assert (out >= 1e-6).all()
    assert (out <= 5.0 + 1e-6).all()


def test_full_retraction_stays_spd():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    D = torch.randn(3, 4, 4, generator=g)
    delta = 0.5 * (D + D.transpose(-1, -2))
    out = retract_spd_full(sigma, delta)
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-4)
    assert (torch.linalg.eigvalsh(out) > 0).all()


def test_full_retraction_identity_tangent_is_identity():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    zero = torch.zeros(3, 4, 4)
    # sigma_max=1e9 keeps the (variance) safety ceiling from binding: this random sigma has
    # eigenvalues > the default sigma_max, and the centering axiom R(Sigma,0)=Sigma is the geodesic
    # exp-map property, tested WITHIN the variance box (the output clamp correctly caps an
    # out-of-box base point at sigma_max on BOTH the diagonal and full arms).
    out = retract_spd_full(sigma, zero, sigma_max=1e9)
    assert torch.allclose(out, sigma, atol=1e-3)


def test_natural_gradient_diagonal_formula():
    g = torch.Generator().manual_seed(3)
    sigma = torch.rand(4, 5, generator=g) + 0.1
    gmu = torch.randn(4, 5, generator=g)
    gsig = torch.randn(4, 5, generator=g)
    nmu, nsig = natural_gradient(gmu, gsig, sigma)
    assert torch.allclose(nmu, sigma * gmu, atol=1e-6)
    assert torch.allclose(nsig, 2.0 * sigma * sigma * gsig, atol=1e-6)


def test_full_retraction_K1_matches_diagonal_formula():
    # For K=1 the affine-invariant SPD exp map reduces to the diagonal rule
    # sigma_new = sigma * exp(tau * delta/sigma).
    sigma = torch.tensor([[[2.0]]])          # (1,1,1) as (B,K,K) with K=1
    delta = torch.tensor([[[0.6]]])
    out = retract_spd_full(sigma, delta, trust_region=0.0)   # disable TR for exact check
    expected = 2.0 * torch.exp(torch.tensor(0.6 / 2.0))
    assert torch.allclose(out.reshape(()), expected, atol=1e-4)


# --- register_retraction / get_retraction seam (roadmap item 4) ------------
def test_retraction_registry_round_trip():
    """register_retraction/get_retraction round-trip and the unknown-name KeyError."""
    sentinel = object()

    @register_retraction("_test_dummy_retraction")
    def _dummy(*args, **kwargs):
        return sentinel

    try:
        assert get_retraction("_test_dummy_retraction")() is sentinel
        with pytest.raises(KeyError):
            get_retraction("nope")
    finally:
        from vfe3.geometry.retraction import _RETRACTIONS
        _RETRACTIONS.pop("_test_dummy_retraction", None)


def test_spd_affine_is_registered():
    """The default affine-invariant SPD retraction is registered under 'spd_affine'."""
    assert callable(get_retraction("spd_affine"))


def test_spd_affine_bit_identical_diagonal():
    """spd_affine reproduces the bare retract_spd_diagonal call bit-for-bit (atol=0)."""
    g = torch.Generator().manual_seed(11)
    sigma = torch.rand(3, 4, 5, generator=g) + 0.1      # (B, N, K) diagonal
    nat_sigma = torch.randn(3, 4, 5, generator=g)
    step = 0.015
    legacy = retract_spd_diagonal(
        sigma, -step * nat_sigma, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    seam = get_retraction("spd_affine")(
        sigma, -step * nat_sigma, mean_ndim=3, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    assert torch.equal(seam, legacy)


def test_spd_affine_bit_identical_full():
    """spd_affine reproduces the bare retract_spd_full call bit-for-bit (atol=0)."""
    g = torch.Generator().manual_seed(12)
    A = torch.randn(3, 4, 5, 5, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(5)       # (B, N, K, K) full SPD
    D = torch.randn(3, 4, 5, 5, generator=g)
    nat_sigma = 0.5 * (D + D.transpose(-1, -2))
    step = 0.015
    legacy = retract_spd_full(
        sigma, -step * nat_sigma, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    seam = get_retraction("spd_affine")(
        sigma, -step * nat_sigma, mean_ndim=3, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    assert torch.equal(seam, legacy)


# --- log_euclidean SPD retraction variant (spec 2a, pure log-chart) --------
def _logm_eigh(sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Independent eigh-based matrix log (test oracle, distinct from the impl)."""
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(sigma)
    evals = evals.clamp(min=eps)
    return evecs @ torch.diag_embed(torch.log(evals)) @ evecs.transpose(-1, -2)


def test_log_euclidean_stays_spd_unconditionally():
    """Random full-cov SPD Sigma + random symmetric tangent -> symmetric PD output
    across a range of step sizes (no trust region). LE is SPD-exact for any step
    (expm of a symmetric matrix is SPD), whereas a naive Euclidean step Sigma +
    step*delta can leave the cone. Steps kept within fp32 dynamic range so neither
    the eps floor nor the sigma_max cap binds (asserted below); beyond that the
    re-eigh of a clamped, extreme-scale matrix measures fp32 noise, not the math."""
    g = torch.Generator().manual_seed(21)
    A = torch.randn(5, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    D = torch.randn(5, 4, 4, generator=g)
    delta = 0.5 * (D + D.transpose(-1, -2))
    eps, sigma_max = 1e-6, 1e9
    for step in (0.1, 0.5, 1.0):
        out = get_retraction("log_euclidean")(
            sigma, delta, mean_ndim=2, step_size=step, trust_region=0.0,
            eps=eps, sigma_max=sigma_max,
        )
        assert torch.allclose(out, out.transpose(-1, -2), atol=1e-4)
        evals = torch.linalg.eigvalsh(out)
        assert (evals > 0).all(), f"not PD at step={step}"
        # neither the eps floor nor the sigma_max cap engaged: the genuine expm map, not a clamp
        assert evals.min() > eps and evals.max() < sigma_max * sigma_max

    # contrast: on an ill-conditioned base a naive Euclidean step leaves the SPD cone,
    # where LE (below) stays PD -- the distinguishing property of the log-chart map.
    sigma_ill = torch.diag(torch.tensor([0.05, 1.0, 1.0, 1.0]))
    tangent   = -0.5 * torch.eye(4)
    assert not (torch.linalg.eigvalsh(sigma_ill + tangent) > 0).all()
    le_ill = get_retraction("log_euclidean")(
        sigma_ill, tangent, mean_ndim=1, step_size=1.0, trust_region=0.0, eps=eps, sigma_max=sigma_max,
    )
    assert (torch.linalg.eigvalsh(le_ill) > 0).all()


def test_log_euclidean_identity_tangent_is_identity():
    """Retraction axiom R(Sigma, 0) = Sigma with the operational trust_region=5.0:
    the trust region must clamp the TANGENT, not the base point logm(Sigma), so a
    zero tangent returns Sigma even when ||logm(Sigma)||_F exceeds the trust region."""
    g = torch.Generator().manual_seed(25)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + 0.1 * torch.eye(4)   # spread spectrum -> large ||logm||_F
    zero = torch.zeros(3, 4, 4)
    out = get_retraction("log_euclidean")(
        sigma, zero, mean_ndim=2, step_size=1.0, trust_region=5.0, eps=1e-6, sigma_max=1e9,
    )
    assert torch.allclose(out, sigma, atol=1e-3)


def test_log_euclidean_full_matches_independent_expm_logm():
    """Full case equals an INDEPENDENTLY computed expm(logm(Sigma) + step*sym(delta))
    via torch.linalg.matrix_exp + an eigh-based logm written here. Well-conditioned
    Sigma + modest step so neither the eps floor nor the sigma_max cap binds."""
    g = torch.Generator().manual_seed(22)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + 2.0 * torch.eye(4)   # well-conditioned SPD
    D = torch.randn(3, 4, 4, generator=g)
    delta = 0.5 * (D + D.transpose(-1, -2))
    step = 0.1
    out = get_retraction("log_euclidean")(
        sigma, delta, mean_ndim=2, step_size=step, trust_region=0.0,
        eps=1e-9, sigma_max=1e9,
    )
    log_sigma = _logm_eigh(sigma, eps=1e-9)
    sym_delta = 0.5 * (delta + delta.transpose(-1, -2))
    expected = torch.linalg.matrix_exp(log_sigma + step * sym_delta)
    assert torch.allclose(out, expected, atol=1e-5)


def test_log_euclidean_diagonal_is_log_chart_step():
    """Diagonal case applies the tangent directly in the log chart:
    sigma_new = sigma * exp(step * delta) (NO affine 1/sigma whitening)."""
    g = torch.Generator().manual_seed(23)
    sigma = torch.rand(4, 5, generator=g) + 0.2
    delta = 0.3 * torch.randn(4, 5, generator=g)
    step = 0.4
    out = get_retraction("log_euclidean")(
        sigma, delta, mean_ndim=2, step_size=step, trust_region=0.0,
        eps=1e-9, sigma_max=1e9,
    )
    expected = sigma * torch.exp(step * delta)
    assert torch.allclose(out, expected, atol=1e-5)


def test_log_euclidean_diagonal_differs_from_affine():
    """Under the seam's pre-whitened tangent convention LE does NOT reduce to
    spd_affine on the diagonal (affine whitens by 1/sigma, LE does not), so they
    differ for a non-identity sigma. Pins the verified scope finding."""
    g = torch.Generator().manual_seed(24)
    sigma = torch.rand(4, 5, generator=g) + 0.5      # non-identity
    delta = 0.3 * torch.randn(4, 5, generator=g)
    step = 0.4
    le = get_retraction("log_euclidean")(
        sigma, delta, mean_ndim=2, step_size=step, trust_region=0.0, eps=1e-9, sigma_max=1e9,
    )
    affine = get_retraction("spd_affine")(
        sigma, delta, mean_ndim=2, step_size=step, trust_region=0.0, eps=1e-9, sigma_max=1e9,
    )
    assert not torch.allclose(le, affine, atol=1e-3)


def test_log_euclidean_registered_and_config_accepts():
    """get_retraction('log_euclidean') resolves and spd_retract_mode validates."""
    from vfe3.config import VFE3Config
    assert callable(get_retraction("log_euclidean"))
    cfg = VFE3Config(spd_retract_mode="log_euclidean", family="gaussian_full",
                     diagonal_covariance=False, decode_mode="full")
    assert cfg.spd_retract_mode == "log_euclidean"


def test_log_euclidean_diagonal_pairing_warns():
    """Config WARNs (not errors) when log_euclidean is paired with a diagonal family
    (the log-chart step lacks the affine Fisher whitening there)."""
    from vfe3.config import VFE3Config
    with pytest.warns(UserWarning, match="log_euclidean"):
        VFE3Config(spd_retract_mode="log_euclidean", family="gaussian_diagonal",
                   diagonal_covariance=True)


def test_log_euclidean_e_step_full_cov_runs():
    """A full-covariance E-step under spd_retract_mode='log_euclidean' produces a
    finite SPD covariance and finite grads (forward+backward)."""
    from vfe3.belief import BeliefState
    from vfe3.geometry.groups import get_group
    from vfe3.inference.e_step import e_step_iteration

    torch.manual_seed(31)
    N, K = 4, 4
    group = get_group("glk")(K)
    mu = torch.randn(1, N, K, requires_grad=True)
    base = torch.randn(1, N, K, K)
    sigma = (base @ base.transpose(-1, -2) + torch.eye(K)).requires_grad_(True)
    phi = torch.zeros(1, N, group.generators.shape[0])
    belief = BeliefState(mu=mu, sigma=sigma, phi=phi)
    mu_p = torch.randn(N, K)
    pbase = torch.randn(N, K, K)
    sigma_p = pbase @ pbase.transpose(-1, -2) + torch.eye(K)

    out = e_step_iteration(
        belief, mu_p, sigma_p, group,
        e_mu_lr=0.1, e_sigma_lr=0.05, e_phi_lr=0.0,
        family="gaussian_full", spd_retract_mode="log_euclidean",
    )
    assert torch.isfinite(out.sigma).all()
    assert (torch.linalg.eigvalsh(out.sigma) > 0).all()
    loss = out.mu.pow(2).sum() + out.sigma.pow(2).sum()
    loss.backward()
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(sigma.grad).all()
