r"""Route B: gauge-COVARIANT non-flat Regime-II transport.

The edge connection is built from gauge-INVARIANT scalar features of the (query,
transported-key) belief pair, so the edge factor exp(delta_ij . G) is invariant and the
transport stays covariant (Omega_ij -> g_i Omega_ij g_j^{-1}) under GL(K) frame changes --
unlike the bilinear ``regime_ii`` connection delta_ij = mu_i^T W mu_j, which is invariant
only at W=0. These tests pin that invariance (the whole point of "principled" Regime-II).
"""

import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators, get_transport
from vfe3.metrics import holonomy_deviation


def _spd(*shape: int, k: int) -> torch.Tensor:
    r"""Random SPD covariance batch of shape (*shape, k, k)."""
    a = torch.randn(*shape, k, k)
    return a @ a.transpose(-1, -2) + k * torch.eye(k)


def test_gauge_invariant_edge_features_are_gl_invariant() -> None:
    r"""delta-features (Mahalanobis, trace, log-det) of KL(q_i || transported q_j) are
    invariant under a common GL(K) push-forward applied to BOTH beliefs."""
    from vfe3.geometry.transport import gauge_invariant_edge_features

    torch.manual_seed(0)
    K = 4
    B, Nq, Nk = 2, 3, 3

    mu_q  = torch.randn(B, Nq, Nk, K)          # (B, Nq, Nk, K) per-edge query mean
    mu_kt = torch.randn(B, Nq, Nk, K)          # (B, Nq, Nk, K) transported key mean
    cov_q  = _spd(B, Nq, Nk, k=K)              # (B, Nq, Nk, K, K) query covariance
    cov_kt = _spd(B, Nq, Nk, k=K)             # (B, Nq, Nk, K, K) transported key covariance

    feats = gauge_invariant_edge_features(mu_q, cov_q, mu_kt, cov_kt)   # (B, Nq, Nk, 3)
    assert feats.shape == (B, Nq, Nk, 3)

    # A generic (non-orthogonal) invertible GL(K) frame change applied to both beliefs.
    g = 2.0 * torch.eye(K) + 0.3 * torch.randn(K, K)
    assert torch.linalg.det(g).abs() > 1e-3

    mu_q2  = torch.einsum("kl,bijl->bijk", g, mu_q)
    mu_kt2 = torch.einsum("kl,bijl->bijk", g, mu_kt)
    cov_q2  = torch.einsum("kl,bijlm,nm->bijkn", g, cov_q, g)
    cov_kt2 = torch.einsum("kl,bijlm,nm->bijkn", g, cov_kt, g)

    feats2 = gauge_invariant_edge_features(mu_q2, cov_q2, mu_kt2, cov_kt2)

    assert torch.allclose(feats, feats2, atol=1e-4, rtol=1e-4), (
        f"edge features not GL(K)-invariant: max abs diff "
        f"{(feats - feats2).abs().max().item():.3e}"
    )


def _phi_mu_sigma(seed=0, B=2, N=3, K=4, group="block_glk", n_heads=2):
    grp = get_group(group)(K, n_heads) if group in ("block_glk", "tied_block_glk") else get_group(group)(K)
    g = torch.Generator().manual_seed(seed)
    n_gen = grp.generators.shape[0]
    phi   = 0.3 * torch.randn(B, N, n_gen, generator=g)
    mu    = torch.randn(B, N, K, generator=g)
    sigma = torch.rand(B, N, K, generator=g) + 0.5
    return phi, mu, sigma, grp


# --- builder registry + flat oracle (M=0 / None -> exactly the flat cocycle) ---
def test_regime_ii_covariant_is_registered():
    assert callable(get_transport("regime_ii_covariant"))


def test_regime_ii_covariant_none_M_reduces_to_flat():
    """connection_M=None (un-threaded default) -> flat cocycle exactly."""
    phi, mu, sigma, grp = _phi_mu_sigma(seed=1)
    flat = compute_transport_operators(phi, grp)["Omega"]
    out = get_transport("regime_ii_covariant")(phi, grp, mu=mu, sigma=sigma, connection_M=None)["Omega"]
    assert out.shape == flat.shape
    assert torch.allclose(out, flat, atol=1e-6, rtol=0.0)


def test_regime_ii_covariant_zero_M_reduces_to_flat():
    """A zero connection_M tensor -> delta=0 -> exp(delta)=I -> flat cocycle to fp32 tol.
    (Not short-circuited, so autograd to M is preserved; delta=0 reduces to flat numerically.)"""
    phi, mu, sigma, grp = _phi_mu_sigma(seed=2)
    n_gen = grp.generators.shape[0]
    M = torch.zeros(n_gen, 3)
    flat = compute_transport_operators(phi, grp)["Omega"]
    out = get_transport("regime_ii_covariant")(phi, grp, mu=mu, sigma=sigma, connection_M=M)["Omega"]
    assert torch.allclose(out, flat, atol=1e-6, rtol=0.0)


def test_regime_ii_covariant_returns_flat_dict_shape():
    phi, mu, sigma, grp = _phi_mu_sigma(seed=4)
    K, n_gen = grp.generators.shape[-1], grp.generators.shape[0]
    M = 0.2 * torch.randn(n_gen, 3)
    out = get_transport("regime_ii_covariant")(phi, grp, mu=mu, sigma=sigma, connection_M=M)
    assert set(out) == {"exp_phi", "exp_neg_phi", "Omega"}
    B, N = phi.shape[0], phi.shape[1]
    assert out["exp_phi"].shape == (B, N, K, K)
    assert out["exp_neg_phi"].shape == (B, N, K, K)
    assert out["Omega"].shape == (B, N, N, K, K)


# --- genuinely non-flat when M != 0: holonomy strictly positive (curvature) ---
def test_regime_ii_covariant_nonzero_M_is_non_flat():
    """A nonzero connection_M gives the non-trivial triangle holonomy (the curved Regime II)."""
    phi, mu, sigma, grp = _phi_mu_sigma(seed=6, B=1, N=4)
    n_gen = grp.generators.shape[0]
    M = 0.2 * torch.randn(n_gen, 3, generator=torch.Generator().manual_seed(11))
    omega_flat = compute_transport_operators(phi, grp)["Omega"][0]
    omega_cov = get_transport("regime_ii_covariant")(
        phi, grp, mu=mu, sigma=sigma, connection_M=M
    )["Omega"][0]
    assert float(holonomy_deviation(omega_flat)) < 1e-4          # flat closes
    assert float(holonomy_deviation(omega_cov)) > 1e-2           # covariant Regime II does not


# --- model wiring: connection_M creation, init-flat == flat, threading, gradient flow ---
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _tiny_cfg(transport_mode="flat", **kw):
    return VFE3Config(
        vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
        n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.0, transport_mode=transport_mode, **kw,
    )


def test_model_regime_ii_covariant_creates_connection_m_zero_init():
    """transport_mode='regime_ii_covariant' creates connection_M as a zero-init (n_gen,3) Parameter."""
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii_covariant"))
    assert hasattr(model, "connection_M")
    assert isinstance(model.connection_M, torch.nn.Parameter)
    n_gen = model.group.generators.shape[0]
    assert model.connection_M.shape == (n_gen, 3)
    assert torch.equal(model.connection_M, torch.zeros(n_gen, 3))


def test_model_flat_has_no_connection_m():
    """The default flat model carries no connection_M (pure path is param-free)."""
    model = VFEModel(_tiny_cfg(transport_mode="flat"))
    assert not hasattr(model, "connection_M")


def test_model_covariant_init_flat_equals_flat_forward():
    """At init (M=0), a regime_ii_covariant model's forward equals the flat model's (init-flat)."""
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    torch.manual_seed(0)
    flat_model = VFEModel(_tiny_cfg(transport_mode="flat"))
    logits_flat, loss_flat, _ = flat_model(tokens, targets)
    torch.manual_seed(0)
    cov_model = VFEModel(_tiny_cfg(transport_mode="regime_ii_covariant"))
    logits_cov, loss_cov, _ = cov_model(tokens, targets)
    assert torch.allclose(logits_flat, logits_cov, atol=1e-6, rtol=0.0)
    assert torch.allclose(loss_flat, loss_cov, atol=1e-6, rtol=0.0)


def test_model_covariant_nonzero_m_changes_forward():
    """A nonzero connection_M actually changes the forward loss (the connection is threaded into
    the live transport, not ignored). Means inflated so the invariant features carry signal."""
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii_covariant"))
    with torch.no_grad():
        model.prior_bank.mu_embed *= 50.0
    _, loss_flat, _ = model(tokens, targets)
    with torch.no_grad():
        model.connection_M += 0.5 * torch.randn_like(model.connection_M)
    _, loss_nonflat, _ = model(tokens, targets)
    assert not torch.allclose(loss_flat, loss_nonflat, atol=1e-4)


def test_model_covariant_gradient_flows_to_m():
    """connection_M enters the loss only through the E-step belief updates; with the differentiable
    oracle (oracle_unroll_grad=True) loss.backward() populates a finite, NONZERO M.grad."""
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii_covariant", oracle_unroll_grad=True))
    with torch.no_grad():
        model.prior_bank.mu_embed *= 50.0
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.connection_M.grad is not None
    assert torch.isfinite(model.connection_M.grad).all()
    assert model.connection_M.grad.abs().sum() > 1e-6


def test_non_flat_regimes_auto_enable_oracle_unroll_grad():
    """The learned connection (connection_W / connection_M) enters the loss ONLY through the
    unrolled E-step, so the config AUTO-enables oracle_unroll_grad for the non-flat regimes
    (rather than warning the user to set it). The flat pure path keeps the default OFF."""
    assert VFE3Config(transport_mode="regime_ii_covariant").oracle_unroll_grad is True
    assert VFE3Config(transport_mode="regime_ii").oracle_unroll_grad is True
    assert VFE3Config(transport_mode="flat").oracle_unroll_grad is False


def test_build_optimizer_groups_connection_m():
    """connection_M must land in exactly one optimizer param group, else build_optimizer's
    coverage guard raises (the connection would never train). Regression: the param was created
    in the model but not added to a group."""
    from vfe3.train import build_optimizer
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii_covariant"))
    opt = build_optimizer(model, model.cfg)               # raises if connection_M is ungrouped
    grouped = {p for g in opt.param_groups for p in g["params"]}
    assert model.connection_M in grouped


# --- OOM fix: query-index chunking of the dense covariant builder (2026-06-18) -------------------
# The covariant builder materializes several dense (B, N, N, K, K) tensors at once (omega0, cov_kt,
# its Cholesky factors, delta_mat, exp_delta, omega). At K=20/2-head/B=64/N=128 that is ~1.68 GB
# EACH, and the K=20 regime_ii_covariant run OOMs while the larger flat K=80 run (factored, no dense
# Omega) fits. The builder now loops over the query index in chunks so only (B, C, N, K, K) is live
# at a time. Chunking must be exactly value- and gradient-equivalent to the one-chunk build (no
# cross-query reduction exists, so each (i, j) operator is byte-identical regardless of chunking).

def test_regime_ii_query_chunk_bounds_working_set():
    """The chunk-size policy chunks the OOM config (B=64, N=128, K=20) below N, and collapses to a
    single chunk for a tiny diagnostic build (B=1)."""
    from vfe3.geometry import transport as T
    assert 1 <= T._regime_ii_query_chunk(64, 128, 20) < 128       # OOM config must chunk
    assert T._regime_ii_query_chunk(1, 4, 4) == 4                 # tiny single-sequence build: one chunk


def test_regime_ii_covariant_chunked_matches_unchunked():
    """Forcing a tiny query chunk gives the SAME Omega and the SAME gradient to connection_M as the
    one-chunk build -- chunking is purely a memory optimization, numerically equivalent."""
    import pytest
    from vfe3.geometry import transport as T

    phi, mu, sigma, grp = _phi_mu_sigma(seed=7, B=2, N=6, K=8, n_heads=2)
    n_gen = grp.generators.shape[0]
    M0 = 0.2 * torch.randn(n_gen, 3, generator=torch.Generator().manual_seed(3))
    build = get_transport("regime_ii_covariant")

    def _omega_and_grad(chunk_elems: int):
        # monkeypatch-free: set/restore the module constant around each build (raises if absent -> RED)
        saved = T._REGIME_II_CHUNK_ELEMS
        T._REGIME_II_CHUNK_ELEMS = chunk_elems
        try:
            M = M0.clone().requires_grad_(True)
            omega = build(phi, grp, mu=mu, sigma=sigma, connection_M=M)["Omega"]
            (omega ** 2).sum().backward()
            return omega.detach().clone(), M.grad.detach().clone()
        finally:
            T._REGIME_II_CHUNK_ELEMS = saved

    omega_one, grad_one = _omega_and_grad(10 ** 12)              # one chunk (size N)
    omega_chunk, grad_chunk = _omega_and_grad(1)                 # forced size-1 chunks
    assert torch.allclose(omega_one, omega_chunk, atol=1e-5, rtol=1e-5)
    assert torch.allclose(grad_one, grad_chunk, atol=1e-5, rtol=1e-5)


# --- diagnostics threading: connection_M must reach the holonomy / converged-state transport ------
def test_diagnostics_holonomy_reflects_connection_m():
    """model.diagnostics builds its holonomy Omega under the ACTIVE regime; under regime_ii_covariant
    a nonzero connection_M must produce non-trivial curvature (holonomy_deviation > 0). Before the
    threading fix the diagnostic Omega fell back to flat (connection_M unpassed) -> holonomy ~0."""
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii_covariant"))
    with torch.no_grad():
        model.prior_bank.mu_embed *= 50.0
        model.connection_M += 0.5 * torch.randn_like(model.connection_M)
    tokens = torch.randint(0, 15, (1, 4))
    d = model.diagnostics(tokens)
    assert d["holonomy_deviation"] > 1e-4


def test_converged_state_omega_reflects_connection_m():
    """viz.extract.converged_state returns the diagnostic Omega; under regime_ii_covariant it must be
    built with connection_M (the F-trajectory / holonomy panels otherwise read a flat transport)."""
    from vfe3.viz.extract import converged_state
    from vfe3.metrics import holonomy_deviation
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii_covariant"))
    with torch.no_grad():
        model.prior_bank.mu_embed *= 50.0
        model.connection_M += 0.5 * torch.randn_like(model.connection_M)
    tokens = torch.randint(0, 15, (1, 4))
    st = converged_state(model, tokens)
    assert float(holonomy_deviation(st["omega"])) > 1e-4
