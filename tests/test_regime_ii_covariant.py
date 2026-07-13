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
    logits_flat = flat_model(tokens)
    _, loss_flat, _ = flat_model(tokens, targets)
    torch.manual_seed(0)
    cov_model = VFEModel(_tiny_cfg(transport_mode="regime_ii_covariant"))
    logits_cov = cov_model(tokens)
    _, loss_cov, _ = cov_model(tokens, targets)
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


# --- numerical robustness of the gauge-invariant edge features (audit 2026-06-18) ---------------
# The transported-key congruence S = Omega^0 Sigma Omega^0^T SQUARES cond(Omega^0) ~ exp(2||phi||)
# on the non-compact block_glk frame, so the edge-feature Cholesky must (a) run in a float64 island
# (an fp32 congruence destroys the gauge-invariant features -- >100% rel error at K=70 in the audit)
# and (b) degrade a non-PD S to NaN via safe_cholesky instead of raising a LinAlgError that aborts
# the whole forward.

def test_gauge_invariant_edge_features_does_not_raise_on_non_pd():
    """A non-PD transported covariance must degrade to NaN (-> kl_max downstream), not raise a
    LinAlgError. Mirrors the safe_cholesky contract used throughout families/gaussian.py."""
    from vfe3.geometry.transport import gauge_invariant_edge_features
    K = 4
    mu_q   = torch.zeros(K)
    mu_kt  = torch.zeros(K)
    cov_q  = torch.eye(K)
    cov_kt = torch.eye(K)
    cov_kt[0, 0] = -1.0                                   # clearly non-PD (not rescued by +eps)
    feats = gauge_invariant_edge_features(mu_q, cov_q, mu_kt, cov_kt)   # must NOT raise
    assert feats.shape == (3,)
    assert torch.isnan(feats).any()


def test_regime_ii_covariant_edge_features_use_float64_congruence():
    """On the non-compact block_glk frame the transported-key congruence is ill-conditioned, so the
    builder must evaluate the edge-feature congruence + Cholesky in float64. The builder INPUTS
    (phi, mu, sigma) are well-conditioned and fp32-exact; the ill-conditioning is born INSIDE the
    congruence. Isolation test: the connection's contribution to the fp32-vs-float64 Omega gap must
    be no larger than the flat (M=0) exp_phi baseline -- i.e. the feature path adds no fp32 blow-up."""
    import dataclasses
    from vfe3.geometry.transport import get_transport
    build = get_transport("regime_ii_covariant")

    phi, mu, sigma, grp = _phi_mu_sigma(seed=5, B=1, N=5, K=12, n_heads=2)
    phi = phi * 4.0                                       # push ||phi|| up -> ill-conditioned Omega^0
    n_gen = grp.generators.shape[0]
    M  = 0.3 * torch.randn(n_gen, 3, generator=torch.Generator().manual_seed(9))
    M0 = torch.zeros(n_gen, 3)
    grp64 = dataclasses.replace(grp, generators=grp.generators.double())

    def omega(dtype, conn):
        g = grp if dtype == torch.float32 else grp64
        return build(phi.to(dtype), g, mu=mu.to(dtype), sigma=sigma.to(dtype),
                     connection_M=conn.to(dtype))["Omega"].double()

    # fp32 features (the bug) either CRASH (non-PD congruence) or give >100% relative error; with the
    # float64 congruence the fp32-input curved build tracks its float64 reference to ~fp32 epsilon,
    # about as tightly as the flat (M=0) exp_phi baseline -- the connection adds no fp32 blow-up.
    ref_flat   = omega(torch.float64, M0)
    ref_curved = omega(torch.float64, M)
    rel_flat   = (omega(torch.float32, M0) - ref_flat).abs().max()   / ref_flat.abs().max()
    rel_curved = (omega(torch.float32, M)  - ref_curved).abs().max() / ref_curved.abs().max()
    assert rel_curved < 5e-3, (
        f"fp32 feature congruence blows up the covariant transport: curved rel err "
        f"{rel_curved:.3e} (flat baseline {rel_flat:.3e})")


def test_regime_ii_covariant_requires_sigma_when_connection_m_set():
    """connection_M provided but sigma=None must raise a clear ValueError, not an opaque
    AttributeError on sigma.dim(). Hardens the public builder contract (e_step always passes sigma)."""
    import pytest
    phi, mu, _sigma, grp = _phi_mu_sigma(seed=8)
    n_gen = grp.generators.shape[0]
    M = 0.2 * torch.randn(n_gen, 3)
    with pytest.raises(ValueError):
        get_transport("regime_ii_covariant")(phi, grp, mu=mu, sigma=None, connection_M=M)


def test_regime_ii_query_chunk_accounts_for_simultaneous_transients():
    """The chunk policy must bound the SUM of the several simultaneous dense (B,C,N,K,K) transients
    the builder holds (omega0, cov_kt, delta_mat, exp_delta, the output chunk), not a single tensor
    -- else the OOM budget underestimates peak by ~5x (audit 2026-06-18)."""
    from vfe3.geometry import transport as T
    B, N, K = 8, 256, 16                                 # N large enough to be budget-limited (chunk < N)
    chunk = T._regime_ii_query_chunk(B, N, K)
    per_row = B * N * K * K
    assert chunk * per_row * T._REGIME_II_LIVE_TRANSIENTS <= T._REGIME_II_CHUNK_ELEMS
    assert 1 <= chunk < N                                # must actually chunk at this size


def test_regime_ii_covariant_omega_transforms_covariantly():
    """End-to-end gauge-covariance law: the assembled Omega satisfies Omega_ij -> g_i Omega_ij g_j^{-1}
    under a coherent per-token GL(K) frame change (phi co-transforms: exp(phi'_i) = g_i exp(phi_i)).
    Pins the wiki C3 claim end-to-end -- previously only edge-feature invariance + non-flatness were
    tested (audit 2026-06-18 coverage gap). Closed-form correct behavior, so a regression guard."""
    from vfe3.geometry.transport import get_transport, build_factored_transport
    build = get_transport("regime_ii_covariant")
    B, N, K, n_heads = 1, 4, 8, 2
    grp = get_group("block_glk")(K, n_heads)
    n_gen = grp.generators.shape[0]
    gen = torch.Generator().manual_seed(21)
    mu    = torch.randn(B, N, K, generator=gen)
    sigma = _spd(B, N, k=K)                               # FULL SPD (g Sigma g^T is non-diagonal)
    M = 0.3 * torch.randn(n_gen, 3, generator=gen)

    phi0 = torch.zeros(B, N, n_gen)                       # base frame phi=0
    a    = 0.2 * torch.randn(B, N, n_gen, generator=gen)  # per-token gauge a_i -> g_i = exp(a_i . G)
    fac  = build_factored_transport(a, grp)
    g, g_inv = fac.exp_phi, fac.exp_neg_phi               # (B,N,K,K) g_i, g_i^{-1}

    omega_base = build(phi0, grp, mu=mu, sigma=sigma, connection_M=M)["Omega"]
    mu_t    = torch.einsum("bnkl,bnl->bnk", g, mu)                       # g_i mu_i
    sigma_t = torch.einsum("bnkl,bnlm,bnpm->bnkp", g, sigma, g)          # g_i Sigma_i g_i^T
    omega_tr = build(a, grp, mu=mu_t, sigma=sigma_t, connection_M=M)["Omega"]

    expected = torch.einsum("bikl,bijlm,bjmn->bijkn", g, omega_base, g_inv)   # g_i Omega_ij g_j^{-1}
    assert torch.allclose(omega_tr, expected, atol=1e-4, rtol=1e-4), (
        f"covariance law violated: max abs diff {(omega_tr - expected).abs().max().item():.3e}")
    assert float(holonomy_deviation(omega_base[0])) > 1e-2               # genuinely curved (non-flat)


# --- audit 2026-07-01 C4: the oracle threads its own sigma leaves into the omega builder ----------
from vfe3.gradients.oracle import belief_gradients_autograd
from vfe3.inference.e_step import build_belief_transport


def _oracle_fixture(seed=0, B=1, N=4, K=8, n_heads=2):
    phi, mu, sigma, grp = _phi_mu_sigma(seed=seed, B=B, N=N, K=K, n_heads=n_heads)
    mu = 3.0 * mu                                        # inflate so the invariant features carry signal
    g = torch.Generator().manual_seed(seed + 100)
    mu_p    = torch.randn(B, N, K, generator=g)
    sigma_p = torch.rand(B, N, K, generator=g) + 0.5
    n_gen = grp.generators.shape[0]
    M = 0.3 * torch.randn(n_gen, 3, generator=g)
    return phi, mu, sigma, mu_p, sigma_p, grp, M


def test_covariant_detached_oracle_includes_sigma_grad():
    """C4: on the DETACHED oracle path (create_graph=False -- eval / diagnostics / no_grad E-step)
    the omega builder must receive the oracle's OWN sigma leaves, so grad_sigma carries
    d Omega/d sigma exactly as the live unrolled path does. Before the fix the builder closed
    over the belief covariance, so the detached path silently dropped it (probe gap ~0.37)."""
    phi, mu, sigma, mu_p, sigma_p, grp, M = _oracle_fixture(seed=12)

    def builder(mu_q, sigma_q, mu_k, sigma_k):
        return build_belief_transport(
            phi, grp, transport_mode="regime_ii_covariant",
            mu=mu_q, sigma=sigma_q, mu_key=mu_k, sigma_key=sigma_k, connection_M=M,
        )

    kw = dict(gradient_mode="filtering", irrep_dims=grp.irrep_dims, omega_builder=builder)
    # detached path: grad-free inputs -> the oracle clones its own leaves (create_graph=False)
    g_mu_det, g_sig_det = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, None, **kw)
    # live reference: create_graph=True with grad-carrying beliefs aliases the leaves, so
    # d Omega/d sigma is included by construction -- the ground truth the detached path must match
    mu_live = mu.clone().requires_grad_(True)
    sigma_live = sigma.clone().requires_grad_(True)
    g_mu_ref, g_sig_ref = belief_gradients_autograd(mu_live, sigma_live, mu_p, sigma_p, None,
                                                    create_graph=True, **kw)
    assert torch.allclose(g_sig_det, g_sig_ref.detach(), atol=1e-5, rtol=1e-4)
    assert torch.allclose(g_mu_det, g_mu_ref.detach(), atol=1e-5, rtol=1e-4)


def test_omega_builder_receives_four_tensor_args():
    """C4 contract pin: the oracle invokes omega_builder with 4 positional tensors
    (mu_q, sigma_q, mu_k, sigma_k); under filtering the key slots are detached and the query
    slots are the differentiation leaves."""
    phi, mu, sigma, mu_p, sigma_p, grp, M = _oracle_fixture(seed=13)
    calls = []

    def spy(mu_q, sigma_q, mu_k, sigma_k):
        calls.append((mu_q, sigma_q, mu_k, sigma_k))
        return build_belief_transport(
            phi, grp, transport_mode="regime_ii_covariant",
            mu=mu_q, sigma=sigma_q, mu_key=mu_k, sigma_key=sigma_k, connection_M=M,
        )

    belief_gradients_autograd(mu, sigma, mu_p, sigma_p, None,
                              gradient_mode="filtering", irrep_dims=grp.irrep_dims,
                              omega_builder=spy)
    assert len(calls) == 1
    mu_q, sigma_q, mu_k, sigma_k = calls[0]
    assert all(isinstance(t, torch.Tensor) for t in calls[0])
    assert mu_q.requires_grad and sigma_q.requires_grad             # differentiation leaves
    assert not mu_k.requires_grad and not sigma_k.requires_grad     # filtering: key slots frozen


# ===========================================================================
# PB-11 (Task 3, 2026-07-12): model channel shares the Route-B covariant connection.
#
# _gamma_energy builds the s-channel transport through regime_ii_covariant when configured,
# forwarding connection_M and (per the needs_mu + needs_sigma registration metadata) the
# s-channel means AND covariances -- so its edge features are channel-local to the s state.
# ===========================================================================

def _covariant_gamma_model(**kw):
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    # prior_source='token' (default) keeps the belief mu_embed table SEPARATE from the s tables, so the
    # channel-local "transport reads the s means/covariances, not the belief q" assertion is meaningful.
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, lambda_h=0.0, lambda_gamma=0.5,
                transport_mode="regime_ii_covariant")
    base.update(kw)
    torch.manual_seed(0)
    m = VFEModel(VFE3Config(**base))
    torch.manual_seed(7)
    with torch.no_grad():
        m.prior_bank.s_mu_embed.normal_(0.0, 0.5)
        m.prior_bank.s_sigma_log_embed.normal_(0.0, 0.3)
        m.prior_bank.mu_embed.normal_(0.0, 0.5)
        m.prior_bank.phi_embed.normal_(0.0, 0.2)
        m.connection_M.normal_(0.0, 0.4)                       # nonzero Route-B connection
    return m


def _covariant_e_s_ref(m, tok, phi, *, mu_state, sigma_state):
    from vfe3.families.base import get_family
    from vfe3.free_energy import pairwise_energy
    from vfe3.geometry.transport import (
        _TRANSPORT_NEEDS_MU, _TRANSPORT_NEEDS_SIGMA, transport_covariance, transport_mean,
    )
    cfg = m.cfg
    fam = get_family(cfg.family)
    s_mu, s_sigma = m.prior_bank.encode_s(tok)
    tm = cfg.transport_mode
    omega = build_belief_transport(
        phi, m.group, transport_mode=tm, gauge_parameterization="phi",
        mu=(mu_state if tm in _TRANSPORT_NEEDS_MU else None),
        sigma=(sigma_state if tm in _TRANSPORT_NEEDS_SIGMA else None),
        connection_M=getattr(m, "connection_M", None),
        link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
        cocycle_relaxation=cfg.cocycle_relaxation,
    )
    s_mu_t = transport_mean(omega, s_mu)
    s_sigma_t = transport_covariance(omega, s_sigma, diagonal_out=(s_sigma.dim() == s_mu.dim()))
    return pairwise_energy(fam(s_mu, s_sigma), fam(s_mu_t, s_sigma_t),
                           alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                           divergence_family=cfg.divergence_family, irrep_dims=m.group.irrep_dims)


def test_gamma_energy_covariant_reads_connection_M():
    m = _covariant_gamma_model()
    tok = torch.randint(0, 6, (2, 5))
    phi = m.prior_bank.encode(tok).phi
    e_s0 = m._gamma_energy(tok, phi)[0].clone()
    with torch.no_grad():
        m.connection_M.mul_(1.9)
    e_s1 = m._gamma_energy(tok, phi)[0]
    assert not torch.allclose(e_s0, e_s1)                       # s-channel now reads connection_M (was flat)


def test_gamma_energy_covariant_transport_reads_s_means_and_covariances():
    m = _covariant_gamma_model()
    tok = torch.randint(0, 6, (2, 5))
    phi = m.prior_bank.encode(tok).phi
    e_s = m._gamma_energy(tok, phi)[0]
    s_mu, s_sigma = m.prior_bank.encode_s(tok)
    enc = m.prior_bank.encode(tok)
    ref_s = _covariant_e_s_ref(m, tok, phi, mu_state=s_mu, sigma_state=s_sigma)
    torch.testing.assert_close(e_s, ref_s, rtol=1e-4, atol=1e-5)   # channel-local: reads the s state
    ref_q = _covariant_e_s_ref(m, tok, phi, mu_state=enc.mu, sigma_state=enc.sigma)
    assert not torch.allclose(e_s, ref_q)                      # ...not the belief (q) means/covariances


def test_gamma_energy_covariant_gradient_reaches_connection_M():
    m = _covariant_gamma_model()
    tok = torch.randint(0, 6, (2, 5))
    phi = m.prior_bank.encode(tok).phi
    m._gamma_energy(tok, phi)[0].sum().backward()
    assert m.connection_M.grad is not None
    assert float(m.connection_M.grad.abs().sum()) > 0.0
