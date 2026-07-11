r"""Regime-II DIRECT-LINK transport: ``regime_ii_link`` (bare) and ``regime_ii_link_charted``.

Two belief-independent, model-owned link modes over the SAME ``connection_L`` table
(shape ``(max_seq_len, max_seq_len, n_gen)``, zero-init), sliced to the active N and
exponentiated after a self-edge mask + embedded-matrix Frobenius soft cap:

  regime_ii_link (bare, default-off): Omega_ij = exp(link_alpha * A_ij . G).
    Reads ONLY connection_L -- no phi, no beliefs. Its flat limit is IDENTITY links
    (Omega=I), NOT the Regime-I vertex cocycle exp(phi_i)exp(-phi_j). It is
    frame-INDEPENDENT, so it does NOT satisfy the gauge-covariance law
    Omega_ij -> g_i Omega_ij g_j^{-1} -- a documented opt-in equivariance break.
    Unlike connection_W (exact at W=0, where regime_ii recovers the covariant flat
    cocycle), the bare link breaks for ALL connection_L: even the A=0 identity links
    satisfy I != g_i g_j^{-1}. Returns Omega at logical (N,N,K,K) (batch-independent,
    broadcast downstream) -- the D3 memory collapse.

  regime_ii_link_charted (opt-in): Omega_ij = exp(phi_i) exp(link_alpha * A_ij . G) exp(-phi_j).
    The co-transforming vertex frames carry the whole conjugation and the constant
    middle factor is insulated, so it IS EXACTLY gauge-covariant for ANY constant A
    (Omega_ij -> g_i Omega_ij g_j^{-1}). Belief-independent (kernel-eligible). Its A=0
    limit is the Regime-I flat cocycle exp(phi_i)exp(-phi_j). Forward inference
    retains the per-sequence vertex factors around the shared edge table and does
    not materialize a batched pairwise Omega; direct registry callers may request
    that dense compatibility representation explicitly.
"""

import math

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    _TRANSPORT_NEEDS_MU,
    _TRANSPORT_NEEDS_SIGMA,
    compute_transport_operators,
    get_transport,
)
from vfe3.metrics import holonomy_deviation


def _grp(k: int = 4, n_heads: int = 2):
    return get_group("block_glk")(k, n_heads)


def _inputs(seed: int = 0, *, B: int = 1, N: int = 4, K: int = 4, n_heads: int = 2):
    """phi (B,N,n_gen) gauge frames and connection_L ((N+2,N+2,n_gen), oversized to test slicing)."""
    grp = _grp(K, n_heads)
    g = torch.Generator().manual_seed(seed)
    n_gen = grp.generators.shape[0]
    phi = 0.3 * torch.randn(B, N, n_gen, generator=g)
    connection_l = 0.15 * torch.randn(N + 2, N + 2, n_gen, generator=g)
    return phi, connection_l, grp


def _identity_omega(n_tok: int, k: int, *, device, dtype):
    eye = torch.eye(k, device=device, dtype=dtype)
    return eye.expand(n_tok, n_tok, k, k).contiguous()


# --- registration + state-routing metadata --------------------------------------------------

def test_regime_ii_link_is_registered():
    assert callable(get_transport("regime_ii_link"))


def test_regime_ii_link_charted_is_registered():
    assert callable(get_transport("regime_ii_link_charted"))


def test_link_modes_have_no_state_routing_metadata():
    """Both link modes are belief-INDEPENDENT: no needs_mu / needs_sigma (keeps kernel eligibility)."""
    for mode in ("regime_ii_link", "regime_ii_link_charted"):
        assert mode not in _TRANSPORT_NEEDS_MU
        assert mode not in _TRANSPORT_NEEDS_SIGMA


# --- bare regime_ii_link: identity-link flat limit, zero-alpha, slicing, dict shape ----------

def test_regime_ii_link_connection_none_returns_identity_links():
    """connection_L=None -> IDENTITY links Omega=I (NOT the Regime-I flat cocycle)."""
    phi, _connection_l, grp = _inputs(seed=1)
    K = grp.generators.shape[-1]
    omega = get_transport("regime_ii_link")(phi, grp, connection_L=None)["Omega"]
    expected = _identity_omega(phi.shape[1], K, device=phi.device, dtype=phi.dtype)
    assert omega.shape == (phi.shape[1], phi.shape[1], K, K)          # (N,N,K,K), batch-independent
    assert torch.equal(omega, expected)


def test_regime_ii_link_zero_alpha_ignores_nonzero_table():
    """link_alpha=0 -> identity links for ANY connection_L."""
    phi, connection_l, grp = _inputs(seed=2)
    K = grp.generators.shape[-1]
    omega = get_transport("regime_ii_link")(phi, grp, connection_L=connection_l, link_alpha=0.0)["Omega"]
    expected = _identity_omega(phi.shape[1], K, device=phi.device, dtype=phi.dtype)
    assert torch.allclose(omega, expected, atol=1e-6, rtol=0.0)


def test_regime_ii_link_dict_shape_is_batch_independent():
    """The bare link returns a (N,N,K,K) Omega (batch-collapsed; broadcast downstream)."""
    phi, connection_l, grp = _inputs(seed=3, B=3, N=4)
    K = grp.generators.shape[-1]
    out = get_transport("regime_ii_link")(phi, grp, connection_L=connection_l, link_alpha=1.0)
    assert set(out) == {"exp_phi", "exp_neg_phi", "Omega"}
    assert out["Omega"].shape == (4, 4, K, K)                        # NO batch axis (D3 collapse)


def test_regime_ii_link_active_length_slicing():
    """connection_L is sliced to the active N; an oversized table covering N works."""
    phi, connection_l, grp = _inputs(seed=4, N=4)
    K = grp.generators.shape[-1]
    omega = get_transport("regime_ii_link")(phi, grp, connection_L=connection_l, link_alpha=1.0)["Omega"]
    assert omega.shape == (4, 4, K, K)


def test_regime_ii_link_raises_when_table_too_small():
    """A connection_L that does not cover the active N is a clear ValueError."""
    phi, _cl, grp = _inputs(seed=5, N=6)
    n_gen = grp.generators.shape[0]
    small = torch.zeros(3, 3, n_gen)                                 # < N=6
    with pytest.raises(ValueError):
        get_transport("regime_ii_link")(phi, grp, connection_L=small, link_alpha=1.0)


def test_regime_ii_link_self_edge_is_identity():
    """The link is an EDGE object: the self-edge carries no link factor -> Omega_ii = I."""
    phi, connection_l, grp = _inputs(seed=6, N=4)
    K = grp.generators.shape[-1]
    omega = get_transport("regime_ii_link")(phi, grp, connection_L=connection_l, link_alpha=1.0)["Omega"]
    idx = torch.arange(omega.shape[0])
    eye = torch.eye(K).expand(omega.shape[0], K, K)
    assert torch.allclose(omega[idx, idx], eye, atol=1e-6)


def test_regime_ii_link_nonzero_is_non_flat():
    """A nonzero connection_L gives non-trivial triangle holonomy (curvature > 0)."""
    phi, connection_l, grp = _inputs(seed=7, N=4)
    omega = get_transport("regime_ii_link")(phi, grp, connection_L=connection_l, link_alpha=1.0)["Omega"]
    assert float(holonomy_deviation(omega)) > 1e-2


# --- bare regime_ii_link gauge property: frame-independent, breaks covariance ----------------

def test_regime_ii_link_is_frame_independent_and_breaks_covariance():
    r"""The bare link Omega_ij = exp(alpha A_ij) reads ONLY connection_L, never the frame phi.
    So (1) rebuilding under a DIFFERENT frame gives an identical Omega (frame-independent), and
    (2) it therefore does NOT satisfy the gauge-covariance law Omega_ij -> g_i Omega_ij g_j^{-1}
    -- the documented opt-in equivariance break. Unlike connection_W (exact at W=0, where regime_ii
    recovers the covariant flat cocycle), the bare link breaks for ALL connection_L: even the A=0
    identity links satisfy I != g_i g_j^{-1}."""
    from vfe3.geometry.transport import build_factored_transport

    K, n_heads, N = 4, 2, 3
    phi, connection_l, grp = _inputs(seed=8, B=1, N=N, K=K, n_heads=n_heads)
    n_gen = grp.generators.shape[0]
    build = get_transport("regime_ii_link")

    # (1) frame-independence: a totally different frame gives the same link
    phi2 = phi + 0.5 * torch.randn_like(phi)
    om1 = build(phi, grp, connection_L=connection_l, link_alpha=1.0)["Omega"]
    om2 = build(phi2, grp, connection_L=connection_l, link_alpha=1.0)["Omega"]
    assert torch.equal(om1, om2)

    # (2) covariance break: per-token g_i = exp(a_i . G); the link does NOT transform as g_i Om g_j^{-1}
    a = 0.2 * torch.randn(1, N, n_gen, generator=torch.Generator().manual_seed(3))
    fac = build_factored_transport(a, grp)
    g, g_inv = fac.exp_phi[0], fac.exp_neg_phi[0]                    # (N,K,K)
    expected_cov = torch.einsum("ikl,ijlm,jmn->ijkn", g, om1, g_inv)
    assert not torch.allclose(om1, expected_cov, atol=1e-4)         # breaks the covariance law

    # break does NOT vanish at connection_L=0 (identity links are themselves non-covariant)
    om0 = build(phi, grp, connection_L=torch.zeros_like(connection_l), link_alpha=1.0)["Omega"]
    expected0 = torch.einsum("ikl,ijlm,jmn->ijkn", g, om0, g_inv)
    assert not torch.allclose(om0, expected0, atol=1e-4)


# --- charted regime_ii_link_charted: exact covariance, flat-cocycle A=0 limit, non-flat ------

def test_regime_ii_link_charted_dict_shape_is_batched():
    """A direct registry caller receives the explicit batched compatibility representation."""
    phi, connection_l, grp = _inputs(seed=10, B=2, N=4)
    K = grp.generators.shape[-1]
    out = get_transport("regime_ii_link_charted")(phi, grp, connection_L=connection_l, link_alpha=1.0)
    assert set(out) == {"exp_phi", "exp_neg_phi", "Omega"}
    assert out["Omega"].shape == (2, 4, 4, K, K)


def test_regime_ii_link_charted_zero_connection_reduces_to_flat_cocycle():
    """A=0 limit: exp(phi_i) exp(0) exp(-phi_j) = the Regime-I flat cocycle (NOT identity links)."""
    phi, connection_l, grp = _inputs(seed=11, B=2, N=4)
    flat = compute_transport_operators(phi, grp)["Omega"]
    out = get_transport("regime_ii_link_charted")(
        phi, grp, connection_L=torch.zeros_like(connection_l), link_alpha=1.0
    )["Omega"]
    assert torch.allclose(out, flat, atol=1e-6, rtol=0.0)


def test_regime_ii_link_charted_none_reduces_to_flat_cocycle():
    """connection_L=None also reduces to the flat cocycle byte-identically."""
    phi, _cl, grp = _inputs(seed=12, B=2, N=4)
    flat = compute_transport_operators(phi, grp)["Omega"]
    out = get_transport("regime_ii_link_charted")(phi, grp, connection_L=None)["Omega"]
    assert torch.allclose(out, flat, atol=1e-6, rtol=0.0)


def test_charted_trivial_gauge_preserves_edge_link():
    """Trivial vertex frames leave the direct edge factor instead of erasing the connection."""
    phi, connection_l, grp = _inputs(seed=121, B=2, N=4)
    connection_l.zero_()
    connection_l[0, 1, 0] = 0.25

    charted = get_transport("regime_ii_link_charted")(
        phi, grp, gauge_mode="trivial", connection_L=connection_l, link_alpha=1.0,
    )["Omega"]
    bare = get_transport("regime_ii_link")(
        phi, grp, gauge_mode="trivial", connection_L=connection_l, link_alpha=1.0,
    )["Omega"]
    expected = bare.unsqueeze(0).expand_as(charted)

    assert torch.allclose(charted, expected, atol=1e-6, rtol=0.0)
    assert not torch.allclose(
        charted[0, 0, 1], torch.eye(grp.generators.shape[-1]), atol=1e-6, rtol=0.0,
    )


def test_regime_ii_link_charted_nonzero_is_non_flat():
    """A nonzero connection_L gives non-trivial triangle holonomy under the charted sandwich."""
    phi, connection_l, grp = _inputs(seed=13, B=1, N=4)
    omega = get_transport("regime_ii_link_charted")(
        phi, grp, connection_L=connection_l, link_alpha=1.0
    )["Omega"][0]
    assert float(holonomy_deviation(omega)) > 1e-2


def test_regime_ii_link_charted_transforms_covariantly():
    r"""EXACT gauge covariance: Omega_ij -> g_i Omega_ij g_j^{-1} under a per-token frame change
    (exp(phi'_i) = g_i exp(phi_i)), for ANY constant connection_L and WITHOUT transforming beliefs
    (the charted middle factor reads nothing). Mirrors test_regime_ii_covariant's covariance law,
    but the charted link needs no belief transform."""
    from vfe3.geometry.transport import build_factored_transport
    build = get_transport("regime_ii_link_charted")
    B, N, K, n_heads = 1, 4, 8, 2
    grp = get_group("block_glk")(K, n_heads)
    n_gen = grp.generators.shape[0]
    gen = torch.Generator().manual_seed(21)
    connection_l = 0.3 * torch.randn(N, N, n_gen, generator=gen)

    phi0 = torch.zeros(B, N, n_gen)                                  # base frame phi=0
    a = 0.2 * torch.randn(B, N, n_gen, generator=gen)               # g_i = exp(a_i . G)
    fac = build_factored_transport(a, grp)
    g, g_inv = fac.exp_phi, fac.exp_neg_phi                          # (B,N,K,K)

    omega_base = build(phi0, grp, connection_L=connection_l, link_alpha=1.0)["Omega"]
    omega_tr = build(a, grp, connection_L=connection_l, link_alpha=1.0)["Omega"]
    expected = torch.einsum("bikl,bijlm,bjmn->bijkn", g, omega_base, g_inv)
    assert torch.allclose(omega_tr, expected, atol=1e-4, rtol=1e-4), (
        f"charted covariance law violated: max abs diff {(omega_tr - expected).abs().max().item():.3e}")
    assert float(holonomy_deviation(omega_base[0])) > 1e-2          # genuinely curved


# --- config: fields, validation, e_phi_lr gate, D2 oracle auto-enable ------------------------

def test_config_link_fields_defaults_and_validation():
    """link_alpha defaults 1.0 in [0,1]; link_soft_cap defaults 6.0, positive AND finite."""
    assert VFE3Config().link_alpha == 1.0
    assert VFE3Config().link_soft_cap == 6.0
    assert VFE3Config(transport_mode="regime_ii_link", link_alpha=0.0).link_alpha == 0.0
    assert VFE3Config(transport_mode="regime_ii_link", link_alpha=0.5).link_alpha == 0.5
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            VFE3Config(transport_mode="regime_ii_link", link_alpha=bad)
    for bad in (0.0, -1.0, float("nan"), float("inf")):                 # inf must be rejected (math.isfinite)
        with pytest.raises(ValueError):
            VFE3Config(transport_mode="regime_ii_link", link_soft_cap=bad)


def test_config_bare_link_rejects_e_phi_lr_positive():
    """The bare link is edge-owned and independent of the vertex frame phi: e_phi_lr>0 is rejected."""
    with pytest.raises(ValueError):
        VFE3Config(transport_mode="regime_ii_link", e_phi_lr=0.1)
    # e_phi_lr=0 is fine
    assert VFE3Config(transport_mode="regime_ii_link", e_phi_lr=0.0).e_phi_lr == 0.0


def test_config_charted_link_accepts_e_phi_lr_positive():
    """The charted sandwich IS phi-dependent, so a nonzero e_phi_lr is legitimate."""
    cfg = VFE3Config(transport_mode="regime_ii_link_charted", e_phi_lr=0.1)
    assert cfg.e_phi_lr == 0.1


def test_config_link_modes_conditional_oracle_auto_enable():
    """D2: both link modes are kernel-eligible on the canonical knobs (oracle_unroll_grad stays
    False), but a non-kernel-eligible config (gaussian_full) auto-enables the differentiable
    oracle so connection_L still trains. The flat pure path is unaffected."""
    for mode in ("regime_ii_link", "regime_ii_link_charted"):
        assert VFE3Config(transport_mode=mode).oracle_unroll_grad is False           # canonical: kernel
        assert VFE3Config(transport_mode=mode, family="gaussian_full",
                          oracle_unroll_grad=False).oracle_unroll_grad is True        # oracle route
    assert VFE3Config(transport_mode="flat").oracle_unroll_grad is False


def test_link_modes_kernel_eligible_at_canonical_knobs():
    """Neither link mode is excluded from the kernel route (they are belief-independent)."""
    from vfe3.gradients.kernels import uses_kernel_route
    base = dict(renyi_order=1.0, gradient_mode="filtering", family="gaussian_diagonal",
                divergence_family="renyi", include_attention_entropy=True)
    assert uses_kernel_route(**base, transport_mode="regime_ii_link")
    assert uses_kernel_route(**base, transport_mode="regime_ii_link_charted")


# --- model wiring: connection_L creation, init, gradient flow, optimizer, cache --------------
from vfe3.model.model import VFEModel


def _tiny_cfg(transport_mode="flat", e_phi_lr=0.0, **kw):
    return VFE3Config(
        vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
        n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=e_phi_lr, transport_mode=transport_mode, **kw,
    )


def test_model_link_modes_create_connection_l_zero_init():
    """Both link modes create connection_L as a zero-init (max_seq, max_seq, n_gen) Parameter."""
    for mode in ("regime_ii_link", "regime_ii_link_charted"):
        model = VFEModel(_tiny_cfg(transport_mode=mode))
        assert isinstance(model.connection_L, torch.nn.Parameter)
        n_gen = model.group.generators.shape[0]
        assert model.connection_L.shape == (4, 4, n_gen)
        assert torch.equal(model.connection_L, torch.zeros(4, 4, n_gen))


def test_model_flat_has_no_connection_l():
    """The default flat model carries no connection_L (pure path is param-free)."""
    assert not hasattr(VFEModel(_tiny_cfg(transport_mode="flat")), "connection_L")


def test_model_charted_init_equals_flat_forward():
    """The charted A=0 limit IS the flat cocycle, so at init (connection_L=0) a charted model's
    forward equals the flat model's. (The BARE link's init is identity links, a different transport,
    so it has no such equality -- pinned separately below.)"""
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    torch.manual_seed(0)
    flat_model = VFEModel(_tiny_cfg(transport_mode="flat"))
    logits_flat = flat_model(tokens)
    _, loss_flat, _ = flat_model(tokens, targets)
    torch.manual_seed(0)
    charted_model = VFEModel(_tiny_cfg(transport_mode="regime_ii_link_charted"))
    logits_ch = charted_model(tokens)
    _, loss_ch, _ = charted_model(tokens, targets)
    assert torch.allclose(logits_flat, logits_ch, atol=1e-5, rtol=0.0)
    assert torch.allclose(loss_flat, loss_ch, atol=1e-5, rtol=0.0)


def test_model_link_modes_nonzero_changes_forward():
    """A nonzero connection_L changes the forward loss for both link modes (the link is threaded
    into the live transport, not ignored)."""
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    for mode in ("regime_ii_link", "regime_ii_link_charted"):
        torch.manual_seed(0)
        model = VFEModel(_tiny_cfg(transport_mode=mode))
        with torch.no_grad():
            model.prior_bank.mu_embed *= 50.0
        _, loss0, _ = model(tokens, targets)
        with torch.no_grad():
            model.connection_L += 0.5 * torch.randn_like(model.connection_L)
        _, loss1, _ = model(tokens, targets)
        assert not torch.allclose(loss0, loss1, atol=1e-4), mode


def test_model_regime_ii_link_gradient_flows_to_l_on_default_kernel_route():
    """THE load-bearing test (F6): the bare link is belief-INDEPENDENT, so the closed-form kernel is
    valid and carries dF/dconnection_L with NO oracle. On the DEFAULT config (oracle_unroll_grad=False,
    the kernel route) loss.backward() must populate a finite, NONZERO off-diagonal connection_L.grad.
    (Unlike regime_ii, which needs oracle_unroll_grad=True because its kernel drops dOmega/dmu.)"""
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii_link"))
    assert model.cfg.oracle_unroll_grad is False                         # canonical kernel route
    with torch.no_grad():
        model.prior_bank.mu_embed *= 50.0
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    g = model.connection_L.grad
    assert g is not None and torch.isfinite(g).all()
    N = tokens.shape[1]
    offdiag = ~torch.eye(N, dtype=torch.bool)
    assert g[:N, :N][offdiag].abs().sum() > 1e-6                          # off-diagonal link trains


def test_model_charted_gradient_flows_to_l_on_default_kernel_route():
    """The charted link is also belief-independent (phi-dependent but not belief-dependent), so it is
    kernel-eligible and trains connection_L on the default route."""
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii_link_charted"))
    assert model.cfg.oracle_unroll_grad is False
    with torch.no_grad():
        model.prior_bank.mu_embed *= 50.0
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    g = model.connection_L.grad
    assert g is not None and torch.isfinite(g).all()
    N = tokens.shape[1]
    offdiag = ~torch.eye(N, dtype=torch.bool)
    assert g[:N, :N][offdiag].abs().sum() > 1e-6


def test_build_optimizer_groups_connection_l_once():
    """connection_L must land in exactly one optimizer param group (else the coverage guard raises
    and the link would never train)."""
    from vfe3.train import build_optimizer
    for mode in ("regime_ii_link", "regime_ii_link_charted"):
        model = VFEModel(_tiny_cfg(transport_mode=mode))
        opt = build_optimizer(model, model.cfg)                          # raises if ungrouped
        grouped = [p for grp in opt.param_groups for p in grp["params"]]
        assert sum(p is model.connection_L for p in grouped) == 1, mode


def test_charted_model_runs_with_e_phi_lr_positive():
    """The charted sandwich is phi-dependent, so e_phi_lr>0 (a live phi E-step) is legitimate and the
    forward runs end-to-end (exercises phi_alignment_loss's link forwarding)."""
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii_link_charted", e_phi_lr=0.05))
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)


def test_cache_rejects_link_modes():
    """The prefix belief cache is flat-only; both link modes are rejected (non-flat transport)."""
    from vfe3.inference.belief_cache import cache_supported
    for mode in ("regime_ii_link", "regime_ii_link_charted"):
        assert not cache_supported(_tiny_cfg(transport_mode=mode)), mode


# --- gradient + kernel verification ----------------------------------------------------------
from vfe3.belief import BeliefState
from vfe3.inference.e_step import build_belief_transport, free_energy_value


def _grad_setup(seed=0, N=3, K=4):
    grp = get_group("block_glk")(K, 2)
    n_gen = grp.generators.shape[0]
    g = torch.Generator().manual_seed(seed)
    mu      = 0.5 * torch.randn(N, K, generator=g)
    sigma   = torch.rand(N, K, generator=g) + 0.5
    phi     = 0.1 * torch.randn(N, n_gen, generator=g)
    mu_p    = 0.5 * torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    L       = 0.3 * torch.randn(N, N, n_gen, generator=g)
    return grp, mu, sigma, phi, mu_p, sigma_p, L


def test_regime_ii_link_df_dconnection_l_matches_fd():
    """dF/dconnection_L against central differences -- transport-DIFFERENTIABILITY of F (a necessary
    building block, NOT the M-step gradient, which flows through the unrolled E-step). Off-diagonal
    (i != j) entries only; the self-edge is masked, so diagonal entries are zero-gradient by design."""
    grp, mu, sigma, phi, mu_p, sigma_p, L = _grad_setup(seed=3)

    def F(l):
        return free_energy_value(BeliefState(mu=mu, sigma=sigma, phi=phi), mu_p, sigma_p, grp,
                                 transport_mode="regime_ii_link", connection_L=l, link_alpha=1.0)

    L_leaf = L.clone().requires_grad_(True)
    (g_l,) = torch.autograd.grad(F(L_leaf), L_leaf)
    assert torch.isfinite(g_l).all()
    assert g_l.abs().sum() > 1e-6
    h = 1e-3
    for (i, j, a) in ((0, 1, 0), (1, 2, 1), (2, 0, 2)):
        e = torch.zeros_like(L); e[i, j, a] = h
        fd = (F(L + e) - F(L - e)) / (2.0 * h)
        assert abs(float(g_l[i, j, a]) - float(fd)) <= 0.05 * abs(float(fd)) + 5e-3


def test_regime_ii_link_self_edge_grad_is_zero():
    """The masked self-edge carries no link, so dF/dconnection_L[i,i,:] = 0 exactly (by design)."""
    grp, mu, sigma, phi, mu_p, sigma_p, L = _grad_setup(seed=4)
    L_leaf = L.clone().requires_grad_(True)
    (g_l,) = torch.autograd.grad(
        free_energy_value(BeliefState(mu=mu, sigma=sigma, phi=phi), mu_p, sigma_p, grp,
                          transport_mode="regime_ii_link", connection_L=L_leaf, link_alpha=1.0),
        L_leaf)
    idx = torch.arange(L.shape[0])
    assert torch.equal(g_l[idx, idx], torch.zeros_like(g_l[idx, idx]))


def test_regime_ii_link_kernel_matches_oracle_for_fixed_transport():
    """The bare link is belief-INDEPENDENT (dOmega/dmu = 0), so the closed-form KERNEL (which treats
    the transported keys as constant in mu -- exactly correct here) matches the autograd ORACLE for
    the SAME fixed transport. This is the contrast with regime_ii, whose kernel drops dOmega/dmu and
    is therefore excluded from the kernel route."""
    from vfe3.gradients.kernels import belief_gradients
    from vfe3.gradients.oracle import belief_gradients_autograd
    grp, mu, sigma, phi, mu_p, sigma_p, L = _grad_setup(seed=5)
    omega = build_belief_transport(phi.unsqueeze(0), grp, transport_mode="regime_ii_link",
                                   connection_L=L, link_alpha=1.0)            # (N,N,K,K), batch-independent
    g_k, gs_k = belief_gradients(mu, sigma, mu_p, sigma_p, omega,
                                 transport_mode="regime_ii_link", gradient_mode="filtering",
                                 irrep_dims=grp.irrep_dims)
    g_o, gs_o = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                          gradient_mode="filtering", irrep_dims=grp.irrep_dims)
    assert torch.allclose(g_k, g_o, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gs_k, gs_o, atol=1e-5, rtol=1e-4)


# --- diagnostics threading: connection_L must reach the holonomy / norm diagnostics --------------

def test_diagnostics_holonomy_reflects_connection_l():
    """model.diagnostics builds its holonomy Omega under the ACTIVE regime; a nonzero connection_L
    must produce non-trivial curvature (holonomy_deviation > 0) and emit connection_l_norm. Before
    the threading fix the diagnostic Omega fell back to flat/identity (connection_L unpassed)."""
    for mode in ("regime_ii_link", "regime_ii_link_charted"):
        torch.manual_seed(0)
        model = VFEModel(_tiny_cfg(transport_mode=mode))
        with torch.no_grad():
            model.prior_bank.mu_embed *= 50.0
            model.connection_L += 0.5 * torch.randn_like(model.connection_L)
        d = model.diagnostics(torch.randint(0, 15, (1, 4)))
        assert d["holonomy_deviation"] > 1e-4, mode
        assert d["connection_l_norm"] > 0.0, mode
        assert "connection_l_offdiag_norm" in d, mode


def test_converged_state_omega_reflects_connection_l():
    """viz.extract.converged_state returns the diagnostic Omega; under a link mode it must be built
    with connection_L (the holonomy / gauge-equivariance figure panels otherwise read flat)."""
    from vfe3.viz.extract import converged_state
    for mode in ("regime_ii_link", "regime_ii_link_charted"):
        torch.manual_seed(0)
        model = VFEModel(_tiny_cfg(transport_mode=mode))
        with torch.no_grad():
            model.prior_bank.mu_embed *= 50.0
            model.connection_L += 0.5 * torch.randn_like(model.connection_L)
        st = converged_state(model, torch.randint(0, 15, (1, 4)))
        assert float(holonomy_deviation(st["omega"])) > 1e-4, mode


# --- numerical / performance guardrails (D3 memory, AMP) -------------------------------------

@pytest.mark.parametrize("mode", ["regime_ii_link", "regime_ii_link_charted"])
def test_direct_link_factored_mean_matches_dense_reference(mode):
    from vfe3.geometry.transport import DirectLinkTransport, transport_mean
    from vfe3.inference.e_step import build_belief_transport

    phi, connection_l, grp = _inputs(seed=19, B=2, N=4)
    mu = torch.randn(2, 4, grp.generators.shape[-1], generator=torch.Generator().manual_seed(190))
    factored = build_belief_transport(
        phi,
        grp,
        transport_mode=mode,
        connection_L=connection_l,
        link_alpha=1.0,
    )

    assert isinstance(factored, DirectLinkTransport)
    dense = factored.to_dense_omega()
    assert torch.allclose(
        transport_mean(factored, mu),
        transport_mean(dense, mu),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("mode", ["regime_ii_link", "regime_ii_link_charted"])
def test_direct_link_factored_diagonal_covariance_matches_dense_reference(mode):
    from vfe3.geometry.transport import DirectLinkTransport, transport_covariance
    from vfe3.inference.e_step import build_belief_transport

    phi, connection_l, grp = _inputs(seed=20, B=2, N=4)
    sigma = torch.rand(2, 4, grp.generators.shape[-1], generator=torch.Generator().manual_seed(200)) + 0.5
    factored = build_belief_transport(
        phi,
        grp,
        transport_mode=mode,
        connection_L=connection_l,
        link_alpha=1.0,
    )

    assert isinstance(factored, DirectLinkTransport)
    dense = factored.to_dense_omega()
    assert torch.allclose(
        transport_covariance(factored, sigma, diagonal_out=True),
        transport_covariance(dense, sigma, diagonal_out=True),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("mode", ["regime_ii_link", "regime_ii_link_charted"])
def test_direct_link_factored_full_covariance_matches_dense_reference(mode):
    from vfe3.geometry.transport import DirectLinkTransport, transport_covariance
    from vfe3.inference.e_step import build_belief_transport

    phi, connection_l, grp = _inputs(seed=21, B=2, N=4)
    K = grp.generators.shape[-1]
    raw = torch.randn(2, 4, K, K, generator=torch.Generator().manual_seed(210))
    sigma = raw @ raw.transpose(-1, -2) + 0.25 * torch.eye(K)
    factored = build_belief_transport(
        phi,
        grp,
        transport_mode=mode,
        connection_L=connection_l,
        link_alpha=1.0,
    )

    assert isinstance(factored, DirectLinkTransport)
    dense = factored.to_dense_omega()
    assert torch.allclose(
        transport_covariance(factored, sigma, diagonal_out=False),
        transport_covariance(dense, sigma, diagonal_out=False),
        atol=1e-5,
        rtol=1e-5,
    )


def test_charted_direct_link_does_not_materialize_dense_pair_transport(monkeypatch):
    from vfe3.geometry.transport import DirectLinkTransport, transport_covariance, transport_mean
    from vfe3.inference.e_step import build_belief_transport

    phi, connection_l, grp = _inputs(seed=22, B=2, N=4)
    K = grp.generators.shape[-1]
    mu = torch.randn(2, 4, K, generator=torch.Generator().manual_seed(220))
    sigma_diag = torch.rand(2, 4, K, generator=torch.Generator().manual_seed(221)) + 0.5
    raw = torch.randn(2, 4, K, K, generator=torch.Generator().manual_seed(222))
    sigma_full = raw @ raw.transpose(-1, -2) + 0.25 * torch.eye(K)

    def _forbid_dense(_self):
        raise AssertionError("charted E-step transport materialized dense pairwise Omega")

    monkeypatch.setattr(DirectLinkTransport, "to_dense_omega", _forbid_dense)
    factored = build_belief_transport(
        phi,
        grp,
        transport_mode="regime_ii_link_charted",
        connection_L=connection_l,
        link_alpha=1.0,
    )

    assert isinstance(factored, DirectLinkTransport)
    transport_mean(factored, mu)
    transport_covariance(factored, sigma_diag, diagonal_out=True)
    transport_covariance(factored, sigma_full, diagonal_out=False)


def test_bare_direct_link_skips_vertex_exponentials(monkeypatch):
    from vfe3.geometry import transport
    from vfe3.inference.e_step import build_belief_transport

    phi, connection_l, grp = _inputs(seed=23, B=3, N=4)

    def _forbid_vertex_build(*args, **kwargs):
        raise AssertionError("bare direct link built unused vertex exponentials")

    monkeypatch.setattr(transport, "build_factored_transport", _forbid_vertex_build)
    factored = build_belief_transport(
        phi,
        grp,
        transport_mode="regime_ii_link",
        connection_L=connection_l,
        link_alpha=1.0,
    )

    assert isinstance(factored, transport.DirectLinkTransport)
    assert factored.exp_phi is None
    assert factored.exp_neg_phi is None
    assert factored.exp_link.shape == (4, 4, grp.generators.shape[-1], grp.generators.shape[-1])


@pytest.mark.parametrize("mode", ["regime_ii_link", "regime_ii_link_charted"])
def test_direct_link_factored_preserves_reflection_and_rope(mode, monkeypatch):
    from vfe3.geometry.transport import DirectLinkTransport, RopeTransport, transport_covariance, transport_mean
    from vfe3.inference.e_step import build_belief_transport

    phi, connection_l, grp = _inputs(seed=24, B=2, N=4)
    K = grp.generators.shape[-1]
    reflection = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
    theta = 0.35
    rope = torch.eye(K).expand(4, K, K).clone()
    rope[:, 0, 0] = math.cos(theta)
    rope[:, 0, 1] = -math.sin(theta)
    rope[:, 1, 0] = math.sin(theta)
    rope[:, 1, 1] = math.cos(theta)
    mu = torch.randn(2, 4, K, generator=torch.Generator().manual_seed(240))
    sigma_diag = torch.rand(2, 4, K, generator=torch.Generator().manual_seed(242)) + 0.5
    raw = torch.randn(2, 4, K, K, generator=torch.Generator().manual_seed(241))
    sigma = raw @ raw.transpose(-1, -2) + 0.25 * torch.eye(K)

    wrapped = build_belief_transport(
        phi,
        grp,
        transport_mode=mode,
        connection_L=connection_l,
        link_alpha=1.0,
        reflection=reflection,
        rope=rope,
        rope_on_cov=True,
    )

    assert isinstance(wrapped, RopeTransport)
    assert isinstance(wrapped.base, DirectLinkTransport)
    dense_base = wrapped.base.to_dense_omega()
    dense_rope = torch.einsum("ikl,...ijlm,jnm->...ijkn", rope, dense_base, rope)

    def _forbid_dense(_self):
        raise AssertionError("RoPE/reflection direct-link path materialized dense pairwise Omega")

    monkeypatch.setattr(DirectLinkTransport, "to_dense_omega", _forbid_dense)
    assert torch.allclose(
        transport_mean(wrapped, mu),
        transport_mean(dense_rope, mu),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        transport_covariance(wrapped, sigma_diag, diagonal_out=True),
        transport_covariance(dense_rope, sigma_diag, diagonal_out=True),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        transport_covariance(wrapped, sigma, diagonal_out=False),
        transport_covariance(dense_rope, sigma, diagonal_out=False),
        atol=1e-5,
        rtol=1e-5,
    )


def test_regime_ii_link_build_belief_transport_is_batch_collapsed():
    """The D3 collapse is realized end-to-end without a batched or pairwise compatibility copy."""
    from vfe3.geometry.transport import DirectLinkTransport
    from vfe3.inference.e_step import build_belief_transport
    phi, connection_l, grp = _inputs(seed=20, B=8, N=4)
    K = grp.generators.shape[-1]
    omega = build_belief_transport(phi, grp, transport_mode="regime_ii_link",
                                   connection_L=connection_l, link_alpha=1.0)
    assert isinstance(omega, DirectLinkTransport)
    assert omega.exp_phi is None and omega.exp_neg_phi is None
    assert omega.exp_link.shape == (4, 4, K, K)                           # NOT (8,4,4,K,K)
    assert omega.to_dense_omega().shape == (4, 4, K, K)                   # explicit compatibility only


def test_direct_link_dense_bytes_estimator():
    """The dense (B,N,N,K,K) byte estimate is the compatibility cost both factored links avoid."""
    from vfe3.geometry.transport import _direct_link_dense_bytes
    assert _direct_link_dense_bytes(64, 128, 64, torch.float32) == 64 * 128 * 128 * 64 * 64 * 4


def test_config_warns_amp_with_link_modes():
    """amp_dtype with a link mode warns about the downstream-sandwich precision (spec bf16 ban)."""
    import warnings
    for mode in ("regime_ii_link", "regime_ii_link_charted"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            VFE3Config(transport_mode=mode, amp_dtype="bf16")
        assert any("amp_dtype" in str(x.message) for x in w), mode


def test_regime_ii_link_self_edge_stays_identity_after_perturbing_diagonal():
    """The self-edge is masked inside the builder, so even a nonzero connection_L[i,i] leaves
    Omega_ii = I (the diagonal can never inject a spurious self-energy)."""
    phi, connection_l, grp = _inputs(seed=21, N=4)
    K = grp.generators.shape[-1]
    connection_l[torch.arange(4), torch.arange(4)] = 3.0                 # perturb the (would-be) self-edges
    omega = get_transport("regime_ii_link")(phi, grp, connection_L=connection_l, link_alpha=1.0)["Omega"]
    idx = torch.arange(4)
    assert torch.allclose(omega[idx, idx], torch.eye(K).expand(4, K, K), atol=1e-6)
