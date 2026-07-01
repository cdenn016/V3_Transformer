r"""Regime-II edge-relaxed (non-flat) transport: oracle-backed tests.

The Regime-II builder ``_build_regime_ii`` realizes the edge-relaxed cocycle
(spec eq:edge_relaxed_omega)

    Omega_ij = exp(phi_i . G) exp(delta_ij . G) exp(-phi_j . G),
    delta_ij^a = cocycle_relaxation * (mu_i^T W^a mu_j),

with ``W`` the learned bilinear connection (an nn.Parameter on the model -- a
DOCUMENTED neural-network exception, default-off; the flat Regime-I builder is the
default and pure path). At ``W=0`` (init) OR ``cocycle_relaxation=0`` the edge factor
exp(delta) = I, so Omega reduces to the flat cocycle exp(phi_i) exp(-phi_j) EXACTLY --
the core oracle. A nonzero W gives non-trivial holonomy (curvature > 0).
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    compute_transport_operators,
    get_transport,
)
from vfe3.metrics import holonomy_deviation
from vfe3.model.model import VFEModel


def _phi_mu(seed=0, B=2, N=3, K=4, group="block_glk", n_heads=2):
    grp = get_group(group)(K, n_heads) if group in ("block_glk", "tied_block_glk") else get_group(group)(K)
    g = torch.Generator().manual_seed(seed)
    n_gen = grp.generators.shape[0]
    phi = 0.3 * torch.randn(B, N, n_gen, generator=g)
    mu = torch.randn(B, N, K, generator=g)
    return phi, mu, grp


# --- the core oracle: W=0 (or cocycle_relaxation=0) reduces to the flat cocycle ---
def test_regime_ii_is_registered():
    assert callable(get_transport("regime_ii"))


def test_regime_ii_w_zero_reduces_to_flat():
    """W=0 -> delta=0 -> exp(delta)=I -> Omega = exp(phi_i)exp(-phi_j) = flat cocycle."""
    phi, mu, grp = _phi_mu(seed=1)
    K, n_gen = grp.generators.shape[-1], grp.generators.shape[0]
    W = torch.zeros(n_gen, K, K)
    flat = compute_transport_operators(phi, grp)["Omega"]
    r2 = get_transport("regime_ii")(phi, grp, mu=mu, connection_W=W, cocycle_relaxation=1.0)["Omega"]
    assert r2.shape == flat.shape
    assert torch.allclose(r2, flat, atol=1e-6, rtol=0.0)


def test_regime_ii_cocycle_relaxation_zero_reduces_to_flat():
    """cocycle_relaxation=0 zeroes delta for ANY W -> flat cocycle exactly."""
    phi, mu, grp = _phi_mu(seed=2)
    K, n_gen = grp.generators.shape[-1], grp.generators.shape[0]
    g = torch.Generator().manual_seed(99)
    W = 0.5 * torch.randn(n_gen, K, K, generator=g)              # arbitrary nonzero W
    flat = compute_transport_operators(phi, grp)["Omega"]
    r2 = get_transport("regime_ii")(phi, grp, mu=mu, connection_W=W, cocycle_relaxation=0.0)["Omega"]
    assert torch.allclose(r2, flat, atol=1e-6, rtol=0.0)


def test_regime_ii_connection_none_reduces_to_flat():
    """connection_W=None (the un-threaded default) also reduces to flat."""
    phi, mu, grp = _phi_mu(seed=3)
    flat = compute_transport_operators(phi, grp)["Omega"]
    r2 = get_transport("regime_ii")(phi, grp, mu=mu, connection_W=None, cocycle_relaxation=1.0)["Omega"]
    assert torch.allclose(r2, flat, atol=1e-6, rtol=0.0)


def test_regime_ii_returns_flat_dict_shape():
    """The builder returns the SAME dict keys/shapes as flat (drop-in Omega producer)."""
    phi, mu, grp = _phi_mu(seed=4)
    K, n_gen = grp.generators.shape[-1], grp.generators.shape[0]
    W = 0.2 * torch.randn(n_gen, K, K)
    out = get_transport("regime_ii")(phi, grp, mu=mu, connection_W=W, cocycle_relaxation=1.0)
    assert set(out) == {"exp_phi", "exp_neg_phi", "Omega"}
    B, N = phi.shape[0], phi.shape[1]
    assert out["exp_phi"].shape == (B, N, K, K)
    assert out["exp_neg_phi"].shape == (B, N, K, K)
    assert out["Omega"].shape == (B, N, N, K, K)


# --- genuinely non-flat when W != 0: Omega differs and holonomy is strictly positive ---
def test_regime_ii_nonzero_w_is_non_flat():
    """A nonzero W makes Omega_ij differ from the flat exp(phi_i)exp(-phi_j) for some i!=j."""
    phi, mu, grp = _phi_mu(seed=5)
    K, n_gen = grp.generators.shape[-1], grp.generators.shape[0]
    g = torch.Generator().manual_seed(7)
    W = 0.4 * torch.randn(n_gen, K, K, generator=g)
    flat = compute_transport_operators(phi, grp)["Omega"]
    r2 = get_transport("regime_ii")(phi, grp, mu=mu, connection_W=W, cocycle_relaxation=1.0)["Omega"]
    # off-diagonal (i != j) edges carry delta != 0, so the transport genuinely differs
    assert not torch.allclose(r2[:, 0, 1], flat[:, 0, 1], atol=1e-3)


def test_regime_ii_holonomy_strictly_positive():
    """The independent curvature oracle: a flat cocycle gives holonomy ~0; a nonzero
    connection W gives the non-trivial triangle holonomy the diagnostic was built for."""
    phi, mu, grp = _phi_mu(seed=6, B=1, N=4)
    K, n_gen = grp.generators.shape[-1], grp.generators.shape[0]
    g = torch.Generator().manual_seed(11)
    W = 0.5 * torch.randn(n_gen, K, K, generator=g)
    omega_flat = compute_transport_operators(phi, grp)["Omega"][0]            # (N,N,K,K)
    omega_r2 = get_transport("regime_ii")(
        phi, grp, mu=mu, connection_W=W, cocycle_relaxation=1.0
    )["Omega"][0]
    assert float(holonomy_deviation(omega_flat)) < 1e-4                       # flat closes
    assert float(holonomy_deviation(omega_r2)) > 1e-2                         # regime II does not


# --- homotopy: cocycle_relaxation interpolates flat (alpha=0) to fully relaxed (alpha=1) ---
def test_regime_ii_cocycle_relaxation_homotopy():
    """Omega(alpha) = exp(phi_i) exp(alpha delta . G) exp(-phi_j) interpolates flat (alpha=0)
    to fully relaxed (alpha=1). At alpha=0.5 with a fixed W, the edge factor uses exactly half
    the connection -- pinned against an independent build that pre-scales W by 0.5 at alpha=1."""
    phi, mu, grp = _phi_mu(seed=9, B=1, N=3)
    K, n_gen = grp.generators.shape[-1], grp.generators.shape[0]
    g = torch.Generator().manual_seed(17)
    W = 0.4 * torch.randn(n_gen, K, K, generator=g)
    build = get_transport("regime_ii")
    flat = compute_transport_operators(phi, grp)["Omega"]
    half = build(phi, grp, mu=mu, connection_W=W, cocycle_relaxation=0.5)["Omega"]
    full = build(phi, grp, mu=mu, connection_W=W, cocycle_relaxation=1.0)["Omega"]
    # alpha=0.5 sits strictly between flat and the fully-relaxed transport
    assert not torch.allclose(half, flat, atol=1e-3)
    assert not torch.allclose(half, full, atol=1e-3)
    # alpha=0.5 with W equals alpha=1.0 with W/2 (the homotopy scales delta = alpha * mu^T W mu)
    full_halfW = build(phi, grp, mu=mu, connection_W=0.5 * W, cocycle_relaxation=1.0)["Omega"]
    assert torch.allclose(half, full_halfW, atol=1e-6, rtol=0.0)


# --- model wiring: init-flat, gradient-to-W, init-flat == flat ---
def _tiny_cfg(transport_mode="flat", cocycle_relaxation=1.0, **kw):
    return VFE3Config(
        vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
        n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.0,
        transport_mode=transport_mode, cocycle_relaxation=cocycle_relaxation, **kw,
    )


def test_model_regime_ii_creates_connection_w_zero_init():
    """transport_mode='regime_ii' creates connection_W as a zero-init nn.Parameter."""
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii"))
    assert hasattr(model, "connection_W")
    assert isinstance(model.connection_W, torch.nn.Parameter)
    n_gen = model.group.generators.shape[0]
    K = model.cfg.embed_dim
    assert model.connection_W.shape == (n_gen, K, K)
    assert torch.equal(model.connection_W, torch.zeros(n_gen, K, K))


def test_model_flat_has_no_connection_w():
    """The default flat model carries no connection_W parameter (pure path is param-free)."""
    model = VFEModel(_tiny_cfg(transport_mode="flat"))
    assert not hasattr(model, "connection_W")


def test_model_init_flat_equals_flat_forward():
    """At init (W=0), a regime_ii model's forward loss/logits equal the flat model's
    (init-flat == flat). Seed BOTH constructions identically so the PriorBank tables match."""
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))

    torch.manual_seed(0)
    flat_model = VFEModel(_tiny_cfg(transport_mode="flat"))
    logits_flat, loss_flat, _ = flat_model(tokens, targets)

    torch.manual_seed(0)
    r2_model = VFEModel(_tiny_cfg(transport_mode="regime_ii"))
    logits_r2, loss_r2, _ = r2_model(tokens, targets)

    assert torch.allclose(logits_flat, logits_r2, atol=1e-6, rtol=0.0)
    assert torch.allclose(loss_flat, loss_r2, atol=1e-6, rtol=0.0)


def test_model_regime_ii_gradient_flows_to_w():
    """connection_W enters the loss only through the E-step belief updates; with
    detach_e_step=False (default) loss.backward() populates a finite, NONZERO W.grad.

    W is zero-init (init-flat), so to get a nonzero gradient the connection must actually
    influence the loss at W=0 -- it does, because d Omega / d W at W=0 is the generator
    structure (exp'(0) = I), not zero. (If the gradient were zero at W=0 the parameter would
    never train; this pins that it does.) The PriorBank means are inflated so the QUADRATIC
    bilinear delta = mu_i^T W^a mu_j carries real signal (at the near-zero default init the
    quadratic delta, hence its gradient, is vanishingly small).

    oracle_unroll_grad=True is REQUIRED since the audit 2026-06-10 F1 reroute: regime_ii's
    belief gradient is served by the autograd oracle (the kernel is the flat-transport gradient),
    and only the differentiable (use_live) oracle keeps the unrolled chain to W. The config-time
    freeze warning pins the inverse case."""
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii", oracle_unroll_grad=True))
    with torch.no_grad():                                                    # non-tiny means -> non-vacuous bilinear
        model.prior_bank.mu_embed *= 50.0
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.connection_W.grad is not None
    assert torch.isfinite(model.connection_W.grad).all()
    assert model.connection_W.grad.abs().sum() > 1e-4                        # genuinely in the graph


def test_model_regime_ii_nonzero_w_changes_forward():
    """After setting W != 0, the regime_ii forward loss differs from the init-flat forward
    (the connection actually does something in the full model, not just the builder). Means are
    inflated so the quadratic-in-mu bilinear connection has signal."""
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii"))
    with torch.no_grad():
        model.prior_bank.mu_embed *= 50.0
    _, loss_flat, _ = model(tokens, targets)
    with torch.no_grad():
        model.connection_W += 0.5 * torch.randn_like(model.connection_W)
    _, loss_nonflat, _ = model(tokens, targets)
    assert not torch.allclose(loss_flat, loss_nonflat, atol=1e-4)


def test_model_diagnostics_holonomy_tracks_regime():
    """diagnostics() uses the forward's transport regime, so holonomy_deviation reads the ACTUAL
    connection: ~0 for a fresh (W=0) regime_ii model (init-flat), strictly > 0 once W != 0 (the
    end-to-end version of the curvature oracle; means inflated for the quadratic bilinear)."""
    tokens = torch.randint(0, 15, (1, 4))
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii"))
    with torch.no_grad():
        model.prior_bank.mu_embed *= 50.0
    diag_flat = model.diagnostics(tokens)
    assert diag_flat["holonomy_deviation"] < 1e-3                             # W=0 -> init-flat -> closes
    with torch.no_grad():
        model.connection_W += 0.5 * torch.randn_like(model.connection_W)
    diag_r2 = model.diagnostics(tokens)
    assert diag_r2["holonomy_deviation"] > 1e-2                               # nonzero W -> curvature


def test_config_cocycle_relaxation_default_and_validated():
    """cocycle_relaxation defaults to 1.0, accepts the [0,1] homotopy range, and rejects
    out-of-range / non-finite values at construction (it feeds the regime_ii connection directly)."""
    assert VFE3Config().cocycle_relaxation == 1.0
    assert VFE3Config(transport_mode="regime_ii", cocycle_relaxation=0.0).cocycle_relaxation == 0.0
    assert VFE3Config(transport_mode="regime_ii", cocycle_relaxation=0.5).cocycle_relaxation == 0.5
    assert VFE3Config(transport_mode="regime_ii", cocycle_relaxation=1.0).cocycle_relaxation == 1.0
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            VFE3Config(transport_mode="regime_ii", cocycle_relaxation=bad)


from vfe3.belief import BeliefState
from vfe3.inference.e_step import e_step_iteration


def test_phi_estep_descends_regime_ii_not_flat():
    # The phi E-step must build its Omega with the ACTIVE transport_mode. Under regime_ii + e_phi_lr>0
    # with a nonzero learned connection, the regime_ii phi update differs from the flat one; the bug
    # (phi_alignment_loss always built the flat Omega) made the phi output independent of
    # transport_mode. e_q_mu_lr=e_q_sigma_lr=0 isolates the phi step; W != 0 makes regime_ii != flat.
    grp = get_group("block_glk")(6, 2)
    n_gen = grp.generators.shape[0]
    torch.manual_seed(0)
    N, K = 4, 6
    belief = BeliefState(mu=torch.randn(N, K), sigma=torch.rand(N, K) + 0.5,
                         phi=0.1 * torch.randn(N, n_gen))
    mu_p, sigma_p = torch.randn(N, K), torch.rand(N, K) + 0.5
    W = 0.5 * torch.randn(n_gen, K, K)                 # nonzero connection -> regime_ii != flat
    kw = dict(e_q_mu_lr=0.0, e_q_sigma_lr=0.0, e_phi_lr=0.1, connection_W=W, cocycle_relaxation=1.0)
    out_flat = e_step_iteration(belief, mu_p, sigma_p, grp, transport_mode="flat", **kw)
    out_rii = e_step_iteration(belief, mu_p, sigma_p, grp, transport_mode="regime_ii", **kw)
    assert not torch.allclose(out_flat.phi, out_rii.phi, atol=1e-6)


# --- audit 2026-06-10 fixes: self-edge identity, soft cap, dOmega/dmu gradient route ---
from vfe3.gradients.kernels import belief_gradients, uses_kernel_route
from vfe3.gradients.oracle import belief_gradients_autograd
from vfe3.inference.e_step import build_belief_transport, free_energy_value


def test_regime_ii_self_edge_equals_flat_identity():
    """Audit F4: the connection is an EDGE object, so the self-edge carries NO edge factor --
    Omega_ii = exp(phi_i)exp(-phi_i) exactly as on the flat path (delta_ii is zeroed before the
    exp). Off-diagonal edges still carry the connection (pinned by the non-flat test above)."""
    phi, mu, grp = _phi_mu(seed=5)
    K, n_gen = grp.generators.shape[-1], grp.generators.shape[0]
    g = torch.Generator().manual_seed(7)
    W = 0.4 * torch.randn(n_gen, K, K, generator=g)
    flat = compute_transport_operators(phi, grp)["Omega"]
    r2 = get_transport("regime_ii")(phi, grp, mu=mu, connection_W=W, cocycle_relaxation=1.0)["Omega"]
    N = phi.shape[1]
    idx = torch.arange(N)
    assert torch.allclose(r2[:, idx, idx], flat[:, idx, idx], atol=1e-6, rtol=0.0)
    eye = torch.eye(K).expand(phi.shape[0], N, K, K)
    assert torch.allclose(r2[:, idx, idx], eye, atol=1e-5)


def test_regime_ii_homotopy_stays_responsive_past_old_clamp():
    """Audit F3: delta is quadratic in the mean scale; before the smooth norm cap, every edge past
    stable_matrix_exp_pair's hard Frobenius clamp received the SAME rescaled operator, so
    cocycle_relaxation became inert (alpha=0.5 and 1.0 produced identical Omegas). The smooth cap
    is strictly monotone in alpha, so the homotopy stays responsive even at large ||delta||, and
    the result stays finite."""
    phi, mu, grp = _phi_mu(seed=8)
    mu = 5.0 * mu                                                # ||delta . G|| >> 15: old-clamp regime
    K, n_gen = grp.generators.shape[-1], grp.generators.shape[0]
    g = torch.Generator().manual_seed(13)
    W = 0.5 * torch.randn(n_gen, K, K, generator=g)
    build = get_transport("regime_ii")
    half = build(phi, grp, mu=mu, connection_W=W, cocycle_relaxation=0.5)["Omega"]
    full = build(phi, grp, mu=mu, connection_W=W, cocycle_relaxation=1.0)["Omega"]
    assert torch.isfinite(half).all() and torch.isfinite(full).all()
    assert not torch.allclose(half, full, atol=1e-5)


def test_regime_ii_excluded_from_kernel_route():
    """Audit F1: the hand kernel is the flat-transport gradient (it drops dOmega/dmu), so the
    kernel-coverage predicate must exclude transport_mode='regime_ii' even at the canonical
    operating point."""
    base = dict(renyi_order=1.0, gradient_mode="filtering", family="gaussian_diagonal",
                divergence_family="renyi", include_attention_entropy=True)
    assert uses_kernel_route(**base, transport_mode="flat")
    assert not uses_kernel_route(**base, transport_mode="regime_ii")


def _grad_setup(seed=0, N=3, K=4):
    grp = get_group("block_glk")(K, 2)
    n_gen = grp.generators.shape[0]
    g = torch.Generator().manual_seed(seed)
    mu      = 0.5 * torch.randn(N, K, generator=g)
    sigma   = torch.rand(N, K, generator=g) + 0.5
    phi     = 0.1 * torch.randn(N, n_gen, generator=g)
    mu_p    = 0.5 * torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    W       = 0.3 * torch.randn(n_gen, K, K, generator=g)
    return grp, mu, sigma, phi, mu_p, sigma_p, W


def test_regime_ii_filtering_gradient_carries_domega_dmu():
    """Audit F1/F2: the belief gradient served under regime_ii must differ from the frozen-Omega
    gradient (a pre-built Omega treated as constant) -- the difference IS the dOmega/dmu term the
    kernel route silently dropped."""
    grp, mu, sigma, phi, mu_p, sigma_p, W = _grad_setup(seed=1)

    def builder(mu_q, sigma_q, mu_k, sigma_k):                   # 4-arg oracle contract (audit C4)
        return build_belief_transport(phi, grp, transport_mode="regime_ii", mu=mu_q, mu_key=mu_k,
                                      connection_W=W, cocycle_relaxation=1.0)

    g_live, _ = belief_gradients(mu, sigma, mu_p, sigma_p, None, gradient_mode="filtering",
                                 transport_mode="regime_ii", omega_builder=builder,
                                 irrep_dims=grp.irrep_dims)
    omega_frozen = build_belief_transport(phi, grp, transport_mode="regime_ii", mu=mu,
                                          connection_W=W, cocycle_relaxation=1.0)
    g_frozen, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega_frozen,
                                            gradient_mode="filtering", irrep_dims=grp.irrep_dims)
    assert torch.isfinite(g_live).all()
    assert not torch.allclose(g_live, g_frozen, atol=1e-6)


def test_regime_ii_smoothing_gradient_matches_autograd_of_global_f():
    """Audit F1/F2 ground truth: under gradient_mode='smoothing' the served gradient must equal
    the autograd of the GLOBAL regime_ii F (free_energy_value, keys=None, which rebuilds Omega
    from the live belief means -- both delta slots live, exactly the smoothing key-role split),
    plus a central-difference spot check on a few coordinates."""
    grp, mu, sigma, phi, mu_p, sigma_p, W = _grad_setup(seed=2)
    fkw = dict(transport_mode="regime_ii", connection_W=W, cocycle_relaxation=1.0)

    def F(m):
        return free_energy_value(BeliefState(mu=m, sigma=sigma, phi=phi), mu_p, sigma_p, grp, **fkw)

    def builder(mu_q, sigma_q, mu_k, sigma_k):                   # 4-arg oracle contract (audit C4)
        return build_belief_transport(phi, grp, transport_mode="regime_ii", mu=mu_q, mu_key=mu_k,
                                      connection_W=W, cocycle_relaxation=1.0)

    g_mu, _ = belief_gradients(mu, sigma, mu_p, sigma_p, None, gradient_mode="smoothing",
                               transport_mode="regime_ii", omega_builder=builder,
                               irrep_dims=grp.irrep_dims)
    mu_leaf = mu.clone().requires_grad_(True)
    (g_ref,) = torch.autograd.grad(F(mu_leaf), mu_leaf)
    assert torch.allclose(g_mu, g_ref, atol=1e-5, rtol=1e-4)
    h = 1e-3                                                     # FD spot check (fp32-loose)
    for (i, k) in ((0, 0), (1, 2), (2, 3)):
        e = torch.zeros_like(mu); e[i, k] = h
        fd = (F(mu + e) - F(mu - e)) / (2.0 * h)
        assert abs(float(g_mu[i, k]) - float(fd)) <= 0.05 * abs(float(fd)) + 5e-3


def test_regime_ii_df_dw_matches_fd():
    """Audit F2: dF/d connection_W against central differences -- the ground-truth gradient check
    for the learned connection that the suite previously lacked (existence-only pin)."""
    grp, mu, sigma, phi, mu_p, sigma_p, W = _grad_setup(seed=3)

    def F(w):
        return free_energy_value(BeliefState(mu=mu, sigma=sigma, phi=phi), mu_p, sigma_p, grp,
                                 transport_mode="regime_ii", connection_W=w, cocycle_relaxation=1.0)

    W_leaf = W.clone().requires_grad_(True)
    (g_w,) = torch.autograd.grad(F(W_leaf), W_leaf)
    assert torch.isfinite(g_w).all()
    assert g_w.abs().sum() > 1e-6
    h = 1e-3
    for (a, r, c) in ((0, 0, 1), (1, 2, 0), (3, 1, 3)):
        e = torch.zeros_like(W); e[a, r, c] = h
        fd = (F(W + e) - F(W - e)) / (2.0 * h)
        assert abs(float(g_w[a, r, c]) - float(fd)) <= 0.05 * abs(float(fd)) + 5e-3


def test_regime_ii_edge_factor_breaks_gauge_invariance_for_nonzero_W():
    r"""Audit F6 (characterization of a documented opt-in impurity): the Regime-II edge factor
    delta_ij = mu_i^T W^a mu_j is gauge-invariant ONLY at W=0. With phi held fixed (so the vertex
    factors exp(phi_i), exp(-phi_j) are unchanged), applying a gauge-group element g to the means
    (mu_i -> g mu_i) leaves Omega unchanged at W=0 (edge factor = I, mu-independent), but for W != 0
    Omega deviates and the deviation GROWS with ||W|| -- the 'exact at zero init, drifts under
    training' equivariance break the CLAUDE.md caveat now records (only W=0 is invariant)."""
    from vfe3.geometry.transport import _build_regime_ii

    K, n_heads = 4, 2
    phi, mu, grp = _phi_mu(seed=1, B=1, N=3, K=K, group="block_glk", n_heads=n_heads)
    n_gen = grp.generators.shape[0]
    g_rng = torch.Generator().manual_seed(7)
    # a gauge-group element g = exp(sum_a x_a G_a), applied per token: mu_i -> g mu_i
    x = 0.2 * torch.randn(n_gen, generator=g_rng)
    g = torch.linalg.matrix_exp(torch.einsum("a,akl->kl", x, grp.generators))
    mu_g = torch.einsum("kl,bnl->bnk", g, mu)

    def omega(mu_in, W):
        return _build_regime_ii(phi, grp, mu=mu_in, connection_W=W, cocycle_relaxation=1.0)["Omega"]

    W0 = torch.zeros(n_gen, K, K)
    assert torch.allclose(omega(mu, W0), omega(mu_g, W0), atol=1e-5)     # W=0: invariant (flat, mu-free)

    W_base = torch.randn(n_gen, K, K, generator=g_rng)
    devs = [(omega(mu, s * W_base) - omega(mu_g, s * W_base)).norm().item() for s in (0.1, 0.5, 1.0)]
    assert devs[0] > 1e-4                                                # nonzero W breaks invariance
    assert devs[0] < devs[1] < devs[2]                                  # break grows with ||W||


# --- audit 2026-07-01 F10: query-chunking of the dense regime_ii build (covariant-builder port) ---

def test_regime_ii_query_chunk_used():
    """The plain regime_ii builder shares the covariant chunk policy: the OOM-scale config chunks
    below N; a tiny diagnostic build collapses to a single chunk (bit-for-bit the unchunked path)."""
    from vfe3.geometry import transport as T
    assert 1 <= T._regime_ii_query_chunk(64, 128, 20) < 128              # OOM config must chunk
    assert T._regime_ii_query_chunk(1, 3, 4) == 3                        # tiny build: one chunk


def test_regime_ii_chunked_matches_unchunked():
    """Forcing size-1 query chunks gives the SAME Omega and the SAME gradient to connection_W as
    the one-chunk build -- chunking is purely a memory optimization (no cross-query reduction),
    mirroring the covariant builder's equivalence pin."""
    from vfe3.geometry import transport as T

    phi, mu, grp = _phi_mu(seed=7, B=2, N=5, K=4)
    K, n_gen = grp.generators.shape[-1], grp.generators.shape[0]
    W0 = 0.3 * torch.randn(n_gen, K, K, generator=torch.Generator().manual_seed(3))
    build = get_transport("regime_ii")

    def _omega_and_grad(chunk_elems: int):
        # monkeypatch-free: set/restore the module constant around each build
        saved = T._REGIME_II_CHUNK_ELEMS
        T._REGIME_II_CHUNK_ELEMS = chunk_elems
        try:
            W = W0.clone().requires_grad_(True)
            omega = build(phi, grp, mu=mu, connection_W=W, cocycle_relaxation=1.0)["Omega"]
            (omega ** 2).sum().backward()
            return omega.detach().clone(), W.grad.detach().clone()
        finally:
            T._REGIME_II_CHUNK_ELEMS = saved

    omega_one, grad_one = _omega_and_grad(10 ** 12)                      # one chunk (size N)
    omega_chunk, grad_chunk = _omega_and_grad(1)                         # forced size-1 chunks
    assert torch.allclose(omega_one, omega_chunk, atol=1e-5, rtol=1e-5)
    assert torch.allclose(grad_one, grad_chunk, atol=1e-5, rtol=1e-5)
