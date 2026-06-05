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
def _tiny_cfg(transport_mode="flat", cocycle_relaxation=1.0):
    return VFE3Config(
        vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
        n_e_steps=2, e_mu_lr=0.05, e_phi_lr=0.0,
        transport_mode=transport_mode, cocycle_relaxation=cocycle_relaxation,
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
    quadratic delta, hence its gradient, is vanishingly small)."""
    torch.manual_seed(0)
    model = VFEModel(_tiny_cfg(transport_mode="regime_ii"))
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
    # transport_mode. e_mu_lr=e_sigma_lr=0 isolates the phi step; W != 0 makes regime_ii != flat.
    grp = get_group("block_glk")(6, 2)
    n_gen = grp.generators.shape[0]
    torch.manual_seed(0)
    N, K = 4, 6
    belief = BeliefState(mu=torch.randn(N, K), sigma=torch.rand(N, K) + 0.5,
                         phi=0.1 * torch.randn(N, n_gen))
    mu_p, sigma_p = torch.randn(N, K), torch.rand(N, K) + 0.5
    W = 0.5 * torch.randn(n_gen, K, K)                 # nonzero connection -> regime_ii != flat
    kw = dict(e_mu_lr=0.0, e_sigma_lr=0.0, e_phi_lr=0.1, connection_W=W, cocycle_relaxation=1.0)
    out_flat = e_step_iteration(belief, mu_p, sigma_p, grp, transport_mode="flat", **kw)
    out_rii = e_step_iteration(belief, mu_p, sigma_p, grp, transport_mode="regime_ii", **kw)
    assert not torch.allclose(out_flat.phi, out_rii.phi, atol=1e-6)
