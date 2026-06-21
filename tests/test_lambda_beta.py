r"""Tests for lambda_beta: the belief-coupling weight.

lambda_beta scales the WHOLE belief-coupling block of F -- sum_ij [ beta_ij E_ij +
tau beta_ij log(beta_ij/pi_ij) ] -- relative to the alpha self-term. The correctness
invariant: it scales coupling AND entropy by the same factor and leaves beta = softmax(-E/tau)
alone, so (a) the gradient is AFFINE in lambda_beta (a lambda-into-softmax leak would make it
nonlinear) and (b) the analytic kernel (which scales only its pair term) agrees with the autograd
oracle (which differentiates lambda_beta*F_red). learnable_lambda_beta adds a scalar nn.Parameter
log_lambda_beta (lambda_beta = exp(log_lambda_beta), init 0 -> 1.0) trained through the unrolled
E-step, mirroring log_alpha.
"""

import math

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.free_energy import free_energy, pairwise_energy, self_divergence
from vfe3.families.base import get_family
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators, transport_covariance, transport_mean
from vfe3.gradients.kernels import belief_gradients
from vfe3.gradients.oracle import belief_gradients_autograd
from vfe3.model.model import VFEModel


def _setup(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    phi = 0.15 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    mu = torch.randn(N, K, generator=g); sigma = torch.rand(N, K, generator=g) + 0.5
    mu_p = torch.randn(N, K, generator=g); sigma_p = torch.rand(N, K, generator=g) + 0.5
    return mu, sigma, mu_p, sigma_p, omega


# ---- the primary gate: kernel == oracle at lambda_beta != 1 -------------------

@pytest.mark.parametrize("lb", [0.5, 2.0])
def test_kernel_matches_oracle_at_lambda_beta(lb):
    args = _setup()
    km, ks = belief_gradients(*args, tau=1.5, gradient_mode="filtering", lambda_beta=lb)
    om, os_ = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering", lambda_beta=lb)
    assert torch.allclose(km, om, atol=1e-5)        # mu grad: kernel pair-scaling == oracle d(lb*F)
    assert torch.allclose(ks, os_, atol=1e-5)       # sigma grad


@pytest.mark.parametrize("lb", [0.5, 2.0])
def test_kernel_matches_oracle_multihead_at_lambda_beta(lb):
    # Per-head (block_glk, 2 heads): the envelope cancellation holds PER head, so the lambda_beta
    # scaling of the per-head coupling+entropy block must still leave kernel == oracle.
    g = torch.Generator().manual_seed(2)
    N, K = 4, 4
    grp = get_group("block_glk")(4, 2)              # irrep_dims [2, 2]
    phi = 0.15 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    mu = torch.randn(N, K, generator=g); sigma = torch.rand(N, K, generator=g) + 0.5
    mu_p = torch.randn(N, K, generator=g); sigma_p = torch.rand(N, K, generator=g) + 0.5
    km, ks = belief_gradients(mu, sigma, mu_p, sigma_p, omega, tau=1.5,
                              gradient_mode="filtering", irrep_dims=grp.irrep_dims, lambda_beta=lb)
    om, os_ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, tau=1.5,
                                        gradient_mode="filtering", irrep_dims=grp.irrep_dims, lambda_beta=lb)
    assert torch.allclose(km, om, atol=1e-5)
    assert torch.allclose(ks, os_, atol=1e-5)


def test_oracle_smoothing_matches_at_lambda_beta():
    # The oracle is the only path for smoothing; lambda_beta must scale its coupling there too.
    args = _setup()
    a = belief_gradients(*args, tau=1.5, gradient_mode="smoothing", lambda_beta=2.0)
    b = belief_gradients_autograd(*args, tau=1.5, gradient_mode="smoothing", lambda_beta=2.0)
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[1], b[1], atol=1e-6)


# ---- beta stays lambda-free: the gradient is AFFINE in lambda_beta ------------

def test_gradient_is_affine_in_lambda_beta_kernel():
    # grad(lb) = self + lb*pair. If lambda leaked into the softmax, beta(lb) would make grad
    # NONLINEAR in lb. Affinity (equal second difference) is the observable that catches the leak.
    args = _setup()
    g0 = belief_gradients(*args, tau=1.5, gradient_mode="filtering", lambda_beta=0.0)
    g1 = belief_gradients(*args, tau=1.5, gradient_mode="filtering", lambda_beta=1.0)
    g2 = belief_gradients(*args, tau=1.5, gradient_mode="filtering", lambda_beta=2.0)
    assert torch.allclose(g2[0] - g1[0], g1[0] - g0[0], atol=1e-6)   # mu grad affine in lb
    assert torch.allclose(g2[1] - g1[1], g1[1] - g0[1], atol=1e-6)   # sigma grad affine in lb


def test_gradient_is_affine_in_lambda_beta_oracle():
    args = _setup()
    g0 = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering", lambda_beta=0.0)
    g1 = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering", lambda_beta=1.0)
    g2 = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering", lambda_beta=2.0)
    assert torch.allclose(g2[0] - g1[0], g1[0] - g0[0], atol=1e-6)
    assert torch.allclose(g2[1] - g1[1], g1[1] - g0[1], atol=1e-6)


def test_lambda_beta_zero_leaves_only_self_term():
    # lambda_beta=0 removes the coupling entirely: only the alpha*D(q||p) self gradient remains.
    # With q == p and identity transport that self term is also zero, so the whole gradient is 0.
    K, N = 2, 3
    omega = torch.eye(K).expand(N, N, K, K).contiguous()
    mu = torch.randn(1, K).expand(N, K).contiguous(); sigma = torch.rand(N, K) + 0.5
    gmu, gsig = belief_gradients(mu, sigma, mu.clone(), sigma.clone(), omega,
                                 tau=1.5, gradient_mode="filtering", lambda_beta=0.0)
    assert torch.allclose(gmu, torch.zeros(N, K), atol=1e-6)
    assert torch.allclose(gsig, torch.zeros(N, K), atol=1e-6)


# ---- pure-path preservation --------------------------------------------------

def test_lambda_beta_one_is_byte_identical_to_default():
    # lambda_beta=1.0 must reproduce the default (no-lambda_beta) path exactly, on both branches.
    args = _setup()
    k1 = belief_gradients(*args, tau=1.5, gradient_mode="filtering", lambda_beta=1.0)
    kd = belief_gradients(*args, tau=1.5, gradient_mode="filtering")
    assert torch.equal(k1[0], kd[0]) and torch.equal(k1[1], kd[1])
    o1 = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering", lambda_beta=1.0)
    od = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering")
    assert torch.equal(o1[0], od[0]) and torch.equal(o1[1], od[1])


def test_free_energy_scales_block_by_lambda_beta():
    # F(lb) = self + lb*(coupling+entropy). The second difference over lb vanishes (affine), and
    # F(0) drops the entire coupling+entropy block (== self-only F).
    mu, sigma, mu_p, sigma_p, omega = _setup()
    fam = get_family("gaussian_diagonal")
    mu_t = transport_mean(omega, mu); sigma_t = transport_covariance(omega, sigma)
    sd = self_divergence(fam(mu, sigma), fam(mu_p, sigma_p))
    energy = pairwise_energy(fam(mu, sigma), fam(mu_t, sigma_t))
    alpha = torch.ones_like(sd)
    f0 = free_energy(sd, energy, alpha, tau=1.5, lambda_beta=0.0)
    f1 = free_energy(sd, energy, alpha, tau=1.5, lambda_beta=1.0)
    f2 = free_energy(sd, energy, alpha, tau=1.5, lambda_beta=2.0)
    assert torch.allclose(f2 - f1, f1 - f0, atol=1e-6)              # affine in lb
    assert torch.allclose(f0, (alpha * sd).sum(), atol=1e-6)        # lb=0 -> self-only F


def test_config_rejects_negative_lambda_beta():
    with pytest.raises(ValueError, match="lambda_beta must be >= 0"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, lambda_beta=-0.5)


# ---- learnable lambda_beta (mirrors log_alpha) -------------------------------

def _cfg(**over):
    base = dict(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                n_e_steps=4, e_q_mu_lr=0.3, e_q_sigma_lr=0.1, e_phi_lr=0.0)
    base.update(over)
    return VFE3Config(**base)


def test_default_off_no_log_lambda_beta_attribute():
    model = VFEModel(_cfg())                         # learnable_lambda_beta defaults to False
    assert not hasattr(model, "log_lambda_beta")


def test_learnable_creates_scalar_param_init_one():
    import torch.nn as nn
    model = VFEModel(_cfg(learnable_lambda_beta=True))
    assert isinstance(model.log_lambda_beta, nn.Parameter)
    assert model.log_lambda_beta.shape == ()
    assert float(model.log_lambda_beta.detach()) == 0.0      # exp(0) = lambda_beta = 1.0


def test_learnable_init_equals_constant_one():
    # learnable-at-init (log_lambda_beta=0 -> 1.0) must match the constant lambda_beta=1.0 path.
    tok = torch.randint(0, 20, (3, 5)); tgt = torch.randint(0, 20, (3, 5))
    torch.manual_seed(0); m_const = VFEModel(_cfg(lambda_beta=1.0))
    torch.manual_seed(0); m_learn = VFEModel(_cfg(learnable_lambda_beta=True))
    lc, loss_c, _ = m_const(tok, tgt)
    ll, loss_l, _ = m_learn(tok, tgt)
    assert torch.equal(lc, ll)
    assert torch.equal(loss_c, loss_l)


def test_learnable_log_lambda_beta_grad_populated():
    # The learned weight trains: log_lambda_beta.grad is finite and nonzero after backward
    # (the coupling block reaches the loss through the unrolled E-step).
    model = VFEModel(_cfg(learnable_lambda_beta=True))
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert model.log_lambda_beta.grad is not None
    assert torch.isfinite(model.log_lambda_beta.grad)
    assert model.log_lambda_beta.grad.abs() > 0


def test_learnable_with_detach_warns_and_freezes():
    with pytest.warns(UserWarning, match="freezes log_lambda_beta"):
        model = VFEModel(_cfg(learnable_lambda_beta=True, detach_e_step=True))
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert model.log_lambda_beta.grad is None        # frozen under the detached E-step


def test_learnable_lambda_beta_in_optimizer_coverage():
    # build_optimizer's exact-coverage guard must include log_lambda_beta (else it would never
    # train and the guard would raise).
    from vfe3.train import build_optimizer
    model = VFEModel(_cfg(learnable_lambda_beta=True))
    opt = build_optimizer(model, model.cfg)
    grouped = {p for grp in opt.param_groups for p in grp["params"]}
    assert model.log_lambda_beta in grouped


def test_learnable_changes_forward_when_lambda_beta_moves():
    # Moving log_lambda_beta away from 0 must change the converged belief / loss (genuinely consumed).
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    torch.manual_seed(0)
    model = VFEModel(_cfg(learnable_lambda_beta=True, pos_phi="none"))
    _, loss0, _ = model(tok, tgt)
    with torch.no_grad():
        model.log_lambda_beta.copy_(torch.log(torch.tensor(3.0)))    # lambda_beta = 3.0
    _, loss1, _ = model(tok, tgt)
    assert not torch.allclose(loss0, loss1, atol=1e-6)


def test_learnable_diagnostics_runs():
    model = VFEModel(_cfg(learnable_lambda_beta=True))
    tok = torch.randint(0, 20, (2, 5))
    d = model.diagnostics(tok)
    assert math.isfinite(d["self_coupling"]) and math.isfinite(d["belief_coupling"])
