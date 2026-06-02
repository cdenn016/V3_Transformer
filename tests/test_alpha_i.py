import torch

from vfe3.alpha_i import alpha_regularizer, register_alpha, self_coupling_alpha


def test_constant_alpha_is_value_zero_reg():
    kl = torch.rand(3, 5)
    a, r = self_coupling_alpha(kl, mode="constant", value=1.0)
    assert torch.allclose(a, torch.ones(3, 5))
    assert torch.allclose(r, torch.zeros(3, 5))


def test_state_dependent_alpha_formula_and_minimizes_objective():
    # alpha* = c0/(b0 + KL) is the stationary point of  alpha*KL + b0*alpha - c0*log(alpha).
    kl = torch.tensor([0.0, 1.0, 4.0])
    b0, c0 = 0.5, 2.0
    a, r = self_coupling_alpha(kl, mode="state_dependent", b0=b0, c0=c0)
    assert torch.allclose(a, c0 / (b0 + kl), atol=1e-6)
    # d/d alpha [alpha*KL + b0*alpha - c0*log alpha] = KL + b0 - c0/alpha == 0 at alpha*
    grad = kl + b0 - c0 / a
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-5)


def test_per_coord_alpha_uses_per_dimension_kl():
    kl = torch.rand(2, 4, 3) + 0.1                       # (..., N, K) per-coordinate KL
    b0 = torch.full((3,), 0.5)
    c0 = torch.full((3,), 2.0)
    a, r = self_coupling_alpha(kl, mode="state_dependent_per_coord", b0=b0, c0=c0)
    assert torch.allclose(a, c0 / (b0 + kl), atol=1e-6)


def test_new_form_with_novel_kwarg_reachable_without_editing_dispatcher():
    # Modularity: a registered form's OWN param must flow through the dispatcher's
    # **kwargs (not a hard-coded value/b0/c0 union), so a new form selects-with-config
    # without editing the call site.
    @register_alpha("_test_scaled")
    def _scaled(kl, *, scale=2.0, **kwargs):
        return scale * torch.ones_like(kl), torch.zeros_like(kl)

    kl = torch.zeros(3)
    a, r = self_coupling_alpha(kl, mode="_test_scaled", scale=5.0)
    assert torch.allclose(a, torch.full((3,), 5.0))


def test_alpha_is_per_coord_declares_reduction_need():
    # Modularity: each alpha form DECLARES whether it consumes a per-coordinate (unsummed)
    # self-divergence, so the routing seam reads that flag rather than hard-coding a mode
    # name at the call sites. A future per-coordinate form slots in by registering with
    # per_coord=True -- no consumer is edited.
    from vfe3.alpha_i import alpha_is_per_coord
    assert alpha_is_per_coord("state_dependent_per_coord") is True
    assert alpha_is_per_coord("state_dependent") is False
    assert alpha_is_per_coord("constant") is False


def test_register_alpha_per_coord_flag_is_modular():
    from vfe3.alpha_i import register_alpha, alpha_is_per_coord

    @register_alpha("_test_pc", per_coord=True)
    def _pc(kl, **kwargs):
        return kl, torch.zeros_like(kl)

    assert alpha_is_per_coord("_test_pc") is True


def test_learnable_alpha_is_exp_log_alpha_zero_reg():
    # The learnable form (NN exception): alpha = exp(log_alpha) broadcast to kl, zero regularizer.
    # log_alpha = 0 -> alpha = 1.0 (reproduces the constant alpha=1.0 default at init).
    kl = torch.rand(3, 5) + 0.1
    a, r = self_coupling_alpha(kl, mode="learnable", log_alpha=torch.zeros(()))
    assert torch.allclose(a, torch.ones(3, 5), atol=1e-7)
    assert torch.allclose(r, torch.zeros(3, 5))
    # log_alpha = log(2) -> alpha = 2.0
    a2, r2 = self_coupling_alpha(kl, mode="learnable", log_alpha=torch.log(torch.tensor(2.0)))
    assert torch.allclose(a2, torch.full((3, 5), 2.0), atol=1e-6)
    assert torch.allclose(r2, torch.zeros(3, 5))


def test_learnable_alpha_gradient_flows_to_log_alpha():
    # The learned scalar must be in the autograd graph: a sum over alpha must produce a grad on log_alpha.
    kl = torch.rand(4) + 0.1
    log_alpha = torch.zeros((), requires_grad=True)
    a, _ = self_coupling_alpha(kl, mode="learnable", log_alpha=log_alpha)
    a.sum().backward()
    assert log_alpha.grad is not None and torch.isfinite(log_alpha.grad).all()


from vfe3.alpha_i import alpha_gradient_coefficient


def test_alpha_grad_coefficient_constant_is_value():
    kl = torch.rand(3, 5)
    assert torch.allclose(alpha_gradient_coefficient(kl, mode="constant", value=2.0),
                          torch.full((3, 5), 2.0))


def test_alpha_grad_coefficient_state_dependent_is_alpha_star():
    # By the alpha-envelope, the effective coefficient is alpha* itself (the
    # alpha'*D and R' paths cancel at the stationary alpha* = c0/(b0+KL)).
    kl = torch.tensor([0.0, 1.0, 4.0])
    b0, c0 = 0.5, 2.0
    coef = alpha_gradient_coefficient(kl, mode="state_dependent", b0=b0, c0=c0)
    assert torch.allclose(coef, c0 / (b0 + kl), atol=1e-6)
