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
