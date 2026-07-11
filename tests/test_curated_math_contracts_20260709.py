r"""Curated C6 contracts for fixed surrogates and frozen-sigma inference."""

from types import SimpleNamespace

import pytest
import torch

import vfe3.gradients.kernels as kernels
import vfe3.gradients.oracle as oracle
import vfe3.inference.e_step as e_step_module
from vfe3.config import VFE3Config
from vfe3.inference.e_step import BeliefState, e_step_iteration
from vfe3.run_artifacts import _pure_path_report


def _warning_messages(records: pytest.WarningsRecorder) -> str:
    """Return captured warning messages in lower case for contract assertions."""
    return "\n".join(str(record.message) for record in records).lower()


def _identity_transport(n_tokens: int, dim: int) -> torch.Tensor:
    """Dense identity transport for every ordered token pair."""
    return torch.eye(dim, dtype=torch.float32).expand(n_tokens, n_tokens, dim, dim).clone()


def test_precision_attention_warning_labels_detached_prior_and_fixed_covariance() -> None:
    with pytest.warns(UserWarning) as records:
        VFE3Config(embed_dim=4, n_heads=2, precision_weighted_attention=True)

    messages = _warning_messages(records)
    assert "detached precision prior" in messages
    assert "fixed covariance" in messages


def test_pure_path_report_exposes_fixed_surrogate_flags() -> None:
    with pytest.warns(UserWarning):
        cfg = VFE3Config(
            embed_dim=4,
            n_heads=2,
            use_prior_bank=False,
            precision_weighted_attention=True,
            query_adaptive_tau=True,
            skip_belief_sigma_update=True,
            e_step_update="mm_exact",
            lambda_alpha_mode="state_dependent",
        )

    toggles = _pure_path_report(cfg, [])["config_toggles"]
    assert toggles["fixed_covariance_surrogate"] is True
    assert toggles["detached_precision_prior"] is True
    assert toggles["detached_query_adaptive_tau"] is True
    assert toggles["state_dependent_alpha_majorizer"] is True


def test_state_dependent_alpha_mm_warning_says_majorizer_not_one_step_exact() -> None:
    with pytest.warns(UserWarning) as records:
        VFE3Config(
            embed_dim=4,
            n_heads=2,
            e_step_update="mm_exact",
            lambda_alpha_mode="state_dependent",
        )

    messages = _warning_messages(records)
    assert "majorizer" in messages
    assert "not one-step exact" in messages


def test_state_dependent_alpha_mm_damped_warning_separates_target_and_step() -> None:
    with pytest.warns(UserWarning) as records:
        VFE3Config(
            embed_dim=4,
            n_heads=2,
            e_step_update="mm_exact",
            mm_damping=0.5,
            lambda_alpha_mode="state_dependent",
        )

    messages = _warning_messages(records)
    assert "computes the frozen state-dependent-alpha majorizer minimizer" in messages
    assert "a damped step for values below 1.0" in messages
    assert "the full step at 1.0" in messages
    assert "exactly minimizes" not in messages


def test_skip_sigma_oracle_never_requests_sigma_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_tokens, dim = 3, 2
    mu = torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]], dtype=torch.float32)
    sigma = torch.full((n_tokens, dim), 1.2, dtype=torch.float32)
    mu_p = torch.zeros_like(mu)
    sigma_p = torch.ones_like(sigma)
    omega = _identity_transport(n_tokens, dim)
    requested_inputs: list[tuple[torch.Tensor, ...]] = []
    real_grad = oracle.torch.autograd.grad

    def _grad_spy(
        outputs: torch.Tensor,
        inputs: list[torch.Tensor],
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]:
        requested_inputs.append(tuple(inputs))
        return real_grad(outputs, inputs, *args, **kwargs)

    monkeypatch.setattr(oracle.torch.autograd, "grad", _grad_spy)
    grad_mu, grad_sigma = oracle.belief_gradients_autograd(
        mu,
        sigma,
        mu_p,
        sigma_p,
        omega,
        need_sigma_grad=False,
    )

    assert grad_mu.shape == mu.shape
    assert grad_sigma is None
    assert len(requested_inputs) == 1
    assert len(requested_inputs[0]) == 1


def test_skip_sigma_mm_skips_sigma_fusion_and_returns_input_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_tokens, dim = 3, 2
    mu = torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]], dtype=torch.float32)
    sigma = torch.full((n_tokens, dim), 1.2, dtype=torch.float32)
    mu_p = torch.zeros_like(mu)
    sigma_p = torch.ones_like(sigma)
    omega = _identity_transport(n_tokens, dim)

    def _forbid_sigma_fusion(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("sigma fusion must be skipped when sigma is frozen")

    monkeypatch.setattr(kernels, "_pair_mass", _forbid_sigma_fusion)
    mu_star, sigma_star = kernels.mm_exact_update(
        mu,
        sigma,
        mu_p,
        sigma_p,
        omega,
        need_sigma_update=False,
    )

    assert mu_star.shape == mu.shape
    assert sigma_star is sigma


def test_skip_sigma_gradient_skips_natural_preconditioning_and_returns_input_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_tokens, dim = 3, 2
    mu = torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]], dtype=torch.float32)
    sigma = torch.full((n_tokens, dim), 1.2, dtype=torch.float32)
    phi = torch.zeros(n_tokens, dim * dim, dtype=torch.float32)
    belief = BeliefState(mu=mu, sigma=sigma, phi=phi)
    mu_p = torch.zeros_like(mu)
    sigma_p = torch.ones_like(sigma)
    omega = _identity_transport(n_tokens, dim)
    group = SimpleNamespace(irrep_dims=[dim])

    monkeypatch.setattr(e_step_module, "build_belief_transport", lambda *args, **kwargs: omega)
    monkeypatch.setattr(e_step_module, "_transport", lambda *args, **kwargs: omega)
    monkeypatch.setattr(
        e_step_module,
        "belief_gradients",
        lambda *args, **kwargs: (torch.ones_like(mu), None),
    )

    natural_calls: list[torch.Tensor | None] = []

    class _MeanOnlyNaturalGradient:
        def __call__(self, *args: object, **kwargs: object) -> "_MeanOnlyNaturalGradient":
            return self

        def natural_gradient(
            self,
            grad_mu: torch.Tensor,
            grad_sigma: torch.Tensor | None,
            *,
            eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            natural_calls.append(grad_sigma)
            return 2.0 * grad_mu, None

    monkeypatch.setattr(e_step_module, "get_family", lambda *args, **kwargs: _MeanOnlyNaturalGradient())
    out = e_step_iteration(
        belief,
        mu_p,
        sigma_p,
        group,
        e_q_mu_lr=0.25,
        e_phi_lr=0.0,
        skip_belief_sigma_update=True,
    )

    assert len(natural_calls) == 1
    assert natural_calls[0] is None
    assert out.sigma is sigma
    assert torch.equal(out.mu, mu - 0.5)
