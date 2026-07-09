import warnings
from typing import Any

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.free_energy import (
    attention_weights,
    free_energy,
    pairwise_energy,
    self_divergence_for_alpha,
)
from vfe3.geometry.transport import transport_covariance, transport_mean
from vfe3.gradients.kernels import belief_gradients, mm_exact_update
from vfe3.gradients.oracle import belief_gradients_autograd
from vfe3.metrics import compute_metrics, free_energy_terms
from vfe3.model.model import VFEModel


def _twohop_case() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Three diagonal Gaussians with zero, live, and clamp-saturated pair energies."""
    mu = torch.tensor([[0.0], [0.0], [0.1]], dtype=torch.float32)
    sigma = torch.tensor([[1.0], [1.0], [0.01]], dtype=torch.float32)
    mu_p = mu.clone()
    sigma_p = sigma.clone()
    omega = torch.eye(1, dtype=torch.float32).expand(3, 3, 1, 1).contiguous()
    return mu, sigma, mu_p, sigma_p, omega


def _twohop_mm_case() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Three live destinations plus exact-zero self pairs for the MM stationarity check."""
    mu = torch.tensor([[-0.6], [0.2], [1.1]], dtype=torch.float32)
    sigma = torch.tensor([[0.7], [1.4], [0.9]], dtype=torch.float32)
    mu_p = torch.tensor([[0.1], [-0.2], [0.4]], dtype=torch.float32)
    sigma_p = torch.tensor([[1.0], [0.8], [1.2]], dtype=torch.float32)
    omega = torch.eye(1, dtype=torch.float32).expand(3, 3, 1, 1).contiguous()
    return mu, sigma, mu_p, sigma_p, omega


def _twohop_scalar_mu_gradient(
    mu:           torch.Tensor,
    sigma:        torch.Tensor,
    mu_p:         torch.Tensor,
    sigma_p:      torch.Tensor,
    omega:        torch.Tensor,

    *,
    kl_max:        float,
    lambda_twohop: float,
    tau:           float,
) -> torch.Tensor:
    r"""Filtering scalar derivative with raw detached two-hop attention weights."""
    mu_q = mu.detach().clone().requires_grad_(True)
    sigma_q = sigma.detach().clone().requires_grad_(True)
    mu_k = mu_q.detach()
    sigma_k = sigma_q.detach()
    mu_t = transport_mean(omega, mu_k)
    sigma_t = transport_covariance(omega, sigma_k, diagonal_out=True)

    q = DiagonalGaussian(mu_q, sigma_q)
    p = DiagonalGaussian(mu_p, sigma_p)
    transported = DiagonalGaussian(mu_t, sigma_t)
    self_div = self_divergence_for_alpha(q, p, alpha=1.0, kl_max=kl_max)
    energy = pairwise_energy(q, transported, alpha=1.0, kl_max=kl_max)
    assert energy[0, 1] == 0.0
    assert energy[0, 2] >= kl_max
    assert 0.0 < energy[2, 0] < kl_max
    alpha = torch.ones_like(self_div)
    beta = attention_weights(energy, tau=tau)
    base_F = free_energy(self_div, energy, alpha, tau=tau)
    w2 = beta.detach() @ beta.detach()
    scalar = base_F + lambda_twohop * (w2 * energy).sum()
    grad_mu, = torch.autograd.grad(scalar, mu_q)
    return grad_mu


def _twohop_mm_scalar_gradients(
    mu:            torch.Tensor,
    sigma:         torch.Tensor,
    mu_p:          torch.Tensor,
    sigma_p:       torch.Tensor,
    omega:         torch.Tensor,

    *,
    kl_max:        float,
    lambda_twohop: float,
    tau:           float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Autograd of the raw-hop, frozen-attention scalar at the MM closed-form point."""
    mu_k = mu.detach()
    sigma_k = sigma.detach()
    mu_t = transport_mean(omega, mu_k)
    sigma_t = transport_covariance(omega, sigma_k, diagonal_out=True)
    family_keys = DiagonalGaussian(mu_t, sigma_t)
    energy_0 = pairwise_energy(
        DiagonalGaussian(mu, sigma), family_keys,
        alpha=1.0, kl_max=kl_max,
    )
    beta_0 = attention_weights(energy_0, tau=tau).detach()
    pair_mask = ((energy_0 > 0.0) & (energy_0 < kl_max)).to(beta_0.dtype).detach()
    w2 = beta_0 @ beta_0
    mu_star, sigma_star = mm_exact_update(
        mu, sigma, mu_p, sigma_p, omega,
        kl_max=kl_max, lambda_twohop=lambda_twohop, tau=tau,
    )
    mu_q = mu_star.detach().requires_grad_(True)
    sigma_q = sigma_star.detach().requires_grad_(True)
    q = DiagonalGaussian(mu_q, sigma_q)
    self_div = self_divergence_for_alpha(
        q, DiagonalGaussian(mu_p, sigma_p),
        alpha=1.0, kl_max=kl_max,
    )
    energy = pairwise_energy(q, family_keys, alpha=1.0, kl_max=kl_max)
    scalar = (self_div.sum()
              + (beta_0 * pair_mask * energy).sum()
              + lambda_twohop * (w2 * pair_mask * energy).sum())
    return torch.autograd.grad(scalar, [mu_q, sigma_q])


def test_twohop_kernel_matches_scalar_autograd_with_zero_and_saturated_pairs() -> None:
    mu, sigma, mu_p, sigma_p, omega = _twohop_case()
    kl_max = 10.0
    lambda_twohop = 0.7
    tau = 1.3
    g_ref = _twohop_scalar_mu_gradient(
        mu, sigma, mu_p, sigma_p, omega,
        kl_max=kl_max, lambda_twohop=lambda_twohop, tau=tau,
    )
    g_kernel, _ = belief_gradients(
        mu, sigma, mu_p, sigma_p, omega,
        kl_max=kl_max, lambda_twohop=lambda_twohop, tau=tau,
        gradient_mode="filtering",
    )
    assert torch.allclose(g_kernel, g_ref, atol=2e-5, rtol=2e-5)
    mm_mu, mm_sigma, mm_mu_p, mm_sigma_p, mm_omega = _twohop_mm_case()
    g_mm_mu, g_mm_sigma = _twohop_mm_scalar_gradients(
        mm_mu, mm_sigma, mm_mu_p, mm_sigma_p, mm_omega,
        kl_max=kl_max, lambda_twohop=lambda_twohop, tau=tau,
    )
    assert torch.allclose(g_mm_mu, torch.zeros_like(g_mm_mu), atol=2e-5, rtol=2e-5)
    assert torch.allclose(g_mm_sigma, torch.zeros_like(g_mm_sigma), atol=2e-5, rtol=2e-5)


def test_oracle_twohop_gradient_matches_scalar_free_energy() -> None:
    mu, sigma, mu_p, sigma_p, omega = _twohop_case()
    kl_max = 10.0
    lambda_twohop = 0.7
    tau = 1.3
    g_ref = _twohop_scalar_mu_gradient(
        mu, sigma, mu_p, sigma_p, omega,
        kl_max=kl_max, lambda_twohop=lambda_twohop, tau=tau,
    )
    g_oracle, _ = belief_gradients_autograd(
        mu, sigma, mu_p, sigma_p, omega,
        kl_max=kl_max, lambda_twohop=lambda_twohop, tau=tau,
        gradient_mode="filtering",
    )
    assert torch.allclose(g_oracle, g_ref, atol=2e-5, rtol=2e-5)


def test_free_energy_terms_matches_scalar_with_twohop_and_value_gauge() -> None:
    torch.manual_seed(7)
    n = 3
    self_div = torch.rand(n, dtype=torch.float32) + 0.1
    score_energy = torch.rand(n, n, dtype=torch.float32)
    coupling_energy = torch.rand(n, n, dtype=torch.float32) + 0.25
    alpha = torch.rand(n, dtype=torch.float32) + 0.5
    alpha_reg = torch.rand(n, dtype=torch.float32) * 0.1
    log_likelihood = torch.rand(n, dtype=torch.float32)
    log_prior = torch.randn(n, n, dtype=torch.float32)
    tau = 1.2
    lambda_beta = 0.8
    lambda_twohop = 0.6
    beta = attention_weights(score_energy, tau=tau, log_prior=log_prior)

    scalar = free_energy(
        self_div, score_energy, alpha,
        tau=tau, lambda_beta=lambda_beta, lambda_twohop=lambda_twohop,
        log_prior=log_prior, alpha_reg=alpha_reg,
        coupling_energy=coupling_energy, log_likelihood=log_likelihood,
    )
    terms = free_energy_terms(
        self_div, score_energy, beta, alpha,
        tau=tau, lambda_beta=lambda_beta, lambda_twohop=lambda_twohop,
        log_prior=log_prior, alpha_reg=alpha_reg,
        coupling_energy=coupling_energy, log_likelihood=log_likelihood,
    )
    expected_twohop = float(((beta.detach() @ beta.detach()) * coupling_energy).sum())
    assert terms["twohop_coupling"] == expected_twohop
    assert terms["observation_likelihood"] == float(log_likelihood.sum())
    torch.testing.assert_close(
        torch.tensor(terms["total"]), scalar.detach(),
        atol=2e-5, rtol=2e-5,
    )
    registered = compute_metrics(
        ["free_energy_terms"],
        self_div=self_div, energy=score_energy, beta=beta, alpha=alpha,
        tau=tau, lambda_beta=lambda_beta, lambda_twohop=lambda_twohop,
        log_prior=log_prior, alpha_reg=alpha_reg,
        coupling_energy=coupling_energy, log_likelihood=log_likelihood,
    )["free_energy_terms"]
    assert registered == terms


def test_diagnostics_threads_all_active_objective_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.metrics as metrics_module

    seen: list[dict[str, Any]] = []
    original = metrics_module.free_energy_terms

    def record_and_compute(*args: Any, **kwargs: Any) -> dict[str, float]:
        terms = original(*args, **kwargs)
        seen.append({"args": args, "kwargs": kwargs, "terms": terms})
        return terms

    monkeypatch.setattr(metrics_module, "free_energy_terms", record_and_compute)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = VFE3Config(
            vocab_size=8,
            embed_dim=4,
            n_heads=2,
            max_seq_len=4,
            n_layers=1,
            n_e_steps=1,
            lambda_twohop=0.5,
            pos_rotation="rope",
            rope_on_value=False,
        )
        torch.manual_seed(11)
        model = VFEModel(cfg)
        token_ids = torch.zeros((1, cfg.max_seq_len), dtype=torch.long)
        log_likelihood = torch.tensor([-0.2, -0.3, -0.4, -0.5], dtype=torch.float32)
        diagnostics = model.diagnostics(token_ids, log_likelihood=log_likelihood)
        per_layer = model.diagnostics_per_layer(token_ids, log_likelihood=log_likelihood)

    assert len(seen) == 2
    for call in seen:
        args = call["args"]
        kwargs = call["kwargs"]
        assert kwargs["lambda_twohop"] == cfg.lambda_twohop
        assert kwargs["coupling_energy"] is not None
        assert not torch.allclose(kwargs["coupling_energy"], args[1])
        assert torch.equal(kwargs["log_likelihood"], log_likelihood)
        scalar = free_energy(
            args[0], args[1], args[3],
            tau=kwargs["tau"], lambda_beta=kwargs["lambda_beta"],
            lambda_twohop=kwargs["lambda_twohop"],
            include_attention_entropy=kwargs["include_attention_entropy"],
            log_prior=kwargs["log_prior"], alpha_reg=kwargs["alpha_reg"],
            coupling_energy=kwargs["coupling_energy"],
            log_likelihood=kwargs["log_likelihood"],
        )
        torch.testing.assert_close(
            torch.tensor(call["terms"]["total"]), scalar,
            atol=2e-5, rtol=2e-5,
        )
    assert diagnostics["total"] == seen[0]["terms"]["total"]
    assert per_layer["total"][0] == seen[1]["terms"]["total"]
    assert diagnostics["twohop_coupling"] == seen[0]["terms"]["twohop_coupling"]
    assert diagnostics["observation_likelihood"] == seen[0]["terms"]["observation_likelihood"]
    assert per_layer["twohop_coupling"][0] == seen[1]["terms"]["twohop_coupling"]
    assert per_layer["observation_likelihood"][0] == seen[1]["terms"]["observation_likelihood"]
