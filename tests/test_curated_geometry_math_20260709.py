r"""Focused regressions for family-aware full-covariance inference geometry."""

import math

import pytest
import torch
import torch.nn.functional as F

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank
from vfe3.numerics import apply_mu_trust_region


def _full_prior_bank(*, eps: float = 1e-3) -> PriorBank:
    cfg = VFE3Config(
        vocab_size=3,
        embed_dim=2,
        n_heads=1,
        max_seq_len=2,
        n_layers=1,
        n_e_steps=1,
        e_step_gradient="straight_through",
        pos_phi="none",
        family="gaussian_full",
        decode_mode="full",
        use_prior_bank=True,
        eps=eps,
    )
    return VFEModel(cfg).prior_bank


def test_halt_tol_full_covariance_uses_full_gaussian_kl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe3.inference.e_step as e_step_module

    group = get_group("glk")(2)
    n_gen = group.generators.shape[0]
    sigma_0 = torch.eye(2).reshape(1, 2, 2)
    sigma_1 = torch.tensor([[[1.0, -0.5], [-0.5, 1.0]]])
    belief = BeliefState(
        mu=torch.zeros(1, 2),
        sigma=sigma_0,
        phi=torch.zeros(1, n_gen),
    )
    calls = 0

    def _fixed_iteration(
        current: BeliefState,
        *args: object,
        **kwargs: object,
    ) -> BeliefState:
        nonlocal calls
        calls += 1
        return current._replace(sigma=sigma_1)

    monkeypatch.setattr(e_step_module, "e_step_iteration", _fixed_iteration)
    with torch.no_grad():
        e_step_module.e_step(
            belief,
            torch.zeros(1, 2),
            sigma_0,
            group,
            n_iter=3,
            e_phi_lr=0.1,
            e_step_halt_tol=0.05,
            family="gaussian_full",
        )

    assert calls == 2


def test_full_chunked_matches_dense_at_variance_floor_without_double_ridge() -> None:
    pb = _full_prior_bank(eps=1e-3)
    with torch.no_grad():
        pb.mu_embed.copy_(torch.tensor([[0.0, 0.0], [0.02, -0.01], [-0.03, 0.04]]))
        pb.sigma_log_embed.fill_(math.log(pb.eps))
    mu_q = torch.tensor([[[0.01, -0.02]]])
    sigma_q = torch.diag_embed(torch.full((1, 1, 2), pb.eps))

    pb.decode_mode = "full"
    dense = pb.decode(mu_q, sigma_q)
    pb.decode_mode = "full_chunked"
    chunked = pb.decode(mu_q, sigma_q)

    assert torch.allclose(chunked, dense, rtol=1e-5, atol=1e-6)

    targets = torch.tensor([[2]])
    dense_ce = F.cross_entropy(dense.reshape(-1, pb.vocab_size), targets.reshape(-1))
    chunked_ce = pb.decode_ce_full_chunked(mu_q, sigma_q, targets, chunk_size=2)
    assert torch.allclose(chunked_ce, dense_ce, rtol=1e-5, atol=1e-6)

    sigma_retry = torch.diag_embed(torch.tensor([[[-0.1, 1.0]]]))
    pb.decode_mode = "full"
    dense_retry = pb.decode(mu_q, sigma_retry)
    pb.decode_mode = "full_chunked"
    chunked_retry = pb.decode(mu_q, sigma_retry)
    assert torch.equal(torch.isneginf(chunked_retry), torch.isneginf(dense_retry))
    assert torch.allclose(chunked_retry, dense_retry, rtol=1e-5, atol=1e-6)


def test_full_cov_query_invariants_use_raw_spd_on_round_zero() -> None:
    pb = _full_prior_bank(eps=1e-3)
    sigma_q = torch.tensor([[[[0.004, 0.001], [0.001, 0.003]]]])

    diag_q, logdet_q = pb._full_cov_query_invariants(sigma_q)
    factor = torch.linalg.cholesky(sigma_q)
    expected_logdet = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)

    assert torch.equal(diag_q, torch.diagonal(sigma_q, dim1=-2, dim2=-1))
    assert torch.equal(logdet_q, expected_logdet)


def test_full_cov_box_binds_in_mahalanobis_units() -> None:
    sigma_q = torch.tensor([[[1.0, 0.9], [0.9, 1.0]]])
    delta_mu = torch.tensor([[0.0, 1.0]])
    factor = torch.linalg.cholesky(sigma_q)

    out = apply_mu_trust_region(
        delta_mu,
        sigma_q,
        trust=1.0,
        mode="box",
        is_diagonal=False,
    )
    whitened = torch.linalg.solve_triangular(factor, out.unsqueeze(-1), upper=False).squeeze(-1)

    assert whitened.abs().max() <= 1.0 + 1e-6
    assert torch.allclose(whitened, torch.tensor([[0.0, 1.0]]), atol=1e-6)


def test_full_cov_ball_bounds_mahalanobis_norm() -> None:
    sigma_q = torch.tensor([[[1.0, 0.9], [0.9, 1.0]]])
    delta_mu = torch.tensor([[0.0, 1.0]])
    factor = torch.linalg.cholesky(sigma_q)

    out = apply_mu_trust_region(
        delta_mu,
        sigma_q,
        trust=1.0,
        mode="ball",
        is_diagonal=False,
    )
    whitened = torch.linalg.solve_triangular(factor, out.unsqueeze(-1), upper=False).squeeze(-1)

    assert torch.allclose(whitened.norm(dim=-1), torch.ones(1), atol=1e-6)


def test_full_cov_failed_cholesky_falls_back_per_element() -> None:
    sigma_q = torch.tensor(
        [
            [[1.0, 0.9], [0.9, 1.0]],
            [[1.0, 2.0], [2.0, 1.0]],
        ]
    )
    delta_mu = torch.tensor([[0.0, 3.0], [0.0, 3.0]])

    out = apply_mu_trust_region(
        delta_mu,
        sigma_q,
        trust=1.0,
        mode="box",
        is_diagonal=False,
    )

    good_factor = torch.linalg.cholesky(sigma_q[:1])
    good_white = torch.linalg.solve_triangular(
        good_factor,
        delta_mu[:1].unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    expected_good = (good_factor @ good_white.clamp(-1.0, 1.0).unsqueeze(-1)).squeeze(-1)
    expected_bad = torch.tensor([[0.0, 1.0]])

    assert torch.allclose(out[:1], expected_good, atol=1e-6)
    assert torch.allclose(out[1:], expected_bad, atol=1e-6)
