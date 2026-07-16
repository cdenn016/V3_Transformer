"""Regression contract for the user-adjudicated decoder architecture."""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _tiny_model(*, use_prior_bank: bool) -> VFEModel:
    cfg = VFE3Config(
        vocab_size=7,
        embed_dim=4,
        n_heads=2,
        max_seq_len=3,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        use_prior_bank=use_prior_bank,
        decode_mode="diagonal",
    )
    return VFEModel(cfg)


@pytest.mark.parametrize("use_prior_bank", [True, False])
def test_both_decoder_routes_construct_and_operate(use_prior_bank: bool) -> None:
    model = _tiny_model(use_prior_bank=use_prior_bank)
    tokens = torch.tensor([[0, 1, 2]])

    logits = model(tokens)

    assert logits.shape == (1, 3, 7)
    assert torch.isfinite(logits).all()
    assert (model.prior_bank.output_proj_weight is None) is use_prior_bank


def test_prior_bank_route_is_geometric_and_has_no_affine_projection() -> None:
    model = _tiny_model(use_prior_bank=True)
    prior_bank = model.prior_bank
    mu_v = torch.tensor(
        [
            [0.0, 0.1, -0.2, 0.3],
            [0.4, -0.1, 0.2, 0.0],
            [-0.3, 0.5, 0.1, -0.2],
            [0.2, 0.2, -0.4, 0.1],
            [0.1, -0.3, 0.4, 0.2],
            [-0.2, 0.0, 0.3, 0.5],
            [0.3, -0.4, 0.0, -0.1],
        ]
    )
    sigma_v = torch.tensor(
        [
            [0.8, 1.1, 1.3, 0.9],
            [1.2, 0.7, 1.0, 1.4],
            [0.9, 1.5, 0.8, 1.1],
            [1.3, 0.9, 1.2, 0.7],
            [0.7, 1.4, 0.9, 1.2],
            [1.1, 0.8, 1.5, 1.0],
            [1.4, 1.0, 0.7, 1.3],
        ]
    )
    mu_q = torch.tensor([[[0.15, -0.05, 0.25, 0.10]]])
    sigma_q = torch.tensor([[[1.0, 0.9, 1.2, 0.8]]])
    with torch.no_grad():
        prior_bank.mu_embed.copy_(mu_v)
        prior_bank.sigma_log_embed.copy_(sigma_v.log())
        prior_bank.decode_log_scale.zero_()

    logits = prior_bank.decode(mu_q, sigma_q)
    delta = mu_q.unsqueeze(-2) - mu_v
    kl = 0.5 * (
        sigma_q.unsqueeze(-2) / sigma_v
        + delta.square() / sigma_v
        - 1.0
        + sigma_v.log()
        - sigma_q.unsqueeze(-2).log()
    ).sum(dim=-1)

    assert prior_bank.output_proj_weight is None
    assert "prior_bank.output_proj_weight" not in dict(model.named_parameters())
    assert torch.allclose(logits, -kl, atol=1e-6, rtol=1e-6)


def test_affine_route_remains_an_operational_projection() -> None:
    model = _tiny_model(use_prior_bank=False)
    prior_bank = model.prior_bank
    mu_q = torch.tensor([[[0.2, -0.1, 0.4, 0.3]]])
    sigma_q = torch.full_like(mu_q, 2.0)
    weight = torch.arange(28, dtype=mu_q.dtype).reshape(7, 4) / 10.0
    with torch.no_grad():
        prior_bank.output_proj_weight.copy_(weight)

    logits = prior_bank.decode(mu_q, sigma_q)

    assert isinstance(prior_bank.output_proj_weight, torch.nn.Parameter)
    assert torch.equal(logits, mu_q @ weight.t())
