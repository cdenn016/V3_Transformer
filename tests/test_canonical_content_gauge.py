r"""Exact canonical-content gauge encoder: provenance, constraints, and reference behavior."""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.families.frame_gaussian import FrameDiagonalGaussian
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import build_factored_transport
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank, get_encode


def _exact_cfg(**overrides: object) -> VFE3Config:
    r"""Small configuration on the exact canonical-content scientific control."""
    values = dict(
        vocab_size=11,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        batch_size=2,
        n_layers=1,
        n_e_steps=1,
        e_step_update="gradient",
        oracle_unroll_grad=True,
        family="gaussian_frame_diagonal",
        transport_mode="flat",
        gauge_parameterization="phi",
        prior_source="token",
        s_e_step=False,
        use_prior_bank=True,
        decode_mode="diagonal",
        encode_mode="canonical_content_gauge",
        omega_reflection="off",
        phi_reflection="off",
        lambda_alpha_mode="constant",
        max_steps=1,
    )
    values.update(overrides)
    return VFE3Config(**values)


def _bank(
    *,
    encode_mode: str = "canonical_content_gauge",
    untie_decode_bank: bool = False,
) -> PriorBank:
    torch.manual_seed(17)
    return PriorBank(
        vocab_size=9,
        K=4,
        n_gen=6,
        family="gaussian_frame_diagonal",
        encode_mode=encode_mode,
        decode_mode="diagonal",
        use_prior_bank=True,
        prior_source="token",
        s_e_step=False,
        gauge_parameterization="phi",
        omega_reflection="off",
        phi_reflection="off",
        untie_decode_bank=untie_decode_bank,
    )


def test_canonical_encoder_is_registered_and_matches_the_per_token_canonical_lookup() -> None:
    r"""Removing the named registration or adding a second table changes this control's result."""
    get_encode("canonical_content_gauge")
    canonical = _bank()
    per_token = _bank(encode_mode="per_token")
    token_ids = torch.tensor([[0, 4, 2], [7, 1, 8]])

    got = canonical.encode(token_ids)
    expected = per_token.encode(token_ids)

    assert torch.equal(got.mu, expected.mu)
    assert torch.equal(got.sigma, expected.sigma)
    assert torch.equal(got.phi, expected.phi)
    assert set(canonical._parameters) == set(per_token._parameters)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", "gaussian_diagonal"),
        ("transport_mode", "regime_ii"),
        ("gauge_parameterization", "omega_direct"),
        ("prior_source", "model_channel"),
        ("s_e_step", True),
        ("use_prior_bank", False),
        ("decode_mode", "family"),
        ("omega_reflection", "init_seed"),
        ("phi_reflection", "init_seed"),
    ],
)
def test_canonical_encoder_rejects_each_incompatible_cross_field_pair(
    field: str,
    value: object,
) -> None:
    r"""Relaxing any exact-control gate would silently change the experimental arm."""
    with pytest.raises(ValueError, match="canonical_content_gauge"):
        _exact_cfg(**{field: value})


def test_direct_prior_bank_construction_rejects_its_locally_knowable_incompatible_mode() -> None:
    r"""Changing the family beneath the named encoder must not bypass config validation."""
    with pytest.raises(ValueError, match="canonical_content_gauge"):
        PriorBank(
            vocab_size=9,
            K=4,
            n_gen=6,
            family="gaussian_diagonal",
            encode_mode="canonical_content_gauge",
        )


def test_canonical_tied_and_untied_banks_keep_canonical_coordinates_and_clone_decode_tables() -> None:
    r"""Untying must begin from the exact canonical table, not a pushed-forward surrogate."""
    tied = _bank()
    untied = _bank(untie_decode_bank=True)
    token_ids = torch.tensor([[3, 0, 5]])

    assert torch.equal(tied.encode(token_ids).mu, tied.mu_embed[token_ids])
    assert torch.equal(tied.encode(token_ids).sigma, torch.exp(tied.sigma_log_embed[token_ids]))
    assert torch.equal(untied.encode(token_ids).mu, untied.mu_embed[token_ids])
    assert torch.equal(untied.encode(token_ids).sigma, torch.exp(untied.sigma_log_embed[token_ids]))
    assert torch.equal(untied.decode_mu_embed, untied.mu_embed)
    assert torch.equal(untied.decode_sigma_log_embed, untied.sigma_log_embed)
    assert untied.decode_mu_embed.data_ptr() != untied.mu_embed.data_ptr()
    assert untied.decode_sigma_log_embed.data_ptr() != untied.sigma_log_embed.data_ptr()


def test_canonical_frame_transport_and_decode_match_the_hand_computed_intrinsic_reference() -> None:
    r"""A dense frame sandwich or fixed-basis decoder would violate the intrinsic KL reference."""
    torch.manual_seed(23)
    bank = _bank()
    with torch.no_grad():
        bank.mu_embed.copy_(torch.tensor([
            [0.0, 0.5, -0.5, 1.0],
            [1.0, -1.0, 0.25, 0.5],
            [-0.25, 0.75, 1.0, -0.5],
            [0.5, 0.0, -1.0, 0.25],
            [-0.5, -0.25, 0.0, 0.5],
            [0.75, 0.5, -0.25, -1.0],
            [-1.0, 0.25, 0.5, 0.0],
            [0.25, -0.75, 1.0, 0.5],
            [0.5, 1.0, 0.75, -0.25],
        ]))
        bank.sigma_log_embed.copy_(torch.log(torch.tensor([
            [0.5, 0.75, 1.0, 1.25],
            [1.5, 1.0, 0.75, 0.5],
            [0.75, 1.25, 1.5, 1.0],
            [1.0, 0.5, 1.25, 0.75],
            [1.25, 1.5, 0.5, 1.0],
            [0.5, 1.0, 1.5, 0.75],
            [1.0, 1.25, 0.75, 1.5],
            [1.5, 0.75, 1.0, 0.5],
            [0.75, 1.0, 1.25, 1.5],
        ])))
    query_mu = torch.tensor([[[0.25, -0.5, 0.75, 1.0]]])
    query_sigma = torch.tensor([[[0.75, 1.25, 0.5, 1.5]]])

    group = get_group("block_glk")(4, 2)
    phi = 0.1 * torch.randn(2, group.generators.shape[0])
    transport = build_factored_transport(phi, group)
    transported = FrameDiagonalGaussian.transport_location(query_mu[0], transport)
    assert torch.equal(transported[0], query_mu[0])

    sigma_v = torch.exp(bank.sigma_log_embed)
    delta = bank.mu_embed - query_mu[0, 0]
    expected = -0.5 * (
        (query_sigma[0, 0] / sigma_v)
        + (delta.square() / sigma_v)
        - 1.0
        + torch.log(sigma_v)
        - torch.log(query_sigma[0, 0])
    ).sum(dim=-1)
    got = bank.decode(query_mu, query_sigma)
    assert torch.allclose(got[0, 0], expected, atol=1e-6, rtol=1e-6)


def test_canonical_model_has_no_supervised_phi_gradient_and_emits_one_provenance_notice() -> None:
    r"""Reconnects of token phi to the belief/decode objective invalidate this exact control."""
    with pytest.warns(UserWarning, match="frame-intrinsic.*phi_embed.*supervised") as notices:
        cfg = _exact_cfg()
    assert len(notices) == 1

    torch.manual_seed(29)
    model = VFEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.prior_bank.phi_embed.grad is None
