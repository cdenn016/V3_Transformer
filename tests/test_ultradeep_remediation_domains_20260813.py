import copy

import pytest
import torch
import torch.nn.functional as F
import vfe3.model.prior_bank as prior_bank_module

from tests.test_amp import _tiny_model
from vfe3.config import VFE3Config
from vfe3.families.gaussian import DiagonalGaussian, FullGaussian
from vfe3.families.laplace import DiagonalLaplace
from vfe3.geometry.groups import get_group
from vfe3.model.cg_coupling import CGCoupling
from vfe3.model.prior_bank import DecodeCEResult, PriorBank, set_decode_av_precision


@pytest.fixture(autouse=True)
def _restore_decode_precision():
    previous = set_decode_av_precision("fp32")
    yield
    set_decode_av_precision(previous)


def test_delta_full_covariant_floor_keeps_literal_singular_jacobian_spd():
    group = get_group("so_n")(
        5, group_n=3, irrep_spec=[("l2", 1)], dtype=torch.float64)
    coupling = CGCoupling(
        3,
        "so",
        group.irrep_dims,
        group.irrep_labels,
        cg_covariance_mode="delta_full",
        cg_covariance_floor=1e-3,
    ).double()
    with torch.no_grad():
        coupling.path_weights.fill_(0.28203803740888295)
    mu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    sigma = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64))

    result = coupling.forward_moments(mu, sigma)
    pushed = result.jacobian @ sigma @ result.jacobian.transpose(-1, -2)
    expected = 0.5 * (pushed + pushed.transpose(-1, -2)) + 1e-3 * sigma

    assert torch.linalg.svdvals(result.jacobian)[-1] < 1e-14
    assert torch.allclose(result.sigma, expected, atol=1e-15, rtol=0.0)
    assert torch.linalg.eigvalsh(result.sigma)[0] > 0.0


@pytest.mark.parametrize("floor", [0.0, -1e-6, float("nan"), float("inf")])
def test_delta_full_rejects_nonpositive_or_nonfinite_covariant_floor(floor):
    group = get_group("so_n")(
        5, group_n=3, irrep_spec=[("l2", 1)], dtype=torch.float64)
    with pytest.raises(ValueError, match="cg_covariance_floor"):
        CGCoupling(
            3,
            "so",
            group.irrep_dims,
            group.irrep_labels,
            cg_covariance_mode="delta_full",
            cg_covariance_floor=floor,
        )


def test_invalid_reflection_checkpoint_fails_before_any_prior_bank_mutation():
    bank = PriorBank(
        4, 2, 1, gauge_parameterization="phi", phi_reflection="metropolis")
    before = {name: value.detach().clone() for name, value in bank.state_dict().items()}
    invalid = copy.deepcopy(bank.state_dict())
    invalid["mu_embed"] = torch.full_like(invalid["mu_embed"], 17.0)
    invalid["reflection_sign"] = torch.tensor([1.0, -1.0, 0.0, 1.0])

    with pytest.raises(RuntimeError, match=r"reflection_sign.*exact.*\{-1, \+1\}"):
        bank.load_state_dict(invalid, strict=True)

    after = bank.state_dict()
    assert all(torch.equal(after[name], value) for name, value in before.items())


def test_renyi_above_one_is_rejected_only_for_family_consistent_prior_decode():
    common = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4,
                  family="gaussian_diagonal", divergence_family="renyi",
                  renyi_order=1.5)
    with pytest.raises(ValueError, match=r"family-consistent.*renyi_order.*<= 1"):
        VFE3Config(**common, use_prior_bank=True, decode_mode="family")

    linear = VFE3Config(**common, use_prior_bank=False, decode_mode="family")
    assert linear.renyi_order == 1.5


def test_family_chunked_all_invalid_row_is_explicitly_excluded_without_nan():
    bank = PriorBank(
        3,
        2,
        1,
        family="gaussian_diagonal",
        divergence_family="renyi",
        renyi_order=1.5,
        decode_mode="family_chunked",
        decode_chunk_size=2,
    )
    with torch.no_grad():
        bank.mu_embed.zero_()
        bank.sigma_log_embed.zero_()
    mu = torch.zeros(1, 2, 2, requires_grad=True)
    sigma = torch.full((1, 2, 2), 10.0, requires_grad=True)
    targets = torch.tensor([[0, 2]])

    result = bank.decode_ce_family_chunked(mu, sigma, targets, return_stats=True)

    assert isinstance(result, DecodeCEResult)
    assert torch.equal(result.ce, torch.tensor(0.0))
    assert int(result.scored_tokens) == 0
    assert int(result.excluded_tokens) == 2
    assert torch.isfinite(result.ce)
    result.ce.backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(sigma.grad).all()


def test_fp32_escalate_uses_error_certificate_and_retains_accurate_float64_kl():
    mu_q = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    mu_p = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
    sigma_q = torch.tensor([[[1.0, 0.0], [0.0, 1e-7]]], dtype=torch.float32)
    sigma_p = torch.tensor([[[2.0, 0.0], [0.0, 2e-7]]], dtype=torch.float32)
    query = FullGaussian(mu_q, sigma_q, _precision_policy="fp32_escalate")
    prior = FullGaussian(mu_p, sigma_p, _precision_policy="fp32_escalate")
    reference = FullGaussian(
        mu_q.double(), sigma_q.double(), _precision_policy="fp64"
    ).renyi_closed_form(
        FullGaussian(mu_p.double(), sigma_p.double(), _precision_policy="fp64"),
        alpha=1.0,
        kl_max=float("inf"),
    )

    got = query.renyi_closed_form(prior, alpha=1.0, kl_max=float("inf"))

    assert got.dtype is torch.float64
    assert torch.equal(got, reference)
    assert abs(float(reference) - 100000.1975) < 1e-3


def test_near_tied_expanded_decoder_matches_promoted_reference_ranking():
    bank = PriorBank(3, 4, 1, mu_init_std=0.0, decode_mode="diagonal")
    table = torch.tensor([
        [10000.0068359375, 9999.9990234375, 10000.001953125, 9999.990234375],
        [9999.953125, 10000.0146484375, 9999.9765625, 9999.982421875],
        [9999.9873046875, 10000.009765625, 9999.99609375, 9999.9716796875],
    ])
    with torch.no_grad():
        bank.mu_embed.copy_(table)
        bank.sigma_log_embed.fill_(-9.0)
    mu = torch.tensor([[[
        10000.01171875, 9999.9912109375, 10000.0234375, 9999.9501953125,
    ]]])
    sigma = torch.full((1, 1, 4), 1e-4)
    promoted = copy.deepcopy(bank).double()
    set_decode_av_precision("fp64")
    reference = promoted.decode(mu.double(), sigma.double())
    set_decode_av_precision("fp32")

    logits = bank.decode(mu, sigma)

    assert int(reference.argmax(dim=-1)) == 0
    assert int(logits.argmax(dim=-1)) == int(reference.argmax(dim=-1))


def test_laplace_public_divergence_and_natural_gradient_promote_mixed_inputs():
    query = DiagonalLaplace(
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[0.5, 2.0]], dtype=torch.float64),
    )
    prior = DiagonalLaplace(
        torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        torch.tensor([[1.5, 0.75]], dtype=torch.float32),
    )
    grad_mu = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    grad_sigma = torch.tensor([[0.5, 0.25]], dtype=torch.float64)

    divergence = query.renyi_closed_form(prior, alpha=0.5)
    nat_mu, nat_sigma = query.natural_gradient(grad_mu, grad_sigma)

    assert divergence.dtype is torch.float64
    assert nat_mu.dtype is torch.float64 and nat_sigma.dtype is torch.float64


def test_decode_last_is_bit_identical_to_last_position_of_full_decoder():
    model = _tiny_model(
        gauge_group="block_glk", n_heads=2, family="gaussian_full",
        decode_mode="full_chunked")
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    with torch.no_grad():
        _, full_logits = model.forward_beliefs(
            token_ids, return_logits=True, decode_last=False)
        _, last_logits = model.forward_beliefs(
            token_ids, return_logits=True, decode_last=True)

    assert torch.equal(last_logits, full_logits[:, -1:])


@pytest.mark.parametrize("checkpoint_mode", ["never", "always"])
@pytest.mark.parametrize("chunk_width", [2, 3])
def test_family_chunked_partial_invalid_vocab_matches_dense_finite_mask_value_and_gradients(
    checkpoint_mode, chunk_width,
):
    vocab = 2 * chunk_width
    bank = PriorBank(
        vocab, 2, 1, family="gaussian_diagonal", divergence_family="renyi",
        renyi_order=1.5, decode_mode="family_chunked", decode_chunk_size=chunk_width,
        decode_ce_checkpoint=checkpoint_mode,
    )
    oracle_bank = copy.deepcopy(bank)
    with torch.no_grad():
        # alpha=1.5 blend = -0.5 Sigma_q + 1.5 Sigma_v. Sigma_q=10 makes
        # the first chunk finite (Sigma_v=4) and the second wholly invalid (Sigma_v=1).
        bank.mu_embed.zero_()
        bank.mu_embed[:chunk_width, 0] = torch.arange(chunk_width) * 0.25
        bank.sigma_log_embed[:chunk_width].fill_(torch.log(torch.tensor(4.0)))
        bank.sigma_log_embed[chunk_width:].zero_()
        oracle_bank.load_state_dict(bank.state_dict())

    mu = torch.tensor([[[0.1, -0.2]]], requires_grad=True)
    sigma = torch.full((1, 1, 2), 10.0, requires_grad=True)
    mu_ref = mu.detach().clone().requires_grad_(True)
    sigma_ref = sigma.detach().clone().requires_grad_(True)
    targets = torch.tensor([[0]])

    got = bank.decode_ce_family_chunked(
        mu, sigma, targets, return_stats=True, chunk_size=chunk_width)

    q = DiagonalGaussian(mu_ref.unsqueeze(-2), sigma_ref.unsqueeze(-2))
    p = DiagonalGaussian(
        oracle_bank.mu_embed[:chunk_width],
        oracle_bank.sigma_log_embed[:chunk_width].exp(),
    )
    finite_logits = -q.renyi_closed_form(
        p, alpha=1.5, kl_max=float("inf"), eps=bank.eps)
    expected = F.cross_entropy(finite_logits.reshape(1, chunk_width), targets.reshape(-1))

    torch.testing.assert_close(got.ce, expected, atol=2e-7, rtol=2e-7)
    assert int(got.scored_tokens) == 1 and int(got.excluded_tokens) == 0
    got.ce.backward()
    expected.backward()
    torch.testing.assert_close(mu.grad, mu_ref.grad, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(sigma.grad, sigma_ref.grad, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(
        bank.mu_embed.grad[:chunk_width], oracle_bank.mu_embed.grad[:chunk_width],
        atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(
        bank.sigma_log_embed.grad[:chunk_width],
        oracle_bank.sigma_log_embed.grad[:chunk_width], atol=2e-7, rtol=2e-7)
    assert torch.equal(
        bank.mu_embed.grad[chunk_width:],
        torch.zeros_like(bank.mu_embed.grad[chunk_width:]))
    assert torch.equal(
        bank.sigma_log_embed.grad[chunk_width:],
        torch.zeros_like(bank.sigma_log_embed.grad[chunk_width:]))


def _assert_fused_and_dense_match_promoted_value_and_gradients(
    bank, mu, sigma, targets, *, chunk_size,
):
    fast_bank = copy.deepcopy(bank)
    dense_bank = copy.deepcopy(bank)
    ref_bank = copy.deepcopy(bank).double()
    mu_fast = mu.detach().clone().requires_grad_(True)
    sigma_fast = sigma.detach().clone().requires_grad_(True)
    mu_dense = mu.detach().clone().requires_grad_(True)
    sigma_dense = sigma.detach().clone().requires_grad_(True)
    mu_ref = mu.detach().double().clone().requires_grad_(True)
    sigma_ref = sigma.detach().double().clone().requires_grad_(True)

    fused = (
        fast_bank.decode_ce_full_chunked
        if sigma.dim() == 4 else fast_bank.decode_ce_diagonal_chunked
    )(mu_fast, sigma_fast, targets, chunk_size=chunk_size, return_stats=True)
    dense_logits = dense_bank.decode(mu_dense, sigma_dense)
    dense_ce = F.cross_entropy(
        dense_logits.reshape(-1, bank.vocab_size), targets.reshape(-1))
    previous = set_decode_av_precision("fp64")
    try:
        ref_logits = ref_bank.decode(mu_ref, sigma_ref)
        ref_ce = F.cross_entropy(
            ref_logits.reshape(-1, bank.vocab_size), targets.reshape(-1))
    finally:
        set_decode_av_precision(previous)

    assert fused.ce.dtype is torch.float64
    assert dense_logits.dtype is torch.float64
    torch.testing.assert_close(fused.ce, ref_ce, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(dense_ce, ref_ce, atol=2e-12, rtol=2e-12)
    assert int(fused.scored_tokens) == targets.numel()
    assert int(fused.excluded_tokens) == 0

    fused.ce.backward()
    dense_ce.backward()
    ref_ce.backward()
    for got_grad, want_grad in (
        (mu_fast.grad, mu_ref.grad), (sigma_fast.grad, sigma_ref.grad),
        (mu_dense.grad, mu_ref.grad), (sigma_dense.grad, sigma_ref.grad),
        (fast_bank.mu_embed.grad, ref_bank.mu_embed.grad),
        (fast_bank.sigma_log_embed.grad, ref_bank.sigma_log_embed.grad),
        (dense_bank.mu_embed.grad, ref_bank.mu_embed.grad),
        (dense_bank.sigma_log_embed.grad, ref_bank.sigma_log_embed.grad),
    ):
        torch.testing.assert_close(
            got_grad.double(), want_grad, atol=5e-6, rtol=1e-6)


@pytest.mark.parametrize("chunk_size", [1, 2])
@pytest.mark.parametrize("full", [False, True])
def test_global_cross_chunk_near_tie_promotes_complete_dense_and_fused_decode(chunk_size, full):
    bank = PriorBank(
        3, 4, 1, mu_init_std=0.0,
        family="gaussian_full" if full else "gaussian_diagonal",
        decode_mode="full_chunked" if full else "diagonal_chunked",
    )
    table = torch.tensor([
        [10000.0068359375, 9999.9990234375, 10000.001953125, 9999.990234375],
        [9999.953125, 10000.0146484375, 9999.9765625, 9999.982421875],
        [9999.9873046875, 10000.009765625, 9999.99609375, 9999.9716796875],
    ])
    with torch.no_grad():
        bank.mu_embed.copy_(table)
        bank.sigma_log_embed.fill_(-9.0)
    mu = torch.tensor([[[10000.01171875, 9999.9912109375, 10000.0234375, 9999.9501953125]]])
    diagonal = torch.full((1, 1, 4), 1e-4)
    sigma = torch.diag_embed(diagonal) if full else diagonal
    _assert_fused_and_dense_match_promoted_value_and_gradients(
        bank, mu, sigma, torch.tensor([[0]]), chunk_size=chunk_size)


def test_global_final_ranking_includes_unigram_cancellation_value_and_gradients():
    bank = PriorBank(
        2, 2, 1, mu_init_std=0.0, decode_mode="diagonal_chunked",
        decode_unigram_prior=True,
    )
    with torch.no_grad():
        bank.mu_embed.copy_(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))
        bank.sigma_log_embed.zero_()
        bank.unigram_log_prior.copy_(torch.tensor([-20.0, -7.5000005]))
        bank._unigram_set = True
    _assert_fused_and_dense_match_promoted_value_and_gradients(
        bank, torch.zeros(1, 1, 2), torch.ones(1, 1, 2), torch.tensor([[0]]),
        chunk_size=1)


def test_global_final_ranking_includes_head_evidence_cancellation_value_and_gradients():
    bank = PriorBank(
        2, 2, 1, mu_init_std=0.0, decode_mode="diagonal_chunked",
        irrep_dims=[1, 1], use_priorbank_head_evidence_mixer=True,
    )
    with torch.no_grad():
        bank.mu_embed.copy_(torch.tensor([[1.0, 0.0], [1.0, 10.0]]))
        bank.sigma_log_embed.zero_()
        bank.head_evidence_logits.copy_(torch.tensor([10.0, -10.0]))
    _assert_fused_and_dense_match_promoted_value_and_gradients(
        bank, torch.zeros(1, 1, 2), torch.ones(1, 1, 2), torch.tensor([[0]]),
        chunk_size=1)


@pytest.mark.parametrize("chunk_size", [1, 2])
@pytest.mark.parametrize("full", [False, True])
def test_global_interval_nonoverlap_keeps_fp32_fast_path(
    monkeypatch, chunk_size, full,
):
    bank = PriorBank(
        3, 2, 1, mu_init_std=0.0,
        family="gaussian_full" if full else "gaussian_diagonal",
        decode_mode="full_chunked" if full else "diagonal_chunked",
    )
    with torch.no_grad():
        bank.mu_embed.copy_(
            torch.tensor([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]]))
        bank.sigma_log_embed.zero_()
    mu = torch.zeros(1, 1, 2, requires_grad=True)
    diagonal = torch.ones(1, 1, 2)
    sigma = torch.diag_embed(diagonal) if full else diagonal

    def _unexpected_promotion(*args, **kwargs):
        raise AssertionError("non-overlapping score intervals must not recompute")

    monkeypatch.setattr(
        prior_bank_module, "_promoted_expanded_decode", _unexpected_promotion)
    logits = bank.decode(mu, sigma)
    fused = (
        bank.decode_ce_full_chunked
        if full else bank.decode_ce_diagonal_chunked
    )(mu, sigma, torch.tensor([[0]]), chunk_size=chunk_size, return_stats=True)
    assert logits.dtype is torch.float32
    assert fused.ce.dtype is torch.float32
    assert int(fused.scored_tokens) == 1 and int(fused.excluded_tokens) == 0
