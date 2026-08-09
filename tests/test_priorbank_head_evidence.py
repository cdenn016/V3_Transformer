"""Opt-in PriorBank-native head-evidence mixer state and ownership."""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank
from vfe3.numerics import bounded_variance_from_log
from vfe3.train import build_optimizer


_MISSING = object()


def diagonal_head_reference(mu_q, var_q, mu_v, var_v, weights, dims):
    pieces = []
    start = 0
    for weight, dim in zip(weights, dims):
        stop = start + dim
        term = (
            (var_q[..., start:stop, None] +
             (mu_q[..., start:stop, None] - mu_v[start:stop]).square()) /
            var_v[start:stop]
            + var_v[start:stop].log()
            - var_q[..., start:stop, None].log()
            - 1.0
        ).sum(dim=-2) * 0.5
        pieces.append(weight * term)
        start = stop
    return torch.stack(pieces).sum(dim=0)


def _reference_bank(
    mode: str,
    dims: tuple[int, ...],
    *,
    enabled: bool,
    unigram: bool = False,
) -> PriorBank:
    torch.manual_seed(711)
    bank = PriorBank(
        5,
        sum(dims),
        9,
        decode_tau=1.7,
        diagonal_covariance=not mode.startswith("full"),
        family="gaussian_full" if mode.startswith("full") else "gaussian_diagonal",
        decode_mode=mode,
        decode_chunk_size=2,
        decode_ce_checkpoint="off",
        decode_unigram_prior=unigram,
        unigram_kappa=0.37,
        irrep_dims=list(dims),
        use_priorbank_head_evidence_mixer=enabled,
    ).double()
    with torch.no_grad():
        bank.mu_embed.copy_(torch.tensor([
            [-0.7, 0.2, 0.9, -0.1, 0.4],
            [0.3, -0.8, 0.1, 0.5, -0.2],
            [0.6, 0.4, -0.5, -0.3, 0.7],
            [-0.2, 0.9, 0.3, -0.6, 0.1],
            [0.8, -0.1, -0.4, 0.2, -0.9],
        ], dtype=torch.float64))
        bank.sigma_log_embed.copy_(torch.tensor([
            [0.9, 1.4, 0.7, 1.8, 1.1],
            [1.6, 0.8, 1.2, 0.6, 1.5],
            [1.1, 1.7, 0.9, 1.3, 0.8],
            [0.7, 1.0, 1.6, 1.1, 1.9],
            [1.8, 1.2, 0.8, 1.5, 0.7],
        ], dtype=torch.float64).log())
        if enabled:
            bank.head_evidence_logits.copy_(torch.tensor([0.8, -0.45], dtype=torch.float64))
    if unigram:
        bank.set_unigram_log_prior(torch.tensor([2, 11, 5, 3, 17]))
    return bank


def _diagonal_query() -> tuple[torch.Tensor, torch.Tensor]:
    mu_q = torch.tensor([
        [[-0.4, 0.7, -0.2, 0.5, 0.1], [0.6, -0.3, 0.8, -0.7, 0.2]],
        [[0.9, 0.1, -0.6, 0.4, -0.5], [-0.8, 0.5, 0.3, -0.2, 0.7]],
    ], dtype=torch.float64, requires_grad=True)
    var_q = torch.tensor([
        [[1.3, 0.6, 1.7, 0.8, 1.1], [0.7, 1.9, 0.9, 1.4, 0.5]],
        [[1.8, 1.0, 0.6, 1.2, 1.5], [0.9, 1.6, 1.3, 0.7, 1.9]],
    ], dtype=torch.float64, requires_grad=True)
    return mu_q, var_q


def _full_query() -> tuple[torch.Tensor, torch.Tensor]:
    mu_q, _ = _diagonal_query()
    factors = torch.tensor([
        [1.2, 0.0, 0.0, 0.0, 0.0],
        [0.2, 1.0, 0.0, 0.0, 0.0],
        [0.35, -0.15, 1.1, 0.0, 0.0],
        [-0.1, 0.25, 0.2, 0.9, 0.0],
        [0.3, 0.1, -0.2, 0.15, 1.05],
    ], dtype=torch.float64)
    base = factors @ factors.T + 0.2 * torch.eye(5, dtype=torch.float64)
    cov_q = torch.stack((base, base * 1.13, base * 0.87, base * 1.29)).reshape(2, 2, 5, 5)
    return mu_q, cov_q.clone().requires_grad_()


def _full_head_reference(
    mu_q: torch.Tensor,
    cov_q: torch.Tensor,
    mu_v: torch.Tensor,
    var_v: torch.Tensor,
    weights: torch.Tensor,
    dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_kls = []
    block_slices = []
    start = 0
    for dim in dims:
        stop = start + dim
        sl = slice(start, stop)
        block_slices.append(sl)
        marginal = cov_q[..., sl, sl]
        diff = mu_q[..., sl, None] - mu_v[sl]
        kl_h = 0.5 * (
            (torch.diagonal(marginal, dim1=-2, dim2=-1)[..., None] + diff.square())
            .div(var_v[sl])
            .sum(dim=-2)
            + var_v[sl].log().sum(dim=-2)
            - torch.linalg.slogdet(marginal).logabsdet[..., None]
            - dim
        )
        block_kls.append(kl_h)
        start = stop
    correction = 0.5 * (
        sum(torch.linalg.slogdet(cov_q[..., sl, sl]).logabsdet for sl in block_slices)
        - torch.linalg.slogdet(cov_q).logabsdet
    )
    weighted = sum(w_h * kl_h for w_h, kl_h in zip(weights, block_kls))
    return weighted + correction[..., None], weighted, correction


def _restore_registry_entry(registry: dict[str, object], name: str, previous: object) -> None:
    if previous is _MISSING:
        registry.pop(name, None)
    else:
        registry[name] = previous


def _enabled_cfg(**overrides: object) -> VFE3Config:
    values = {
        "vocab_size": 11,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 8,
        "n_layers": 1,
        "gauge_group": "block_glk",
        "use_prior_bank": True,
        "use_priorbank_head_evidence_mixer": True,
        "family": "gaussian_diagonal",
        "divergence_family": "renyi",
        "renyi_order": 1.0,
        "decode_mode": "diagonal_chunked",
    }
    values.update(overrides)
    return VFE3Config(**values)


def test_head_evidence_is_default_off_and_does_not_change_state_dict():
    assert VFE3Config().use_priorbank_head_evidence_mixer is False
    torch.manual_seed(7)
    base = PriorBank(11, 4, 8, irrep_dims=[2, 2], use_prior_bank=True)
    torch.manual_seed(7)
    explicit = PriorBank(
        11,
        4,
        8,
        irrep_dims=[2, 2],
        use_prior_bank=True,
        use_priorbank_head_evidence_mixer=False,
    )
    assert base.state_dict().keys() == explicit.state_dict().keys()
    for key in base.state_dict():
        torch.testing.assert_close(base.state_dict()[key], explicit.state_dict()[key], rtol=0, atol=0)
    assert not hasattr(base, "head_evidence_logits")


def test_zero_logits_produce_identity_head_and_coordinate_weights():
    pb = PriorBank(
        11,
        4,
        8,
        irrep_dims=[1, 3],
        use_prior_bank=True,
        use_priorbank_head_evidence_mixer=True,
    )
    head, coord = pb.head_evidence_weights(dtype=torch.float64, device=torch.device("cpu"))
    torch.testing.assert_close(head, torch.ones(2, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(coord, torch.ones(4, dtype=torch.float64), rtol=0, atol=0)


def test_head_evidence_parameter_is_enabled_only_and_starts_at_exact_zero():
    off = PriorBank(11, 4, 8, irrep_dims=[2, 2], use_prior_bank=True)
    on = PriorBank(
        11,
        4,
        8,
        irrep_dims=[2, 2],
        use_prior_bank=True,
        use_priorbank_head_evidence_mixer=True,
    )
    assert not hasattr(off, "head_evidence_logits")
    assert isinstance(on.head_evidence_logits, torch.nn.Parameter)
    torch.testing.assert_close(on.head_evidence_logits, torch.zeros(2), rtol=0, atol=0)


def test_disabled_head_evidence_preserves_caller_irrep_dims_list_identity():
    dims = [2, 2]
    pb = PriorBank(11, 4, 8, irrep_dims=dims, use_prior_bank=True)
    assert pb.irrep_dims is dims
    assert pb.irrep_dims == [2, 2]


@pytest.mark.registry_mutation
def test_head_evidence_rejects_overridden_builtin_family_registration():
    from vfe3.families.base import _FAMILIES, register_family
    from vfe3.families.gaussian import DiagonalGaussian

    name = "gaussian_diagonal"
    previous = _FAMILIES.get(name, _MISSING)
    try:
        @register_family(name, override=True)
        class _ReplacementDiagonal(DiagonalGaussian):
            pass

        with pytest.raises(ValueError, match="built-in canonical registry identities"):
            _enabled_cfg()
    finally:
        _restore_registry_entry(_FAMILIES, name, previous)


@pytest.mark.registry_mutation
def test_head_evidence_rejects_overridden_builtin_renyi_functional():
    from vfe3.families.base import _FUNCTIONALS, register_functional

    name = "renyi"
    previous = _FUNCTIONALS.get(name, _MISSING)
    try:
        @register_functional(name, override=True)
        def _replacement_renyi(*args: object, **kwargs: object) -> torch.Tensor:
            return previous(*args, **kwargs)

        with pytest.raises(ValueError, match="built-in canonical registry identities"):
            _enabled_cfg()
    finally:
        _restore_registry_entry(_FUNCTIONALS, name, previous)


@pytest.mark.registry_mutation
def test_head_evidence_rejects_replaced_canonical_decoder_registration():
    from vfe3.model import prior_bank as prior_bank_mod

    name = "diagonal_chunked"
    previous = prior_bank_mod._DECODERS[name]
    try:
        prior_bank_mod.register_decode(
            name,
            supports_full=previous.supports_full,
            supports_chunked=previous.supports_chunked,
            fused_ce=previous.fused_ce,
            family_consistent=previous.family_consistent,
            covariance_kinds=previous.covariance_kinds,
            can_omit_base_mean=previous.can_omit_base_mean,
            can_omit_base_variance=previous.can_omit_base_variance,
            override=True,
        )(previous.callable)

        with pytest.raises(ValueError, match="built-in canonical registry identities"):
            _enabled_cfg()
    finally:
        prior_bank_mod._DECODERS[name] = previous


@pytest.mark.registry_mutation
def test_head_evidence_rejects_replaced_canonical_decoder_callable():
    from vfe3.model import prior_bank as prior_bank_mod

    name = "diagonal_chunked"
    previous = prior_bank_mod._DECODERS[name]
    try:
        @prior_bank_mod.register_decode(
            name,
            supports_full=previous.supports_full,
            supports_chunked=previous.supports_chunked,
            fused_ce=previous.fused_ce,
            family_consistent=previous.family_consistent,
            covariance_kinds=previous.covariance_kinds,
            can_omit_base_mean=previous.can_omit_base_mean,
            can_omit_base_variance=previous.can_omit_base_variance,
            override=True,
        )
        def _replacement_decoder(*args: object, **kwargs: object) -> torch.Tensor:
            return previous.callable(*args, **kwargs)

        with pytest.raises(ValueError, match="built-in canonical registry identities"):
            _enabled_cfg()
    finally:
        prior_bank_mod._DECODERS[name] = previous


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"use_prior_bank": False}, "use_prior_bank=True"),
        ({"gauge_group": "glk"}, "at least two gauge blocks"),
        ({"renyi_order": 0.5}, "divergence_family='renyi'.*renyi_order=1.0"),
        ({"divergence_family": "squared_hellinger"}, "divergence_family='renyi'.*renyi_order=1.0"),
        ({"decode_mode": "family_chunked"}, "canonical KL decode"),
        ({"family": "gaussian_full", "decode_mode": "diagonal_chunked"}, "gaussian_full.*full.*full_chunked"),
    ],
)
def test_head_evidence_rejects_invalid_configurations(overrides: dict[str, object], match: str):
    with pytest.raises(ValueError, match=match):
        _enabled_cfg(**overrides)


def test_optimizer_owns_head_evidence_logits_once_with_mu_hyperparameters():
    cfg = _enabled_cfg(m_p_mu_lr=0.0123)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    owned = [
        group for group in optimizer.param_groups
        if any(parameter is model.prior_bank.head_evidence_logits for parameter in group["params"])
    ]
    assert len(owned) == 1
    group = owned[0]
    assert group["lr"] == cfg.m_p_mu_lr
    assert group["weight_decay"] == 0.0
    assert group["role"] == "mu"


def test_head_mixer_and_head_evidence_are_legal_and_separately_owned():
    cfg = _enabled_cfg(use_head_mixer=True)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    evidence_groups = [
        group for group in optimizer.param_groups
        if any(parameter is model.prior_bank.head_evidence_logits for parameter in group["params"])
    ]
    mixer_parameters = tuple(model.head_mixer.parameters())
    mixer_groups = [
        group for group in optimizer.param_groups
        if any(parameter is candidate for parameter in group["params"] for candidate in mixer_parameters)
    ]
    assert model.head_mixer is not None
    assert len(evidence_groups) == 1
    assert len(mixer_groups) == 1
    assert evidence_groups[0] is not mixer_groups[0]


@pytest.mark.parametrize("mode", ["diagonal", "diagonal_chunked"])
def test_diagonal_reference_matches_nonuniform_unequal_head_blocks_and_gradients(mode: str):
    dims = (2, 3)
    bank = _reference_bank(mode, dims, enabled=True)
    mu_q, var_q = _diagonal_query()
    head, _ = bank.head_evidence_weights(dtype=mu_q.dtype, device=mu_q.device)
    mu_v = bank._decode_mu_table().T
    var_v = bounded_variance_from_log(bank._decode_sigma_log_table(), eps=bank.eps).T
    expected_divergence = diagonal_head_reference(mu_q, var_q, mu_v, var_v, head, dims)
    expected = -expected_divergence / bank._tau_eff()
    actual = bank.decode(mu_q, var_q)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    probe = torch.linspace(-0.9, 1.1, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    leaves = (mu_q, var_q, bank.mu_embed, bank.sigma_log_embed, bank.head_evidence_logits)
    actual_grads = torch.autograd.grad((actual * probe).sum(), leaves, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), leaves)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-11, atol=2e-11)
    assert bank.head_evidence_logits.grad is None
    assert actual_grads[-1].abs().sum() > 0


@pytest.mark.parametrize("mode", ["full", "full_chunked"])
def test_full_reference_retains_unweighted_cross_block_correction_and_gradients(mode: str):
    dims = (2, 3)
    bank = _reference_bank(mode, dims, enabled=True)
    mu_q, cov_q = _full_query()
    head, _ = bank.head_evidence_weights(dtype=mu_q.dtype, device=mu_q.device)
    mu_v = bank._decode_mu_table().T
    var_v = bounded_variance_from_log(bank._decode_sigma_log_table(), eps=bank.eps).T
    expected_divergence, weighted_blocks, correction = _full_head_reference(
        mu_q, cov_q, mu_v, var_v, head, dims)
    assert correction.abs().min() > 0
    expected = -expected_divergence / bank._tau_eff()
    actual = bank.decode(mu_q, cov_q)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)

    probe = torch.linspace(1.1, -0.7, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    leaves = (mu_q, cov_q, bank.mu_embed, bank.sigma_log_embed, bank.head_evidence_logits)
    actual_grads = torch.autograd.grad((actual * probe).sum(), leaves, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), leaves, retain_graph=True)
    weighted_only_logit_grad = torch.autograd.grad(
        ((-weighted_blocks / bank._tau_eff()) * probe).sum(), bank.head_evidence_logits)[0]
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=4e-10, atol=4e-10)
    torch.testing.assert_close(actual_grads[-1], weighted_only_logit_grad, rtol=2e-11, atol=2e-11)
    assert actual_grads[-1].abs().sum() > 0


@pytest.mark.parametrize("mode", ["diagonal", "diagonal_chunked", "full", "full_chunked"])
def test_initial_identity_is_bitwise_for_outputs_and_legacy_gradients(mode: str):
    dims = (2, 3)
    disabled = _reference_bank(mode, dims, enabled=False, unigram=True).float()
    enabled = _reference_bank(mode, dims, enabled=True, unigram=True).float()
    incompatible = enabled.load_state_dict(disabled.state_dict(), strict=False)
    assert incompatible.missing_keys == ["head_evidence_logits"]
    assert incompatible.unexpected_keys == []
    with torch.no_grad():
        enabled.head_evidence_logits.zero_()

    query_factory = _full_query if mode.startswith("full") else _diagonal_query
    mu_off_raw, sigma_off_raw = query_factory()
    mu_off = mu_off_raw.float().detach().requires_grad_()
    sigma_off = sigma_off_raw.float().detach().requires_grad_()
    mu_on = mu_off.detach().clone().requires_grad_()
    sigma_on = sigma_off.detach().clone().requires_grad_()
    logits_off = disabled.decode(mu_off, sigma_off)
    logits_on = enabled.decode(mu_on, sigma_on)
    assert torch.equal(logits_on, logits_off)

    probe = torch.linspace(-0.6, 0.8, logits_off.numel(), dtype=logits_off.dtype).reshape_as(logits_off)
    off_leaves = (mu_off, sigma_off, disabled.mu_embed, disabled.sigma_log_embed)
    on_leaves = (mu_on, sigma_on, enabled.mu_embed, enabled.sigma_log_embed)
    off_grads = torch.autograd.grad((logits_off * probe).sum(), off_leaves)
    on_grads = torch.autograd.grad(
        (logits_on * probe).sum(), on_leaves + (enabled.head_evidence_logits,))
    for off_grad, on_grad in zip(off_grads, on_grads[:-1]):
        assert torch.equal(on_grad, off_grad)
    assert on_grads[-1].abs().sum() > 0
