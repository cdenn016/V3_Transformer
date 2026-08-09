"""Opt-in PriorBank-native head-evidence mixer state and ownership."""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank
from vfe3.train import build_optimizer


_MISSING = object()


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
