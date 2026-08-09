"""Opt-in PriorBank-native head-evidence mixer state and ownership."""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank
from vfe3.train import build_optimizer


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
