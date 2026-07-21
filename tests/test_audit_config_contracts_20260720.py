"""Regressions for exact public configuration and data-loader contracts."""

import re
from dataclasses import fields

import pytest

from vfe3.config import VFE3Config
from vfe3.data import datasets as dsmod


_EXACT_CONFIG_TYPE_CASES = [
    pytest.param(
        field.name,
        "false" if field.type is bool else True,
        id=f"{field.name}-rejects-nonexact-{field.type.__name__}",
    )
    for field in fields(VFE3Config)
    if field.type in (bool, int)
] + [
    pytest.param("n_layers", 1.5, id="n_layers-rejects-float"),
    pytest.param("e_steps_backprop_last", 0.5, id="e_steps_backprop_last-rejects-float"),
]


@pytest.mark.parametrize(("field_name", "wrong_value"), _EXACT_CONFIG_TYPE_CASES)
def test_exact_bool_and_int_config_fields_reject_other_plain_types(field_name, wrong_value):
    with pytest.raises(TypeError, match=rf"\b{re.escape(field_name)}\b"):
        VFE3Config(**{field_name: wrong_value})


def test_include_attention_entropy_rejects_string_false():
    with pytest.raises(TypeError, match=r"\binclude_attention_entropy\b"):
        VFE3Config(include_attention_entropy="false")


def test_additive_encoder_rejects_pullback_group_update():
    with pytest.raises(ValueError, match="per_token_additive.*pullback_group"):
        VFE3Config(
            embed_dim=10,
            n_heads=1,
            gauge_group="glk",
            gauge_parameterization="phi",
            m_phi_update_mode="pullback_group",
            phi_precond_mode="pullback",
            transport_chart_max_norm=6.0,
            pos_phi="none",
            encode_mode="per_token_additive",
        )


def test_irrep_multiplicity_rejects_bool():
    with pytest.raises(ValueError, match="mult.*plain int"):
        VFE3Config(
            embed_dim=1,
            n_heads=1,
            gauge_group="so_n",
            group_n=3,
            irrep_spec=[("l0", True)],
        )


@pytest.mark.parametrize(
    ("argument", "wrong_value"),
    [
        pytest.param("shuffle", "false", id="shuffle-rejects-string"),
        pytest.param("drop_last", 1, id="drop_last-rejects-int"),
    ],
)
def test_make_dataloader_rejects_nonexact_booleans_before_side_effects(
    monkeypatch,
    argument,
    wrong_value,
):
    def _unexpected_side_effect(*args, **kwargs):
        del args, kwargs
        pytest.fail("invalid loader booleans must fail before data loading or construction")

    monkeypatch.setattr(dsmod, "_load_identity_bound_tokens", _unexpected_side_effect)
    monkeypatch.setattr(dsmod, "TokenWindows", _unexpected_side_effect)
    monkeypatch.setattr(dsmod, "DataLoader", _unexpected_side_effect)

    with pytest.raises(TypeError, match=rf"\b{argument}\b"):
        dsmod.make_dataloader("dataset", "train", 8, 2, **{argument: wrong_value})
