"""Architectural contract for exact fast paths that are mandatory in production."""

import ast
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.block import e_step_shared_kwargs
from vfe3.model.model import VFEModel


ROOT = Path(__file__).resolve().parents[1]
FORMER_CONFIG_FIELDS = frozenset({
    "compact_phi_block_transport",
    "reuse_pairwise_kl_stats",
    "transport_mean_per_head",
})
PRODUCTION_MODULES = (
    "vfe3/model/block.py",
    "vfe3/model/model.py",
    "vfe3/viz/extract.py",
)


def _keyword_values(relative: str, keyword_name: str) -> list[ast.expr]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == keyword_name
    ]


def test_public_configuration_and_drivers_have_no_fastpath_controls() -> None:
    assert FORMER_CONFIG_FIELDS.isdisjoint(field.name for field in fields(VFE3Config))
    for relative in ("vfe3/config.py", "train_vfe3.py", "scaling.py", "ablation.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert all(name not in source for name in FORMER_CONFIG_FIELDS), relative


def test_production_does_not_read_removed_fastpath_attributes() -> None:
    for relative in PRODUCTION_MODULES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        reads = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in FORMER_CONFIG_FIELDS
        }
        assert reads == set(), relative


@pytest.mark.parametrize(
    ("relative", "keyword_name"),
    [
        ("vfe3/model/block.py", "reuse_pairwise_kl_stats"),
        ("vfe3/model/block.py", "transport_mean_per_head"),
        ("vfe3/model/model.py", "reuse_pairwise_kl_stats"),
        ("vfe3/model/model.py", "transport_mean_per_head"),
        ("vfe3/viz/extract.py", "transport_mean_per_head"),
    ],
)
def test_production_fastpath_requests_are_literal_true(
    relative:     str,
    keyword_name: str,
) -> None:
    values = _keyword_values(relative, keyword_name)
    assert values, (relative, keyword_name)
    assert all(isinstance(value, ast.Constant) and value.value is True for value in values)


def test_shared_e_step_kwargs_always_request_pairwise_reuse() -> None:
    kwargs = e_step_shared_kwargs(VFE3Config(), torch.device("cpu"))
    assert kwargs["reuse_pairwise_kl_stats"] is True


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"gauge_parameterization": "omega_direct", "pos_phi": "none"}, False),
        ({"transport_mode": "regime_ii"}, False),
        ({"phi_reflection": "init_seed"}, False),
        ({"n_heads": 1}, False),
    ],
)
def test_compact_phi_route_is_automatic(
    overrides: dict[str, object],
    expected:  bool,
) -> None:
    values: dict[str, object] = {
        "vocab_size": 9,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 3,
        "n_layers": 1,
        "n_e_steps": 1,
    }
    values.update(overrides)
    model = VFEModel(VFE3Config(**values)).eval()
    assert model._compact_phi_blocks_enabled() is expected
