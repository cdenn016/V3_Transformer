import inspect

import pytest
import torch

import ablation
import scaling
import scaling_analysis
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext
from vfe3.model import block_mlp
from vfe3.model.block_mlp import build_block_mlp
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import DecodeRegistration
from vfe3.run_artifacts import _pure_path_report


def _block_mlp_kwargs(mode: str, covariance: str = "passthrough") -> dict:
    return {
        "mode": mode,
        "embed_dim": 4,
        "irrep_dims": (2, 2),
        "expansion": 2,
        "activation": "gelu",
        "dropout": 0.0,
        "covariance_contract": covariance,
        "covariance_floor": 1e-4,
    }


def test_all_registered_block_mlps_share_frame_context_protocol() -> None:
    mu = torch.zeros(1, 2, 4)
    sigma = torch.eye(4).expand(1, 2, 4, 4).clone()
    frame_context = CanonicalFrameContext(torch.eye(4), torch.eye(4))

    registrations = getattr(block_mlp, "BLOCK_MLP_REGISTRATIONS", {})
    assert registrations, "BlockMLP mode registry is missing"
    for mode in registrations:
        mlp = build_block_mlp(**_block_mlp_kwargs(mode))
        assert "frame_context" in inspect.signature(mlp.forward_moments).parameters
        result = mlp.forward_moments(mu, sigma, frame_context=frame_context)
        assert result.mu.shape == mu.shape
        assert result.sigma.shape == sigma.shape


def test_canonical_block_mlp_rejects_missing_frame_context() -> None:
    mlp = build_block_mlp(**_block_mlp_kwargs("canonical_frame"))
    mu = torch.zeros(1, 2, 4)
    sigma = torch.eye(4).expand(1, 2, 4, 4).clone()

    with pytest.raises(ValueError, match="requires a realized canonical frame"):
        mlp.forward_moments(mu, sigma)


@pytest.mark.parametrize("mode", ("coordinate", "gauge_gate", "canonical_frame"))
def test_invalid_block_mlp_covariance_fails_at_common_construction_boundary(mode: str) -> None:
    with pytest.raises(ValueError, match="unknown BlockMLP covariance contract 'invalid'"):
        build_block_mlp(**_block_mlp_kwargs(mode, covariance="invalid"))


@pytest.mark.parametrize(
    ("forward", "inverse", "message"),
    [
        (torch.ones(2, 3), torch.ones(2, 3), "square trailing matrix axes"),
        (torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]), torch.eye(2), "finite"),
        (torch.eye(2), 2.0 * torch.eye(2), "mutual inverses"),
    ],
)
def test_canonical_frame_context_rejects_invalid_factors(
    forward: torch.Tensor,
    inverse: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CanonicalFrameContext(forward, inverse)


def _decode(*_args):
    raise AssertionError("not called")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"supports_full": True, "covariance_kinds": frozenset({"diagonal"})},
        {"supports_chunked": True, "fused_ce": None},
        {"supports_chunked": False, "fused_ce": _decode},
        {"fused_ce_supports_stats": True, "fused_ce": None},
        {"family_consistent": True, "covariance_kinds": frozenset()},
    ],
)
def test_decode_registration_rejects_contradictory_capabilities(kwargs: dict) -> None:
    values = {
        "callable": _decode,
        "supports_full": False,
        "supports_chunked": False,
        "fused_ce": None,
        **kwargs,
    }
    with pytest.raises((TypeError, ValueError), match="contradictory|nonempty|fused|chunk"):
        DecodeRegistration(**values)


def test_model_executable_build_report_is_immutable_after_config_mutation() -> None:
    cfg = VFE3Config(
        vocab_size=11,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        use_block_mlp=True,
        block_mlp_mode="coordinate",
    )
    model = VFEModel(cfg)
    cfg.use_block_mlp = False
    cfg.block_mlp_mode = "gauge_gate"

    report = _pure_path_report(cfg, [], executable_build=model.executable_build)
    assert report["config_toggles"]["use_block_mlp"] is True
    assert report["config_toggles"]["block_mlp_mode"] == "coordinate"
    assert (
        report["config_toggles"]["block_mlp_structural_mode"]
        == "coordinate_mean_only_nonintertwiner"
    )


def test_scaling_signature_distinguishes_equal_parameter_block_mlp_structures() -> None:
    common = {field: None for field in scaling_analysis._SCALING_STRUCTURAL_FIELDS}
    coordinate = {
        **common,
        "use_block_mlp": True,
        "block_mlp_mode": "coordinate",
        "block_mlp_covariance": "passthrough",
        "block_mlp_expansion": 2,
        "block_mlp_activation": "gelu",
        "block_mlp_dropout": 0.0,
        "block_mlp_covariance_floor": 1e-4,
    }
    canonical = {
        **coordinate,
        "block_mlp_mode": "canonical_frame",
        "block_mlp_covariance": "delta_full",
    }
    assert coordinate["n_params"] == canonical["n_params"]
    assert scaling_analysis._structural_signature(coordinate) != (
        scaling_analysis._structural_signature(canonical)
    )


def test_reachable_ablation_tables_use_combined_gauge_label(capsys) -> None:
    row = {
        "label": "best",
        "primary_val_ppl": "2.0",
        "n_params": "10",
        "head_mixer_compatibility": "disabled",
        "block_mlp_structural_mode": "invariant_scalar_gate",
    }
    ablation._print_ablation_rows([row], parameter_matched=False)
    assert ablation._sensitivity_gauge_label(row) in capsys.readouterr().out


def test_inactive_scaling_decode_temperature_has_no_effective_value() -> None:
    cfg = VFE3Config(use_prior_bank=False, decode_tau=1.0)
    report = scaling._scaling_knob_report(cfg, "decode_tau")
    assert report == {
        "name": "decode_tau",
        "active": False,
        "configured_value": 1.0,
        "effective_value": None,
    }
