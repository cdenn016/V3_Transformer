import torch

import scaling
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import _cost_model_fields, _pure_path_report


def _pure_cfg(mode, covariance="passthrough"):
    return VFE3Config(
        vocab_size=17,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=2,
        n_e_steps=1,
        family="gaussian_full",
        decode_mode="full_chunked",
        use_prior_bank=True,
        use_block_mlp=True,
        block_mlp_mode=mode,
        block_mlp_covariance=covariance,
        gauge_group="block_glk",
        transport_mode="flat",
        pos_phi="none",
        pos_phi_compose="bch",
        pos_rotation="none",
        e_phi_lr=0.0,
    )


def test_pure_path_report_distinguishes_strict_gate_from_frame_fixed_mode():
    gate = _pure_path_report(_pure_cfg("gauge_gate", "delta_full"), [])
    canonical = _pure_path_report(_pure_cfg("canonical_frame"), [])

    assert gate["config_toggles"]["block_mlp_mode"] == "gauge_gate"
    assert gate["config_toggles"]["block_mlp_structural_mode"] == "invariant_scalar_gate"
    assert gate["config_toggles"]["block_mlp_covariance_contract"] == "delta_full_plus_covariant_floor"
    assert gate["config_toggles"]["block_mlp_covariance_floor"] == 1e-4
    assert gate["gauge_flags"]["block_mlp_intertwiner_compatible"] is True
    assert gate["on_gauge_pure_path"] is True
    assert gate["pure_flags"]["no_block_mlp"] is False
    assert gate["on_pure_path"] is False

    assert canonical["config_toggles"]["block_mlp_mode"] == "canonical_frame"
    assert (
        canonical["config_toggles"]["block_mlp_structural_mode"]
        == "canonical_frame_left_equivariant_right_fixed"
    )
    assert canonical["gauge_flags"]["block_mlp_intertwiner_compatible"] is False
    assert canonical["on_gauge_pure_path"] is False


def test_gauge_gate_parameter_accounting_uses_invariant_block_width():
    cfg = _pure_cfg("gauge_gate")
    model = VFEModel(cfg)
    n_blocks = len(model.group.irrep_dims)
    expansion = cfg.block_mlp_expansion
    per_layer = (
        2 * expansion * n_blocks * n_blocks
        + (expansion + 1) * n_blocks
    )
    predicted, _ = scaling.predict_n_params(cfg)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert predicted == actual

    cost = _cost_model_fields(model, cfg, n_params=actual, tokens_seen=13)
    assert cost["block_mlp_params"] == cfg.n_layers * per_layer


@torch.no_grad()
def test_selected_modes_execute_complete_model_forward():
    tokens = torch.tensor([[1, 2, 3, 4]])
    for mode, covariance in (
        ("coordinate", "delta_full"),
        ("gauge_gate", "passthrough"),
        ("gauge_gate", "delta_full"),
        ("canonical_frame", "passthrough"),
        ("canonical_frame", "delta_full"),
    ):
        model = VFEModel(_pure_cfg(mode, covariance))
        logits = model(tokens)
        assert logits.shape == (1, 4, 17)
        assert torch.isfinite(logits).all()


@torch.no_grad()
def test_canonical_frame_executes_with_right_full_gauge_rope():
    cfg = _pure_cfg("canonical_frame")
    cfg = VFE3Config(**{
        **cfg.__dict__,
        "pos_rotation": "rope",
        "rope_insertion": "right",
        "rope_full_gauge": True,
        "rope_on_value": True,
    })
    logits = VFEModel(cfg)(torch.tensor([[1, 2, 3, 4]]))
    assert torch.isfinite(logits).all()
