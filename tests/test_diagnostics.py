r"""One-forward diagnostic snapshots and replay-free evaluation consumers."""

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import _val_diagnostics
from vfe3.viz import extract


def _model(**overrides) -> VFEModel:
    base = dict(
        vocab_size=12,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=2,
        n_e_steps=2,
        pos_phi="none",
        pos_rotation="none",
    )
    base.update(overrides)
    torch.manual_seed(17)
    model = VFEModel(VFE3Config(**base))
    model.eval()
    return model


def _tokens() -> torch.Tensor:
    return torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])


def _assert_scalar_dict_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], list):
            assert actual[key] == pytest.approx(expected[key], rel=1e-5, abs=1e-6), key
        else:
            assert math.isclose(actual[key], expected[key], rel_tol=1e-5, abs_tol=1e-6), key


def test_eval_diagnostics_builds_one_snapshot(monkeypatch) -> None:
    model = _model(lambda_gamma=0.25)
    tokens = _tokens()
    targets = torch.roll(tokens, shifts=-1, dims=1)
    calls = 0
    real_forward_beliefs = model.forward_beliefs

    def counted_forward_beliefs(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_forward_beliefs(*args, **kwargs)

    monkeypatch.setattr(model, "forward_beliefs", counted_forward_beliefs)
    diagnostics = _val_diagnostics(model, [(tokens, targets)], torch.device("cpu"))

    assert calls == 1
    assert math.isfinite(diagnostics["val_free_energy_total"])
    assert math.isfinite(diagnostics["estep_f_drop"])
    assert math.isfinite(diagnostics["pos_loss_ratio"])


def test_snapshot_and_independent_diagnostics_are_value_equal() -> None:
    model = _model(lambda_gamma=0.25)
    tokens = _tokens()
    snapshot = model.build_diagnostic_snapshot(tokens)

    with pytest.raises(FrozenInstanceError):
        snapshot.logits = torch.empty_like(snapshot.logits)

    torch.manual_seed(29)
    expected = model.diagnostics(tokens)
    torch.manual_seed(29)
    actual = model.diagnostics(tokens, snapshot=snapshot)
    _assert_scalar_dict_equal(actual, expected)

    assert torch.allclose(
        model.attention_maps(tokens, snapshot=snapshot),
        model.attention_maps(tokens),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        model.gamma_attention_maps(tokens, snapshot=snapshot),
        model.gamma_attention_maps(tokens),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.manual_seed(31)
    per_layer_actual = model.diagnostics_per_layer(tokens, snapshot=snapshot)
    torch.manual_seed(31)
    per_layer_expected = model.diagnostics_per_layer(tokens)
    _assert_scalar_dict_equal(per_layer_actual, per_layer_expected)
    assert torch.allclose(snapshot.logits, model(tokens), atol=1e-6, rtol=1e-6)


def test_attention_and_trace_reuse_snapshot_without_forward_replay(monkeypatch) -> None:
    import vfe3.model.block as block_module
    import vfe3.model.model as model_module
    import vfe3.model.stack as stack_module

    model = _model(lambda_gamma=0.25)
    tokens = _tokens()
    snapshot = model.build_diagnostic_snapshot(tokens)

    def replay_forbidden(*args, **kwargs):
        raise AssertionError("snapshot consumer replayed model inference")

    monkeypatch.setattr(model, "forward_beliefs", replay_forbidden)
    monkeypatch.setattr(model.prior_bank, "encode", replay_forbidden)
    monkeypatch.setattr(model.prior_bank, "encode_s", replay_forbidden)
    monkeypatch.setattr(model_module, "vfe_stack", replay_forbidden)
    monkeypatch.setattr(model_module, "vfe_block", replay_forbidden)
    monkeypatch.setattr(stack_module, "vfe_stack", replay_forbidden)
    monkeypatch.setattr(block_module, "e_step", replay_forbidden)
    monkeypatch.setattr(extract, "_encode_one", replay_forbidden)
    monkeypatch.setattr(extract, "e_step_iteration", replay_forbidden)
    monkeypatch.setattr(extract, "vfe_block", replay_forbidden)

    assert model.diagnostics(tokens, snapshot=snapshot)
    assert model.attention_maps(tokens, snapshot=snapshot).shape == (2, 2, 4, 4)
    assert model.diagnostics_per_layer(tokens, snapshot=snapshot)
    assert model.gamma_attention_maps(tokens, snapshot=snapshot).shape == (2, 4, 4)
    assert extract.e_step_belief_trace(model, tokens, snapshot=snapshot)["mu"].shape[0] == 3
    assert extract.across_layer_belief_trace(model, tokens, snapshot=snapshot)["mu"].shape[0] == 2
    assert extract.numerical_health(model, tokens, snapshot=snapshot)
    assert extract.converged_state(model, tokens, snapshot=snapshot)
    assert extract.gamma_attention(model, tokens, snapshot=snapshot)["gamma"].shape == (2, 4, 4)
