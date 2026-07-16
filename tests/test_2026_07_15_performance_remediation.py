import gc
import importlib
import math
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vfe3 import metrics
from vfe3.config import VFE3Config
from vfe3.ema import EMA
from vfe3.gauge_optim import GaugeNaturalGradAdamW, project_phi_parameter_rows_
from vfe3.geometry.groups import get_group
from vfe3.geometry.lie_ops import extract_phi, gram_pinv, retract_omega
from vfe3.geometry.transport import CompactFactoredTransport
from vfe3.model.model import VFEModel
from vfe3.train import evaluate
from vfe3.viz import extract
from vfe3.viz import report


e_step_module = importlib.import_module("vfe3.inference.e_step")
gauge_optim_module = importlib.import_module("vfe3.gauge_optim")
lie_ops_module = importlib.import_module("vfe3.geometry.lie_ops")
train_module = importlib.import_module("vfe3.train")


def _tiny_config(**overrides: object) -> VFE3Config:
    values = {
        "vocab_size": 9,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 4,
        "n_layers": 1,
        "n_e_steps": 1,
        "e_phi_lr": 0.0,
        "gauge_group": "block_glk",
        "pos_phi": "none",
        "use_head_mixer": False,
    }
    values.update(overrides)
    return VFE3Config(**values)


def _assert_scalar_dict_close(
    actual:   dict,
    expected: dict,

    *,
    atol:     float = 2e-5,
    rtol:     float = 1e-5,
) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float) and math.isnan(expected_value):
            assert math.isnan(actual_value), key
        else:
            assert actual_value == pytest.approx(expected_value, abs=atol, rel=rtol), key


def _assert_mapping_close(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, torch.Tensor):
            torch.testing.assert_close(actual_value, expected_value, rtol=2e-5, atol=2e-5)
        elif isinstance(expected_value, float):
            assert actual_value == pytest.approx(expected_value, rel=2e-5, abs=2e-5)
        else:
            assert actual_value == expected_value


def _record_tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if hasattr(value, "blocks") and isinstance(value.blocks, torch.Tensor):
        return [value.blocks]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _record_tensors(item)]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _record_tensors(item)]
    return []


def test_compact_diagnostics_keep_pairwise_transport_factored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(17)
    dense_model = VFEModel(_tiny_config(compact_phi_block_transport=False)).eval()
    compact_model = VFEModel(_tiny_config(compact_phi_block_transport=True)).eval()
    compact_model.load_state_dict(dense_model.state_dict())
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    dense_snapshot = dense_model.build_diagnostic_snapshot(token_ids)
    expected = dense_model.diagnostics(token_ids, snapshot=dense_snapshot)

    dense_pairwise_builds = 0
    original_transport = e_step_module._transport

    def _transport_spy(*args: object, **kwargs: object) -> object:
        nonlocal dense_pairwise_builds
        dense_pairwise_builds += 1
        return original_transport(*args, **kwargs)

    monkeypatch.setattr(e_step_module, "_transport", _transport_spy)
    compact_snapshot = compact_model.build_diagnostic_snapshot(token_ids)
    actual = compact_model.diagnostics(token_ids, snapshot=compact_snapshot)

    assert dense_pairwise_builds == 0
    torch.testing.assert_close(
        compact_snapshot.beta_maps,
        dense_snapshot.beta_maps,
        rtol=1e-5,
        atol=2e-5,
    )
    for key in (
        "attn_entropy",
        "self_coupling",
        "belief_coupling",
        "attention_entropy",
        "total",
        "holonomy_deviation",
        "holonomy_wilson",
        "cocycle_residual",
        "transport_asymmetry",
    ):
        assert actual[key] == pytest.approx(expected[key], rel=2e-5, abs=2e-5), key


def test_compact_trace_fallback_keeps_free_energy_transport_factored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(23)
    dense_model = VFEModel(_tiny_config(compact_phi_block_transport=False)).eval()
    compact_model = VFEModel(_tiny_config(compact_phi_block_transport=True)).eval()
    compact_model.load_state_dict(dense_model.state_dict())
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    expected = extract.e_step_belief_trace(dense_model, token_ids, n_iter=2)

    dense_pairwise_builds = 0
    original_transport = e_step_module._transport

    def _transport_spy(*args: object, **kwargs: object) -> object:
        nonlocal dense_pairwise_builds
        dense_pairwise_builds += 1
        return original_transport(*args, **kwargs)

    monkeypatch.setattr(e_step_module, "_transport", _transport_spy)
    actual = extract.e_step_belief_trace(compact_model, token_ids, n_iter=2)

    assert dense_pairwise_builds == 0
    _assert_mapping_close(actual, expected)


def test_compact_report_fallbacks_keep_transport_factored_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(29)
    dense_model = VFEModel(_tiny_config(compact_phi_block_transport=False)).eval()
    compact_model = VFEModel(_tiny_config(compact_phi_block_transport=True)).eval()
    compact_model.load_state_dict(dense_model.state_dict())
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    expected_maps = dense_model.attention_maps(token_ids)
    expected_layers = dense_model.diagnostics_per_layer(token_ids)
    expected_health = extract.numerical_health(dense_model, token_ids)
    expected_state = extract.converged_state(dense_model, token_ids)
    compact_reference_state = extract.converged_state(compact_model, token_ids)
    assert isinstance(compact_reference_state["omega"], CompactFactoredTransport)
    expected_equivariance = metrics.gauge_equivariance_residual(
        compact_reference_state["mu"],
        compact_reference_state["sigma"],
        compact_reference_state["omega"].to_dense_omega(),
        compact_model.group,
        kappa=compact_model.cfg.kappa_beta,
        diagonal=compact_model.cfg.diagonal_covariance,
        n_samples=2,
        seed=7,
    )

    dense_pairwise_builds = 0
    dense_materializations = 0
    original_transport = e_step_module._transport

    def _transport_spy(*args: object, **kwargs: object) -> object:
        nonlocal dense_pairwise_builds
        dense_pairwise_builds += 1
        return original_transport(*args, **kwargs)

    def _dense_forbidden(self: CompactFactoredTransport) -> torch.Tensor:
        del self
        nonlocal dense_materializations
        dense_materializations += 1
        raise AssertionError("compact report fallback materialized pairwise transport")

    monkeypatch.setattr(e_step_module, "_transport", _transport_spy)
    monkeypatch.setattr(extract, "_transport", _transport_spy)
    monkeypatch.setattr(CompactFactoredTransport, "to_dense_omega", _dense_forbidden)

    actual_maps = compact_model.attention_maps(token_ids)
    actual_layers = compact_model.diagnostics_per_layer(token_ids)
    actual_health = extract.numerical_health(compact_model, token_ids)
    actual_state = extract.converged_state(compact_model, token_ids)
    compact_snapshot = compact_model.build_diagnostic_snapshot(token_ids)
    snapshot_state = extract.converged_state(
        compact_model,
        token_ids,
        snapshot=compact_snapshot,
    )
    actual_curvature = metrics.curvature_field(actual_state["omega"])
    actual_equivariance = metrics.gauge_equivariance_residual(
        actual_state["mu"],
        actual_state["sigma"],
        actual_state["omega"],
        compact_model.group,
        kappa=compact_model.cfg.kappa_beta,
        diagonal=compact_model.cfg.diagonal_covariance,
        n_samples=2,
        seed=7,
    )

    assert dense_pairwise_builds == 0
    assert dense_materializations == 0
    assert isinstance(actual_state["omega"], CompactFactoredTransport)
    assert isinstance(snapshot_state["omega"], CompactFactoredTransport)
    torch.testing.assert_close(actual_state["mu"], compact_reference_state["mu"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_state["sigma"], compact_reference_state["sigma"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_maps, expected_maps, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        actual_curvature,
        metrics.curvature_field(expected_state["omega"]),
        rtol=2e-5,
        atol=2e-5,
    )
    for key, expected_value in expected_equivariance.items():
        torch.testing.assert_close(
            actual_equivariance[key],
            expected_value,
            rtol=2e-4,
            atol=(1e-3 if key.startswith("energy_") else 2e-5),
            msg=lambda message, label=key: f"{label}\n{message}",
        )
    for key, expected_values in expected_layers.items():
        assert actual_layers[key] == pytest.approx(expected_values, rel=2e-5, abs=2e-5), key
    _assert_scalar_dict_close(actual_health, expected_health)
    for key in ("mu", "sigma", "phi", "exp_phi", "energy", "beta", "self_div"):
        torch.testing.assert_close(actual_state[key], expected_state[key], rtol=2e-5, atol=2e-5)


def test_direct_omega_reuses_basis_factorization_and_defers_determinant_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("glk")(K=2, dtype=torch.float64)
    initial = torch.tensor(
        [
            [[1.1, 0.1], [0.0, 0.9]],
            [[0.8, 0.0], [0.2, 1.2]],
        ],
        dtype=torch.float64,
    )
    gradients = [
        torch.tensor(
            [
                [[0.2, -0.1], [0.05, 0.1]],
                [[-0.1, 0.2], [0.15, -0.05]],
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                [[-0.05, 0.1], [0.2, -0.1]],
                [[0.1, -0.15], [0.05, 0.2]],
            ],
            dtype=torch.float64,
        ),
    ]
    expected = initial.clone()
    basis_inverse = gram_pinv(group.generators)
    for gradient in gradients:
        tangent = extract_phi(
            torch.einsum("...lk,...lm->...km", expected, gradient),
            group.generators,
            gram_pinv_=basis_inverse,
        )
        expected = retract_omega(
            expected,
            -0.03 * tangent,
            group.generators,
            mode="lie_exp",
        )

    parameter = nn.Parameter(initial.clone())
    optimizer = GaugeNaturalGradAdamW(
        [{"params": [parameter], "lr": 0.03, "omega": True, "weight_decay": 0.0}],
        group.generators,
        group.irrep_dims,
        group_name=group.name,
        weight_decay=0.0,
    )
    gram_calls = 0
    slogdet_calls = 0
    real_gram_pinv = lie_ops_module.gram_pinv
    real_slogdet = torch.linalg.slogdet

    def _gram_spy(basis: torch.Tensor) -> torch.Tensor:
        nonlocal gram_calls
        gram_calls += 1
        return real_gram_pinv(basis)

    def _slogdet_spy(value: torch.Tensor) -> object:
        nonlocal slogdet_calls
        slogdet_calls += 1
        return real_slogdet(value)

    monkeypatch.setattr(lie_ops_module, "gram_pinv", _gram_spy)
    monkeypatch.setattr(torch.linalg, "slogdet", _slogdet_spy)
    for gradient in gradients:
        parameter.grad = gradient.clone()
        optimizer.step()

    torch.testing.assert_close(parameter, expected, rtol=0.0, atol=1e-12)
    assert gram_calls == 1
    assert slogdet_calls == 0

    optimizer._collect_gauge_diag = True
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert gram_calls == 1
    assert slogdet_calls == 1


def test_direct_omega_validation_has_one_host_decision_and_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("glk")(K=2)
    parameter = nn.Parameter(torch.eye(2).expand(2, 2, 2).clone())
    optimizer = GaugeNaturalGradAdamW(
        [{"params": [parameter], "lr": 0.01, "omega": True, "weight_decay": 0.0}],
        group.generators,
        group.irrep_dims,
        group_name=group.name,
        weight_decay=0.0,
    )
    bool_calls = 0
    real_bool = torch.Tensor.__bool__

    def _bool_spy(value: torch.Tensor) -> bool:
        nonlocal bool_calls
        if value.dtype == torch.bool and value.numel() == 1:
            bool_calls += 1
        return real_bool(value)

    monkeypatch.setattr(torch.Tensor, "__bool__", _bool_spy)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert bool_calls == 1

    with torch.no_grad():
        parameter[0, 1].copy_(parameter[0, 0])
    before = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(FloatingPointError, match="nonfinite or singular"):
        optimizer.step()
    torch.testing.assert_close(parameter, before, rtol=0.0, atol=0.0)


def _four_phi_table_model() -> VFEModel:
    return VFEModel(_tiny_config(
        n_heads=1,
        prior_source="model_channel",
        s_frame_mode="phi_tilde",
        s_e_step=True,
        lambda_h=1.0,
        lambda_gamma=1.0,
        pos_phi="learned",
    ))


def test_phi_projection_uses_dynamic_chunks_and_one_aggregate_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _four_phi_table_model()
    tables = [
        model.prior_bank.phi_embed,
        model.prior_bank.s_phi_embed,
        model.pos_phi_free,
        model.s_pos_phi_free,
    ]
    generator = torch.Generator().manual_seed(19)
    expected_tables = []
    projected_rows = 0
    total_rows = 0
    norm_max = 0.0
    scale_min = 1.0
    with torch.no_grad():
        for table in tables:
            table.copy_(torch.randn(table.shape, generator=generator) * 3.0)
            expected = table.detach().clone()
            embedded = torch.einsum("...a,aij->...ij", expected, model.group.generators)
            norm = torch.linalg.matrix_norm(embedded, ord="fro", dim=(-2, -1))
            scale = (2.0 / norm.clamp(min=1e-12)).clamp(max=1.0)
            expected.mul_(scale.unsqueeze(-1))
            expected_tables.append(expected)
            projected_rows += int((scale < 1.0).sum().item())
            total_rows += table.shape[0]
            norm_max = max(norm_max, norm.max().item())
            scale_min = min(scale_min, scale.min().item())

    norm_calls = 0
    cpu_calls = 0
    tensor_float_calls = 0
    real_norm = gauge_optim_module.embedded_phi_frobenius_norm
    real_cpu = torch.Tensor.cpu
    real_float = torch.Tensor.__float__

    def _norm_spy(phi: torch.Tensor, group: object, **kwargs: object) -> torch.Tensor:
        nonlocal norm_calls
        norm_calls += 1
        return real_norm(phi, group, **kwargs)

    def _cpu_spy(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal cpu_calls
        cpu_calls += 1
        return real_cpu(value, *args, **kwargs)

    def _float_spy(value: torch.Tensor) -> float:
        nonlocal tensor_float_calls
        tensor_float_calls += 1
        return real_float(value)

    monkeypatch.setattr(gauge_optim_module, "embedded_phi_frobenius_norm", _norm_spy)
    monkeypatch.setattr(torch.Tensor, "cpu", _cpu_spy)
    monkeypatch.setattr(torch.Tensor, "__float__", _float_spy)
    stats = project_phi_parameter_rows_(
        model,
        2.0,
        temporary_bytes=1 << 30,
        collect_stats=True,
    )

    assert norm_calls == len(tables)
    assert cpu_calls == 1
    assert tensor_float_calls == 0
    for table, expected in zip(tables, expected_tables):
        torch.testing.assert_close(table, expected)
    assert stats == pytest.approx({
        "phi_chart_projected_rows": projected_rows,
        "phi_chart_total_rows": total_rows,
        "phi_chart_projected_fraction": projected_rows / total_rows,
        "phi_chart_preproject_max": norm_max,
        "phi_chart_projection_scale_min": scale_min,
    })


class _EvaluationModel(nn.Module):
    def __init__(self, cross_entropies: list[float]) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.prior_bank = SimpleNamespace(mu_embed=self.anchor)
        self._cross_entropies = iter(cross_entropies)

    def forward(
        self,
        tokens:  torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[None, None, torch.Tensor]:
        del tokens, targets
        return None, None, self.anchor.new_tensor(next(self._cross_entropies))


def test_evaluate_transfers_one_aggregate_and_preserves_token_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _EvaluationModel([0.5, 1.25, 2.0])
    loader = [
        (torch.zeros(1, 4, dtype=torch.long), torch.tensor([[0, 1, 2, 3]])),
        (torch.zeros(1, 4, dtype=torch.long), torch.tensor([[0, 1, 2, -100]])),
        (torch.zeros(1, 4, dtype=torch.long), torch.tensor([[0, 1, -100, -100]])),
    ]
    float_calls = 0
    int_calls = 0
    cpu_calls = 0
    real_float = torch.Tensor.__float__
    real_int = torch.Tensor.__int__
    real_cpu = torch.Tensor.cpu

    def _float_spy(value: torch.Tensor) -> float:
        nonlocal float_calls
        float_calls += 1
        return real_float(value)

    def _int_spy(value: torch.Tensor) -> int:
        nonlocal int_calls
        int_calls += 1
        return real_int(value)

    def _cpu_spy(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal cpu_calls
        cpu_calls += 1
        return real_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "__float__", _float_spy)
    monkeypatch.setattr(torch.Tensor, "__int__", _int_spy)
    monkeypatch.setattr(torch.Tensor, "cpu", _cpu_spy)
    result = evaluate(model, loader, tokens_per_char=0.4)

    expected_ce = (0.5 * 4 + 1.25 * 3 + 2.0 * 2) / 9
    assert result["ce"] == pytest.approx(expected_ce)
    assert result["ppl"] == pytest.approx(math.exp(expected_ce))
    assert result["bits_per_token"] == pytest.approx(expected_ce / math.log(2.0))
    assert result["bpc"] == pytest.approx(expected_ce / math.log(2.0) * 0.4)
    assert float_calls == 0
    assert int_calls == 0
    assert cpu_calls == 1


def test_fixed_point_diagnostics_reuse_snapshot_and_run_only_one_new_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VFEModel(_tiny_config(n_e_steps=2, e_step_update="gradient")).eval()
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    expected = extract.e_step_fixed_point_diagnostics(model, token_ids)
    snapshot = model.build_diagnostic_snapshot(token_ids)
    iteration_calls = 0
    real_iteration = extract.e_step_iteration

    def _iteration_spy(*args: object, **kwargs: object) -> object:
        nonlocal iteration_calls
        iteration_calls += 1
        return real_iteration(*args, **kwargs)

    monkeypatch.setattr(extract, "e_step_iteration", _iteration_spy)
    actual = extract.e_step_fixed_point_diagnostics(
        model,
        token_ids,
        snapshot=snapshot,
    )

    assert iteration_calls == 1
    _assert_scalar_dict_close(actual, expected)


def test_fixed_point_snapshot_reuses_realized_early_halt_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VFEModel(_tiny_config(
        n_e_steps=4,
        e_step_update="gradient",
        e_step_halt_tol=1e9,
    )).eval()
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    snapshot = model.build_diagnostic_snapshot(token_ids)
    assert 1 < len(snapshot.trace_states) < model.cfg.n_e_steps + 1
    iteration_calls = 0
    real_iteration = extract.e_step_iteration

    def _iteration_spy(*args: object, **kwargs: object) -> object:
        nonlocal iteration_calls
        iteration_calls += 1
        return real_iteration(*args, **kwargs)

    monkeypatch.setattr(extract, "e_step_iteration", _iteration_spy)
    actual = extract.e_step_fixed_point_diagnostics(
        model,
        token_ids,
        snapshot=snapshot,
    )

    assert iteration_calls == 1
    assert math.isnan(actual["estep_target_gap"])
    assert all(math.isfinite(value)
               for key, value in actual.items()
               if key != "estep_target_gap")


def test_shared_report_inference_bank_serves_all_population_consumers_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VFEModel(_tiny_config(
        n_heads=1,
        prior_source="model_channel",
        s_frame_mode="phi_tilde",
        s_e_step=True,
        lambda_h=1.0,
        lambda_gamma=0.0,
    )).eval()
    loader = [
        (
            torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        ),
        (
            torch.tensor([[4, 3, 2, 1]], dtype=torch.long),
            torch.tensor([[3, 2, 1, 0]], dtype=torch.long),
        ),
    ]
    token_batches = [tokens for tokens, _ in loader]
    expected_belief = extract.belief_bank(model, token_batches)
    expected_ce = extract.belief_ce_bank(model, loader)
    expected_model = extract.model_channel_bank(model, token_batches)
    expected_vocab = extract.vocab_prediction_stats(model, token_batches)

    inference_calls = 0
    cpu_transfer_ids: set[int] = set()
    real_forward_beliefs = model.forward_beliefs
    real_cpu = torch.Tensor.cpu

    def _forward_spy(*args: object, **kwargs: object) -> object:
        nonlocal inference_calls
        inference_calls += 1
        return real_forward_beliefs(*args, **kwargs)

    def _cpu_spy(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        cpu_transfer_ids.add(id(value))
        return real_cpu(value, *args, **kwargs)

    monkeypatch.setattr(model, "forward_beliefs", _forward_spy)
    monkeypatch.setattr(torch.Tensor, "cpu", _cpu_spy)
    inference_bank = extract.collect_inference_bank(
        model,
        loader,
        max_batches=len(loader),
        return_logits=True,
    )
    retained_tensors = [
        tensor
        for record in inference_bank
        for tensor in _record_tensors(record)
    ]
    assert len(inference_bank) == len(loader)
    assert retained_tensors
    assert all(tensor.device.type == "cpu" for tensor in retained_tensors)
    assert all(id(tensor) in cpu_transfer_ids for tensor in retained_tensors)
    expected_bytes = 0
    seen_storages: set[tuple[str, int]] = set()
    for tensor in retained_tensors:
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), storage.data_ptr())
        if storage_key not in seen_storages:
            expected_bytes += storage.nbytes()
            seen_storages.add(storage_key)
    assert extract.inference_bank_nbytes(inference_bank) == expected_bytes
    actual_belief = extract.belief_bank(
        model,
        token_batches,
        inference_bank=inference_bank,
    )
    actual_ce = extract.belief_ce_bank(
        model,
        loader,
        inference_bank=inference_bank,
    )
    actual_model = extract.model_channel_bank(
        model,
        token_batches,
        inference_bank=inference_bank,
    )
    actual_vocab = extract.vocab_prediction_stats(
        model,
        token_batches,
        inference_bank=inference_bank,
    )

    assert inference_calls == len(loader)
    _assert_mapping_close(actual_belief, expected_belief)
    _assert_mapping_close(actual_ce, expected_ce)
    assert actual_model is not None and expected_model is not None
    _assert_mapping_close(actual_model, expected_model)
    _assert_mapping_close(actual_vocab, expected_vocab)


def test_inference_bank_releases_prior_device_batch_before_next_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VFEModel(_tiny_config(
        n_heads=1,
        prior_source="model_channel",
        s_frame_mode="phi_tilde",
        s_e_step=True,
        lambda_h=1.0,
        lambda_gamma=0.0,
    )).eval()
    loader = [
        (torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])),
        (torch.tensor([[4, 3, 2, 1]]), torch.tensor([[3, 2, 1, 0]])),
    ]
    previous_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    liveness_checks: list[bool] = []
    calls = 0
    real_forward = model.forward_beliefs

    def _forward_spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        if previous_refs:
            gc.collect()
            liveness_checks.append(all(ref() is None for ref in previous_refs))
        result = real_forward(*args, **kwargs)
        if calls == 0:
            capture = kwargs["capture"]
            tensors = (
                result[0].mu,
                result[1],
                capture["out"].sigma,
                capture["prior"].mu,
                capture["prior"].phi,
            )
            previous_refs.extend(weakref.ref(tensor) for tensor in tensors)
        calls += 1
        return result

    monkeypatch.setattr(model, "forward_beliefs", _forward_spy)
    extract.collect_inference_bank(model, loader, return_logits=True)

    assert calls == 2
    assert liveness_checks == [True]


def test_vocab_consumer_releases_previous_logits_and_probabilities_before_softmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VFEModel(_tiny_config()).eval()
    token_batches = [
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[4, 3, 2, 1]]),
    ]
    inference_bank = extract.collect_inference_bank(
        model,
        token_batches,
        return_logits=True,
    )
    previous_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    liveness_checks: list[bool] = []
    real_softmax = torch.softmax

    def _softmax_spy(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        if previous_refs:
            gc.collect()
            liveness_checks.append(all(ref() is None for ref in previous_refs))
        result = real_softmax(value, *args, **kwargs)
        previous_refs[:] = [weakref.ref(value), weakref.ref(result)]
        return result

    monkeypatch.setattr(torch, "softmax", _softmax_spy)
    extract.vocab_prediction_stats(
        model,
        token_batches,
        inference_bank=inference_bank,
    )

    assert liveness_checks == [True]


def test_report_token_fallback_keeps_all_collected_batches_off_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_batches = [
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[4, 3, 2, 1]]),
    ]
    device_moves: list[int] = []

    def _to_spy(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        device_moves.append(id(value))
        return value

    monkeypatch.setattr(torch.Tensor, "to", _to_spy)
    actual = report._collect_token_batches(
        token_batches,
        torch.device("cuda"),
        len(token_batches),
    )

    assert device_moves == []
    assert all(torch.equal(actual_batch, source_batch)
               for actual_batch, source_batch in zip(actual, token_batches))
    assert all(tokens.device.type == "cpu" for tokens in actual)


def test_full_vocab_memory_guard_scales_with_all_retained_batches() -> None:
    cfg = _tiny_config(vocab_size=101, max_seq_len=7, batch_size=3)
    one_batch = report._estimated_full_vocab_bank_bytes(cfg, 1)
    four_batches = report._estimated_full_vocab_bank_bytes(cfg, 4)

    assert one_batch == 8 * cfg.vocab_size * cfg.max_seq_len * cfg.batch_size
    assert four_batches == 4 * one_batch


def test_eval_only_step_runs_sparse_omega_determinant_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VFEModel(_tiny_config(
        gauge_parameterization="omega_direct",
        omega_compact_storage=True,
        m_phi_lr=0.01,
        use_ema=False,
    ))
    batch = (
        torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
    )
    determinant_calls = 0
    original = gauge_optim_module._omega_determinant_failure

    def _determinant_spy(value: torch.Tensor) -> torch.Tensor:
        nonlocal determinant_calls
        determinant_calls += 1
        return original(value)

    monkeypatch.setattr(gauge_optim_module, "_omega_determinant_failure", _determinant_spy)
    train_module.train(
        model,
        [batch],
        model.cfg,
        n_steps=1,
        log_interval=None,
        eval_interval=1,
        val_loader=[batch],
        artifacts=None,
        generate_samples=False,
    )

    assert determinant_calls > 0


def test_direct_omega_retraction_is_atomic_across_parameter_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("glk")(K=2, dtype=torch.float64)
    first = nn.Parameter(torch.eye(2, dtype=torch.float64).unsqueeze(0))
    second = nn.Parameter((1.1 * torch.eye(2, dtype=torch.float64)).unsqueeze(0))
    optimizer = GaugeNaturalGradAdamW(
        [
            {"params": [first], "lr": 0.1, "omega": True, "weight_decay": 0.0},
            {"params": [second], "lr": 0.1, "omega": True, "weight_decay": 0.0},
        ],
        group.generators,
        group.irrep_dims,
        gauge_momentum=0.0,
        weight_decay=0.0,
    )
    first.grad = torch.full_like(first, 0.1)
    second.grad = torch.full_like(second, 0.2)
    first_before = first.detach().clone()
    second_before = second.detach().clone()
    first_grad_before = first.grad.clone()
    second_grad_before = second.grad.clone()
    optimizer_state_before = optimizer.state_dict()
    omega_step_before = optimizer._omega_step
    calls = 0

    def _late_invalid_retraction(
        value:      torch.Tensor,
        tangent:    torch.Tensor,
        generators: torch.Tensor,

        *,
        mode:       str,
    ) -> torch.Tensor:
        del tangent, generators, mode
        nonlocal calls
        calls += 1
        if calls == 2:
            return torch.zeros_like(value)
        return value + 0.01 * torch.eye(2, dtype=value.dtype, device=value.device)

    monkeypatch.setattr(lie_ops_module, "retract_omega", _late_invalid_retraction)

    with pytest.raises(FloatingPointError, match="nonfinite or singular"):
        optimizer.step()

    assert calls == 2
    assert torch.equal(first, first_before)
    assert torch.equal(second, second_before)
    assert torch.equal(first.grad, first_grad_before)
    assert torch.equal(second.grad, second_grad_before)
    assert not optimizer.state
    assert optimizer.state_dict() == optimizer_state_before
    assert optimizer._omega_step == omega_step_before


def test_direct_omega_reorth_cadence_is_one_clock_for_all_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("so_k")(K=2, dtype=torch.float64)
    first = nn.Parameter(torch.eye(2, dtype=torch.float64).repeat(2, 1, 1))
    second = nn.Parameter(torch.eye(2, dtype=torch.float64).repeat(2, 1, 1))
    optimizer = GaugeNaturalGradAdamW(
        [
            {"params": [first], "lr": 0.05, "omega": True, "weight_decay": 0.0},
            {"params": [second], "lr": 0.05, "omega": True, "weight_decay": 0.0},
        ],
        group.generators,
        group.irrep_dims,
        gauge_momentum=0.0,
        skew_symmetric=True,
        omega_reorth_every=2,
        weight_decay=0.0,
    )
    polar_calls: list[torch.Size] = []
    original_polar = gauge_optim_module._polar_orthogonalize

    def _polar_spy(value: torch.Tensor) -> torch.Tensor:
        polar_calls.append(value.shape)
        return original_polar(value)

    monkeypatch.setattr(gauge_optim_module, "_polar_orthogonalize", _polar_spy)
    for parameter in (first, second):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert optimizer._omega_step == 1
    assert polar_calls == []
    assert all(bool(optimizer.state[p]["omega_dirty"].all()) for p in (first, second))

    for parameter in (first, second):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert optimizer._omega_step == 2
    assert polar_calls == [torch.Size([2, 2, 2]), torch.Size([2, 2, 2])]
    assert all(not bool(optimizer.state[p]["omega_dirty"].any()) for p in (first, second))


def test_gamma_compatibility_surfaces_delegate_to_production_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VFEModel(_tiny_config(
        n_heads=1,
        prior_source="model_channel",
        lambda_gamma=1.0,
        include_attention_entropy=True,
    ))
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    phi = model.prior_bank.encode(token_ids).phi
    calls: list[str] = []

    def _rows(
        tokens:         torch.Tensor,
        model_phi:      torch.Tensor,

        *,
        head_reduction: str,
        **kwargs:       object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model_phi, kwargs
        calls.append(head_reduction)
        shape = tokens.shape
        coupling = torch.arange(1, shape[1] + 1, dtype=torch.float32).reshape(shape)
        meta = torch.full(shape, 0.5)
        return coupling, meta

    monkeypatch.setattr(model, "_gamma_coupling_rows", _rows)
    term = model._gamma_coupling_term(token_ids, phi)
    terms = model._gamma_coupling_terms(token_ids, phi)

    expected_coupling = torch.tensor(10.0)
    expected_meta = torch.tensor(2.0)
    assert calls == ["mean", "sum"]
    torch.testing.assert_close(term, (expected_coupling + expected_meta) / token_ids.numel())
    torch.testing.assert_close(terms["coupling"], expected_coupling)
    torch.testing.assert_close(terms["meta_entropy"], expected_meta)
    torch.testing.assert_close(terms["total"], expected_coupling + expected_meta)


def test_ema_transfers_one_finiteness_vector_and_preserves_per_parameter_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Module()
    model.left = nn.Parameter(torch.tensor([1.0, 1.0]))
    model.right = nn.Parameter(torch.tensor([2.0, 2.0]))
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.left.fill_(3.0)
        model.right[0] = float("nan")

    bool_calls = 0
    cpu_calls = 0
    real_bool = torch.Tensor.__bool__
    real_cpu = torch.Tensor.cpu

    def _bool_spy(value: torch.Tensor) -> bool:
        nonlocal bool_calls
        if value.dtype == torch.bool and value.numel() == 1:
            bool_calls += 1
        return real_bool(value)

    def _cpu_spy(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal cpu_calls
        cpu_calls += 1
        return real_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "__bool__", _bool_spy)
    monkeypatch.setattr(torch.Tensor, "cpu", _cpu_spy)
    ema.update(model)

    assert bool_calls == 0
    assert cpu_calls == 1
    torch.testing.assert_close(ema.shadow["left"], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(ema.shadow["right"], torch.tensor([2.0, 2.0]))
