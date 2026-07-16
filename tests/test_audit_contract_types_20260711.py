"""Regression pins for the July 11 type-contract and half-Fisher audit fixes."""

from importlib import import_module, util
from inspect import Parameter, signature
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, get_args, get_type_hints, is_typeddict

import torch

from vfe3.belief import BeliefState
from vfe3.inference.e_step import e_step, e_step_iteration
from vfe3.model.block import _as_coeff, vfe_block
from vfe3.model.model import VFEModel
from vfe3.model.stack import vfe_stack
from vfe3.run_artifacts import RunArtifacts, load_checkpoint
from vfe3.viz import figures


def _contracts():
    spec = util.find_spec("vfe3.contracts")
    assert spec is not None, "vfe3.contracts must define the shared audited TypedDict contracts"
    return import_module("vfe3.contracts")


def test_shared_contracts_pin_only_the_cited_mutable_dict_schemas() -> None:
    contracts = _contracts()
    expected = {
        contracts.MStepCapture: {
            "converged": BeliefState,
            "final_block_prior": Tuple[torch.Tensor, torch.Tensor],
            "final_block_tau": float | torch.Tensor,
            "prior": BeliefState,
            "out": BeliefState,
            "beta_prior_context": contracts.EffectiveBetaPriorContext,
            "cg_moment_energy_rows": List[torch.Tensor],
            "cg_pre_moments": List[Tuple[torch.Tensor, torch.Tensor]],
        },
        contracts.EStepGradientRecord: {
            "mu": torch.Tensor,
            "sigma": torch.Tensor,
            "phi": torch.Tensor,
        },
        contracts.EStepGradientOutput: {
            "mu": float,
            "sigma": float,
            "phi": float,
        },
        contracts.DataStateBuffer: {
            "epoch_start_generator_state": Optional[torch.Tensor],
            "batches_consumed": int,
            "epoch": int,
            "data_identity": Dict[str, object],
        },
        contracts.DataState: {
            "epoch_start_generator_state": Optional[torch.Tensor],
            "batches_consumed": int,
            "epoch": int,
            "data_identity": Dict[str, object],
        },
    }

    for contract, hints in expected.items():
        assert is_typeddict(contract)
        assert get_type_hints(contract) == hints

    assert contracts.MStepCapture.__required_keys__ == frozenset()
    assert contracts.EStepGradientRecord.__required_keys__ == frozenset()
    assert contracts.EStepGradientOutput.__required_keys__ == frozenset()
    assert contracts.DataStateBuffer.__required_keys__ == frozenset()
    assert contracts.DataState.__required_keys__ == frozenset(expected[contracts.DataState])


def test_prior_bank_registries_use_concrete_callable_aliases() -> None:
    prior_bank = import_module("vfe3.model.prior_bank")

    assert prior_bank.EncodeCallable == Callable[[prior_bank.PriorBank, torch.Tensor], BeliefState]
    assert prior_bank.DecodeCallable == Callable[
        [prior_bank.PriorBank, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]
    assert prior_bank.FusedCECallable != Callable[..., torch.Tensor]
    assert set(get_args(prior_bank.FusedCECallable)) == {
        prior_bank.GeometricFusedCECallable,
        prior_bank.LinearFusedCECallable,
    }

    geometric = signature(prior_bank.GeometricFusedCECallable.__call__)
    assert [
        (parameter.name, parameter.kind)
        for parameter in geometric.parameters.values()
    ] == [
        ("self", Parameter.POSITIONAL_OR_KEYWORD),
        ("pb", Parameter.POSITIONAL_OR_KEYWORD),
        ("mu_q", Parameter.POSITIONAL_OR_KEYWORD),
        ("sigma_q", Parameter.POSITIONAL_OR_KEYWORD),
        ("targets", Parameter.POSITIONAL_OR_KEYWORD),
        ("z_loss_weight", Parameter.KEYWORD_ONLY),
        ("tau", Parameter.KEYWORD_ONLY),
        ("chunk_size", Parameter.KEYWORD_ONLY),
        ("ignore_index", Parameter.KEYWORD_ONLY),
    ]
    geometric_hints = get_type_hints(prior_bank.GeometricFusedCECallable.__call__)
    assert geometric_hints == {
        "pb": prior_bank.PriorBank,
        "mu_q": torch.Tensor,
        "sigma_q": torch.Tensor,
        "targets": torch.Tensor,
        "z_loss_weight": float,
        "tau": Optional[float],
        "chunk_size": Optional[int],
        "ignore_index": int,
        "return": torch.Tensor,
    }
    assert geometric.parameters["z_loss_weight"].default == 0.0
    assert geometric.parameters["tau"].default is None
    assert geometric.parameters["chunk_size"].default is None
    assert geometric.parameters["ignore_index"].default == -100

    linear = signature(prior_bank.LinearFusedCECallable.__call__)
    assert [
        (parameter.name, parameter.kind)
        for parameter in linear.parameters.values()
    ] == [
        ("self", Parameter.POSITIONAL_OR_KEYWORD),
        ("pb", Parameter.POSITIONAL_OR_KEYWORD),
        ("mu_q", Parameter.POSITIONAL_OR_KEYWORD),
        ("targets", Parameter.POSITIONAL_OR_KEYWORD),
        ("z_loss_weight", Parameter.KEYWORD_ONLY),
        ("chunk_size", Parameter.KEYWORD_ONLY),
        ("ignore_index", Parameter.KEYWORD_ONLY),
    ]
    linear_hints = get_type_hints(prior_bank.LinearFusedCECallable.__call__)
    assert linear_hints == {
        "pb": prior_bank.PriorBank,
        "mu_q": torch.Tensor,
        "targets": torch.Tensor,
        "z_loss_weight": float,
        "chunk_size": Optional[int],
        "ignore_index": int,
        "return": torch.Tensor,
    }
    assert linear.parameters["z_loss_weight"].default == 0.0
    assert linear.parameters["chunk_size"].default is None
    assert linear.parameters["ignore_index"].default == -100

    module_hints = get_type_hints(prior_bank)
    assert module_hints["_ENCODERS"] == Dict[str, prior_bank.EncodeCallable]
    record_hints = get_type_hints(prior_bank.DecodeRegistration)
    assert record_hints["callable"] == prior_bank.DecodeCallable
    assert record_hints["fused_ce"] == Optional[prior_bank.FusedCECallable]

    encode_register_hints = get_type_hints(prior_bank.register_encode)
    assert encode_register_hints["return"] == Callable[
        [prior_bank.EncodeCallable], prior_bank.EncodeCallable
    ]
    assert get_type_hints(prior_bank.get_encode)["return"] == prior_bank.EncodeCallable

    decode_register_hints = get_type_hints(prior_bank.register_decode)
    assert decode_register_hints["fused_ce"] == Optional[prior_bank.FusedCECallable]
    assert decode_register_hints["return"] == Callable[
        [prior_bank.DecodeCallable], prior_bank.DecodeCallable
    ]
    assert get_type_hints(prior_bank.get_decode)["return"] == prior_bank.DecodeCallable


def test_cited_capture_and_gradient_seams_use_shared_typeddicts() -> None:
    contracts = _contracts()
    cases = (
        (vfe_block, "capture", Optional[contracts.MStepCapture]),
        (vfe_block, "grad_record", Optional[contracts.EStepGradientRecord]),
        (vfe_stack, "capture", Optional[contracts.MStepCapture]),
        (vfe_stack, "grad_record", Optional[contracts.EStepGradientRecord]),
        (e_step_iteration, "grad_record", Optional[contracts.EStepGradientRecord]),
        (e_step, "grad_record", Optional[contracts.EStepGradientRecord]),
        (VFEModel.forward_beliefs, "capture", Optional[contracts.MStepCapture]),
        (VFEModel.forward_beliefs, "estep_grad_out", Optional[contracts.EStepGradientOutput]),
        (VFEModel.forward, "estep_grad_out", Optional[contracts.EStepGradientOutput]),
    )

    for fn, parameter, expected in cases:
        assert signature(fn).parameters[parameter].annotation not in (dict, Optional[dict])
        assert get_type_hints(fn)[parameter] == expected


def test_checkpoint_and_map_writer_contracts_are_precise() -> None:
    contracts = _contracts()

    assert get_type_hints(RunArtifacts.save_checkpoint)["data_state"] == Optional[
        contracts.DataState
    ]
    assert get_type_hints(load_checkpoint)["data_state"] == Optional[contracts.DataStateBuffer]
    expected_paths = Optional[List[Path]]
    assert get_type_hints(RunArtifacts.save_attention_maps)["return"] == expected_paths
    assert get_type_hints(RunArtifacts.save_gamma_attention_maps)["return"] == expected_paths


def test_as_coeff_accepts_and_documents_tuple_coefficients() -> None:
    annotation = get_type_hints(_as_coeff)["v"]
    assert set(get_args(annotation)) == {float, list, tuple}
    assert "tuple" in (_as_coeff.__doc__ or "").lower()
    assert torch.equal(
        _as_coeff((1.0, 2.0), torch.device("cpu")),
        torch.tensor([1.0, 2.0]),
    )


def test_half_fisher_human_labels_name_the_quantity_and_show_the_factor() -> None:
    history = {"step": [0, 1], "fisher_trace_mean": [1.0, 2.0]}
    fig = figures.plot_geometry_health(history)
    validation = figures.plot_validation_sanity(
        {"step": [0, 1], "val_fisher_trace_mean": [1.0, 2.0]}
    )
    try:
        for dashboard in (fig, validation):
            text = " ".join(
                [axis.get_ylabel() for axis in dashboard.axes]
                + [
                    item.get_text()
                    for axis in dashboard.axes
                    if axis.get_legend() is not None
                    for item in axis.get_legend().texts
                ]
            )
            assert "Half Fisher trace" in text
            assert "/2" in text or r"\frac{1}{2}" in text
    finally:
        figures.plt.close(fig)
        figures.plt.close(validation)

    for key in ("fisher_trace_mean", "fisher_trace_median"):
        label = figures.pub_label(key)
        assert "Half Fisher trace" in label
        assert "/2" in label or r"\frac{1}{2}" in label
