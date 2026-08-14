"""Task 6 review regressions: exact metrics, labels, and belief-state boundaries."""
from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import nn


def _scaling_payload(ce: float) -> dict[str, object]:
    try:
        ppl = math.exp(ce)
    except OverflowError:
        ppl = float("inf")
    metrics = {
        "n_params": 17,
        "test_ce": ce,
        "test_ppl": ppl,
        "test_bits_per_token": ce / math.log(2.0),
        "test_bpc": None,
    }
    return {**metrics, "scaling_point": dict(metrics)}


@pytest.mark.parametrize("ce", [25.0, 1_000.0])
def test_exact_ppl_contract_is_shared_by_both_scaling_consumers(ce: float) -> None:
    import scaling
    import scaling_analysis

    payload = _scaling_payload(ce)
    assert scaling._scaling_result_status(payload) == "complete"
    assert scaling_analysis._validated_scaling_metrics(payload) is not None


def test_training_partition_uses_target_size_not_decoder_self_report() -> None:
    from vfe3.train import _target_accounting_for_targets

    targets = torch.tensor([[1, 2], [3, -100]], dtype=torch.int64)
    with pytest.raises(RuntimeError, match=r"4 != 1 \+ 1"):
        _target_accounting_for_targets(
            targets,
            torch.tensor(1, dtype=torch.int64),
            torch.tensor(1, dtype=torch.int64),
        )


def test_evaluation_count_transfer_preserves_first_integer_above_float64_exactness() -> None:
    from vfe3.train import _evaluation_totals_from_device

    boundary = 2**53 + 1
    nats, accounting = _evaluation_totals_from_device(
        torch.tensor(7.5, dtype=torch.float64),
        torch.tensor(boundary, dtype=torch.int64),
        torch.tensor(boundary - 4, dtype=torch.int64),
        torch.tensor(4, dtype=torch.int64),
    )
    assert nats == 7.5
    assert accounting == {
        "expected_targets": boundary,
        "scored_targets": boundary - 4,
        "excluded_targets": 4,
    }


def test_device_count_transfer_rejects_non_int64_inputs() -> None:
    from vfe3.train import _target_accounting_from_device

    with pytest.raises(TypeError, match="int64"):
        _target_accounting_from_device(
            torch.tensor(3.0, dtype=torch.float64),
            torch.tensor(2, dtype=torch.int64),
            torch.tensor(1, dtype=torch.int64),
        )


def test_scheduler_metadata_matches_executable_positive_zero_and_mixed_groups() -> None:
    from vfe3.config import VFE3Config
    from vfe3.run_artifacts import _scheduler_metadata
    from vfe3.train import _floor_lr_lambdas

    cfg = VFE3Config(warmup_steps=1, max_steps=4, min_lr=0.1, min_lr_frac=0.25)
    base_lrs = [1.0, 0.0, 0.2]
    metadata = _scheduler_metadata(
        min_lr=cfg.min_lr,
        min_lr_frac=cfg.min_lr_frac,
        base_lrs=base_lrs,
        group_roles=["mu", "phi", "sigma"],
    )
    executable = [base * fn(cfg.max_steps)
                  for base, fn in zip(base_lrs, _floor_lr_lambdas(base_lrs, cfg))]
    assert metadata["zero_base_groups_remain_frozen"] is True
    assert [group["base_lr"] for group in metadata["groups"]] == base_lrs
    assert [group["effective_floor"] for group in metadata["groups"]] == pytest.approx(executable)
    assert [group["frozen"] for group in metadata["groups"]] == [False, True, False]


def test_builtin_functionals_have_typed_display_metadata() -> None:
    from vfe3.families.base import FunctionalDisplayMetadata, get_functional_display

    for name in ("renyi", "squared_hellinger", "bhattacharyya", "jeffreys"):
        display = get_functional_display(name)
        assert isinstance(display, FunctionalDisplayMetadata)
        assert display.label
        assert display.units


def test_real_registry_override_preserves_or_replaces_display_metadata() -> None:
    from vfe3.families import base

    name = "task6_review_override"
    display_a = base.FunctionalDisplayMetadata("review A", "units-A")
    display_b = base.FunctionalDisplayMetadata("review B", "units-B")

    def first(*_args, **_kwargs):
        return torch.tensor(1.0)

    def second(*_args, **_kwargs):
        return torch.tensor(2.0)

    def third(*_args, **_kwargs):
        return torch.tensor(3.0)

    try:
        base.register_functional(name, display=display_a)(first)
        base.register_functional(name, override=True)(second)
        assert base.get_functional(name) is second
        assert base.get_functional_display(name) == display_a
        base.register_functional(name, display=display_b, override=True)(third)
        assert base.get_functional(name) is third
        assert base.get_functional_display(name) == display_b
    finally:
        base._FUNCTIONALS.pop(name, None)
        base._FUNCTIONAL_DISPLAYS.pop(name, None)


def _figure_text(fig) -> str:
    chunks: list[str] = []
    for axis in fig.axes:
        chunks.extend((axis.get_title(), axis.get_xlabel(), axis.get_ylabel()))
        chunks.extend(text.get_text() for text in axis.texts)
    return " ".join(chunks).lower()


def test_all_objective_plots_use_real_registered_metadata_and_neutral_defaults() -> None:
    from vfe3.viz import figures

    evidence = {
        "iterate_label": "final_iterate",
        "descent_evidence": False,
        "convergence_evidence": False,
        "fixed_point_evidence": False,
    }
    history = {
        "step": np.arange(6),
        "self_coupling": np.linspace(3.0, 2.0, 6),
        "belief_coupling": np.linspace(1.0, 0.8, 6),
        "attention_entropy": np.linspace(0.4, 0.3, 6),
        "val_ce": np.linspace(2.5, 2.2, 6),
    }
    trace = {
        "free_energy": np.array([3.0, 2.8]),
        "mu": torch.stack((torch.zeros((3, 2)), torch.full((3, 2), 0.1))),
        "sigma": torch.stack((torch.ones((3, 2)), torch.full((3, 2), 1.1))),
        "phi": torch.stack((torch.zeros((3, 4)), torch.full((3, 4), 0.05))),
    }
    figures_to_check = [
        figures.plot_free_energy_relationship(
            history, divergence_family="squared_hellinger"),
        figures.plot_iterate_trajectory(
            trace, divergence_family="squared_hellinger", evidence=evidence),
        figures.plot_estep_capacity(
            [1, 2], [3.0, 2.9], [5.0, 4.9],
            divergence_family="squared_hellinger", evidence=evidence),
        figures.plot_f_ce_relationship(
            [{"n_e_steps": 1, "final_f": 5.0, "ce": 3.0},
             {"n_e_steps": 2, "final_f": 4.9, "ce": 2.9}],
            divergence_family="squared_hellinger"),
    ]
    try:
        text = " ".join(_figure_text(fig) for fig in figures_to_check)
        assert "squared hellinger divergence" in text
        assert "dimensionless" in text
        for unsupported in ("convergence", "converged", "fixed point", "descent", "decorrelation"):
            assert unsupported not in text
    finally:
        for fig in figures_to_check:
            plt.close(fig)


def test_no_halt_default_artifact_names_are_evidence_neutral() -> None:
    import scaling_analysis
    from vfe3.run_artifacts import _estep_endpoint_artifact
    from vfe3.viz import figures
    from vfe3.viz.report import plan_single_run_figures

    evidence = {
        "descent_evidence": False,
        "convergence_evidence": False,
        "fixed_point_evidence": False,
    }
    endpoint = _estep_endpoint_artifact(evidence)
    planned = plan_single_run_figures("synthetic", {"iterate_trajectory": True})
    assert endpoint["filename"] == "estep_endpoint_delta.png"
    assert endpoint["title"] == "E-step endpoint delta"
    assert "iterate_trajectory.png" in planned
    assert scaling_analysis.F_CE_RELATIONSHIP_FILENAME == "f_ce_relationship.png"
    assert figures.get_figure("iterate_trajectory") is figures.plot_iterate_trajectory
    assert figures.get_figure("estep_convergence") is figures.plot_iterate_trajectory
    assert figures.get_figure("f_ce_relationship") is figures.plot_f_ce_relationship
    assert figures.get_figure("f_ce_decorrelation") is figures.plot_f_ce_relationship
    surfaces = " ".join((*planned, endpoint["filename"], endpoint["title"],
                         scaling_analysis.F_CE_RELATIONSHIP_FILENAME)).lower()
    for unsupported in ("convergence", "descent", "decorrelation", "fixed_point"):
        assert unsupported not in surfaces


class _PairAdd(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor):
        return mu + self.amount, sigma


class _MeanAdd(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return mu + self.amount


@pytest.mark.parametrize("with_mlp", [False, True])
def test_block_capture_is_the_immediate_mlp_boundary_after_mixer_cg_and_norm(
    with_mlp: bool,
) -> None:
    from vfe3.config import VFE3Config
    from vfe3.model.block import vfe_block
    from vfe3.model.model import VFEModel

    cfg = VFE3Config(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=3,
                     n_layers=1, n_e_steps=1, e_phi_lr=0.0)
    model = VFEModel(cfg)
    belief = model.prior_bank.encode(torch.tensor([[1, 2, 3]]))
    capture: dict[str, object] = {}
    mlp = _MeanAdd(4.0) if with_mlp else None
    out = vfe_block(
        belief,
        belief.mu,
        belief.sigma,
        model.group,
        cfg,
        head_mixer=_PairAdd(1.0),
        cg_coupling=_PairAdd(2.0),
        block_norm=lambda mu, sigma: mu + 3.0,
        block_mlp=mlp,
        capture=capture,
    )
    converged = capture["converged"]
    mlp_input = capture["block_mlp_input"]
    mlp_output = capture["block_mlp_output"]
    assert torch.allclose(mlp_input.mu, converged.mu + 6.0)
    if with_mlp:
        assert torch.allclose(mlp_output.mu, mlp_input.mu + 4.0)
        assert mlp_output is out
    else:
        assert mlp_input is mlp_output is out


@pytest.mark.parametrize("use_block_mlp", [False, True])
@pytest.mark.parametrize("model_channel", [False, True])
def test_state_diagnostics_reconcile_all_terms_at_actual_mlp_states(
    use_block_mlp: bool, model_channel: bool,
) -> None:
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_kwargs = {
            "prior_source": "model_channel",
            "s_e_step": True,
            "use_prior_bank": False,
            "lambda_h": 0.25,
            "lambda_h_mode": "state_dependent",
            "lambda_gamma": 0.5,
            "decode_mode": "diagonal_chunked",
        } if model_channel else {}
        cfg = VFE3Config(
            vocab_size=9,
            embed_dim=4,
            n_heads=1,
            max_seq_len=4,
            n_layers=1,
            n_e_steps=1,
            e_phi_lr=0.0,
            gauge_group="so_k",
            phi_reflection="init_seed",
            pos_phi="none",
            lambda_twohop=0.2,
            use_block_mlp=use_block_mlp,
            **model_kwargs,
        )
        torch.manual_seed(13)
        model = VFEModel(cfg).eval()
    if use_block_mlp:
        with torch.no_grad():
            model.block_mlps[0].fc2.bias.add_(0.25)
    tokens = torch.tensor([[0, 1, 2, 3]])
    diagnostic = model.diagnostics(
        tokens,
        log_likelihood=torch.tensor([-0.1, -0.2, -0.3, -0.4]),
    )
    components = (
        "self_coupling",
        "belief_coupling",
        "attention_entropy",
        "twohop_coupling",
        "hyper_prior",
        "model_coupling",
        "meta_entropy",
        "observation_nll",
    )
    for prefix in ("pre_block_mlp_", "post_block_mlp_"):
        assert diagnostic[prefix + "total"] == pytest.approx(
            sum(diagnostic[prefix + component] for component in components), abs=1e-5)
        assert prefix + "self_divergence" in diagnostic
        assert prefix + "observation_likelihood" in diagnostic
        assert prefix + "hyper_prior_raw" in diagnostic
        assert prefix + "gamma_coupling_raw" in diagnostic
        assert prefix + "gamma_meta_entropy_raw" in diagnostic
        if model_channel:
            assert abs(diagnostic[prefix + "hyper_prior_raw"]) > 0.0
            assert diagnostic[prefix + "model_coupling"] == pytest.approx(
                cfg.lambda_gamma * diagnostic[prefix + "gamma_coupling_raw"], abs=1e-6)
            assert diagnostic[prefix + "meta_entropy"] == pytest.approx(
                cfg.lambda_gamma * diagnostic[prefix + "gamma_meta_entropy_raw"], abs=1e-6)
    assert diagnostic["total"] == diagnostic["post_block_mlp_total"]
    assert bool(diagnostic["block_mlp_state_equality_proven"]) is (not use_block_mlp)
