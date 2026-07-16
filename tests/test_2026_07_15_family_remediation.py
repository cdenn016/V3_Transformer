"""Regressions for the second-panel family/mixer remediation (S2-I1--I6, T2, C1)."""

import copy
import logging
import math
from types import SimpleNamespace

import pytest
import torch

from vfe3 import metrics
from vfe3.config import VFE3Config
from vfe3.geometry.norms import MahalanobisNorm
from vfe3.model.head_mixer import HeadMixer
from vfe3.model.model import VFEModel, _precision_key_bias
from vfe3.numerics import apply_mu_trust_region
from vfe3.run_artifacts import _pure_path_report
from vfe3.viz.extract import belief_ce_bank, model_channel_belief, numerical_health


def _tiny_config(**overrides) -> VFE3Config:
    values = dict(
        vocab_size=8,
        embed_dim=2,
        n_heads=1,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        m_phi_lr=0.0,
        e_step_gradient="detach",
        pos_phi="none",
    )
    values.update(overrides)
    return VFE3Config(**values)


def test_gaussian_public_statistical_paths_keep_current_values() -> None:
    """Pin the pre-refactor Gaussian formulas at every public consumer seam."""
    variance = torch.tensor([[4.0, 1.0]])
    mean = torch.tensor([[2.0, 3.0]])

    normalized = MahalanobisNorm(2)(mean, variance)
    s2 = (mean.square() / variance).sum(dim=-1, keepdim=True)
    assert torch.equal(normalized, mean * torch.sqrt(2.0 / s2))

    bounded = apply_mu_trust_region(
        torch.tensor([[12.0]]),
        torch.tensor([[4.0]]),
        trust=5.0,
        mode="box",
        is_diagonal=True,
    )
    assert torch.equal(bounded, torch.tensor([[10.0]]))

    bias = _precision_key_bias(variance, b0=1.0)
    assert torch.equal(bias, -torch.log(torch.tensor([6.0])))
    assert torch.equal(metrics.sigma_trace(variance), variance.sum(dim=-1))
    assert torch.equal(metrics.half_fisher_trace(variance), (0.5 / variance).sum(dim=-1))

    mixer = HeadMixer([1, 1])
    with torch.no_grad():
        mixer.mixer_delta.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    _, mixed = mixer(torch.zeros_like(variance), variance)
    assert torch.equal(mixed, torch.tensor([[5.0, 1.0]]))

    indefinite = torch.diag(torch.tensor([-1e-6, 1.0]))
    spectrum = metrics.belief_spectrum(
        indefinite,
        diagonal=False,
        eps=1e-6,
        family="gaussian_full",
    )
    assert torch.equal(spectrum["eigenvalues"], torch.tensor([1.0, -1e-6]))
    assert torch.isinf(spectrum["condition"])


def test_laplace_head_mixer_uses_moment_matched_scale() -> None:
    mixer = HeadMixer([1, 1], family="laplace_diagonal")
    with torch.no_grad():
        mixer.mixer_delta.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    scale = torch.tensor([[2.0, 3.0]])

    _, mixed = mixer(torch.zeros_like(scale), scale)

    assert torch.allclose(mixed, torch.tensor([[math.sqrt(13.0), 3.0]]))


def test_laplace_mahalanobis_norm_uses_mean_fisher_precision() -> None:
    mean = torch.tensor([[2.0, 3.0]])
    scale = torch.tensor([[4.0, 1.0]])
    norm = MahalanobisNorm(2, family="laplace_diagonal")

    actual = norm(mean, scale)

    fisher2 = (mean.square() / scale.square()).sum(dim=-1, keepdim=True)
    expected = mean * torch.sqrt(2.0 / fisher2)
    assert torch.equal(actual, expected)


def test_gaussian_mahalanobis_preserves_division_arithmetic_order() -> None:
    mean = torch.tensor([[
        -8.92149829864502,
        5.149660110473633,
        -7.426316738128662,
        -0.3939628601074219,
        -5.77911376953125,
        0.2099761962890625,
        9.311418533325195,
    ]])
    variance = torch.tensor([[
        950.94677734375,
        0.0007257265388034284,
        0.00260880496352911,
        0.07082255184650421,
        120.90992736816406,
        57.44921875,
        0.06778386980295181,
    ]])
    direct_s2 = ((mean ** 2) / variance).sum(dim=-1, keepdim=True)
    reciprocal_s2 = ((mean ** 2) * variance.reciprocal()).sum(dim=-1, keepdim=True)
    expected = mean * torch.sqrt(7.0 / direct_s2)
    reciprocal_result = mean * torch.sqrt(7.0 / reciprocal_s2)

    actual = MahalanobisNorm(7)(mean, variance)

    assert not torch.equal(expected, reciprocal_result)
    assert torch.equal(actual, expected)


def test_laplace_trust_region_whitens_by_scale_b() -> None:
    actual = apply_mu_trust_region(
        torch.tensor([[30.0]]),
        torch.tensor([[4.0]]),
        trust=5.0,
        mode="box",
        is_diagonal=True,
        family="laplace_diagonal",
    )

    assert torch.equal(actual, torch.tensor([[20.0]]))


def test_laplace_attention_reliability_uses_covariance_trace() -> None:
    scale = torch.tensor([[[2.0, 3.0]]])

    actual = _precision_key_bias(scale, b0=1.0, family="laplace_diagonal")

    expected = -torch.log(torch.tensor([[1.0 + 2.0 * (2.0**2 + 3.0**2)]]))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("field", ["kl_max", "renyi_order"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_config_rejects_nonfinite_or_nonpositive_divergence_controls(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be finite and positive"):
        VFE3Config(**{field: value})


def test_laplace_metrics_use_family_fisher_and_covariance_statistics() -> None:
    scale = torch.tensor([[2.0, 3.0]])

    trace = metrics.sigma_trace(scale, family="laplace_diagonal")
    fisher = metrics.half_fisher_trace(scale, family="laplace_diagonal")
    spectrum = metrics.belief_spectrum(scale, family="laplace_diagonal")
    registered_rank = metrics.compute_metrics(
        ["effective_rank"],
        sigma=scale,
        diagonal=True,
        family="laplace_diagonal",
    )["effective_rank"]

    assert torch.equal(trace, torch.tensor([26.0]))
    assert torch.equal(fisher, torch.tensor([0.5 * (1.0 / 4.0 + 1.0 / 9.0)]))
    assert torch.equal(spectrum["eigenvalues"], torch.tensor([[18.0, 8.0]]))
    assert spectrum["dispersion_label"] == "Laplace scale b"
    assert spectrum["spectrum_label"] == "marginal covariance variance"
    assert registered_rank == pytest.approx(float((26.0**2) / (18.0**2 + 8.0**2)))


def test_laplace_near_floor_spectrum_uses_covariance_units() -> None:
    from vfe3.viz import figures

    scale = torch.tensor([[1e-6, 2e-6]])
    spectrum = metrics.belief_spectrum(
        scale,
        eps=1e-6,
        family="laplace_diagonal",
    )
    per_token_rank = metrics.effective_rank_per_token(
        scale,
        eps=1e-6,
        family="laplace_diagonal",
    )

    assert torch.allclose(
        spectrum["eigenvalues"],
        torch.tensor([[8e-12, 2e-12]]),
        rtol=1e-6,
        atol=0.0,
    )
    assert spectrum["condition"].item() == pytest.approx(4.0)
    assert spectrum["effective_rank"].item() == pytest.approx(100.0 / 68.0)
    assert per_token_rank.item() == pytest.approx(100.0 / 68.0)

    figure = figures.plot_belief_spectrum(
        scale,
        eps=1e-6,
        family="laplace_diagonal",
    )
    try:
        floor_line = figure.axes[1].lines[1]
        assert floor_line.get_ydata()[0] == pytest.approx(2e-12)
    finally:
        figures.plt.close(figure)


def test_gaussian_near_floor_effective_rank_keeps_legacy_stabilizer() -> None:
    variance = torch.tensor([[1e-6, 2e-6]])
    baseline = metrics.effective_rank(variance, eps=1e-6)
    per_token = metrics.effective_rank_per_token(
        variance,
        eps=1e-6,
        family="gaussian_diagonal",
    )
    spectrum = metrics.belief_spectrum(
        variance,
        eps=1e-6,
        family="gaussian_diagonal",
    )

    assert baseline.item() == pytest.approx(9e-6)
    assert torch.equal(per_token, baseline)
    assert torch.equal(spectrum["effective_rank"], baseline)


def test_laplace_fisher_figure_labels_name_scale_precision() -> None:
    from vfe3.viz import figures

    geometry = figures.plot_geometry_health(
        {"step": [0, 1], "fisher_trace_mean": [2.0, 1.0]},
        family="laplace_diagonal",
    )
    validation = figures.plot_validation_sanity(
        {"step": [0, 1], "val_fisher_trace_mean": [2.0, 1.0]},
        family="laplace_diagonal",
    )
    try:
        for dashboard in (geometry, validation):
            text = " ".join(
                [axis.get_ylabel() for axis in dashboard.axes]
                + [
                    item.get_text()
                    for axis in dashboard.axes
                    if axis.get_legend() is not None
                    for item in axis.get_legend().texts
                ]
            )
            assert r"b_k^{-2}" in text
            assert r"\Sigma^{-1}" not in text
    finally:
        figures.plt.close(geometry)
        figures.plt.close(validation)

    publication_label = figures.pub_label(
        "fisher_trace_mean",
        family="laplace_diagonal",
    )
    assert r"b_k^{-2}" in publication_label
    assert r"\Sigma^{-1}" not in publication_label


def test_run_artifact_fisher_dashboards_receive_family(tmp_path, monkeypatch) -> None:
    from vfe3.run_artifacts import _save_figures
    from vfe3.viz import figures

    received = {}

    def _geometry(history, *, family=None, **kwargs):
        received["geometry"] = family
        return figures.plt.figure()

    def _validation(history, *, family=None, **kwargs):
        received["validation"] = family
        return figures.plt.figure()

    monkeypatch.setattr(figures, "plot_geometry_health", _geometry)
    monkeypatch.setattr(figures, "plot_validation_sanity", _validation)
    artifacts = SimpleNamespace(
        run_dir=tmp_path,
        history=[{
            "step": 0,
            "fisher_trace_mean": 1.0,
            "val_fisher_trace_mean": 1.0,
        }],
        cfg=SimpleNamespace(
            family="laplace_diagonal",
            transport_mode="flat",
        ),
    )

    _save_figures(artifacts, None, logging.getLogger("task-3-label-wiring"))

    assert received == {
        "geometry": "laplace_diagonal",
        "validation": "laplace_diagonal",
    }


def test_laplace_model_diagnostics_use_family_statistics() -> None:
    torch.manual_seed(0)
    model = VFEModel(_tiny_config(family="laplace_diagonal"))
    tokens = torch.tensor([[1, 2, 3, 4]])
    snapshot = model.build_diagnostic_snapshot(tokens)

    diagnostic = model.diagnostics(tokens, snapshot=snapshot)
    health = numerical_health(model, tokens, snapshot=snapshot)

    scale = snapshot.stack_output.sigma[0]
    fisher = 0.5 * scale.clamp_min(model.cfg.eps).square().reciprocal().sum(dim=-1)
    covariance = 2.0 * scale.square()
    condition = covariance.max(dim=-1).values / covariance.min(dim=-1).values
    effective_rank = covariance.sum(dim=-1).square() / covariance.square().sum(dim=-1)
    assert diagnostic["fisher_trace_mean"] == pytest.approx(float(fisher.mean()), abs=1e-6)
    assert diagnostic["belief_cond_median"] == pytest.approx(float(condition.median()), abs=1e-6)
    assert diagnostic["effective_rank"] == pytest.approx(float(effective_rank.mean()), abs=1e-6)
    assert health["max_condition"] == pytest.approx(float(condition.max()), abs=1e-6)


def test_laplace_extraction_reports_covariance_not_raw_scale() -> None:
    torch.manual_seed(1)
    model = VFEModel(_tiny_config(family="laplace_diagonal"))
    tokens = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    snapshot = model.build_diagnostic_snapshot(tokens)

    bank = belief_ce_bank(model, [(tokens, targets)], max_batches=1)

    expected = (2.0 * snapshot.stack_output.sigma.square()).sum(dim=-1).reshape(-1)
    assert torch.allclose(bank["tr_sigma"], expected, atol=1e-6, rtol=0.0)


def test_laplace_model_channel_extraction_labels_scale_and_covariance() -> None:
    torch.manual_seed(2)
    model = VFEModel(_tiny_config(
        family="laplace_diagonal",
        prior_source="model_channel",
    ))
    tokens = torch.tensor([[1, 2, 3, 4]])
    _, scale = model.prior_bank.encode_s(tokens[:1])

    extracted = model_channel_belief(model, tokens)

    assert extracted is not None
    expected = torch.sort(2.0 * scale[0].square(), dim=-1, descending=True).values.cpu()
    assert torch.equal(extracted["spectrum"], expected)
    assert extracted["dispersion_label"] == "Laplace scale b"
    assert extracted["spectrum_label"] == "marginal covariance variance"


def test_independent_head_mixer_metadata_matches_nonzero_commutator() -> None:
    mixer = HeadMixer([1, 1])
    with torch.no_grad():
        mixer.mixer_delta.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    mixing = mixer._dense_m(torch.device("cpu"), torch.float32)
    independent_gauge = torch.diag(torch.tensor([2.0, 3.0]))
    commutator = mixing @ independent_gauge - independent_gauge @ mixing
    assert float(commutator.detach().abs().max()) > 0.0

    incompatible = VFE3Config(use_head_mixer=True, gauge_group="block_glk")
    tied_diagonal = VFE3Config(use_head_mixer=True, gauge_group="tied_block_glk")
    tied_full = VFE3Config(
        use_head_mixer=True,
        gauge_group="tied_block_glk",
        family="gaussian_full",
        use_prior_bank=True,
        decode_mode="family",
        e_step_gradient="detach",
    )
    disabled = VFE3Config(use_head_mixer=False, gauge_group="block_glk")
    assert incompatible.head_mixer_gauge_compatible is False
    assert incompatible.head_mixer_compatibility == "independent_head_nonintertwiner"
    assert tied_diagonal.head_mixer_gauge_compatible is False
    assert tied_diagonal.head_mixer_compatibility == "tied_diagonal_projection_nonintertwiner"
    assert tied_full.head_mixer_gauge_compatible is True
    assert tied_full.head_mixer_compatibility == "tied_intertwiner"
    assert disabled.head_mixer_gauge_compatible is True
    assert disabled.head_mixer_compatibility == "disabled"

    report = _pure_path_report(incompatible, [])
    assert report["gauge_flags"]["head_mixer_intertwiner_compatible"] is False
    assert report["on_gauge_pure_path"] is False
    assert report["config_toggles"]["head_mixer_compatibility"] == "independent_head_nonintertwiner"


def test_no_grad_identity_shortcut_does_not_call_tensor_item(monkeypatch) -> None:
    mixer = HeadMixer([1, 1])
    mean = torch.randn(3, 2)
    variance = torch.rand(3, 2)

    def _forbid_item(self, *args, **kwargs):
        raise AssertionError("HeadMixer.forward synchronized through Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
    with torch.no_grad():
        mixed_mean, mixed_variance = mixer(mean, variance)

    assert mixed_mean is mean
    assert mixed_variance is variance

    trained = HeadMixer([1, 1])
    with torch.no_grad():
        trained.mixer_delta.copy_(torch.tensor([[0.0, 0.25], [0.0, 0.0]]))
    copied = copy.deepcopy(trained)
    with torch.no_grad():
        copied_mean, _ = copied(mean, variance)

    assert copied_mean is not mean
    assert torch.allclose(copied_mean, mean @ torch.tensor([[1.0, 0.0], [0.25, 1.0]]))
