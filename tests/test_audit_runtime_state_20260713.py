"""Regression tests for the validated 2026-07-13 runtime/state audit findings."""

import logging
import types
import warnings

import pytest
import torch

from vfe3 import metrics
from vfe3.config import VFE3Config
from vfe3.ema import EMA
from vfe3.families.base import get_family
from vfe3.model.model import VFEModel
from vfe3.train import _save_eval_attention_maps, _val_diagnostics


def _tiny_config(**overrides) -> VFE3Config:
    values = dict(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        n_layers=1,
        n_e_steps=1,
        e_q_mu_lr=0.05,
        e_phi_lr=0.0,
    )
    values.update(overrides)
    return VFE3Config(**values)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_sigma_trust_radius_requires_positive_finite_value(value):
    with pytest.raises(ValueError, match="e_sigma_q_trust"):
        _tiny_config(e_sigma_q_trust=value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_phi_metropolis_requires_positive_finite_temperature(value):
    with pytest.raises(ValueError, match="omega_metropolis_temperature"):
        _tiny_config(phi_reflection="metropolis", omega_metropolis_temperature=value)


def test_sigma_max_none_disables_ceiling_diagnostic():
    sigma = torch.ones(2, 4)
    result = metrics.guard_saturation(
        sigma,
        torch.zeros(2, 2),
        torch.zeros(2),
        sigma_max=None,
    )
    assert result["sigma_ceil_frac"] == 0.0


def test_linear_decode_warns_when_nondefault_decode_tau_is_inert():
    with pytest.warns(UserWarning, match="decode_tau=.*inert.*use_prior_bank=False"):
        _tiny_config(use_prior_bank=False, decode_tau=0.25)


def test_free_energy_metric_wrapper_keeps_tensor_annotations():
    annotations = metrics._m_free_energy_terms.__annotations__
    assert annotations["self_div"] is torch.Tensor
    assert annotations["energy"] is torch.Tensor
    assert annotations["beta"] is torch.Tensor
    assert annotations["alpha"] is torch.Tensor


def test_model_channel_tables_are_encoded_once_for_hyper_and_gamma(monkeypatch):
    model = VFEModel(_tiny_config(lambda_h=0.2, lambda_gamma=0.3))
    token_ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    original = model.prior_bank.encode_s
    calls = []

    def counted(ids):
        calls.append(ids)
        return original(ids)

    monkeypatch.setattr(model.prior_bank, "encode_s", counted)
    model(token_ids, targets)
    assert len(calls) == 1


def test_cg_energy_diagnostics_remain_device_tensors():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(_tiny_config(
            gauge_group="so_n",
            group_n=3,
            irrep_spec=[("l0", 1), ("l1", 1)],
            use_cg_coupling=True,
            cg_energy_weight=0.5,
        ))
    token_ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    model(token_ids, targets)
    diagnostic = model._cg_energy_diagnostics
    assert diagnostic["cg_moment_energy"].device == token_ids.device
    assert diagnostic["objective_total_with_cg"].device == token_ids.device
    assert diagnostic["cg_moment_energy_layers"].device == token_ids.device


class _OneParameter(torch.nn.Module):
    def __init__(self, *, trainable: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=trainable)


def test_ema_load_reconciles_same_key_trainability_drift():
    frozen = _OneParameter(trainable=False)
    saved = EMA(frozen).state_dict()

    current = _OneParameter(trainable=True)
    ema = EMA(current)
    with torch.no_grad():
        current.weight.fill_(7.0)
    with pytest.warns(UserWarning, match="EMA shadow keys differed"):
        ema.load_state_dict(saved)

    assert set(ema.shadow) == {"weight"}
    assert torch.equal(ema.shadow["weight"], current.weight)


def test_ema_load_drops_keys_that_are_no_longer_trainable():
    saved = EMA(_OneParameter(trainable=True)).state_dict()
    current = _OneParameter(trainable=False)
    ema = EMA(current)
    with pytest.warns(UserWarning, match="EMA shadow keys differed"):
        ema.load_state_dict(saved)
    assert ema.shadow == {}


def test_ema_copy_recomputes_barycenter_and_restore_returns_raw_centroid():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(_tiny_config(
            lambda_h=0.2,
            learnable_r=True,
            r_update_mode="barycenter",
            prior_source="model_channel",
        ))
    ema = EMA(model, decay=0.9)
    averaged_s = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10.0
    ema.shadow["prior_bank.s_mu_embed"] = averaged_s.clone()
    with torch.no_grad():
        model.prior_bank.r_mu.fill_(-9.0)
    raw_r = model.prior_bank.r_mu.detach().clone()

    ema.store(model)
    ema.copy_to(model)
    assert torch.allclose(model.prior_bank.r_mu, averaged_s.mean(dim=0))

    ema.restore(model)
    assert torch.equal(model.prior_bank.r_mu, raw_r)


def test_eval_attention_pair_uses_one_shared_post_eval_snapshot():
    snapshot = object()

    class FakeModel:
        def __init__(self):
            self.built = 0
            self.seen = []

        def build_diagnostic_snapshot(self, tokens):
            self.built += 1
            self.seen.append(tokens.clone())
            return snapshot

        def attention_maps(self, tokens, *, snapshot=None):
            assert snapshot is not None
            self.seen.append(tokens.clone())
            return torch.ones(1, 1, 2, 2)

        def gamma_attention_maps(self, tokens, *, snapshot=None):
            assert snapshot is not None
            self.seen.append(tokens.clone())
            return torch.ones(1, 2, 2)

    class FakeArtifacts:
        def __init__(self):
            self.saved = []

        def save_attention_maps(self, step, maps, logger=None):
            self.saved.append(("beta", step, maps))

        def save_gamma_attention_maps(self, step, maps, logger=None):
            self.saved.append(("gamma", step, maps))

    model = FakeModel()
    artifacts = FakeArtifacts()
    tokens = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    _save_eval_attention_maps(
        tokens,
        snapshot,
        model,
        artifacts,
        logging.getLogger(__name__),
        step=3,
    )
    assert model.built == 0
    assert len(model.seen) == 2
    assert all(torch.equal(seen, tokens) for seen in model.seen)
    assert [item[0] for item in artifacts.saved] == ["beta", "gamma"]


def test_gamma_meta_entropy_uses_exact_log_softmax_prior(monkeypatch):
    model = VFEModel(_tiny_config(lambda_gamma=0.2))
    energy = torch.zeros(1, 1, 2, 2)
    log_prior = torch.tensor([[0.0, -40.0], [0.0, -40.0]])
    gamma = torch.full_like(energy, 0.5)

    def fake_energy(self, token_ids, phi, **kwargs):
        return energy, 1.0, log_prior

    model._gamma_energy = types.MethodType(fake_energy, model)
    monkeypatch.setattr("vfe3.free_energy.attention_weights", lambda *args, **kwargs: gamma)
    coupling, meta = model._gamma_coupling_rows(
        torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1, 2, model.group.generators.shape[0]),
        head_reduction="mean",
    )
    expected_log_prior = torch.log_softmax(log_prior, dim=-1)
    expected = (gamma * (torch.log(gamma) - expected_log_prior)).sum(dim=-1).mean(dim=1)
    assert torch.equal(coupling, torch.zeros_like(coupling))
    torch.testing.assert_close(meta, expected)


def test_model_diagnostics_dispatch_dispersion_transport_through_family(monkeypatch):
    family = get_family("laplace_diagonal")
    original = family.transport_dispersion
    calls = []

    def tracked(cls, dispersion, omega, *, diagonal_out=None):
        calls.append((dispersion.shape, diagonal_out))
        return original(dispersion, omega, diagonal_out=diagonal_out)

    monkeypatch.setattr(family, "transport_dispersion", classmethod(tracked))
    model = VFEModel(_tiny_config(family="laplace_diagonal", lambda_gamma=0.2))
    token_ids = torch.tensor([[0, 1, 2, 3]])
    snapshot = model.build_diagnostic_snapshot(token_ids)
    calls.clear()

    model.diagnostics(token_ids, snapshot=snapshot)
    model.diagnostics_per_layer(token_ids, snapshot=snapshot)
    model._attention_map_for_belief(
        snapshot.final_belief,
        model._first_sequence_log_prior(snapshot.log_prior, token_ids.shape[0]),
        snapshot.rope,
    )
    model._gamma_energy(token_ids, snapshot.model_phi, s_belief=snapshot.s_belief)

    assert len(calls) >= 4


def test_head_mixer_builder_failure_is_logged(caplog, monkeypatch):
    model = VFEModel(_tiny_config(use_head_mixer=True))
    tokens = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])

    def fail(*args, **kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(metrics, "head_mixer_gauge_residual", fail)
    with caplog.at_level(logging.WARNING, logger="vfe3.train"):
        result = _val_diagnostics(model, [(tokens, targets)], torch.device("cpu"))

    assert "val_builder_resid" not in result.metrics
    assert "head-mixer builder-residual diagnostic failed" in caplog.text
    assert "probe failed" in caplog.text
