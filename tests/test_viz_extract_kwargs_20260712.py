r"""Audit 2026-07-12 N5: the viz extractors must thread the SAME cfg-derived E-step knob bag
as production (``vfe_block`` -> ``e_step``), so opt-in iteration knobs (``e_step_update``,
``mm_damping``, ``lambda_twohop``, ``skip_belief_sigma_update``, ...) cannot silently diverge
between the trained model and its persisted diagnostics.

The committed baselines set three of the four previously-dropped knobs off-default
(``e_step_update='mm_exact'``, ``mm_damping=0.75``, ``skip_belief_sigma_update=True``), so the
divergence was live on the committed workload, not merely under opt-in toggles.
"""

import inspect

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.inference.e_step import e_step_iteration, free_energy_value
from vfe3.model.block import e_step_shared_kwargs
from vfe3.model.model import VFEModel
from vfe3.viz.extract import _fe_kwargs, _iter_kwargs, e_step_belief_trace

_TOKEN_IDS = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)


def _tiny_offdefault_cfg() -> VFE3Config:
    r"""Tiny (K=4) config with every previously-dropped E-step knob set OFF-default."""
    return VFE3Config(
        vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1, n_e_steps=1,
        e_phi_lr=0.0,
        e_step_update="mm_exact", mm_damping=0.75,
        skip_belief_sigma_update=True, lambda_twohop=0.05,
    )


def _build_model(cfg: VFE3Config) -> VFEModel:
    torch.manual_seed(13)
    return VFEModel(cfg).eval()


def test_iter_kwargs_threads_offdefault_estep_knobs() -> None:
    r"""_iter_kwargs must carry the four audit-N5 knobs from model.cfg (previously silently
    dropped, so e_step_iteration fell back to the pure defaults regardless of the config)."""
    model = _build_model(_tiny_offdefault_cfg())
    ikw = _iter_kwargs(model, log_prior=None, rope=None)
    assert ikw["e_step_update"] == "mm_exact"
    assert ikw["mm_damping"] == 0.75
    assert ikw["skip_belief_sigma_update"] is True
    assert ikw["lambda_twohop"] == 0.05


def test_fe_kwargs_threads_lambda_twohop() -> None:
    r"""free_energy_value HONORS lambda_twohop (e_step.py adds the detached two-hop block), so
    the logged F trajectory must carry it."""
    model = _build_model(_tiny_offdefault_cfg())
    fkw = _fe_kwargs(model, log_prior=None)
    assert fkw["lambda_twohop"] == 0.05
    assert fkw["e_step_update"] == "mm_exact"          # accepted-and-ignored, rides the shared bag


def test_configured_chart_bound_reaches_direct_extractor_evaluators() -> None:
    cfg = VFE3Config(
        vocab_size=9,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        pos_phi="none",
        transport_chart_max_norm=0.1,
    )
    model = _build_model(cfg)
    with torch.no_grad():
        model.prior_bank.phi_embed.fill_(1.0)

    with pytest.raises(ValueError, match="transport chart validity bound"):
        e_step_belief_trace(model, _TOKEN_IDS)


def test_shared_bag_accepted_by_both_consumers() -> None:
    r"""Drift guard: every key of the shared cfg-derived bag must be a declared parameter of
    BOTH e_step_iteration and free_energy_value (the production ``e_step`` forwards one knob bag
    to both; a new knob added to the mapping without declaring it downstream must fail here,
    not silently)."""
    cfg = _tiny_offdefault_cfg()
    shared = e_step_shared_kwargs(cfg, torch.device("cpu"))
    iter_params = set(inspect.signature(e_step_iteration).parameters)
    fe_params = set(inspect.signature(free_energy_value).parameters)
    missing_iter = set(shared) - iter_params
    missing_fe = set(shared) - fe_params
    assert not missing_iter, f"shared-bag keys unknown to e_step_iteration: {sorted(missing_iter)}"
    assert not missing_fe, f"shared-bag keys unknown to free_energy_value: {sorted(missing_fe)}"


def test_extractor_bags_are_supersets_of_shared_bag() -> None:
    r"""_iter_kwargs and _fe_kwargs must contain the shared bag verbatim (the runtime extras --
    tau, log_prior, rope, connections -- ride on top)."""
    model = _build_model(_tiny_offdefault_cfg())
    shared = e_step_shared_kwargs(model.cfg, torch.device("cpu"))
    ikw = _iter_kwargs(model, log_prior=None, rope=None)
    fkw = _fe_kwargs(model, log_prior=None)
    for key, value in shared.items():
        assert key in ikw and ikw[key] == value, f"_iter_kwargs diverges on {key!r}"
        assert key in fkw and fkw[key] == value, f"_fe_kwargs diverges on {key!r}"


def test_production_e_step_receives_shared_bag(monkeypatch) -> None:
    r"""vfe_block must feed production ``e_step`` the SAME shared bag (single source of truth):
    a hand-rolled production bag that drops a knob would desynchronize from the extractors
    again."""
    from vfe3.model import block as block_module

    captured: dict = {}
    real_e_step = block_module.e_step

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return real_e_step(*args, **kwargs)

    monkeypatch.setattr(block_module, "e_step", _capture)
    model = _build_model(_tiny_offdefault_cfg())
    with torch.no_grad():
        model(_TOKEN_IDS)
    shared = e_step_shared_kwargs(model.cfg, torch.device("cpu"))
    for key, value in shared.items():
        assert key in captured and captured[key] == value, (
            f"production e_step bag diverges from the shared mapping on {key!r}")
