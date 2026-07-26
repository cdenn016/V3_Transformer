r"""``parameter_motion_report``: did every trainable parameter ACTUALLY move over the run?

The existing ``parameter_report`` is a CAPABILITY check -- one synthetic backward at init, flagging
``p.grad is None``. It catches a table severed from the loss (measured: under R-1's
``rope + rope_on_value=False + oracle_unroll_grad=False``, ``phi_embed`` lands in ``dead_names``).
It cannot catch a table that receives a gradient and still never moves. The blind spot pinned here
is a zero group learning rate: with ``m_phi_lr=0.0`` the init probe reports ``phi_embed`` live while
it moves exactly ``0.0`` across real optimizer steps.

Every model here is TINY (K=4, single-digit everything) and CPU-bound per CLAUDE.md.
"""

import json
import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import (
    build_optimizer,
    parameter_motion_report,
    parameter_report,
    snapshot_parameters,
)

BASE = dict(embed_dim=4, n_heads=2, vocab_size=11, max_seq_len=8, max_steps=1,
            n_layers=1, n_e_steps=2, gauge_group="block_glk")


def _model(**overrides) -> VFEModel:
    torch.manual_seed(0)
    return VFEModel(VFE3Config(**BASE, **overrides))


def _train_a_few_steps(model: VFEModel, cfg: VFE3Config, steps: int = 3) -> None:
    optimizer = build_optimizer(model, cfg)
    generator = torch.Generator().manual_seed(1)
    for _ in range(steps):
        ids = torch.randint(0, cfg.vocab_size, (2, 6), generator=generator)
        tgt = torch.randint(0, cfg.vocab_size, (2, 6), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        out = model(ids, tgt)
        (out[1] if isinstance(out, tuple) else out).backward()
        optimizer.step()


def test_a_healthy_run_reports_no_frozen_parameters() -> None:
    r"""Nothing is flagged when every live table moves.

    ``known_dead`` is threaded exactly as ``train`` threads it, because the repo default is
    ``use_prior_bank=False``, under which ``decode_log_scale`` is config-dead and correctly immobile.
    """
    model = _model()
    known_dead = parameter_report(model, batch=2, seqlen=6)["dead_names"]
    snapshot = snapshot_parameters(model)
    _train_a_few_steps(model, model.cfg)

    report = parameter_motion_report(model, snapshot, known_dead=known_dead)

    assert report["frozen"] == [], report["frozen"]
    assert report["missing"] == []
    assert report["n_parameters"] > 0
    assert any(m["rel"] and m["rel"] > 0 for m in report["motion"].values())


def test_a_config_dead_table_is_reported_but_not_warned_about() -> None:
    r"""``decode_log_scale`` under the default ``use_prior_bank=False`` is dead by construction.

    It must still be RECORDED as immobile, but routed to ``frozen_known_dead`` so the end-of-run
    warning does not fire on every run of an intentional ablation -- a guard that always fires is a
    guard that gets tuned out, which is how the R-1 freeze went unread for 15000 steps.
    """
    model = _model()
    assert model.cfg.use_prior_bank is False, "precondition: the repo default is the linear decode"
    known_dead = parameter_report(model, batch=2, seqlen=6)["dead_names"]
    assert any("decode_log_scale" in name for name in known_dead)

    snapshot = snapshot_parameters(model)
    _train_a_few_steps(model, model.cfg)
    report = parameter_motion_report(model, snapshot, known_dead=known_dead)

    name = next(n for n in report["motion"] if "decode_log_scale" in n)
    assert report["motion"][name]["frozen"] is True          # measured immobile ...
    assert report["motion"][name]["known_dead"] is True
    assert name in report["frozen_known_dead"]               # ... and explained, so not news
    assert name not in report["frozen"]


def test_the_novel_signal_survives_the_known_dead_split() -> None:
    r"""A zero-LR freeze is NOT explained by the init probe, so it must stay in ``frozen``."""
    model = _model(m_phi_lr=0.0)
    known_dead = parameter_report(model, batch=2, seqlen=6)["dead_names"]
    snapshot = snapshot_parameters(model)
    _train_a_few_steps(model, model.cfg)

    report = parameter_motion_report(model, snapshot, known_dead=known_dead)

    assert any("phi_embed" in name for name in report["frozen"]), report["frozen"]


def test_zero_learning_rate_freezes_a_parameter_the_init_grad_probe_calls_live() -> None:
    r"""The blind spot this report exists to cover: live at init, immobile for the whole run."""
    model = _model(m_phi_lr=0.0)

    capability = parameter_report(model, batch=2, seqlen=6)
    assert not any("phi_embed" in name for name in capability["dead_names"]), (
        "precondition: the init probe must consider phi_embed live under m_phi_lr=0.0"
    )

    snapshot = snapshot_parameters(model)
    _train_a_few_steps(model, model.cfg)
    report = parameter_motion_report(model, snapshot)

    assert any("phi_embed" in name for name in report["frozen"]), report["frozen"]
    frozen_entry = next(m for name, m in report["motion"].items() if "phi_embed" in name)
    assert frozen_entry["abs"] == 0.0
    assert frozen_entry["frozen"] is True


def test_a_deliberately_frozen_parameter_is_not_reported() -> None:
    r"""``requires_grad=False`` is a decision, not a defect: the frozen hyper-prior centroid r under
    ``learnable_r=False`` must never appear in ``frozen``."""
    model = _model(lambda_h=0.1, learnable_r=False)
    fixed = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert fixed, "precondition: this config must hold at least one requires_grad=False parameter"

    snapshot = snapshot_parameters(model)
    _train_a_few_steps(model, model.cfg)
    report = parameter_motion_report(model, snapshot)

    assert not (set(fixed) & set(report["frozen"]))
    for name in fixed:
        assert report["motion"][name]["frozen"] is True         # it did not move ...
        assert report["motion"][name]["requires_grad"] is False  # ... and that is correct


def test_a_zero_init_table_is_judged_on_absolute_motion() -> None:
    r"""``||p_0|| = 0`` makes relative motion undefined, so any nonzero motion counts as trained."""
    model = _model()
    snapshot = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    report = parameter_motion_report(model, snapshot)

    for name, entry in report["motion"].items():
        assert entry["rel"] is None, name
        assert entry["init_norm"] == 0.0
        assert entry["frozen"] is (entry["abs"] == 0.0)


def test_motion_is_measured_relative_to_the_parameter_norm() -> None:
    model = _model()
    snapshot = snapshot_parameters(model)
    with torch.no_grad():                                        # a known 1% nudge on one table
        model.prior_bank.phi_embed.add_(0.01 * model.prior_bank.phi_embed)

    report = parameter_motion_report(model, snapshot)
    entry = next(m for name, m in report["motion"].items() if name.endswith("phi_embed"))

    assert entry["rel"] == pytest.approx(0.01, rel=1e-4)
    assert entry["frozen"] is False


def test_rel_tol_sets_the_frozen_threshold() -> None:
    model = _model()
    snapshot = snapshot_parameters(model)
    with torch.no_grad():
        model.prior_bank.phi_embed.add_(1e-5 * model.prior_bank.phi_embed)

    name = next(n for n, _ in model.named_parameters() if n.endswith("phi_embed"))
    assert name not in parameter_motion_report(model, snapshot, rel_tol=1e-6)["frozen"]
    assert name in parameter_motion_report(model, snapshot, rel_tol=1e-3)["frozen"]


def test_a_parameter_absent_from_the_snapshot_is_reported_missing_not_frozen() -> None:
    model = _model()
    snapshot = snapshot_parameters(model)
    dropped = next(iter(snapshot))
    del snapshot[dropped]

    report = parameter_motion_report(model, snapshot)

    assert report["missing"] == [dropped]
    assert dropped not in report["motion"]
    assert dropped not in report["frozen"]
    assert dropped not in report["frozen_known_dead"]


def test_snapshot_is_detached_on_the_host_and_does_not_alias_the_model() -> None:
    model = _model()
    snapshot = snapshot_parameters(model)
    with torch.no_grad():
        model.prior_bank.phi_embed.add_(1.0)                     # must not write through to the snapshot

    name = next(n for n, _ in model.named_parameters() if n.endswith("phi_embed"))
    assert snapshot[name].device.type == "cpu"
    assert not snapshot[name].requires_grad
    assert parameter_motion_report(model, snapshot)["motion"][name]["abs"] > 0.0


def _loader(vocab: int = 11, seq_len: int = 6, batch_size: int = 2, n: int = 96):
    from torch.utils.data import DataLoader

    from vfe3.data.datasets import TokenWindows

    tokens = torch.arange(n).remainder(vocab).to(torch.long)     # deterministic period-vocab stream
    return DataLoader(TokenWindows(tokens, seq_len), batch_size=batch_size, drop_last=True)


def test_train_warns_by_name_when_a_live_parameter_never_moves() -> None:
    r"""End-to-end: the guard has to fire from ``train`` itself, not just from the report helper."""
    from vfe3.train import train

    cfg = VFE3Config(**{**BASE, "max_steps": 3}, m_phi_lr=0.0)
    torch.manual_seed(0)
    model = VFEModel(cfg)

    with pytest.warns(UserWarning, match=r"parameter motion:.*phi_embed"):
        train(model, _loader(), cfg, n_steps=3)


def test_train_is_silent_when_every_live_parameter_moves() -> None:
    from vfe3.train import train

    cfg = VFE3Config(**{**BASE, "max_steps": 3})
    torch.manual_seed(0)
    model = VFEModel(cfg)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train(model, _loader(), cfg, n_steps=3)

    assert not [w for w in caught if "parameter motion" in str(w.message)]


def test_train_writes_the_motion_report_to_the_run_artifacts(tmp_path) -> None:
    from vfe3.train import train

    cfg = VFE3Config(**{**BASE, "max_steps": 3}, m_phi_lr=0.0)
    torch.manual_seed(0)
    model = VFEModel(cfg)

    written = {}

    class _Artifacts:
        def save_json(self, name, obj):
            written[name] = obj
            return tmp_path / name

        def log_metrics(self, row):
            pass

    with pytest.warns(UserWarning, match="parameter motion"):
        train(model, _loader(), cfg, n_steps=3, artifacts=_Artifacts())

    report = written["parameter_motion.json"]
    assert report["steps"] == 3
    assert any("phi_embed" in name for name in report["frozen"])
    assert report["rel_tol"] == cfg.parameter_motion_rel_tol
    assert json.loads(json.dumps(report))                        # the payload must be JSON-writable


def test_the_check_can_be_turned_off() -> None:
    from vfe3.train import train

    cfg = VFE3Config(**{**BASE, "max_steps": 3}, m_phi_lr=0.0, check_parameter_motion=False)
    torch.manual_seed(0)
    model = VFEModel(cfg)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train(model, _loader(), cfg, n_steps=3)

    assert not [w for w in caught if "parameter motion" in str(w.message)]


def test_config_exposes_the_toggle_and_validates_the_tolerance() -> None:
    cfg = VFE3Config(**BASE)
    assert cfg.check_parameter_motion is True
    assert cfg.parameter_motion_rel_tol == 1e-6

    with pytest.raises(ValueError, match="parameter_motion_rel_tol"):
        VFE3Config(**BASE, parameter_motion_rel_tol=-1.0)
