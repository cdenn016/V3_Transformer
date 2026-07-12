r"""Training run artifacts: run dir, config.json, metrics.csv, checkpoints, best_model.pt,
end-of-run TEST eval, summary.json, and figures.

These pin the persistence plumbing the user found missing (training ran but saved nothing).
The proof is files on disk, so the integration tests assert the actual artifacts appear; the
silent path (no artifacts object) must write nothing and stay bitwise-identical (the latter is
covered by tests/test_train.py::test_silent_and_logging_paths_are_bitwise_identical).
"""

import hashlib
import json
import logging
import math
import os
import subprocess
import types
from dataclasses import asdict

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3 import run_artifacts
from vfe3.run_artifacts import (
    RunArtifacts,
    _calibration_and_strata,
    _pure_path_report,
    finalize_run,
    semantic_config_fingerprint,
)
from vfe3.train import build_optimizer, train


def _loader(seed=0, n=600, seq_len=8, bs=8):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)               # period-3 stream over {0,1,2}
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, generator=g)


def _cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return VFE3Config(**base)


def test_config_checkpoint_interval_default_and_validated():
    assert VFE3Config().checkpoint_interval == 25000
    assert VFE3Config(checkpoint_interval=1000).checkpoint_interval == 1000
    with pytest.raises(ValueError):
        VFE3Config(checkpoint_interval=-1)


def test_creates_run_dir_and_config_json(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic", device="cpu")
    assert (tmp_path / "run").is_dir()
    assert (tmp_path / "run" / "checkpoints").is_dir()
    meta = json.loads((tmp_path / "run" / "config.json").read_text())
    assert meta["dataset"] == "synthetic"
    assert meta["n_params"] == sum(p.numel() for p in model.parameters())
    assert meta["config"]["embed_dim"] == 4


def test_log_metrics_writes_csv_with_header(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    art.log_metrics({"step": 1, "val_ppl": 3.0})
    art.log_metrics({"step": 2, "val_ppl": 2.5})
    lines = (tmp_path / "r" / "metrics.csv").read_text().strip().splitlines()
    assert lines[0].split(",") == ["step", "val_ppl"]
    assert len(lines) == 3                                     # header + 2 rows


def test_maybe_save_best_only_on_improvement(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    assert art.maybe_save_best(1, model, 10.0) is True
    assert (tmp_path / "r" / "best_model.pt").exists()
    assert art.maybe_save_best(2, model, 12.0) is False       # worse PPL -> no save
    assert art.maybe_save_best(3, model, 8.0) is True
    assert art.best_val_ppl == 8.0 and art.best_step == 3


def test_best_model_bundle_embeds_semantic_config_fingerprint(tmp_path):
    cfg = _cfg(n_e_steps=3)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)

    assert art.maybe_save_best(1, model, 5.0) is True
    bundle = torch.load(art.best_path, weights_only=True)

    assert set(bundle) == {"model_state", "config", "config_fingerprint"}
    assert bundle["config"] == asdict(cfg)
    expected = hashlib.sha256(json.dumps(
        bundle["config"], sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    assert bundle["config_fingerprint"] == expected
    assert bundle["config_fingerprint"] == semantic_config_fingerprint(bundle["config"])
    reordered = dict(reversed(list(bundle["config"].items())))
    assert semantic_config_fingerprint(reordered) == bundle["config_fingerprint"]


def test_save_checkpoint_is_loadable(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    p = art.save_checkpoint(4, model, opt, cfg)
    assert p.exists()
    ckpt = torch.load(p, weights_only=False)
    assert ckpt["step"] == 4
    assert "model_state" in ckpt and "optimizer_state" in ckpt
    # model-selection state is bundled so a resumed run reports the run-wide best (audit 2026-07-01 C2)
    assert "best_val_ppl" in ckpt and "best_step" in ckpt


def test_writes_are_atomic_no_temp_left(tmp_path):
    # C11 (audit 2026-07-01): every writer publishes via same-dir .tmp + os.replace, so no temp
    # file survives a successful write and every final artifact loads cleanly.
    cfg = _cfg()
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    art.save_json("summary.json", {"a": 1})
    assert art.maybe_save_best(1, model, 5.0) is True
    p = art.save_checkpoint(2, model, opt, cfg)
    assert list((tmp_path / "r").rglob("*.tmp")) == []          # run_dir AND ckpt_dir hold no temps
    assert json.loads((tmp_path / "r" / "summary.json").read_text()) == {"a": 1}
    best = torch.load(tmp_path / "r" / "best_model.pt", weights_only=True)
    assert set(best["model_state"]) == set(model.state_dict())
    ckpt = torch.load(p, weights_only=True)
    assert ckpt["step"] == 2


@pytest.mark.parametrize("name", [
    "", ".", "..", "a/b", r"a\b", "C:evil", "C:/evil", r"C:\evil",
])
def test_save_json_rejects_non_bare_filename(tmp_path, name):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)

    with pytest.raises(ValueError, match="bare filename"):
        art.save_json(name, {"unsafe": True})


def test_save_json_rejects_absolute_filename(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    outside = tmp_path / "outside.json"

    with pytest.raises(ValueError, match="bare filename"):
        art.save_json(str(outside), {"unsafe": True})

    assert not outside.exists()


def test_best_model_overwrite_replaces(tmp_path):
    # C11: os.replace over an EXISTING best_model.pt succeeds (Windows lock retry path aside),
    # and the file reloads to the improved (second) state_dict.
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    assert art.maybe_save_best(1, model, 10.0) is True
    with torch.no_grad():
        model.prior_bank.mu_embed.add_(1.0)                     # make the second save distinguishable
    assert art.maybe_save_best(2, model, 8.0) is True           # improved -> replaces the existing file
    loaded = torch.load(tmp_path / "r" / "best_model.pt", weights_only=True)["model_state"]
    cur = model.state_dict()
    assert all(torch.equal(loaded[k], cur[k]) for k in cur)     # the SECOND state won


# --------------------------------------------------------------------------- pure-path report labels

def _report_cfg(**over):
    r"""SimpleNamespace with every attribute _pure_path_report reads (incl. family for the
    regime_ii_covariant exactness flag)."""
    base = dict(include_attention_entropy=True, transport_mode="flat", lambda_alpha_mode="constant",
                use_prior_bank=True, use_head_mixer=False,
                lambda_beta=1.0, precision_weighted_attention=False,
                gauge_transport="on", pos_rotation="none", rope_full_gauge=False, rope_on_value=True,
                lambda_gamma=0.0, s_e_step=False,
                skip_belief_sigma_update=False, lambda_twohop=0.0,
                gauge_parameterization="phi", omega_reflection="off", phi_reflection="off",
                gauge_group="glk", family="gaussian_diagonal")
    base.update(over)
    return types.SimpleNamespace(**base)


def test_pure_path_report_regime_ii_covariant_exact_flag():
    # C5 (audit 2026-07-01): the diagonal cone is not closed under GL congruence, so a diagonal
    # regime_ii_covariant run is a CONTROLLED APPROXIMATION -- never reported as exact Route B.
    diag = _pure_path_report(_report_cfg(transport_mode="regime_ii_covariant",
                                         family="gaussian_diagonal"), [])
    assert diag["config_toggles"]["regime_ii_covariant_exact"] is False
    full = _pure_path_report(_report_cfg(transport_mode="regime_ii_covariant",
                                         family="gaussian_full"), [])
    assert full["config_toggles"]["regime_ii_covariant_exact"] is True
    flat = _pure_path_report(_report_cfg(), [])
    assert flat["config_toggles"]["regime_ii_covariant_exact"] is True


def test_pure_path_report_transport_covariance_class():
    # C7 (audit 2026-07-01): plain regime_ii's bilinear edge is gauge-FIXED; the report must never
    # group it with the covariant Route B. Every shipped registration owns its exact class.
    expected = {"flat":                   "covariant (flat)",
                "regime_ii":              "gauge-fixed (non-covariant)",
                "regime_ii_covariant":    "covariant",
                "regime_ii_link":         "gauge-fixed",
                "regime_ii_link_charted": "covariant"}
    for mode, label in expected.items():
        rep = _pure_path_report(_report_cfg(transport_mode=mode), [])
        assert rep["config_toggles"]["transport_covariance_class"] == label, mode
    with pytest.raises(KeyError, match="no transport"):
        _pure_path_report(_report_cfg(transport_mode="future_mode"), [])


def test_pure_path_report_reads_transport_registry_metadata():
    from vfe3.geometry import transport

    original = transport.get_transport_registration("flat")
    try:
        transport.register_transport(
            "flat",
            covariance_class="registry-probe",
            needs_mu=original.needs_mu,
            needs_sigma=original.needs_sigma,
            batch_independent=original.batch_independent,
            override=True,
        )(original.callable)

        report = _pure_path_report(_report_cfg(transport_mode="flat"), [])

        assert report["config_toggles"]["transport_covariance_class"] == "registry-probe"
    finally:
        transport.register_transport(
            "flat",
            covariance_class=original.covariance_class,
            needs_mu=original.needs_mu,
            needs_sigma=original.needs_sigma,
            batch_independent=original.batch_independent,
            override=True,
        )(original.callable)


def test_diagonal_gl_route_reports_not_exactly_gauge_invariant():
    diagonal = _pure_path_report(_report_cfg(family="gaussian_diagonal", gauge_group="glk"), [])
    full = _pure_path_report(_report_cfg(family="gaussian_full", gauge_group="glk"), [])

    assert diagonal["gauge_flags"]["family_group_invariant"] is False
    assert diagonal["on_gauge_pure_path"] is False
    assert full["gauge_flags"]["family_group_invariant"] is True
    assert full["on_gauge_pure_path"] is True
    assert diagonal["config_toggles"]["group_invariant_families"] == ["gaussian", "gaussian_full"]


def test_pure_path_queries_builder_metadata_without_constructing_group():
    from vfe3.geometry import groups

    name = "_test_metadata_only_group"
    try:
        @groups.register_group(name, invariant_families=("gaussian_full",))
        def _must_not_build(K):
            raise AssertionError("pure-path reporting must not construct a gauge group")

        report = _pure_path_report(_report_cfg(gauge_group=name, family="gaussian_full"), [])
        assert report["gauge_flags"]["family_group_invariant"] is True
    finally:
        groups._GROUPS.pop(name, None)


def test_group_invariance_metadata_fails_closed_and_override_is_atomic():
    from vfe3.geometry import groups

    name = "_test_invariance_metadata_group"
    try:
        @groups.register_group(name, omega_direct_capable=True)
        def _undeclared(K):
            return groups.GaugeGroup(
                name=name,
                generators=torch.zeros(1, K, K),
                irrep_dims=[K],
                skew_symmetric=False,
                invariant_families=("gaussian_full",),
            )

        undeclared = groups.get_group(name)
        assert undeclared.invariant_families == ()
        assert undeclared(K=2).invariant_families == ()

        @groups.register_group(
            name,
            override=True,
            omega_direct_capable=False,
            invariant_families=("gaussian", "gaussian_full"),
        )
        def _declared(K):
            return groups.GaugeGroup(
                name=name,
                generators=torch.zeros(1, K, K),
                irrep_dims=[K],
                skew_symmetric=False,
            )

        declared = groups.get_group(name)
        built = declared(K=2)
        assert declared.omega_direct_capable is False
        assert declared.invariant_families == ("gaussian", "gaussian_full")
        assert built.omega_direct_capable is False
        assert built.invariant_families == ("gaussian", "gaussian_full")
    finally:
        groups._GROUPS.pop(name, None)


def test_shipped_group_builders_declare_full_gaussian_invariance():
    from vfe3.geometry.groups import get_group

    for name in ("glk", "block_glk", "tied_block_glk", "so_k", "so_n", "sp", "sp_n"):
        assert get_group(name).invariant_families == ("gaussian", "gaussian_full")


def test_train_with_artifacts_writes_files(tmp_path):
    cfg = _cfg(checkpoint_interval=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    train(model, _loader(), cfg, n_steps=4, eval_interval=2, val_loader=_loader(seed=1), artifacts=art)
    assert (tmp_path / "run" / "metrics.csv").exists()
    assert (tmp_path / "run" / "best_model.pt").exists()
    assert any((tmp_path / "run" / "checkpoints").glob("step_*.pt"))


def test_train_with_artifacts_writes_attention_pngs(tmp_path):
    # Per eval, one LOG-scaled attention/step_<N>_layer<l>_head<h>.png per (layer, head).
    cfg = _cfg(n_layers=2, prior_handoff_rho=0.5)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    train(model, _loader(), cfg, n_steps=4, eval_interval=2, val_loader=_loader(seed=1), artifacts=art)
    L, H = cfg.n_layers, len(model.group.irrep_dims)
    expected = sorted(f"step_{s}_layer{l}_head{h}.png"
                      for s in (2, 4) for l in range(L) for h in range(H))
    pngs = sorted((tmp_path / "run" / "attention").glob("step_*.png"))
    assert [p.name for p in pngs] == expected
    assert all(p.stat().st_size > 0 for p in pngs)


def test_save_attention_maps_is_best_effort(tmp_path):
    # A viz/plotting error must be swallowed (logged, never raised) so it cannot kill a run.
    cfg = _cfg()
    art = RunArtifacts(tmp_path / "run", cfg, VFEModel(cfg))
    assert art.save_attention_maps(1, object()) is None         # bad maps -> None, no exception


@pytest.mark.parametrize(("writer", "maps"), [
    ("save_attention_maps", torch.ones(1, 1, 2, 2)),
    ("save_gamma_attention_maps", torch.ones(1, 2, 2)),
])
def test_attention_writer_closes_only_new_figures_on_plot_failure(
    tmp_path,
    monkeypatch,
    writer,
    maps,
):
    from vfe3.viz import figures as figs

    cfg = _cfg()
    art = RunArtifacts(tmp_path / "run", cfg, VFEModel(cfg))
    existing = figs.plt.figure()
    created = []

    def _raise_after_creating_figure(*args, **kwargs):
        fig = figs.plt.figure()
        created.append(fig.number)
        raise RuntimeError("plot failed after allocating a figure")

    monkeypatch.setattr(figs, "plot_attention_heatmap", _raise_after_creating_figure)
    try:
        assert getattr(art, writer)(1, maps) is None
        open_figures = set(figs.plt.get_fignums())
        assert existing.number in open_figures
        assert open_figures.isdisjoint(created)
    finally:
        figs.plt.close(existing)
        for number in created:
            figs.plt.close(number)


def test_train_without_artifacts_writes_nothing(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    train(model, _loader(), cfg, n_steps=4, eval_interval=2, val_loader=_loader(seed=1))
    assert list(tmp_path.iterdir()) == []                     # no artifacts object -> no writes


def test_finalize_run_writes_test_results_and_figures(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    losses = train(model, _loader(), cfg, n_steps=4, eval_interval=2,
                   val_loader=_loader(seed=1), artifacts=art)
    res = finalize_run(model, art, cfg, test_loader=_loader(seed=2), losses=losses)
    assert "test_ppl" in res and math.isfinite(res["test_ppl"])
    assert (tmp_path / "run" / "test_results.json").exists()
    assert (tmp_path / "run" / "summary.json").exists()
    assert (tmp_path / "run" / "loss_curve.png").exists()
    assert (tmp_path / "run" / "val_ppl.png").exists()
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert "test_ppl" in summary and "best_val_ppl" in summary
    assert "reloaded_best" in summary   # m26: surface whether best_model.pt was reloaded (cross-dir resume honesty)


def test_frequency_strata_use_training_corpus_counts():
    class _FixedLogits(torch.nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, _tokens: torch.Tensor) -> torch.Tensor:
            return self.logits

    targets = torch.tensor([[0, 0, 1, 1, 2, 2]])
    tokens = torch.zeros_like(targets)
    logits = torch.tensor([[[4.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                            [0.0, 3.0, 0.0], [0.0, 1.0, 0.0],
                            [0.0, 0.0, 2.0], [0.0, 0.0, 0.5]]])
    corpus_counts = torch.tensor([1, 10, 100])

    out = _calibration_and_strata(
        corpus_counts,
        _FixedLogits(logits),
        [(tokens, targets)],
        torch.device("cpu"),
    )

    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    strata = out["corpus_freq_strata_ce"]
    assert strata["rare"] == pytest.approx(float(ce[:2].mean()))
    assert strata["mid"] == pytest.approx(float(ce[2:4].mean()))
    assert strata["frequent"] == pytest.approx(float(ce[4:].mean()))
    assert "freq_strata_ce" not in out


def test_frequency_strata_cutoffs_ignore_imbalanced_evaluation_duplicates():
    class _FixedLogits(torch.nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, _tokens: torch.Tensor) -> torch.Tensor:
            return self.logits

    targets = torch.tensor([[1, 1, 1, 1, 1, 3, 4, 0]])
    tokens = torch.zeros_like(targets)
    logits = torch.tensor([[
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.5, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 2.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 3.0],
        [0.25, 0.0, 0.0, 0.0, 0.0],
    ]])
    corpus_counts = torch.tensor([0, 1, 10, 100, 1000])

    out = _calibration_and_strata(
        corpus_counts,
        _FixedLogits(logits),
        [(tokens, targets)],
        torch.device("cpu"),
    )

    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    strata = out["corpus_freq_strata_ce"]
    expected_rare = torch.cat((ce[:5], ce[7:])).mean()
    assert strata["rare"] == pytest.approx(float(expected_rare))
    assert strata["mid"] == pytest.approx(float(ce[5]))
    assert strata["frequent"] == pytest.approx(float(ce[6]))


def test_provenance_records_all_split_hashes_and_data_knobs(tmp_path):
    cfg = _cfg(generate_figures=False)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    train_loader = _loader(seed=1, n=300)
    val_loader = _loader(seed=2, n=360)
    test_loader = _loader(seed=3, n=420)

    finalize_run(
        model,
        art,
        cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        data_seed=17,
        max_tokens=300,
        tokenizer_tag="synthetic-v1",
    )

    prov = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        tokens = loader.dataset.tokens
        expected = hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()
        assert prov[f"{split}_data_sha256"] == expected
        assert prov[f"{split}_data_n_tokens"] == int(tokens.numel())
    assert prov["data_seed"] == 17
    assert prov["max_tokens"] == 300
    assert prov["tokenizer_tag"] == "synthetic-v1"
    assert prov["data_sha256"] == prov["test_data_sha256"]
    assert prov["data_n_tokens"] == prov["test_data_n_tokens"]


def test_provenance_git_probe_timeout_records_error(tmp_path, monkeypatch):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    timeout = subprocess.TimeoutExpired(["git", "rev-parse", "HEAD"], 5)
    def _which(name):
        assert name == "git"
        return "C:/trusted/git.exe"

    monkeypatch.setattr(run_artifacts.shutil, "which", _which)

    def _time_out(*args, **kwargs):
        assert kwargs["timeout"] == 5
        assert kwargs["env"] is not os.environ
        assert kwargs["env"]["GIT_CONFIG_NOSYSTEM"] == "1"
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        raise timeout

    monkeypatch.setattr(run_artifacts.subprocess, "check_output", _time_out)
    run_artifacts._write_provenance(
        art,
        cfg,
        model,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        data_seed=None,
        max_tokens=None,
        tokenizer_tag=None,
        logger=logging.getLogger("test-provenance"),
    )

    prov = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["git_sha"] is None
    assert prov["git_dirty"] is None
    assert prov["git_dirty_fingerprint"] is None
    assert prov["git_error"] == repr(timeout)


def test_metrics_csv_includes_gauge_geometry_columns(tmp_path):
    # Part 1 (diagnostics tier): the curvature/gauge probes (holonomy deviation + gauge trace
    # spread) must be surfaced in the per-eval CSV, not only the free-energy terms.
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    train(model, _loader(), cfg, n_steps=4, eval_interval=2, val_loader=_loader(seed=1), artifacts=art)
    header = (tmp_path / "run" / "metrics.csv").read_text().splitlines()[0]
    assert "holonomy_deviation" in header
    assert "gauge_trace_spread" in header


def test_finalize_writes_gauge_geometry_figure(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    losses = train(model, _loader(), cfg, n_steps=4, eval_interval=2,
                   val_loader=_loader(seed=1), artifacts=art)
    finalize_run(model, art, cfg, test_loader=_loader(seed=2), losses=losses)
    assert (tmp_path / "run" / "holonomy.png").exists()


def test_finalize_reloads_best_checkpoint(tmp_path):
    # finalize must report the TEST metric on the reloaded best-val checkpoint, not the final
    # (possibly worse) live weights. Pin the reload happened.
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    losses = train(model, _loader(), cfg, n_steps=4, eval_interval=2,
                   val_loader=_loader(seed=1), artifacts=art)
    res = finalize_run(model, art, cfg, test_loader=_loader(seed=2), losses=losses)
    assert res["reloaded_best"] is True


def test_fd_gradient_check_restores_param_on_midloop_failure():
    """Audit 2026-07-12 N1: the FD probe perturbs a LIVE decode parameter through a
    storage-sharing view; a forward that raises between the +eps write and the restore
    previously left the parameter perturbed for every subsequent probe (the caller catches
    broadly). The perturbation loop must restore the coordinate on the way out."""
    cfg = _cfg(generate_figures=False)
    model = VFEModel(cfg)
    loader = _loader(seed=1, n=120)
    pb = model.prior_bank
    p = (pb.output_proj_weight
         if getattr(pb, "output_proj_weight", None) is not None else pb.decode_log_scale)
    before = p.detach().clone()

    calls = {"n": 0}
    real_forward = model.forward

    def _failing_forward(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:            # first PERTURBED eval: after flat[j] = orig + fd_eps
            raise RuntimeError("injected mid-probe failure")
        return real_forward(*args, **kwargs)

    model.forward = _failing_forward
    with pytest.raises(RuntimeError, match="injected mid-probe"):
        run_artifacts._fd_gradient_check(model, loader, torch.device("cpu"))
    assert calls["n"] >= 2, "probe never reached the perturbation loop"
    assert torch.equal(p.detach(), before), "decode parameter left perturbed by +/-fd_eps"


def test_provenance_hash_failure_warns_not_silent(tmp_path, caplog):
    """Audit 2026-07-12 N2: a failure while hashing a split's corpus previously hit a bare
    ``except Exception: pass`` -- provenance.json silently recorded null data hashes. The
    failure must be logged (the keys stay best-effort None so finalize never crashes)."""

    class _ExplodingTokens:
        def detach(self):
            raise RuntimeError("corrupt token stream")

    train_loader = types.SimpleNamespace(
        dataset=types.SimpleNamespace(tokens=_ExplodingTokens()))

    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    with caplog.at_level(logging.WARNING, logger="test-provenance-n2"):
        run_artifacts._write_provenance(
            art, cfg, model,
            train_loader=train_loader, val_loader=None, test_loader=None,
            data_seed=None, max_tokens=None, tokenizer_tag=None,
            logger=logging.getLogger("test-provenance-n2"),
        )
    prov = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["train_data_sha256"] is None                  # best-effort null, not a crash
    assert any("corrupt token stream" in rec.getMessage() and "train" in rec.getMessage()
               for rec in caplog.records), "hash failure was swallowed silently"
