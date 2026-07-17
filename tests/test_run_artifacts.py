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
from dataclasses import asdict, dataclass

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
from vfe3.viz import figures as figs


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


@dataclass(frozen=True)
class TrainedArtifactEvidence:
    relative_files:  frozenset[str]
    checkpoint_names: tuple[str, ...]
    metrics_columns: frozenset[str]


@pytest.fixture(scope="module")
def trained_artifact_evidence(tmp_path_factory) -> TrainedArtifactEvidence:
    cpu_rng_state   = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    open_figures    = frozenset(figs.plt.get_fignums())
    cfg = model = artifacts = train_loader = val_loader = run_dir = None
    evidence = None
    try:
        run_dir     = tmp_path_factory.mktemp("trained-artifact-evidence") / "run"
        cfg         = _cfg(checkpoint_interval=2)
        train_loader = _loader()
        val_loader   = _loader(seed=1)
        torch.manual_seed(0)
        model     = VFEModel(cfg)
        artifacts = RunArtifacts(run_dir, cfg, model, dataset="synthetic")
        train(
            model,
            train_loader,
            cfg,
            n_steps=4,
            eval_interval=2,
            val_loader=val_loader,
            artifacts=artifacts,
        )
        relative_files = frozenset(
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*")
            if path.is_file()
        )
        checkpoint_names = tuple(sorted(
            path.name for path in (run_dir / "checkpoints").glob("step_*.pt")
        ))
        metrics_columns = frozenset(
            (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
        )
        evidence = TrainedArtifactEvidence(
            relative_files=relative_files,
            checkpoint_names=checkpoint_names,
            metrics_columns=metrics_columns,
        )
    finally:
        for figure_number in set(figs.plt.get_fignums()).difference(open_figures):
            figs.plt.close(figure_number)
        cfg = model = artifacts = train_loader = val_loader = run_dir = None
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        cpu_rng_state = cuda_rng_states = None
    assert evidence is not None
    return evidence


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


@pytest.mark.parametrize("grad_clip", [None, 0.0, 0.25])
def test_config_json_persists_grad_clip(tmp_path, grad_clip):
    cfg = _cfg(grad_clip=grad_clip)
    model = VFEModel(cfg)
    RunArtifacts(tmp_path / "run", cfg, model)
    meta = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    assert meta["config"]["grad_clip"] == grad_clip


def test_log_metrics_writes_csv_with_header(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    art.log_metrics({"step": 1, "val_ppl": 3.0})
    art.log_metrics({"step": 2, "val_ppl": 2.5})
    lines = (tmp_path / "r" / "metrics.csv").read_text().strip().splitlines()
    assert lines[0].split(",") == ["step", "val_ppl"]
    assert len(lines) == 3                                     # header + 2 rows


def test_history_figures_run_in_disposable_process_without_mutating_parent_openmp_env(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    cfg = _cfg(generate_figures=False)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "isolated-figures", cfg, model, dataset="synthetic")
    for step, train_loss, val_ce in ((1, 1.4, 1.3), (2, 1.2, 1.1), (3, 1.0, 0.9)):
        art.log_metrics({
            "step":              step,
            "train_loss":        train_loss,
            "val_ce":            val_ce,
            "val_ppl":           math.exp(val_ce),
            "self_coupling":     0.20,
            "belief_coupling":   0.15,
            "attention_entropy": 0.05,
        })

    completed = run_artifacts._run_figures_isolated(
        model,
        art,
        [1.4, 1.2, 1.0],
        logging.getLogger("test-isolated-figures"),
        generate_publication=False,
    )

    assert completed is True
    assert "KMP_DUPLICATE_LIB_OK" not in os.environ
    assert (art.run_dir / "loss_curve.png").is_file()
    assert (art.run_dir / "free_energy_codescent.png").is_file()


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

    assert set(bundle) == {
        "model_state", "config", "config_fingerprint", "code_identity_sha256",
        "selection_data_identity",
    }
    assert isinstance(bundle["code_identity_sha256"], str)
    assert len(bundle["code_identity_sha256"]) == 64
    assert bundle["selection_data_identity"] is None
    assert bundle["config"] == asdict(cfg)
    assert bundle["config"]["grad_clip"] == cfg.grad_clip
    expected = hashlib.sha256(json.dumps(
        bundle["config"], sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    assert bundle["config_fingerprint"] == expected
    assert bundle["config_fingerprint"] == semantic_config_fingerprint(bundle["config"])
    reordered = dict(reversed(list(bundle["config"].items())))
    assert semantic_config_fingerprint(reordered) == bundle["config_fingerprint"]


def test_best_model_bundle_grad_clip_changes_fingerprint(tmp_path):
    # PB-15: grad_clip is part of the persisted config, so a bundle saved under a different
    # grad_clip carries both the value AND a distinct semantic_config_fingerprint.
    cfg_a = _cfg(grad_clip=1.0)
    cfg_b = _cfg(grad_clip=0.25)
    model_a, model_b = VFEModel(cfg_a), VFEModel(cfg_b)
    art_a = RunArtifacts(tmp_path / "a", cfg_a, model_a)
    art_b = RunArtifacts(tmp_path / "b", cfg_b, model_b)

    assert art_a.maybe_save_best(1, model_a, 5.0) is True
    assert art_b.maybe_save_best(1, model_b, 5.0) is True
    bundle_a = torch.load(art_a.best_path, weights_only=True)
    bundle_b = torch.load(art_b.best_path, weights_only=True)

    assert bundle_a["config"]["grad_clip"] == 1.0
    assert bundle_b["config"]["grad_clip"] == 0.25
    assert bundle_a["config_fingerprint"] != bundle_b["config_fingerprint"]


def test_save_checkpoint_is_loadable(tmp_path):
    cfg = _cfg(grad_clip=0.25)
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
    assert ckpt["config"]["grad_clip"] == 0.25


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


def test_train_with_artifacts_writes_files(trained_artifact_evidence):
    assert "metrics.csv" in trained_artifact_evidence.relative_files
    assert "best_model.pt" in trained_artifact_evidence.relative_files
    assert any(
        name.startswith("step_") and name.endswith(".pt")
        for name in trained_artifact_evidence.checkpoint_names
    )


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


def test_provenance_data_hash_is_dtype_independent(tmp_path):
    # PB-08 follow-up: TokenWindows now stores tokens in their native cache dtype, so the SAME
    # corpus content can sit in memory as int32 (uncapped .bin memmap) or int64 (capped load).
    # The provenance hash is a CONTENT identity pooled by scaling_analysis.py's mixed_corpus
    # gate; a storage-dtype-only divergence would false-positive that confound check.
    cfg = _cfg(generate_figures=False)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    content = torch.arange(3).repeat(100)                     # (300,) identical token content
    loader32 = DataLoader(TokenWindows(content.to(torch.int32), 8), batch_size=8)
    loader64 = DataLoader(TokenWindows(content.to(torch.int64), 8), batch_size=8)

    finalize_run(model, art, cfg,
                 train_loader=loader32, val_loader=loader64, test_loader=loader64)

    prov = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    expected = hashlib.sha256(content.to(torch.int64).numpy().tobytes()).hexdigest()
    assert prov["train_data_sha256"] == expected              # int32 storage hashes as content
    assert prov["val_data_sha256"] == expected                # int64 storage: unchanged identity
    assert prov["train_data_sha256"] == prov["val_data_sha256"]
    assert prov["train_data_n_tokens"] == prov["val_data_n_tokens"] == 300


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


def test_metrics_csv_includes_gauge_geometry_columns(trained_artifact_evidence):
    # Part 1 (diagnostics tier): the curvature/gauge probes (holonomy deviation + gauge trace
    # spread) must be surfaced in the per-eval CSV, not only the free-energy terms.
    assert "holonomy_deviation" in trained_artifact_evidence.metrics_columns
    assert "gauge_trace_spread" in trained_artifact_evidence.metrics_columns


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


# ======================================================================================
# PB-06: model_behavior_fingerprint + sigma_behavior_config (non-policy behavior projection).
# ======================================================================================
from vfe3.run_artifacts import model_behavior_fingerprint, sigma_behavior_config


def _sigma_model(seed=0, **kw):
    d = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
             n_e_steps=1, e_phi_lr=0.0, family="gaussian_diagonal", seed=seed)
    d.update(kw)
    torch.manual_seed(seed)
    return VFEModel(VFE3Config(**d))


def test_sigma_behavior_config_drops_every_policy_field():
    cfg = VFE3Config(policy_mode="none")
    proj = sigma_behavior_config(cfg)
    assert not any(k.startswith("policy_") for k in proj)
    assert "decode_tau" in proj and "family" in proj and "divergence_family" in proj


def test_sigma_behavior_config_invariant_to_policy_fields():
    # A checkpoint config (policy_mode='none') and a consumer config (efe_rollout + different
    # preference/score/top-k/horizon/gate fields) share the SAME behavior projection.
    ckpt = VFE3Config(policy_mode="none")
    consumer = VFE3Config(policy_mode="efe_rollout", policy_preference="flat", policy_horizon=2,
                          policy_top_k=4, policy_score_terms=("risk",), policy_precision=2.0)
    assert sigma_behavior_config(ckpt) == sigma_behavior_config(consumer)


def test_sigma_behavior_config_accepts_mapping_and_matches_dataclass():
    cfg = VFE3Config()
    assert sigma_behavior_config(cfg) == sigma_behavior_config(asdict(cfg))


def test_model_behavior_fingerprint_is_key_order_invariant():
    m = _sigma_model()
    sd = m.state_dict()
    reordered = dict(reversed(list(sd.items())))
    cfg_proj = sigma_behavior_config(m.cfg)
    assert model_behavior_fingerprint(cfg_proj, sd) == model_behavior_fingerprint(cfg_proj, reordered)


def test_model_behavior_fingerprint_sensitive_to_one_changed_value():
    m = _sigma_model()
    sd = m.state_dict()
    cfg_proj = sigma_behavior_config(m.cfg)
    base = model_behavior_fingerprint(cfg_proj, sd)
    key = next(iter(sd))
    perturbed = {k: (v.clone() if k != key else v.clone()) for k, v in sd.items()}
    perturbed[key].reshape(-1)[0] += 1.0
    assert model_behavior_fingerprint(cfg_proj, perturbed) != base


def test_model_behavior_fingerprint_sensitive_to_dtype_and_shape():
    m = _sigma_model()
    sd = m.state_dict()
    cfg_proj = sigma_behavior_config(m.cfg)
    base = model_behavior_fingerprint(cfg_proj, sd)
    key = next(iter(sd))
    dtype_changed = dict(sd); dtype_changed[key] = sd[key].to(torch.float64)
    assert model_behavior_fingerprint(cfg_proj, dtype_changed) != base
    shape_changed = dict(sd); shape_changed[key] = sd[key].reshape(-1)
    assert model_behavior_fingerprint(cfg_proj, shape_changed) != base


def test_model_behavior_fingerprint_sensitive_to_non_policy_config():
    import dataclasses
    m = _sigma_model()
    sd = m.state_dict()
    base = model_behavior_fingerprint(sigma_behavior_config(m.cfg), sd)
    cfg2 = dataclasses.replace(m.cfg, decode_tau=m.cfg.decode_tau + 0.25)
    assert model_behavior_fingerprint(sigma_behavior_config(cfg2), sd) != base


def test_model_behavior_fingerprint_survives_state_dict_save_load(tmp_path):
    m = _sigma_model()
    cfg_proj = sigma_behavior_config(m.cfg)
    base = model_behavior_fingerprint(cfg_proj, m.state_dict())
    p = tmp_path / "sd.pt"
    torch.save(m.state_dict(), p)
    reloaded = torch.load(p, weights_only=True)
    assert model_behavior_fingerprint(cfg_proj, reloaded) == base


def test_model_behavior_fingerprint_rejects_non_tensor_value():
    m = _sigma_model()
    sd = dict(m.state_dict())
    sd["__bogus__"] = "not a tensor"
    with pytest.raises((TypeError, ValueError)):
        model_behavior_fingerprint(sigma_behavior_config(m.cfg), sd)
