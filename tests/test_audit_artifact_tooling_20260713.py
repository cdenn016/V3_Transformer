import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import make_figures
import scaling
import scaling_analysis
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.path_utils import filesystem_slug
from vfe3.run_artifacts import (
    _save_figures,
    _sha256_tensor_content,
    semantic_config_fingerprint,
)
from vfe3.viz import figures as figs
from vfe3.viz.sweep_adapters import pareto_frontier_kwargs


def _write_finalized_figure_run(
    run_dir: Path,
    cfg:     VFE3Config,

    *,
    model_state: dict[str, torch.Tensor] | None = None,
) -> None:
    """Write one finalized run fixture whose checkpoint is bound to ``cfg``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config = asdict(cfg)
    (run_dir / "config.json").write_text(
        json.dumps({"config": config}), encoding="utf-8",
    )
    model = VFEModel(cfg)
    (run_dir / "summary.json").write_text(json.dumps({
        "n_steps": 1,
        "n_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_val_ppl": 2.0,
    }), encoding="utf-8")
    if model_state is None:
        model_state = model.state_dict()
    torch.save({
        "model_state": model_state,
        "config": config,
        "config_fingerprint": semantic_config_fingerprint(config),
    }, run_dir / "best_model.pt")


def test_save_figures_isolates_failure_and_closes_leaked_figure(tmp_path, monkeypatch, caplog):
    calls = []

    def _plot(*args, **kwargs):
        calls.append(kwargs.get("title"))
        fig = plt.figure()
        if len(calls) == 1:
            raise RuntimeError("synthetic plot failure")
        return fig

    monkeypatch.setattr(figs, "plot_trajectory", _plot)
    artifacts = SimpleNamespace(
        run_dir=tmp_path,
        history=[{"step": 1, "val_ppl": 2.0}],
        cfg=SimpleNamespace(),
    )
    before = set(plt.get_fignums())
    with caplog.at_level(logging.WARNING):
        _save_figures(artifacts, [1.0], logging.getLogger("test.figures"))

    assert calls == ["Training cross-entropy", "Validation perplexity"]
    assert set(plt.get_fignums()) == before
    assert "plot_trajectory" in caplog.text


def test_tensor_content_hash_streams_canonical_int64_chunks():
    tokens = torch.tensor([1, 2, 3, 2**31 - 1, -4], dtype=torch.int32)
    expected = hashlib.sha256(tokens.to(torch.int64).numpy().tobytes()).hexdigest()
    assert _sha256_tensor_content(tokens, chunk_tokens=2) == expected


def test_scaling_loader_constructs_an_explicit_seeded_generator(monkeypatch):
    captured = []

    def _make(*args, **kwargs):
        captured.append(kwargs["generator"])
        return SimpleNamespace(generator=kwargs["generator"])

    scaling._LOADER_CACHE.clear()
    monkeypatch.setattr(scaling, "make_dataloader", _make)
    loader = scaling.get_loader("fixture", 4, 2, "train", data_seed=17)

    assert isinstance(captured[0], torch.Generator)
    assert captured[0].initial_seed() == 17
    assert loader.generator is captured[0]

    second = scaling.get_loader("fixture", 4, 2, "train", data_seed=23)
    assert second is not loader
    assert second.generator.initial_seed() == 23


def test_offset_bootstrap_never_switches_to_fallback_estimator(monkeypatch):
    def _fit(n, loss, *, weights=None, with_offset=False):
        distinct = np.unique(np.asarray(n)).size
        if with_offset and distinct >= 4:
            return {"alpha": 0.5, "form": "offset_power_law"}
        return {"alpha": 0.37, "form": "power_law_fallback_underdetermined"}

    monkeypatch.setattr(figs, "_fit_power_law", _fit)
    points = [
        {"n_params": n, "ce_seeds": [loss - 0.01, loss + 0.01]}
        for n, loss in zip((10, 20, 40, 80), (4.0, 3.5, 3.1, 2.8))
    ]
    alpha, lo, hi = scaling_analysis.bootstrap_exponent_ci(
        points, n_boot=100, with_offset=True,
    )

    assert (alpha, lo, hi) == (0.5, 0.5, 0.5)


def test_fallback_bootstrap_rejects_offset_estimator_replicates(monkeypatch):
    calls = 0

    def _fit(_n, _loss, *, weights=None, with_offset=False):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"alpha": 0.37, "form": "power_law_fallback_solver"}
        return {"alpha": 9.0, "form": "offset_power_law"}

    monkeypatch.setattr(figs, "_fit_power_law", _fit)
    points = [
        {"n_params": n, "ce_seeds": [loss - 0.01, loss + 0.01]}
        for n, loss in zip((10, 20, 40, 80), (4.0, 3.5, 3.1, 2.8))
    ]

    alpha, lo, hi = scaling_analysis.bootstrap_exponent_ci(
        points, n_boot=8, with_offset=True,
    )

    assert alpha == 0.37
    assert np.isnan(lo) and np.isnan(hi)


def test_pareto_omits_wall_time_when_any_point_lacks_finite_timing():
    points = [
        {"route": "grow_K", "label": "a", "n_params": 10.0,
         "val_bits_per_token_mean": 4.0, "wall_time_mean": 2.0},
        {"route": "grow_K", "label": "b", "n_params": 20.0,
         "val_bits_per_token_mean": 3.5, "wall_time_mean": float("nan")},
    ]
    kwargs = pareto_frontier_kwargs(points)
    assert kwargs is not None
    assert "wall_time" not in kwargs["points"]


def test_package_discovery_uses_exact_package_prefix():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "vfe3", "vfe3.*",
    ]


def test_filesystem_slug_is_shared_allowlist_policy():
    assert filesystem_slug("a/b:c") == "a_b_c__3c63076c"
    assert filesystem_slug("../../") == "artifact__323e3584"
    assert filesystem_slug("a b") != filesystem_slug("a:b")
    assert "/" not in filesystem_slug("", fallback="../fallback")
    assert len(filesystem_slug("x" * 500)) <= 130


def test_make_figures_inherits_trained_large_figure_opt_in(tmp_path, monkeypatch):
    _write_finalized_figure_run(
        tmp_path,
        VFE3Config(force_large_figures=True),
    )
    captured = {}

    def _generate(run_dir, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(make_figures, "_generate_figures_isolated", _generate)
    monkeypatch.setitem(make_figures.CONFIG, "run_dir", str(tmp_path))
    monkeypatch.setitem(make_figures.CONFIG, "allow_large", None)
    make_figures.main()

    assert captured["allow_large"] is True


def test_make_figures_surfaces_successful_child_diagnostics(tmp_path, monkeypatch, caplog):
    def _run_process_tree(command, **kwargs):
        del command
        request_path = Path(kwargs["env"]["VFE3_FIGURE_REQUEST"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        Path(request["result_path"]).write_text(
            json.dumps({"paths": []}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="child figure warning",
        )

    monkeypatch.setattr(make_figures, "run_process_tree", _run_process_tree)
    with caplog.at_level(logging.INFO, logger=make_figures.__name__):
        paths = make_figures._generate_figures_isolated(
            tmp_path,
            device="cpu",
            split="validation",
            max_sequences=1,
            n_e_steps=None,
            allow_large=False,
        )

    assert paths == []
    assert "child figure warning" in caplog.text


def test_make_figures_discovers_nested_finalized_run_and_ignores_shells(tmp_path):
    shallow_shell = tmp_path / "newer_shell"
    shallow_shell.mkdir()
    (shallow_shell / "config.json").write_text('{"config": {}}', encoding="utf-8")
    nested = tmp_path / "multiseed_group" / "seed_7" / "actual_run"
    _write_finalized_figure_run(nested, VFE3Config())

    assert make_figures._newest_run(str(tmp_path)) == nested


def test_make_figures_rejects_explicit_incomplete_run(tmp_path):
    (tmp_path / "config.json").write_text('{"config": {}}', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        make_figures._validated_run_dir(tmp_path)


def test_make_figures_skips_newest_corrupt_checkpoint_but_rejects_it_explicitly(tmp_path):
    cfg = VFE3Config()
    valid = tmp_path / "valid"
    corrupt = tmp_path / "corrupt"
    _write_finalized_figure_run(valid, cfg)
    _write_finalized_figure_run(corrupt, cfg)
    corrupt_best = corrupt / "best_model.pt"
    corrupt_best.write_bytes(b"not a checkpoint")
    newer = valid.joinpath("best_model.pt").stat().st_mtime + 10.0
    os.utime(corrupt_best, (newer, newer))

    assert make_figures._newest_run(str(tmp_path)) == valid
    with pytest.raises(ValueError, match="invalid best_model"):
        make_figures._validated_run_dir(corrupt)


def test_make_figures_skips_checkpoint_with_incompatible_model_state(tmp_path):
    cfg = VFE3Config()
    valid = tmp_path / "valid"
    incompatible = tmp_path / "incompatible"
    _write_finalized_figure_run(valid, cfg)
    _write_finalized_figure_run(
        incompatible,
        cfg,
        model_state={"weight": torch.ones(1)},
    )
    newer = valid.joinpath("best_model.pt").stat().st_mtime + 10.0
    os.utime(incompatible / "best_model.pt", (newer, newer))

    assert make_figures._newest_run(str(tmp_path)) == valid
    with pytest.raises(ValueError, match="invalid best_model"):
        make_figures._validated_run_dir(incompatible)
