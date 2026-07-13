import hashlib
import json
import logging
import tomllib
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

import make_figures
import scaling
import scaling_analysis
from vfe3.path_utils import filesystem_slug
from vfe3.run_artifacts import _save_figures, _sha256_tensor_content
from vfe3.viz import figures as figs
from vfe3.viz.sweep_adapters import pareto_frontier_kwargs


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
    with open("pyproject.toml", "rb") as handle:
        config = tomllib.load(handle)
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == ["vfe3", "vfe3.*"]


def test_filesystem_slug_is_shared_allowlist_policy():
    assert filesystem_slug("a/b:c") == "a_b_c__3c63076c"
    assert filesystem_slug("../../") == "artifact__323e3584"
    assert filesystem_slug("a b") != filesystem_slug("a:b")
    assert "/" not in filesystem_slug("", fallback="../fallback")
    assert len(filesystem_slug("x" * 500)) <= 130


def test_make_figures_inherits_trained_large_figure_opt_in(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({
        "config": {"force_large_figures": True},
    }), encoding="utf-8")
    captured = {}

    def _generate(run_dir, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(make_figures, "generate_figures", _generate)
    monkeypatch.setitem(make_figures.CONFIG, "run_dir", str(tmp_path))
    monkeypatch.setitem(make_figures.CONFIG, "allow_large", None)
    make_figures.main()

    assert captured["allow_large"] is True
