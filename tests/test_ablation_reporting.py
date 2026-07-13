r"""PB-07 ablation-report wiring: opt-in report sweeps, paired-token artifact publication + identity
binding, and the persisted-artifact adapters (component forest, joint-LR grid).

The two new sweeps (``component_ablation_forest``, ``e_q_mu_sigma_lr_grid``) are opt-in and stay OUT
of ``SWEEP_ORDER``; the grid is the exact Cartesian product of the live one-dimensional E-step LR
entries plus their baseline point. ``run_single`` publishes ``val_token_nats.pt`` atomically and
records its exact byte/tensor identity so the resume validator (``_paired_token_artifact_is_current``)
and the forest adapter (``sweep_adapters._load_token_vector``) both reject a stale, tampered, or
same-shape-different-bytes file. Most tests are pure fixtures; two build a TINY real model (K=4,
n_heads=1, two batches) to exercise the real per-unit extraction and terminal-merge path.
"""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

import ablation
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.viz.sweep_adapters import ablation_forest_kwargs, lr_grid_heatmap_kwargs

DATASET = "wikitext-103"
FIXED_CODE_IDENTITY = {"git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None}


# =============================================================================
# Opt-in report sweeps: registered, derived, and NOT in SWEEP_ORDER
# =============================================================================

def test_component_ablation_forest_sweep_registered_and_opt_in():
    sweep = ablation.SWEEPS["component_ablation_forest"]
    labels = [arm["label"] for arm in sweep["configs"]]
    assert labels == ["baseline", "head_mixer_off", "precision_attention_off"]
    assert sweep["paired_token_bootstrap"] is True
    assert sweep["forest_baseline_label"] == "baseline"
    assert "component_ablation_forest" not in ablation.SWEEP_ORDER
    # The arms flip exactly the two named toggles (plus a no-op baseline arm).
    runs = dict(ablation.make_run_overrides("component_ablation_forest"))
    assert runs["baseline"] == {}
    assert runs["head_mixer_off"] == {"use_head_mixer": False}
    assert runs["precision_attention_off"] == {"precision_weighted_attention": False}


def test_e_q_mu_sigma_lr_grid_cartesian_complete_and_n_runs():
    sweep = ablation.SWEEPS["e_q_mu_sigma_lr_grid"]
    assert "e_q_mu_sigma_lr_grid" not in ablation.SWEEP_ORDER
    assert sweep["grid_x"] == "e_q_mu_lr" and sweep["grid_y"] == "e_q_sigma_lr"
    mus, sigmas = ablation._GRID_MU_LRS, ablation._GRID_SIGMA_LRS
    assert sweep["grid_x_values"] == mus and sweep["grid_y_values"] == sigmas
    assert sweep["grid_baseline"] == (ablation.BASELINE_CONFIG["e_q_mu_lr"],
                                      ablation.BASELINE_CONFIG["e_q_sigma_lr"])
    # sweep_n_runs pins the product; every (mu, sigma) pair appears exactly once.
    assert ablation.sweep_n_runs(sweep) == len(mus) * len(sigmas)
    pairs = [(arm["e_q_mu_lr"], arm["e_q_sigma_lr"]) for arm in sweep["configs"]]
    assert len(pairs) == len(mus) * len(sigmas)
    assert set(pairs) == {(mu, sigma) for sigma in sigmas for mu in mus}
    assert len(set(pairs)) == len(pairs)                          # no duplicate cell


def test_paired_token_fields_in_csv_columns_for_ablation_forest():
    for field in ("paired_token_bootstrap", "val_token_nats_path", "val_token_nats_sha256",
                  "val_token_nats_size_bytes", "val_token_nats_numel", "val_token_nats_dtype"):
        assert field in ablation._CSV_COLUMNS


# =============================================================================
# _paired_token_artifact_is_current: fails closed on any identity drift
# =============================================================================

def _write_token_marker(run_dir: Path, tensor, *, paired=True, write_file=True, **overrides) -> Path:
    r"""A successful cell marker plus (optionally) its ``val_token_nats.pt``; ``overrides`` replaces
    any recorded identity field so a single test can drift exactly one of them."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tpath = run_dir / "val_token_nats.pt"
    if write_file:
        torch.save(tensor, tpath)
    real_sha = hashlib.sha256(tpath.read_bytes()).hexdigest() if tpath.exists() else None
    real_size = tpath.stat().st_size if tpath.exists() else None
    marker = {
        "label": "cell", "status": "success", "error_kind": None,
        "primary_val_ppl": 9.0, "final_val_ppl": 10.0, "seed": 6,
        "paired_token_bootstrap": paired,
        "val_token_nats_path": "val_token_nats.pt" if paired else None,
        "val_token_nats_sha256": real_sha,
        "val_token_nats_size_bytes": real_size,
        "val_token_nats_numel": int(tensor.numel()),
        "val_token_nats_dtype": str(tensor.dtype),
    }
    marker.update(overrides)
    (run_dir / "ablation_result.json").write_text(json.dumps(marker), encoding="utf-8")
    return tpath


def test_paired_token_artifact_is_current_gates_ablation_forest_reuse(tmp_path):
    vec = torch.arange(16, dtype=torch.float32)

    # required=False -> nothing to bind -> always current (even on an empty dir).
    assert ablation._paired_token_artifact_is_current(tmp_path / "empty", required=False) is True

    # A valid artifact whose identity matches the marker -> current.
    ok = tmp_path / "ok"
    _write_token_marker(ok, vec)
    assert ablation._paired_token_artifact_is_current(ok, required=True) is True

    # Missing file -> fail closed.
    missing = tmp_path / "missing"
    _write_token_marker(missing, vec, write_file=False)
    assert ablation._paired_token_artifact_is_current(missing, required=True) is False

    # Stale SHA (bytes changed without updating the marker) -> fail closed.
    stale = tmp_path / "stale"
    _write_token_marker(stale, vec, val_token_nats_sha256="0" * 64)
    assert ablation._paired_token_artifact_is_current(stale, required=True) is False

    # Wrong recorded numel -> fail closed.
    bad_numel = tmp_path / "bad_numel"
    _write_token_marker(bad_numel, vec, val_token_nats_numel=999)
    assert ablation._paired_token_artifact_is_current(bad_numel, required=True) is False

    # Marker says the artifact was not requested, but the sweep now requires it -> fail closed.
    flag_off = tmp_path / "flag_off"
    _write_token_marker(flag_off, vec, paired=False)
    assert ablation._paired_token_artifact_is_current(flag_off, required=True) is False

    # A non-1-D tensor whose numel/dtype/sha all still match the marker -> fail closed (shape rule).
    twod = tmp_path / "twod"
    mat = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    _write_token_marker(twod, mat)
    assert ablation._paired_token_artifact_is_current(twod, required=True) is False


# =============================================================================
# Same-shape finite replacement: BOTH the cache validator and the forest adapter reject it
# =============================================================================

def _write_forest_cell(sweep_dir: Path, label: str, tensor: torch.Tensor) -> Path:
    cell = sweep_dir / label
    cell.mkdir(parents=True, exist_ok=True)
    tpath = cell / "val_token_nats.pt"
    torch.save(tensor, tpath)
    sha = hashlib.sha256(tpath.read_bytes()).hexdigest()
    marker = {
        "sweep": sweep_dir.name, "label": label, "status": "success", "error_kind": None,
        "primary_val_ppl": 10.0, "final_val_ppl": 10.0, "seed": 6,
        "paired_token_bootstrap": True, "val_token_nats_path": "val_token_nats.pt",
        "val_token_nats_sha256": sha, "val_token_nats_size_bytes": tpath.stat().st_size,
        "val_token_nats_numel": int(tensor.numel()), "val_token_nats_dtype": str(tensor.dtype),
    }
    (cell / "ablation_result.json").write_text(json.dumps(marker), encoding="utf-8")
    return cell


def test_same_shape_finite_replacement_rejected_by_cache_and_ablation_forest(tmp_path):
    sweep = tmp_path / "component_ablation_forest"
    _write_forest_cell(sweep, "baseline", torch.ones(32, dtype=torch.float32))
    arm_cell = _write_forest_cell(sweep, "head_mixer_off", torch.ones(32, dtype=torch.float32))

    # Both the cache validator and the forest adapter accept the pristine artifacts.
    assert ablation._paired_token_artifact_is_current(arm_cell, required=True) is True
    assert ablation_forest_kwargs(sweep, "baseline") is not None

    # Overwrite the arm with DIFFERENT finite values of the same shape/dtype: numel/size still match,
    # only the digest changes. Both must now reject (the marker's SHA is stale).
    torch.save(torch.full((32,), 2.0, dtype=torch.float32), arm_cell / "val_token_nats.pt")
    assert ablation._paired_token_artifact_is_current(arm_cell, required=True) is False
    assert ablation_forest_kwargs(sweep, "baseline") is None


# =============================================================================
# sweep_meta.json persistence (adapters must work after a process restart)
# =============================================================================

def _fake_source_ok(dataset, split, *, cache_dir=None):
    return {"format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split),
            "sha256": "0" * 64 + split, "meta": None, "meta_sha256": None}


def _stub_sweep_identity(monkeypatch):
    monkeypatch.setattr(ablation, "_git_code_identity",
                        lambda: dict(FIXED_CODE_IDENTITY))
    monkeypatch.setattr(ablation, "cache_source_identity", _fake_source_ok)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)

    def fake_run_single(label, overrides, run_dir, **kwargs):
        return {"label": label, "error_kind": None, "primary_val_ppl": 8.0,
                "final_val_ppl": 9.0, "seed": 6, "overrides": ablation._jsonable(overrides)}

    monkeypatch.setattr(ablation, "run_single", fake_run_single)


def test_sweep_meta_persists_ablation_forest_report_keys(tmp_path, monkeypatch):
    _stub_sweep_identity(monkeypatch)
    ablation.run_sweep("component_ablation_forest", tmp_path, dataset=DATASET,
                       device=None, seed=6, resume=False)
    meta = json.loads((tmp_path / "component_ablation_forest" / "sweep_meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["paired_token_bootstrap"] is True
    assert meta["forest_baseline_label"] == "baseline"
    assert meta["grid_x"] is None and meta["grid_y"] is None


def test_sweep_meta_persists_e_q_mu_sigma_lr_grid_report_keys(tmp_path, monkeypatch):
    _stub_sweep_identity(monkeypatch)
    ablation.run_sweep("e_q_mu_sigma_lr_grid", tmp_path, dataset=DATASET,
                       device=None, seed=6, resume=False)
    meta = json.loads((tmp_path / "e_q_mu_sigma_lr_grid" / "sweep_meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["grid_x"] == "e_q_mu_lr" and meta["grid_y"] == "e_q_sigma_lr"
    assert meta["grid_x_values"] == ablation._GRID_MU_LRS
    assert meta["grid_y_values"] == ablation._GRID_SIGMA_LRS
    assert meta["grid_baseline"] == list(ablation.SWEEPS["e_q_mu_sigma_lr_grid"]["grid_baseline"])
    assert meta["paired_token_bootstrap"] is False               # a pure grid sweep, no token vectors


# =============================================================================
# Tiny real model: run_single publishes the token vector + preserves grid metadata
# =============================================================================

_TINY_BASELINE = dict(
    vocab_size=6, embed_dim=4, n_heads=1, max_seq_len=8, batch_size=4,
    max_steps=2, n_layers=1, n_e_steps=1, seed=6, warmup_steps=1,
    gauge_group="glk", use_head_mixer=False, generate_figures=False,
    e_q_mu_lr=0.5, e_q_sigma_lr=0.05, e_phi_lr=0.0,
    m_p_mu_lr=0.1, m_p_sigma_lr=0.05, m_phi_lr=0.0,
    kl_max=32,
)


def _tiny_train_loader(seed=7, n=64, seq_len=8, bs=4):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, generator=g)


def _fake_get_loader(dataset, seq_len, batch_size, split, *, max_tokens=None, vocab_size=None):
    if split == "train":
        return _tiny_train_loader(seq_len=seq_len, bs=batch_size)
    base = torch.arange(3).repeat(48 // 3 + 2)
    ds = TokenWindows(base[:48].long(), seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


def _patch_tiny_cell(monkeypatch):
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", dict(_TINY_BASELINE))
    monkeypatch.setattr(ablation, "get_loader", _fake_get_loader)
    monkeypatch.setattr(ablation, "tokens_per_char", lambda *a, **k: 1.0)
    monkeypatch.setattr(ablation, "_tokenizer_tag", lambda *a, **k: "tiktoken")
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: dict(FIXED_CODE_IDENTITY))


def test_run_single_publishes_paired_token_vector_for_ablation_forest(tmp_path, monkeypatch):
    _patch_tiny_cell(monkeypatch)
    torch.manual_seed(0)
    run_dir = tmp_path / "cell"
    result = ablation.run_single("baseline", {}, run_dir, dataset=DATASET,
                                 device=torch.device("cpu"), seed=6,
                                 paired_token_bootstrap=True, max_steps=2)

    assert result["error_kind"] is None
    assert result["paired_token_bootstrap"] is True
    assert result["val_token_nats_path"] == "val_token_nats.pt"

    tpath = run_dir / "val_token_nats.pt"
    assert tpath.is_file()
    # The recorded identity matches the file on disk exactly.
    assert result["val_token_nats_sha256"] == hashlib.sha256(tpath.read_bytes()).hexdigest()
    assert result["val_token_nats_size_bytes"] == tpath.stat().st_size
    assert result["val_token_nats_dtype"] == "torch.float32"

    vec = torch.load(tpath, map_location="cpu", weights_only=True)
    assert vec.ndim == 1 and vec.numel() > 0
    assert int(vec.numel()) == result["val_token_nats_numel"]
    assert bool(torch.isfinite(vec).all())

    # run_sweep persists the marker from the result; the resume validator then accepts the artifact.
    (run_dir / "ablation_result.json").write_text(json.dumps(result, default=str), encoding="utf-8")
    assert ablation._paired_token_artifact_is_current(run_dir, required=True) is True


def test_e_q_mu_sigma_lr_grid_cell_preserves_metadata_through_merge(tmp_path, monkeypatch):
    r"""A grid cell threads BOTH learning rates through overrides; label / seed / overrides / n_params
    / token cap / primary_val_ppl must all survive the terminal-field merge."""
    _patch_tiny_cell(monkeypatch)
    torch.manual_seed(0)
    overrides = {"e_q_mu_lr": 0.5, "e_q_sigma_lr": 0.0}
    result = ablation.run_single("mu=0.5,sigma=0", overrides, tmp_path / "cell", dataset=DATASET,
                                 device=torch.device("cpu"), seed=6, max_tokens=123, max_steps=2)
    assert result["label"] == "mu=0.5,sigma=0"
    assert result["seed"] == 6
    assert result["overrides"] == {"e_q_mu_lr": 0.5, "e_q_sigma_lr": 0.0}
    assert isinstance(result["n_params"], int) and result["n_params"] > 0
    assert result["max_tokens"] == 123
    assert math.isfinite(result["primary_val_ppl"])
    assert result["error_kind"] is None
