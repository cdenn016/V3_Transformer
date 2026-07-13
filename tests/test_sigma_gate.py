r"""Tests for the sigma-validation gate measurement (vfe3/inference/sigma_gate.py; spec Section 4.5).
Pure-statistic tests on synthetic per-token data with known properties, plus the PASS/FAIL decision and
the artifact writer. No model needed (the orchestrator measure_sigma_gate is exercised by the EFE
pipeline; here we pin the statistics that decide the binding gate)."""
import json
from pathlib import Path

import pytest
import torch

from vfe3.inference import sigma_gate as sigma_gate_module
from vfe3.inference.sigma_gate import (
    evaluate_sigma_gate,
    permutation_floor,
    sigma_binned_ece,
    sigma_stratified_ce,
    spearman_bootstrap_ci,
    verify_gate_artifact,
    write_sigma_gate_artifact,
)


def _calibrated(n, sigma, slope, seed):
    # accuracy decreases with sigma; conf == the true Bernoulli prob -> perfectly calibrated (ece ~ 0)
    g = torch.Generator().manual_seed(seed)
    p = (1.0 - slope * sigma).clamp(0.02, 0.98)
    correct = (torch.rand(n, generator=g) < p).float()
    return p, correct


def test_spearman_bootstrap_ci_brackets_and_bounded():
    g = torch.Generator().manual_seed(0)
    sigma = torch.rand(2000, generator=g)
    ce = 2.0 * sigma + 0.1 * torch.randn(2000, generator=g)     # strong positive signal
    rho, lo, hi = spearman_bootstrap_ci(sigma, ce, n_boot=200, seed=1)
    assert -1.0 <= lo <= hi <= 1.0
    assert rho > 0.8 and lo > 0.0                                # clearly positive, CI excludes 0


def test_spearman_bootstrap_ci_point_estimate_uses_average_tie_ranks():
    sigma = torch.tensor([1.0, 1.0, 2.0, 3.0])
    ce = torch.tensor([1.0, 2.0, 2.0, 3.0])
    rank_sigma = torch.tensor([0.5, 0.5, 2.0, 3.0], dtype=torch.float64)
    rank_ce = torch.tensor([0.0, 1.5, 1.5, 3.0], dtype=torch.float64)
    expected = float(torch.corrcoef(torch.stack([rank_sigma, rank_ce]))[0, 1])

    rho, lo, hi = spearman_bootstrap_ci(sigma, ce, n_boot=32, seed=2)

    assert abs(rho - expected) < 1e-12
    assert -1.0 <= lo <= hi <= 1.0


def test_permutation_floor_is_small_under_the_null():
    g = torch.Generator().manual_seed(0)
    sigma = torch.rand(4000, generator=g)
    ce = torch.rand(4000, generator=g)                          # independent of sigma
    floor = permutation_floor(sigma, ce, n_perm=300, seed=2)
    assert 0.0 < floor < 0.1                                     # noise band ~ 1/sqrt(n)


def test_sigma_stratified_ce_monotone_iff_signal_increasing():
    g = torch.Generator().manual_seed(0)
    sigma = torch.rand(3000, generator=g)
    up = sigma_stratified_ce(sigma, 3.0 * sigma + 0.05 * torch.randn(3000, generator=g))
    assert up["monotone"] is True and up["mono_spearman"] > 0.9
    down = sigma_stratified_ce(sigma, -3.0 * sigma)            # CE decreases with sigma
    assert down["monotone"] is False and down["mono_spearman"] < -0.9


def test_sigma_binned_ece_zero_when_calibrated_high_when_not():
    g = torch.Generator().manual_seed(0)
    sigma = torch.rand(5000, generator=g)
    conf, correct = _calibrated(5000, sigma, slope=0.5, seed=3)
    assert sigma_binned_ece(sigma, conf, correct) < 0.05        # calibrated
    bad_conf = torch.ones(5000)                                 # claims certainty
    bad_correct = torch.zeros(5000)                             # always wrong
    assert sigma_binned_ece(sigma, bad_conf, bad_correct) > 0.9


def test_gate_passes_on_informative_sigma():
    g = torch.Generator().manual_seed(0)
    n = 5000
    sigma = torch.rand(n, generator=g)
    ce = 2.0 * sigma + 0.1 * torch.randn(n, generator=g)        # sigma strongly predicts CE
    conf, correct = _calibrated(n, sigma, slope=0.5, seed=4)
    rec = evaluate_sigma_gate(sigma, ce, conf, correct, n_boot=200, n_perm=200, seed=5)
    assert rec["status"] == "PASS"
    assert rec["sigma_ce_spearman"] >= 0.2 and rec["spearman_ci"][0] > rec["permutation_floor"]
    assert rec["stratified_ce"]["monotone"] and rec["sigma_binned_ece"] < 0.05


def test_gate_fails_on_uninformative_sigma():
    g = torch.Generator().manual_seed(0)
    n = 5000
    sigma = torch.rand(n, generator=g)
    ce = torch.rand(n, generator=g)                            # sigma carries no CE signal
    conf = torch.full((n,), 0.7)
    correct = (torch.rand(n, generator=g) < 0.7).float()
    rec = evaluate_sigma_gate(sigma, ce, conf, correct, n_boot=200, n_perm=200, seed=6)
    assert rec["status"] == "FAIL"                              # spearman ~ 0 < 0.2
    assert rec["sigma_ce_spearman"] < 0.2


def test_write_sigma_gate_artifact_has_required_fields(tmp_path):
    g = torch.Generator().manual_seed(0)
    n = 1500
    sigma = torch.rand(n, generator=g)
    ce = 2.0 * sigma + 0.1 * torch.randn(n, generator=g)
    conf, correct = _calibrated(n, sigma, slope=0.4, seed=7)
    rec = evaluate_sigma_gate(sigma, ce, conf, correct, n_boot=100, n_perm=100, seed=8)
    path = write_sigma_gate_artifact(rec, checkpoint_id="ckpt_test", spec_commit="deadbeef",
                                     seeds=(6, 23, 64), out_dir=str(tmp_path))
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    for key in ("checkpoint_id", "spec_commit", "seeds", "sigma_ce_spearman", "spearman_ci",
                "permutation_floor", "stratified_ce", "sigma_binned_ece", "status"):
        assert key in payload
    assert payload["checkpoint_id"] == "ckpt_test" and payload["spec_commit"] == "deadbeef"
    assert payload["seeds"] == [6, 23, 64] and payload["status"] in ("PASS", "FAIL")


def test_write_sigma_gate_artifact_publishes_with_same_directory_replace(tmp_path, monkeypatch):
    replace_calls = []
    real_replace = sigma_gate_module.os.replace

    def _record_replace(src, dst):
        replace_calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(sigma_gate_module.os, "replace", _record_replace)
    path = Path(write_sigma_gate_artifact(
        {"status": "PASS"}, checkpoint_id="atomic", spec_commit="x", seeds=(6,),
        out_dir=str(tmp_path),
    ))

    assert len(replace_calls) == 1
    tmp, final = replace_calls[0]
    assert final == path
    assert tmp.parent == final.parent == tmp_path
    assert tmp.name.startswith(final.name + ".")
    assert tmp.suffix == ".tmp"
    assert path.is_file()
    assert not tmp.exists()


def test_write_sigma_gate_artifact_uses_unique_temps_for_same_final(tmp_path, monkeypatch):
    replace_sources = []
    real_replace = sigma_gate_module.os.replace

    def _record_replace(src, dst):
        replace_sources.append(Path(src))
        real_replace(src, dst)

    monkeypatch.setattr(sigma_gate_module.os, "replace", _record_replace)
    for status in ("PASS", "FAIL"):
        write_sigma_gate_artifact(
            {"status": status}, checkpoint_id="shared_final", spec_commit="x", seeds=(6,),
            out_dir=str(tmp_path),
        )

    assert len(replace_sources) == 2
    assert replace_sources[0] != replace_sources[1]
    assert all(path.parent == tmp_path and path.suffix == ".tmp" for path in replace_sources)
    assert all(not path.exists() for path in replace_sources)


def test_write_sigma_gate_artifact_cleans_temp_when_replace_fails(tmp_path, monkeypatch):
    path = Path(write_sigma_gate_artifact(
        {"status": "PASS"}, checkpoint_id="replace_failure", spec_commit="x", seeds=(6,),
        out_dir=str(tmp_path),
    ))
    original = path.read_bytes()

    def _fail_replace(_src, _dst):
        raise OSError("publish failed")

    monkeypatch.setattr(sigma_gate_module.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="publish failed"):
        write_sigma_gate_artifact(
            {"status": "FAIL"}, checkpoint_id="replace_failure", spec_commit="x", seeds=(6,),
            out_dir=str(tmp_path),
        )

    assert list(tmp_path.glob("*.tmp")) == []
    assert path.read_bytes() == original


def test_write_sigma_gate_artifact_cleans_temp_when_serialization_fails(tmp_path):
    path = Path(write_sigma_gate_artifact(
        {"status": "PASS"}, checkpoint_id="json_failure", spec_commit="x", seeds=(6,),
        out_dir=str(tmp_path),
    ))
    original = path.read_bytes()

    with pytest.raises(TypeError):
        write_sigma_gate_artifact(
            {"status": "PASS", "not_json": object()}, checkpoint_id="json_failure",
            spec_commit="x", seeds=(6,), out_dir=str(tmp_path),
        )

    assert list(tmp_path.glob("*.tmp")) == []
    assert path.read_bytes() == original


def test_write_artifact_slugs_checkpoint_id(tmp_path):
    r"""A traversal-shaped checkpoint_id must not escape out_dir: the FILENAME is slugified while
    the payload keeps the raw id for provenance."""
    path = write_sigma_gate_artifact({"status": "PASS"}, checkpoint_id="../../evil",
                                     spec_commit="x", seeds=(6,), out_dir=str(tmp_path))
    assert Path(path).resolve().parent == tmp_path.resolve()     # written INSIDE out_dir
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["checkpoint_id"] == "../../evil"              # raw id preserved in the record


def test_write_artifact_separator_id(tmp_path):
    r"""Separators / drive colons in checkpoint_id map to one flat file directly under out_dir."""
    path = write_sigma_gate_artifact({"status": "PASS"}, checkpoint_id="a/b:c",
                                     spec_commit="x", seeds=(6,), out_dir=str(tmp_path))
    assert Path(path).resolve().parent == tmp_path.resolve()
    # slug + stable 8-hex hash of the RAW id (collision disambiguator, audit 2026-07-01 round-3)
    assert Path(path).name.startswith("a_b_c__") and Path(path).name.endswith(".json")
    assert not (tmp_path / "a").exists()                         # no nested directory created


def test_measure_script_checkpoint_not_hardcoded():
    r"""Guards against re-introducing a machine-absolute checkpoint default in the click-to-run
    CONFIG (the empty value fails closed via the guard in sigma_gate_measure.main)."""
    import sigma_gate_measure
    assert not sigma_gate_measure.CONFIG["checkpoint"]


def test_verify_gate_artifact_accepts_pass_rejects_others(tmp_path):
    # Guard 4 content check: only a PASS record (with matching spec_commit when checked) validates.
    ok = tmp_path / "p.json"
    ok.write_text(json.dumps({"status": "PASS", "spec_commit": "c1"}), encoding="utf-8")
    assert verify_gate_artifact(str(ok))["status"] == "PASS"
    assert verify_gate_artifact(str(ok), expected_spec_commit="c1")["status"] == "PASS"
    with pytest.raises(ValueError):                            # missing file
        verify_gate_artifact(str(tmp_path / "nope.json"))
    fail = tmp_path / "f.json"
    fail.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    with pytest.raises(ValueError):                            # FAIL stamp
        verify_gate_artifact(str(fail))
    with pytest.raises(ValueError):                            # spec_commit mismatch
        verify_gate_artifact(str(ok), expected_spec_commit="OTHER")


# ======================================================================================
# PB-06: content-based spec identity, consumer code identity, sealed measurement context,
# preregistry manifest, and the strict sigma-consumer gate (Task 3).
# ======================================================================================
from types import SimpleNamespace

from vfe3.config import VFE3Config
from vfe3.inference.sigma_gate import (
    canonical_json_sha256,
    load_sigma_gate_preregistry,
    sigma_consumer_code_identity,
    sigma_gate_spec_identity,
    sigma_measurement_context,
    verify_sigma_consumer_gate,
)
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import (
    model_behavior_fingerprint,
    semantic_config_fingerprint,
    sigma_behavior_config,
)

SPEC_IDENTITY_GOLDEN = "c136c3242abb6a091d091c67020f0a73746401f478e19fb74b2d6d1e53096691"
FAIL_ARTIFACT_CANON = "f2b55e2f45e9d7146c9f96b371c2a971df43c7a2c6affb6bf1b2941a28205d9f"

_PREREG_REL = "docs/research/active-inference/2026-06-28-sigma-gate-prereg.md"
_SPEC_REL = "docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md"


def _tiny_sigma_model(seed=0, **kw):
    d = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
             n_e_steps=1, e_phi_lr=0.0, family="gaussian_diagonal", seed=seed)
    d.update(kw)
    torch.manual_seed(seed)
    return VFEModel(VFE3Config(**d))


def _seed_wikitext_cache(cache_dir):
    from vfe3.data.datasets import cache_path
    p = cache_path("wikitext-103", "test", suffix="pt", cache_dir=cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.arange(64, dtype=torch.int64), p)
    return p


def _governing_root(tmp, a="prereg-body\n", b="spec-body\n"):
    root = tmp / "gov"
    (root / "docs/research/active-inference").mkdir(parents=True, exist_ok=True)
    (root / "docs/superpowers/specs").mkdir(parents=True, exist_ok=True)
    (root / _PREREG_REL).write_text(a, encoding="utf-8", newline="")
    (root / _SPEC_REL).write_text(b, encoding="utf-8", newline="")
    return root


def _code_root(tmp):
    root = tmp / "code"
    (root / "vfe3/inference").mkdir(parents=True, exist_ok=True)
    (root / "vfe3/foo.py").write_text("x = 1\n", encoding="utf-8", newline="")
    (root / "vfe3/inference/bar.py").write_text("y = 2\n", encoding="utf-8", newline="")
    (root / "sigma_gate_measure.py").write_text("z = 3\n", encoding="utf-8", newline="")
    return root


# ---------- spec identity (content-based, CRLF-normalized, length-delimited) ----------

def test_sigma_gate_spec_identity_is_known_on_tracked_tree():
    # Golden: the restored governing docs on the tracked tree yield the pinned content identity. Never a
    # git SHA -- content-only, so the restore commit is not circular.
    assert sigma_gate_spec_identity() == SPEC_IDENTITY_GOLDEN


def test_spec_identity_lf_crlf_cr_parity(tmp_path):
    a, b = "line1\nline2\n", "alpha\nbeta\n"
    lf = _governing_root(tmp_path / "lf", a, b)
    crlf = _governing_root(tmp_path / "crlf", a.replace("\n", "\r\n"), b.replace("\n", "\r\n"))
    cr = _governing_root(tmp_path / "cr", a.replace("\n", "\r"), b.replace("\n", "\r"))
    i_lf = sigma_gate_spec_identity(root=lf)
    assert i_lf == sigma_gate_spec_identity(root=crlf) == sigma_gate_spec_identity(root=cr)
    assert i_lf != "unknown"


def test_spec_identity_changes_when_either_file_changes(tmp_path):
    base = sigma_gate_spec_identity(root=_governing_root(tmp_path / "b", "A\n", "B\n"))
    only_a = sigma_gate_spec_identity(root=_governing_root(tmp_path / "a", "A2\n", "B\n"))
    only_b = sigma_gate_spec_identity(root=_governing_root(tmp_path / "c", "A\n", "B2\n"))
    assert base != only_a and base != only_b and only_a != only_b


def test_spec_identity_unknown_on_missing_or_undecodable(tmp_path):
    assert sigma_gate_spec_identity(root=tmp_path / "empty") == "unknown"   # missing files
    root = _governing_root(tmp_path)
    (root / _SPEC_REL).write_bytes(b"\xff\xfe not utf8 \x00")
    assert sigma_gate_spec_identity(root=root) == "unknown"                 # undecodable


# ---------- consumer code identity ----------

def test_code_identity_stable_across_json_and_preregistry_writes(tmp_path):
    root = _code_root(tmp_path)
    before = sigma_consumer_code_identity(root=root)
    (root / "vfe3/inference/sigma_gate_preregistry.json").write_text("{}", encoding="utf-8")
    (root / "vfe3/some_artifact.json").write_text('{"a": 1}', encoding="utf-8")
    assert sigma_consumer_code_identity(root=root) == before                # JSON is excluded
    (root / "vfe3/inference/sigma_gate_preregistry.json").write_text(
        '{"k": {"status": "PASS"}}', encoding="utf-8")
    assert sigma_consumer_code_identity(root=root) == before                # updating it changes nothing


def test_code_identity_changes_when_a_python_source_changes(tmp_path):
    root = _code_root(tmp_path)
    before = sigma_consumer_code_identity(root=root)
    (root / "vfe3/foo.py").write_text("x = 999\n", encoding="utf-8", newline="")
    assert sigma_consumer_code_identity(root=root) != before


def test_code_identity_raises_when_a_declared_source_is_unreadable(tmp_path):
    root = tmp_path / "nocode"          # no vfe3/ and no sigma_gate_measure.py
    root.mkdir()
    with pytest.raises(ValueError):
        sigma_consumer_code_identity(root=root)


# ---------- sealed measurement context ----------

def test_sigma_measurement_context_is_sealed(tmp_path):
    _seed_wikitext_cache(tmp_path)
    m = _tiny_sigma_model(max_seq_len=64)
    ctx = sigma_measurement_context(m.cfg, cache_dir=tmp_path)
    assert ctx["dataset"] == "wikitext-103" and ctx["split"] == "test"
    assert ctx["requested_seq_len"] == 128 and ctx["effective_seq_len"] == 64   # min(128, max_seq_len)
    assert ctx["batch_size"] == 16 and ctx["max_batches"] == 20
    assert ctx["shuffle"] is False and ctx["drop_last"] is True
    assert ctx["seeds"] == [6, 23, 64] and ctx["sigma_samples"] == 16
    assert ctx["mc_seed"] == 0 and ctx["sampling_rule"] == "antithetic_shared_v1"
    assert ctx["thresholds"]["spearman_min"] == 0.2 and ctx["thresholds"]["ece_max"] == 0.05
    assert "cache_source_identity" in ctx and "tokenizer_tag" in ctx


def test_sigma_measurement_context_fails_closed_without_corpus(tmp_path):
    m = _tiny_sigma_model()
    with pytest.raises(FileNotFoundError):
        sigma_measurement_context(m.cfg, cache_dir=tmp_path)   # no cache -> forbid


# ---------- preregistry manifest + canonical JSON ----------

def test_repository_sigma_gate_artifact_is_immutable_fail():
    # The shipped manifest binds the production spec identity to FAIL + the historical canonical hash;
    # the restored artifact still hashes to it (LF/CRLF/indentation independent).
    manifest = load_sigma_gate_preregistry()
    entry = manifest[SPEC_IDENTITY_GOLDEN]
    assert entry["status"] == "FAIL"
    assert entry["artifact_sha256"] == FAIL_ARTIFACT_CANON
    repo_artifact = Path(__file__).resolve().parent.parent / \
        "vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json"
    assert canonical_json_sha256(repo_artifact) == FAIL_ARTIFACT_CANON


def test_canonical_json_sha256_ignores_formatting(tmp_path):
    obj = {"b": 2, "a": [1, 2], "c": {"y": 1, "x": 2}}
    p1 = tmp_path / "compact.json"; p1.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    p2 = tmp_path / "pretty.json"; p2.write_text(json.dumps(obj, indent=4, sort_keys=False), encoding="utf-8")
    assert canonical_json_sha256(p1) == canonical_json_sha256(p2)


# ---------- the strict sigma-consumer gate ----------

def _valid_gate(tmp_path, monkeypatch, model=None):
    from vfe3.inference import sigma_gate as sg
    model = model or _tiny_sigma_model(seed=0)
    _seed_wikitext_cache(tmp_path)
    spec = sg.sigma_gate_spec_identity(root=_governing_root(tmp_path))
    code = sg.sigma_consumer_code_identity(root=_code_root(tmp_path))
    meas = sg.sigma_measurement_context(model.cfg, cache_dir=tmp_path)
    behavior = model_behavior_fingerprint(sigma_behavior_config(model.cfg), model.state_dict())
    ctx_fp = semantic_config_fingerprint(meas)
    record = {
        "status": "PASS",
        "checkpoint_id": "synthetic-test-checkpoint",
        "model_behavior_sha256": behavior,
        "spec_commit": spec,
        "code_identity_sha256": code,
        "measurement_context": meas,
        "measurement_context_sha256": ctx_fp,
        "seeds": [6, 23, 64],
        "sigma_ce_spearman": 0.5,
        "spearman_ci": [0.3, 0.7],
        "permutation_floor": 0.1,
        "stratified_ce": {"monotone": True},
        "sigma_binned_ece": 0.01,
        "thresholds": meas["thresholds"],
    }
    path = tmp_path / "synthetic_gate.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    manifest = {spec: {"status": "PASS", "artifact_sha256": sg.canonical_json_sha256(path),
                       "test_only": True}}
    monkeypatch.setattr(sg, "load_sigma_gate_preregistry", lambda *a, **k: manifest)
    return SimpleNamespace(sg=sg, model=model, spec=spec, code=code, meas=meas, behavior=behavior,
                           live_ctx_fp=ctx_fp, record=record, path=path, manifest=manifest)


def _reserialize(g):
    """Rewrite the artifact and keep the temporary manifest's byte hash consistent with the new bytes."""
    g.path.write_text(json.dumps(g.record), encoding="utf-8")
    g.manifest[g.spec]["artifact_sha256"] = g.sg.canonical_json_sha256(g.path)


def _verify(g, **over):
    kw = dict(actual_model_behavior_sha256=g.behavior, actual_spec_identity=g.spec,
              actual_code_identity_sha256=g.code, actual_measurement_context_sha256=g.live_ctx_fp)
    kw.update(over)
    return g.sg.verify_sigma_consumer_gate(str(g.path), **kw)


def test_verify_sigma_consumer_gate_synthetic_pass_plumbing(tmp_path, monkeypatch):
    """Synthetic PASS plumbing test: an artificial PASS record + temporary manifest, spec, code, and
    cache prove the consumer control-flow gate OPENS. It does NOT validate the empirical sigma arm."""
    g = _valid_gate(tmp_path, monkeypatch)
    out = _verify(g)
    assert out["status"] == "PASS"                              # gate opens on the synthetic PASS


def test_verify_rejects_wrong_live_model_same_checkpoint_id(tmp_path, monkeypatch):
    # An artifact measured from model A cannot open sigma_mc for a differently-parametrized model B,
    # even though both would carry the same human checkpoint_id.
    g = _valid_gate(tmp_path, monkeypatch, model=_tiny_sigma_model(seed=0))
    model_b = _tiny_sigma_model(seed=1)
    behavior_b = model_behavior_fingerprint(sigma_behavior_config(model_b.cfg), model_b.state_dict())
    assert behavior_b != g.behavior
    with pytest.raises(ValueError, match="model-behavior"):
        _verify(g, actual_model_behavior_sha256=behavior_b)


def test_verify_rejects_changed_behavior_field_with_fixed_state_dict(tmp_path, monkeypatch):
    import dataclasses
    g = _valid_gate(tmp_path, monkeypatch)
    cfg_tau = dataclasses.replace(g.model.cfg, decode_tau=g.model.cfg.decode_tau + 0.5)
    behavior_tau = model_behavior_fingerprint(sigma_behavior_config(cfg_tau), g.model.state_dict())
    assert behavior_tau != g.behavior                          # same weights, different behavior config
    with pytest.raises(ValueError, match="model-behavior"):
        _verify(g, actual_model_behavior_sha256=behavior_tau)


def test_verify_rejects_stale_spec(tmp_path, monkeypatch):
    g = _valid_gate(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not registered as PASS"):
        _verify(g, actual_spec_identity="0" * 64)             # unregistered live spec identity


def test_verify_rejects_stale_code(tmp_path, monkeypatch):
    g = _valid_gate(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="code identity"):
        _verify(g, actual_code_identity_sha256="0" * 64)


def test_verify_rejects_missing_and_unreadable(tmp_path, monkeypatch):
    g = _valid_gate(tmp_path, monkeypatch)
    g.path.unlink()
    with pytest.raises(ValueError):
        _verify(g)                                            # missing artifact
    g.path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError):
        _verify(g)                                            # unreadable JSON


def test_verify_rejects_fail_record(tmp_path, monkeypatch):
    g = _valid_gate(tmp_path, monkeypatch)
    g.record["status"] = "FAIL"
    _reserialize(g)
    g.manifest[g.spec]["status"] = "PASS"                     # even with a PASS manifest, the record FAILs
    with pytest.raises(ValueError):
        _verify(g)


def test_verify_rejects_missing_behavior_fingerprint(tmp_path, monkeypatch):
    g = _valid_gate(tmp_path, monkeypatch)
    del g.record["model_behavior_sha256"]
    _reserialize(g)
    with pytest.raises(ValueError, match="model-behavior"):
        _verify(g)


@pytest.mark.parametrize("mutate,ident", [
    (lambda r: r["measurement_context"].__setitem__("dataset", "other-corpus"), "dataset"),
    (lambda r: r["measurement_context"].__setitem__("split", "train"), "split"),
    (lambda r: r["measurement_context"].__setitem__("effective_seq_len", 64), "seqlen"),
    (lambda r: r["measurement_context"].__setitem__("batch_size", 8), "batch"),
    (lambda r: r["measurement_context"].__setitem__("max_batches", 5), "maxbatch"),
    (lambda r: r["measurement_context"].__setitem__("mc_seed", 7), "mcseed"),
    (lambda r: r["measurement_context"].__setitem__("sampling_rule", "other_rule"), "rule"),
    (lambda r: r["measurement_context"]["thresholds"].__setitem__("spearman_min", 0.9), "threshold"),
    (lambda r: (r["measurement_context"].__setitem__("seeds", [1, 2, 3]),
                r.__setitem__("seeds", [1, 2, 3])), "seeds"),
    (lambda r: r.__setitem__("measurement_context_sha256", "d" * 64), "fingerprint"),
])
def test_verify_sigma_consumer_gate_rejects_context_tampering(tmp_path, monkeypatch, mutate, ident):
    g = _valid_gate(tmp_path, monkeypatch)
    mutate(g.record)
    if ident != "fingerprint":                               # keep the stored fingerprint self-consistent
        g.record["measurement_context_sha256"] = semantic_config_fingerprint(g.record["measurement_context"])
    _reserialize(g)                                          # byte-hash passes; the context still mismatches live
    with pytest.raises(ValueError):
        _verify(g)


def test_verify_rejects_flipped_historical_fail_under_production_manifest(tmp_path):
    # Copy the restored FAIL artifact, flip only status to PASS, and require rejection: the PRODUCTION
    # manifest entry for the live spec identity remains FAIL (and the statistics still recompute FAIL).
    repo_artifact = Path(__file__).resolve().parent.parent / \
        "vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json"
    record = json.loads(repo_artifact.read_text(encoding="utf-8"))
    record["status"] = "PASS"
    flipped = tmp_path / "flipped.json"
    flipped.write_text(json.dumps(record), encoding="utf-8")
    live_spec = sigma_gate_spec_identity()                   # production manifest, NOT patched
    with pytest.raises(ValueError, match="not registered as PASS"):
        verify_sigma_consumer_gate(
            str(flipped),
            actual_model_behavior_sha256="x", actual_spec_identity=live_spec,
            actual_code_identity_sha256="y", actual_measurement_context_sha256="z")
