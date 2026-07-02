r"""Tests for the sigma-validation gate measurement (vfe3/inference/sigma_gate.py; spec Section 4.5).
Pure-statistic tests on synthetic per-token data with known properties, plus the PASS/FAIL decision and
the artifact writer. No model needed (the orchestrator measure_sigma_gate is exercised by the EFE
pipeline; here we pin the statistics that decide the binding gate)."""
import json
from pathlib import Path

import pytest
import torch

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
