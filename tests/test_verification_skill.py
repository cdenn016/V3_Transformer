"""Behavioral contract checks for the installable verification skill."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from agent_tooling.verification.skill.scripts.verification_gate import render_skill_markdown


SKILL_ROOT = Path(__file__).parents[1] / "agent_tooling" / "verification" / "skill"


def _skill_body() -> str:
    return (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")


def test_skill_frontmatter_has_only_the_installable_identity_fields() -> None:
    text = _skill_body()
    assert text.startswith("---\n")
    _, frontmatter, _ = text.split("---", 2)
    assert yaml.safe_load(frontmatter) == {
        "name": "verification",
        "description": "Run evidence-gated, multi-view verification for code, mathematics, sources, and experiments.",
    }


def test_skill_activates_the_gate_and_validates_the_named_ledger() -> None:
    text = _skill_body()
    assert "{{VERIFICATION_GATE_COMMAND}} start --cwd . --ledger .verification/ledger.json" in text
    assert "{{VERIFICATION_GATE_COMMAND}} validate .verification/ledger.json" in text
    assert "final response" in text
    assert ".verification/ledger.json" in text


def test_copied_skill_renders_an_absolute_gate_command_for_an_unrelated_cwd(tmp_path: Path) -> None:
    installed_skill = tmp_path / "installed skill with spaces" / "verification"
    shutil.copytree(SKILL_ROOT, installed_skill)
    rendered = render_skill_markdown(installed_skill, shell="powershell")
    (installed_skill / "SKILL.md").write_text(rendered, encoding="utf-8")

    assert "{{VERIFICATION_GATE_COMMAND}}" not in rendered
    assert str((installed_skill / "scripts" / "verification_gate.py").resolve()) in rendered

    unrelated_cwd = tmp_path / "unrelated cwd"
    unrelated_cwd.mkdir()
    start_command = next(line.strip() for line in rendered.splitlines() if line.strip().endswith("--mode closure"))
    validate_command = next(line.strip() for line in rendered.splitlines() if line.strip().endswith(".verification/ledger.json"))
    expected_prefix = f'& "{Path(sys.executable).resolve()}" "{(installed_skill / "scripts" / "verification_gate.py").resolve()}"'
    assert start_command == f"{expected_prefix} start --cwd . --ledger .verification/ledger.json --mode closure"
    assert validate_command == f"{expected_prefix} validate .verification/ledger.json"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", start_command], cwd=unrelated_cwd, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    ledger_path = unrelated_cwd / ".verification" / "ledger.json"
    assert ledger_path.is_file()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["artifact_revision"] = "rendered"
    ledger["claims"] = [
        {
            "id": "RENDER-001", "domain": "code", "statement": "The installed gate executes.", "severity": "low",
            "state": "EVIDENCE_VERIFIED", "artifact_revision": "rendered", "criteria": [{"name": "execution", "score": 20}], "escalation_triggers": [],
            "views": {
                "calibration_kind": "independent_0_20", "unresolved_disagreement": False,
                "comparison": {"method": "pairwise", "candidate_count": 2, "candidate_ids": ["A", "B"], "pivot_ids": [], "orders": ["AB", "BA"], "matches": [{"left": "A", "right": "B"}, {"left": "B", "right": "A"}]},
                "scores": [{"view_id": "render-a", "criteria": [{"name": "execution", "score": 20}]}, {"view_id": "render-b", "criteria": [{"name": "execution", "score": 20}]}],
            },
            "evidence": [{"kind": "mechanical", "location": "render-test", "artifact_revision": "rendered"}],
            "counterevidence": [], "verifiers": [{"role": "verifier-code"}], "open_obligations": [], "evidence_invalidated": False,
        }
    ]
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    validate_result = subprocess.run(["powershell", "-NoProfile", "-Command", validate_command], cwd=unrelated_cwd, check=False, capture_output=True, text=True)
    assert validate_result.returncode == 0, validate_result.stdout


def test_copied_skill_renders_a_posix_command_for_git_bash(tmp_path: Path) -> None:
    installed_skill = tmp_path / "installed skill with spaces" / "verification"
    shutil.copytree(SKILL_ROOT, installed_skill)
    rendered = render_skill_markdown(installed_skill, shell="posix")
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    assert git_bash.is_file()
    start_command = next(line.strip() for line in rendered.splitlines() if line.strip().endswith("--mode closure"))
    validate_command = next(line.strip() for line in rendered.splitlines() if line.strip().endswith(".verification/ledger.json"))
    assert not start_command.startswith("& ")

    unrelated_cwd = tmp_path / "unrelated posix cwd"
    unrelated_cwd.mkdir()
    start_result = subprocess.run([str(git_bash), "-lc", start_command], cwd=unrelated_cwd, check=False, capture_output=True, text=True)
    assert start_result.returncode == 0, start_result.stderr
    ledger_path = unrelated_cwd / ".verification" / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["artifact_revision"] = "rendered"
    ledger["claims"] = [{
        "id": "POSIX-001", "domain": "code", "statement": "The POSIX command executes.", "severity": "low",
        "state": "EVIDENCE_VERIFIED", "artifact_revision": "rendered", "criteria": [{"name": "execution", "score": 20}], "escalation_triggers": [],
        "views": {"calibration_kind": "independent_0_20", "unresolved_disagreement": False,
                  "comparison": {"method": "pairwise", "candidate_count": 2, "candidate_ids": ["A", "B"], "pivot_ids": [], "orders": ["AB", "BA"], "matches": [{"left": "A", "right": "B"}, {"left": "B", "right": "A"}]},
                  "scores": [{"view_id": "posix-a", "criteria": [{"name": "execution", "score": 20}]}, {"view_id": "posix-b", "criteria": [{"name": "execution", "score": 20}]}]},
        "evidence": [{"kind": "mechanical", "location": "render-test", "artifact_revision": "rendered"}],
        "counterevidence": [], "verifiers": [{"role": "verifier-code"}], "open_obligations": [], "evidence_invalidated": False,
    }]
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    validate_result = subprocess.run([str(git_bash), "-lc", validate_command], cwd=unrelated_cwd, check=False, capture_output=True, text=True)
    assert validate_result.returncode == 0, validate_result.stderr


def test_skill_requires_audits_proofs_and_current_domain_correctness_evidence() -> None:
    text = _skill_body()
    for trigger in ("audit", "proof", "correctness", "verifier", "current mechanical", "reproduced-output"):
        assert trigger in text
    assert "LLM judgment alone" in text
    assert "derivation or formal proof" in text


def test_skill_links_all_domain_criteria_and_spells_out_closure_states() -> None:
    text = _skill_body()
    for reference in (
        "references/criteria-code.md",
        "references/criteria-math.md",
        "references/criteria-evidence.md",
        "references/criteria-experiment.md",
        "references/criteria-general.md",
        "references/contract.md",
    ):
        assert reference in text
        assert (SKILL_ROOT / reference).is_file()
    for state in ("CANDIDATE", "LLM_SUPPORTED", "EVIDENCE_VERIFIED", "REFUTED", "INCONCLUSIVE"):
        assert state in text


def test_skill_requires_adaptive_views_reversed_order_and_abstention() -> None:
    text = _skill_body()
    for phrase in (
        "0 to 20",
        "two views",
        "four or eight",
        "small margins",
        "high dispersion",
        "criterion disagreement",
        "high severity",
        "Reverse A/B order",
        "balanced pivot tournament",
        "INCONCLUSIVE",
        "never majority-vote acceptance",
        "verifier-skeptic",
        "verifier-adjudicator",
    ):
        assert phrase in text


def test_eval_corpus_contains_six_required_behavioral_cases() -> None:
    evaluations = json.loads((SKILL_ROOT / "evals" / "evals.json").read_text(encoding="utf-8"))
    cases = evaluations["evals"]
    assert len(cases) == 6
    corpus = json.dumps(cases).lower()
    for topic in ("code", "mathematics", "primary source", "experiment", "stale", "inconclusive"):
        assert topic in corpus
    stale_case = next(case for case in cases if "stale" in case["prompt"].lower())
    disagreement_case = next(case for case in cases if "majority vote" in case["prompt"].lower())
    assert "if fresh evidence later permits closure" in json.dumps(stale_case).lower()
    assert "a/b reversal" in json.dumps(disagreement_case).lower()
    assert "tournament" not in json.dumps(disagreement_case).lower()
