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
    installed_skill = tmp_path / "installed" / "verification"
    shutil.copytree(SKILL_ROOT, installed_skill)
    rendered = render_skill_markdown(installed_skill)
    (installed_skill / "SKILL.md").write_text(rendered, encoding="utf-8")

    assert "{{VERIFICATION_GATE_COMMAND}}" not in rendered
    assert str((installed_skill / "scripts" / "verification_gate.py").resolve()) in rendered

    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    result = subprocess.run(
        [sys.executable, str(installed_skill / "scripts" / "verification_gate.py"), "start", "--cwd", str(unrelated_cwd)],
        cwd=unrelated_cwd,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (unrelated_cwd / ".verification" / "ledger.json").is_file()


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
