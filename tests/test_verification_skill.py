"""Behavioral contract checks for the installable verification skill."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from agent_tooling.verification.skill.scripts.verification_gate import render_skill_markdown


SKILL_ROOT = Path(__file__).parents[1] / "agent_tooling" / "verification" / "skill"


def _skill_body() -> str:
    return (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")


def _initialize_git_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "verification@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Verification Test"], cwd=path, check=True)
    (path / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    subprocess.run(["git", "add", "artifact.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "artifact"], cwd=path, check=True)


def _rendered_claim(revision: str, claim_id: str, view_prefix: str) -> dict[str, object]:
    evidence_id = f"{claim_id.lower()}-evidence"
    view_ids = [f"{view_prefix}-a", f"{view_prefix}-b"]
    return {
        "id": claim_id,
        "domain": "code",
        "statement": "The installed gate executes.",
        "severity": "low",
        "state": "EVIDENCE_VERIFIED",
        "artifact_revision": revision,
        "criteria": [{"name": "execution", "score": 20}],
        "escalation_triggers": [],
        "escalation_target": 2,
        "views": {
            "calibration_kind": "independent_0_20",
            "unresolved_disagreement": False,
            "comparison": {
                "method": "pairwise",
                "candidate_count": 2,
                "candidate_ids": ["A", "B"],
                "candidate_descriptions": [
                    {"id": "A", "description": "The execution claim is supported."},
                    {"id": "B", "description": "The execution claim is not supported."},
                ],
                "pivot_ids": [],
                "orders": ["AB", "BA"],
                "matches": [
                    {
                        "left": "A", "right": "B", "view_id": view_ids[0], "outcome": "left",
                        "criteria": [{"name": "execution", "score": 20}],
                        "result_location": f".verification/results/{view_ids[0]}.json",
                    },
                    {
                        "left": "B", "right": "A", "view_id": view_ids[1], "outcome": "right",
                        "criteria": [{"name": "execution", "score": 20}],
                        "result_location": f".verification/results/{view_ids[1]}.json",
                    },
                ],
            },
            "scores": [
                {"view_id": view_ids[0], "criteria": [{"name": "execution", "score": 20}]},
                {"view_id": view_ids[1], "criteria": [{"name": "execution", "score": 20}]},
            ],
        },
        "evidence": [
            {"id": evidence_id, "kind": "mechanical", "location": "render-test", "artifact_revision": revision}
        ],
        "counterevidence": [],
        "verifiers": [
            {
                "role": "verifier-code", "view_ids": view_ids, "result": "support",
                "evidence_ids": [evidence_id], "result_location": ".verification/results/code.json",
            },
            {
                "role": "verifier-adjudicator", "view_ids": view_ids, "result": "support",
                "evidence_ids": [evidence_id], "result_location": ".verification/results/adjudicator.json",
            },
        ],
        "open_obligations": [],
        "evidence_invalidated": False,
    }


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
    assert "{{VERIFICATION_GATE_COMMAND}} validate .verification/ledger.json --cwd ." in text
    assert "final response" in text
    assert ".verification/ledger.json" in text


def test_copied_skill_renders_an_absolute_gate_command_for_an_unrelated_cwd(tmp_path: Path) -> None:
    installed_skill = tmp_path / "installed skill with spaces" / "verification"
    shutil.copytree(SKILL_ROOT, installed_skill)
    rendered = render_skill_markdown(installed_skill, shell="powershell")
    (installed_skill / "SKILL.md").write_text(rendered, encoding="utf-8")

    assert "{{VERIFICATION_GATE_COMMAND}}" not in rendered
    assert "{{VERIFICATION_GATE_SHELL}}" not in rendered
    assert "```powershell" in rendered
    assert str((installed_skill / "scripts" / "verification_gate.py").resolve()) in rendered

    unrelated_cwd = tmp_path / "unrelated cwd"
    unrelated_cwd.mkdir()
    _initialize_git_repository(unrelated_cwd)
    start_command = next(line.strip() for line in rendered.splitlines() if line.strip().endswith("--mode closure"))
    validate_command = next(line.strip() for line in rendered.splitlines() if line.strip().endswith(".verification/ledger.json --cwd ."))
    expected_prefix = f'& "{Path(sys.executable).resolve()}" "{(installed_skill / "scripts" / "verification_gate.py").resolve()}"'
    assert start_command == f"{expected_prefix} start --cwd . --ledger .verification/ledger.json --mode closure"
    assert validate_command == f"{expected_prefix} validate .verification/ledger.json --cwd ."
    result = subprocess.run(["powershell", "-NoProfile", "-Command", start_command], cwd=unrelated_cwd, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    ledger_path = unrelated_cwd / ".verification" / "ledger.json"
    assert ledger_path.is_file()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    revision = ledger["artifact_revision"]
    assert isinstance(revision, str)
    ledger["claims"] = [_rendered_claim(revision, "RENDER-001", "render")]
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    validate_result = subprocess.run(["powershell", "-NoProfile", "-Command", validate_command], cwd=unrelated_cwd, check=False, capture_output=True, text=True)
    assert validate_result.returncode == 0, validate_result.stdout


def test_copied_skill_renders_a_posix_command_for_git_bash(tmp_path: Path) -> None:
    installed_skill = tmp_path / "installed skill with spaces" / "verification"
    shutil.copytree(SKILL_ROOT, installed_skill)
    rendered = render_skill_markdown(installed_skill, shell="posix")
    assert "{{VERIFICATION_GATE_COMMAND}}" not in rendered
    assert "{{VERIFICATION_GATE_SHELL}}" not in rendered
    assert "```bash" in rendered
    bash_path = shutil.which("bash")
    if bash_path is None and sys.platform == "win32":
        fallback = Path("C:/Program Files/Git/bin/bash.exe")
        bash_path = str(fallback) if fallback.is_file() else None
    if bash_path is None:
        pytest.skip("No POSIX shell is available for rendered-command verification.")
    start_command = next(line.strip() for line in rendered.splitlines() if line.strip().endswith("--mode closure"))
    validate_command = next(line.strip() for line in rendered.splitlines() if line.strip().endswith(".verification/ledger.json --cwd ."))
    assert not start_command.startswith("& ")

    unrelated_cwd = tmp_path / "unrelated posix cwd"
    unrelated_cwd.mkdir()
    _initialize_git_repository(unrelated_cwd)
    start_result = subprocess.run([bash_path, "-lc", start_command], cwd=unrelated_cwd, check=False, capture_output=True, text=True)
    assert start_result.returncode == 0, start_result.stderr
    ledger_path = unrelated_cwd / ".verification" / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    revision = ledger["artifact_revision"]
    assert isinstance(revision, str)
    ledger["claims"] = [_rendered_claim(revision, "POSIX-001", "posix")]
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    validate_result = subprocess.run([bash_path, "-lc", validate_command], cwd=unrelated_cwd, check=False, capture_output=True, text=True)
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


def test_skill_documents_revision_binding_candidates_and_terminal_adjudication() -> None:
    text = _skill_body()
    for phrase in (
        "git:<HEAD>:sha256:<digest>",
        "without fabricated criteria, views, or comparison results",
        "stable `id`",
        "candidate descriptions",
        "result location",
        "evidence_invalidated",
        "structured `verifier-adjudicator`",
        "exactly one structured `verifier-adjudicator`",
        "current domain-eligible supporting evidence ID",
        "current domain-eligible `supports: false` counterevidence ID",
    ):
        assert phrase in text


def test_schema_encodes_provenance_and_structured_verifier_records() -> None:
    schema = json.loads((SKILL_ROOT / "schemas" / "claim-ledger.schema.json").read_text(encoding="utf-8"))
    serialized = json.dumps(schema)
    for field in ("candidate_descriptions", "view_id", "outcome", "criteria", "result_location", "evidence_ids"):
        assert field in serialized
    claim_schema = schema["properties"]["claims"]["items"]
    assert "criteria" not in claim_schema["required"]
    assert "views" not in claim_schema["required"]
    assert any("if" in rule and "then" in rule for rule in claim_schema["allOf"])
    assert "[Uu][Nn][Ss][Pp][Ee][Cc][Ii][Ff][Ii][Ee][Dd]" in serialized


def test_contract_requires_recursive_revalidation_and_exact_adjudication() -> None:
    contract = (SKILL_ROOT / "references" / "contract.md").read_text(encoding="utf-8")
    for phrase in (
        "recursive Stop-hook invocation revalidates",
        "exactly one structured `verifier-adjudicator`",
        "current domain-eligible supporting evidence ID",
        "current domain-eligible `supports: false` counterevidence ID",
    ):
        assert phrase in contract


def test_adjudicator_agent_emits_view_and_result_provenance() -> None:
    spec = json.loads(
        (SKILL_ROOT.parent / "agents" / "verifier-adjudicator.json").read_text(encoding="utf-8")
    )
    required = spec["output"]["required"]
    for field in ("result", "view_ids", "evidence_ids", "result_location"):
        assert field in required
    instructions = spec["instructions"]
    for phrase in (
        "exactly one structured result",
        "current domain-eligible supporting evidence ID",
        "current domain-eligible supports:false counterevidence ID",
    ):
        assert phrase in instructions


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
    assert "stable evidence ids" in corpus
    assert "structured adjudicator" in corpus
    assert "current domain-eligible" in corpus
