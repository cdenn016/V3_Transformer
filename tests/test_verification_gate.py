"""Behavioral tests for the deterministic verification Stop gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from agent_tooling.verification.skill.scripts.verification_gate import run_hook, validate_ledger


def valid_code_claim() -> dict[str, object]:
    return {
        "id": "CODE-001",
        "domain": "code",
        "statement": "The parser rejects malformed ledger entries.",
        "severity": "medium",
        "state": "EVIDENCE_VERIFIED",
        "artifact_revision": "abc123",
        "criteria": [{"name": "reachability", "score": 20}],
        "evidence": [
            {
                "kind": "mechanical",
                "location": "tests/test_parser.py::test_rejects_malformed",
                "artifact_revision": "abc123",
            }
        ],
        "counterevidence": [],
        "verifiers": [{"role": "verifier-code"}],
        "open_obligations": [],
        "evidence_invalidated": False,
    }


def valid_ledger() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "artifact_revision": "abc123",
        "claims": [valid_code_claim()],
    }


def test_mechanically_verified_code_claim_validates() -> None:
    assert validate_ledger(valid_ledger()) == []


def test_llm_only_code_claim_cannot_be_evidence_verified() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["evidence"] = [
        {"kind": "llm_judgment", "location": "agent-output.md", "artifact_revision": "abc123"}
    ]

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "mechanical or reproduced_output" in error for error in errors)


def test_numerical_only_mathematical_claim_cannot_be_evidence_verified() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["domain"] = "mathematics"
    claim["evidence"] = [
        {"kind": "numerical", "location": "probe.json", "artifact_revision": "abc123"}
    ]

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "derivation or formal_proof" in error for error in errors)


def test_stale_evidence_is_rejected() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    evidence = claim["evidence"][0]
    assert isinstance(evidence, dict)
    evidence["artifact_revision"] = "old-revision"

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "stale evidence" in error for error in errors)


def test_high_claim_requires_skeptic_and_adjudicator() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["severity"] = "high"

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "verifier-skeptic" in error for error in errors)
    assert any("CODE-001" in error and "verifier-adjudicator" in error for error in errors)


def test_inconclusive_claim_requires_open_obligation() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["state"] = "INCONCLUSIVE"

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "open obligation" in error for error in errors)


def test_errors_are_reported_in_claim_id_order() -> None:
    ledger = valid_ledger()
    first = copy.deepcopy(ledger["claims"][0])
    second = copy.deepcopy(ledger["claims"][0])
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    first["id"] = "Z-001"
    first["state"] = "INCONCLUSIVE"
    second["id"] = "A-001"
    second["state"] = "INCONCLUSIVE"
    ledger["claims"] = [first, second]

    errors = validate_ledger(ledger)

    assert errors[0].startswith("A-001:")
    assert errors[-1].startswith("Z-001:")


def activate(tmp_path: Path, ledger: dict[str, object], ledger_name: str = "ledger.json") -> Path:
    verification_dir = tmp_path / ".verification"
    verification_dir.mkdir()
    ledger_path = verification_dir / ledger_name
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    (verification_dir / "active.json").write_text(
        json.dumps({"ledger": f".verification/{ledger_name}"}), encoding="utf-8"
    )
    return ledger_path


def test_inactive_directory_passes() -> None:
    exit_code, response = run_hook({"cwd": ".", "stop_hook_active": True, "last_assistant_message": "done"})

    assert exit_code == 0
    assert response is None


def test_active_invalid_ledger_blocks(tmp_path: Path) -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["evidence"] = []
    activate(tmp_path, ledger)

    exit_code, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": True, "last_assistant_message": ".verification/ledger.json"}
    )

    assert exit_code == 0
    assert response is not None
    assert response["decision"] == "block"
    assert (tmp_path / ".verification" / "active.json").exists()


def test_active_valid_ledger_requires_final_message_reference(tmp_path: Path) -> None:
    activate(tmp_path, valid_ledger())

    exit_code, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": True, "last_assistant_message": "Implementation is complete."}
    )

    assert exit_code == 0
    assert response is not None
    assert response["decision"] == "block"
    assert (tmp_path / ".verification" / "active.json").exists()


def test_active_valid_referenced_ledger_passes_and_removes_only_marker(tmp_path: Path) -> None:
    ledger_path = activate(tmp_path, valid_ledger())
    ledger_before = ledger_path.read_bytes()

    exit_code, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": True, "last_assistant_message": "Ledger: .verification/ledger.json"}
    )

    assert exit_code == 0
    assert response is None
    assert not (tmp_path / ".verification" / "active.json").exists()
    assert ledger_path.read_bytes() == ledger_before


def test_hook_rejects_ledger_path_traversal(tmp_path: Path) -> None:
    verification_dir = tmp_path / ".verification"
    verification_dir.mkdir()
    (verification_dir / "active.json").write_text(json.dumps({"ledger": "../outside.json"}), encoding="utf-8")

    exit_code, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": True, "last_assistant_message": "../outside.json"}
    )

    assert exit_code == 0
    assert response is not None
    assert response["decision"] == "block"
    assert "traversal" in response["reason"]
