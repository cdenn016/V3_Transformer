"""Behavioral tests for the deterministic verification Stop gate."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from agent_tooling.verification.skill.scripts.verification_gate import main, run_hook, validate_ledger


def verifier_record(
    role: str,
    *,
    result: str = "support",
    view_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "role": role,
        "view_ids": view_ids or ["code-a", "code-b"],
        "result": result,
        "evidence_ids": ["evidence-code-1"] if evidence_ids is None else evidence_ids,
        "result_location": f".verification/results/{role}.json",
    }


def comparison_match(left: str, right: str, view_id: str, outcome: str = "left") -> dict[str, object]:
    return {
        "left": left,
        "right": right,
        "view_id": view_id,
        "outcome": outcome,
        "criteria": [{"name": "reachability", "score": 20}],
        "result_location": f".verification/results/{view_id}-{left}-{right}.json",
    }


def valid_code_claim() -> dict[str, object]:
    return {
        "id": "CODE-001",
        "domain": "code",
        "statement": "The parser rejects malformed ledger entries.",
        "severity": "medium",
        "state": "EVIDENCE_VERIFIED",
        "artifact_revision": "abc123",
        "criteria": [{"name": "reachability", "score": 20}],
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
                    {"id": "A", "description": "The claim is supported."},
                    {"id": "B", "description": "The claim is not supported."},
                ],
                "pivot_ids": [],
                "orders": ["AB", "BA"],
                "matches": [
                    comparison_match("A", "B", "code-a"),
                    comparison_match("B", "A", "code-b", outcome="right"),
                ],
            },
            "scores": [
                {"view_id": "code-a", "criteria": [{"name": "reachability", "score": 20}]},
                {"view_id": "code-b", "criteria": [{"name": "reachability", "score": 20}]},
            ],
        },
        "evidence": [
            {
                "id": "evidence-code-1",
                "kind": "mechanical",
                "location": "tests/test_parser.py::test_rejects_malformed",
                "artifact_revision": "abc123",
            }
        ],
        "counterevidence": [],
        "verifiers": [verifier_record("verifier-code"), verifier_record("verifier-adjudicator")],
        "open_obligations": [],
        "evidence_invalidated": False,
    }


def valid_ledger() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "mode": "closure",
        "artifact_revision": "abc123",
        "claims": [valid_code_claim()],
    }


def initialize_git_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "verification@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Verification Test"], cwd=path, check=True)
    (path / "artifact.txt").write_text("initial artifact\n", encoding="utf-8")
    (path / ".gitignore").write_text("ignored-artifact.txt\n", encoding="utf-8")
    subprocess.run(["git", "add", "artifact.txt", ".gitignore"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial artifact"], cwd=path, check=True)


def set_artifact_revision(ledger: dict[str, object], revision: str) -> None:
    ledger["artifact_revision"] = revision
    claims = ledger["claims"]
    assert isinstance(claims, list)
    for claim in claims:
        assert isinstance(claim, dict)
        claim["artifact_revision"] = revision
        for field in ("evidence", "counterevidence"):
            entries = claim[field]
            assert isinstance(entries, list)
            for entry in entries:
                assert isinstance(entry, dict)
                entry["artifact_revision"] = revision


def test_mechanically_verified_code_claim_validates() -> None:
    assert validate_ledger(valid_ledger()) == []


def test_closed_claim_rejects_placeholder_artifact_revision() -> None:
    ledger = valid_ledger()
    set_artifact_revision(ledger, "UNSPECIFIED")

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "placeholder artifact_revision" in error for error in errors)


def test_candidate_can_be_queued_without_scores_views_or_comparison() -> None:
    ledger = valid_ledger()
    ledger["mode"] = "triage"
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["state"] = "CANDIDATE"
    claim.pop("criteria")
    claim.pop("views")
    claim["evidence"] = []
    claim["counterevidence"] = []
    claim["verifiers"] = []

    assert validate_ledger(ledger) == []


@pytest.mark.parametrize("field", ("id",))
def test_evidence_requires_a_stable_id(field: str) -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    entry = claim["evidence"][0]
    assert isinstance(entry, dict)
    entry.pop(field)

    assert any("evidence[0]" in error and field in error for error in validate_ledger(ledger))


def test_comparison_requires_candidate_descriptions() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    comparison = views["comparison"]
    assert isinstance(comparison, dict)
    comparison.pop("candidate_descriptions")

    assert any("candidate_descriptions" in error for error in validate_ledger(ledger))


@pytest.mark.parametrize("field", ("view_id", "outcome", "criteria", "result_location"))
def test_comparison_matches_require_result_provenance(field: str) -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    comparison = views["comparison"]
    assert isinstance(comparison, dict)
    match = comparison["matches"][0]
    assert isinstance(match, dict)
    match.pop(field)

    assert any("matches[0]" in error and field in error for error in validate_ledger(ledger))


@pytest.mark.parametrize("state", ("EVIDENCE_VERIFIED", "REFUTED"))
def test_invalidated_evidence_blocks_both_closed_states(state: str) -> None:
    ledger = valid_ledger() if state == "EVIDENCE_VERIFIED" else refuted_code_ledger(
        [{"id": "counter-1", "kind": "mechanical", "location": "tests/test_parser.py", "artifact_revision": "abc123", "supports": False}]
    )
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["evidence_invalidated"] = True

    assert any("invalidated evidence" in error and state in error for error in validate_ledger(ledger))


def test_stale_entries_remain_as_invalidated_inconclusive_audit_history() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["state"] = "INCONCLUSIVE"
    claim["open_obligations"] = ["Reproduce evidence for the current artifact."]
    claim["evidence_invalidated"] = True
    claim["evidence"][0]["artifact_revision"] = "old-revision"
    claim["counterevidence"] = [
        {"id": "counter-old", "kind": "mechanical", "location": "old-output.txt", "artifact_revision": "old-revision", "supports": False}
    ]
    adjudicator = next(item for item in claim["verifiers"] if item["role"] == "verifier-adjudicator")
    adjudicator["result"] = "abstain"
    adjudicator["evidence_ids"] = ["evidence-code-1", "counter-old"]

    errors = validate_ledger(ledger)

    assert not any("stale evidence" in error or "stale counterevidence" in error for error in errors)
    assert errors == []


@pytest.mark.parametrize("state", ("EVIDENCE_VERIFIED", "REFUTED", "INCONCLUSIVE"))
def test_terminal_states_require_a_structured_adjudicator_result(state: str) -> None:
    if state == "REFUTED":
        ledger = refuted_code_ledger(
            [{"id": "counter-1", "kind": "mechanical", "location": "tests/test_parser.py", "artifact_revision": "abc123", "supports": False}]
        )
    else:
        ledger = valid_ledger()
        claim = ledger["claims"][0]
        assert isinstance(claim, dict)
        if state == "INCONCLUSIVE":
            claim["state"] = state
            claim["open_obligations"] = ["Resolve the claim."]
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["verifiers"] = [item for item in claim["verifiers"] if item["role"] != "verifier-adjudicator"]

    assert any("structured verifier-adjudicator result" in error for error in validate_ledger(ledger))


def test_adjudicator_result_must_link_known_views_evidence_and_location() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    adjudicator = next(item for item in claim["verifiers"] if item["role"] == "verifier-adjudicator")
    adjudicator["view_ids"] = ["unknown-view"]
    adjudicator["evidence_ids"] = ["unknown-evidence"]
    adjudicator["result_location"] = ""

    errors = validate_ledger(ledger)

    assert any("unknown view IDs" in error for error in errors)
    assert any("unknown evidence IDs" in error for error in errors)
    assert any("result_location" in error for error in errors)


def test_terminal_claim_requires_exactly_one_adjudicator_record() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["verifiers"].append(verifier_record("verifier-adjudicator", result="refute"))

    assert any("exactly one structured verifier-adjudicator" in error for error in validate_ledger(ledger))


def test_verified_adjudicator_must_link_current_domain_eligible_supporting_evidence() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["evidence"].append(
        {
            "id": "evidence-llm-1",
            "kind": "llm_judgment",
            "location": "agent-output.md",
            "artifact_revision": "abc123",
        }
    )
    adjudicator = next(item for item in claim["verifiers"] if item["role"] == "verifier-adjudicator")
    adjudicator["evidence_ids"] = ["evidence-llm-1"]

    assert any(
        "current domain-eligible supporting evidence ID" in error for error in validate_ledger(ledger)
    )


@pytest.mark.parametrize("bad_link_kind", ("llm", "opposite_polarity"))
def test_refuted_adjudicator_must_link_current_eligible_negative_counterevidence(bad_link_kind: str) -> None:
    ledger = refuted_code_ledger(
        [
            {
                "id": "counter-valid",
                "kind": "mechanical",
                "location": "tests/test_parser.py",
                "artifact_revision": "abc123",
                "supports": False,
            }
        ]
    )
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    bad_entry = {
        "id": "counter-bad",
        "kind": "llm_judgment" if bad_link_kind == "llm" else "mechanical",
        "location": "counter-output.json",
        "artifact_revision": "abc123",
        "supports": False if bad_link_kind == "llm" else True,
    }
    claim["counterevidence"].append(bad_entry)
    adjudicator = next(item for item in claim["verifiers"] if item["role"] == "verifier-adjudicator")
    adjudicator["evidence_ids"] = ["counter-bad"]

    assert any(
        "current domain-eligible supports:false counterevidence ID" in error
        for error in validate_ledger(ledger)
    )


@pytest.mark.parametrize("entry_kind", ("evidence", "counterevidence"))
def test_invalidated_history_still_rejects_placeholder_entry_revisions(entry_kind: str) -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["state"] = "INCONCLUSIVE"
    claim["open_obligations"] = ["Reproduce evidence for the current artifact."]
    claim["evidence_invalidated"] = True
    adjudicator = next(item for item in claim["verifiers"] if item["role"] == "verifier-adjudicator")
    adjudicator["result"] = "abstain"
    if entry_kind == "evidence":
        claim["evidence"][0]["artifact_revision"] = "unspecified"
    else:
        claim["counterevidence"] = [
            {
                "id": "counter-placeholder",
                "kind": "mechanical",
                "location": "old-output.txt",
                "artifact_revision": "placeholder",
                "supports": False,
            }
        ]

    assert any(
        f"{entry_kind}[0]" in error and "placeholder artifact_revision" in error
        for error in validate_ledger(ledger)
    )


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
    claim["verifiers"] = [verifier_record("verifier-code")]

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


def test_closure_mode_rejects_intermediate_states() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["state"] = "LLM_SUPPORTED"

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "closure mode" in error for error in errors)


def test_closure_mode_requires_inconclusive_for_unresolved_disagreement() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    views["unresolved_disagreement"] = True

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "unresolved disagreement" in error and "INCONCLUSIVE" in error for error in errors)


def test_triage_mode_allows_a_candidate_with_auditable_views() -> None:
    ledger = valid_ledger()
    ledger["mode"] = "triage"
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["state"] = "CANDIDATE"
    claim["evidence"] = []
    claim["verifiers"] = []

    assert validate_ledger(ledger) == []


def test_triage_mode_rejects_terminal_evidence_verified_state() -> None:
    ledger = valid_ledger()
    ledger["mode"] = "triage"

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "triage mode" in error and "EVIDENCE_VERIFIED" in error for error in errors)


def test_triage_mode_rejects_terminal_refuted_state() -> None:
    ledger = refuted_code_ledger(
        [{"kind": "mechanical", "location": "tests/test_parser.py", "artifact_revision": "abc123", "supports": False}]
    )
    ledger["mode"] = "triage"

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "triage mode" in error and "REFUTED" in error for error in errors)


def test_ledger_requires_at_least_one_claim() -> None:
    ledger = valid_ledger()
    ledger["claims"] = []

    assert any("ledger: claims must contain at least one claim" == error for error in validate_ledger(ledger))


def test_start_defaults_to_a_closure_ledger(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)
    assert main(["start", "--cwd", str(tmp_path)]) == 0

    ledger = json.loads((tmp_path / ".verification" / "ledger.json").read_text(encoding="utf-8"))
    activation = json.loads((tmp_path / ".verification" / "active.json").read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert ledger["mode"] == "closure"
    assert ledger["artifact_revision"].startswith(f"git:{head}:sha256:")
    assert ledger["artifact_revision"] != "UNSPECIFIED"
    assert activation["artifact_revision"] == ledger["artifact_revision"]


def test_start_fails_closed_outside_a_git_worktree(tmp_path: Path) -> None:
    assert main(["start", "--cwd", str(tmp_path)]) == 2
    assert not (tmp_path / ".verification").exists()


def test_start_with_custom_ledger_creates_ledger_and_activation_parents(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)

    assert main(["start", "--cwd", str(tmp_path), "--ledger", "artifacts/nested/ledger.json"]) == 0

    assert (tmp_path / "artifacts" / "nested" / "ledger.json").is_file()
    assert (tmp_path / ".verification" / "active.json").is_file()


def test_duplicate_view_ids_are_rejected() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    scores = views["scores"]
    assert isinstance(scores, list)
    second = scores[1]
    assert isinstance(second, dict)
    second["view_id"] = "code-a"

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "unique view IDs" in error for error in errors)


def test_view_scores_must_reconstruct_aggregate_criterion_score() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    scores = views["scores"]
    assert isinstance(scores, list)
    first = scores[0]
    assert isinstance(first, dict)
    criteria = first["criteria"]
    assert isinstance(criteria, list)
    criterion = criteria[0]
    assert isinstance(criterion, dict)
    criterion["score"] = 10

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "does not equal mean view score" in error for error in errors)


def test_two_candidate_pairwise_view_requires_reversed_order() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    comparison = views["comparison"]
    assert isinstance(comparison, dict)
    comparison["orders"] = ["AB"]

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "AB and BA" in error for error in errors)


def test_more_than_four_candidates_requires_a_pivot_tournament() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    comparison = views["comparison"]
    assert isinstance(comparison, dict)
    comparison["candidate_count"] = 5

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "pivot_tournament" in error for error in errors)


def test_two_to_four_candidates_require_a_pairwise_comparison() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    views["comparison"] = {
        "method": "pivot_tournament",
        "candidate_count": 4,
        "candidate_ids": ["A", "B", "C", "D"],
        "candidate_descriptions": [
            {"id": candidate, "description": f"Candidate {candidate}."}
            for candidate in ("A", "B", "C", "D")
        ],
        "pivot_ids": ["A"],
        "orders": ["pivot_tournament"],
        "matches": [
            comparison_match("A", "B", "code-a"), comparison_match("B", "A", "code-b"),
            comparison_match("A", "C", "code-a"), comparison_match("C", "A", "code-b"),
            comparison_match("A", "D", "code-a"), comparison_match("D", "A", "code-b"),
        ],
    }

    assert any("CODE-001" in error and "two through four candidates require pairwise" in error for error in validate_ledger(ledger))


def test_high_claim_requires_at_least_four_unique_views() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["severity"] = "high"
    claim["escalation_triggers"] = ["high_severity"]
    claim["escalation_target"] = 4
    claim["verifiers"] = [
        verifier_record("verifier-code"),
        verifier_record("verifier-skeptic"),
        verifier_record("verifier-adjudicator"),
    ]

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "four unique views" in error for error in errors)


def test_high_claim_accepts_four_unique_views_with_challenge_roles() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["severity"] = "high"
    claim["escalation_triggers"] = ["high_severity"]
    claim["escalation_target"] = 4
    claim["verifiers"] = [
        verifier_record("verifier-code", view_ids=["code-a", "code-b", "code-c", "code-d"]),
        verifier_record("verifier-skeptic", view_ids=["code-a", "code-b", "code-c", "code-d"]),
        verifier_record("verifier-adjudicator", view_ids=["code-a", "code-b", "code-c", "code-d"]),
    ]
    views = claim["views"]
    assert isinstance(views, dict)
    scores = views["scores"]
    assert isinstance(scores, list)
    scores.extend(
        [
            {"view_id": "code-c", "criteria": [{"name": "reachability", "score": 20}]},
            {"view_id": "code-d", "criteria": [{"name": "reachability", "score": 20}]},
        ]
    )

    assert validate_ledger(ledger) == []


def test_high_closure_skeptic_must_link_current_evidence() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["severity"] = "high"
    claim["escalation_triggers"] = ["high_severity"]
    claim["escalation_target"] = 4
    claim["verifiers"] = [
        verifier_record("verifier-code", view_ids=["code-a", "code-b", "code-c", "code-d"]),
        verifier_record(
            "verifier-skeptic",
            view_ids=["code-a", "code-b", "code-c", "code-d"],
            evidence_ids=[],
        ),
        verifier_record("verifier-adjudicator", view_ids=["code-a", "code-b", "code-c", "code-d"]),
    ]
    views = claim["views"]
    assert isinstance(views, dict)
    scores = views["scores"]
    assert isinstance(scores, list)
    scores.extend(
        [
            {"view_id": "code-c", "criteria": [{"name": "reachability", "score": 20}]},
            {"view_id": "code-d", "criteria": [{"name": "reachability", "score": 20}]},
        ]
    )

    assert any("structured verifier-skeptic linkage" in error for error in validate_ledger(ledger))


def test_views_require_an_aggregate_criterion_and_nonempty_per_view_criteria() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["criteria"] = []
    views = claim["views"]
    assert isinstance(views, dict)
    scores = views["scores"]
    assert isinstance(scores, list)
    first = scores[0]
    assert isinstance(first, dict)
    first["criteria"] = []

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "at least one aggregate criterion" in error for error in errors)
    assert any("CODE-001" in error and "at least one criterion" in error for error in errors)


def test_view_criteria_must_exactly_cover_aggregate_criteria() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    scores = views["scores"]
    assert isinstance(scores, list)
    second = scores[1]
    assert isinstance(second, dict)
    second["criteria"] = [{"name": "unrelated", "score": 20}]

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "exactly cover aggregate criteria" in error for error in errors)


def valid_pivot_tournament() -> dict[str, object]:
    return {
        "method": "pivot_tournament",
        "candidate_count": 5,
        "candidate_ids": ["A", "B", "C", "D", "E"],
        "candidate_descriptions": [
            {"id": candidate, "description": f"Candidate {candidate}."}
            for candidate in ("A", "B", "C", "D", "E")
        ],
        "pivot_ids": ["A"],
        "orders": ["pivot_tournament"],
        "matches": [
            comparison_match("A", "B", "code-a"), comparison_match("B", "A", "code-b"),
            comparison_match("A", "C", "code-a"), comparison_match("C", "A", "code-b"),
            comparison_match("A", "D", "code-a"), comparison_match("D", "A", "code-b"),
            comparison_match("A", "E", "code-a"), comparison_match("E", "A", "code-b"),
        ],
    }


def test_pivot_tournament_requires_balanced_reconstructible_matches() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    views["comparison"] = valid_pivot_tournament()
    comparison = views["comparison"]
    assert isinstance(comparison, dict)
    matches = comparison["matches"]
    assert isinstance(matches, list)
    matches.pop()

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "both left and right orientations" in error for error in errors)


def test_balanced_pivot_tournament_validates() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    views["comparison"] = valid_pivot_tournament()

    assert validate_ledger(ledger) == []


def test_pairwise_comparisons_require_nonempty_balanced_unique_matches() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    comparison = views["comparison"]
    assert isinstance(comparison, dict)
    comparison["matches"] = [{"left": "A", "right": "B"}, {"left": "A", "right": "B"}]

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "duplicate ordered match" in error for error in errors)
    assert any("CODE-001" in error and "both ordered orientations" in error for error in errors)


def test_pairwise_matches_reject_unknown_or_identical_candidate_ids() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    comparison = views["comparison"]
    assert isinstance(comparison, dict)
    comparison["matches"] = [{"left": "A", "right": "C"}]

    assert any("CODE-001" in error and "distinct known candidate IDs" in error for error in validate_ledger(ledger))


def test_pairwise_comparisons_require_the_complete_ordered_grid() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    comparison = views["comparison"]
    assert isinstance(comparison, dict)
    comparison["candidate_count"] = 3
    comparison["candidate_ids"] = ["A", "B", "C"]

    assert any("CODE-001" in error and "complete ordered pair grid" in error for error in validate_ledger(ledger))


def test_pivot_tournament_requires_every_pivot_for_every_nonpivot_in_both_orientations() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    views["comparison"] = valid_pivot_tournament()
    comparison = views["comparison"]
    assert isinstance(comparison, dict)
    comparison["pivot_ids"] = ["A", "B"]
    comparison["matches"] = [
        {"left": "A", "right": "C"}, {"left": "C", "right": "A"},
        {"left": "A", "right": "D"}, {"left": "D", "right": "A"},
        {"left": "A", "right": "E"}, {"left": "E", "right": "A"},
    ]

    assert any("CODE-001" in error and "complete nonpivot-by-pivot grid" in error for error in validate_ledger(ledger))


def test_pivot_tournament_rejects_duplicate_ordered_matches() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    views = claim["views"]
    assert isinstance(views, dict)
    views["comparison"] = valid_pivot_tournament()
    comparison = views["comparison"]
    assert isinstance(comparison, dict)
    matches = comparison["matches"]
    assert isinstance(matches, list)
    matches.append({"left": "A", "right": "B"})

    assert any("CODE-001" in error and "duplicate ordered match" in error for error in validate_ledger(ledger))


def test_escalation_triggers_require_four_views_for_low_severity_claims() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["escalation_triggers"] = ["small_margin"]
    claim["escalation_target"] = 4

    assert any("CODE-001" in error and "four unique views" in error for error in validate_ledger(ledger))


def test_unresolved_low_severity_disagreement_requires_the_eight_view_stage() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["state"] = "INCONCLUSIVE"
    claim["open_obligations"] = ["Resolve reviewer disagreement."]
    views = claim["views"]
    assert isinstance(views, dict)
    views["unresolved_disagreement"] = True

    assert any("CODE-001" in error and "escalation_target 8" in error for error in validate_ledger(ledger))


def test_four_views_satisfy_escalation_trigger_requirement() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["escalation_triggers"] = ["criterion_disagreement"]
    claim["escalation_target"] = 4
    views = claim["views"]
    assert isinstance(views, dict)
    scores = views["scores"]
    assert isinstance(scores, list)
    scores.extend([
        {"view_id": "code-c", "criteria": [{"name": "reachability", "score": 20}]},
        {"view_id": "code-d", "criteria": [{"name": "reachability", "score": 20}]},
    ])

    assert validate_ledger(ledger) == []


def test_escalation_target_must_equal_actual_views_and_follow_protocol() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["escalation_target"] = 4

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "must equal escalation_target" in error for error in errors)
    assert any("CODE-001" in error and "requires escalation_target 2" in error for error in errors)


def test_high_severity_requires_high_severity_trigger_and_target_four() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["severity"] = "high"
    claim["escalation_target"] = 4
    claim["verifiers"] = [{"role": "verifier-code"}, {"role": "verifier-skeptic"}, {"role": "verifier-adjudicator"}]
    views = claim["views"]
    assert isinstance(views, dict)
    scores = views["scores"]
    assert isinstance(scores, list)
    scores.extend([
        {"view_id": "code-c", "criteria": [{"name": "reachability", "score": 20}]},
        {"view_id": "code-d", "criteria": [{"name": "reachability", "score": 20}]},
    ])

    assert any("CODE-001" in error and "high_severity" in error for error in validate_ledger(ledger))


def test_unresolved_disagreement_requires_criterion_trigger_and_target_eight() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["state"] = "INCONCLUSIVE"
    claim["open_obligations"] = ["Resolve disagreement."]
    claim["escalation_target"] = 4
    claim["escalation_triggers"] = ["criterion_disagreement"]
    views = claim["views"]
    assert isinstance(views, dict)
    views["unresolved_disagreement"] = True
    scores = views["scores"]
    assert isinstance(scores, list)
    scores.extend([
        {"view_id": "code-c", "criteria": [{"name": "reachability", "score": 20}]},
        {"view_id": "code-d", "criteria": [{"name": "reachability", "score": 20}]},
    ])

    assert any("CODE-001" in error and "requires escalation_target 8" in error for error in validate_ledger(ledger))


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


def test_duplicate_claim_ids_are_rejected() -> None:
    ledger = valid_ledger()
    ledger["claims"].append(copy.deepcopy(ledger["claims"][0]))

    assert any("ledger: duplicate claim ID 'CODE-001'" == error for error in validate_ledger(ledger))


def activate(
    tmp_path:         Path,
    ledger:           dict[str, object],
    ledger_reference: str = ".verification/ledger.json",
) -> Path:
    initialize_git_repository(tmp_path)
    assert main(["start", "--cwd", str(tmp_path), "--ledger", ledger_reference]) == 0
    verification_dir = tmp_path / ".verification"
    ledger_path = tmp_path / ledger_reference
    activation = json.loads((verification_dir / "active.json").read_text(encoding="utf-8"))
    revision = activation["artifact_revision"]
    assert isinstance(revision, str)
    set_artifact_revision(ledger, revision)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return ledger_path


def started_valid_ledger(tmp_path: Path) -> tuple[Path, str]:
    ledger_path = activate(tmp_path, valid_ledger())
    activation = json.loads((tmp_path / ".verification" / "active.json").read_text(encoding="utf-8"))
    revision = activation["artifact_revision"]
    assert isinstance(revision, str)
    return ledger_path, revision


def test_inactive_directory_passes() -> None:
    exit_code, response = run_hook({"cwd": ".", "stop_hook_active": True, "last_assistant_message": "done"})

    assert exit_code == 0
    assert response is None


@pytest.mark.parametrize("failure_kind", ("invalid", "stale"))
def test_active_recursive_stop_remains_fail_closed(tmp_path: Path, failure_kind: str) -> None:
    if failure_kind == "invalid":
        ledger = valid_ledger()
        claim = ledger["claims"][0]
        assert isinstance(claim, dict)
        claim["evidence"] = []
        activate(tmp_path, ledger)
    else:
        started_valid_ledger(tmp_path)
        (tmp_path / "artifact.txt").write_text("changed artifact\n", encoding="utf-8")

    _, first_response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": ".verification/ledger.json"}
    )

    exit_code, recursive_response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": True, "last_assistant_message": "User interrupted."}
    )

    assert first_response is not None
    assert first_response["decision"] == "block"
    assert exit_code == 0
    assert recursive_response is not None
    assert recursive_response["decision"] == "block"
    assert (tmp_path / ".verification" / "active.json").is_file()


def test_active_recursive_stop_passes_after_ledger_and_reference_are_valid(tmp_path: Path) -> None:
    started_valid_ledger(tmp_path)
    _, first_response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": "Work is complete."}
    )

    exit_code, recursive_response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": True, "last_assistant_message": ".verification/ledger.json"}
    )

    assert first_response is not None
    assert first_response["decision"] == "block"
    assert exit_code == 0
    assert recursive_response is None
    assert not (tmp_path / ".verification" / "active.json").exists()


def test_active_hook_rejects_nonboolean_recursion_marker(tmp_path: Path) -> None:
    started_valid_ledger(tmp_path)

    _, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": "false", "last_assistant_message": ".verification/ledger.json"}
    )

    assert response is not None
    assert response["decision"] == "block"
    assert "stop_hook_active" in response["reason"]


def test_active_valid_ledger_is_bound_to_unchanged_live_artifact(tmp_path: Path) -> None:
    started_valid_ledger(tmp_path)

    exit_code, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": "Ledger: .verification/ledger.json"}
    )

    assert exit_code == 0
    assert response is None


@pytest.mark.parametrize("change_kind", ("tracked", "untracked"))
def test_active_hook_blocks_worktree_changes_after_activation(tmp_path: Path, change_kind: str) -> None:
    started_valid_ledger(tmp_path)
    if change_kind == "tracked":
        (tmp_path / "artifact.txt").write_text("changed artifact\n", encoding="utf-8")
    else:
        (tmp_path / "new-artifact.txt").write_text("new artifact\n", encoding="utf-8")

    _, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": "Ledger: .verification/ledger.json"}
    )

    assert response is not None
    assert response["decision"] == "block"
    assert "live artifact changed" in response["reason"]


def test_active_hook_blocks_a_staged_index_change_with_original_worktree_content(tmp_path: Path) -> None:
    started_valid_ledger(tmp_path)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("staged artifact\n", encoding="utf-8")
    subprocess.run(["git", "add", "artifact.txt"], cwd=tmp_path, check=True)
    artifact.write_text("initial artifact\n", encoding="utf-8")

    _, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": ".verification/ledger.json"}
    )

    assert response is not None
    assert response["decision"] == "block"
    assert "live artifact changed" in response["reason"]


def test_active_hook_ignores_gitignored_cache_or_output_mutations(tmp_path: Path) -> None:
    started_valid_ledger(tmp_path)
    (tmp_path / "ignored-artifact.txt").write_text("generated output\n", encoding="utf-8")

    exit_code, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": "Ledger: .verification/ledger.json"}
    )

    assert exit_code == 0
    assert response is None


def test_cli_validate_checks_live_artifact_binding(tmp_path: Path) -> None:
    ledger_path, _ = started_valid_ledger(tmp_path)
    assert main(["validate", str(ledger_path), "--cwd", str(tmp_path)]) == 0

    (tmp_path / "artifact.txt").write_text("changed artifact\n", encoding="utf-8")

    assert main(["validate", str(ledger_path), "--cwd", str(tmp_path)]) == 1


def test_cli_validate_fails_closed_outside_a_git_worktree(tmp_path: Path) -> None:
    verification_dir = tmp_path / ".verification"
    verification_dir.mkdir()
    ledger_path = verification_dir / "ledger.json"
    ledger_path.write_text(json.dumps(valid_ledger()), encoding="utf-8")
    (verification_dir / "active.json").write_text(
        json.dumps({"ledger": ".verification/ledger.json", "artifact_revision": "abc123"}),
        encoding="utf-8",
    )

    assert main(["validate", str(ledger_path), "--cwd", str(tmp_path)]) == 1


def test_active_hook_fails_closed_outside_a_git_worktree(tmp_path: Path) -> None:
    verification_dir = tmp_path / ".verification"
    verification_dir.mkdir()
    (verification_dir / "ledger.json").write_text(json.dumps(valid_ledger()), encoding="utf-8")
    (verification_dir / "active.json").write_text(
        json.dumps({"ledger": ".verification/ledger.json", "artifact_revision": "abc123"}),
        encoding="utf-8",
    )

    _, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": ".verification/ledger.json"}
    )

    assert response is not None
    assert response["decision"] == "block"
    assert "git" in response["reason"].lower()


def test_active_hook_blocks_activation_ledger_revision_mismatch(tmp_path: Path) -> None:
    ledger_path, _ = started_valid_ledger(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    set_artifact_revision(ledger, "sha256:" + "0" * 64)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    _, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": "Ledger: .verification/ledger.json"}
    )

    assert response is not None
    assert response["decision"] == "block"
    assert "activation artifact_revision" in response["reason"]


def test_custom_active_ledger_validates_and_passes_unchanged(tmp_path: Path) -> None:
    ledger_reference = "artifacts/nested/ledger.json"
    ledger_path = activate(tmp_path, valid_ledger(), ledger_reference)

    assert main(["validate", str(ledger_path), "--cwd", str(tmp_path)]) == 0
    exit_code, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": ledger_reference}
    )

    assert exit_code == 0
    assert response is None
    assert not (tmp_path / ".verification" / "active.json").exists()


def test_custom_active_ledger_exclusion_does_not_cover_sibling_files(tmp_path: Path) -> None:
    ledger_reference = "artifacts/nested/ledger.json"
    activate(tmp_path, valid_ledger(), ledger_reference)
    (tmp_path / "artifacts" / "nested" / "result.json").write_text("{}\n", encoding="utf-8")

    _, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": ledger_reference}
    )

    assert response is not None
    assert response["decision"] == "block"
    assert "live artifact changed" in response["reason"]


def test_active_invalid_ledger_blocks(tmp_path: Path) -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["evidence"] = []
    activate(tmp_path, ledger)

    exit_code, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": ".verification/ledger.json"}
    )

    assert exit_code == 0
    assert response is not None
    assert response["decision"] == "block"
    assert (tmp_path / ".verification" / "active.json").exists()


def test_active_valid_ledger_requires_final_message_reference(tmp_path: Path) -> None:
    activate(tmp_path, valid_ledger())

    exit_code, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": "Implementation is complete."}
    )

    assert exit_code == 0
    assert response is not None
    assert response["decision"] == "block"
    assert (tmp_path / ".verification" / "active.json").exists()


def test_active_valid_referenced_ledger_passes_and_removes_only_marker(tmp_path: Path) -> None:
    ledger_path = activate(tmp_path, valid_ledger())
    ledger_before = ledger_path.read_bytes()

    exit_code, response = run_hook(
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": "Ledger: .verification/ledger.json"}
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
        {"cwd": str(tmp_path), "stop_hook_active": False, "last_assistant_message": "../outside.json"}
    )

    assert exit_code == 0
    assert response is not None
    assert response["decision"] == "block"
    assert "traversal" in response["reason"]


def refuted_code_ledger(counterevidence: list[dict[str, object]]) -> dict[str, object]:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    for index, entry in enumerate(counterevidence):
        entry.setdefault("id", f"counter-{index + 1}")
    claim["state"] = "REFUTED"
    claim["evidence"] = []
    claim["counterevidence"] = counterevidence
    counterevidence_ids = [str(entry["id"]) for entry in counterevidence]
    for verifier in claim["verifiers"]:
        verifier["evidence_ids"] = counterevidence_ids
    adjudicator = next(item for item in claim["verifiers"] if item["role"] == "verifier-adjudicator")
    adjudicator["result"] = "refute"
    return ledger


def test_refuted_code_claim_requires_counterevidence() -> None:
    errors = validate_ledger(refuted_code_ledger([]))

    assert any("CODE-001" in error and "requires INCONCLUSIVE" in error for error in errors)


def test_refuted_code_claim_rejects_llm_only_counterevidence() -> None:
    errors = validate_ledger(
        refuted_code_ledger(
            [{"kind": "llm_judgment", "location": "agent-output.md", "artifact_revision": "abc123", "supports": False}]
        )
    )

    assert any("CODE-001" in error and "requires INCONCLUSIVE" in error for error in errors)


def test_refuted_code_claim_rejects_stale_counterevidence() -> None:
    errors = validate_ledger(
        refuted_code_ledger(
            [{"kind": "mechanical", "location": "tests/test_parser.py", "artifact_revision": "old-revision", "supports": False}]
        )
    )

    assert any("CODE-001" in error and "stale counterevidence" in error for error in errors)


def test_refuted_code_claim_rejects_wrong_polarity_counterevidence() -> None:
    errors = validate_ledger(
        refuted_code_ledger(
            [{"kind": "mechanical", "location": "tests/test_parser.py", "artifact_revision": "abc123", "supports": True}]
        )
    )

    assert any("CODE-001" in error and "requires INCONCLUSIVE" in error for error in errors)


def test_refuted_code_claim_accepts_current_mechanical_counterevidence() -> None:
    errors = validate_ledger(
        refuted_code_ledger(
            [{"kind": "mechanical", "location": "tests/test_parser.py", "artifact_revision": "abc123", "supports": False}]
        )
    )

    assert errors == []


def test_refuted_source_claim_accepts_reproduced_source_counterevidence() -> None:
    ledger = refuted_code_ledger(
        [{"kind": "reproduced_source", "location": "sources/reproduction.md", "artifact_revision": "abc123", "supports": False}]
    )
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["domain"] = "source"

    assert validate_ledger(ledger) == []
