"""Behavioral tests for the deterministic verification Stop gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from agent_tooling.verification.skill.scripts.verification_gate import main, run_hook, validate_ledger


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
                "pivot_ids": [],
                "orders": ["AB", "BA"],
                "matches": [{"left": "A", "right": "B"}, {"left": "B", "right": "A"}],
            },
            "scores": [
                {"view_id": "code-a", "criteria": [{"name": "reachability", "score": 20}]},
                {"view_id": "code-b", "criteria": [{"name": "reachability", "score": 20}]},
            ],
        },
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
        "mode": "closure",
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
    assert main(["start", "--cwd", str(tmp_path)]) == 0

    ledger = json.loads((tmp_path / ".verification" / "ledger.json").read_text(encoding="utf-8"))

    assert ledger["mode"] == "closure"


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
        "pivot_ids": ["A"],
        "orders": ["pivot_tournament"],
        "matches": [
            {"left": "A", "right": "B"}, {"left": "B", "right": "A"},
            {"left": "A", "right": "C"}, {"left": "C", "right": "A"},
            {"left": "A", "right": "D"}, {"left": "D", "right": "A"},
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
    claim["verifiers"] = [{"role": "verifier-code"}, {"role": "verifier-skeptic"}, {"role": "verifier-adjudicator"}]

    errors = validate_ledger(ledger)

    assert any("CODE-001" in error and "four unique views" in error for error in errors)


def test_high_claim_accepts_four_unique_views_with_challenge_roles() -> None:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["severity"] = "high"
    claim["escalation_triggers"] = ["high_severity"]
    claim["escalation_target"] = 4
    claim["verifiers"] = [{"role": "verifier-code"}, {"role": "verifier-skeptic"}, {"role": "verifier-adjudicator"}]
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
        "pivot_ids": ["A"],
        "orders": ["pivot_tournament"],
        "matches": [
            {"left": "A", "right": "B"}, {"left": "B", "right": "A"},
            {"left": "A", "right": "C"}, {"left": "C", "right": "A"},
            {"left": "A", "right": "D"}, {"left": "D", "right": "A"},
            {"left": "A", "right": "E"}, {"left": "E", "right": "A"},
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
        {"cwd": str(tmp_path), "stop_hook_active": True, "last_assistant_message": "../outside.json"}
    )

    assert exit_code == 0
    assert response is not None
    assert response["decision"] == "block"
    assert "traversal" in response["reason"]


def refuted_code_ledger(counterevidence: list[dict[str, object]]) -> dict[str, object]:
    ledger = valid_ledger()
    claim = ledger["claims"][0]
    assert isinstance(claim, dict)
    claim["state"] = "REFUTED"
    claim["evidence"] = []
    claim["counterevidence"] = counterevidence
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
