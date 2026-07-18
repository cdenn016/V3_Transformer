"""Deterministic validation and opt-in Stop-hook gate for claim ledgers."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any


CLAIM_STATES = frozenset({"CANDIDATE", "LLM_SUPPORTED", "EVIDENCE_VERIFIED", "REFUTED", "INCONCLUSIVE"})
MODES = frozenset({"triage", "closure"})
DOMAINS = frozenset({"code", "experiment", "mathematics", "evidence", "research", "source", "general"})
SEVERITIES = frozenset({"low", "medium", "high", "critical"})
ESCALATION_TRIGGERS = frozenset({"small_margin", "high_dispersion", "criterion_disagreement", "high_severity"})
EVIDENCE_KINDS = frozenset(
    {
        "llm_judgment",
        "mechanical",
        "reproduced_output",
        "derivation",
        "formal_proof",
        "numerical",
        "primary_source",
        "reproduced_source",
    }
)

_ROOT_FIELDS = frozenset({"schema_version", "mode", "artifact_revision", "claims"})
_CLAIM_FIELDS = frozenset(
    {
        "id",
        "domain",
        "statement",
        "severity",
        "state",
        "artifact_revision",
        "criteria",
        "escalation_triggers",
        "escalation_target",
        "views",
        "evidence",
        "counterevidence",
        "verifiers",
        "open_obligations",
        "evidence_invalidated",
    }
)
_EVIDENCE_FIELDS = frozenset({"kind", "location", "artifact_revision"})
_COUNTEREVIDENCE_FIELDS = frozenset({"kind", "location", "artifact_revision", "supports"})
_CRITERION_FIELDS = frozenset({"name", "score"})
_VERIFIER_FIELDS = frozenset({"role"})
_VIEW_FIELDS = frozenset({"calibration_kind", "unresolved_disagreement", "comparison", "scores"})
_VIEW_COMPARISON_FIELDS = frozenset({"method", "candidate_count", "candidate_ids", "pivot_ids", "orders", "matches"})
_VIEW_SCORE_FIELDS = frozenset({"view_id", "criteria"})
_MATCH_FIELDS = frozenset({"left", "right"})

GATE_COMMAND_TOKEN = "{{VERIFICATION_GATE_COMMAND}}"
GATE_SHELL_TOKEN = "{{VERIFICATION_GATE_SHELL}}"


def _as_dict(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _posix_path(path: Path) -> str:
    """Return a Git-Bash-compatible representation of a local path."""

    text = str(path.resolve()).replace("\\", "/")
    if len(text) >= 3 and text[1:3] == ":/":
        return f"/{text[0].lower()}{text[2:]}"
    return text


def render_skill_markdown(skill_root: Path, shell: str = "powershell") -> str:
    """Render the source skill template for an installed skill directory.

    Task 3 copies the skill before calling this seam, so the gate command names
    the installed absolute script path rather than a repository-relative path.
    """

    root = skill_root.resolve()
    template_path = root / "SKILL.md"
    gate_path = root / "scripts" / "verification_gate.py"
    template = template_path.read_text(encoding="utf-8")
    if GATE_COMMAND_TOKEN not in template or GATE_SHELL_TOKEN not in template:
        raise ValueError("skill template has unresolved verification render tokens")
    if not gate_path.is_file():
        raise FileNotFoundError(gate_path)
    python_path = Path(sys.executable).resolve()
    if shell == "powershell":
        command = f'& "{python_path}" "{gate_path}"'
        shell_label = "powershell"
    elif shell == "posix":
        command = f"{shlex.quote(_posix_path(python_path))} {shlex.quote(_posix_path(gate_path))}"
        shell_label = "bash"
    else:
        raise ValueError("shell must be powershell or posix")
    return template.replace(GATE_COMMAND_TOKEN, command).replace(GATE_SHELL_TOKEN, shell_label)


def _field_errors(prefix: str, value: dict[str, object], fields: frozenset[str]) -> list[str]:
    errors: list[str] = []
    for field in sorted(fields - value.keys()):
        errors.append(f"{prefix}: missing required field '{field}'")
    for field in sorted(value.keys() - fields):
        errors.append(f"{prefix}: unexpected field '{field}'")
    return errors


def _closure_evidence(domain: str) -> frozenset[str]:
    if domain in {"code", "experiment"}:
        return frozenset({"mechanical", "reproduced_output"})
    if domain == "mathematics":
        return frozenset({"derivation", "formal_proof"})
    return frozenset({"primary_source", "reproduced_source"})


def _refutation_evidence(domain: str) -> frozenset[str]:
    if domain in {"code", "experiment"}:
        return frozenset({"mechanical", "reproduced_output"})
    if domain == "mathematics":
        return frozenset({"derivation", "formal_proof"})
    return frozenset({"primary_source", "reproduced_source"})


def _score_is_valid(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 <= value <= 20


def _validate_views(
    prefix:              str,
    value:               object,
    aggregate_criteria:  dict[str, float],
    severity:            object,
    escalation_target:   object,
) -> tuple[list[str], bool]:
    """Validate auditable independent views and their criterion aggregation."""

    errors: list[str] = []
    views = _as_dict(value)
    if views is None:
        return [f"{prefix}: views must be an object"], False
    errors.extend(_field_errors(f"{prefix}: views", views, _VIEW_FIELDS))
    calibration_kind = views.get("calibration_kind")
    if not _nonempty_string(calibration_kind):
        errors.append(f"{prefix}: views calibration_kind must be a nonempty string")
    unresolved = views.get("unresolved_disagreement")
    if not isinstance(unresolved, bool):
        errors.append(f"{prefix}: views unresolved_disagreement must be a boolean")

    comparison = _as_dict(views.get("comparison"))
    if comparison is None:
        errors.append(f"{prefix}: views comparison must be an object")
    else:
        errors.extend(_field_errors(f"{prefix}: views comparison", comparison, _VIEW_COMPARISON_FIELDS))
        method = comparison.get("method")
        candidate_count = comparison.get("candidate_count")
        candidate_ids = comparison.get("candidate_ids")
        pivot_ids = comparison.get("pivot_ids")
        orders = comparison.get("orders")
        matches = comparison.get("matches")
        if method not in {"pairwise", "pivot_tournament"}:
            errors.append(f"{prefix}: views comparison method must be pairwise or pivot_tournament")
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 2:
            errors.append(f"{prefix}: views comparison candidate_count must be an integer of at least 2")
        if not isinstance(candidate_ids, list) or any(not _nonempty_string(item) for item in candidate_ids):
            errors.append(f"{prefix}: views comparison candidate_ids must be a list of nonempty strings")
            candidate_set: set[str] = set()
        else:
            candidate_set = {str(item) for item in candidate_ids}
            if len(candidate_set) != len(candidate_ids):
                errors.append(f"{prefix}: views comparison candidate_ids must be unique")
            if isinstance(candidate_count, int) and not isinstance(candidate_count, bool) and len(candidate_ids) != candidate_count:
                errors.append(f"{prefix}: views comparison candidate_ids must match candidate_count")
        if not isinstance(pivot_ids, list) or any(not _nonempty_string(item) for item in pivot_ids):
            errors.append(f"{prefix}: views comparison pivot_ids must be a list of nonempty strings")
            pivot_set: set[str] = set()
        else:
            pivot_set = {str(item) for item in pivot_ids}
            if len(pivot_set) != len(pivot_ids):
                errors.append(f"{prefix}: views comparison pivot_ids must be unique")
        if not isinstance(orders, list) or any(not _nonempty_string(item) for item in orders):
            errors.append(f"{prefix}: views comparison orders must be a list of nonempty strings")
        elif candidate_count == 2 and method == "pairwise" and not {"AB", "BA"}.issubset(set(orders)):
            errors.append(f"{prefix}: two-candidate pairwise comparison requires AB and BA orders")
        if isinstance(candidate_count, int) and not isinstance(candidate_count, bool) and 2 <= candidate_count <= 4 and method != "pairwise":
            errors.append(f"{prefix}: two through four candidates require pairwise")
        if isinstance(candidate_count, int) and not isinstance(candidate_count, bool) and candidate_count > 4 and method != "pivot_tournament":
            errors.append(f"{prefix}: more than four candidates require pivot_tournament")
        if not isinstance(matches, list):
            errors.append(f"{prefix}: views comparison matches must be a list")
            match_items: list[dict[str, object]] = []
        else:
            match_items = []
            for index, item in enumerate(matches):
                match = _as_dict(item)
                item_prefix = f"{prefix}: views comparison matches[{index}]"
                if match is None:
                    errors.append(f"{item_prefix} must be an object")
                    continue
                errors.extend(_field_errors(item_prefix, match, _MATCH_FIELDS))
                left = match.get("left")
                right = match.get("right")
                if not _nonempty_string(left) or not _nonempty_string(right):
                    errors.append(f"{item_prefix}: left and right must be nonempty strings")
                    continue
                match_items.append(match)
        ordered_matches: set[tuple[str, str]] = set()
        valid_matches: set[tuple[str, str]] = set()
        for index, match in enumerate(match_items):
            left = str(match["left"])
            right = str(match["right"])
            item_prefix = f"{prefix}: views comparison matches[{index}]"
            if left not in candidate_set or right not in candidate_set or left == right:
                errors.append(f"{item_prefix}: match must use distinct known candidate IDs")
                continue
            ordered = (left, right)
            if ordered in ordered_matches:
                errors.append(f"{item_prefix}: duplicate ordered match")
                continue
            ordered_matches.add(ordered)
            valid_matches.add(ordered)
        if method == "pairwise":
            if not valid_matches:
                errors.append(f"{prefix}: pairwise comparison requires at least one explicit match")
            expected_matches = {(left, right) for left in candidate_set for right in candidate_set if left != right}
            if valid_matches != expected_matches:
                errors.append(f"{prefix}: pairwise comparison requires the complete ordered pair grid with both ordered orientations")
        if method == "pivot_tournament":
            if not pivot_set or not pivot_set < candidate_set:
                errors.append(f"{prefix}: pivot_tournament requires pivot_ids as a nonempty proper subset of candidate_ids")
            for index, (left, right) in enumerate(valid_matches):
                pivots_in_match = int(left in pivot_set) + int(right in pivot_set)
                if pivots_in_match != 1:
                    errors.append(f"{prefix}: views comparison matches[{index}] must contain exactly one pivot ID")
            expected_matches = {
                (pivot, nonpivot)
                for pivot in pivot_set
                for nonpivot in candidate_set - pivot_set
            }
            expected_matches.update((nonpivot, pivot) for pivot in pivot_set for nonpivot in candidate_set - pivot_set)
            if valid_matches != expected_matches:
                errors.append(f"{prefix}: pivot_tournament requires a complete nonpivot-by-pivot grid in both left and right orientations")

    score_records = views.get("scores")
    per_view_criteria: list[dict[str, float]] = []
    view_ids: list[str] = []
    if not isinstance(score_records, list) or len(score_records) < 2:
        errors.append(f"{prefix}: views scores must contain at least two views")
    else:
        for index, item in enumerate(score_records):
            record = _as_dict(item)
            item_prefix = f"{prefix}: views scores[{index}]"
            if record is None:
                errors.append(f"{item_prefix} must be an object")
                continue
            errors.extend(_field_errors(item_prefix, record, _VIEW_SCORE_FIELDS))
            view_id = record.get("view_id")
            if not _nonempty_string(view_id):
                errors.append(f"{item_prefix}: view_id must be a nonempty string")
            else:
                view_ids.append(str(view_id))
            criteria = record.get("criteria")
            current: dict[str, float] = {}
            if not isinstance(criteria, list):
                errors.append(f"{item_prefix}: criteria must be a list")
            elif not criteria:
                errors.append(f"{item_prefix}: criteria must contain at least one criterion")
            else:
                for criterion_index, criterion_value in enumerate(criteria):
                    criterion = _as_dict(criterion_value)
                    criterion_prefix = f"{item_prefix}: criteria[{criterion_index}]"
                    if criterion is None:
                        errors.append(f"{criterion_prefix} must be an object")
                        continue
                    errors.extend(_field_errors(criterion_prefix, criterion, _CRITERION_FIELDS))
                    name = criterion.get("name")
                    score = criterion.get("score")
                    if not _nonempty_string(name):
                        errors.append(f"{criterion_prefix}: name must be a nonempty string")
                    elif _score_is_valid(score):
                        if str(name) in current:
                            errors.append(f"{criterion_prefix}: name must not repeat within a view")
                        else:
                            current[str(name)] = float(score)
                    else:
                        errors.append(f"{criterion_prefix}: score must be a number from 0 to 20")
            per_view_criteria.append(current)
    if len(view_ids) != len(set(view_ids)):
        errors.append(f"{prefix}: views must contain unique view IDs")
    if escalation_target in {2, 4, 8} and len(set(view_ids)) != escalation_target:
        target_word = {2: "two", 4: "four", 8: "eight"}[escalation_target]
        errors.append(f"{prefix}: requires {target_word} unique views; actual count must equal escalation_target {escalation_target}")
    if not aggregate_criteria:
        errors.append(f"{prefix}: criteria must contain at least one aggregate criterion")
    for name, aggregate_score in aggregate_criteria.items():
        values = [criteria.get(name) for criteria in per_view_criteria]
        if len(values) < 2 or any(score is None for score in values):
            errors.append(f"{prefix}: every view must score aggregate criterion '{name}'")
        elif not math.isclose(aggregate_score, sum(values) / len(values), rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"{prefix}: aggregate criterion '{name}' does not equal mean view score")
    if aggregate_criteria and any(set(criteria) != set(aggregate_criteria) for criteria in per_view_criteria):
        errors.append(f"{prefix}: view criteria must exactly cover aggregate criteria")
    return errors, unresolved is True


def validate_ledger(data: dict[str, object]) -> list[str]:
    """Return all deterministic contract violations in claim-ID order.

    The validator evaluates evidence eligibility and freshness, not the truth of a
    claim. Errors for each claim remain grouped and claims are sorted by ID so
    callers get stable machine-readable output.
    """

    root = _as_dict(data)
    if root is None:
        return ["ledger: expected an object"]

    errors = _field_errors("ledger", root, _ROOT_FIELDS)
    if root.get("schema_version") != "1.0":
        errors.append("ledger: schema_version must be '1.0'")
    mode = root.get("mode")
    if mode not in MODES:
        errors.append("ledger: mode must be triage or closure")
    revision = root.get("artifact_revision")
    if not _nonempty_string(revision):
        errors.append("ledger: artifact_revision must be a nonempty string")
    claims_value = root.get("claims")
    if not isinstance(claims_value, list):
        return errors + ["ledger: claims must be a list"]
    if not claims_value:
        errors.append("ledger: claims must contain at least one claim")

    claim_items: list[tuple[str, int, object]] = []
    claim_ids: set[str] = set()
    for index, claim in enumerate(claims_value):
        claim_dict = _as_dict(claim)
        claim_id = claim_dict.get("id") if claim_dict is not None else None
        if _nonempty_string(claim_id):
            claim_id_text = str(claim_id)
            if claim_id_text in claim_ids:
                errors.append(f"ledger: duplicate claim ID '{claim_id_text}'")
            claim_ids.add(claim_id_text)
        sort_id = claim_id if _nonempty_string(claim_id) else f"~{index:08d}"
        claim_items.append((str(sort_id), index, claim))

    for _, index, value in sorted(claim_items, key=lambda item: (item[0], item[1])):
        claim = _as_dict(value)
        prefix = f"claim[{index}]"
        if claim is None:
            errors.append(f"{prefix}: expected an object")
            continue
        claim_id = claim.get("id")
        prefix = str(claim_id) if _nonempty_string(claim_id) else prefix
        errors.extend(_field_errors(prefix, claim, _CLAIM_FIELDS))
        if not _nonempty_string(claim_id):
            errors.append(f"{prefix}: id must be a nonempty string")
        domain = claim.get("domain")
        if domain not in DOMAINS:
            errors.append(f"{prefix}: domain must be one of {', '.join(sorted(DOMAINS))}")
        if not _nonempty_string(claim.get("statement")):
            errors.append(f"{prefix}: statement must be a nonempty string")
        severity = claim.get("severity")
        if severity not in SEVERITIES:
            errors.append(f"{prefix}: severity must be one of {', '.join(sorted(SEVERITIES))}")
        escalation_triggers = claim.get("escalation_triggers")
        trigger_set: set[str] = set()
        if not isinstance(escalation_triggers, list):
            errors.append(f"{prefix}: escalation_triggers must be a list")
        elif any(trigger not in ESCALATION_TRIGGERS for trigger in escalation_triggers):
            errors.append(f"{prefix}: escalation_triggers must use only {', '.join(sorted(ESCALATION_TRIGGERS))}")
        elif len(set(escalation_triggers)) != len(escalation_triggers):
            errors.append(f"{prefix}: escalation_triggers must be unique")
        else:
            trigger_set = {str(trigger) for trigger in escalation_triggers}
        escalation_target = claim.get("escalation_target")
        if isinstance(escalation_target, bool) or escalation_target not in {2, 4, 8}:
            errors.append(f"{prefix}: escalation_target must be 2, 4, or 8")
        state = claim.get("state")
        if state not in CLAIM_STATES:
            errors.append(f"{prefix}: state must be one of {', '.join(sorted(CLAIM_STATES))}")
        claim_revision = claim.get("artifact_revision")
        if not _nonempty_string(claim_revision):
            errors.append(f"{prefix}: artifact_revision must be a nonempty string")
        elif _nonempty_string(revision) and claim_revision != revision:
            errors.append(f"{prefix}: artifact_revision does not match ledger artifact_revision")
        if not isinstance(claim.get("evidence_invalidated"), bool):
            errors.append(f"{prefix}: evidence_invalidated must be a boolean")

        aggregate_criteria: dict[str, float] = {}
        criteria = claim.get("criteria")
        if not isinstance(criteria, list):
            errors.append(f"{prefix}: criteria must be a list")
        else:
            for criterion_index, value in enumerate(criteria):
                criterion = _as_dict(value)
                item_prefix = f"{prefix}: criteria[{criterion_index}]"
                if criterion is None:
                    errors.append(f"{item_prefix} must be an object")
                    continue
                errors.extend(_field_errors(item_prefix, criterion, _CRITERION_FIELDS))
                if not _nonempty_string(criterion.get("name")):
                    errors.append(f"{item_prefix}: name must be a nonempty string")
                score = criterion.get("score")
                if not _score_is_valid(score):
                    errors.append(f"{item_prefix}: score must be a number from 0 to 20")
                elif _nonempty_string(criterion.get("name")):
                    name = str(criterion["name"])
                    if name in aggregate_criteria:
                        errors.append(f"{item_prefix}: aggregate criterion name must not repeat")
                    else:
                        aggregate_criteria[name] = float(score)

        view_errors, unresolved_disagreement = _validate_views(
            prefix,
            claim.get("views"),
            aggregate_criteria,
            severity,
            escalation_target,
        )
        errors.extend(view_errors)
        if severity in {"high", "critical"} and "high_severity" not in trigger_set:
            errors.append(f"{prefix}: high and critical severity require high_severity in escalation_triggers")
        if unresolved_disagreement and "criterion_disagreement" not in trigger_set:
            errors.append(f"{prefix}: unresolved disagreement requires criterion_disagreement in escalation_triggers")
        escalation_required = bool(trigger_set) or severity in {"high", "critical"} or unresolved_disagreement
        if not escalation_required and escalation_target != 2:
            errors.append(f"{prefix}: no escalation requirement requires escalation_target 2")
        elif unresolved_disagreement and escalation_target != 8:
            errors.append(f"{prefix}: unresolved disagreement after a four-view pass requires escalation_target 8")
        elif escalation_required and escalation_target not in {4, 8}:
            errors.append(f"{prefix}: escalation requires escalation_target 4 or 8")

        evidence_kinds: set[str] = set()
        evidence = claim.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{prefix}: evidence must be a list")
        else:
            for evidence_index, value in enumerate(evidence):
                entry = _as_dict(value)
                item_prefix = f"{prefix}: evidence[{evidence_index}]"
                if entry is None:
                    errors.append(f"{item_prefix} must be an object")
                    continue
                errors.extend(_field_errors(item_prefix, entry, _EVIDENCE_FIELDS))
                kind = entry.get("kind")
                if kind not in EVIDENCE_KINDS:
                    errors.append(f"{item_prefix}: kind must be one of {', '.join(sorted(EVIDENCE_KINDS))}")
                else:
                    evidence_kinds.add(str(kind))
                if not _nonempty_string(entry.get("location")):
                    errors.append(f"{item_prefix}: location must be a nonempty string")
                evidence_revision = entry.get("artifact_revision")
                if not _nonempty_string(evidence_revision):
                    errors.append(f"{item_prefix}: artifact_revision must be a nonempty string")
                elif _nonempty_string(revision) and evidence_revision != revision:
                    errors.append(f"{prefix}: stale evidence at evidence[{evidence_index}]")

        counterevidence_kinds: set[str] = set()
        counterevidence = claim.get("counterevidence")
        if not isinstance(counterevidence, list):
            errors.append(f"{prefix}: counterevidence must be a list")
        else:
            for counterevidence_index, value in enumerate(counterevidence):
                entry = _as_dict(value)
                item_prefix = f"{prefix}: counterevidence[{counterevidence_index}]"
                if entry is None:
                    errors.append(f"{item_prefix} must be an object")
                    continue
                errors.extend(_field_errors(item_prefix, entry, _COUNTEREVIDENCE_FIELDS))
                kind = entry.get("kind")
                supports = entry.get("supports")
                if kind not in EVIDENCE_KINDS:
                    errors.append(f"{item_prefix}: kind must be one of {', '.join(sorted(EVIDENCE_KINDS))}")
                elif supports is False:
                    counterevidence_kinds.add(str(kind))
                if not _nonempty_string(entry.get("location")):
                    errors.append(f"{item_prefix}: location must be a nonempty string")
                counterevidence_revision = entry.get("artifact_revision")
                if not _nonempty_string(counterevidence_revision):
                    errors.append(f"{item_prefix}: artifact_revision must be a nonempty string")
                elif _nonempty_string(revision) and counterevidence_revision != revision:
                    errors.append(f"{prefix}: stale counterevidence at counterevidence[{counterevidence_index}]")
                if not isinstance(supports, bool):
                    errors.append(f"{item_prefix}: supports must be a boolean")

        roles: set[str] = set()
        verifiers = claim.get("verifiers")
        if not isinstance(verifiers, list):
            errors.append(f"{prefix}: verifiers must be a list")
        else:
            for verifier_index, value in enumerate(verifiers):
                verifier = _as_dict(value)
                item_prefix = f"{prefix}: verifiers[{verifier_index}]"
                if verifier is None:
                    errors.append(f"{item_prefix} must be an object")
                    continue
                errors.extend(_field_errors(item_prefix, verifier, _VERIFIER_FIELDS))
                role = verifier.get("role")
                if not _nonempty_string(role):
                    errors.append(f"{item_prefix}: role must be a nonempty string")
                else:
                    roles.add(str(role))

        obligations = claim.get("open_obligations")
        if not isinstance(obligations, list):
            errors.append(f"{prefix}: open_obligations must be a list")
        elif any(not _nonempty_string(item) for item in obligations):
            errors.append(f"{prefix}: every open obligation must be a nonempty string")

        if state == "INCONCLUSIVE" and (not isinstance(obligations, list) or not obligations):
            errors.append(f"{prefix}: INCONCLUSIVE claims require at least one open obligation")
        if state in {"EVIDENCE_VERIFIED", "REFUTED"} and isinstance(obligations, list) and obligations:
            errors.append(f"{prefix}: closed claims may not retain open obligations")
        if state == "EVIDENCE_VERIFIED":
            if mode == "closure" and unresolved_disagreement:
                errors.append(f"{prefix}: unresolved disagreement in closure mode requires INCONCLUSIVE")
            if claim.get("evidence_invalidated") is True:
                errors.append(f"{prefix}: invalidated evidence cannot support EVIDENCE_VERIFIED")
            if isinstance(domain, str) and domain in DOMAINS:
                eligible = _closure_evidence(domain)
                if not evidence_kinds.intersection(eligible):
                    required = " or ".join(sorted(eligible))
                    if mode == "closure":
                        errors.append(f"{prefix}: closure mode requires INCONCLUSIVE when EVIDENCE_VERIFIED {domain} claims lack {required} evidence")
                    else:
                        errors.append(f"{prefix}: EVIDENCE_VERIFIED {domain} claims require {required} evidence")
        if state == "REFUTED" and isinstance(domain, str) and domain in DOMAINS:
            if mode == "closure" and unresolved_disagreement:
                errors.append(f"{prefix}: unresolved disagreement in closure mode requires INCONCLUSIVE")
            eligible = _refutation_evidence(domain)
            if not counterevidence_kinds.intersection(eligible):
                required = " or ".join(sorted(eligible))
                if mode == "closure":
                    errors.append(f"{prefix}: closure mode requires INCONCLUSIVE when REFUTED {domain} claims lack current {required} counterevidence with supports false")
                else:
                    errors.append(f"{prefix}: REFUTED {domain} claims require current {required} counterevidence with supports false")
        if severity in {"high", "critical"} and state in {"EVIDENCE_VERIFIED", "REFUTED"}:
            for required_role in ("verifier-skeptic", "verifier-adjudicator"):
                if required_role not in roles:
                    errors.append(f"{prefix}: {severity} closure requires {required_role}")
        if mode == "closure" and state in {"CANDIDATE", "LLM_SUPPORTED"}:
            errors.append(f"{prefix}: closure mode requires INCONCLUSIVE instead of {state}")
        if mode == "triage" and state in {"EVIDENCE_VERIFIED", "REFUTED"}:
            errors.append(f"{prefix}: triage mode permits only CANDIDATE, LLM_SUPPORTED, or INCONCLUSIVE instead of {state}")

    return errors


def _block(reason: str) -> tuple[int, dict[str, object]]:
    return 0, {"decision": "block", "reason": reason}


def _resolve_under_cwd(cwd: Path, ledger_reference: object) -> tuple[Path | None, str | None]:
    if not _nonempty_string(ledger_reference):
        return None, "activation marker ledger must be a nonempty string"
    candidate = Path(str(ledger_reference))
    if candidate.is_absolute():
        return None, "ledger path traversal is not permitted"
    resolved = (cwd / candidate).resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        return None, "ledger path traversal is not permitted"
    return resolved, None


def _message_references_ledger(message: object, cwd: Path, ledger_path: Path, raw_reference: str) -> bool:
    if not isinstance(message, str):
        return False
    relative = ledger_path.relative_to(cwd).as_posix()
    normalized_message = message.replace("\\", "/")
    return any(reference in normalized_message for reference in (raw_reference.replace("\\", "/"), relative, str(ledger_path).replace("\\", "/")))


def run_hook(payload: dict[str, object]) -> tuple[int, dict[str, object] | None]:
    """Validate the active ledger and return a Stop-hook decision.

    A missing activation marker is deliberately a no-op. A successful active
    ledger is permitted only when the final assistant message names that ledger.
    """

    cwd_value = payload.get("cwd")
    if not _nonempty_string(cwd_value):
        return _block("hook payload cwd must be a nonempty string")
    cwd = Path(str(cwd_value)).resolve()
    if not cwd.is_dir():
        return _block("hook payload cwd must name an existing directory")
    marker = cwd / ".verification" / "active.json"
    if not marker.exists():
        return 0, None
    try:
        activation = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _block(f"cannot read activation marker: {exc}")
    activation_data = _as_dict(activation)
    if activation_data is None:
        return _block("activation marker must be a JSON object")
    raw_ledger = activation_data.get("ledger")
    ledger_path, path_error = _resolve_under_cwd(cwd, raw_ledger)
    if path_error is not None or ledger_path is None:
        return _block(path_error or "invalid ledger path")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _block("referenced ledger does not exist")
    except (OSError, json.JSONDecodeError) as exc:
        return _block(f"cannot read referenced ledger: {exc}")
    if not isinstance(ledger, dict):
        return _block("referenced ledger must be a JSON object")
    errors = validate_ledger(ledger)
    if errors:
        return _block("; ".join(errors))
    if not _message_references_ledger(payload.get("last_assistant_message"), cwd, ledger_path, str(raw_ledger)):
        return _block("final assistant message must reference the validated ledger")
    try:
        marker.unlink()
    except OSError as exc:
        return _block(f"cannot remove activation marker: {exc}")
    return 0, None


def _candidate_ledger(mode: str) -> dict[str, object]:
    return {"schema_version": "1.0", "mode": mode, "artifact_revision": "UNSPECIFIED", "claims": []}


def _command_start(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    ledger_path, error = _resolve_under_cwd(cwd, args.ledger)
    if error is not None or ledger_path is None:
        print(error or "invalid ledger path", file=sys.stderr)
        return 2
    marker = cwd / ".verification" / "active.json"
    if ledger_path.exists() or marker.exists():
        print("verification activation or ledger already exists", file=sys.stderr)
        return 2
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(_candidate_ledger(args.mode), indent=2) + "\n", encoding="utf-8")
    marker.write_text(json.dumps({"ledger": ledger_path.relative_to(cwd).as_posix()}, indent=2) + "\n", encoding="utf-8")
    print(ledger_path.relative_to(cwd).as_posix())
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ledger: cannot read {path}: {exc}")
        return 1
    if not isinstance(data, dict):
        print("ledger: expected an object")
        return 1
    errors = validate_ledger(data)
    if errors:
        print("\n".join(errors))
        return 1
    return 0


def _command_hook() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        _, response = _block(f"invalid hook JSON: {exc}")
        print(json.dumps(response))
        return 0
    if not isinstance(payload, dict):
        _, response = _block("hook JSON must be an object")
        print(json.dumps(response))
        return 0
    _, response = run_hook(payload)
    print(json.dumps(response or {}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start", help="create an active candidate ledger")
    start_parser.add_argument("--cwd", default=".")
    start_parser.add_argument("--ledger", default=".verification/ledger.json")
    start_parser.add_argument("--mode", choices=sorted(MODES), default="closure")
    validate_parser = subparsers.add_parser("validate", help="validate a ledger file")
    validate_parser.add_argument("ledger")
    subparsers.add_parser("hook", help="read Stop-hook JSON from standard input")
    args = parser.parse_args(argv)
    if args.command == "start":
        return _command_start(args)
    if args.command == "validate":
        return _command_validate(args)
    return _command_hook()


if __name__ == "__main__":
    raise SystemExit(main())
