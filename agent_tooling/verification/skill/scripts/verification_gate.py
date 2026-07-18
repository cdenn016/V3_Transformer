"""Deterministic validation and opt-in Stop-hook gate for claim ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CLAIM_STATES = frozenset({"CANDIDATE", "LLM_SUPPORTED", "EVIDENCE_VERIFIED", "REFUTED", "INCONCLUSIVE"})
DOMAINS = frozenset({"code", "experiment", "mathematics", "evidence", "research", "source", "general"})
SEVERITIES = frozenset({"low", "medium", "high", "critical"})
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

_ROOT_FIELDS = frozenset({"schema_version", "artifact_revision", "claims"})
_CLAIM_FIELDS = frozenset(
    {
        "id",
        "domain",
        "statement",
        "severity",
        "state",
        "artifact_revision",
        "criteria",
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


def _as_dict(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


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
    return frozenset({"primary_source", "reproduced_output"})


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
    revision = root.get("artifact_revision")
    if not _nonempty_string(revision):
        errors.append("ledger: artifact_revision must be a nonempty string")
    claims_value = root.get("claims")
    if not isinstance(claims_value, list):
        return errors + ["ledger: claims must be a list"]

    claim_items: list[tuple[str, int, object]] = []
    for index, claim in enumerate(claims_value):
        claim_dict = _as_dict(claim)
        claim_id = claim_dict.get("id") if claim_dict is not None else None
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
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 20:
                    errors.append(f"{item_prefix}: score must be a number from 0 to 20")

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
            if claim.get("evidence_invalidated") is True:
                errors.append(f"{prefix}: invalidated evidence cannot support EVIDENCE_VERIFIED")
            if isinstance(domain, str) and domain in DOMAINS:
                eligible = _closure_evidence(domain)
                if not evidence_kinds.intersection(eligible):
                    required = " or ".join(sorted(eligible))
                    errors.append(f"{prefix}: EVIDENCE_VERIFIED {domain} claims require {required} evidence")
        if state == "REFUTED" and isinstance(domain, str) and domain in DOMAINS:
            eligible = _refutation_evidence(domain)
            if not counterevidence_kinds.intersection(eligible):
                required = " or ".join(sorted(eligible))
                errors.append(f"{prefix}: REFUTED {domain} claims require current {required} counterevidence with supports false")
        if severity in {"high", "critical"} and state in {"EVIDENCE_VERIFIED", "REFUTED"}:
            for required_role in ("verifier-skeptic", "verifier-adjudicator"):
                if required_role not in roles:
                    errors.append(f"{prefix}: {severity} closure requires {required_role}")

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


def _candidate_ledger() -> dict[str, object]:
    return {"schema_version": "1.0", "artifact_revision": "UNSPECIFIED", "claims": []}


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
    ledger_path.write_text(json.dumps(_candidate_ledger(), indent=2) + "\n", encoding="utf-8")
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
