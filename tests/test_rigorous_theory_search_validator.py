"""Behavioral tests for the version-one rigorous-theory-search run validator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from agent_tooling.rigorous_theory_search.skill.scripts.validate_run import main, validate_run


SCHEMA = "rigorous-theory-search/v1"
TARGET_STATEMENT = "For every permitted input, the construction succeeds."
TARGET_QUANTIFIERS = "for all permitted inputs"
FINAL_REPORT_HEADINGS = (
    "# Rigorous theory search report",
    "## Frozen contract",
    "## Terminal status",
    "## Certificate",
    "## Strongest verified result",
    "## Dependency closure",
    "## Independent reconstruction",
    "## Oracle erasure",
    "## Unresolved obligations",
    "## Scope and limitations",
)
PROOF_BYTES = b"A complete formal proof artifact.\n"
ATTACK_BYTES = b"Attack A checks the target's boundary case and is answered at proof step 4.\n"
RECONSTRUCTION_BYTES = b"An independent reconstruction derives the target from the frozen premises.\n"
ERASURE_BYTES = b"Oracle erasure removes the search prior and reconstructs every dependency.\n"
ARTIFACTS = (
    "problem-contract.json",
    "approach-registry.json",
    "claim-ledger.json",
    "dependency-dag.json",
    "counterexample-register.md",
    "construction-or-strongest-theorem.md",
    "adversarial-report.json",
    "release.json",
    "final-report.md",
)


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def canonical_target_digest(target: dict[str, object]) -> str:
    encoded = json.dumps(
        target, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadata_line(contract_id: str, target_digest: str) -> str:
    payload = json.dumps(
        {
            "contract_id": contract_id,
            "schema_version": SCHEMA,
            "target_digest": target_digest,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"<!-- rigorous-theory-search-metadata {payload} -->"


def write_json(run_dir: Path, name: str, payload: dict[str, object]) -> None:
    (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def evidence(
    evidence_id: str = "proof-1", kind: str = "FORMAL_PROOF", *,
    artifact_path: str = "proofs/proof.txt", supports: bool | None = None,
) -> dict[str, object]:
    if supports is None:
        supports = kind not in {"COUNTEREXAMPLE", "NONEXISTENCE_PROOF"}
    return {
        "id": evidence_id,
        "kind": kind,
        "supports": supports,
        "artifact_path": artifact_path,
        "artifact_sha256": sha256_bytes(PROOF_BYTES),
        "scope": "the stated target",
        "side_conditions": [],
    }


def claim(
    claim_id: str = "target", *, state: str = "EVIDENCE_VERIFIED",
    kind: str = "MATHEMATICAL", evidence_ids: list[str] | None = None,
    assumptions: list[str] | None = None, bridges: list[str] | None = None,
    target_digest: str = "",
) -> dict[str, object]:
    return {
        "id": claim_id,
        "statement": TARGET_STATEMENT,
        "quantifiers": TARGET_QUANTIFIERS,
        "kind": kind,
        "target_digest": target_digest,
        "state": state,
        "assumption_ids": assumptions or [],
        "evidence_ids": evidence_ids if evidence_ids is not None else ["proof-1"],
        "bridge_premise_ids": bridges or [],
        "falsifier": "a counterexample within the stated domain",
    }


def make_valid_run(
    tmp_path: Path, *, status: str | None = "COMPLETE_AFFIRMATIVE",
    target_quantifier: str = "UNIVERSAL",
    negative_certificate_kind: str | None = None,
    target_statement: str = TARGET_STATEMENT,
    target_kind: str = "MATHEMATICAL",
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    target = {
        "statement": target_statement,
        "kind": target_kind,
        "quantifier_class": target_quantifier,
        "quantifiers": TARGET_QUANTIFIERS,
        "domains": ["X"], "codomains": ["Y"], "regularity": "continuous",
        "measures": ["Lebesgue"], "boundary_conditions": ["compact support"],
        "symmetries": ["identity"], "equivalence": "equality",
        "premises": ["P"], "modeling_postulates": [], "search_priors": [],
        "permitted_theorems": ["T"],
        "negative_certificate_kind": negative_certificate_kind or (
            "COUNTEREXAMPLE" if target_quantifier == "UNIVERSAL" else "NONEXISTENCE_PROOF"
        ),
        "falsification_criterion": "a counterexample", "literature_policy": "primary sources",
    }
    target_digest = canonical_target_digest(target)
    contract_id = f"contract-sha256-{target_digest}"
    common = {
        "schema_version": SCHEMA,
        "contract_id": contract_id,
        "target_digest": target_digest,
    }
    (run_dir / "proofs").mkdir()
    (run_dir / "proofs" / "proof.txt").write_bytes(PROOF_BYTES)
    (run_dir / "proofs" / "attack.md").write_bytes(ATTACK_BYTES)
    (run_dir / "proofs" / "independent-reconstruction.md").write_bytes(RECONSTRUCTION_BYTES)
    (run_dir / "proofs" / "oracle-erasure.md").write_bytes(ERASURE_BYTES)
    write_json(run_dir, "problem-contract.json", {
        **common, "target": target,
    })
    write_json(run_dir, "approach-registry.json", {
        **common,
        "mechanism_families": [{
            "family_id": "family-001", "core_mechanism": "direct construction",
            "target_obligation_ids": ["target"],
            "representation": "a typed map from X to Y",
            "invariant_or_obstruction": "the map preserves I",
            "obligations": ["prove preservation of I"],
            "bridge": "lemma L establishes the interface",
            "failure_test": "an input in X whose image violates I",
            "verified_results": ["lemma L"], "open_gaps": [],
            "novelty_fingerprint": "direct invariant-preserving construction",
            "disposition": "certified",
        }],
    })
    write_json(run_dir, "claim-ledger.json", {
        **common, "assumptions": [], "evidence": [evidence()],
        "claims": [
            claim(target_digest=target_digest, kind=target_kind)
            | {"statement": target_statement},
        ],
    })
    write_json(run_dir, "dependency-dag.json", {
        **common, "edges": [],
    })
    write_json(run_dir, "adversarial-report.json", {
        **common,
        "attacks": [{
            "id": "attack-1",
            "claim_ids": ["target"],
            "attack": "Test the boundary of X where preservation of I is least direct.",
            "disposition": "REJECTED",
            "response": "Proof step 4 handles the boundary without a stronger premise.",
            "artifact_path": "proofs/attack.md",
            "artifact_sha256": sha256_bytes(ATTACK_BYTES),
        }],
        "independent_reconstruction": {
            "completed": True,
            "claim_ids": ["target"],
            "method": "Re-derive the construction from the frozen contract and dependency DAG.",
            "result": "PASS",
            "conclusion": "The reconstruction reaches the exact target statement.",
            "artifact_path": "proofs/independent-reconstruction.md",
            "artifact_sha256": sha256_bytes(RECONSTRUCTION_BYTES),
        },
        "oracle_erasure": {
            "completed": True,
            "claim_ids": ["target"],
            "method": "Remove search priors, rescan every artifact, and recheck dependency closure.",
            "result": "PASS",
            "conclusion": "The target remains supported without a search prior.",
            "artifact_path": "proofs/oracle-erasure.md",
            "artifact_sha256": sha256_bytes(ERASURE_BYTES),
        },
    })
    write_json(run_dir, "release.json", {
        **common,
        "checkpoint": "release", "target_claim": "target", "terminal_status": status,
        "certificate_claim": "target", "strongest_result": "target", "unresolved_obligations": [],
    })
    header = metadata_line(contract_id, target_digest)
    (run_dir / "counterexample-register.md").write_text(
        f"{header}\n# Counterexample register\n\n"
        "Attack A tested the boundary of X and was rejected by proof step 4.\n",
        encoding="utf-8",
    )
    (run_dir / "construction-or-strongest-theorem.md").write_text(
        f"{header}\n# Construction or strongest theorem\n\n"
        "The typed construction maps every x in X to a unique y in Y and preserves I.\n",
        encoding="utf-8",
    )
    report_sections = "\n\n".join(
        f"{heading}\n\nSubstantive release content for {heading.lstrip('# ').lower()}."
        for heading in FINAL_REPORT_HEADINGS
    )
    (run_dir / "final-report.md").write_text(
        f"{header}\n{report_sections}\n", encoding="utf-8",
    )
    return run_dir


def load_json(run_dir: Path, name: str) -> dict[str, object]:
    return json.loads((run_dir / name).read_text(encoding="utf-8"))


def replace_json(run_dir: Path, name: str, mutate: object) -> None:
    payload = load_json(run_dir, name)
    mutate(payload)
    write_json(run_dir, name, payload)


def bound_claim(run_dir: Path, claim_id: str = "target", **overrides: object) -> dict[str, object]:
    contract = load_json(run_dir, "problem-contract.json")
    target = contract["target"]
    assert isinstance(target, dict)
    record = claim(
        claim_id,
        target_digest=str(contract["target_digest"]),
        kind=str(target["kind"]),
    )
    record.update({
        "statement": target["statement"],
        "quantifiers": target["quantifiers"],
        **overrides,
    })
    return record


def set_adversarial_coverage(run_dir: Path, claim_ids: list[str]) -> None:
    def mutate(payload: dict[str, object]) -> None:
        attacks = payload["attacks"]
        assert isinstance(attacks, list) and isinstance(attacks[0], dict)
        attacks[0]["claim_ids"] = claim_ids
        for field in ("independent_reconstruction", "oracle_erasure"):
            record = payload[field]
            assert isinstance(record, dict)
            record["claim_ids"] = claim_ids

    replace_json(run_dir, "adversarial-report.json", mutate)


def test_valid_complete_affirmative_run_has_no_errors(tmp_path: Path) -> None:
    assert validate_run(make_valid_run(tmp_path), mode="release") == []


@pytest.mark.parametrize("status", ["COMPLETE_NEGATIVE", "INCONCLUSIVE"])
def test_valid_terminal_statuses_have_no_errors(tmp_path: Path, status: str) -> None:
    run_dir = make_valid_run(tmp_path, status=status)
    if status == "COMPLETE_NEGATIVE":
        replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"evidence": [evidence("counter", "COUNTEREXAMPLE")], "claims": [bound_claim(run_dir, state="REFUTED", evidence_ids=["counter"])]}))
        replace_json(run_dir, "release.json", lambda p: p.update({"certificate_claim": "target"}))
    else:
        replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"claims": [bound_claim(run_dir, state="INCONCLUSIVE")]}))
        replace_json(run_dir, "release.json", lambda p: p.update({"strongest_result": "partial theorem", "unresolved_obligations": ["prove lemma"]}))
    assert validate_run(run_dir, mode="release") == []


def test_checkpoint_permits_null_terminal_status(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path, status=None)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"claims": [bound_claim(run_dir, state="CANDIDATE")]}))
    assert validate_run(run_dir) == []


@pytest.mark.parametrize("target_claim", [None, "missing"])
def test_checkpoint_requires_a_bound_target_claim(
    tmp_path: Path, target_claim: object,
) -> None:
    run_dir = make_valid_run(tmp_path, status=None)
    replace_json(
        run_dir,
        "release.json",
        lambda payload: payload.update({"target_claim": target_claim}),
    )
    assert any(
        "release.json: target_claim must name a claim" in error
        for error in validate_run(run_dir, mode="checkpoint")
    )


@pytest.mark.parametrize("status", sorted({"COMPLETE_AFFIRMATIVE", "COMPLETE_NEGATIVE", "INCONCLUSIVE"}))
def test_checkpoint_rejects_every_nonnull_terminal_status(tmp_path: Path, status: str) -> None:
    run_dir = make_valid_run(tmp_path, status=status)
    assert any(
        "checkpoint mode requires terminal_status null" in error
        for error in validate_run(run_dir, mode="checkpoint")
    )


@pytest.mark.parametrize("name", ARTIFACTS)
def test_every_required_artifact_must_exist_and_markdown_must_be_nonempty(tmp_path: Path, name: str) -> None:
    run_dir = make_valid_run(tmp_path)
    (run_dir / name).unlink()
    errors = validate_run(run_dir, mode="release")
    assert any(name in error for error in errors)


def test_rejects_unsupported_schema_and_mismatched_contract_ids(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "approach-registry.json", lambda p: p.update({"schema_version": "other/v1", "contract_id": "wrong"}))
    errors = validate_run(run_dir, mode="release")
    assert any("schema_version" in error for error in errors)
    assert any("contract_id" in error for error in errors)


def test_recomputes_full_target_digest_and_rejects_target_mutation(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir,
        "problem-contract.json",
        lambda payload: payload["target"].update({"boundary_conditions": ["changed boundary"]}),
    )
    errors = validate_run(run_dir, mode="release")
    assert any("canonical target digest" in error for error in errors)


@pytest.mark.parametrize("field", ["statement", "kind", "quantifiers", "target_digest"])
def test_target_claim_must_match_the_frozen_target_exactly(tmp_path: Path, field: str) -> None:
    run_dir = make_valid_run(tmp_path)
    replacement: object = "different"
    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload["claims"][0].update({field: replacement}),
    )
    errors = validate_run(run_dir, mode="release")
    assert any(f"target claim target: mismatched {field}" in error for error in errors)


def test_cross_contract_json_and_markdown_swaps_fail(tmp_path: Path) -> None:
    first = make_valid_run(tmp_path / "first")
    second = make_valid_run(
        tmp_path / "second",
        target_statement="There exists a permitted input with a successful construction.",
    )
    (first / "approach-registry.json").write_bytes(
        (second / "approach-registry.json").read_bytes(),
    )
    (first / "counterexample-register.md").write_bytes(
        (second / "counterexample-register.md").read_bytes(),
    )
    errors = validate_run(first, mode="release")
    assert any("approach-registry.json: mismatched contract_id" in error for error in errors)
    assert any("counterexample-register.md: mismatched contract_id" in error for error in errors)


def test_every_json_and_markdown_artifact_requires_target_binding(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "dependency-dag.json", lambda payload: payload.pop("target_digest"))
    final_report = run_dir / "final-report.md"
    final_report.write_text(
        "\n".join(final_report.read_text(encoding="utf-8").splitlines()[1:]),
        encoding="utf-8",
    )
    errors = validate_run(run_dir, mode="release")
    assert any("dependency-dag.json: mismatched target_digest" in error for error in errors)
    assert any("final-report.md: missing metadata header" in error for error in errors)


def test_rejects_missing_target_and_mechanism_family_schema_fields(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "problem-contract.json", lambda p: p["target"].pop("quantifier_class"))
    replace_json(run_dir, "approach-registry.json", lambda p: p["mechanism_families"][0].pop("failure_test"))
    errors = validate_run(run_dir, mode="release")
    assert any("target: missing quantifier_class" in error for error in errors)
    assert any("mechanism family 0: missing failure_test" in error for error in errors)


@pytest.mark.parametrize(
    "field,value",
    [
        ("statement", ""),
        ("kind", 7),
        ("quantifiers", []),
        ("domains", []),
        ("codomains", [""]),
        ("regularity", "  "),
        ("premises", "P"),
        ("falsification_criterion", ""),
    ],
)
def test_target_fields_are_typed_and_required_nonempty(
    tmp_path: Path, field: str, value: object,
) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir,
        "problem-contract.json",
        lambda payload: payload["target"].update({field: value}),
    )
    assert any(
        f"target: {field}" in error for error in validate_run(run_dir, mode="release")
    )


@pytest.mark.parametrize("value", [None, "", "NUMERICAL", 7])
def test_target_requires_explicit_valid_negative_certificate_kind(
    tmp_path: Path, value: object,
) -> None:
    run_dir = make_valid_run(tmp_path, target_quantifier="MIXED")
    replace_json(
        run_dir,
        "problem-contract.json",
        lambda payload: payload["target"].update({"negative_certificate_kind": value}),
    )
    assert any(
        "negative_certificate_kind" in error
        for error in validate_run(run_dir, mode="release")
    )


@pytest.mark.parametrize("certificate", ["COUNTEREXAMPLE", "NONEXISTENCE_PROOF"])
def test_mixed_target_uses_its_explicit_negative_certificate_kind(
    tmp_path: Path, certificate: str,
) -> None:
    run_dir = make_valid_run(
        tmp_path,
        status="COMPLETE_NEGATIVE",
        target_quantifier="MIXED",
        negative_certificate_kind=certificate,
    )
    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload.update({
            "evidence": [evidence("negative", certificate)],
            "claims": [
                bound_claim(
                    run_dir,
                    state="REFUTED",
                    evidence_ids=["negative"],
                ),
            ],
        }),
    )
    assert validate_run(run_dir, mode="release") == []


@pytest.mark.parametrize("field", ["family_id", "core_mechanism", "target_obligation_ids"])
def test_rejects_missing_stable_mechanism_family_identity_fields(
    tmp_path: Path, field: str
) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir, "approach-registry.json",
        lambda payload: payload["mechanism_families"][0].pop(field),
    )
    assert f"mechanism family 0: missing {field}" in validate_run(run_dir, mode="release")


def test_rejects_duplicate_or_malformed_mechanism_family_identity(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir, "approach-registry.json",
        lambda payload: payload["mechanism_families"].extend([
            {
                **copy.deepcopy(payload["mechanism_families"][0]),
                "core_mechanism": "",
                "target_obligation_ids": "target",
            }
        ]),
    )
    errors = validate_run(run_dir, mode="release")
    assert "duplicate mechanism family ID: family-001" in errors
    assert "mechanism family 1: core_mechanism must be a nonempty string" in errors
    assert "mechanism family 1: target_obligation_ids must be a list of nonempty strings" in errors


def test_rejects_duplicate_and_unknown_references(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({
        "assumptions": [{"id": "a", "statement": "A", "kind": "MODELING_POSTULATE"}, {"id": "a", "statement": "A", "kind": "MODELING_POSTULATE"}],
        "claims": [claim("target", assumptions=["missing"], evidence_ids=["missing"], bridges=["missing"]) , claim("target")],
    }))
    replace_json(run_dir, "dependency-dag.json", lambda p: p.update({"edges": [{"from": "target", "to": "missing"}]}))
    errors = validate_run(run_dir, mode="release")
    assert any("duplicate assumption ID" in error for error in errors)
    assert any("duplicate claim ID" in error for error in errors)
    assert any("unknown assumption" in error for error in errors)
    assert any("unknown evidence" in error for error in errors)
    assert any("unknown bridge" in error for error in errors)
    assert any("unknown dependency" in error for error in errors)


def test_rejects_canonical_dependency_cycle_and_unverified_transitive_ancestor(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"claims": [claim("a"), claim("b", state="LLM_SUPPORTED")]}))
    replace_json(run_dir, "dependency-dag.json", lambda p: p.update({"edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}]}))
    replace_json(run_dir, "release.json", lambda p: p.update({"target_claim": "a", "certificate_claim": "a", "strongest_result": "a"}))
    errors = validate_run(run_dir, mode="release")
    assert "dependency cycle: a -> b -> a" in errors
    assert any("ancestor b" in error for error in errors)


@pytest.mark.parametrize("status", sorted({"COMPLETE_AFFIRMATIVE", "COMPLETE_NEGATIVE", "INCONCLUSIVE"}))
@pytest.mark.parametrize("intermediate_state", ["CANDIDATE", "LLM_SUPPORTED"])
def test_every_terminal_status_rejects_intermediate_transitive_dependencies(
    tmp_path: Path, status: str, intermediate_state: str,
) -> None:
    run_dir = make_valid_run(tmp_path, status=status)
    target_state = {
        "COMPLETE_AFFIRMATIVE": "EVIDENCE_VERIFIED",
        "COMPLETE_NEGATIVE": "REFUTED",
        "INCONCLUSIVE": "INCONCLUSIVE",
    }[status]
    target_evidence_kind = (
        "COUNTEREXAMPLE" if status == "COMPLETE_NEGATIVE" else "FORMAL_PROOF"
    )
    target_evidence_id = "counter" if status == "COMPLETE_NEGATIVE" else "proof-1"
    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload.update({
            "evidence": [
                evidence(target_evidence_id, target_evidence_kind),
                evidence("ancestor-proof", "FORMAL_PROOF"),
            ],
            "claims": [
                bound_claim(
                    run_dir,
                    state=target_state,
                    evidence_ids=[target_evidence_id],
                ),
                claim(
                    "ancestor",
                    state=intermediate_state,
                    evidence_ids=["ancestor-proof"],
                ),
            ],
        }),
    )
    replace_json(
        run_dir,
        "dependency-dag.json",
        lambda payload: payload.update({"edges": [{"from": "target", "to": "ancestor"}]}),
    )
    if status == "INCONCLUSIVE":
        replace_json(
            run_dir,
            "release.json",
            lambda payload: payload.update({
                "strongest_result": "A dependency-local lemma.",
                "unresolved_obligations": ["Close the target obstruction."],
            }),
        )
    set_adversarial_coverage(run_dir, ["target", "ancestor"])
    assert any(
        f"dependency ancestor {intermediate_state}" in error
        for error in validate_run(run_dir, mode="release")
    )


@pytest.mark.parametrize("status", ["COMPLETE_NEGATIVE", "INCONCLUSIVE"])
def test_negative_and_inconclusive_release_accept_closed_dependency_ancestors(
    tmp_path: Path, status: str,
) -> None:
    run_dir = make_valid_run(tmp_path, status=status)
    target_state = "REFUTED" if status == "COMPLETE_NEGATIVE" else "INCONCLUSIVE"
    target_kind = "COUNTEREXAMPLE" if status == "COMPLETE_NEGATIVE" else "FORMAL_PROOF"
    target_evidence_id = "counter" if status == "COMPLETE_NEGATIVE" else "proof-1"
    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload.update({
            "evidence": [
                evidence(target_evidence_id, target_kind),
                evidence("ancestor-proof", "FORMAL_PROOF"),
            ],
            "claims": [
                bound_claim(
                    run_dir,
                    state=target_state,
                    evidence_ids=[target_evidence_id],
                ),
                bound_claim(
                    run_dir,
                    "ancestor",
                    evidence_ids=["ancestor-proof"],
                ),
            ],
        }),
    )
    replace_json(
        run_dir,
        "dependency-dag.json",
        lambda payload: payload.update({"edges": [{"from": "target", "to": "ancestor"}]}),
    )
    if status == "INCONCLUSIVE":
        replace_json(
            run_dir,
            "release.json",
            lambda payload: payload.update({
                "strongest_result": "The dependency ancestor is proved.",
                "unresolved_obligations": ["Close the target obstruction."],
            }),
        )
    set_adversarial_coverage(run_dir, ["target", "ancestor"])
    assert validate_run(run_dir, mode="release") == []


@pytest.mark.parametrize("forged_state", ["REFUTED", "INCONCLUSIVE"])
def test_complete_negative_rejects_nonverified_certificate_dependencies(
    tmp_path: Path, forged_state: str,
) -> None:
    run_dir = make_valid_run(tmp_path, status="COMPLETE_NEGATIVE")
    ancestor_evidence = "ancestor-counter" if forged_state == "REFUTED" else "ancestor-proof"
    ancestor_kind = "COUNTEREXAMPLE" if forged_state == "REFUTED" else "FORMAL_PROOF"
    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload.update({
            "evidence": [
                evidence("target-counter", "COUNTEREXAMPLE"),
                evidence(ancestor_evidence, ancestor_kind),
            ],
            "claims": [
                bound_claim(
                    run_dir,
                    state="REFUTED",
                    evidence_ids=["target-counter"],
                ),
                claim(
                    "ancestor",
                    state=forged_state,
                    evidence_ids=[ancestor_evidence],
                ),
            ],
        }),
    )
    replace_json(
        run_dir,
        "dependency-dag.json",
        lambda payload: payload.update({"edges": [{"from": "target", "to": "ancestor"}]}),
    )
    set_adversarial_coverage(run_dir, ["target", "ancestor"])
    assert any(
        f"COMPLETE_NEGATIVE dependency ancestor {forged_state}: ancestor must be EVIDENCE_VERIFIED"
        in error
        for error in validate_run(run_dir, mode="release")
    )


def test_rejects_verified_universal_mathematics_with_only_numerical_evidence(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"evidence": [evidence(kind="NUMERICAL")]}))
    assert any("mathematical claim target" in error for error in validate_run(run_dir, mode="release"))


@pytest.mark.parametrize("where", ["assumption", "evidence", "dependency"])
def test_rejects_search_prior_leakage_outside_target_priors(tmp_path: Path, where: str) -> None:
    run_dir = make_valid_run(tmp_path)
    if where == "assumption":
        replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"assumptions": [{"id": "a", "statement": "SEARCH_PRIOR_AFFIRMATIVE", "kind": "MODELING_POSTULATE"}]}))
    elif where == "evidence":
        replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"evidence": [dict(evidence(), scope="SEARCH_PRIOR_AFFIRMATIVE")]}))
    else:
        replace_json(run_dir, "dependency-dag.json", lambda p: p.update({"edges": [{"from": "SEARCH_PRIOR_AFFIRMATIVE", "to": "target"}]}))
    assert any("SEARCH_PRIOR_AFFIRMATIVE" in error for error in validate_run(run_dir, mode="release"))


def test_physical_interpretation_requires_declared_bridge_and_direct_eligible_evidence(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path, target_kind="PHYSICAL_INTERPRETATION")
    errors = validate_run(run_dir, mode="release")
    assert any("physical interpretation target" in error for error in errors)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({
        "assumptions": [{"id": "bridge", "statement": "operational bridge", "kind": "OPERATIONAL_IDENTIFICATION"}],
        "evidence": [evidence("source", "PRIMARY_SOURCE")],
        "claims": [bound_claim(
            run_dir,
            evidence_ids=["source"],
            bridge_premise_ids=["bridge"],
        )],
    }))
    assert validate_run(run_dir, mode="release") == []


@pytest.mark.parametrize(
    "status,state,evidence_kind",
    [
        ("COMPLETE_AFFIRMATIVE", "EVIDENCE_VERIFIED", "NUMERICAL"),
        ("COMPLETE_NEGATIVE", "REFUTED", "COUNTEREXAMPLE"),
    ],
)
def test_physical_interpretation_accepts_documented_evidence_with_bridge(
    tmp_path: Path,
    status: str,
    state: str,
    evidence_kind: str,
) -> None:
    run_dir = make_valid_run(
        tmp_path,
        status=status,
        target_kind="PHYSICAL_INTERPRETATION",
    )
    replace_json(run_dir, "claim-ledger.json", lambda payload: payload.update({
        "assumptions": [{
            "id": "bridge",
            "statement": "The measured quantity operationalizes the physical observable.",
            "kind": "OPERATIONAL_IDENTIFICATION",
        }],
        "evidence": [evidence("physical-evidence", evidence_kind)],
        "claims": [bound_claim(
            run_dir,
            state=state,
            evidence_ids=["physical-evidence"],
            bridge_premise_ids=["bridge"],
        )],
    }))
    assert validate_run(run_dir, mode="release") == []


@pytest.mark.parametrize("quantifier, certificate", [("UNIVERSAL", "COUNTEREXAMPLE"), ("EXISTENTIAL", "NONEXISTENCE_PROOF")])
def test_negative_release_requires_quantifier_appropriate_certificate(tmp_path: Path, quantifier: str, certificate: str) -> None:
    run_dir = make_valid_run(tmp_path, status="COMPLETE_NEGATIVE", target_quantifier=quantifier)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"evidence": [evidence("certificate", certificate)], "claims": [bound_claim(run_dir, state="REFUTED", evidence_ids=["certificate"])]}))
    assert validate_run(run_dir, mode="release") == []
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"evidence": [evidence("wrong", "COUNTEREXAMPLE" if certificate == "NONEXISTENCE_PROOF" else "NUMERICAL")], "claims": [claim("target", state="REFUTED", evidence_ids=["wrong"])]}))
    assert any("negative certificate" in error for error in validate_run(run_dir, mode="release"))


def test_inconclusive_requires_strongest_result_and_unresolved_obligations(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path, status="INCONCLUSIVE")
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"claims": [claim("target", state="INCONCLUSIVE")]}))
    replace_json(run_dir, "release.json", lambda p: p.update({"strongest_result": "", "unresolved_obligations": []}))
    errors = validate_run(run_dir, mode="release")
    assert any("strongest_result" in error for error in errors)
    assert any("unresolved_obligations" in error for error in errors)


def test_release_rejects_empty_and_scaffold_mechanism_portfolios(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir,
        "approach-registry.json",
        lambda payload: payload.update({"mechanism_families": []}),
    )
    assert any(
        "release requires a nonempty mechanism portfolio" in error
        for error in validate_run(run_dir, mode="release")
    )

    replace_json(
        run_dir,
        "approach-registry.json",
        lambda payload: payload.update({
            "mechanism_families": [{
                "family_id": "family-001",
                "core_mechanism": "Candidate mechanism to be specified.",
                "target_obligation_ids": ["target"],
                "representation": "Candidate representation to be specified.",
                "invariant_or_obstruction": "Candidate invariant or obstruction to be investigated.",
                "obligations": ["State the precise proof or counterexample obligation."],
                "bridge": "No bridge result has been established.",
                "failure_test": "Specify a falsifying instance or failed obligation.",
                "verified_results": [],
                "open_gaps": ["All mathematical obligations remain open."],
                "novelty_fingerprint": "Candidate only; novelty is unassessed.",
                "disposition": "open",
            }],
        }),
    )
    assert any(
        "mechanism portfolio contains scaffold sentinel content" in error
        for error in validate_run(run_dir, mode="release")
    )


@pytest.mark.parametrize("field", ["statement", "quantifiers", "target_digest", "falsifier"])
def test_complete_release_rejects_malformed_dependency_claim_fields(
    tmp_path: Path,
    field: str,
) -> None:
    run_dir = make_valid_run(tmp_path)
    ancestor = bound_claim(run_dir, "ancestor")
    ancestor[field] = ""
    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload["claims"].append(ancestor),
    )
    replace_json(
        run_dir,
        "dependency-dag.json",
        lambda payload: payload.update({"edges": [{"from": "target", "to": "ancestor"}]}),
    )
    set_adversarial_coverage(run_dir, ["target", "ancestor"])
    expected = (
        "claim ancestor: mismatched target_digest"
        if field == "target_digest"
        else f"claim ancestor: {field}"
    )
    assert any(expected in error for error in validate_run(run_dir, mode="release"))


@pytest.mark.parametrize("disposition", ["SUSTAINED", "PARTIALLY_SUSTAINED"])
def test_sustained_attack_blocks_complete_release(
    tmp_path: Path,
    disposition: str,
) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir,
        "adversarial-report.json",
        lambda payload: payload["attacks"][0].update({
            "disposition": disposition,
            "response": "The attack remains an unresolved proof obligation.",
        }),
    )
    assert any(
        f"{disposition} attack blocks a complete release" in error
        for error in validate_run(run_dir, mode="release")
    )


@pytest.mark.parametrize(
    "name,body",
    [
        (
            "counterexample-register.md",
            "# Counterexample register\n\nNo counterexample search has been recorded.",
        ),
        (
            "construction-or-strongest-theorem.md",
            "# Construction or strongest theorem\n\nNo construction or theorem has been established.",
        ),
        (
            "final-report.md",
            "# Candidate-run report\n\nThis scaffold is a structural checkpoint only.",
        ),
    ],
)
def test_release_rejects_scaffold_markdown(
    tmp_path: Path, name: str, body: str,
) -> None:
    run_dir = make_valid_run(tmp_path)
    contract = load_json(run_dir, "problem-contract.json")
    (run_dir / name).write_text(
        f"{metadata_line(str(contract['contract_id']), str(contract['target_digest']))}\n{body}\n",
        encoding="utf-8",
    )
    assert any(
        f"{name}: scaffold content is not releasable" in error
        for error in validate_run(run_dir, mode="release")
    )


@pytest.mark.parametrize("mutation", ["missing", "renamed", "extra", "reordered"])
def test_release_requires_the_exact_final_report_headings(
    tmp_path: Path, mutation: str,
) -> None:
    run_dir = make_valid_run(tmp_path)
    path = run_dir / "final-report.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    if mutation == "missing":
        lines.remove("## Certificate")
    elif mutation == "renamed":
        lines[lines.index("## Certificate")] = "## Certification"
    elif mutation == "extra":
        lines.extend(["", "## Appendix", "", "Unexpected heading."])
    else:
        left = lines.index("## Certificate")
        right = lines.index("## Terminal status")
        lines[left], lines[right] = lines[right], lines[left]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert any(
        "final-report.md: headings must exactly match the release contract" in error
        for error in validate_run(run_dir, mode="release")
    )


@pytest.mark.parametrize("field", ["independent_reconstruction", "oracle_erasure"])
def test_release_requires_completed_adversarial_records(tmp_path: Path, field: str) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "adversarial-report.json", lambda p: p.update({field: {"completed": False}}))
    assert any(field in error for error in validate_run(run_dir, mode="release"))


def test_boolean_only_adversarial_completion_is_not_release_evidence(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir,
        "adversarial-report.json",
        lambda payload: payload.update({
            "attacks": [],
            "independent_reconstruction": {"completed": True},
            "oracle_erasure": {"completed": True},
        }),
    )
    errors = validate_run(run_dir, mode="release")
    assert any("attacks must be a substantive nonempty list" in error for error in errors)
    assert any("independent_reconstruction must be artifact-backed" in error for error in errors)
    assert any("oracle_erasure must be artifact-backed" in error for error in errors)


@pytest.mark.parametrize("field", ["attack", "response", "artifact_path", "artifact_sha256"])
def test_adversarial_attacks_require_substantive_artifact_backing(
    tmp_path: Path, field: str,
) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir,
        "adversarial-report.json",
        lambda payload: payload["attacks"][0].update({field: ""}),
    )
    assert any(
        f"attack attack-1: {field}" in error
        for error in validate_run(run_dir, mode="release")
    )


def test_adversarial_records_must_cover_target_dependency_closure(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload.update({
            "claims": [
                bound_claim(run_dir),
                claim("ancestor", evidence_ids=["proof-1"]),
            ],
        }),
    )
    replace_json(
        run_dir,
        "dependency-dag.json",
        lambda payload: payload.update({"edges": [{"from": "target", "to": "ancestor"}]}),
    )
    errors = validate_run(run_dir, mode="release")
    assert any("attacks omit dependency-closure claims: ancestor" in error for error in errors)
    assert any(
        "independent_reconstruction omits dependency-closure claims: ancestor" in error
        for error in errors
    )
    assert any("oracle_erasure omits dependency-closure claims: ancestor" in error for error in errors)


def test_adversarial_artifact_hash_is_recomputed(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    (run_dir / "proofs" / "attack.md").write_text("forged attack result\n", encoding="utf-8")
    assert any(
        "attack attack-1: artifact_sha256 mismatch" in error
        for error in validate_run(run_dir, mode="release")
    )


def test_rejects_invalid_mode_and_status_and_sorts_multiple_errors(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path, status="NOT_A_STATUS")
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"evidence": [dict(evidence(), artifact_path="../unsafe", artifact_sha256="bad")]}))
    errors = validate_run(run_dir, mode="release")
    assert errors == sorted(errors)
    assert any("terminal_status" in error for error in errors)
    assert any("artifact_path" in error for error in errors)
    assert any("artifact_sha256" in error for error in errors)
    assert validate_run(run_dir, mode="invalid") == ["invalid mode: invalid"]


@pytest.mark.parametrize("status, target_state, evidence_kind", [
    ("COMPLETE_AFFIRMATIVE", "EVIDENCE_VERIFIED", "FORMAL_PROOF"),
    ("COMPLETE_NEGATIVE", "REFUTED", "COUNTEREXAMPLE"),
    ("INCONCLUSIVE", "INCONCLUSIVE", "FORMAL_PROOF"),
])
def test_terminal_status_requires_matching_target_state(
    tmp_path: Path, status: str, target_state: str, evidence_kind: str
) -> None:
    run_dir = make_valid_run(tmp_path, status=status)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({
        "evidence": [evidence("certificate", evidence_kind)],
        "claims": [bound_claim(run_dir, state=target_state, evidence_ids=["certificate"])],
    }))
    if status == "INCONCLUSIVE":
        replace_json(run_dir, "release.json", lambda p: p.update({"strongest_result": "partial", "unresolved_obligations": ["open"]}))
    assert validate_run(run_dir, mode="release") == []
    wrong = "REFUTED" if target_state != "REFUTED" else "EVIDENCE_VERIFIED"
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"claims": [claim("target", state=wrong, evidence_ids=["certificate"])]}))
    assert any("target claim target" in error for error in validate_run(run_dir, mode="release"))


def test_terminal_certificate_must_resolve_and_match_target_relationship(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "release.json", lambda p: p.update({"certificate_claim": "missing"}))
    assert any("certificate_claim" in error for error in validate_run(run_dir, mode="release"))
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"claims": [claim("target"), claim("other")]}))
    replace_json(run_dir, "release.json", lambda p: p.update({"certificate_claim": "other"}))
    assert any("certificate relationship" in error for error in validate_run(run_dir, mode="release"))


def test_checkpoint_checks_verified_evidence_and_all_physical_claims(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path, status=None)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({
        "evidence": [evidence("numeric", "NUMERICAL")],
        "claims": [claim("target", state="CANDIDATE"), claim("side", kind="PHYSICAL_INTERPRETATION", evidence_ids=["numeric"])],
    }))
    errors = validate_run(run_dir, mode="checkpoint")
    assert any("physical interpretation side" in error for error in errors)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({
        "claims": [claim("target", state="CANDIDATE"), claim("side", state="REFUTED", kind="PHYSICAL_INTERPRETATION", evidence_ids=["numeric"])],
    }))
    assert any("physical interpretation side" in error for error in validate_run(run_dir, mode="checkpoint"))
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({
        "claims": [claim("target", state="CANDIDATE"), claim("side", kind="MATHEMATICAL", evidence_ids=["numeric"])],
    }))
    assert any("mathematical claim side" in error for error in validate_run(run_dir, mode="checkpoint"))


def test_checkpoint_requires_direct_refutation_evidence_for_refuted_mathematics(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path, status=None)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({
        "evidence": [evidence("numeric", "NUMERICAL")],
        "claims": [claim("target", state="REFUTED", evidence_ids=["numeric"])],
    }))
    assert any("refuted mathematical claim target" in error for error in validate_run(run_dir, mode="checkpoint"))


@pytest.mark.parametrize(
    "kind,state,eligible_kind",
    [
        ("MATHEMATICAL", "EVIDENCE_VERIFIED", "FORMAL_PROOF"),
        ("MATHEMATICAL", "REFUTED", "COUNTEREXAMPLE"),
        ("PHYSICAL_INTERPRETATION", "EVIDENCE_VERIFIED", "PRIMARY_SOURCE"),
        ("PHYSICAL_INTERPRETATION", "REFUTED", "PRIMARY_SOURCE"),
        ("MODELING_POSTULATE", "EVIDENCE_VERIFIED", "PRIMARY_SOURCE"),
        ("MODELING_POSTULATE", "REFUTED", "PRIMARY_SOURCE"),
        ("OPERATIONAL_IDENTIFICATION", "EVIDENCE_VERIFIED", "NUMERICAL"),
        ("OPERATIONAL_IDENTIFICATION", "REFUTED", "NUMERICAL"),
        ("NUMERICAL_OBSERVATION", "EVIDENCE_VERIFIED", "NUMERICAL"),
        ("NUMERICAL_OBSERVATION", "REFUTED", "NUMERICAL"),
    ],
)
def test_every_closable_claim_kind_has_explicit_evidence_eligibility(
    tmp_path: Path, kind: str, state: str, eligible_kind: str,
) -> None:
    run_dir = make_valid_run(tmp_path, status=None)
    supports = state == "EVIDENCE_VERIFIED"
    assumptions = []
    bridges: list[str] = []
    if kind == "PHYSICAL_INTERPRETATION":
        assumptions = [{
            "id": "bridge",
            "statement": "The recorded quantity operationalizes the physical observable.",
            "kind": "OPERATIONAL_IDENTIFICATION",
        }]
        bridges = ["bridge"]
    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload.update({
            "assumptions": assumptions,
            "evidence": [evidence("direct", eligible_kind, supports=supports)],
            "claims": [
                bound_claim(run_dir, state="CANDIDATE", evidence_ids=[]),
                bound_claim(
                    run_dir,
                    "side",
                    state=state,
                    kind=kind,
                    evidence_ids=["direct"],
                    bridge_premise_ids=bridges,
                ),
            ],
        }),
    )
    assert validate_run(run_dir, mode="checkpoint") == []

    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload.update({
            "evidence": [
                evidence("direct", "AGENT_ASSESSMENT", supports=supports),
            ],
        }),
    )
    assert any(
        f"claim side lacks eligible {state} evidence" in error
        for error in validate_run(run_dir, mode="checkpoint")
    )


@pytest.mark.parametrize("supports", [None, "yes", 1])
def test_evidence_requires_boolean_support_polarity(tmp_path: Path, supports: object) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload["evidence"][0].update({"supports": supports}),
    )
    assert any(
        "evidence proof-1: supports must be boolean" in error
        for error in validate_run(run_dir, mode="release")
    )


def test_invalid_negative_certificate_does_not_emit_inconclusive_payload_errors(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path, status="COMPLETE_NEGATIVE")
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({
        "evidence": [evidence("counter", "COUNTEREXAMPLE")],
        "claims": [claim("target", state="REFUTED", evidence_ids=["counter"])],
    }))
    replace_json(run_dir, "release.json", lambda p: p.update({"certificate_claim": "missing", "strongest_result": "", "unresolved_obligations": []}))
    errors = validate_run(run_dir, mode="release")
    assert any("certificate_claim" in error for error in errors)
    assert not any("inconclusive release" in error for error in errors)


def test_release_checks_verified_physical_ancestors_not_only_target(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({
        "claims": [claim("target"), claim("ancestor", kind="PHYSICAL_INTERPRETATION")],
    }))
    replace_json(run_dir, "dependency-dag.json", lambda p: p.update({"edges": [{"from": "target", "to": "ancestor"}]}))
    assert any("physical interpretation ancestor" in error for error in validate_run(run_dir, mode="release"))


def test_rejects_search_prior_recursively_in_all_nonprior_artifacts(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "approach-registry.json", lambda p: p["mechanism_families"][0].update({"bridge": {"nested": "SEARCH_PRIOR_AFFIRMATIVE"}}))
    replace_json(run_dir, "claim-ledger.json", lambda p: p["claims"][0].update({"statement": "SEARCH_PRIOR_AFFIRMATIVE"}))
    assert any("SEARCH_PRIOR_AFFIRMATIVE" in error for error in validate_run(run_dir, mode="release"))


@pytest.mark.parametrize("unsafe", ["C:/absolute", "C:relative", "//server/share", "\\\\server\\share", "/rooted", "../traversal", "nested/../traversal", "bad\x00path"])
def test_evidence_paths_must_be_portable_safe_relative_paths(tmp_path: Path, unsafe: str) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"evidence": [dict(evidence(), artifact_path=unsafe)]}))
    assert any("unsafe artifact_path" in error for error in validate_run(run_dir, mode="release"))


def test_evidence_file_must_exist_and_match_its_declared_sha256(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    (run_dir / "proofs" / "proof.txt").unlink()
    errors = validate_run(run_dir, mode="release")
    assert any("evidence proof-1: artifact is missing" in error for error in errors)

    (run_dir / "proofs" / "proof.txt").write_bytes(b"forged proof\n")
    errors = validate_run(run_dir, mode="release")
    assert any("evidence proof-1: artifact_sha256 mismatch" in error for error in errors)


def test_evidence_path_must_name_a_regular_file(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(
        run_dir,
        "claim-ledger.json",
        lambda payload: payload.update({
            "evidence": [
                evidence("proof-1", artifact_path="proofs")
                | {"artifact_sha256": sha256_bytes(PROOF_BYTES)},
            ],
        }),
    )
    assert any(
        "evidence proof-1: artifact is not a regular file" in error
        for error in validate_run(run_dir, mode="release")
    )


def test_evidence_symlink_escape_is_rejected(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    outside = tmp_path / "outside-proof.txt"
    outside.write_bytes(PROOF_BYTES)
    proof = run_dir / "proofs" / "proof.txt"
    proof.unlink()
    try:
        proof.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlinks unavailable on this platform: {error}")
    assert proof.is_symlink()
    assert any(
        "evidence proof-1: artifact path contains a symlink" in error
        for error in validate_run(run_dir, mode="release")
    )


def test_cli_main_returns_exit_code_and_help_explains_limit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = make_valid_run(tmp_path)
    assert main([str(run_dir), "--mode", "release"]) == 0
    assert capsys.readouterr().out == ""
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    assert "cannot establish mathematical truth" in capsys.readouterr().out


@pytest.mark.parametrize("field", ["evidence_ids", "assumption_ids", "bridge_premise_ids"])
def test_malformed_claim_reference_containers_are_diagnostics_not_exceptions(tmp_path: Path, field: str) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "claim-ledger.json", lambda p: p["claims"][0].update({field: 7}))
    errors = validate_run(run_dir, mode="release")
    assert errors == sorted(errors)
    assert any(field in error for error in errors)


def test_list_target_in_negative_release_is_a_diagnostic_not_an_exception(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path, status="COMPLETE_NEGATIVE")
    replace_json(run_dir, "problem-contract.json", lambda p: p.update({"target": []}))
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"evidence": [evidence("counter", "COUNTEREXAMPLE")], "claims": [claim("target", state="REFUTED", evidence_ids=["counter"])]}))
    errors = validate_run(run_dir, mode="release")
    assert errors == sorted(errors)
    assert any("missing target" in error for error in errors)


def test_nonlist_claim_collections_are_diagnostics_not_exceptions(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    replace_json(run_dir, "claim-ledger.json", lambda p: p.update({"claims": {"target": claim()}}))
    errors = validate_run(run_dir, mode="release")
    assert errors == sorted(errors)
    assert any("claims: expected list" in error for error in errors)
