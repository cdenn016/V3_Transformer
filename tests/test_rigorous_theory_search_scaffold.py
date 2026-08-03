"""Behavioral tests for deterministic rigorous-theory-search run scaffolding."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from agent_tooling.rigorous_theory_search.skill.scripts.scaffold_run import main, scaffold_run
from agent_tooling.rigorous_theory_search.skill.scripts.validate_run import validate_run


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
METADATA_PATTERN = re.compile(r"<!-- rigorous-theory-search-metadata (\{.*\}) -->\Z")


def canonical_target_digest(target: dict[str, object]) -> str:
    encoded = json.dumps(
        target, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_contents(run_dir: Path) -> dict[str, str]:
    return {name: (run_dir / name).read_text(encoding="utf-8") for name in ARTIFACTS}


def test_scaffold_creates_a_valid_candidate_run(tmp_path: Path) -> None:
    run_dir = scaffold_run(tmp_path, slug="pullback-geometry", date="2026-08-02")
    assert run_dir == tmp_path / "2026-08-02-pullback-geometry"
    assert validate_run(run_dir, mode="checkpoint") == []


def test_scaffold_creates_every_required_artifact_with_one_contract_id(tmp_path: Path) -> None:
    run_dir = scaffold_run(tmp_path, slug="pullback-geometry", date="2026-08-02")
    assert all((run_dir / name).is_file() for name in ARTIFACTS)
    payloads = [json.loads((run_dir / name).read_text(encoding="utf-8")) for name in ARTIFACTS if name.endswith(".json")]
    contract_ids = {payload["contract_id"] for payload in payloads}
    assert len(contract_ids) == 1
    contract = json.loads((run_dir / "problem-contract.json").read_text(encoding="utf-8"))
    digest = canonical_target_digest(contract["target"])
    assert contract_ids == {f"contract-sha256-{digest}"}
    assert {payload["target_digest"] for payload in payloads} == {digest}
    for name in ARTIFACTS:
        if not name.endswith(".md"):
            continue
        first_line = (run_dir / name).read_text(encoding="utf-8").splitlines()[0]
        match = METADATA_PATTERN.fullmatch(first_line)
        assert match is not None
        assert json.loads(match.group(1)) == {
            "contract_id": f"contract-sha256-{digest}",
            "schema_version": "rigorous-theory-search/v1",
            "target_digest": digest,
        }
    ledger = json.loads((run_dir / "claim-ledger.json").read_text(encoding="utf-8"))
    assert ledger["claims"][0]["state"] == "CANDIDATE"
    registry = json.loads((run_dir / "approach-registry.json").read_text(encoding="utf-8"))
    family = registry["mechanism_families"][0]
    assert family["family_id"] == "family-001"
    assert family["core_mechanism"] == "Candidate mechanism to be specified."
    assert family["target_obligation_ids"] == ["target"]


@pytest.mark.parametrize("slug", ["", "UPPER", "bad slug", "../escape", "safe/escape", "a--b", "-edge", "edge-"])
def test_scaffold_rejects_invalid_slugs(tmp_path: Path, slug: str) -> None:
    with pytest.raises(ValueError, match="slug"):
        scaffold_run(tmp_path, slug=slug, date="2026-08-02")


@pytest.mark.parametrize("date", ["20260802", "2026-2-02", "2026-02-30", "2026-08-02/escape"])
def test_scaffold_rejects_invalid_dates(tmp_path: Path, date: str) -> None:
    with pytest.raises(ValueError, match="date"):
        scaffold_run(tmp_path, slug="pullback-geometry", date=date)


def test_scaffold_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-08-02-pullback-geometry"
    run_dir.mkdir()
    (run_dir / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        scaffold_run(tmp_path, slug="pullback-geometry", date="2026-08-02")
    assert (run_dir / "keep.txt").read_text(encoding="utf-8") == "preserve"


def test_scaffold_cleans_failed_staging_and_allows_immediate_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write = Path.write_text
    writes = 0

    def fail_second_artifact(path: Path, data: str, **kwargs: object) -> int:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise PermissionError("injected artifact write failure")
        return original_write(path, data, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_second_artifact)
    with pytest.raises(PermissionError, match="injected artifact write failure"):
        scaffold_run(tmp_path, slug="pullback-geometry", date="2026-08-02")
    assert not (tmp_path / "2026-08-02-pullback-geometry").exists()
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(Path, "write_text", original_write)
    run_dir = scaffold_run(tmp_path, slug="pullback-geometry", date="2026-08-02")
    assert validate_run(run_dir, mode="checkpoint") == []


def test_scaffold_preserves_a_concurrently_created_final_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "2026-08-02-pullback-geometry"

    def concurrent_target(staging: Path, destination: Path) -> Path:
        assert destination == target
        destination.mkdir()
        (destination / "concurrent.txt").write_text("preserve", encoding="utf-8")
        raise FileExistsError("concurrent final target")

    monkeypatch.setattr(Path, "rename", concurrent_target)
    with pytest.raises(FileExistsError, match="concurrent final target"):
        scaffold_run(tmp_path, slug="pullback-geometry", date="2026-08-02")
    assert (target / "concurrent.txt").read_text(encoding="utf-8") == "preserve"
    assert [path.name for path in tmp_path.iterdir()] == [target.name]


def test_scaffold_output_is_stable_and_remains_within_explicit_root(tmp_path: Path) -> None:
    first = scaffold_run(tmp_path / "first", slug="pullback-geometry", date="2026-08-02")
    second = scaffold_run(tmp_path / "second", slug="pullback-geometry", date="2026-08-02")
    assert first.parent == tmp_path / "first"
    assert second.parent == tmp_path / "second"
    assert file_contents(first) == file_contents(second)


def test_scaffold_contract_binding_depends_on_target_not_date_or_slug(tmp_path: Path) -> None:
    first = scaffold_run(tmp_path / "first", slug="pullback-geometry", date="2026-08-02")
    second = scaffold_run(tmp_path / "second", slug="different-search", date="2027-01-03")
    first_contract = json.loads((first / "problem-contract.json").read_text(encoding="utf-8"))
    second_contract = json.loads((second / "problem-contract.json").read_text(encoding="utf-8"))
    assert first_contract["target"] == second_contract["target"]
    assert first_contract["contract_id"] == second_contract["contract_id"]
    assert first_contract["target_digest"] == second_contract["target_digest"]


def test_cli_creates_a_checkpoint_candidate_and_returns_zero(tmp_path: Path) -> None:
    assert main([str(tmp_path), "--slug", "pullback-geometry", "--date", "2026-08-02"]) == 0
    assert validate_run(tmp_path / "2026-08-02-pullback-geometry", mode="checkpoint") == []


def test_cli_returns_nonzero_for_invalid_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path), "--slug", "bad slug", "--date", "2026-08-02"]) == 2
    assert "slug" in capsys.readouterr().err
