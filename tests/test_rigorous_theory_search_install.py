from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from agent_tooling.rigorous_theory_search import install as installer


TARGET_NAME = "rigorous-theory-search"


def _write_source(root: Path) -> Path:
    (root / "references").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "SKILL.md").write_bytes(b"# Rigorous theory search\n")
    (root / "references" / "proof.md").write_bytes(b"proof obligations\n")
    (root / "scripts" / "validate.py").write_bytes(b"print('ok')\n")
    return root


def _make_homes(tmp_path: Path) -> tuple[Path, Path, Path]:
    homes = (
        tmp_path / "agents-home",
        tmp_path / "claude-home",
        tmp_path / "codex-home",
    )
    for home in homes:
        home.mkdir()
    return homes


def _target(home: Path) -> Path:
    return home / "skills" / TARGET_NAME


def _entry_snapshot(path: Path) -> tuple[tuple[str, str, bytes], ...] | None:
    if not os.path.lexists(path):
        return None
    paths = [path]
    if path.is_dir() and not path.is_symlink():
        paths.extend(sorted(path.rglob("*"), key=lambda item: item.as_posix()))
    entries: list[tuple[str, str, bytes]] = []
    for item in paths:
        relative = "." if item == path else item.relative_to(path).as_posix()
        mode = item.lstat().st_mode
        if stat.S_ISDIR(mode):
            entries.append((relative, "directory", b""))
        elif stat.S_ISREG(mode):
            entries.append((relative, "file", item.read_bytes()))
        elif stat.S_ISLNK(mode):
            entries.append((relative, "link", os.readlink(item).encode()))
        else:
            entries.append((relative, "special", b""))
    return tuple(entries)


def _seed_existing_target(target: Path, label: str) -> None:
    (target / "nested" / "empty").mkdir(parents=True)
    (target / ".pytest_cache").mkdir()
    (target / "SKILL.md").write_bytes(f"old-{label}\n".encode())
    (target / "nested" / "stale.txt").write_bytes(f"stale-{label}\n".encode())
    (target / ".pytest_cache" / "state").write_bytes(f"cache-{label}\n".encode())


def _install_args(source: Path, homes: tuple[Path, Path, Path]) -> list[str]:
    return [
        "--source-root",
        os.fspath(source),
        "--agents-home",
        os.fspath(homes[0]),
        "--claude-home",
        os.fspath(homes[1]),
        "--codex-home",
        os.fspath(homes[2]),
        "--mirrored",
    ]


def _canonical_args(source: Path, agents_home: Path) -> list[str]:
    return [
        "--source-root",
        os.fspath(source),
        "--agents-home",
        os.fspath(agents_home),
    ]


def _make_directory_symlink(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        if os.name == "nt":
            result = subprocess.run(
                ["cmd.exe", "/c", "mklink", "/J", os.fspath(link), os.fspath(target)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return
        pytest.skip(f"directory links are unavailable: {exc}")


def _transaction_residue(homes: tuple[Path, Path, Path]) -> list[Path]:
    residue: list[Path] = []
    prefixes = (
        f".{TARGET_NAME}.stage-",
        f".{TARGET_NAME}.backup-",
        f".{TARGET_NAME}.install.lock",
    )
    for home in homes:
        skills = home / "skills"
        if not skills.is_dir():
            continue
        residue.extend(
            entry
            for entry in skills.iterdir()
            if entry.name.startswith(prefixes)
        )
    return sorted(residue, key=lambda path: path.as_posix())


def _remove_unknown_target(target: Path) -> None:
    for path in sorted(target.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    target.rmdir()


def test_build_manifest_hashes_sorted_managed_files_and_ignores_caches(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "ignored.pyc").write_bytes(b"cache")
    (source / ".ruff_cache").mkdir()
    (source / ".ruff_cache" / "ignored").write_bytes(b"cache")
    (source / "ignored.pyo").write_bytes(b"cache")

    manifest = installer.build_manifest(source)

    expected_bytes = {
        "SKILL.md": b"# Rigorous theory search\n",
        "references/proof.md": b"proof obligations\n",
        "scripts/validate.py": b"print('ok')\n",
    }
    expected = {
        relative: {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        for relative, content in expected_bytes.items()
    }
    assert manifest == expected
    assert list(manifest) == sorted(manifest)


@pytest.mark.parametrize("missing_skill", [False, True])
def test_build_manifest_rejects_missing_source_or_skill(
    tmp_path: Path,
    missing_skill: bool,
) -> None:
    source = tmp_path / "source"
    if missing_skill:
        source.mkdir()

    with pytest.raises(installer.InstallError):
        installer.build_manifest(source)


def test_build_manifest_detects_source_mutation_during_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    real_hash = installer._hash_regular_file

    def mutating_hash(path: Path, before: os.stat_result) -> dict[str, object]:
        result = real_hash(path, before)
        if path.name == "SKILL.md":
            with path.open("ab") as handle:
                handle.write(b"mutated")
        return result

    monkeypatch.setattr(installer, "_hash_regular_file", mutating_hash)

    with pytest.raises(installer.InstallError, match="changed while hashing"):
        installer.build_manifest(source)


@pytest.mark.parametrize("cache_name", ["ordinary-link", "__pycache__"])
def test_source_links_are_rejected_before_cache_ignores(
    tmp_path: Path,
    cache_name: str,
) -> None:
    source = _write_source(tmp_path / "source")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"untouched")
    _make_directory_symlink(source / cache_name, outside)

    with pytest.raises(installer.InstallError, match="unsafe"):
        installer.build_manifest(source)

    assert sentinel.read_bytes() == b"untouched"


def test_source_root_link_is_rejected(tmp_path: Path) -> None:
    real_source = _write_source(tmp_path / "real-source")
    linked_source = tmp_path / "linked-source"
    _make_directory_symlink(linked_source, real_source)

    with pytest.raises(installer.InstallError, match="unsafe"):
        installer.build_manifest(linked_source)


def test_destination_component_link_is_rejected_without_touching_outside(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"untouched")
    _make_directory_symlink(homes[0] / "skills", outside)
    before = _entry_snapshot(outside)

    with pytest.raises(installer.InstallError, match="unsafe"):
        installer.install(source, *homes)

    assert _entry_snapshot(outside) == before
    assert not (homes[1] / "skills").exists()
    assert not (homes[2] / "skills").exists()


def test_cache_named_destination_link_is_unsafe_not_ignored(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    installer.install(source, *homes)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"untouched")
    _make_directory_symlink(_target(homes[0]) / ".pytest_cache", outside)
    before = _entry_snapshot(outside)

    diagnostics = installer.verify_install(source, *homes)
    with pytest.raises(installer.InstallError, match="unsafe"):
        installer.install(source, *homes)

    assert diagnostics == sorted(diagnostics)
    assert any("agents" in item and "unsafe" in item for item in diagnostics)
    assert _entry_snapshot(outside) == before


def test_duplicate_targets_are_rejected_before_any_write(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source")
    shared_home = tmp_path / "shared-home"
    codex_home = tmp_path / "codex-home"
    shared_home.mkdir()
    codex_home.mkdir()
    before = _entry_snapshot(tmp_path)

    with pytest.raises(installer.InstallError, match="duplicate"):
        installer.install(source, shared_home, shared_home, codex_home)

    assert _entry_snapshot(tmp_path) == before


def test_source_destination_overlap_is_rejected_before_other_home_writes(
    tmp_path: Path,
) -> None:
    homes = _make_homes(tmp_path)
    source = _write_source(_target(homes[0]))
    before = _entry_snapshot(tmp_path)

    with pytest.raises(installer.InstallError, match="overlap"):
        installer.install(source, *homes)

    assert _entry_snapshot(tmp_path) == before


def test_nondirectory_skills_parent_is_rejected_without_writes(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    skills_file = homes[1] / "skills"
    skills_file.write_bytes(b"not a directory")
    before = _entry_snapshot(tmp_path)

    with pytest.raises(installer.InstallError, match="not a directory"):
        installer.install(source, *homes)

    assert _entry_snapshot(tmp_path) == before


def test_install_copies_byte_identical_manifests_to_three_clean_mirrors(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    for home in homes:
        sibling = home / "skills" / "other-skill"
        sibling.mkdir(parents=True)
        (sibling / "keep.txt").write_bytes(b"keep")

    installer.install(source, *homes)

    source_manifest = installer.build_manifest(source)
    assert installer.verify_install(source, *homes) == []
    for home in homes:
        target = _target(home)
        assert installer.build_manifest(target) == source_manifest
        assert (home / "skills" / "other-skill" / "keep.txt").read_bytes() == b"keep"
        residue = [
            entry.name
            for entry in (home / "skills").iterdir()
            if entry.name.startswith(f".{TARGET_NAME}.")
        ]
        assert residue == []


def test_matching_reinstall_is_a_true_noop_and_ignores_destination_cache(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    installer.install(source, *homes)
    cache = _target(homes[0]) / ".pytest_cache" / "state"
    cache.parent.mkdir()
    cache.write_bytes(b"preserve")
    observed = [
        _target(home) / "SKILL.md"
        for home in homes
    ] + [cache]
    before = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in observed
    }

    installer.install(source, *homes)

    after = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in observed
    }
    assert after == before
    assert installer.verify_install(source, *homes) == []


def test_reinstall_repairs_corruption_and_removes_stale_managed_entries(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    installer.install(source, *homes)
    target = _target(homes[1])
    (target / "SKILL.md").write_bytes(b"corrupt")
    (target / "stale" / "nested").mkdir(parents=True)
    (target / "stale" / "nested" / "old.txt").write_bytes(b"old")
    (target / "stale-empty" / "nested" / "leaf").mkdir(parents=True)
    sibling = homes[1] / "skills" / "sibling-skill"
    sibling.mkdir()
    (sibling / "keep.txt").write_bytes(b"keep")

    installer.install(source, *homes)

    assert installer.verify_install(source, *homes) == []
    assert not (target / "stale").exists()
    assert not (target / "stale-empty").exists()
    assert (sibling / "keep.txt").read_bytes() == b"keep"


@pytest.mark.parametrize("fail_at", [2, 3])
@pytest.mark.parametrize("targets_present", [False, True])
def test_stage_failure_on_later_mirror_restores_exact_prestates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: int,
    targets_present: bool,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    if targets_present:
        for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
            _seed_existing_target(_target(home), label)
    before = {home: _entry_snapshot(home) for home in homes}
    real_create_stage = installer._create_stage
    calls = 0

    def failing_create_stage(*args: object, **kwargs: object) -> Path:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(f"injected stage {fail_at}")
        return real_create_stage(*args, **kwargs)

    monkeypatch.setattr(installer, "_create_stage", failing_create_stage)

    with pytest.raises(installer.InstallError, match=f"injected stage {fail_at}"):
        installer.install(source, *homes)

    assert {home: _entry_snapshot(home) for home in homes} == before


@pytest.mark.parametrize("fail_at", [2, 3])
@pytest.mark.parametrize("targets_present", [False, True])
def test_promotion_failure_on_later_mirror_restores_exact_prestates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: int,
    targets_present: bool,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    if targets_present:
        for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
            _seed_existing_target(_target(home), label)
    before = {home: _entry_snapshot(home) for home in homes}
    real_promote = installer._promote_stage
    calls = 0

    def failing_promote(stage: Path, target: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(f"injected promotion {fail_at}")
        return real_promote(stage, target)

    monkeypatch.setattr(installer, "_promote_stage", failing_promote)

    with pytest.raises(installer.InstallError, match=f"injected promotion {fail_at}"):
        installer.install(source, *homes)

    assert {home: _entry_snapshot(home) for home in homes} == before


@pytest.mark.parametrize(
    "interruption",
    [KeyboardInterrupt("injected interruption"), SystemExit(73)],
    ids=("keyboard-interrupt", "system-exit"),
)
def test_interruption_during_later_promotion_restores_exact_prestates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
        _seed_existing_target(_target(home), label)
    before_targets = {home: _entry_snapshot(_target(home)) for home in homes}
    before_manifests = {
        home: installer.build_manifest(_target(home)) for home in homes
    }
    real_promote = installer._promote_stage
    calls = 0

    def interrupting_promote(stage: Path, target: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interruption
        return real_promote(stage, target)

    monkeypatch.setattr(installer, "_promote_stage", interrupting_promote)

    with pytest.raises(type(interruption)) as raised:
        installer.install(source, *homes)

    assert raised.value is interruption
    assert {home: _entry_snapshot(_target(home)) for home in homes} == before_targets
    assert {
        home: installer.build_manifest(_target(home)) for home in homes
    } == before_manifests
    assert _transaction_residue(homes) == []


@pytest.mark.parametrize(
    "interruption_type",
    [KeyboardInterrupt, SystemExit],
    ids=("keyboard-interrupt", "system-exit"),
)
@pytest.mark.parametrize("fail_at", [1, 3], ids=("first-lock", "last-lock"))
@pytest.mark.parametrize("targets_present", [False, True])
def test_interruption_during_lock_token_write_restores_exact_prestates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_type: type[BaseException],
    fail_at: int,
    targets_present: bool,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    if targets_present:
        for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
            _seed_existing_target(_target(home), label)
    before = {home: _entry_snapshot(home) for home in homes}
    interruption = (
        KeyboardInterrupt("injected lock interruption")
        if interruption_type is KeyboardInterrupt
        else SystemExit(73)
    )
    real_write = installer._write_lock_token
    calls = 0

    def interrupting_write(descriptor: int, token: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise interruption
        real_write(descriptor, token)

    monkeypatch.setattr(installer, "_write_lock_token", interrupting_write)

    with pytest.raises(interruption_type) as raised:
        installer.install(source, *homes)

    assert raised.value is interruption
    assert {home: _entry_snapshot(home) for home in homes} == before
    assert _transaction_residue(homes) == []


@pytest.mark.parametrize(
    "interruption_type",
    [KeyboardInterrupt, SystemExit],
    ids=("keyboard-interrupt", "system-exit"),
)
@pytest.mark.parametrize(
    "fail_at",
    [2, 8],
    ids=("partial-first-stage", "partial-last-stage"),
)
@pytest.mark.parametrize("targets_present", [False, True])
def test_interruption_during_stage_copy_restores_exact_prestates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_type: type[BaseException],
    fail_at: int,
    targets_present: bool,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    if targets_present:
        for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
            _seed_existing_target(_target(home), label)
    before = {home: _entry_snapshot(home) for home in homes}
    interruption = (
        KeyboardInterrupt("injected stage interruption")
        if interruption_type is KeyboardInterrupt
        else SystemExit(73)
    )
    real_copy = installer._copy_managed_file
    calls = 0

    def interrupting_copy(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise interruption
        real_copy(*args, **kwargs)

    monkeypatch.setattr(installer, "_copy_managed_file", interrupting_copy)

    with pytest.raises(interruption_type) as raised:
        installer.install(source, *homes)

    assert raised.value is interruption
    assert {home: _entry_snapshot(home) for home in homes} == before
    assert _transaction_residue(homes) == []


@pytest.mark.parametrize(
    "interruption_type",
    [KeyboardInterrupt, SystemExit],
    ids=("keyboard-interrupt", "system-exit"),
)
def test_interruption_during_committed_backup_cleanup_retains_new_mirrors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_type: type[BaseException],
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
        _seed_existing_target(_target(home), label)
    interruption = (
        KeyboardInterrupt("injected committed cleanup interruption")
        if interruption_type is KeyboardInterrupt
        else SystemExit(73)
    )
    real_remove = installer._remove_owned_tree
    backup_removals = 0

    def interrupt_after_first_backup_removal(path: Path) -> None:
        nonlocal backup_removals
        real_remove(path)
        if ".backup-" in path.name:
            backup_removals += 1
            if backup_removals == 1:
                raise interruption

    monkeypatch.setattr(
        installer,
        "_remove_owned_tree",
        interrupt_after_first_backup_removal,
    )

    with pytest.raises(interruption_type) as raised:
        installer.install(source, *homes)

    assert raised.value is interruption
    assert installer.verify_install(source, *homes) == []
    assert all(_target(home).is_dir() for home in homes)
    residue = _transaction_residue(homes)
    assert len(residue) == 2
    assert all(".backup-" in path.name for path in residue)
    assert raised.value.__notes__ == [
        "installation committed before interruption; installed targets were retained"
    ]


def test_interruption_preserves_cleanup_diagnostics_on_original_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
        _seed_existing_target(_target(home), label)
    before_targets = {home: _entry_snapshot(_target(home)) for home in homes}
    interruption = KeyboardInterrupt("injected interruption")
    real_promote = installer._promote_stage
    real_rollback = installer._rollback
    calls = 0

    def interrupting_promote(stage: Path, target: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interruption
        return real_promote(stage, target)

    def rollback_with_diagnostic(*args: object, **kwargs: object) -> list[str]:
        errors = real_rollback(*args, **kwargs)
        return [*errors, "injected rollback diagnostic"]

    monkeypatch.setattr(installer, "_promote_stage", interrupting_promote)
    monkeypatch.setattr(installer, "_rollback", rollback_with_diagnostic)

    with pytest.raises(KeyboardInterrupt) as raised:
        installer.install(source, *homes)

    assert raised.value is interruption
    assert raised.value.__notes__ == [
        "installation interrupted; cleanup errors: injected rollback diagnostic"
    ]
    assert {home: _entry_snapshot(_target(home)) for home in homes} == before_targets
    assert _transaction_residue(homes) == []


def test_source_drift_before_live_mutation_preserves_exact_prestates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
        _seed_existing_target(_target(home), label)
    before = {home: _entry_snapshot(home) for home in homes}
    real_build_manifest = installer.build_manifest
    calls = 0

    def drifting_manifest(path: Path) -> dict[str, dict[str, object]]:
        nonlocal calls
        result = real_build_manifest(path)
        if Path(path) == source:
            calls += 1
            if calls == 2:
                (source / "SKILL.md").write_bytes(b"source drift")
                result = real_build_manifest(path)
        return result

    monkeypatch.setattr(installer, "build_manifest", drifting_manifest)

    with pytest.raises(installer.InstallError, match="source changed"):
        installer.install(source, *homes)

    assert {home: _entry_snapshot(home) for home in homes} == before


@pytest.mark.parametrize("race_index", [0, 1, 2])
def test_target_appearing_during_staging_is_preserved_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race_index: int,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    before = {home: _entry_snapshot(home) for home in homes}
    real_prepare = installer._prepare_stages
    raced_snapshot: tuple[tuple[str, str, bytes], ...] | None = None

    def racing_prepare(*args: object, **kwargs: object) -> None:
        nonlocal raced_snapshot
        real_prepare(*args, **kwargs)
        raced_target = _target(homes[race_index])
        raced_target.mkdir()
        (raced_target / "UNKNOWN-OWNER.txt").write_bytes(b"external owner\n")
        raced_snapshot = _entry_snapshot(raced_target)

    monkeypatch.setattr(installer, "_prepare_stages", racing_prepare)

    with pytest.raises(installer.InstallError, match="target changed"):
        installer.install(source, *homes)

    raced_target = _target(homes[race_index])
    assert _entry_snapshot(raced_target) == raced_snapshot
    for index, home in enumerate(homes):
        if index != race_index:
            assert _entry_snapshot(home) == before[home]
    assert _transaction_residue(homes) == []

    _remove_unknown_target(raced_target)
    monkeypatch.setattr(installer, "_prepare_stages", real_prepare)
    installer.install(source, *homes)
    assert installer.verify_install(source, *homes) == []


@pytest.mark.parametrize("mutation", ["content", "disappearance", "replacement"])
def test_existing_target_drift_during_staging_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
        _seed_existing_target(_target(home), label)
    before = {home: _entry_snapshot(home) for home in homes}
    raced_target = _target(homes[0])
    held_target = homes[0] / "skills" / "externally-held-target"
    real_prepare = installer._prepare_stages
    external_snapshot: tuple[tuple[str, str, bytes], ...] | None = None
    held_snapshot: tuple[tuple[str, str, bytes], ...] | None = None

    def racing_prepare(*args: object, **kwargs: object) -> None:
        nonlocal external_snapshot, held_snapshot
        real_prepare(*args, **kwargs)
        if mutation == "content":
            (raced_target / "SKILL.md").write_bytes(b"external mutation\n")
        else:
            os.rename(raced_target, held_target)
            held_snapshot = _entry_snapshot(held_target)
            if mutation == "replacement":
                raced_target.mkdir()
                (raced_target / "UNKNOWN-OWNER.txt").write_bytes(b"replacement\n")
        external_snapshot = _entry_snapshot(raced_target)

    monkeypatch.setattr(installer, "_prepare_stages", racing_prepare)

    with pytest.raises(installer.InstallError, match="target changed"):
        installer.install(source, *homes)

    assert _entry_snapshot(raced_target) == external_snapshot
    assert _entry_snapshot(held_target) == held_snapshot
    assert _entry_snapshot(homes[1]) == before[homes[1]]
    assert _entry_snapshot(homes[2]) == before[homes[2]]
    assert _transaction_residue(homes) == []


def test_target_appearing_after_backup_preserves_itself_and_original_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
        _seed_existing_target(_target(home), label)
    before = {home: _entry_snapshot(home) for home in homes}
    before_targets = {home: _entry_snapshot(_target(home)) for home in homes}
    raced_target = _target(homes[1])
    real_backup = installer._backup_targets
    raced_snapshot: tuple[tuple[str, str, bytes], ...] | None = None

    def racing_backup(*args: object, **kwargs: object) -> None:
        nonlocal raced_snapshot
        real_backup(*args, **kwargs)
        raced_target.mkdir()
        (raced_target / "UNKNOWN-OWNER.txt").write_bytes(b"after backup\n")
        raced_snapshot = _entry_snapshot(raced_target)

    monkeypatch.setattr(installer, "_backup_targets", racing_backup)

    with pytest.raises(installer.InstallError, match="preserved"):
        installer.install(source, *homes)

    backups = list(raced_target.parent.glob(f".{TARGET_NAME}.backup-*"))
    assert len(backups) == 1
    assert _entry_snapshot(raced_target) == raced_snapshot
    assert _entry_snapshot(backups[0]) == before_targets[homes[1]]
    assert _entry_snapshot(homes[0]) == before[homes[0]]
    assert _entry_snapshot(homes[2]) == before[homes[2]]
    assert _transaction_residue(homes) == backups

    _remove_unknown_target(raced_target)
    os.rename(backups[0], raced_target)
    monkeypatch.setattr(installer, "_backup_targets", real_backup)
    installer.install(source, *homes)
    assert installer.verify_install(source, *homes) == []


def test_replaced_promoted_target_is_not_demoted_or_deleted_on_later_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    for label, home in zip(("agents", "claude", "codex"), homes, strict=True):
        _seed_existing_target(_target(home), label)
    before = {home: _entry_snapshot(home) for home in homes}
    before_targets = {home: _entry_snapshot(_target(home)) for home in homes}
    target = _target(homes[0])
    displaced = homes[0] / "skills" / "externally-held-promotion"
    real_promote = installer._promote_stage
    real_assert = installer._assert_manifest
    calls = 0
    replacement_snapshot: tuple[tuple[str, str, bytes], ...] | None = None
    displaced_snapshot: tuple[tuple[str, str, bytes], ...] | None = None

    def racing_promote(stage: Path, destination: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected later promotion failure")
        return real_promote(stage, destination)

    def racing_assert(
        root: Path,
        expected: dict[str, dict[str, object]],
        label: str,
    ) -> None:
        nonlocal replacement_snapshot, displaced_snapshot
        real_assert(root, expected, label)
        if label == "agents":
            os.rename(root, displaced)
            displaced_snapshot = _entry_snapshot(displaced)
            root.mkdir()
            (root / "UNKNOWN-OWNER.txt").write_bytes(b"replacement\n")
            replacement_snapshot = _entry_snapshot(root)

    monkeypatch.setattr(installer, "_promote_stage", racing_promote)
    monkeypatch.setattr(installer, "_assert_manifest", racing_assert)

    with pytest.raises(installer.InstallError, match="injected later promotion failure"):
        installer.install(source, *homes)

    backups = list(target.parent.glob(f".{TARGET_NAME}.backup-*"))
    assert len(backups) == 1
    assert _entry_snapshot(target) == replacement_snapshot
    assert _entry_snapshot(displaced) == displaced_snapshot
    assert _entry_snapshot(backups[0]) == before_targets[homes[0]]
    assert _entry_snapshot(homes[1]) == before[homes[1]]
    assert _entry_snapshot(homes[2]) == before[homes[2]]
    assert _transaction_residue(homes) == backups


def test_existing_install_lock_is_preserved_and_fails_closed(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    lock_path = installer._lock_path(_target(homes[0]))
    lock_path.parent.mkdir()
    lock_path.write_bytes(b"unknown owner")
    before = {home: _entry_snapshot(home) for home in homes}

    with pytest.raises(installer.InstallError, match="lock"):
        installer.install(source, *homes)

    assert lock_path.read_bytes() == b"unknown owner"
    assert {home: _entry_snapshot(home) for home in homes} == before


def test_partial_lock_write_removes_only_the_exact_created_lock_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    before = {home: _entry_snapshot(home) for home in homes}
    real_write = installer.os.write

    def failing_write(descriptor: int, content: bytes) -> int:
        raise OSError("injected lock write failure")

    monkeypatch.setattr(installer.os, "write", failing_write)

    with pytest.raises(installer.InstallError, match="injected lock write failure"):
        installer.install(source, *homes)

    assert {home: _entry_snapshot(home) for home in homes} == before
    assert _transaction_residue(homes) == []

    monkeypatch.setattr(installer.os, "write", real_write)
    installer.install(source, *homes)
    assert installer.verify_install(source, *homes) == []


def test_parent_creation_failure_removes_only_directories_created_by_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    before = {home: _entry_snapshot(home) for home in homes}
    real_mkdir_owned = installer._mkdir_owned
    calls = 0

    def failing_mkdir_owned(path: Path, label: str, created: list[Path]) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected parent creation")
        real_mkdir_owned(path, label, created)

    monkeypatch.setattr(installer, "_mkdir_owned", failing_mkdir_owned)

    with pytest.raises(installer.InstallError, match="injected parent creation"):
        installer.install(source, *homes)

    assert {home: _entry_snapshot(home) for home in homes} == before


def test_verify_install_returns_sorted_labeled_drift_without_writes(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = (
        tmp_path / "agents-home",
        tmp_path / "claude-home",
        tmp_path / "codex-home",
    )
    before = _entry_snapshot(tmp_path)

    diagnostics = installer.verify_install(source, *homes)

    assert diagnostics == sorted(diagnostics)
    assert {item.split(":", 1)[0] for item in diagnostics} == {
        "agents",
        "claude",
        "codex",
    }
    assert _entry_snapshot(tmp_path) == before


def test_check_mode_is_read_only_and_cli_statuses_distinguish_drift_and_invalid(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    args = _install_args(source, homes)
    before = {home: _entry_snapshot(home) for home in homes}

    assert installer.main([*args, "--check"]) == 1
    assert {home: _entry_snapshot(home) for home in homes} == before
    assert installer.main(args) == 0
    assert installer.main([*args, "--check"]) == 0
    (_target(homes[2]) / "SKILL.md").write_bytes(b"drift")
    assert installer.main([*args, "--check"]) == 1
    invalid_args = _install_args(tmp_path / "missing-source", homes)
    assert installer.main(invalid_args) == 2
    assert "codex" in capsys.readouterr().err


def test_cli_defaults_to_one_canonical_agents_mirror(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    agents_home, claude_home, codex_home = _make_homes(tmp_path)
    _seed_existing_target(_target(claude_home), "claude")
    _seed_existing_target(_target(codex_home), "codex")
    runtime_before = {
        claude_home: _entry_snapshot(claude_home),
        codex_home: _entry_snapshot(codex_home),
    }
    args = _canonical_args(source, agents_home)

    assert installer.main(args) == 0
    assert installer.main([*args, "--check"]) == 0
    assert installer.verify_canonical_install(source, agents_home) == []
    assert {
        claude_home: _entry_snapshot(claude_home),
        codex_home: _entry_snapshot(codex_home),
    } == runtime_before


def test_cli_rejects_runtime_homes_without_explicit_mirrored_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_source(tmp_path / "source")
    homes = _make_homes(tmp_path)
    args = [
        *_canonical_args(source, homes[0]),
        "--claude-home",
        os.fspath(homes[1]),
        "--codex-home",
        os.fspath(homes[2]),
    ]
    before = {home: _entry_snapshot(home) for home in homes}

    assert installer.main(args) == 2
    assert {home: _entry_snapshot(home) for home in homes} == before
    assert "--mirrored" in capsys.readouterr().err


def test_cli_mirrored_mode_requires_both_runtime_homes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_source(tmp_path / "source")
    agents_home = tmp_path / "agents-home"
    agents_home.mkdir()
    before = _entry_snapshot(tmp_path)

    assert installer.main([*_canonical_args(source, agents_home), "--mirrored"]) == 2
    assert _entry_snapshot(tmp_path) == before
    assert "requires --claude-home and --codex-home" in capsys.readouterr().err


def test_module_cli_exits_one_for_drift_without_creating_homes(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    homes = (
        tmp_path / "agents-home",
        tmp_path / "claude-home",
        tmp_path / "codex-home",
    )
    command = [
        sys.executable,
        "-m",
        "agent_tooling.rigorous_theory_search.install",
        *_install_args(source, homes),
        "--check",
    ]
    before = _entry_snapshot(tmp_path)

    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert _entry_snapshot(tmp_path) == before
