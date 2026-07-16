import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

import pytest

import run_cpu_tests


_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _junit_path(command: list[str]) -> Path:
    argument = next(part for part in command if part.startswith("--junitxml="))
    return Path(argument.split("=", 1)[1])


def _passing_junit(path: Path, *, tests: int = 1) -> None:
    path.write_text(
        (
            f'<testsuites><testsuite tests="{tests}" failures="0" '
            'errors="0" skipped="0"/></testsuites>'
        ),
        encoding="utf-8",
    )


def _expected_logical_cpu_count() -> int:
    return 24


def test_resolve_cpu_workers_accepts_all_physical_cores_on_expected_host() -> None:
    assert run_cpu_tests.resolve_cpu_workers(12, 24) == 12


@pytest.mark.parametrize("workers", [True, False, 0, -1, 25])
def test_resolve_cpu_workers_rejects_invalid_counts(workers: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_cpu_tests.resolve_cpu_workers(workers, 24)


def test_fast_lane_command_uses_explicit_workers_and_marker_union(tmp_path: Path) -> None:
    junit_path = tmp_path / "fast.xml"

    command = run_cpu_tests.build_cpu_lane_command("fast", junit_path, 24)

    assert command[:3] == [sys.executable, "-m", "pytest"]
    assert command[3:7] == ["-n", "12", "--dist", "loadscope"]
    assert command[-3:-1] == ["-m", "not slow and not cuda and not external"]
    assert f"--junitxml={junit_path}" in command
    assert "auto" not in command


def test_slow_lane_command_uses_explicit_workers_and_marker_union(tmp_path: Path) -> None:
    junit_path = tmp_path / "slow.xml"

    command = run_cpu_tests.build_cpu_lane_command("slow", junit_path, 24)

    assert command[:4] == [sys.executable, "-m", "pytest", "--runslow"]
    assert command[4:8] == ["-n", "3", "--dist", "loadgroup"]
    assert command[-3:-1] == ["-m", "slow and not cuda and not external"]
    assert f"--junitxml={junit_path}" in command
    assert "auto" not in command


def test_build_cpu_environment_caps_threads_without_mutating_parent() -> None:
    parent = {
        "UNCHANGED": "sentinel",
        "OMP_NUM_THREADS": "64",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    original = parent.copy()

    child = run_cpu_tests.build_cpu_environment(parent)

    assert parent == original
    assert child is not parent
    assert child["UNCHANGED"] == "sentinel"
    assert child["VFE3_TEST_DEVICE"] == "cpu"
    assert child["CUDA_VISIBLE_DEVICES"] == "-1"
    assert {name: child[name] for name in _THREAD_VARIABLES} == {
        name: "1" for name in _THREAD_VARIABLES
    }


def test_run_lane_uses_fresh_subprocess_and_junit_path_per_lane(
    monkeypatch: pytest.MonkeyPatch,
    capsys:      pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[list[str], Mapping[str, str], bool]] = []
    child_environment = run_cpu_tests.build_cpu_environment({"PARENT": "copied"})

    def _fake_run(
        command: list[str],

        *,
        env:   Mapping[str, str],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        copied_command = list(command)
        calls.append((copied_command, env, check))
        _passing_junit(_junit_path(copied_command), tests=len(calls))
        return subprocess.CompletedProcess(copied_command, 0, stdout="999 passed")

    monkeypatch.setattr(run_cpu_tests.os, "cpu_count", _expected_logical_cpu_count)
    monkeypatch.setattr(run_cpu_tests.subprocess, "run", _fake_run)

    assert run_cpu_tests.run_lane("fast", child_environment) == 0
    assert run_cpu_tests.run_lane("slow", child_environment) == 0

    assert len(calls) == 2
    paths = [_junit_path(command) for command, _, _ in calls]
    assert paths[0] != paths[1]
    assert all(path.parent == Path(tempfile.gettempdir()) for path in paths)
    assert all(command[:3] == [sys.executable, "-m", "pytest"] for command, _, _ in calls)
    assert all(environment == child_environment for _, environment, _ in calls)
    assert all(environment is not child_environment for _, environment, _ in calls)
    assert all(check is False for _, _, check in calls)
    assert all(not path.exists() for path in paths)
    output = capsys.readouterr().out
    assert "fast: tests=1 failures=0 errors=0 skipped=0" in output
    assert "slow: tests=2 failures=0 errors=0 skipped=0" in output
    assert "999 passed" not in output


def test_main_stops_after_first_nonzero_lane_and_passes_exact_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_environment = {"EXACT": "child"}
    calls: list[tuple[str, Mapping[str, str]]] = []

    def _fake_build_cpu_environment(_parent: Mapping[str, str]) -> dict[str, str]:
        return child_environment

    def _fake_run_lane(lane: str, environment: Mapping[str, str]) -> int:
        calls.append((lane, environment))
        return 7

    monkeypatch.setattr(
        run_cpu_tests,
        "build_cpu_environment",
        _fake_build_cpu_environment,
    )
    monkeypatch.setattr(run_cpu_tests, "run_lane", _fake_run_lane)

    assert run_cpu_tests.main() == 7
    assert calls == [("fast", child_environment)]
    assert calls[0][1] is child_environment


def test_run_lane_cleans_junit_path_when_subprocess_launch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    junit_paths: list[Path] = []

    def _raise_oserror(
        command: list[str],

        *,
        env:   Mapping[str, str],
        check: bool,
    ) -> NoReturn:
        junit_paths.append(_junit_path(list(command)))
        assert env == {"EXACT": "child"}
        assert check is False
        raise OSError("cannot launch pytest")

    monkeypatch.setattr(run_cpu_tests.os, "cpu_count", _expected_logical_cpu_count)
    monkeypatch.setattr(run_cpu_tests.subprocess, "run", _raise_oserror)

    assert run_cpu_tests.run_lane("fast", {"EXACT": "child"}) == 1
    assert len(junit_paths) == 1
    assert not junit_paths[0].exists()


@pytest.mark.parametrize(
    ("xml_payload", "pytest_status"),
    [
        (None, 0),
        ("<testsuites>", 0),
        ('<testsuite tests="1" failures="1" errors="0" skipped="0"/>', 0),
        ('<testsuite tests="1" failures="0" errors="1" skipped="0"/>', 0),
        ('<testsuite tests="1" failures="0" errors="0" skipped="0"/>', 5),
    ],
)
def test_run_lane_rejects_missing_malformed_failing_or_nonzero_results(
    monkeypatch:   pytest.MonkeyPatch,
    xml_payload:   str | None,
    pytest_status: int,
) -> None:
    junit_paths: list[Path] = []

    def _fake_run(
        command: list[str],

        *,
        env:   Mapping[str, str],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert env == {"EXACT": "child"}
        assert check is False
        junit_paths.append(_junit_path(list(command)))
        if xml_payload is not None:
            junit_paths[-1].write_text(xml_payload, encoding="utf-8")
        return subprocess.CompletedProcess(command, pytest_status)

    monkeypatch.setattr(run_cpu_tests.os, "cpu_count", _expected_logical_cpu_count)
    monkeypatch.setattr(run_cpu_tests.subprocess, "run", _fake_run)

    assert run_cpu_tests.run_lane("fast", {"EXACT": "child"}) != 0
    assert len(junit_paths) == 1
    assert not junit_paths[0].exists()
