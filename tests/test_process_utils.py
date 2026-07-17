import subprocess

import pytest

from vfe3 import path_utils, process_utils


class _FakeProcess:
    pid = 123
    returncode = 0
    _handle = 456

    def __init__(self, *, timeout_once: bool = False) -> None:
        self.timeout_once = timeout_once
        self.calls = 0

    def communicate(self, timeout=None):
        self.calls += 1
        if self.timeout_once and self.calls == 1:
            raise subprocess.TimeoutExpired(["worker"], timeout)
        return "stdout", "stderr"


class _InterruptingProcess(_FakeProcess):
    def communicate(self, timeout=None):
        self.calls += 1
        if self.calls == 1:
            raise KeyboardInterrupt
        return "stdout", "stderr"


class _Gate:
    def __init__(self) -> None:
        self.value = None
        self.closed = False

    def write(self, value):
        self.value = value

    def flush(self):
        return None

    def close(self):
        self.closed = True


def test_run_process_tree_uses_fresh_posix_group_and_reaps_descendants(monkeypatch):
    captured = {}
    process = _FakeProcess()

    def _popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return process

    killed = []
    monkeypatch.setattr(process_utils.os, "name", "posix")
    monkeypatch.setattr(process_utils.subprocess, "Popen", _popen)
    monkeypatch.setattr(
        process_utils,
        "_kill_process_tree",
        lambda child, job: killed.append((child, job)),
    )

    completed = process_utils.run_process_tree(["worker"], capture_output=True, text=True)

    assert captured["start_new_session"] is True
    assert completed.stdout == "stdout"
    assert killed == [(process, None)]


def test_run_process_tree_terminates_whole_group_before_timeout_escapes(monkeypatch):
    process = _FakeProcess(timeout_once=True)
    monkeypatch.setattr(process_utils.os, "name", "posix")
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *_args, **_kwargs: process)
    killed = []
    monkeypatch.setattr(
        process_utils,
        "_kill_process_tree",
        lambda child, job: killed.append((child, job)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        process_utils.run_process_tree(["worker"], timeout=1.0)

    assert killed == [(process, None)]
    assert process.calls == 2


def test_run_process_tree_terminates_and_reaps_after_base_exception(monkeypatch):
    process = _InterruptingProcess()
    monkeypatch.setattr(process_utils.os, "name", "posix")
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *_args, **_kwargs: process)
    killed = []
    monkeypatch.setattr(
        process_utils,
        "_kill_process_tree",
        lambda child, job: killed.append((child, job)),
    )

    with pytest.raises(KeyboardInterrupt):
        process_utils.run_process_tree(["worker"])

    assert killed == [(process, None)]
    assert process.calls == 2


def test_owned_output_child_rejects_reparse_point_before_use(tmp_path, monkeypatch):
    redirected = tmp_path / "figures"
    redirected.mkdir()
    monkeypatch.setattr(
        path_utils,
        "path_is_reparse_point",
        lambda path: path == redirected,
    )

    with pytest.raises(ValueError, match="symlink, junction, or reparse point"):
        path_utils.prepare_owned_output_child(
            tmp_path,
            "figures",
            role="single-run figure",
        )


def test_windows_workload_is_released_only_after_job_assignment(monkeypatch):
    process = _FakeProcess()
    gate = _Gate()
    process.stdin = gate
    captured = {}
    events = []

    class _Job:
        def assign(self, child):
            events.append(("assigned", child.stdin.value))

        def close(self):
            events.append(("closed", None))

    def _popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return process

    monkeypatch.setattr(process_utils.os, "name", "nt")
    monkeypatch.setattr(process_utils.subprocess, "Popen", _popen)
    monkeypatch.setattr(process_utils, "_WindowsJob", _Job)

    completed = process_utils.run_process_tree(["worker", "argument"], text=True)

    assert captured["command"][-3:] == ["vfe3-process-gate", "worker", "argument"]
    assert "sys.argv[2:]" in process_utils._WINDOWS_GATED_LAUNCHER
    assert events[0] == ("assigned", None)
    assert events[-1] == ("closed", None)
    assert gate.value == "1" and gate.closed is True
    assert completed.returncode == 0
