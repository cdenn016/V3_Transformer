"""Subprocess helpers that contain every descendant in one disposable process tree."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
from ctypes import wintypes
from typing import Mapping, Optional, Sequence


_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_WINDOWS_GATED_LAUNCHER = (
    "import subprocess,sys; "
    "gate=sys.stdin.buffer.read(1); "
    "sys.exit(125) if gate != b'1' else None; "
    "sys.exit(126) if len(sys.argv) < 3 or sys.argv[1] != 'vfe3-process-gate' else None; "
    "child=subprocess.Popen(sys.argv[2:]); "
    "sys.exit(child.wait())"
)


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_STRUCT(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsJob:
    """Own one Windows Job Object whose closure kills all assigned descendants."""

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_STRUCT()
        information.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise ctypes.WinError(error)
        self._kernel32 = kernel32
        self._handle = handle

    def assign(self, process: subprocess.Popen[str]) -> None:
        if not self._kernel32.AssignProcessToJobObject(self._handle, process._handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self, exit_code: int = 1) -> None:
        if self._handle:
            self._kernel32.TerminateJobObject(self._handle, exit_code)

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _kill_process_tree(
    process: subprocess.Popen[str],
    job:     Optional[_WindowsJob],
) -> None:
    """Terminate a child and every descendant without trusting the child to cooperate."""
    if os.name == "nt":
        if job is not None:
            job.terminate()
            return
        subprocess.Popen(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).wait()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _terminate_and_reap_after_interruption(
    process: subprocess.Popen[str],
    job:     Optional[_WindowsJob],
) -> None:
    """Best-effort tree termination and reaping without masking an active ``BaseException``."""
    try:
        _kill_process_tree(process, job)
    except BaseException:
        try:
            process.kill()
        except BaseException:
            pass
    try:
        process.communicate()
    except BaseException:
        pass


def run_process_tree(
    command: Sequence[str],

    *,
    creationflags:  int                         = 0,
    capture_output: bool                        = False,
    text:           bool                        = False,
    cwd:            Optional[str]               = None,
    env:            Optional[Mapping[str, str]] = None,
    timeout:        Optional[float]             = None,
    encoding:       Optional[str]               = None,
    errors:         Optional[str]               = None,
) -> subprocess.CompletedProcess[str]:
    r"""Run ``command`` in a process tree that is destroyed on timeout or parent exit.

    Windows descendants are assigned to a Job Object with
    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. POSIX descendants share a fresh process group. This
    prevents native plotting workers, UMAP helpers, or xdist workers from surviving a timeout.
    """
    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    popen_kwargs: dict[str, object] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            int(creationflags) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True

    launch_command = (
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _WINDOWS_GATED_LAUNCHER,
            "vfe3-process-gate",
            *command,
        ]
        if os.name == "nt"
        else list(command)
    )
    process = subprocess.Popen(
        launch_command,
        cwd=cwd,
        env=(dict(env) if env is not None else None),
        stdin=(subprocess.PIPE if os.name == "nt" else None),
        stdout=stdout,
        stderr=stderr,
        text=text,
        encoding=encoding,
        errors=errors,
        **popen_kwargs,
    )
    job: Optional[_WindowsJob] = None
    try:
        if os.name == "nt":
            try:
                job = _WindowsJob()
                job.assign(process)
            except OSError as exc:
                if job is not None:
                    job.close()
                job = None
                _kill_process_tree(process, None)
                process.communicate()
                raise OSError(
                    "could not contain the child in a Windows Job Object"
                ) from exc
            gate = process.stdin
            if gate is None:
                _kill_process_tree(process, job)
                process.communicate()
                raise OSError("Windows process gate was not created")
            gate.write("1" if text else b"1")
            gate.flush()
            gate.close()
            process.stdin = None
        try:
            output, error = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_process_tree(process, job)
            try:
                output, error = process.communicate()
            except BaseException:
                _terminate_and_reap_after_interruption(process, job)
                raise
            exc.output = output
            exc.stderr = error
            raise
        except BaseException:
            _terminate_and_reap_after_interruption(process, job)
            raise
        completed = subprocess.CompletedProcess(list(command), process.returncode, output, error)
        if os.name != "nt":
            _kill_process_tree(process, None)
        return completed
    finally:
        if job is not None:
            job.close()
