from __future__ import annotations

import builtins
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
_NUMERICAL_IMPORTS = {"matplotlib", "numpy", "torch", "vfe3"}
_DRIVERS = (
    ("compare_vocab_figures.py", "_VFE3_COMPARE_VOCAB_FIGURES_CHILD"),
    ("multiseed_analysis.py", "_VFE3_MULTISEED_ANALYSIS_CHILD"),
    ("scaling_analysis.py", "_VFE3_SCALING_ANALYSIS_CHILD"),
)


class _NumericalImportReached(RuntimeError):
    pass


@pytest.mark.parametrize(("filename", "child_sentinel"), _DRIVERS)
def test_click_to_run_parent_is_stdlib_only_and_scopes_openmp_to_child(
    filename:       str,
    child_sentinel: str,
    monkeypatch:    pytest.MonkeyPatch,
    capsys:         pytest.CaptureFixture[str],
) -> None:
    driver = (ROOT / filename).resolve()
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.delenv(child_sentinel, raising=False)

    imported: list[str] = []
    calls: list[tuple[list[str], dict[str, Any]]] = []
    real_import = builtins.__import__

    def _record_import(
        name:     str,
        globals_: Any = None,
        locals_:  Any = None,
        fromlist: Any = (),
        level:    int = 0,
    ) -> Any:
        imported.append(name.partition(".")[0])
        return real_import(name, globals_, locals_, fromlist, level)

    def _record_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(
            command,
            7,
            stdout="isolated stdout\n",
            stderr="isolated stderr\n",
        )

    monkeypatch.setattr(builtins, "__import__", _record_import)
    monkeypatch.setattr(subprocess, "run", _record_run)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(driver), run_name="__main__")

    assert exc_info.value.code == 7
    assert len(calls) == 1
    command, kwargs = calls[0]
    child_environment = kwargs["env"]
    assert command == [sys.executable, str(driver)]
    assert kwargs["cwd"] == str(ROOT)
    assert kwargs["check"] is False
    assert child_environment is not os.environ
    assert child_environment[child_sentinel] == "1"
    assert child_environment["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert child_environment["PYTHONUNBUFFERED"] == "1"
    assert child_sentinel not in os.environ
    assert "KMP_DUPLICATE_LIB_OK" not in os.environ
    assert not (_NUMERICAL_IMPORTS & set(imported))
    captured = capsys.readouterr()
    assert captured.out == "isolated stdout\n"
    assert captured.err == "isolated stderr\n"


@pytest.mark.parametrize(("filename", "child_sentinel"), _DRIVERS)
def test_isolated_child_enables_duplicate_openmp_before_numerical_imports(
    filename:       str,
    child_sentinel: str,
    monkeypatch:    pytest.MonkeyPatch,
) -> None:
    driver = (ROOT / filename).resolve()
    monkeypatch.setenv(child_sentinel, "1")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)

    observed: list[str] = []
    real_import = builtins.__import__

    def _stop_at_numerical_import(
        name:     str,
        globals_: Any = None,
        locals_:  Any = None,
        fromlist: Any = (),
        level:    int = 0,
    ) -> Any:
        top_level = name.partition(".")[0]
        if top_level in _NUMERICAL_IMPORTS:
            observed.append(top_level)
            assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"
            raise _NumericalImportReached(name)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _stop_at_numerical_import)

    with pytest.raises(_NumericalImportReached):
        runpy.run_path(str(driver), run_name="__main__")

    assert observed
