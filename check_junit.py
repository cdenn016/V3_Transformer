"""Shared machine-readable pytest accounting for click-to-run verification scripts."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Sequence
from xml.etree import ElementTree


_COUNT_FIELDS = ("tests", "failures", "errors", "skipped")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_junit_counts(path: Path) -> Dict[str, int]:
    """Return aggregate pytest counts from either supported JUnit root form."""
    root = ElementTree.parse(path).getroot()
    root_name = _local_name(root.tag)
    if root_name not in ("testsuite", "testsuites"):
        raise ValueError(f"unsupported JUnit root element {root_name!r}")

    suites = [element for element in root.iter() if _local_name(element.tag) == "testsuite"]
    if not suites:
        raise ValueError("JUnit XML contains no testsuite elements")

    counts = {field: 0 for field in _COUNT_FIELDS}
    for suite in suites:
        for field in _COUNT_FIELDS:
            try:
                value = int(suite.attrib.get(field, "0"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid JUnit {field!r} count") from exc
            if value < 0:
                raise ValueError(f"JUnit {field!r} count must be nonnegative")
            counts[field] += value

    counts["passes"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    if counts["passes"] < 0:
        raise ValueError("JUnit failure/error/skip counts exceed the test count")
    return counts


def run_pytest_junit(
    args: Sequence[str],

    *,
    prefix: str,
) -> tuple[int, Dict[str, int]]:
    """Run pytest with a temporary JUnit report, return its exit code and derived counts."""
    cli_args = [str(arg) for arg in args]
    if any(arg.startswith(("--junitxml", "--junit-xml")) for arg in cli_args):
        raise ValueError("run_pytest_junit owns the temporary --junitxml argument")

    handle = NamedTemporaryFile(prefix=prefix, suffix=".xml", delete=False)
    junit_path = Path(handle.name)
    handle.close()
    try:
        import pytest

        code = int(pytest.main([*cli_args, f"--junitxml={junit_path}"]))
        counts = read_junit_counts(junit_path)
        return code, counts
    finally:
        junit_path.unlink(missing_ok=True)
