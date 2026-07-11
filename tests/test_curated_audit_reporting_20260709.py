from pathlib import Path

import pytest


LEDGER_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "audits"
    / "curated-audit-closure-ledger-2026-07-09.md"
)
LEDGER_COLUMNS = (
    "ID",
    "Class",
    "Test",
    "Command",
    "Commit",
    "Evidence",
    "Status",
)
ALLOWED_CLASSES = {
    "FIXED",
    "FAIL_CLOSED",
    "RELABELED",
    "INTENTIONAL",
    "DEFERRED_PERFORMANCE",
}
ALLOWED_STATUSES = {"OPEN", "CLOSED"}
PRECLOSED_CLASSES = {
    "1": "INTENTIONAL",
    "22": "INTENTIONAL",
    "26": "INTENTIONAL",
    "56": "INTENTIONAL",
    "62": "FIXED",
    "75": "INTENTIONAL",
    "81": "INTENTIONAL",
    "92": "FIXED",
    "93": "INTENTIONAL",
}
DEFERRED_IDS = {f"P{index}" for index in range(1, 7)}
ACTIONABLE_IDS = (
    {str(index) for index in range(1, 108)}
    - set(PRECLOSED_CLASSES)
    | {f"M{index}" for index in range(1, 12)}
    | {f"L{index}" for index in range(1, 9)}
)
EXPECTED_IDS = ACTIONABLE_IDS | set(PRECLOSED_CLASSES) | DEFERRED_IDS
LedgerRow = tuple[str, str, str, str, str, str, str]


def _parse_ledger_rows(ledger_text: str) -> dict[str, LedgerRow]:
    rows: dict[str, LedgerRow] = {}
    header_seen = False
    for line in ledger_text.splitlines():
        if not line.startswith("|"):
            continue
        columns = tuple(column.strip() for column in line.strip("|").split("|"))
        if columns[0] == "ID":
            assert columns == LEDGER_COLUMNS, (
                "ledger header must contain the canonical seven cells"
            )
            header_seen = True
            continue
        if columns[0] == "---":
            continue

        assert len(columns) == 7, (
            f"ledger data row must contain exactly seven cells: {line}"
        )
        identifier, class_name, _, _, _, _, status = columns
        assert class_name in ALLOWED_CLASSES, f"unknown class: {class_name}"
        assert status in ALLOWED_STATUSES, f"unknown status: {status}"
        assert identifier not in rows, f"duplicate ledger identifier: {identifier}"
        rows[identifier] = columns

    assert header_seen, "ledger header is missing"
    assert set(rows) == EXPECTED_IDS, "ledger identifier universe must contain exactly 132 IDs"
    for identifier, class_name in PRECLOSED_CLASSES.items():
        assert rows[identifier][1] == class_name
        assert rows[identifier][-1] == "CLOSED"
    for identifier in DEFERRED_IDS:
        assert rows[identifier][1] == "DEFERRED_PERFORMANCE"
        assert rows[identifier][-1] == "CLOSED"
    for identifier in ACTIONABLE_IDS:
        assert rows[identifier][1] != "DEFERRED_PERFORMANCE"
    return rows


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        (
            "",
            "\n| EXTRA | FIXED | pending | pending | pending | pending | CLOSED |",
            "identifier universe",
        ),
        (
            "| 2 | FIXED | pending | pending | pending | pending | OPEN |",
            "| 2 | FIXED | pending | pending | pending | OPEN |",
            "seven cells",
        ),
        ("| 2 | FIXED |", "| 2 | UNKNOWN |", "unknown class"),
        (
            "| 2 | FIXED | pending | pending | pending | pending | OPEN |",
            "| 2 | FIXED | pending | pending | pending | pending | UNKNOWN |",
            "unknown status",
        ),
    ],
    ids=("extra-id", "malformed-schema", "unknown-class", "unknown-status"),
)
def test_closure_ledger_rejects_invalid_rows(
    needle:      str,
    replacement: str,
    message:     str,
) -> None:
    ledger_text = LEDGER_PATH.read_text(encoding="utf-8")
    if needle:
        assert needle in ledger_text
        ledger_text = ledger_text.replace(needle, replacement, 1)
    else:
        ledger_text += replacement

    with pytest.raises(AssertionError, match=message):
        _parse_ledger_rows(ledger_text)


def test_closure_ledger_allows_actionable_rows_to_close() -> None:
    ledger_text = LEDGER_PATH.read_text(encoding="utf-8")
    open_row = "| 2 | FIXED | pending | pending | pending | pending | OPEN |"
    closed_row = "| 2 | FIXED | pending | pending | pending | pending | CLOSED |"
    assert open_row in ledger_text

    rows = _parse_ledger_rows(ledger_text.replace(open_row, closed_row, 1))

    assert rows["2"][-1] == "CLOSED"


def test_closure_ledger_inventory() -> None:
    rows = _parse_ledger_rows(LEDGER_PATH.read_text(encoding="utf-8"))

    assert set(rows) == EXPECTED_IDS
    assert len(rows) == 132
    assert len(ACTIONABLE_IDS) == 117
    assert len(PRECLOSED_CLASSES) == 9
    assert len(DEFERRED_IDS) == 6
