from pathlib import Path

import pytest


LEDGER_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "audits"
    / "curated-audit-closure-ledger-2026-07-09.md"
)

pytestmark = pytest.mark.skipif(
    not LEDGER_PATH.exists(),
    reason=(
        "closure ledger document was removed from the repo on 2026-07-12 "
        "(docs cleanup); these tests revive automatically if it returns"
    ),
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


def _format_ledger_row(columns: tuple[str, ...]) -> str:
    return "| " + " | ".join(columns) + " |"


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
    ("mutation", "message"),
    [
        ("extra_id", "identifier universe"),
        ("malformed_schema", "seven cells"),
        ("unknown_class", "unknown class"),
        ("unknown_status", "unknown status"),
    ],
    ids=("extra-id", "malformed-schema", "unknown-class", "unknown-status"),
)
def test_closure_ledger_rejects_invalid_rows(
    mutation: str,
    message:  str,
) -> None:
    ledger_text = LEDGER_PATH.read_text(encoding="utf-8")
    if mutation == "extra_id":
        ledger_text += "\n| EXTRA | FIXED | complete | n/a | n/a | synthetic | CLOSED |"
    else:
        row = _parse_ledger_rows(ledger_text)["2"]
        original = _format_ledger_row(row)
        replacement = list(row)
        if mutation == "malformed_schema":
            replacement = replacement[:-1]
        elif mutation == "unknown_class":
            replacement[1] = "UNKNOWN"
        else:
            replacement[-1] = "UNKNOWN"
        assert original in ledger_text
        ledger_text = ledger_text.replace(
            original, _format_ledger_row(tuple(replacement)), 1)

    with pytest.raises(AssertionError, match=message):
        _parse_ledger_rows(ledger_text)


def test_closure_ledger_allows_actionable_rows_to_close() -> None:
    ledger_text = LEDGER_PATH.read_text(encoding="utf-8")
    closed_row = _parse_ledger_rows(ledger_text)["2"]
    assert closed_row[-1] == "CLOSED"
    open_row = closed_row[:-1] + ("OPEN",)
    closed_text = _format_ledger_row(closed_row)
    open_text = _format_ledger_row(open_row)
    assert closed_text in ledger_text

    open_ledger = ledger_text.replace(closed_text, open_text, 1)
    assert _parse_ledger_rows(open_ledger)["2"][-1] == "OPEN"
    rows = _parse_ledger_rows(open_ledger.replace(open_text, closed_text, 1))

    assert rows["2"][-1] == "CLOSED"


def test_closure_ledger_inventory() -> None:
    rows = _parse_ledger_rows(LEDGER_PATH.read_text(encoding="utf-8"))

    assert set(rows) == EXPECTED_IDS
    assert len(rows) == 132
    assert len(ACTIONABLE_IDS) == 117
    assert len(PRECLOSED_CLASSES) == 9
    assert len(DEFERRED_IDS) == 6
    assert {row[-1] for row in rows.values()} == {"CLOSED"}
    assert all(
        cell.casefold() != "pending"
        for row in rows.values()
        for cell in row[2:6]
    )
