from pathlib import Path


LEDGER_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "audits"
    / "curated-audit-closure-ledger-2026-07-09.md"
)


def test_closure_ledger_inventory() -> None:
    rows: list[tuple[str, str, str]] = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if columns[0] in {"ID", "---"}:
            continue
        rows.append((columns[0], columns[1], columns[-1]))

    preclosed_classes = {
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
    deferred_ids = {f"P{index}" for index in range(1, 7)}
    actionable_ids = (
        {str(index) for index in range(1, 108)}
        - set(preclosed_classes)
        | {f"M{index}" for index in range(1, 12)}
        | {f"L{index}" for index in range(1, 9)}
    )

    row_ids = [identifier for identifier, _, _ in rows]
    assert len(row_ids) == len(set(row_ids))
    assert {identifier for identifier, _, status in rows if status == "OPEN"} == actionable_ids
    assert len(actionable_ids) == 117
    assert {
        identifier: class_name
        for identifier, class_name, status in rows
        if status == "CLOSED" and identifier in preclosed_classes
    } == preclosed_classes
    assert {
        identifier
        for identifier, class_name, status in rows
        if class_name == "DEFERRED_PERFORMANCE" and status == "CLOSED"
    } == deferred_ids
