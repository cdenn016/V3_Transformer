# Curated Audit Closure Ledger — 2026-07-09

This ledger is the canonical closure inventory for the 2026-07-09 curated audit salvage. It combines Findings 1–107 from `ultradeep-audit-findings-investigation-2026-07-09.md` with M1–M11 and L1–L8 from the frozen `deep-audit-and-wikitext103-performance-investigation-2026-07-09.md` source. The imported source has SHA-256 `1D80E75C99DDC942AAFE3FA96D824D98C593DC41472F07C475C9AE63318E961F`.

The Class value is provisional while a row is `OPEN` and final once the row is `CLOSED`. Closed classifications are restricted to `FIXED`, `FAIL_CLOSED`, `RELABELED`, `INTENTIONAL`, and `DEFERRED_PERFORMANCE`. The second audit's addenda remain nested in Findings 14 and 87 and do not introduce extra identifiers.

| ID | Class | Test | Command | Commit | Evidence | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | INTENTIONAL | existing audit ruling | n/a | e504f1c | user-owned route labels | CLOSED |
| 2 | FIXED | pending | pending | pending | pending | OPEN |
| 3 | FIXED | pending | pending | pending | pending | OPEN |
| 4 | FIXED | pending | pending | pending | pending | OPEN |
| 5 | FIXED | pending | pending | pending | pending | OPEN |
| 6 | FIXED | pending | pending | pending | pending | OPEN |
| 7 | FIXED | pending | pending | pending | pending | OPEN |
| 8 | FIXED | pending | pending | pending | pending | OPEN |
| 9 | FIXED | pending | pending | pending | pending | OPEN |
| 10 | FIXED | pending | pending | pending | pending | OPEN |
| 11 | FIXED | pending | pending | pending | pending | OPEN |
| 12 | FIXED | pending | pending | pending | pending | OPEN |
| 13 | FIXED | pending | pending | pending | pending | OPEN |
| 14 | FIXED | pending | pending | pending | pending; addendum: remove host synchronization and unbounded ownership in the bracket-closure cache | OPEN |
| 15 | FIXED | pending | pending | pending | pending | OPEN |
| 16 | FIXED | pending | pending | pending | pending | OPEN |
| 17 | FIXED | pending | pending | pending | pending | OPEN |
| 18 | FIXED | pending | pending | pending | pending | OPEN |
| 19 | FIXED | pending | pending | pending | pending | OPEN |
| 20 | FIXED | pending | pending | pending | pending | OPEN |
| 21 | FIXED | pending | pending | pending | pending | OPEN |
| 22 | INTENTIONAL | existing audit ruling | n/a | e504f1c | 2026-07-09 exclusion-ledger ruling | CLOSED |
| 23 | FIXED | pending | pending | pending | pending | OPEN |
| 24 | FIXED | pending | pending | pending | pending | OPEN |
| 25 | FIXED | pending | pending | pending | pending | OPEN |
| 26 | INTENTIONAL | existing audit ruling | n/a | e504f1c | 2026-07-09 exclusion-ledger ruling | CLOSED |
| 27 | FIXED | pending | pending | pending | pending | OPEN |
| 28 | FIXED | pending | pending | pending | pending | OPEN |
| 29 | FIXED | pending | pending | pending | pending | OPEN |
| 30 | FIXED | pending | pending | pending | pending | OPEN |
| 31 | FIXED | pending | pending | pending | pending | OPEN |
| 32 | FIXED | pending | pending | pending | pending | OPEN |
| 33 | FIXED | pending | pending | pending | pending | OPEN |
| 34 | FIXED | pending | pending | pending | pending | OPEN |
| 35 | FIXED | pending | pending | pending | pending | OPEN |
| 36 | FIXED | pending | pending | pending | pending | OPEN |
| 37 | FIXED | pending | pending | pending | pending | OPEN |
| 38 | FIXED | pending | pending | pending | pending | OPEN |
| 39 | FIXED | pending | pending | pending | pending | OPEN |
| 40 | FIXED | pending | pending | pending | pending | OPEN |
| 41 | FIXED | pending | pending | pending | pending | OPEN |
| 42 | FIXED | pending | pending | pending | pending | OPEN |
| 43 | FIXED | pending | pending | pending | pending | OPEN |
| 44 | FIXED | pending | pending | pending | pending | OPEN |
| 45 | FIXED | pending | pending | pending | pending | OPEN |
| 46 | FIXED | pending | pending | pending | pending | OPEN |
| 47 | FIXED | pending | pending | pending | pending | OPEN |
| 48 | FIXED | pending | pending | pending | pending | OPEN |
| 49 | FIXED | pending | pending | pending | pending | OPEN |
| 50 | FIXED | pending | pending | pending | pending | OPEN |
| 51 | FIXED | pending | pending | pending | pending | OPEN |
| 52 | FIXED | pending | pending | pending | pending | OPEN |
| 53 | FIXED | pending | pending | pending | pending | OPEN |
| 54 | FIXED | pending | pending | pending | pending | OPEN |
| 55 | FIXED | pending | pending | pending | pending | OPEN |
| 56 | INTENTIONAL | existing audit ruling | n/a | e504f1c | 2026-07-09 exclusion-ledger ruling | CLOSED |
| 57 | FIXED | pending | pending | pending | pending | OPEN |
| 58 | FIXED | pending | pending | pending | pending | OPEN |
| 59 | FIXED | pending | pending | pending | pending | OPEN |
| 60 | FIXED | pending | pending | pending | pending | OPEN |
| 61 | FIXED | pending | pending | pending | pending | OPEN |
| 62 | FIXED | tests/test_omega_direct.py | python -m pytest tests/test_omega_direct.py | 0f0ffd3 | BeliefState._replace preserves omega | CLOSED |
| 63 | FIXED | pending | pending | pending | pending | OPEN |
| 64 | FIXED | pending | pending | pending | pending | OPEN |
| 65 | FIXED | pending | pending | pending | pending | OPEN |
| 66 | FIXED | pending | pending | pending | pending | OPEN |
| 67 | FIXED | pending | pending | pending | pending | OPEN |
| 68 | FIXED | pending | pending | pending | pending | OPEN |
| 69 | FIXED | pending | pending | pending | pending | OPEN |
| 70 | FIXED | pending | pending | pending | pending | OPEN |
| 71 | FIXED | pending | pending | pending | pending | OPEN |
| 72 | FIXED | pending | pending | pending | pending | OPEN |
| 73 | FIXED | pending | pending | pending | pending | OPEN |
| 74 | FIXED | pending | pending | pending | pending | OPEN |
| 75 | INTENTIONAL | existing audit ruling | n/a | e504f1c | 2026-07-09 exclusion-ledger ruling | CLOSED |
| 76 | FIXED | pending | pending | pending | pending | OPEN |
| 77 | FIXED | pending | pending | pending | pending | OPEN |
| 78 | FIXED | pending | pending | pending | pending | OPEN |
| 79 | FIXED | pending | pending | pending | pending | OPEN |
| 80 | FIXED | pending | pending | pending | pending | OPEN |
| 81 | INTENTIONAL | existing audit ruling | n/a | e504f1c | 2026-07-09 exclusion-ledger ruling | CLOSED |
| 82 | FIXED | pending | pending | pending | pending | OPEN |
| 83 | FIXED | pending | pending | pending | pending | OPEN |
| 84 | FIXED | pending | pending | pending | pending | OPEN |
| 85 | FIXED | pending | pending | pending | pending | OPEN |
| 86 | FIXED | pending | pending | pending | pending | OPEN |
| 87 | FIXED | pending | pending | pending | pending; addendum: require a successful terminal state before an ablation cell is current | OPEN |
| 88 | FIXED | pending | pending | pending | pending | OPEN |
| 89 | FIXED | pending | pending | pending | pending | OPEN |
| 90 | FIXED | pending | pending | pending | pending | OPEN |
| 91 | FIXED | pending | pending | pending | pending | OPEN |
| 92 | FIXED | tests/test_omega_direct.py | python -m pytest tests/test_omega_direct.py | 0f0ffd3 | BeliefState._replace preserves omega | CLOSED |
| 93 | INTENTIONAL | existing audit ruling | n/a | e504f1c | 2026-07-09 exclusion-ledger ruling | CLOSED |
| 94 | FIXED | pending | pending | pending | pending | OPEN |
| 95 | FIXED | pending | pending | pending | pending | OPEN |
| 96 | FIXED | pending | pending | pending | pending | OPEN |
| 97 | FIXED | pending | pending | pending | pending | OPEN |
| 98 | FIXED | pending | pending | pending | pending | OPEN |
| 99 | FIXED | pending | pending | pending | pending | OPEN |
| 100 | FIXED | pending | pending | pending | pending | OPEN |
| 101 | FIXED | pending | pending | pending | pending | OPEN |
| 102 | FIXED | pending | pending | pending | pending | OPEN |
| 103 | FIXED | pending | pending | pending | pending | OPEN |
| 104 | FIXED | pending | pending | pending | pending | OPEN |
| 105 | FIXED | pending | pending | pending | pending | OPEN |
| 106 | FIXED | pending | pending | pending | pending | OPEN |
| 107 | FIXED | pending | pending | pending | pending | OPEN |
| M1 | FIXED | pending | pending | pending | pending | OPEN |
| M2 | FIXED | pending | pending | pending | pending | OPEN |
| M3 | FIXED | pending | pending | pending | pending | OPEN |
| M4 | FIXED | pending | pending | pending | pending | OPEN |
| M5 | FIXED | pending | pending | pending | pending | OPEN |
| M6 | FIXED | pending | pending | pending | pending | OPEN |
| M7 | FIXED | pending | pending | pending | pending | OPEN |
| M8 | FIXED | pending | pending | pending | pending | OPEN |
| M9 | FIXED | pending | pending | pending | pending | OPEN |
| M10 | FIXED | pending | pending | pending | pending | OPEN |
| M11 | FIXED | pending | pending | pending | pending | OPEN |
| L1 | FIXED | pending | pending | pending | pending | OPEN |
| L2 | FIXED | pending | pending | pending | pending | OPEN |
| L3 | FIXED | pending | pending | pending | pending | OPEN |
| L4 | FIXED | pending | pending | pending | pending | OPEN |
| L5 | FIXED | pending | pending | pending | pending | OPEN |
| L6 | FIXED | pending | pending | pending | pending | OPEN |
| L7 | FIXED | pending | pending | pending | pending | OPEN |
| L8 | FIXED | pending | pending | pending | pending | OPEN |
| P1 | DEFERRED_PERFORMANCE | n/a | n/a | n/a | user-directed performance branch | CLOSED |
| P2 | DEFERRED_PERFORMANCE | n/a | n/a | n/a | user-directed performance branch | CLOSED |
| P3 | DEFERRED_PERFORMANCE | n/a | n/a | n/a | user-directed performance branch | CLOSED |
| P4 | DEFERRED_PERFORMANCE | n/a | n/a | n/a | user-directed performance branch | CLOSED |
| P5 | DEFERRED_PERFORMANCE | n/a | n/a | n/a | user-directed performance branch | CLOSED |
| P6 | DEFERRED_PERFORMANCE | n/a | n/a | n/a | user-directed performance branch | CLOSED |
