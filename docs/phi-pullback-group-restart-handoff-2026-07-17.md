# Phi Pullback-Group Build Restart Handoff

## Durable state

The active implementation worktree is
`C:\tmp\vfe3-phi-pullback-group-descent-20260717` on branch
`codex/phi-pullback-group-descent-20260717`. The branch is rooted on the local
approved design and plan commits `deb73ec` and `090bcfd`; remote publication was
denied because the GitHub destination was not verified as trusted or private.
Do not remove the separate investigation worktree at
`C:\tmp\vfe3-mphi-ng-investigation-20260717`.

Tasks 1 through 4 are complete and independently approved. The Task 1 restart
checkpoint `b0bf9cb` was completed by fix commit `843b45c`. Task 2 is recorded
by runtime commit `71e5481` and atomic nonfinite-covector review fix `085777f`.
Task 3 is recorded by migration commit `43ed6d3` and live-topology slot-schema
review fix `045a245`. Task 4 is recorded by integration-test commit `574602f`
and real-training-seam review fix `6ad1cdc`. No Critical or Important findings
remain in these tasks. Resume at Task 5 rather than dispatching or reviewing
Tasks 1 through 4 again.

## Verification captured before restart

- `python -m pytest tests/test_phi_preconditioner.py -k pullback_group`: 20
  passed and 24 deselected.
- `python -m pytest tests/test_phi_preconditioner.py -k
  estep_preconditioner_modes_are_byte_identical`: one passed and 43
  deselected.
- `python -m pytest tests/test_phi_preconditioner.py`: 44 passed with one
  warning from the legacy warning-only E-step series test.
- Accepted pre-edit full-suite baseline: 3,772 tests, 14 failures, zero errors,
  and 17 skips. The user authorized proceeding while requiring zero new
  failures.

Machine-readable checkpoint artifacts live outside the repository at
`C:\tmp\vfe3-phi-task1-checkpoint-20260717.xml`,
`C:\tmp\vfe3-estep-phi-checkpoint-20260717.xml`, and
`C:\tmp\vfe3-phi-task1-file-checkpoint-20260717.xml`.

Task 2 evidence is at
`C:\tmp\vfe3-phi-task2-review-red-20260717.xml`,
`C:\tmp\vfe3-phi-task2-review-green-20260717.xml`, and
`C:\tmp\vfe3-phi-task2-review-focused-20260717.xml`. The focused artifact
records 531 tests, with 530 passes and the one accepted baseline failure.

Task 3 evidence is at `C:\tmp\vfe3-phi-migration-red-20260717.xml`,
`C:\tmp\vfe3-phi-migration-slot-bypass-red-20260717.xml`,
`C:\tmp\vfe3-phi-migration-slot-bypass-green-20260717.xml`, and
`C:\tmp\vfe3-phi-migration-review-green-20260717.xml`. The final focused
artifact records 384 tests, zero failures, zero errors, and two skips.

Task 4 evidence is at `C:\tmp\vfe3-phi-task4-review-red-20260717.xml`,
`C:\tmp\vfe3-phi-task4-review-green-20260717.xml`, and
`C:\tmp\vfe3-phi-integration-review-fix-green-20260717.xml`. The final exact
eight-file artifact records 162 tests, zero failures, zero errors, and zero
skips.

## Resume sequence

1. Read `.superpowers/sdd/progress.md`, this handoff, and the Task 5 section of
   the approved implementation plan.
2. Preserve commits `b0bf9cb`, `843b45c`, `71e5481`, `085777f`, `43ed6d3`, and
   `045a245`, `574602f`, and `6ad1cdc`; do not re-dispatch Tasks 1 through 4.
3. Proceed sequentially from Task 5 using the approved strict geometry,
   stateless runtime, and serialized migration interfaces.
