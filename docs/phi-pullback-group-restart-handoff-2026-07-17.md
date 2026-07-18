# Phi Pullback-Group Build Restart Handoff

## Durable state

The active implementation worktree is
`C:\tmp\vfe3-phi-pullback-group-descent-20260717` on branch
`codex/phi-pullback-group-descent-20260717`. The branch is rooted on the local
approved design and plan commits `deb73ec` and `090bcfd`; remote publication was
denied because the GitHub destination was not verified as trusted or private.
Do not remove the separate investigation worktree at
`C:\tmp\vfe3-mphi-ng-investigation-20260717`.

Tasks 1 and 2 are complete and independently approved. The Task 1 restart
checkpoint `b0bf9cb` was completed by fix commit `843b45c`. Task 2 is recorded
by runtime commit `71e5481` and atomic nonfinite-covector review fix `085777f`.
No Critical or Important findings remain in either task. Resume at Task 3 rather
than dispatching or reviewing Tasks 1 or 2 again.

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
records 531 tests, with 530 passes and the one accepted baseline failure. Ten
legacy expectation nodes remain assigned to Tasks 3 through 5 in
`.superpowers/sdd/progress.md`.

## Resume sequence

1. Read `.superpowers/sdd/progress.md`, this handoff, and the Task 3 section of
   the approved implementation plan.
2. Preserve commits `b0bf9cb`, `843b45c`, `71e5481`, and `085777f`; do not
   re-dispatch Tasks 1 or 2.
3. Proceed sequentially from Task 3 using the approved strict geometry and
   stateless runtime interfaces.
