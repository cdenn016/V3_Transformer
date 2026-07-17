# Phi Pullback-Group Build Restart Handoff

## Durable state

The active implementation worktree is
`C:\tmp\vfe3-phi-pullback-group-descent-20260717` on branch
`codex/phi-pullback-group-descent-20260717`. The branch is rooted on the local
approved design and plan commits `deb73ec` and `090bcfd`; remote publication was
denied because the GitHub destination was not verified as trusted or private.
Do not remove the separate investigation worktree at
`C:\tmp\vfe3-mphi-ng-investigation-20260717`.

Task 1 is complete and independently approved. The restart checkpoint
`b0bf9cb` was completed by fix commit `843b45c`; no Critical or Important
review findings remain. Resume at Task 2 rather than dispatching or reviewing
Task 1 again.

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

## Resume sequence

1. Read `.superpowers/sdd/progress.md`, this handoff, and the Task 2 brief.
2. Preserve commits `b0bf9cb` and `843b45c`; do not re-dispatch Task 1.
3. Proceed sequentially from Task 2 using the approved Task 1 public geometry
   interface.
