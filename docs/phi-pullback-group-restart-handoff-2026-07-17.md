# Phi Pullback-Group Build Restart Handoff

## Durable state

The active implementation worktree is
`C:\tmp\vfe3-phi-pullback-group-descent-20260717` on branch
`codex/phi-pullback-group-descent-20260717`. The branch is rooted on the local
approved design and plan commits `deb73ec` and `090bcfd`; remote publication was
denied because the GitHub destination was not verified as trusted or private.
Do not remove the separate investigation worktree at
`C:\tmp\vfe3-mphi-ng-investigation-20260717`.

Task 1 has a tested local implementation in
`vfe3/geometry/phi_preconditioner.py` and
`tests/test_phi_preconditioner.py`. It is not yet independently task-reviewed,
so resume at the Task 1 review gate rather than dispatching Task 1 again.

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

1. Read `.superpowers/sdd/progress.md`, this handoff, and the Task 1 brief.
2. Inspect the Task 1 checkpoint commit and generate a review package from
   `090bcfd` to that commit.
3. Dispatch a fresh read-only Task 1 reviewer for specification compliance and
   code quality. Fix and re-review every Critical or Important finding.
4. Run the Task 1 geometry acceptance matrix while retaining the known stale
   curated-geometry baseline failure as an explicit exception.
5. Mark Task 1 complete only after review, then proceed sequentially to Task 2.
