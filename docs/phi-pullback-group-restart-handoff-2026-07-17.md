# Phi Pullback-Group Build Restart Handoff

## Durable state

The active implementation worktree is
`C:\tmp\vfe3-phi-pullback-group-descent-20260717` on branch
`codex/phi-pullback-group-descent-20260717`. The branch is rooted on the local
approved design and plan commits `deb73ec` and `090bcfd`; remote publication was
denied because the GitHub destination was not verified as trusted or private.
Do not remove the separate investigation worktree at
`C:\tmp\vfe3-mphi-ng-investigation-20260717`.

Tasks 1 through 5 are complete and independently approved. The Task 1 restart
checkpoint `b0bf9cb` was completed by fix commit `843b45c`. Task 2 is recorded
by runtime commit `71e5481` and atomic nonfinite-covector review fix `085777f`.
Task 3 is recorded by migration commit `43ed6d3` and live-topology slot-schema
review fix `045a245`. Task 4 is recorded by integration-test commit `574602f`
and real-training-seam review fix `6ad1cdc`. No Critical or Important findings
remain in these tasks. Task 5 is recorded by cleanup commit `2ae3ce2` and
review-fix commit `023686f`. Resume at Task 6 rather than dispatching or
reviewing Tasks 1 through 5 again.

Task 6 is at a pre-CUDA checkpoint. Preserve the uncommitted task-owned files
`benchmarks/benchmark_phi_pullback_group.py` and
`tests/test_phi_pullback_group_benchmark.py`. The missing-module RED artifact
is `C:\tmp\vfe3-phi-benchmark-red-20260717.xml`. The project-interpreter GREEN
artifact is `C:\tmp\vfe3-phi-benchmark-green-20260717.xml`, and the actual
CUDA-interpreter GREEN artifact is
`C:\tmp\vfe3-phi-benchmark-cuda-env-green-20260717.xml`; both GREEN artifacts
record eight tests, zero failures, zero errors, and zero skips.

Use `C:\anaconda\python.exe` for the real run. It reports Python 3.12.7,
PyTorch `2.10.0.dev20251210+cu128`, CUDA available, and NVIDIA GeForce RTX
5090. Do not use the default `C:\Python314\python.exe` or the live `.venv` for
CUDA; both PyTorch builds are CPU-only. The real run was deliberately paused
because unrelated PID `25088` was using approximately 7.3 GiB and 85 percent
of the GPU. Do not stop that process. After restart, verify the GPU is idle,
rerun the eight-test contract under the Anaconda interpreter, then run
`C:\anaconda\python.exe benchmarks\benchmark_phi_pullback_group.py`. No
success or temporary JSON existed at the checkpoint; do not commit Task 6
unless the fixed 20 percent UCB gate passes all three cases.

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

Task 5 evidence is at `C:\tmp\vfe3-phi-cleanup-contract-red-20260717.xml`,
`C:\tmp\vfe3-phi-cleanup-green-20260717.xml`,
`C:\tmp\vfe3-phi-task5-review-fix-matrix-20260717.xml`, and
`C:\tmp\vfe3-phi-task5-rereview-focused-20260717.xml`. The exact cleanup
artifact records 214 tests, zero failures, zero errors, and one skip. The
expanded artifact records 500 tests, one accepted baseline failure, zero
errors, and one skip.

## Resume sequence

1. Read `.superpowers/sdd/progress.md`, this handoff, and the Task 6 section of
   the approved implementation plan.
2. Preserve commits `b0bf9cb`, `843b45c`, `71e5481`, `085777f`, `43ed6d3`, and
   `045a245`, `574602f`, `6ad1cdc`, `2ae3ce2`, and `023686f`; do not re-dispatch
   Tasks 1 through 5.
3. Resume Task 6 from its pre-CUDA checkpoint using the approved strict
   geometry, stateless runtime, and serialized migration interfaces.
