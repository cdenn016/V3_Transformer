# Phi Pullback-Group Build Restart Handoff

## Current state

The implementation worktree is
`C:\tmp\vfe3-phi-pullback-group-descent-20260717` on branch
`codex/phi-pullback-group-descent-20260717`. Tasks 1 through 7 are implemented
and independently reviewed. Preserve the separate investigation worktree
`C:\tmp\vfe3-mphi-ng-investigation-20260717`; it is not owned by this buildout.

The completed route uses the existing completed outer supervised scalar as the
phi covector, then applies stateless strict pullback geometry and a right-group
candidate. It does not claim the canonical fixed-returned-state VFE phi
M-step. The single canonical TODO remains at the completed-loss seam. Stateful
heavy-ball and coordinatewise Adam phi modes, their public controls, and their
moment slots are removed, with typed serialized migration at the historical
boundary.

The final RTX 5090 artifact is
`docs/testing/2026-07-17-phi-pullback-group-rtx5090.json`. All K10 two-GL(5)
cases at 128, 512, and 2,048 active rows pass the unchanged 20 percent upper
confidence-bound gate. The final 2,048-row interval is
`[-0.00431866, 0.0410821]`. The JSON SHA-256 is
`35F0C761EEC971CC764770C92D01BABEC7239D33369DBF3675922383F82EE1BF`.

The final focused XML at
`C:\tmp\vfe3-phi-focused-final-rerun-20260718.xml` records 1,004 tests, one
accepted baseline failure, zero errors, and two skips. The final full XML at
`C:\tmp\vfe3-full-phi-final-rerun-20260718.xml` records 3,919 tests, the same
14 failure node IDs as the pre-edit baseline, zero errors, and 17 skips. There
are zero new failure IDs. The changed phi preconditioner, optimizer, and
generation-migration matrix records 133 tests with zero failures, errors, or
skips on both CPU and the RTX 5090 CUDA interpreter.

The complete reviewed implementation and evidence are committed locally as
`84c5333`. The task worktree is clean.

## Remaining lifecycle work

A fresh fetch found `origin/main` at `d7cb434`, two documentation-only commits
beyond this task's base. GitHub reports `cdenn016/V3_Transformer` as public, so
publication remains blocked by the unverified-public-destination policy. No
push or merge was attempted. If publication is later authorized, integrate
`origin/main`, rerun any affected verification, push the task branch, merge and
push `main`, and remove only this task worktree and local task branch.

The live `main` checkout is already at `d7cb434` but contains user-owned edits
to `scaling.py` and `train_vfe3.py` plus seven user-owned deletions under
`vfe3_scaling_results/grow_K_GL10`. Do not alter those paths. The separate
m-phi investigation worktree is clean and must remain intact.
