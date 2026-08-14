# Task 8 consolidated CPU closure review

## Status

The repository-wide all-green claim is **REFUTED**: the final CPU-fast policy lane has six
failures and the final CPU-slow policy lane has one failure. The narrower comparative claim is
**EVIDENCE_VERIFIED** independently for each lane: every remaining current failure has the same
node ID and first JUnit failure root in the direct `714e3c5` base artifact, and neither lane has a
new or changed root. The exact targeted audit seam is all green at the current revision.

All final behavioral evidence was produced from clean detached worktree
`C:/tmp/V3_Transformer_task8_final_tests_20260814` at code/test revision
`f8ab5f809fb7c31521182a176b2b71b9a4adb212` (tree
`f38f8c615e32993034b52432bc5c3c3967c2988a`). The interpreter was
`C:/Python314/python.exe`. The full CPU environment from `run_cpu_tests.py` was applied:
`CUDA_VISIBLE_DEVICES=-1`, `VFE3_TEST_DEVICE=cpu`, and `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `NUMBA_NUM_THREADS`,
`BLIS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` were all `1`. This report makes no CUDA claim.
A01/A02 drift rejection remains waived as directed.

Task 8 commits through the tested code revision are:

- `7947965fa95c1839f5b9941771a06b6b2076eae0` — close remediation Ruff regressions.
- `c9e52e29aa015475e5d9c8328bd26309bc8dcbe8` — reconcile targeted contracts.
- `87cda15738455ae2d0e813454cad3e48f1213996` — reconcile CPU compatibility fixtures.
- `50c1c91ce4ec6db17534d24bac8b4323b7c312aa` — preserve decoder registry identity.
- `c20f549ea5c53768c8ab41bfb73be1de3d6f34fe` — close the three review-rejected roots.
- `7c2bd70f91350cd15aa4cdfdda64cb6cdb110205` — normalize mixed line endings and range whitespace.
- `f8ab5f809fb7c31521182a176b2b71b9a4adb212` — retry transient UMAP workdir cleanup within the
  explicit cleanup budget.

## Review-rejected roots and TDD closure

The review-current red artifact records the three required failures. The test-first artifact then
records the Renyi registry failure while the new omega/geometry assertions pass. The final focused
artifact is 10/10 green.

- Omega diagnostics now permit `None` only for
  `phi_mstep_max_matrix_norm` and `transport_chart_max_norm`; every other diagnostic is required
  and finite. The RoPE fixture uses the same exact optional set.
- The family-consistent Renyi prior-bank sweep registers only `0.5`, `0.8`, and `1.0`. Every
  registered arm constructs, retaining one-factor sweep semantics while respecting Task 5's
  `alpha <= 1` supported scope.
- The singular covariance has its own fail-closed certification test. Jitter-status propagation is
  exercised separately with a valid SPD production forward, so Task 4 certification is not
  weakened.
- The stale precision comment now describes the retained float64 island.

An xdist-loaded run subsequently exposed a current-only Task 7 scratch-cleanup flake. Eight
isolated repetitions passed, while a deterministic fail-once regression proved that `close()` did
not retry a transient Windows sharing failure. The red test, 4/4 affected green seam, and 11/11
complete Task 7 module are preserved. `f8ab5f8` adds a condition-wait retry bounded by the existing
`cleanup_timeout`; only a terminal failure is published as `workdir_cleanup`.

## Static closure

The final revision changes 73 Python files relative to `714e3c5`; 64 of those paths exist at the
base revision. Ruff was run over all 73 current files and the same 64 applicable files in the
direct base worktree. Current Ruff exits `1` with 224 inherited diagnostics; the base exits `1`
with 226. Relative-path/code/message multiset comparison has zero increased groups and one
decreased group: `tests/test_viz.py|E702` falls from 45 to 43.

```powershell
$files = @(git diff --name-only 714e3c5..HEAD -- '*.py')
ruff check --output-format json --output-file current-f8ab5f8-ruff.json -- $files
git diff --check 714e3c5..HEAD
C:/Python314/python.exe -m compileall -q $files
```

The range-bound diff check and compileall both exit `0`. Direct evidence is
`.verification/remediation-2026-08-14/current-f8ab5f8-ruff.json`,
`base-714e3c5-ruff-current-fileset.json`, `ruff-delta.json`, and
`static-exit-summary.json`.

## Exact targeted audit seam

```powershell
C:/Python314/python.exe -m pytest -p no:cacheprovider `
  --basetemp=C:/tmp/task8-clean-targeted-final `
  --junitxml=<evidence>/current-f8ab5f8-targeted-clean.xml `
  tests/test_block_mlp.py `
  tests/test_block_mlp_ablation_reporting.py `
  tests/test_block_mlp_launchers.py `
  tests/test_gauge_block_mlp.py `
  tests/test_gauge_block_mlp_diagnostics.py `
  tests/test_gauge_block_mlp_integration.py `
  tests/test_gauge_block_mlp_reporting.py `
  tests/test_gauge_block_mlp_training.py `
  tests/test_config.py `
  tests/test_checkpoint_resume.py `
  tests/test_numerics.py `
  tests/test_retraction.py
```

Final JUnit: **427 tests, 427 passed, 0 failures, 0 errors, 0 skipped, 28.330 s**;
console summary: 120 warnings. SHA-256:
`6D4FC44DDB4B94EBC80D7FC8AB76D8579E24890BB4F085BB8E69A537AA83F42D`.

The preserved original audit artifact has 421 nodes (400 passed, 21 failed at that audit
revision). Mechanical node comparison finds six additions and no removals. The additions are the
Cartesian cases of
`test_registered_block_mlp_accounting_and_cost_fields_preserve_legacy_parity` for
`{passthrough,delta_full}` by `{coordinate,gauge_gate,canonical_frame}`. The current result is
therefore reported as 427, never as 421.

## CPU-fast policy

```powershell
C:/Python314/python.exe -m pytest -n 12 --dist loadscope `
  -m 'not slow and not cuda and not external' -p no:cacheprovider `
  --basetemp=C:/tmp/task8-clean-cpu-fast-policy-final `
  --junitxml=<evidence>/current-f8ab5f8-cpu-fast-clean-policy.xml
```

Final JUnit: **5,469 tests, 5,426 passed outcomes, 6 failures, 0 errors, 37 skipped,
70.944 s**. The passing outcomes comprise 5,414 ordinary passes and 12 passing subtests. Console
summary: 946 warnings. The pytest process exit is `1`. XML SHA-256:
`64D92956AC26B496D4C85CDE53B0823079D05E99464DF1623D7710E2A81D94E4`.

The direct base artifact has 5,215 tests, 4,936 passed outcomes, 242 failures, 0 errors, 37
skipped, and 65.457 s. All six final current failure nodes have identical first roots in that base
artifact; `node-first-root-delta.json` records six `persistent_same_root` and zero new or changed
roots. The remaining clusters are one older omega ablation prerequisite assertion, two exact-bit
straight-through/detach assertions, one incomplete-sweep callback signature, one trust-region
default assertion, and one E-step sweep prerequisite rejection.

The raw final stdout, stderr, and exit JSON are preserved. Stderr contains a Windows fatal
access-violation traceback while the JUnit artifact is complete. The evidence does not mechanically
establish its cause or exact inter-stream ordering, so no causal attribution is made.

## CPU-slow policy

```powershell
C:/Python314/python.exe -m pytest --runslow -n 3 --dist loadgroup `
  -m 'slow and not cuda and not external' -p no:cacheprovider `
  --basetemp=C:/tmp/task8-clean-cpu-slow-policy-final `
  --junitxml=<evidence>/current-f8ab5f8-cpu-slow-clean-policy.xml
```

Final JUnit: **3 tests, 2 passed, 1 failure, 0 errors, 0 skipped, 25.999 s**; console summary:
1 warning; process exit `1`. SHA-256:
`6C0311B3C28FA52ACA1B3045F4002055C01BE9E3CDD3D06983D921C822B8C19A`.

The sole failure is
`test_finalize_writes_tier3_research_and_provenance`: the expected dictionary omits four newer
precision/TF32 provenance fields. The direct base lane has the same node and first root (3 tests,
2 passed, 1 failure, 28.056 s), so the slow comparative delta has zero new or changed roots. This
does not make the slow lane green.

## Durable evidence and revision transfer

All direct evidence is under `.verification/remediation-2026-08-14/`. Required anchors are:

- `artifact-manifest.json` — SHA-256 and byte length for every retained base/current artifact.
- `base-714e3c5-cpu-fast.xml`, `base-714e3c5-cpu-slow.xml`, and
  `base-714e3c5-ruff-current-fileset.json` — direct base evidence.
- `audit-original-targeted-421.xml` — original audit targeted evidence.
- `current-f8ab5f8-targeted-clean.xml`, `current-f8ab5f8-cpu-fast-clean-policy.xml`,
  `current-f8ab5f8-cpu-slow-clean-policy.xml`, and `current-f8ab5f8-ruff.json` — current
  equivalents.
- `node-first-root-delta.json` and `ruff-delta.json` — machine-readable deltas.
- `current-f8ab5f8-cpu-fast-clean-policy.stdout.txt`, `.stderr.txt`, and `-exit.json` — raw final
  fast process evidence.
- `code-test-revision.json` — clean worktree, revision, tree, and inactive-verification proof.

The report commit is documentation-only after the tested code revision. After that commit,
`docs-head-transfer.json` records the final documentation head, tree IDs, and the exact
`f8ab5f8..HEAD` name/status diff; it must show no production or test path. The fresh closure ledger
is `task-08-review-closure-ledger.json` and is validated before handoff.
