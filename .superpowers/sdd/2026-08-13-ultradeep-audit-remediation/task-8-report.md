# Task 8 consolidated CPU closure

## Status

Task 8 is **bounded INCONCLUSIVE for a repository-wide green claim** and
**EVIDENCE_VERIFIED for closure of Task 1-7 remediation regressions on the exercised CPU
seams**. The final targeted seam passes completely. The final CPU-fast and CPU-slow lanes retain
only failures reproduced mechanically at the pinned audit base `714e3c5`; no current-only failure
remains. A01/A02 drift rejection remains waived as directed.

All behavioral verification was run at code/test revision
`50c1c91ce4ec6db17534d24bac8b4323b7c312aa` using `C:/Python314/python.exe` with
`CUDA_VISIBLE_DEVICES=-1`, `VFE3_TEST_DEVICE=cpu`, and the CPU thread variables set to `1`.
This report makes no CUDA claim.

Task 8 repair commits are:

- `7947965fa95c1839f5b9941771a06b6b2076eae0` — close two remediation-introduced Ruff defects.
- `c9e52e29aa015475e5d9c8328bd26309bc8dcbe8` — reconcile targeted production and fixture contracts.
- `87cda15738455ae2d0e813454cad3e48f1213996` — reconcile CPU closure compatibility and stale fixtures.
- `50c1c91ce4ec6db17534d24bac8b4323b7c312aa` — restore the exact canonical decoder registry record after a mutating test.

## Static closure

The final revision changes 70 Python files relative to `714e3c5`. Ruff was run over all 70:

```powershell
$files = @(git diff --name-only 714e3c5..HEAD -- '*.py')
& 'C:/Users/chris and christine/AppData/Local/Programs/Python/Python313/Scripts/ruff.exe' check --output-format json --output-file .verification/remediation-2026-08-13/task-08-ruff-final-head.json -- $files
```

Ruff exits `1` because the selected files contain inherited diagnostics: 204 at the verified
revision versus 206 when the same applicable file set is checked in the `714e3c5` worktree.
Relative-path/code/message multiset comparison has zero increased groups; one existing `E702`
group in `tests/test_viz.py` decreases from 45 to 43. Therefore Task 8 introduced no Ruff
diagnostic, but this is not a repository lint-clean claim.

```powershell
git diff --check
C:/Python314/python.exe -m compileall -q $files
```

Both commands exit `0`.

## Exact targeted audit seam

```powershell
C:/Python314/python.exe -m pytest -p no:cacheprovider `
  --basetemp=C:/tmp/task8-targeted-final `
  --junitxml=.verification/remediation-2026-08-13/targeted-final.xml `
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

Fresh JUnit: **427 tests, 427 passed, 0 failures, 0 errors, 0 skipped, 26.032 s**.
Console warning summary: 120 warnings. XML SHA-256:
`B3B63B959CC297097951A93C0E47D3ADBFB717C3BB611C6FAD78C629C8122F17`.

The original audit-revision JUnit contains 421 tests. Mechanical node-ID comparison finds six
additions and no removals. All six are the Cartesian cases of
`test_registered_block_mlp_accounting_and_cost_fields_preserve_legacy_parity` for
`{passthrough,delta_full}` x `{coordinate,gauge_gate,canonical_frame}`. The current result is
therefore reported as 427, never as 421.

## CPU-fast policy

The repository policy resolved from `pytest.ini` and `scripts/test_cpu_fast.ps1` to:

```powershell
C:/Python314/python.exe -m pytest -n 12 --dist loadscope `
  -m 'not slow and not cuda and not external' -p no:cacheprovider `
  --basetemp=C:/tmp/task8-cpu-fast-final `
  --junitxml=.verification/remediation-2026-08-13/cpu-fast-final.xml
```

Fresh JUnit: **5,467 tests, 5,421 passed outcomes, 9 failures, 0 errors, 37 skipped,
65.133 s**. The 5,421 passed outcomes comprise 5,409 ordinary passes and 12 passing subtests.
Console warning summary: 945 warnings. XML SHA-256:
`32CDDB9A50117D105E6D24A1CCE0FE8F1E44E5DF9317AA3EE9C7F501CC496B21`.

For classification, the identical policy was executed at `714e3c5`: 5,215 tests, 4,936 passed
outcomes, 242 failures, 0 errors, 37 skipped, 65.457 s; SHA-256
`7BFA4E6A4E392FC586002AEB0DDE4B903BCE00C609A293F92FBFCD359C332FFD`.
All nine final current failure node IDs occur in that base artifact; the current-only set is empty.
The remaining executable clusters are:

- diagnostics/schema assumptions: the compact omega diagnostic test tries to tensorize a current
  `None` diagnostic;
- launcher/config expectation drift: four tests retain older ablation prerequisite/default
  expectations;
- source-identity/incomplete-sweep behavior: one older skip-all-analysis expectation;
- exact-bit forward assertions: two straight-through/detach byte-identity expectations;
- trust-region default expectation: one older default assertion.

The first current run exposed 48 failures: 12 also failed at the base and 36 were current-only.
Commit `87cda15` closed the 36 remediation compatibility/fixture regressions. A subsequent full run
had 16 failures; seven current-only projected-fidelity failures passed 18/18 in isolation.
Executable ordering with `tests/test_round3_registry_guards.py` reproduced them: its cleanup rebuilt
`full_chunked` with equivalent fields but a new registration object, invalidating the exact
import-time registry identity required by projected decoding. Commit `50c1c91` restores the saved
record itself. The contaminating module plus fidelity module then passed 52/52, and the final full
lane reduced to the nine base-reproduced failures above.

After the complete pytest summary and complete JUnit write, the Python 3.14 parent process emits a
Windows access violation while importing `pyarrow` through `pandas`/`sklearn` during `execnet`
teardown. This native teardown fault is not represented as a JUnit test error and occurs after all
testcase outcomes are serialized. It remains an environment/toolchain concern; the report does not
reinterpret it as a passing process exit.

## CPU-slow policy

```powershell
C:/Python314/python.exe -m pytest --runslow -n 3 --dist loadgroup `
  -m 'slow and not cuda and not external' -p no:cacheprovider `
  --basetemp=C:/tmp/task8-cpu-slow-final `
  --junitxml=.verification/remediation-2026-08-13/cpu-slow-final.xml
```

Fresh JUnit: **3 tests, 2 passed, 1 failure, 0 errors, 0 skipped, 27.691 s**.
Console warning summary: 1 warning. XML SHA-256:
`6D7BDDBCCF84D5A58E78620C7104189FD0157EF852201FDE3417D3E15C7695B7`.

The sole failure is
`test_finalize_writes_tier3_research_and_provenance`: its strict expected dictionary lists the
older four deterministic-state fields, while production also records four newer precision/TF32
fields. The exact base lane reproduces the same failure: 3 tests, 2 passed, 1 failure, 0 errors,
0 skipped, 28.056 s; SHA-256
`406CBCAC1B38BA04FC89609E11D78014794F8D83BE67F3CA7C38A24E1DF7E887`.
It is therefore pre-existing and unrelated to the approved remediation, and Task 8 does not change
it.

## Closure boundary

The fresh closure ledger is
`.verification/remediation-2026-08-13/task-08-final-closure-ledger.json`. Its claims separate the
verified absence of current-only remediation failures from the inconclusive repository-wide green
claim. The ignored JUnit artifacts and ledger are evidence only. The tracked checkout is clean after
the report commit.
