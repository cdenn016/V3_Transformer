# Test execution lanes

The ordinary CPU entry point is the click-to-run driver. It accepts no command-line arguments, captures the logical CPU topology once, validates both configured worker counts before constructing the child environment or starting either subprocess, then launches the 12-worker fast lane and 3-worker slow lane separately. It stops after the first nonzero result and reports counts only from each lane's temporary JUnit XML file. A successful pytest exit with an empty JUnit suite is rejected. Every `testsuite` must explicitly provide its `tests`, `failures`, `errors`, and `skipped` counters; missing attributes are invalid rather than silently interpreted as zero. The shared JUnit runner removes its temporary report even when validation raises. Both subprocesses receive every thread cap documented below.

```powershell
python run_cpu_tests.py
```

Pytest already receives `-q`, strict configuration validation, and strict marker validation from `pyproject.toml`; do not add another `-q`, because `-qq` suppresses the terminal pass-count summary. Use JUnit XML whenever exact counts are reported. For manual lane execution, define this helper once in the current PowerShell session; every lane below then restores each pre-existing environment value even when pytest fails.

```powershell
function Invoke-VFE3TestEnv {
    param([hashtable]$Variables, [scriptblock]$Command)
    $previous = @{}
    foreach ($name in $Variables.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
        [Environment]::SetEnvironmentVariable($name, [string]$Variables[$name], "Process")
    }
    $exitCode = $null
    try {
        & $Command
        $exitCode = $LASTEXITCODE
    } finally {
        foreach ($name in $Variables.Keys) {
            [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
        }
    }
    if ($null -ne $exitCode -and $exitCode -ne 0) {
        throw "Test command exited with code $exitCode."
    }
}
```

Both CPU lanes pin `VFE3_TEST_DEVICE=cpu`, hide CUDA from libraries that probe it directly, and exclude the dedicated CUDA and external cohorts. Every supported native numerical-library thread control is set to one before pytest imports numerical libraries. Giving each xdist worker one native numerical thread prevents nested OpenMP and BLAS pools from oversubscribing the host's 24 logical processors.

The fast lane uses all 12 physical cores with an explicit worker count. Never use `-n auto`. `loadscope` keeps module-scoped immutable artifact evidence on one worker. Its manual equivalent is:

```powershell
$cpuParallelEnv = @{
    VFE3_TEST_DEVICE      = "cpu"
    CUDA_VISIBLE_DEVICES = "-1"
    OMP_NUM_THREADS       = "1"
    MKL_NUM_THREADS       = "1"
    OPENBLAS_NUM_THREADS  = "1"
    NUMEXPR_NUM_THREADS   = "1"
    NUMBA_NUM_THREADS     = "1"
    BLIS_NUM_THREADS      = "1"
    VECLIB_MAXIMUM_THREADS = "1"
}
Invoke-VFE3TestEnv $cpuParallelEnv {
    python -m pytest -n 12 --dist loadscope -m "not slow and not cuda and not external" --junitxml=C:\tmp\vfe3-fast-n12.xml --durations=100
}
```

The remaining slow CPU lane contains only three tests. It enables `--runslow`, uses three explicit workers with `loadgroup`, and is the complement of the fast CPU lane for ordinary CPU verification. Its manual equivalent is:

```powershell
Invoke-VFE3TestEnv $cpuParallelEnv {
    python -m pytest --runslow -n 3 --dist loadgroup -m "slow and not cuda and not external" --junitxml=C:\tmp\vfe3-slow-n3.xml --durations=100
}
```

All eleven native UMAP integration tests were intentionally removed from the suite. Production UMAP support and the `umap-learn` visualization dependency remain, as do mocked worker-protocol tests and pure report-planning tests. The retained slow lane therefore contains exactly the three nodes selected by `tests/pytest_policy.py`; it is not a native UMAP lane.

CUDA is a dedicated serial lane selected canonically with `-m cuda`. `tests/pytest_policy.py` currently defines six CUDA-only hardware tests and sixteen ordinary numerical contracts that join the CUDA marker only when the requested device type is CUDA. Collection policy, rather than a duplicated literal node list or a brittle hardcoded pass count, defines the executable matrix. The expected count is computed as `len(CUDA_TESTS | CUDA_MIRROR_TESTS)`, and success requires both a zero pytest exit and a positive, internally consistent JUnit result in which every expected node passed with no failures, errors, or skips. The shared parser first requires all four JUnit count attributes, so a truncated report cannot pass by omission; its temporary XML cleanup also runs when parsing fails.

The default shell interpreter can have a CPU-only Torch build even when another environment contains CUDA Torch. On the development workstation the CUDA interpreter is `C:\anaconda\python.exe`; replace that path when the environment moves. Both `VFE3_TEST_DEVICE=cuda` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` must be present before that interpreter imports Torch. After the safe Torch import, `tests/conftest.py` enters the deterministic CUDA policy during module initialization before plugin-driven collection. One idempotent lifecycle owner is shared by the session fixture and `pytest_unconfigure`, so normal session teardown and collection-abort teardown both restore state exactly once. It removes `CUBLAS_WORKSPACE_CONFIG` when it was initially absent, reinstates its exact preexisting value when present, and leaves CPU policy environment state untouched.

```powershell
$cudaPython = "C:\anaconda\python.exe"
$cudaEnv = @{
    VFE3_TEST_DEVICE       = "cuda"
    CUBLAS_WORKSPACE_CONFIG = ":4096:8"
}
Invoke-VFE3TestEnv $cudaEnv {
    & $cudaPython -c "import sys, torch; ok = torch.cuda.is_available(); name = torch.cuda.get_device_name(0) if ok else 'no CUDA device'; print(torch.__version__, ok, name); sys.exit(0 if ok and 'RTX 5090' in name else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "The selected interpreter does not expose the intended RTX 5090 CUDA device."
    }
    $expectedCudaCount = [int](& $cudaPython -c "from tests.pytest_policy import CUDA_MIRROR_TESTS, CUDA_TESTS; print(len(CUDA_TESTS | CUDA_MIRROR_TESTS))")
    if ($LASTEXITCODE -ne 0 -or $expectedCudaCount -le 0) {
        throw "The canonical CUDA policy did not provide a positive expected count."
    }
    & $cudaPython -m pytest -m cuda --junitxml=C:\tmp\vfe3-cuda.xml --durations=100
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA pytest exited with code $LASTEXITCODE."
    }
    [xml]$cudaResult = Get-Content -Raw -LiteralPath C:\tmp\vfe3-cuda.xml
    $cudaSuite = $cudaResult.testsuites.testsuite
    $requiredCudaCounts = @("tests", "failures", "errors", "skipped")
    $missingCudaCounts = @($requiredCudaCounts | Where-Object { -not $cudaSuite.HasAttribute($_) })
    if ($missingCudaCounts.Count -ne 0) {
        throw "The canonical CUDA JUnit report omitted required count attributes."
    }
    if ([int]$cudaSuite.tests -ne $expectedCudaCount -or [int]$cudaSuite.failures -ne 0 `
            -or [int]$cudaSuite.errors -ne 0 -or [int]$cudaSuite.skipped -ne 0) {
        throw "The canonical CUDA lane did not pass its exact expected count."
    }
}
```

CPU goldens, host-side contracts, and the complete ordinary CPU lanes remain mandatory. CUDA adds a bounded device matrix for the six hardware-only tests and the dynamically mirrored numerical contracts; it does not replace CPU verification.

The external identity probe is also a one-worker lane. Both files must exist and must be the independently generated branch-base and feature bundles described by `tests/hierarchy_identity_probe.py`.

```powershell
Invoke-VFE3TestEnv @{
    VFE3_BASELINE_BUNDLE = "C:\path\to\baseline.pt"
    VFE3_FEATURE_BUNDLE  = "C:\path\to\feature.pt"
} {
    python -m pytest -m external --junitxml=C:\tmp\vfe3-external.xml
}
```

Branch coverage is measured over `vfe3`. The checked-in configuration enables branch measurement and parallel data-file support. The command below intentionally measures the complete retained CPU union in one simpler 12-worker `loadscope` invocation while excluding unavailable CUDA and external prerequisites. This union-measurement command deviates from the ordinary execution design, which separately uses 12 workers for the fast lane and 3 workers for the three-node slow lane. It does not replace the 12/3 click-to-run driver or provide a like-for-like timing comparison with that sharded runner.

```powershell
Invoke-VFE3TestEnv $cpuParallelEnv {
    python -m pytest --runslow -n 12 --dist loadscope -m "not cuda and not external" --cov=vfe3 --cov-branch --cov-report=xml:C:\tmp\vfe3-coverage.xml --junitxml=C:\tmp\vfe3-coverage-junit.xml --durations=25
}
```

A skipped prerequisite is not executed coverage. Report CUDA and external results from their own JUnit files, including exact skipped counts and reasons when their prerequisites are absent. The semantic suite is the union of fast CPU, the three-test slow CPU lane, CUDA, and external lanes; speed comparisons must use identical marker expressions and fixed worker counts.
