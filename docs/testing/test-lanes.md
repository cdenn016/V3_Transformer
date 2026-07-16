# Test execution lanes

The ordinary CPU entry point is the click-to-run driver. It accepts no command-line arguments, launches the fast and slow lanes as separate subprocesses, stops after the first nonzero result, and reports counts only from each lane's temporary JUnit XML file.

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

CUDA is a dedicated serial lane. The default shell interpreter can have a CPU-only Torch build even when another environment contains CUDA Torch, so verify the selected interpreter before invoking pytest. On the development workstation the CUDA interpreter is `C:\anaconda\python.exe`; replace that path when the environment moves. The environment variable activates tests whose device is selected during module import, and the marker selects the explicit CUDA resource cohort.

```powershell
$cudaPython = "C:\anaconda\python.exe"
& $cudaPython -c "import sys, torch; ok = torch.cuda.is_available(); name = torch.cuda.get_device_name(0) if ok else 'no CUDA device'; print(torch.__version__, ok, name); sys.exit(0 if ok and 'RTX 5090' in name else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "The selected interpreter does not expose the intended RTX 5090 CUDA device."
}
Invoke-VFE3TestEnv @{ VFE3_TEST_DEVICE = "cuda" } {
    & $cudaPython -m pytest -m cuda --junitxml=C:\tmp\vfe3-cuda.xml --durations=100
}
[xml]$cudaResult = Get-Content -Raw -LiteralPath C:\tmp\vfe3-cuda.xml
$cudaSuite = $cudaResult.testsuites.testsuite
if ([int]$cudaSuite.tests -ne 6 -or [int]$cudaSuite.failures -ne 0 `
        -or [int]$cudaSuite.errors -ne 0 -or [int]$cudaSuite.skipped -ne 0) {
    throw "CUDA lane did not execute all six tests successfully."
}
```

The external identity probe is also a one-worker lane. Both files must exist and must be the independently generated branch-base and feature bundles described by `tests/hierarchy_identity_probe.py`.

```powershell
Invoke-VFE3TestEnv @{
    VFE3_BASELINE_BUNDLE = "C:\path\to\baseline.pt"
    VFE3_FEATURE_BUNDLE  = "C:\path\to\feature.pt"
} {
    python -m pytest -m external --junitxml=C:\tmp\vfe3-external.xml
}
```

Branch coverage is measured over `vfe3`. The checked-in configuration enables branch measurement and parallel data-file support. This command covers the complete available CPU union while excluding unavailable CUDA and external prerequisites.

```powershell
Invoke-VFE3TestEnv @{
    VFE3_TEST_DEVICE    = "cpu"
    CUDA_VISIBLE_DEVICES = "-1"
} {
    python -m pytest --runslow -m "not cuda and not external" --cov=vfe3 --cov-branch --cov-report=term-missing --cov-report=xml:C:\tmp\vfe3-coverage.xml --junitxml=C:\tmp\vfe3-coverage-junit.xml
}
```

A skipped prerequisite is not executed coverage. Report CUDA and external results from their own JUnit files, including exact skipped counts and reasons when their prerequisites are absent. The semantic suite is the union of fast CPU, the three-test slow CPU lane, CUDA, and external lanes; speed comparisons must use identical marker expressions and fixed worker counts.
