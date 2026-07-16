# Test execution lanes

The authoritative compatibility check is the serial suite. Pytest already receives `-q`, strict configuration validation, and strict marker validation from `pyproject.toml`; do not add another `-q`, because `-qq` suppresses the terminal pass-count summary. Use JUnit XML whenever exact counts are reported.

```powershell
python -m pytest --junitxml=C:\tmp\vfe3-default.xml --durations=100
python -m pytest --runslow --junitxml=C:\tmp\vfe3-full-cpu-serial.xml --durations=100
```

The parallel CPU fast lane excludes every prerequisite-specific or heavyweight integration case. Cap numerical-library threads so each pytest worker does not create its own competing thread pool. Use a fixed worker count selected from recorded two-worker and four-worker trials; never use `-n auto` as a repository default. `loadscope` keeps module-scoped immutable artifact evidence on one worker.

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m pytest -n 2 --dist loadscope -m "not slow and not cuda and not external" --junitxml=C:\tmp\vfe3-fast-n2.xml --durations=100
python -m pytest -n 4 --dist loadscope -m "not slow and not cuda and not external" --junitxml=C:\tmp\vfe3-fast-n4.xml --durations=100
Remove-Item Env:OMP_NUM_THREADS
Remove-Item Env:MKL_NUM_THREADS
```

The slow CPU lane enables `--runslow` and uses `loadgroup`, which keeps the real UMAP cohort in one resource group while other slow nodes remain schedulable. It is the complement of the fast CPU lane for ordinary CPU verification.

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m pytest --runslow -n 4 --dist loadgroup -m "slow and not cuda and not external" --junitxml=C:\tmp\vfe3-slow-cpu.xml --durations=100
Remove-Item Env:OMP_NUM_THREADS
Remove-Item Env:MKL_NUM_THREADS
```

CUDA is a dedicated one-worker lane. Run it only when the selected Python interpreter has a CUDA-enabled Torch build and the RTX 5090 is available. The environment variable activates tests whose device is selected during module import; the marker selects the explicit CUDA resource cohort.

```powershell
$env:VFE3_TEST_DEVICE = "cuda"
python -m pytest -n 1 -m cuda --junitxml=C:\tmp\vfe3-cuda.xml --durations=100
Remove-Item Env:VFE3_TEST_DEVICE
```

The external identity probe is also a one-worker lane. Both files must exist and must be the independently generated branch-base and feature bundles described by `tests/hierarchy_identity_probe.py`.

```powershell
$env:VFE3_BASELINE_BUNDLE = "C:\path\to\baseline.pt"
$env:VFE3_FEATURE_BUNDLE = "C:\path\to\feature.pt"
python -m pytest -n 1 -m external --junitxml=C:\tmp\vfe3-external.xml
Remove-Item Env:VFE3_BASELINE_BUNDLE
Remove-Item Env:VFE3_FEATURE_BUNDLE
```

Branch coverage is measured over `vfe3`. The checked-in configuration enables branch measurement and parallel data-file support. This command covers the complete available CPU union while excluding unavailable CUDA and external prerequisites.

```powershell
python -m pytest --runslow -m "not cuda and not external" --cov=vfe3 --cov-branch --cov-report=term-missing --cov-report=xml:C:\tmp\vfe3-coverage.xml --junitxml=C:\tmp\vfe3-coverage-junit.xml
```

A skipped prerequisite is not executed coverage. Report CUDA and external results from their own JUnit files, including exact skipped counts and reasons when their prerequisites are absent. The semantic suite is the union of fast CPU, slow CPU/UMAP, CUDA, and external lanes; speed comparisons must use identical marker expressions and fixed worker counts.
