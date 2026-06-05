# 2026-06-05 — OpenMP duplicate-runtime crash workaround (KMP_DUPLICATE_LIB_OK)

(Separate same-day doc to avoid a merge conflict with the belief-UMAP branch's append to
`2026-06-05-renyi-ablation-confound.md`; both are unmerged in parallel.)

## Symptom

`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized` ->
`Fatal Python error: Aborted`, hit by the user when running `train_vfe3.py` with `n_e_steps > 1`
(repro config: n_e_steps=5, batch_size=32, max_steps=500). The traceback is under the
`C:\anaconda` interpreter.

## Diagnosis (environment, not a V3 bug)

Two copies of Intel's `libiomp5md.dll` are loaded into one process: PyTorch ships its own, and
Anaconda's MKL (behind numpy/scipy/scikit-learn) ships another, loaded lazily on the first MKL/LAPACK
call. When the second initializes, the Intel runtime aborts. The `n_e_steps>1` correlation is timing/
path-dependent (the deeper E-step runs more `torch.linalg.matrix_exp` and its unrolled backward, a
LAPACK/MKL path that lazily loads MKL's OpenMP after torch's); the deterministic root cause is the
duplicate runtime in the env. Not reproducible on the box's pip `Python314` interpreter (no MKL
duplicate), so this is reasoned from the error + code, not observed locally.

## Fix (option B: the env-var workaround)

Set `KMP_DUPLICATE_LIB_OK=TRUE` BEFORE `import torch`, via `os.environ.setdefault(...)` at the very
top of each click-to-run entry point: `train_vfe3.py`, `ablation.py`, `make_figures.py`. `setdefault`
lets the user override by exporting the variable themselves. Intel labels the flag "unsafe (may
silently produce wrong results)", but that warning targets mixing DIFFERENT OpenMP implementations
(GNU + Intel); here both copies are the same Intel `libiomp5md`, so it is the standard, benign
workaround for this Anaconda+PyTorch case.

The cleaner, correctness-safe alternative (option A, env-side, not code): remove the duplicate so a
single OpenMP exists — `conda install nomkl` (numpy/scipy -> OpenBLAS, drops MKL's libiomp), or run
with the pip `Python314` interpreter that has no conflict.

## What is committed vs working-tree

`make_figures.py` carries the guard committed (mine, clean). `train_vfe3.py` and `ablation.py` also
have the guard added in the WORKING TREE (so the user's runs work immediately) but were left UNSTAGED
because they also hold the user's live config edits — committing would bundle that smoke config into
main. The guard's top ~3 lines should be kept when those files are next committed.

## Verification

Cannot reproduce the crash on this box (pip Python314, no MKL duplicate). Verified the guard executes
before torch: importing all three entry points sets `os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"`.
The user must confirm the `n_e_steps=5` run no longer aborts in their anaconda env.
