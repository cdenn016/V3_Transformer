# Phi Projection Hot-Path Optimization Design

**Date:** 2026-07-15

**Status:** Proposed; approved in conversation and awaiting review of this written specification.

## Purpose

This change preserves the exact hard bound selected by `phi_mstep_max_matrix_norm` while removing the dense matrix-embedding work and repeated host synchronization that make the current implementation unusable in training. The default `None` path remains byte-identical and performs no projection work.

The optimization applies to every built-in or custom gauge group whose generator builder can certify a Frobenius-orthogonal basis. Groups without that certificate retain an exact dense fallback. The production `block_glk` path receives the fast route without hard-coding behavior at the training call site.

## Confirmed performance defect

Two adjacent WikiText-103 K20 runs have identical saved configurations except for the projection bound and timestamp. With `phi_mstep_max_matrix_norm=4`, the run recorded `train_step_ms_mean=340.7574`; with the bound disabled, it recorded `83.5599`. The projected route is therefore 4.08 times slower at K20.

At step 100, only 6 of 50,385 eligible rows were projected, a fraction of `0.0001191`. The slowdown is consequently independent of useful projection work.

The current projector loops over every row of every trainable phi table after each accepted optimizer step. For each 64-row chunk it evaluates

$$
X_r = \sum_a \phi_{ra}G_a
$$

through the dense contraction `einsum("ra,aij->rij")`, computes the matrix Frobenius norm, synchronizes projected counts and extrema to the host, and writes a scale for every row. At K20 with 50,257 vocabulary rows, 200 generators, and 20 by 20 represented matrices, the token table alone requests approximately 4.02 billion scalar contraction terms per optimizer step.

At K240 with eight `block_glk` heads, there are 7,200 generators and 240 by 240 represented matrices. The corresponding dense token-table contraction requests approximately 20.8 trillion scalar terms per optimizer step. This is a hot-path implementation defect, not an unavoidable cost of projected optimization.

The current 64-row loop also converts device tensors to Python values three times per chunk. At 50,257 vocabulary rows, this creates roughly 786 chunks and up to 2,358 device-to-host synchronization points per table per optimizer step. Statistics are collected even when `metrics_out` is absent and no training row will persist them.

## Baseline repository state

The clean `origin/main` baseline suite recorded 2,897 tests, 1 failure, 0 errors, and 31 skipped tests in 124.341 seconds. The sole failure is `tests/test_runnable_tail_buildout.py::test_runnable_cluster_sweeps_build[sigma_max]`: the test requires every selected sweep to produce at least two cells, while the intentionally preserved live `sigma_max` sweep contains the single value `[15]`. This configuration/test mismatch predates the optimization branch and is outside this design's scope. The implementation must not change the user's sweep values or conceal the baseline failure.

## Design goals

The optimized route must preserve the following contracts.

1. Every trainable phi-table row satisfies the configured embedded matrix Frobenius bound after each accepted optimizer step.
2. Skipped or overflowed optimizer steps do not project.
3. AdamW and natural-gradient optimizer moments remain unchanged by projection.
4. Token, positional, independent model-frame token, and independent model-frame positional tables remain covered.
5. `phi_mstep_max_matrix_norm=None` remains a true no-op.
6. The fast route is enabled only by explicit generator-basis metadata that certifies the required algebraic identity.
7. Uncertified bases remain mathematically exact through a dense fallback and are visibly labeled as the slow route.
8. Persisted metrics distinguish projection work from diagnostic synchronization and retain compatibility with existing figures.

The change does not alter the configured bound, add a new approximation, change the matrix exponential, change BCH composition, reset optimizer moments, or introduce a bounded forward reparameterization.

## Considered approaches

### Certified coordinate-norm projection

This is the selected design. For a Frobenius-orthogonal generator basis,

$$
\langle G_a,G_b\rangle_F = 0 \quad (a\ne b),
$$

so

$$
\left\|\sum_a \phi_aG_a\right\|_F^2
=\sum_a \phi_a^2\left\|G_a\right\|_F^2.
$$

The embedded matrix norm is therefore computed directly from the algebra coordinates and the diagonal of the generator Gram matrix. For the untied `block_glk` basis, the generators are Frobenius-orthonormal and the calculation reduces to the ordinary rowwise Euclidean norm of phi.

This lowers the production projection cost from `O(rows * n_gen * K^2)` to `O(rows * n_gen)`. It preserves the exact existing radial projection.

### Current-batch active-row projection

This approach is rejected as the primary implementation. PyTorch AdamW maintains dense first- and second-moment buffers. A row that was active on an earlier batch can continue moving under decaying momentum even when its current gradient is zero. The custom natural-gradient optimizer likewise has persistent state under supported update rules. Projecting only the tokens present in the current batch would therefore fail to certify a global hard bound.

Tracking every row with nonzero optimizer state does not solve the scaling problem cleanly because the tracked set grows toward the full vocabulary over training. A strict full-table coordinate scan is simpler, exact, and the same asymptotic order as the optimizer's existing dense parameter and moment updates.

### Bounded forward reparameterization

A radial sigmoid or hyperbolic-tangent map could bound the effective coordinate used by the forward pass. This is rejected because it changes the optimization geometry, checkpoint semantics, gradients near the boundary, and the mathematical object being trained. It would be a separate ablation, not a repair of projected Adam.

## Gauge-group capability

`GaugeGroup` receives an explicit optional capability declaring that its generator Frobenius Gram matrix is diagonal. The capability is supplied by a group builder whose construction proves the property; runtime code does not infer orthogonality from a group name.

The group exposes a cached diagonal-weight method. The weights are

$$
w_a = \left\|G_a\right\|_F^2.
$$

The cache follows the existing `gram_pinv` cache discipline and refreshes if generator device, dtype, object identity, or version changes. Builders for `glk`, ordinary `block_glk`, `tied_block_glk`, and other provably orthogonal built-ins opt in individually. A `block_glk` builder that modifies or closes a basis may advertise the capability only when the resulting construction remains proven orthogonal. Unproven paths fail closed to the exact fallback.

Unit tests compare every advertised diagonal against the full generator Gram matrix on small dimensions. A false capability declaration is a test failure.

## Norm kernel

A single geometry helper computes squared embedded Frobenius norms for phi rows.

For a certified basis it evaluates the diagonal quadratic form directly. The unit-weight case uses `torch.linalg.vector_norm` without allocating a weighted copy. Constant-weight and nonuniform-diagonal cases use their exact simplified forms. The helper returns tensors on the input device and does not convert results to Python values.

For an uncertified basis, the helper retains the dense embedded-matrix calculation in bounded chunks. This is the correctness oracle and compatibility fallback. It is not presented as a high-performance route. A one-time warning identifies the group and explains that enabling a hard per-step projection on a nonorthogonal basis can be expensive.

The transport-clamp warning and projection code reuse the same norm helper so the two paths cannot drift to different definitions of embedded matrix norm. The existing dense implementation remains directly testable as an oracle.

## Projection kernel

`project_phi_parameter_rows_` retains its external meaning and radial update:

$$
\phi_r \leftarrow \phi_r\min\left(1,\frac{R}{\|X_r\|_F}\right).
$$

The optimized implementation processes the full tables in memory-budgeted chunks. Chunk sizing is based on coordinate width and a fixed temporary-memory budget rather than the current hard-coded 64 rows. The operation never allocates a vocabulary-by-K-by-K tensor.

Chunk counts, projected-row counts, maximum pre-projection norm, and minimum scale accumulate as device tensors. Silent training steps request no Python statistics and perform no diagnostic device-to-host synchronization. Logged steps reduce each persisted statistic once after all tables are processed.

The projector may continue to multiply every checked row by a scale in `[0,1]`. Avoiding a full-table write is secondary to eliminating dense embedding and host synchronization because AdamW already performs dense table and moment updates. A selective-write variant is admitted only if a benchmark demonstrates a material additional gain without adding a hidden `nonzero` synchronization or changing exact outputs.

## Training integration

The existing post-optimizer placement remains unchanged. Projection runs only when `did_step` is true and the bound is configured. `train_step` passes `collect_stats=metrics_out is not None`, so ordinary silent steps remain entirely on device.

No current-batch token list is required for correctness. Gradient accumulation, learned positional tables, independent model frames, plain AdamW, natural-gradient AdamW, and enabled GradScaler all use the same post-step projection contract.

The scheduler order remains unchanged. Optimizer moments are not rescaled, cleared, or recomputed.

## Metrics and reporting

The existing fields remain available:

- `phi_chart_projected_rows`.
- `phi_chart_total_rows`.
- `phi_chart_projected_fraction`.
- `phi_chart_preproject_max`.
- `phi_chart_projection_scale_min`.

Their mathematical meanings remain unchanged because the complete eligible table remains checked. New fields record:

- `phi_chart_norm_route`, with values such as `diagonal_gram` or `dense_fallback` in configuration or summary artifacts rather than numeric CSV columns.
- `phi_chart_projection_ms`, recorded only on logging steps using CUDA events on CUDA and a monotonic CPU timer on CPU.
- `phi_chart_projection_stats_collected`, distinguishing timed diagnostic steps from silent steps.

The existing geometry figure continues to plot projected fraction and minimum scale. Summary artifacts identify the norm route so results from a slow fallback cannot be mistaken for production-path performance.

Timing instrumentation must not add synchronization to silent steps. CUDA timing events are created and synchronized only when metrics are already being collected.

## Correctness tests

Tests use the existing dense embedding as the oracle and cover the following behavior.

1. The diagonal-Gram norm equals dense embedded Frobenius norm for every group that advertises the capability, across float32 and float64 small-dimensional cases.
2. The optimized projection produces the same phi coordinates as the dense oracle within dtype tolerance.
3. Every projected token, positional, model-token, and model-positional row lies within the configured radius.
4. Rows below the radius are unchanged exactly where the selected tensor operations permit exact identity scaling.
5. An uncertified nonorthogonal custom basis dispatches to the dense fallback and remains exact.
6. A falsely certified basis is rejected by builder validation or a focused invariant test.
7. `collect_stats=False` performs no `.item()`, `float(tensor)`, or equivalent host extraction inside the chunk loop.
8. `collect_stats=True` returns the existing metrics with values matching the dense oracle.
9. Skipped optimizer and GradScaler-overflow steps do not project.
10. The `None` path does not call the norm or projection kernel.
11. Plain AdamW, natural-gradient paths, gradient accumulation, and learned positional frames preserve the configured bound.
12. Checkpoint save and resume preserve configuration provenance and optimizer state while continuing to enforce the bound after accepted steps.

## Performance verification

A dedicated CUDA benchmark compares the same model, batch, seed, and warmed optimizer state with the bound disabled and enabled. Performance assertions do not run as ordinary pytest pass/fail thresholds on shared CPU environments; the benchmark emits machine-readable JSON containing configuration, device, warmup count, measured step count, median, p95, and tokens per second.

The benchmark covers K20 and K240 `block_glk` on the RTX 5090. It uses enough warmup steps to exclude first-use compilation, allocator, and CUDA-context effects. It alternates bounded and unbounded measurements or rebuilds identical seeded models so thermal drift and optimizer-state drift do not bias one arm.

Acceptance requires:

1. The K20 projected median step overhead is at most 10 percent, with 5 percent as the engineering target.
2. K240 projection never constructs a vocabulary-by-K-by-K tensor and completes without an out-of-memory event.
3. The K240 projected route is `diagonal_gram`, not `dense_fallback`.
4. Projection statistics collected at log cadence do not change silent-step throughput materially.
5. The benchmark reproduces the existing slow implementation as a regression baseline before replacement or records the already observed 340.7574 versus 83.5599 millisecond evidence in the benchmark report.

## Expected files

The implementation is expected to modify `vfe3/geometry/groups.py`, `vfe3/gauge_optim.py`, `vfe3/train.py`, `vfe3/run_artifacts.py`, and focused phi-numerics tests. A dedicated benchmark module or script will be added under the repository's existing performance-tooling convention after that convention is confirmed during implementation planning. The dated `docs/2026-07-15-edits.md` record will be extended rather than duplicated.

No training configuration value, ablation sweep value, or `SWEEP_ORDER` entry will change.

## Theoretical interpretation

This optimization changes only how the embedded chart radius is computed. It does not alter the local nature of finite BCH composition, enlarge the exponential image in GL, or reinterpret flat-path numerical closure as curvature. The hard bound remains an experimental chart-control intervention, and its hit rate remains part of the result rather than a hidden stabilizer.
