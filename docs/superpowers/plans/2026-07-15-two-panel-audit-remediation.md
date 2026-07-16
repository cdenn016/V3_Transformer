# Two-Panel Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair every actionable finding in both July 15 audits while preserving all current configuration choices and every reachable mathematically pure path.

**Architecture:** The work is partitioned into eight independently reviewable units. Family-specific mathematics moves behind family dispatch, integrity checks fail closed at their owning boundary, training side effects share one accepted-step gate, and performance work reuses authoritative state without changing mathematical outputs.

**Tech Stack:** Python 3, PyTorch, pytest with JUnit XML, dataclasses, NumPy/memmap, Git worktrees.

## Global Constraints

- Preserve the affine output projection. `use_prior_bank=False` is allowed, and `use_prior_bank=True` remains the opt-in pure decode path.
- Pure paths must exist and be reachable; they do not need to be default or current values.
- Do not change user-selected configuration values merely to select a pure route.
- Preserve float32 execution except for narrow, explicitly bounded numerical reductions whose result is cast back to the caller dtype.
- Keep all registry seams and add variants or family hooks behind registries rather than editing every call site.
- Add a failing regression test before each production-code repair and record the red and green commands.
- Update only `docs/2026-07-15-edits.md` for the dated post-edit record.
- Read executable paths rather than relying on comments as proof.

---

### Task 1: Objective Fidelity and Accepted-Step State Transitions

**Findings:** B1, B2, B3, B6, S2-V1, S2-M1, S2-M2, S2-M3.

**Files:**
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/free_energy.py`
- Modify: `vfe3/gradients/oracle.py`
- Modify: `vfe3/train.py`
- Test: `tests/test_2026_07_15_objective_and_step_remediation.py`

**Interfaces:**
- Produces one authoritative proposal-sensitive frame objective used by both live scoring and Metropolis acceptance.
- Produces a boolean accepted-update result that gates scheduler, projection, EMA, and Metropolis transitions.
- Preserves passive gamma-frame gradients when configured, but reports that scope accurately.

- [ ] **Step 1: Add failing objective and transition regressions.** Reproduce the joint-objective sign reversal, RoPE-sensitive gamma energy, incompatible-support prior row sums, entropy-tail derivative mismatch, scheduler advance on rejected update, inner-autocast dtype leakage, and Metropolis mutation after rejection.
- [ ] **Step 2: Run the focused tests and confirm every new case fails for the cited behavior.**
- [ ] **Step 3: Consolidate proposal scoring.** Make the Metropolis scorer call the same complete fixed-belief objective components as production, including active beta, gamma/meta, hyperprior-sensitive, RoPE, and configured two-hop terms.
- [ ] **Step 4: Correct support mixing and entropy evaluation.** Mask before normalization and use a zero-safe `torch.special.xlogy(beta, beta)` form for the categorical term.
- [ ] **Step 5: Introduce one accepted-step decision.** Use the GradScaler scale transition or explicit optimizer result to compute `did_step`; advance scheduler and optional state transitions only when true.
- [ ] **Step 6: Add the fp32 inner-differentiation island.** Disable autocast around inner objective and derivative construction while preserving the outer autocast contract.
- [ ] **Step 7: Run focused tests until green, then run all objective, training, and Metropolis test modules.**

### Task 2: Gauge, SPD, Retraction, and Pairwise Numerics

**Findings:** B4, D5, S2-G1, S2-G2, S2-G3, S2-G4, S2-N1, S2-N2.

**Files:**
- Modify: `vfe3/geometry/transport.py`
- Modify: `vfe3/geometry/retraction.py`
- Modify: `vfe3/geometry/lie_ops.py`
- Modify: `vfe3/model/positional_phi.py`
- Modify: `vfe3/gradients/pairwise_stats.py`
- Modify: `vfe3/run_artifacts.py`
- Modify: `vfe3/config.py`
- Test: `tests/test_2026_07_15_geometry_remediation.py`

**Interfaces:**
- Exact SPD feature factorization has no coordinate-fixed ridge for valid inputs; recovery returns an explicit approximation status.
- `sigma_max=None` remains the exact AIRM path; finite caps remain allowed and are labeled projected.
- Learned BCH position remains selectable; its route gains a chart/residual validity contract while exact group product remains reachable.
- Retraction dispatch uses the existing registry.

- [ ] **Step 1: Add failing geometry tests.** Cover nonorthogonal GL congruence, bounded versus unbounded AIRM metadata, clamp activation, large BCH residual, exact inverse/cocycle behavior, small diagonal KL, and near-floor spectral derivatives.
- [ ] **Step 2: Run the focused tests and record the expected failures.**
- [ ] **Step 3: Remove the fixed ridge from valid SPD feature evaluation.** Try an exact factorization first; if recovery jitter is required, return or record `exact=False` and never publish the result as gauge-covariant exactness.
- [ ] **Step 4: Make approximation contracts explicit without changing defaults.** Preserve finite `sigma_max`, exponential clamp, and BCH4 choices, but expose activation/residual status and fail closed only when the configured validity bound is exceeded.
- [ ] **Step 5: Route Lie retraction through the registry and remove transpose-as-inverse heuristics unless an exact orthogonality predicate is satisfied.**
- [ ] **Step 6: Replace the eigengap floor with dtype- and scale-aware divided differences, using the analytic repeated-eigenvalue limit.**
- [ ] **Step 7: Stabilize diagonal KL using a narrow float64 scalar reduction or a cancellation-stable series, then cast results back to the input dtype before downstream use.**
- [ ] **Step 8: Run focused tests until green and execute all geometry, transport, retraction, and pairwise-stat test modules.**

### Task 3: Family-Aware Dispersion, Fisher Geometry, and Head Mixing

**Findings:** S2-I1, S2-I2, S2-I3, S2-I4, S2-I5/B5, S2-I6, S2-T2, S2-C1.

**Files:**
- Modify: `vfe3/families/base.py`
- Modify: `vfe3/families/gaussian.py`
- Modify: `vfe3/families/laplace.py`
- Modify: `vfe3/model/head_mixer.py`
- Modify: `vfe3/geometry/norms.py`
- Modify: `vfe3/numerics.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/metrics.py`
- Modify: `vfe3/viz/extract.py`
- Modify: `vfe3/config.py`
- Test: `tests/test_2026_07_15_family_remediation.py`

**Interfaces:**
- Family hooks provide covariance diagonal, mean Fisher precision, trust-region scale, mixed dispersion, and diagnostic labels.
- Independent-head gauge purity remains reachable with the mixer disabled or with a compatible intertwiner; current mixer defaults remain unchanged.
- No forward path converts a CUDA tensor to a Python scalar to detect mixer identity.

- [ ] **Step 1: Add failing Gaussian and Laplace regressions for all family hooks, finite config validation, mixer equivariance metadata, and identity shortcut behavior.**
- [ ] **Step 2: Run the focused tests and confirm the Laplace cases fail while Gaussian compatibility cases pin existing behavior.**
- [ ] **Step 3: Add family-dispatched statistical hooks.** Implement Gaussian variance semantics and Laplace `2*b**2`, `1/b**2`, `b`, and moment-matched scale formulas in the owning family implementations.
- [ ] **Step 4: Make norm, trust-region, reliability, mixer, metrics, and extraction paths consume those hooks.**
- [ ] **Step 5: Reject nonfinite or nonpositive `kl_max` and `renyi_order` values during configuration construction.**
- [ ] **Step 6: Remove `.item()` identity detection and represent independent-head mixer compatibility explicitly in configuration/artifact metadata without changing current selections.**
- [ ] **Step 7: Run focused tests until green and execute all family, mixer, norm, metric, and extraction tests.**

### Task 4: Data Identity, Cache Bytes, Evaluation Coverage, and Bounded Memory

**Findings:** S2-D1 through S2-D7, P3, P8/S2-D6.

**Files:**
- Modify: `vfe3/data/datasets.py`
- Modify: `vfe3/run_artifacts.py`
- Modify: `vfe3/train.py`
- Modify: `train_vfe3.py`
- Modify: `ablation.py`
- Test: `tests/test_2026_07_15_data_integrity_remediation.py`

**Interfaces:**
- A stable data-source contract binds resume state to corpus, tokenizer, cap, cache metadata, and content digest.
- Binary cache mapping validates `n_tokens > 0` and `file_bytes == n_tokens * dtype.itemsize`.
- Evaluation emits every real transition exactly once and masks padded targets.
- Character normalization reports nullable BPC and separately named bits per token.

- [ ] **Step 1: Add failing tests for resume mismatch, truncated/extended caches, final partial windows, unavailable character normalization, stale memo identity, bounded token decoding, bounded unigram counting, and repeated corpus hashing.**
- [ ] **Step 2: Run the focused tests and record all expected failures.**
- [ ] **Step 3: Define and persist the exact data-source contract, then reject resume before cursor restoration when any identity field differs.**
- [ ] **Step 4: Validate binary cache bytes before memmap construction and include source identity in character-count memoization.**
- [ ] **Step 5: Add padded final evaluation windows with ignored padding targets.**
- [ ] **Step 6: Stream character counts and unigram counts in bounded chunks and reuse immutable split digests across ablation cells.**
- [ ] **Step 7: Publish `bpc=None` when character normalization is unavailable and publish bits per token under its own field.**
- [ ] **Step 8: Run focused tests until green and execute dataset, artifact, train, and ablation artifact tests.**

### Task 5: Belief Cache, Checkpoint Schema, Serialization, and Comparison Contracts

**Findings:** T1, T2, T3, T4, Q4.

**Files:**
- Modify: `vfe3/inference/belief_cache.py`
- Modify: `vfe3/inference/e_step.py`
- Modify: `vfe3/config.py`
- Modify: `vfe3/viz/embedding_comparison.py`
- Modify: `sigma_gate_measure.py`
- Test: `tests/test_2026_07_15_cache_serialization_remediation.py`

**Interfaces:**
- Cache eligibility operates on canonical E-step modes and complete positional state.
- Configuration deserialization normalizes real booleans and rejects ambiguous strings.
- Checkpoint measurement fails closed on unsupported schema drift.
- Embedding comparison binds sequence count and sequence length.

- [ ] **Step 1: Add failing tests reproducing the alias cache mismatch, missing right frame, string truthiness reversal, partition mismatch acceptance, and old-schema reinterpretation.**
- [ ] **Step 2: Run the focused tests and confirm each failure matches the audit probe.**
- [ ] **Step 3: Canonicalize E-step modes before cache eligibility and include `right_phi` in cache state or reject that route explicitly.**
- [ ] **Step 4: Normalize only recognized boolean spellings and reject other strings before dataclass construction.**
- [ ] **Step 5: Require checkpoint-schema compatibility in sigma-gate measurement and extend embedding comparison identity with partition dimensions.**
- [ ] **Step 6: Run focused tests until green and execute the full belief-cache, config, checkpoint, and embedding-comparison test groups.**

### Task 6: Driver Reliability, Reproducibility, and Atomic Publication

**Findings:** Q1 through Q9, D1 through D4.

**Files:**
- Modify: `train_vfe3.py`
- Modify: `scaling.py`
- Modify: `ablation.py`
- Modify: `efe_ring_experiment.py`
- Modify: `generate_efe.py`
- Modify: `scaling_analysis.py`
- Modify: `multiseed_analysis.py`
- Modify: `make_figures.py`
- Modify: `compare_vocab_figures.py`
- Modify: `vfe3/inference/ring_task.py`
- Modify: `vfe3/run_artifacts.py`
- Modify: `vfe3/viz/report.py`
- Modify: `vfe3/viz/figures.py`
- Modify: `pyproject.toml`
- Test: `tests/test_2026_07_15_driver_reliability_remediation.py`

**Interfaces:**
- Run directory reservation is atomic and temporary files are unique per writer.
- Requested experiment failures remain visible in artifacts, analysis, and process status.
- Every expanded ablation/scaling arm is construction-validated after full merge.
- Deterministic and generation seeds are applied and persisted.

- [ ] **Step 1: Add failing regressions for collisions, deterministic state, survivor-only analysis, atomic publication, duplicate/zero seeds, unrecorded generation RNG, invalid expanded cells, false completion, dependency metadata, and global OpenMP suppression.**
- [ ] **Step 2: Run the focused tests and record the expected failures.**
- [ ] **Step 3: Reserve paths atomically with unique sibling temporary names and never remove a valid final figure because a later sidecar failed.**
- [ ] **Step 4: Persist failed-route records and return a failing process status when any requested scaling route fails; make analysis report missing/failed cells rather than silently fitting survivors.**
- [ ] **Step 5: Validate positive unique seeds and construction-validate every fully merged ablation/scaling configuration. Repair invalid arms by supplying their required route prerequisites without changing unrelated active defaults.**
- [ ] **Step 6: Apply the shared deterministic-runtime helper to the ring experiment and add an explicit persisted generation seed for stochastic EFE generation.**
- [ ] **Step 7: Declare NetworkX in the visualization extra and replace unconditional `KMP_DUPLICATE_LIB_OK` mutation with an explicit opt-in compatibility toggle.**
- [ ] **Step 8: Run focused tests until green and execute all driver, scaling, ablation, report, and runtime tests.**

### Task 7: Runtime Performance, Reuse, and Synchronization

**Findings:** P1, P2, P4, P5, P6, P7, S2-C2, S2-C3, S2-C4.

**Files:**
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/gauge_optim.py`
- Modify: `vfe3/train.py`
- Modify: `vfe3/ema.py`
- Modify: `vfe3/viz/extract.py`
- Modify: `vfe3/viz/report.py`
- Test: `tests/test_2026_07_15_performance_remediation.py`

**Interfaces:**
- Diagnostics preserve compact transport and reuse one authoritative inference snapshot.
- Direct-omega generator metrics are cached by immutable basis identity; validation transfers one aggregate status at a configured diagnostic cadence.
- EMA transfers at most one aggregate finiteness status per update.
- Production owns one authoritative implementation for objective helpers retained after dead-surface cleanup.

- [ ] **Step 1: Add behavior-preserving tests that count dense transport construction, inference calls, basis factorizations, host scalar transfers, and EMA finiteness decisions.**
- [ ] **Step 2: Run focused tests and confirm the counters expose each repeated operation.**
- [ ] **Step 3: Keep diagnostic computation on compact transport and pass the existing fixed-point snapshot through extraction and reporting.**
- [ ] **Step 4: Cache immutable direct-omega basis factorizations, aggregate validation status on device, and run determinant checks only at an explicit sparse diagnostic cadence.**
- [ ] **Step 5: Chunk phi projection work without changing exact projection semantics, accumulate validation metrics on device, and aggregate EMA finiteness into one device status.**
- [ ] **Step 6: Delete test-only objective surfaces only after tests use the production authority, or make those methods delegate directly to it.**
- [ ] **Step 7: Run focused tests until green and execute model diagnostics, gauge optimizer, EMA, extraction, and reporting tests.**

### Task 8: Architecture Adjudication, Documentation, and Consolidated Verification

**Finding:** S2-T1 plus closure of the complete two-panel ledger.

**Files:**
- Modify: `docs/2026-07-15-edits.md`
- Test: `tests/test_use_prior_bank.py`
- Test: `tests/test_2026_07_15_architecture_contract.py`

**Interfaces:**
- `use_prior_bank=False` remains an allowed affine output projection.
- `use_prior_bank=True` remains a reachable opt-in pure decode path.

- [ ] **Step 1: Add a regression proving both decode routes construct and that the pure route uses the prior bank without an affine vocabulary projection.**
- [ ] **Step 2: Run the architecture tests and preserve all current/default values.**
- [ ] **Step 3: Complete `docs/2026-07-15-edits.md` with one row per unique finding, including the two deduplications and S2-T1's user-adjudicated disposition.**
- [ ] **Step 4: Run an independent whole-branch review and fix every Critical or Important finding, then re-review.**
- [ ] **Step 5: Run `python -m pytest --junitxml=C:\tmp\vfe3-two-panel-remediation-final-20260715.xml` and read counts from XML.**
- [ ] **Step 6: Run `python -m pytest --runslow --junitxml=C:\tmp\vfe3-two-panel-remediation-runslow-20260715.xml` and read counts from XML.**
- [ ] **Step 7: Parse every tracked Python file with `ast.parse`, run `python -m pip check`, inspect `git status --short` and the complete diff, then complete the mandatory push/merge/cleanup lifecycle.**
