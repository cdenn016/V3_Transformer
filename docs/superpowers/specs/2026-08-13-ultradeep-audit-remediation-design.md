# Ultradeep Audit Remediation

**Date:** 2026-08-13
**Status:** Approved design
**Source revision:** `714e3c5be458ef489a93e720468efa8f637a838b` (`origin/main`)
**Branch:** `codex/ultradeep-remediation-20260813`
**Audit:** `docs/audits/ultradeep-audit-codebase-2026-08-13.md`
**Scope:** Implement punch-list items 3 through 16; explicitly waive items 1 and 2

## 1. Goal and owner policy

Repair the verified configuration, scientific-certificate, numerical, reporting, and process-
robustness defects from the 2026-08-13 ultradeep audit. Work occurs only in the isolated
remediation worktree. The user's dirty live checkout and its intentional launcher/config values
remain byte-for-byte untouched.

The owner explicitly waives the two provenance-policy findings:

- `A01`: do not make process/disk source drift fail closed.
- `A02`: do not reject persisted artifacts or cohorts because their source identity is marked
  `drifted`.

No implementation task may reintroduce either policy indirectly. Existing identity metadata may
remain descriptive, but it must not become a load, resume, reuse, cohort, or analysis gate.

## 2. Selected approach

Use a **compatibility-first, dependency-ordered remediation** in four implementation clusters:

1. reusable configuration, migration, sweep, registry, and executable-build contracts;
2. gauge, causal, and exactness certificates;
3. finite-precision and mathematical-domain integrity;
4. diagnostic semantics, evaluation accounting, and process cleanup.

Every verified defect begins with a failing regression test. Public return types and serialized
keys remain stable where a correction does not require additive metadata. New metadata is
additive and versioned. Experiment-local launcher choices remain explicit and unchanged; only
reusable defaults and invalid compatibility contracts are corrected.

Rejected alternatives:

- **Minimal symptom patches:** smaller, but would preserve fragmented registry contracts and allow
  the same drift between construction, reporting, and analysis to recur.
- **Full schema/registry rewrite:** conceptually cleaner, but too disruptive for a correctness
  remediation and unnecessary to close the audit findings.

## 3. Configuration, migration, and sweep contracts

### 3.1 Reusable defaults and legacy migration

Restore library-safe defaults without changing checked-in experiment launchers that explicitly
select other values:

- `VFE3Config.pos_phi_compose` returns to `"bch"`.
- `VFE3Config.decode_tau` returns to `1.0`.
- `VFE3Config.decode_mode` remains `"diagonal_chunked"`.
- Deserializing a historical config that omits either restored field receives the historical
  reusable default, not the current experiment's launcher choice.
- Historical checkpoints that predate `decode_ce_checkpoint` migrate to the historical
  `"always"` behavior. New configs keep the documented current default.

This closes `A03`, `A04`, and `A16`. Explicit launcher values such as
`pos_phi_compose="group_product"` and `decode_tau=0.01` remain unchanged and continue to be
validated as deliberate experiment choices.

### 3.2 Sweep construction and one-factor honesty

Reinstate `decode_mode="family_chunked"` as a shared prerequisite for the registered
`covariance` and `renyi_order` sweeps. Replace late arm-by-arm `pos_phi_compose` repair with a
precomputed sweep baseline that is compatible with every arm unless the sweep explicitly studies
positional composition. A named one-factor sweep must not silently mutate a second scientific
factor.

Sweep validation reports the exact incompatible field set and aborts before scheduling when a
registry declaration is internally inconsistent. This closes `A05` and `A06`.

### 3.3 Registry-owned contracts

Introduce one immutable BlockMLP registration record containing:

- constructor/factory;
- required frame context;
- supported covariance contracts;
- gauge-compatibility class;
- parameter-count/accounting hook;
- durable reporting label.

All BlockMLP modes expose one `forward_moments(mu, sigma, *, frame_context=None)` protocol. Modes
that require canonical frames reject a missing context; no inherited frame-free call silently
executes. Covariance strings are validated once against the registration before any mode-specific
fallback. `CanonicalFrameContext` validates shape, finiteness, and mutual-inverse residuals at its
construction boundary.

`DecodeRegistration` gains construction-time consistency validation for supported covariance
kinds, `supports_full`, fused CE hooks, chunking, and family consistency. Scaling signatures and
cohort keys include every BlockMLP structural field, not only parameter count.

Executable build metadata is captured immutably when `VFEModel` is constructed. Reports query the
actual built modules/registrations rather than a later-mutated config object. This closes `A07`,
`A11` through `A15`, and `M31`.

### 3.4 Dead/duplicated contract cleanup

- Replace the dead `NON_SWEPT_FIELDS` prose tuple with an executable coverage declaration checked
  against live registries, or remove it if the same contract already exists elsewhere.
- Route all ablation gauge labels through one helper that includes BlockMLP compatibility.
- Remove the weaker unused `_sweep_is_complete` predicate.
- Correct `run_process_tree`'s binary `CompletedProcess[bytes]` annotation.
- Record whether an inert scaling knob is active; do not imply that `decode_tau` affects a linear
  decoder.

These changes close `A08` through `A10`, `A19`, and `M32` without altering active experiment
values.

## 4. Scientific certificate contract

One overloaded boolean cannot represent gauge equivariance, causal language-model validity, and
runtime exactness. Preserve compatibility while splitting those meanings.

### 4.1 Additive certificate facets

The pure-path report adds versioned, additive fields:

- `on_gauge_pure_path`: conjunction of all executable GL(K)-equivariance obligations;
- `on_causal_lm_path`: both attention priors are causally masked for next-token training;
- `transport_exactness_status`: `"exact"`, `"approximate"`, `"not_applicable"`, or `"unknown"`;
- `on_theory_pure_path`: conjunction of gauge purity, causal LM validity, and affirmative
  exactness where exactness is applicable.

Existing fields remain present, but their values are derived from these facets rather than from
partial config checks.

### 4.2 Complete gauge predicate

`on_gauge_pure_path` includes:

- the active transport registration's covariance/equivariance class;
- block and E-step normalization class (including LayerNorm exclusion);
- fixed-coordinate spectral caps and trust/clipping policies;
- BlockMLP mode/covariance contract from the executable registration;
- query-adaptive temperature when its trace rule is live;
- all other currently recorded gauge flags.

An explicitly warning-labeled GL-breaking baseline remains executable, but it can never receive a
gauge-pure certificate. This closes `M02` through `M05` and `M29`.

### 4.3 Causality and exactness

Attention-prior registrations declare causal scope. Next-token training with a noncausal prior is
allowed only as an explicitly labeled diagnostic route; artifacts set `on_causal_lm_path=false`
and `on_theory_pure_path=false`. Default checked-in launchers remain causal. This closes `M20`
without incorrectly treating causality as a gauge transformation property.

Missing Regime-II runtime exactness evidence maps to `"unknown"`, never `"exact"`. An exact
certificate requires an affirmative runtime observation over the reported history. Empty or
incomplete history remains unknown. This closes `M30` while preserving the actual group-product
transport implementation.

Reflection metadata records the effective residual subgroup and the fact that block-GL reflection
acts on block zero rather than implying all `2^H` components are accessible. This closes `M07`.

## 5. Numerical and mathematical-domain integrity

### 5.1 Full-covariance congruence

`fp32_escalate` validates mathematical validity, not only finiteness. The fast result must be
finite, symmetric within tolerance, Cholesky-valid without semantic jitter, and below the
configured condition/residual threshold. Failure recomputes the whole affected batch in float64
and retains the float64 result rather than casting it back into the failing representation.

The public result remains the exact congruence of the selected precision policy; no spectral
projection silently changes it. This closes `M13`.

### 5.2 GaugeGate invariant solve

GaugeGate computes each Mahalanobis invariant using a Cholesky solve with residual and condition
checks. Well-conditioned inputs may remain on the fast path. Uncertain or failed rows escalate to
float64, and the invariant is cast back only after the scalar has been computed. A row that cannot
be certified fails visibly rather than generating a gauge-dependent gate. This closes `M14`.

### 5.3 KL, decoder, Laplace, and Rényi paths

- Full-Gaussian `fp32_escalate` checks factorization residual, conditioning, and cancellation risk
  even when Cholesky reports success; risky rows are recomputed and retained in float64 (`M10`).
- Expanded decoder tiles use an error-bound-triggered float64 recomputation before committing a
  near-tied ranking. `decode_last` and full decoding share the same kernel/reduction order so
  slicing the last full position is numerically identical (`M11`, `M22`).
- Laplace divergence and natural-gradient public APIs promote input dtypes instead of
  unconditionally downcasting float64 (`M12`).
- A family-consistent prior-bank decoder with Rényi order above one is rejected at config
  validation unless its domain can be guaranteed for the entire scored prior set. Runtime kernels
  still guard an all-invalid row and return an explicit excluded-token result rather than NaN
  (`M08`).
- Reflection buffers loaded from checkpoints are validated for shape, finiteness, and the exact
  `{−1,+1}` domain before state mutation (`M06`).
- CG `delta_full` adds the configured covariant positive floor after the Jacobian pushforward so a
  singular Jacobian cannot map an SPD covariance to the PSD boundary (`M01`).

## 6. Diagnostics, evaluation, and experiment semantics

### 6.1 State-consistent free-energy diagnostics

Never add terms evaluated on different belief states. Persist separately labeled pre-BlockMLP and
post-BlockMLP components, and expose a total only when every constituent is evaluated on the same
state. Historical mixed totals are not silently reinterpreted. This closes `M15`.

One-step/no-halt outputs are labeled `final_iterate`. `converged`, `fixed_point`, and `descent`
require their respective residual, halt, or monotonicity evidence. Target blindness is reported as
a structural data-flow property, not as a prediction about Pearson-correlation sign or magnitude.
This closes `M16` and `M18`.

Figure labels use the active divergence registration's display name and units. Non-KL objectives
are never labeled `KL`; generic plots say `divergence` when no stronger name is available. This
closes `M17`.

### 6.2 Denominators and reported metrics

Evaluation/finalization persist exact expected, scored, and excluded target counts by split.
CE/PPL/BPC are explicitly conditional on scored targets. A zero-scored split fails visibly. This
closes `M24`.

Perplexity is the mathematical `exp(CE)`: overflow becomes `inf`, not the indistinguishable cap
`exp(20)`. Directory naming and exploratory summaries use validation metrics; the held-out test
metric is evaluated/persisted only at the final reporting boundary and is not the default model-
selection or run-sorting key. This closes `M25` and `M28`.

When `data_seed=None`, reports say that data order follows each resolved run seed and therefore is
not shared across a multiseed cohort. Shared-data-order claims require one explicit common data
seed. This closes `M26`. The scheduler comment and metadata accurately describe
`min_lr_frac=0.01` as a one-percent floor, closing `M27`.

## 7. UMAP/process finalization

A timed-out UMAP job invalidates its worker executor. The timed-out process tree is terminated,
cleanup is bounded, and subsequent requests receive a fresh executor. No poisoned worker remains
in a reusable pool. Post-kill waits have an explicit short timeout and report cleanup failure
without hanging run finalization. This closes `A17` and `A18`.

## 8. Compatibility and non-goals

- Do not modify the user's live checkout or copy implementation changes into it during this task.
- Do not change explicit values in `train_vfe3.py`, `ablation.py`, or `scaling.py` merely to match
  restored dataclass defaults. Change a launcher only when its own semantics/comment is the
  verified defect, and preserve the user's live WIP value during later integration.
- Do not reject a checkpoint, run, artifact, seed, or cohort because source identity is drifted.
- Do not turn source drift into a warning escalation, exception, or reuse gate.
- Do not run CUDA tests while another GPU workload is resident.
- Do not promote performance candidates `P01` through `P10`; they remain profiling obligations,
  not verified defects in this remediation.
- Do not redesign the model's scientific objective or add a new architecture.

## 9. Test and evidence strategy

Every task uses red/green TDD. CPU commands use `C:/Python314/python.exe` with
`CUDA_VISIBLE_DEVICES=-1` and `VFE3_TEST_DEVICE=cpu`. JUnit XML is written inside the isolated
worktree and totals are parsed from XML.

Verification proceeds in increasing scope:

1. one focused regression group per root cause;
2. adjacent subsystem suites after each implementation cluster;
3. targeted audit seam;
4. full CPU-fast lane;
5. CPU-slow integration lane;
6. static checks over every modified Python file;
7. independent code review plus a new revision-bound closure ledger.

Before CUDA work, verify `C:/anaconda/python.exe` reports CUDA and inspect GPU utilization,
resident memory, and active compute processes. If the GPU is busy, wait and recheck rather than
interfere. Once idle, run the project's CUDA lane with `VFE3_TEST_DEVICE=cuda`, then run the
`M19` cadence-parity regression: identical seeds/config/data with periodic generation toggled,
checking model state, optimizer state, CPU RNG, CUDA RNG, and training metrics. Generation must
temporarily enter evaluation mode and restore both model mode and all RNG states.

The final evidence record distinguishes:

- mechanically fixed and passing claims;
- waived `A01`/`A02` policy findings;
- any genuinely unavailable CUDA obligation as `INCONCLUSIVE` rather than inferred from CPU
  evidence.

## 10. Delivery

Implementation is split into small dependency-ordered commits, each independently reviewed.
After all gates pass, append remediation status to the dated audit report, validate the new claim
ledger, and hand off the isolated branch, commits, exact JUnit totals, and any remaining open
obligation. Do not merge, push, or alter the live branch without a separate user request.
