# Ultradeep Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair punch-list items 3 through 16 from the 2026-08-13 ultradeep audit while
explicitly waiving source-drift enforcement and artifact rejection.

**Architecture:** Correct reusable configuration and executable registries first, derive all
scientific certificates from the built path, then harden numerical kernels and reporting/process
boundaries. Every root cause receives a red/green regression, a small commit, task review, and
revision-bound evidence.

**Tech Stack:** Python 3.14 CPU interpreter, Python 3.12 Anaconda/CUDA interpreter, PyTorch,
pytest, JUnit XML, Ruff, Git, JSON/SHA-256.

**Spec:** `docs/superpowers/specs/2026-08-13-ultradeep-audit-remediation-design.md`

**Status:** Complete on `codex/ultradeep-remediation-20260813`. Source/test evidence is bound to
`659487ccd8ad9dd3c7b6afa042a4db04f578c04b`; the containing documentation commit is a docs-only
descendant and is recorded exactly in the final transfer artifact and closure ledger. No merge or
push was performed.

## Global Constraints

- Work only in `C:/tmp/V3_Transformer_ultradeep_remediation_20260813` on
  `codex/ultradeep-remediation-20260813`.
- Preserve the live checkout's modified `ablation.py`, `train_vfe3.py`, `vfe3/config.py`, and
  untracked `zzzzz.py`; never copy implementation changes into the live checkout during this plan.
- Do not reject or gate checkpoints, artifacts, cohorts, or reuse because source identity is
  `drifted`; `A01` and `A02` are owner-waived.
- Do not alter explicit launcher values merely to match restored dataclass defaults.
- Use `C:/Python314/python.exe` with `CUDA_VISIBLE_DEVICES=-1` and `VFE3_TEST_DEVICE=cpu` for CPU
  tests.
- Make no CUDA correctness claim without first verifying `C:/anaconda/python.exe`, checking GPU
  utilization/processes, and setting `VFE3_TEST_DEVICE=cuda`.
- Do not disturb a resident GPU workload. Wait and recheck until idle before Task 9's CUDA lane.
- Each production behavior follows RED -> verify expected failure -> GREEN -> adjacent tests.
- Record JUnit XML under `.verification/remediation-2026-08-13/` and derive totals from XML.
- Use additive/versioned serialized metadata; preserve existing public keys and default return
  shapes unless the spec explicitly corrects the value.
- Performance candidates `P01` through `P10` remain out of scope.

---

### Task 1: Restore reusable defaults and honest sweep construction

**Findings:** `A03`, `A04`, `A05`, `A06`, `A16`

**Files:**

- Modify: `vfe3/config.py`
- Modify: `ablation.py`
- Test: `tests/test_config.py`
- Test: `tests/test_2026_07_15_driver_reliability_remediation.py`
- Test: `tests/test_run_artifacts.py`
- Create: `tests/test_ultradeep_remediation_config_20260813.py`

**Interfaces:**

- `VFE3Config().pos_phi_compose == "bch"`
- `VFE3Config().decode_tau == 1.0`
- `VFE3Config().decode_mode == "diagonal_chunked"`
- `migrate_serialized_config(...)` assigns historical defaults when fields are absent.
- Covariance and Rényi sweep arms carry `decode_mode="family_chunked"`.
- General sweep construction selects one compatible baseline before arm application; it does not
  repair only selected arms after labeling.

- [x] **Step 1: Write RED default and migration regressions**

  Add literal assertions for the three reusable defaults. Serialize an old config with
  `pos_phi_compose`, `decode_tau`, and `decode_ce_checkpoint` absent and assert migration yields
  `"bch"`, `1.0`, and historical `"always"`. Add a configuration that was rejected only because
  omitted `pos_phi_compose` inherited `group_product`; assert it constructs with the reusable
  default.

- [x] **Step 2: Write RED sweep regressions**

  Assert every `covariance` and `renyi_order` arm resolves with `family_chunked`. Construct an
  incompatible baseline plus a one-factor arm and assert the resolved arm label and overrides
  contain no hidden positional-composition mutation.

- [x] **Step 3: Prove RED**

  Run:

  ```powershell
  $env:CUDA_VISIBLE_DEVICES='-1'; $env:VFE3_TEST_DEVICE='cpu'
  & 'C:/Python314/python.exe' -m pytest -q -p no:cacheprovider `
    tests/test_ultradeep_remediation_config_20260813.py `
    tests/test_config.py::test_pos_phi_default_is_learned_and_validates `
    tests/test_2026_07_15_driver_reliability_remediation.py `
    --junitxml=.verification/remediation-2026-08-13/task-01-red.xml
  ```

  Expected: default/migration assertions fail, and the registered sweep prerequisite assertions
  fail on the audited source.

- [x] **Step 4: Implement the minimal contract correction**

  Restore reusable defaults, make historical migration explicit by schema/version rather than
  current dataclass construction, restore the two sweep prerequisites, and replace selective
  `_repair_pos_phi_compose_prerequisite` behavior with one compatible resolved baseline.

- [x] **Step 5: Prove GREEN and adjacent construction coverage**

  Run the three listed modules plus `tests/test_ablation_sweep_route_compatibility_20260711.py`,
  `tests/test_ablation_reporting.py`, and `tests/test_ablation_tackon.py` into
  `.verification/remediation-2026-08-13/task-01-green.xml`. Parse the XML and record totals.

- [x] **Step 6: Commit**

  ```powershell
  git add vfe3/config.py ablation.py tests/test_config.py `
    tests/test_2026_07_15_driver_reliability_remediation.py tests/test_run_artifacts.py `
    tests/test_ultradeep_remediation_config_20260813.py
  git commit -m "fix: restore reusable configuration contracts"
  ```

---

### Task 2: Centralize BlockMLP, decode, scaling, and executable-build contracts

**Findings:** `A07` through `A15`, `A19`, `M31`, `M32`

**Files:**

- Modify: `vfe3/model/block_mlp.py`
- Modify: `vfe3/contracts.py`
- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/config.py`
- Modify: `scaling.py`
- Modify: `scaling_analysis.py`
- Modify: `ablation.py`
- Modify: `vfe3/process_utils.py`
- Test: `tests/test_gauge_block_mlp.py`
- Test: `tests/test_scaling_mup.py`
- Test: `tests/test_config.py`
- Test: `tests/test_prior_bank.py`
- Create: `tests/test_ultradeep_remediation_contracts_20260813.py`

**Interfaces:**

- `BlockMLPRegistration` owns factory, frame requirement, covariance kinds, gauge class,
  accounting hook, and report label.
- All modes implement
  `forward_moments(mu, sigma, *, frame_context: CanonicalFrameContext | None = None)`.
- `CanonicalFrameContext` rejects nonfinite, wrong-shape, or noninverse matrix pairs.
- `DecodeRegistration.__post_init__` rejects contradictory rank/capability metadata.
- `VFEModel` stores immutable executable build metadata used by reports.
- `_SCALING_STRUCTURAL_FIELDS` includes BlockMLP mode, covariance, expansion, activation, dropout,
  covariance floor, and enabled state.

- [x] **Step 1: Write RED contract regressions**

  Assert canonical mode rejects a missing frame; all modes accept the common `forward_moments`
  signature; invalid covariance strings fail identically at construction; noninverse frame pairs
  fail before propagation; contradictory decode registrations fail at construction; and mutating
  `cfg.use_block_mlp` after model construction cannot change executable-build reporting.

- [x] **Step 2: Write RED scaling/ablation/dead-path regressions**

  Create two equal-parameter scaling rows that differ only by BlockMLP mode/covariance and assert
  their structural signatures differ. Assert the reachable ablation table uses the same combined
  gauge label as sensitivity output. Assert inactive scaling decode temperature is reported
  `active=false`, not as an effective temperature.

- [x] **Step 3: Prove RED**

  Run the new contract module, BlockMLP tests, and focused scaling tests into
  `.verification/remediation-2026-08-13/task-02-red.xml`.

- [x] **Step 4: Implement registration and validation boundaries**

  Add the immutable registration, route constructors/accounting/reporting through it, validate
  frame inverses with a dtype-scaled residual, validate decode metadata in `__post_init__`, capture
  immutable build metadata, and extend scaling identity. Remove `NON_SWEPT_FIELDS` if the live
  coverage registry fully replaces it; otherwise turn it into the executable registry check.
  Remove `_sweep_is_complete`, unify gauge-label formatting, and correct
  `CompletedProcess[bytes]`.

- [x] **Step 5: Prove GREEN and call-site completeness**

  Run the five listed test modules plus `tests/test_model.py`,
  `tests/test_ablation_reporting.py`, and `tests/test_block_mlp_ablation_reporting.py` into
  `.verification/remediation-2026-08-13/task-02-green.xml`. Use `rg` to confirm BlockMLP modes are
  not duplicated in construction, accounting, and reporting switch statements.

- [x] **Step 6: Commit**

  ```powershell
  git add vfe3/model/block_mlp.py vfe3/contracts.py vfe3/model/prior_bank.py `
    vfe3/model/model.py vfe3/config.py scaling.py scaling_analysis.py ablation.py `
    vfe3/process_utils.py tests/test_gauge_block_mlp.py tests/test_scaling_mup.py `
    tests/test_config.py tests/test_prior_bank.py `
    tests/test_ultradeep_remediation_contracts_20260813.py
  git commit -m "refactor: centralize executable model contracts"
  ```

---

### Task 3: Make purity, causality, reflection scope, and exactness certificates truthful

**Findings:** `M02` through `M05`, `M07`, `M20`, `M29`, `M30`

**Files:**

- Modify: `vfe3/run_artifacts.py`
- Modify: `vfe3/geometry/transport.py`
- Modify: `vfe3/geometry/norms.py`
- Modify: `vfe3/attention_prior.py`
- Modify: `vfe3/model/free_energy.py`
- Modify: `ablation.py`
- Test: `tests/test_run_artifacts.py`
- Test: `tests/test_validated_geometry_numerics_20260713.py`
- Test: `tests/test_tier12_attention.py`
- Create: `tests/test_ultradeep_remediation_certificates_20260813.py`

**Interfaces:**

- Add `on_causal_lm_path: bool`.
- Add `transport_exactness_status` with exact values `exact`, `approximate`, `not_applicable`,
  `unknown`.
- Add `on_theory_pure_path: bool`.
- Preserve `on_gauge_pure_path`, but calculate it from every active GL-breaking executable seam.
- Persist reflection effective subgroup/scope.

- [x] **Step 1: Write RED certificate truth-table tests**

  Table-drive flat/covariant/gauge-fixed transports, LayerNorm, spectral cap, trust/exponential
  clipping, BlockMLP modes, and adaptive temperature. For each, assert the literal expected gauge,
  causal, exactness, and theory-pure outputs. Assert empty Regime-II history is `unknown`; an
  affirmative complete history is `exact`; a negative observation is `approximate`.

- [x] **Step 2: Write RED causality/reflection tests**

  Assert `uniform`/bidirectional priors produce `on_causal_lm_path=false` under next-token
  training, causal priors produce true, and gauge purity is not used as a synonym for causality.
  Assert block-GL reflection metadata identifies block zero and the accessible component count.

- [x] **Step 3: Prove RED**

  Run the new module and focused existing certificate tests into
  `.verification/remediation-2026-08-13/task-03-red.xml`.

- [x] **Step 4: Implement additive certificate facets**

  Derive flags from the active registrations and immutable Task-2 build metadata. Treat missing
  runtime evidence as unknown. Keep GL-breaking and noncausal diagnostic configurations
  executable but truthfully labeled.

- [x] **Step 5: Prove GREEN and report-schema compatibility**

  Run the four listed modules plus `tests/test_ablation_reporting.py` into
  `.verification/remediation-2026-08-13/task-03-green.xml`. Assert all preexisting report keys
  remain present and new keys are additive.

- [x] **Step 6: Commit**

  ```powershell
  git add vfe3/run_artifacts.py vfe3/geometry/transport.py vfe3/geometry/norms.py `
    vfe3/attention_prior.py vfe3/model/free_energy.py ablation.py `
    tests/test_run_artifacts.py tests/test_validated_geometry_numerics_20260713.py `
    tests/test_tier12_attention.py tests/test_ultradeep_remediation_certificates_20260813.py
  git commit -m "fix: derive scientific certificates from execution"
  ```

---

### Task 4: Enforce SPD and gauge-invariant precision policies

**Findings:** `M13`, `M14`

**Files:**

- Modify: `vfe3/geometry/transport.py`
- Modify: `vfe3/model/block_mlp.py`
- Modify: `vfe3/numerics.py`
- Test: `tests/test_full_gaussian_transport_precision_20260721.py`
- Test: `tests/test_gauge_block_mlp.py`
- Test: `tests/test_validated_geometry_numerics_20260713.py`
- Create: `tests/test_ultradeep_remediation_precision_20260813.py`

**Interfaces:**

- A fast congruence is accepted only when finite, symmetric, Cholesky-valid without semantic
  jitter, and within the configured residual/conditioning bound.
- Failed fast congruence recomputes and returns float64 rather than recasting to the failing dtype.
- GaugeGate uses checked Cholesky solves and float64 escalation for uncertain rows.

- [x] **Step 1: Write RED congruence regression**

  Use the audited strictly SPD fixture whose fp32 sandwich is finite and indefinite. Assert the
  public active helper returns an SPD float64 result matching direct float64 congruence and finite
  backward gradients.

- [x] **Step 2: Write RED GaugeGate regression**

  Construct two gauge-related, in-bounds ill-conditioned SPD inputs. Assert Mahalanobis invariants
  and final learned gates agree to a literal tolerance derived from float64 reference, while a
  well-conditioned fp32 case stays on the fast path.

- [x] **Step 3: Prove RED**

  Run the four files with a focused expression into
  `.verification/remediation-2026-08-13/task-04-red.xml`.

- [x] **Step 4: Implement validity-aware escalation**

  Add reusable residual/condition validation in `vfe3/numerics.py`, use it in congruence and
  GaugeGate, retain escalated precision, and fail visibly when neither path certifies the result.

- [x] **Step 5: Prove GREEN and geometry invariants**

  Run the full four modules plus `tests/test_transport.py` into
  `.verification/remediation-2026-08-13/task-04-green.xml`.

- [x] **Step 6: Commit**

  ```powershell
  git add vfe3/geometry/transport.py vfe3/model/block_mlp.py vfe3/numerics.py `
    tests/test_full_gaussian_transport_precision_20260721.py tests/test_gauge_block_mlp.py `
    tests/test_validated_geometry_numerics_20260713.py `
    tests/test_ultradeep_remediation_precision_20260813.py
  git commit -m "fix: validate covariance and gauge solves"
  ```

---

### Task 5: Harden remaining mathematical domains and decoder parity

**Findings:** `M01`, `M06`, `M08`, `M10` through `M12`, `M22`

**Files:**

- Modify: `vfe3/model/cg_coupling.py`
- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/families/gaussian.py`
- Modify: `vfe3/families/laplace.py`
- Modify: `vfe3/config.py`
- Modify: `vfe3/inference/e_step.py`
- Test: `tests/test_cg.py`
- Test: `tests/test_prior_bank.py`
- Test: `tests/test_families.py`
- Test: `tests/test_decode_nonpd_fallback_20260806.py`
- Create: `tests/test_ultradeep_remediation_domains_20260813.py`

**Interfaces:**

- CG `delta_full` returns `J Sigma J^T + floor * Sigma` with positive validated floor.
- Reflection state loading validates exact `{-1,+1}` values before module mutation.
- Family-consistent prior-bank decoding rejects Rényi `alpha > 1` during config validation.
- Runtime all-invalid rows return explicit excluded-token accounting rather than NaN.
- KL and expanded decoder policies escalate on numerical error bounds, not only failed operations.
- Laplace APIs use promoted public dtype.
- `decode_last` delegates to the identical last-position full-decoder kernel.

- [x] **Step 1: Write RED domain regressions**

  Add literal fixtures for a singular CG Jacobian, invalid reflection checkpoint, Rényi all-invalid
  row, successful-but-inaccurate fp32 KL, near-tied decoder rank reversal, mixed Laplace dtypes,
  and last-position/full-decode parity.

- [x] **Step 2: Prove RED**

  Run only the new cases into `.verification/remediation-2026-08-13/task-05-red.xml`; confirm each
  failure is the audited behavior rather than test setup.

- [x] **Step 3: Implement minimal domain guards**

  Add the covariant CG floor, load-state validation hook, config restriction plus runtime invalid
  row guard, error-bound-driven KL/decoder escalation, dtype promotion, and shared decode kernel.

- [x] **Step 4: Prove GREEN and adjacent decoder/family behavior**

  Run all five listed modules plus `tests/test_tier12_decode.py` into
  `.verification/remediation-2026-08-13/task-05-green.xml`.

- [x] **Step 5: Commit**

  ```powershell
  git add vfe3/model/cg_coupling.py vfe3/model/prior_bank.py `
    vfe3/families/gaussian.py vfe3/families/laplace.py vfe3/config.py `
    vfe3/inference/e_step.py tests/test_cg.py tests/test_prior_bank.py `
    tests/test_families.py tests/test_decode_nonpd_fallback_20260806.py `
    tests/test_ultradeep_remediation_domains_20260813.py
  git commit -m "fix: harden divergence and decoder domains"
  ```

---

### Task 6: Correct diagnostics, denominators, split discipline, and metric labels

**Findings:** `M15` through `M18`, `M24` through `M28`, `T01`

**Files:**

- Modify: `vfe3/model/model.py`
- Modify: `vfe3/train.py`
- Modify: `vfe3/run_artifacts.py`
- Modify: `vfe3/viz/figures.py`
- Modify: `scaling_analysis.py`
- Modify: `multiseed_analysis.py`
- Modify: `train_vfe3.py` comments/naming boundary only
- Test: `tests/test_train.py`
- Test: `tests/test_run_artifacts.py`
- Test: `tests/test_run_diagnostics_2026_06_13.py`
- Test: `tests/test_multiseed.py`
- Test: `tests/test_viz.py`
- Create: `tests/test_ultradeep_remediation_reporting_20260813.py`

**Interfaces:**

- Free-energy totals contain terms from one belief state; pre/post MLP values are separately named.
- `final_iterate` is unconditional; convergence/fixed-point/descent fields require evidence.
- Divergence labels come from the active family registration.
- Split artifacts persist `expected_targets`, `scored_targets`, `excluded_targets`.
- PPL is `exp(CE)` with overflow represented as `inf`.
- Exploratory naming/selection uses validation metrics, not held-out test PPL.
- `data_seed=None` reports nonshared data order.

- [x] **Step 1: Write RED diagnostic-state and label regressions**

  Assert no mixed pre/post total is emitted, a one-step/no-halt run is `final_iterate` without a
  convergence claim, target blindness carries no correlation-sign expectation, and non-KL plots
  never label the objective KL.

- [x] **Step 2: Write RED accounting/split regressions**

  Evaluate a fixture with known expected/scored/excluded counts and assert exact persisted values.
  Assert CE 25 yields PPL `exp(25)` rather than `exp(20)`, overflow yields `inf`, run sorting uses
  validation PPL, and `data_seed=None` does not claim shared order. Update the stale exact
  diagnostics dictionary to include the four production numerical-policy fields.

- [x] **Step 3: Prove RED**

  Run the new module and focused existing cases into
  `.verification/remediation-2026-08-13/task-06-red.xml`.

- [x] **Step 4: Implement reporting corrections**

  Split state-specific diagnostics, gate scientific labels on evidence, source figure labels from
  registrations, carry exact denominator counts, remove the PPL cap, move exploratory naming to
  validation metrics, correct data-seed and scheduler metadata, and update tests only after the
  production contract is fixed.

- [x] **Step 5: Prove GREEN and serialized compatibility**

  Run the six listed test modules plus `tests/test_run_naming.py` and
  `tests/test_scaling_mup.py` into `.verification/remediation-2026-08-13/task-06-green.xml`.

- [x] **Step 6: Commit**

  ```powershell
  git add vfe3/model/model.py vfe3/train.py vfe3/run_artifacts.py `
    vfe3/viz/figures.py scaling_analysis.py multiseed_analysis.py train_vfe3.py `
    tests/test_train.py tests/test_run_artifacts.py tests/test_run_diagnostics_2026_06_13.py `
    tests/test_multiseed.py tests/test_viz.py `
    tests/test_ultradeep_remediation_reporting_20260813.py
  git commit -m "fix: make diagnostics and evaluation accounting honest"
  ```

---

### Task 7: Recover UMAP workers and isolate periodic generation RNG

**Findings:** `A17`, `A18`, `M19`

**Files:**

- Modify: `vfe3/viz/figures.py`
- Modify: `vfe3/viz/report.py`
- Modify: `vfe3/process_utils.py`
- Modify: `vfe3/train.py`
- Test: `tests/test_figures_tail.py`
- Test: `tests/test_train.py`
- Create: `tests/test_ultradeep_remediation_process_20260813.py`

**Interfaces:**

- A timed-out UMAP request invalidates and replaces its executor.
- Process-tree termination and post-kill wait are bounded and return structured cleanup status.
- Periodic generation enters eval mode temporarily and restores model mode, CPU RNG, and every
  CUDA RNG state after generation.

- [x] **Step 1: Write RED UMAP lifecycle regressions**

  Use a controlled hanging worker followed by a successful worker. Assert the first request times
  out within the bound, the executor identity changes, the second request succeeds, and finalizer
  cleanup cannot wait indefinitely.

- [x] **Step 2: Write RED generation-state regressions**

  With active BlockMLP dropout, assert periodic generation leaves model/dropout training mode and
  CPU RNG exactly as found. If CUDA is unavailable in the CPU lane, assert the saved/restored CUDA
  RNG helper is invoked only when CUDA state exists; Task 9 supplies device integration.

- [x] **Step 3: Prove RED**

  Run the three listed test modules into
  `.verification/remediation-2026-08-13/task-07-red.xml`.

- [x] **Step 4: Implement bounded lifecycle and RNG guards**

  Replace the poisoned executor after timeout, use condition-based bounded cleanup, wrap generation
  in a mode/RNG preservation context, and avoid touching unrelated worker processes.

- [x] **Step 5: Prove GREEN**

  Run the full three modules into `.verification/remediation-2026-08-13/task-07-green.xml`.

- [x] **Step 6: Commit**

  ```powershell
  git add vfe3/viz/figures.py vfe3/viz/report.py vfe3/process_utils.py vfe3/train.py `
    tests/test_figures_tail.py tests/test_train.py `
    tests/test_ultradeep_remediation_process_20260813.py
  git commit -m "fix: bound diagnostics workers and generation state"
  ```

---

### Task 8: Run static, targeted, CPU-fast, and CPU-slow closure

**Files:**

- Modify only if a current failure demonstrates a remediation regression.
- Add JUnit XML under `.verification/remediation-2026-08-13/`.

**Interfaces:**

- Exact totals come from fresh JUnit XML.
- No known audit failure is relabeled stale without tracing the executable contract.

- [x] **Step 1: Run static checks**

  Run `ruff check` over every Python file changed since `714e3c5` and `git diff --check`. Repair
  only task-owned defects through a failing regression when behavior changes.

- [x] **Step 2: Run the targeted audit seam**

  Re-run the audit's 421-test targeted selection into
  `.verification/remediation-2026-08-13/targeted-final.xml`.

- [x] **Step 3: Run CPU-fast**

  Use the repository's CPU-fast policy/selection with the CPU interpreter and write
  `.verification/remediation-2026-08-13/cpu-fast-final.xml`.

- [x] **Step 4: Run CPU-slow**

  Use the repository's CPU-slow policy/selection and write
  `.verification/remediation-2026-08-13/cpu-slow-final.xml`.

- [x] **Step 5: Parse and reconcile failures**

  Parse XML, cluster failures by traceback root, and fix only regressions within the approved
  scope. Re-run every affected lane after the last fix; no stale pre-fix XML can support closure.

- [x] **Step 6: Commit any evidence-driven repairs**

  Commit production/tests only. Keep JUnit XML ignored.

---

### Task 9: Run idle-GPU cadence verification and close the remediation ledger

**Files:**

- Create: `.verification/remediation-2026-08-13/ledger.json` (ignored)
- Create: `.verification/remediation-2026-08-13/math-derivations.md` (ignored)
- Modify: `docs/audits/ultradeep-audit-codebase-2026-08-13.md`
- Modify: this plan's checkbox state

**Interfaces:**

- CUDA evidence uses `C:/anaconda/python.exe` and `VFE3_TEST_DEVICE=cuda`.
- The cadence comparison varies only periodic generation and compares model, optimizer, CPU RNG,
  CUDA RNG, and training metrics.
- Ledger states distinguish fixed, waived, refuted, and genuinely unavailable obligations.

- [x] **Step 1: Wait for an idle GPU gate**

  Verify Anaconda torch/CUDA, then inspect utilization, memory, and compute processes. If occupied,
  wait with bounded rechecks and do not launch tests until idle.

- [x] **Step 2: Run targeted CUDA coverage**

  Execute the project's CUDA lane with the required device environment into
  `.verification/remediation-2026-08-13/cuda-final.xml`.

- [x] **Step 3: Run the M19 cadence-parity experiment**

  Run two deterministic short training trajectories differing only in periodic generation.
  Persist inputs, config, environment, and exact comparison output. Require equality of model and
  optimizer state plus CPU/CUDA RNG state at the comparison boundary.

- [x] **Step 4: Dispatch broad final review and fix once**

  Review `714e3c5..HEAD` against the spec, plan, SDD rulings, and fresh evidence. Send all accepted
  findings to one fix worker, run covering tests, and perform one scoped re-review.

- [x] **Step 5: Build and validate the closure ledger**

  Start a fresh closure ledger after the final source commit. Code/experiment claims link current
  mechanical output; mathematical claims link current derivations. Record `A01`/`A02` as owner-
  waived policy decisions rather than fixed claims. Validate with the installed verification gate
  and retire the active marker through the hook.

- [x] **Step 6: Append remediation status and commit documentation**

  Append final revision, exact JUnit totals, claim states, owner waivers, and any open obligation to
  the audit. Update plan checkboxes, commit documentation, and revalidate any revision-bound
  evidence affected by the documentation commit.

- [x] **Step 7: Handoff without merge or push**

  Report branch, commits, exact evidence totals, validated ledger path, rulings, and remaining
  obligations. Do not merge, push, or modify the live checkout without a separate request.
