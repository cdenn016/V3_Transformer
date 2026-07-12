# July 11 Deep-Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Repair every actionable defect retained by the post-merge investigation while leaving
Informational, Refuted, and explicitly correct-first roadmap findings unchanged.

**Architecture:** The cache repairs preserve tensor identity without strong ownership: bounded,
weak-reference, version-aware front caches eliminate repeated work and release device tensors when
their owning gauge group dies. The remaining repairs are surgical test-oracle, numerical-contract,
error-cleanup, and type/naming corrections. Existing entry-point configurations and mathematical
paths remain unchanged.

**Tech Stack:** Python 3, PyTorch, pytest, Matplotlib, Git, Windows PowerShell.

## Global Constraints

- docs/audits/audit-2026-07-11-investigation.md is the authoritative finding record.
- Do not change train_vfe3.py, ablation.py, scaling.py, or checked-in configuration choices.
- Preserve the default float32 path and every theoretically pure path.
- Preserve values except where F12 explicitly restores float64 input accuracy.
- Caches must not strongly own generator tensors and must detect in-place mutation.
- Cache capacity is 32 live entries and identity reuse must not return stale values.
- Keep PriorBank production decode unchanged; F2 is a golden-oracle coverage repair.
- Preserve checkpoint fields, CSV metric keys, and the fisher_trace compatibility name.
- Do not implement or delete F5, F6, F9, F16-F18, or F20-F22.
- Add regression tests before production edits and observe RED. The user waived a completed
  baseline; an initially started run was stopped and discarded, so validation used for the
  remediation begins only after a test file changes.
- Run pytest without an extra -q and derive counts from JUnit XML.
- Use American English and aligned function signatures.

### Task 1: Bound the Killing inverse cache and cache per-block results by parent basis

**Files:**
- Modify: vfe3/geometry/phi_preconditioner.py
- Modify: tests/test_phi_preconditioner.py

**Interfaces:**
- Consumes: build_killing_preconditioner and build_killing_preconditioner_per_block
- Produces: identical inverse tensors through a bounded weak cache shared by both modes

- [ ] **Step 1: Write failing lifetime and per-block-hit regressions**

Add tests that clear phi_preconditioner._KILLING_INV_CACHE, call the per-block builder twice on one
block_glk basis, and assert the second result is the first cached tensor. Add a weak-reference test
that deletes the only caller-owned basis, runs gc.collect(), and asserts the basis and entry are
gone. Add an in-place mutation test that asserts the inverse is recomputed.

~~~python
def test_killing_per_block_caches_parent_without_strong_retention() -> None:
    cache = phi_preconditioner._KILLING_INV_CACHE
    cache.clear()
    group = get_group("block_glk")(4, 2)
    generators = group.generators
    first = build_killing_preconditioner_per_block(generators, group.irrep_dims)
    second = build_killing_preconditioner_per_block(generators, group.irrep_dims)
    assert second is first
    ref = weakref.ref(generators)
    del group, generators, first, second
    gc.collect()
    assert ref() is None
    assert not cache
~~~

- [ ] **Step 2: Run focused RED verification**

Run: python -m pytest tests/test_phi_preconditioner.py --junitxml=C:\tmp\vfe3-f4-red.xml

Expected: per-block identity and weak cleanup fail against the current fresh-sub-basis,
strong-reference cache.

- [ ] **Step 3: Implement the bounded weak identity/version cache**

Replace the data_ptr dictionary with an OrderedDict capped at 32. The key contains id, tensor
version, shape, dtype, device, variant metadata, center_reg, and tol. The value contains a weakref
and inverse. A hit is valid only when the reference resolves to the exact caller. A callback removes
only its own entry. Factor the eigendecomposition into an uncached private helper. Cache the full
inverse under variant full and the assembled per-block inverse under the stable parent basis plus
tuple(irrep_dims). Compute transient local blocks through the uncached helper.

- [ ] **Step 4: Run focused GREEN verification**

Run: python -m pytest tests/test_phi_preconditioner.py tests/test_gauge_optim.py
--junitxml=C:\tmp\vfe3-f4-green.xml

Expected: zero failures and errors with existing inverse-value pins unchanged.

- [ ] **Step 5: Commit**

Commit message: fix(geometry): bound Killing inverse cache

### Task 2: Put BCH closure lookup before hashing and keep BCH arithmetic in fp32

**Files:**
- Modify: vfe3/geometry/lie_ops.py
- Modify: tests/test_fix_gauge_audit.py
- Modify: tests/test_p1_compact_phi_block_transport_20260711.py

**Interfaces:**
- Consumes: warn_if_basis_not_closed and compose_bch
- Produces: one value hash per live basis identity/version and an fp32 BCH island under autocast

- [ ] **Step 1: Write failing cache-hit and autocast regressions**

Clear closure caches. Wrap _basis_value_signature with a counter, call the closure diagnostic twice
on one basis, and assert one signature call. Assert an equal-value clone hashes once but reuses the
value result, while in-place mutation forces a new signature. Add a CPU-bfloat16 autocast test that
calls dense and compact compose_bch on float32 inputs and requires float32 results equal to their
no-autocast references.

- [ ] **Step 2: Run focused RED verification**

Run: python -m pytest tests/test_fix_gauge_audit.py
tests/test_p1_compact_phi_block_transport_20260711.py
--junitxml=C:\tmp\vfe3-f1-f15-red.xml

Expected: the same identity hashes twice and BCH returns an autocast dtype.

- [ ] **Step 3: Implement the weak identity-to-value-signature cache**

Add a 32-entry OrderedDict keyed by id, tensor version, shape, dtype, and device. Store only a weak
reference and stable value signature. On an identity hit, reach the existing value cache before any
detach, CPU copy, byte conversion, or SHA-256. On a miss, hash once, store the weak mapping, and
retain existing equal-value sharing. Reject stale identity reuse by checking reference identity.

- [ ] **Step 4: Implement the conditional fp32 BCH island**

When float32 coordinates enter compose_bch under autocast, call the same implementation once inside
torch.amp.autocast for that device with enabled=False through a private recursion guard. Dense and
compact Dynkin paths consume float32. Float64 remains float64; no-autocast execution enters no new
context.

- [ ] **Step 5: Run focused GREEN verification**

Run: python -m pytest tests/test_fix_gauge_audit.py tests/test_perf_equivalence.py
tests/test_p1_compact_phi_block_transport_20260711.py tests/test_amp.py
--junitxml=C:\tmp\vfe3-f1-f15-green.xml

Expected: zero failures and errors.

- [ ] **Step 6: Commit**

Commit message: fix(geometry): cache BCH closure identity

### Task 3: Replace the drifted decode twin with the production reference oracle

**Files:**
- Modify: tests/test_prior_bank.py

**Interfaces:**
- Consumes: PriorBank.reference_decode
- Produces: fused-versus-reference pins for tied token, untied token, tied model-channel, and
  untied model-channel tables, including bounded-variance extremes

- [ ] **Step 1: Add an off-default test that exposes the local twin**

Parameterize the four table routes. Perturb the active decode mean table and set active log-variance
values across the lower floor, interior, and upper cap. Compare fused decode against the existing
local _reference_decode first.

- [ ] **Step 2: Run focused RED verification**

Run: python -m pytest tests/test_prior_bank.py --junitxml=C:\tmp\vfe3-f2-red.xml

Expected: untied/model-channel cases fail because the local twin reads encode tables, and extreme
log-variances expose its bare-exp mismatch.

- [ ] **Step 3: Use PriorBank.reference_decode everywhere**

Delete the local twin and private divergence/family imports. Replace all golden calls with
pb.reference_decode(mu_q, sigma_q, tau=...). Do not edit production decode.

- [ ] **Step 4: Run focused GREEN verification**

Run: python -m pytest tests/test_prior_bank.py tests/test_tier12_decode.py
--junitxml=C:\tmp\vfe3-f2-green.xml

Expected: zero failures and errors.

- [ ] **Step 5: Commit**

Commit message: test(decode): use PriorBank reference oracle

### Task 4: Repair the remaining behavioral and numerical Low findings

**Files:**
- Modify: vfe3/run_artifacts.py
- Modify: vfe3/geometry/retraction.py
- Modify: vfe3/free_energy.py
- Modify: vfe3/config.py
- Modify: vfe3/inference/e_step.py
- Modify: tests/test_run_artifacts.py
- Modify: tests/test_retraction.py
- Modify: tests/test_tier12_attention.py
- Modify: tests/test_tier12_estep.py
- Modify: tests/test_config.py

**Interfaces:**
- Produces: F10 figure cleanup, F12 float64 preservation, and direct/config guards for F13/F14

- [ ] **Step 1: Write failing regressions**

For both attention writers, make plot_attention_heatmap create a figure and raise, then assert only
new figures close. For F12, compare a float64 full-SPD diagonal 1e-8 tangent against its analytic
affine exponential at 1e-12 tolerance. For F13, reject negative, NaN, and positive-infinite c and
query_tau_c. For F14, reject direct mm_exact calls with zero, negative, greater-than-one, NaN, and
infinite mm_damping.

- [ ] **Step 2: Run focused RED verification**

Run: python -m pytest tests/test_run_artifacts.py tests/test_retraction.py
tests/test_tier12_attention.py tests/test_tier12_estep.py tests/test_config.py
--junitxml=C:\tmp\vfe3-low-runtime-red.xml

Expected: new cases fail on figure leakage, rounded float64 output, and invalid direct values.

- [ ] **Step 3: Implement the four surgical repairs**

Mirror vfe3.viz.report._emit by snapshotting figure numbers before try and closing only newly
registered figures in except. Fix touched spelling to color. In retract_spd_full compute in float64
for float64 input and float32 otherwise inside the disabled-autocast island. Reject c and
query_tau_c unless finite and nonnegative. Under mm_exact, reject mm_damping unless finite and in
(0, 1] before kernel work.

- [ ] **Step 4: Run focused GREEN verification**

Run the same five files with --junitxml=C:\tmp\vfe3-low-runtime-green.xml.

Expected: zero failures and errors with valid values unchanged.

- [ ] **Step 5: Commit**

Commit message: fix(runtime): harden audit boundary cases

### Task 5: Correct typing seams and the half-Fisher naming contract

**Files:**
- Create: vfe3/contracts.py
- Create: tests/test_audit_contract_types_20260711.py
- Modify: vfe3/model/prior_bank.py
- Modify: vfe3/model/block.py
- Modify: vfe3/model/stack.py
- Modify: vfe3/model/model.py
- Modify: vfe3/inference/e_step.py
- Modify: vfe3/run_artifacts.py
- Modify: vfe3/train.py
- Modify: vfe3/metrics.py
- Modify: vfe3/viz/figures.py
- Modify: tests/test_metrics.py

**Interfaces:**
- Produces: concrete registry callables; TypedDict data/capture/gradient contracts;
  Optional[List[Path]] figure returns; tuple-aware _as_coeff; half_fisher_trace plus compatibility
  alias fisher_trace

- [ ] **Step 1: Write failing contract tests**

Use get_type_hints and inspect.signature to require concrete callables or TypedDicts instead of bare
Callable, dict, and list on cited seams. Require tuple in _as_coeff. Import half_fisher_trace,
assert one-half tr(Sigma^-1), and assert fisher_trace is its compatibility alias. Require dashboard
labels to say Half Fisher trace or display the divided-by-two formula.

- [ ] **Step 2: Run focused RED verification**

Run: python -m pytest tests/test_audit_contract_types_20260711.py tests/test_metrics.py
--junitxml=C:\tmp\vfe3-contracts-red.xml

Expected: the new symbol and concrete contracts do not exist.

- [ ] **Step 3: Add minimal shared contracts**

Create total=False TypedDicts for mutable M-step capture, tensor E-step gradient records, float
E-step gradient output, and load-time DataStateBuffer; create a required DataState for checkpoint
save input. Update only cited signatures and direct callers. Runtime dictionaries stay unchanged.

- [ ] **Step 4: Tighten registry, coefficient, and figure annotations**

Define EncodeCallable, DecodeCallable, and FusedCECallable aliases and apply them to PriorBank
registry storage, records, decorators, and getters. Change _as_coeff to float | list | tuple and
mention tuples in its docstring. Change map-writer returns to Optional[List[Path]].

- [ ] **Step 5: Correct the metric name compatibly**

Rename the implementation to half_fisher_trace, update production consumers, and retain
fisher_trace = half_fisher_trace. Keep fisher_trace_mean and fisher_trace_median artifact keys.
Describe the quantity as the KL quadratic coefficient and one-half mean-block Fisher trace; remove
the stale UMAP claim and use Half Fisher trace in human-facing labels.

- [ ] **Step 6: Run focused GREEN verification**

Run: python -m pytest tests/test_audit_contract_types_20260711.py tests/test_metrics.py
tests/test_run_diagnostics_2026_06_13.py tests/test_checkpoint_resume.py
--junitxml=C:\tmp\vfe3-contracts-green.xml

Expected: zero failures and errors.

- [ ] **Step 7: Commit**

Commit message: refactor(contracts): type audit seams precisely

### Task 6: Record closure, run consolidated verification, and prepare review

**Files:**
- Modify: docs/2026-07-11-edits.md
- Modify: docs/audits/audit-2026-07-11-investigation.md

**Interfaces:**
- Produces: a durable finding-to-commit and verification record

- [ ] **Step 1: Update the investigation and dated record**

Record each repaired finding, focused JUnit evidence, and reasons Informational/Refuted items remain
unchanged. State that no baseline completed or contributed a result before code changes.

- [ ] **Step 2: Run source and syntax gates**

Run: git diff --check

Run: python -m compileall -q vfe3

Expected: both exit zero.

- [ ] **Step 3: Run consolidated focused CPU verification**

Run all changed/directly covering tests with
--junitxml=C:\tmp\vfe3-audit-fixes-focused.xml.

Expected: zero failures and errors; read exact counts from XML.

- [ ] **Step 4: Run targeted RTX 5090 verification**

Set VFE3_TEST_DEVICE=cuda and run geometry, BCH, retraction, decode, and MM tests with
--junitxml=C:\tmp\vfe3-audit-fixes-cuda.xml.

Expected: zero failures and errors; report skips.

- [ ] **Step 5: Run the complete default suite once**

Run: python -m pytest --junitxml=C:\tmp\vfe3-audit-fixes-full.xml

Expected: zero failures and errors; derive counts from JUnit attributes.

- [ ] **Step 6: Commit**

Commit message: docs(audit): record July 11 remediation

- [ ] **Step 7: Request whole-branch review**

Review cache lifetime/mutation safety, hash avoidance, numerical identity, oracle independence,
float64 accuracy, direct-call validation, compatibility aliases, test sensitivity, and scope.
