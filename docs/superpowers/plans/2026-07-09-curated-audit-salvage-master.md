# Curated Audit Salvage Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all 117 nonexcluded 2026-07-09 audit identifiers with executable evidence while preserving the live checkout, existing configuration choices, and every mathematically pure path.

**Architecture:** Work proceeds in four independently reviewable subsystem plans followed by a repository-wide closure pass. The live Fable tree is only a read-only donor; every salvaged hunk is reconstructed on this branch and must pass the audit counterexample that justifies it.

**Tech Stack:** Python 3.10+, PyTorch 2.11, pytest 9, JUnit XML, NumPy, SciPy where already optional, matplotlib, Git worktrees, Windows PowerShell.

## Global Constraints

- Work only in `C:\tmp\V3_Transformer_curated_audit_salvage_20260709` on `fix/curated-audit-salvage-20260709`; never modify, stash, reset, or commit the live desktop checkout.
- Base is `origin/main` at `e504f1c5ad5d277f653534cfc7fb63fd3b1bee61`; the design commit is `ea46bc6`.
- Do not import Fable's configuration edits in `ablation.py` or `train_vfe3.py`, and do not import its `vfe3_policy_results` deletions.
- Preserve current config values and defaults. Broken generated sweep cells may receive structural compatibility fixes; baseline experiment choices may not change.
- P1-P6 and H1-H7 are deferred. Old numbered findings remain in scope even when their remedy improves performance.
- No neural networks, CLI argument parsing, or hidden precision downgrade. Keep float32 public behavior and explicit float64 numerical islands only where specified.
- Preserve a theoretically pure path under the existing toggles. Any approximation must be labeled and tested as an approximation.
- Use TDD: add the counterexample, run it and observe the expected failure, apply the minimal fix, rerun focused tests, then commit named files only.
- Never add a second `-q` to pytest. Full counts come from JUnit attributes.
- The current `python` has `torch=2.11.0+cpu`; locate a CUDA-enabled interpreter before final RTX 5090 validation or record the inability explicitly.
- Update the existing `docs/2026-07-09-edits.md`; do not create a second same-day edit note.

---

## File Structure

- `docs/audits/deep-audit-and-wikitext103-performance-investigation-2026-07-09.md`: imported second source ledger.
- `docs/audits/curated-audit-closure-ledger-2026-07-09.md`: one row per actionable identifier, evidence command, test, commit, and closure class.
- `docs/superpowers/plans/2026-07-09-curated-audit-core-math.md`: inference, hierarchy, free energy, SPD, and numerical repairs.
- `docs/superpowers/plans/2026-07-09-curated-audit-state-data.md`: deterministic, checkpoint, resume, data, provenance, and optimizer-loop repairs.
- `docs/superpowers/plans/2026-07-09-curated-audit-omega-policy.md`: omega, reflection, cache, registry, policy, and generation repairs.
- `docs/superpowers/plans/2026-07-09-curated-audit-reporting-tooling.md`: scaling, reports, figures, statistics, scripts, and API-style repairs.
- `tests/test_curated_objective_math_20260709.py`, `tests/test_curated_inference_math_20260709.py`, `tests/test_curated_geometry_math_20260709.py`, `tests/test_curated_math_contracts_20260709.py`, and `tests/test_curated_audit_reporting_20260709.py`: new focused regression entry points; state/omega work extends existing domain tests plus the explicitly created donor-regression modules.

## Authoritative Coverage Partition

| Plan | Prior-audit findings | New findings |
| --- | --- | --- |
| Core mathematics | 4, 6-10, 16-19, 23, 33-34, 36, 49-55, 106 | M1-M4, M6-M7, M11 |
| State and data | 25, 27, 30, 39-46, 57, 76-80, 85-87 | M5, M8, M10, L4, L6, L8 |
| Omega and policy | 11, 20-21, 35, 47, 58-59, 61, 63-74, 82-84, 90-91, 94-100 | M9, L1-L3, L5, L7 |
| Reporting and tooling | 2-3, 5, 12-15, 24, 28-29, 31-32, 37-38, 48, 60, 88-89, 101-105, 107 | none |

Findings 1, 22, 26, 56, 75, 81, and 93 retain their verified intentional-design status. Findings 62 and 92 were fixed before this branch. Finding 14 and Finding 87 each include the new audit addendum in their assigned plan.

### Task 1: Import the audit source and establish the closure ledger

**Files:** Create `docs/audits/deep-audit-and-wikitext103-performance-investigation-2026-07-09.md`; create `docs/audits/curated-audit-closure-ledger-2026-07-09.md`; modify `docs/2026-07-09-edits.md`.

**Interfaces:** The ledger produces the canonical row schema consumed by every later task: `ID | class | test | command | commit | evidence | status`. Valid classes are `FIXED`, `FAIL_CLOSED`, `RELABELED`, `INTENTIONAL`, and `DEFERRED_PERFORMANCE`.

- [ ] **Step 1: Verify and import the frozen audit artifact.** Run:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'C:\Users\chris and christine\Desktop\V3_Transformer\docs\audits\deep-audit-and-wikitext103-performance-investigation-2026-07-09.md'
```

Expected hash: `1D80E75C99DDC942AAFE3FA96D824D98C593DC41472F07C475C9AE63318E961F`. Copy that exact file into this worktree and verify the destination hash matches.

- [ ] **Step 2: Create the ledger with all identifiers.** Start every actionable row as `OPEN`; prefill the exclusions as follows:

```markdown
| ID | Class | Test | Command | Commit | Evidence | Status |
| 1 | INTENTIONAL | existing audit ruling | n/a | e504f1c | user-owned route labels | CLOSED |
| 62 | FIXED | tests/test_omega_direct.py | python -m pytest tests/test_omega_direct.py | 0f0ffd3 | BeliefState._replace preserves omega | CLOSED |
| P1-P6 | DEFERRED_PERFORMANCE | n/a | n/a | n/a | user-directed performance branch | CLOSED |
```

Enumerate 2-107, M1-M11, and L1-L8 explicitly; nest the addenda under 14 and 87. Do not use ranges in the final ledger.

- [ ] **Step 3: Validate the ledger inventory.** Add a small one-shot test in `tests/test_curated_audit_reporting_20260709.py` that parses the first table column and asserts 117 unique actionable IDs plus the nine preclosed prior rows and six deferred P rows. Run:

```powershell
python -m pytest tests/test_curated_audit_reporting_20260709.py::test_closure_ledger_inventory
```

Expected: PASS.

- [ ] **Step 4: Update the dated edit note and commit.** Commit only the two audit files, the inventory test, and `docs/2026-07-09-edits.md`:

```powershell
git commit -m "docs(audit): establish curated closure ledger"
```

### Task 2: Execute the core-mathematics plan

**Files:** Follow `docs/superpowers/plans/2026-07-09-curated-audit-core-math.md` exactly.

**Interfaces:** Produces shared contracts used later: differentiable truncation leaves, unified two-hop convention, reflection-aware phi objective, valid Log-Euclidean retraction, actual final-layer prior capture, and model-channel control parity.

- [ ] **Step 1:** Execute each core-plan task with a fresh subagent and two-stage review.
- [ ] **Step 2:** Run `python -m pytest tests/test_curated_objective_math_20260709.py tests/test_curated_inference_math_20260709.py tests/test_curated_geometry_math_20260709.py tests/test_curated_math_contracts_20260709.py tests/test_e_step.py tests/test_retraction.py tests/test_phi_reflection.py tests/test_model_channel_diagnostics_2026_06_13.py`.
- [ ] **Step 3:** Require all selected tests to pass before advancing; update each core ledger row with its exact command and commit.

### Task 3: Execute the state-and-data plan

**Files:** Follow `docs/superpowers/plans/2026-07-09-curated-audit-state-data.md` exactly.

**Interfaces:** Consumes the model contracts from Task 2; produces reversible deterministic setup, exact auxiliary/data resume state, semantic checkpoint fingerprints, authoritative `did_step`, and per-split provenance.

- [ ] **Step 1:** Execute each state-plan task with focused review.
- [ ] **Step 2:** Run `python -m pytest tests/test_deterministic.py tests/test_checkpoint_resume.py tests/test_run_artifacts.py tests/test_train.py tests/test_data.py tests/test_fixes_20260709_data.py tests/test_fixes_20260709_scripts.py tests/test_multiseed.py`.
- [ ] **Step 3:** Update the corresponding ledger rows only after the focused suite passes.

### Task 4: Execute the omega-and-policy plan

**Files:** Follow `docs/superpowers/plans/2026-07-09-curated-audit-omega-policy.md` exactly.

**Interfaces:** Consumes the transport and checkpoint contracts from Tasks 2-3; produces coherent omega/cache inverse behavior, fail-closed reflection caching, registry metadata, policy window contracts, and safe last-position generation.

- [ ] **Step 1:** Execute each omega/policy task with focused review.
- [ ] **Step 2:** Run `python -m pytest tests/test_omega_direct.py tests/test_phi_reflection.py tests/test_belief_cache.py tests/test_efe_scorer.py tests/test_generate.py tests/test_policy_registry.py`.
- [ ] **Step 3:** Update the corresponding ledger rows only after the focused suite passes.

### Task 5: Execute the reporting-and-tooling plan

**Files:** Follow `docs/superpowers/plans/2026-07-09-curated-audit-reporting-tooling.md` exactly.

**Interfaces:** Consumes all prior runtime contracts; produces honest scaling/report schemas, guarded figure generation, stable statistics, JUnit-derived helper counts, transport registry metadata, and the corrected nine-test baseline.

- [ ] **Step 1:** Execute each reporting/tooling task with focused review.
- [ ] **Step 2:** Run `python -m pytest tests/test_curated_audit_reporting_20260709.py tests/test_reporting_additions.py tests/test_report.py tests/test_metrics.py tests/test_viz.py tests/test_run_naming.py tests/test_phase0_forward_beliefs.py`.
- [ ] **Step 3:** Update the corresponding ledger rows only after the focused suite passes.

### Task 6: Repository-wide closure and final validation

**Files:** Modify `docs/audits/curated-audit-closure-ledger-2026-07-09.md` and `docs/2026-07-09-edits.md`; no source behavior changes are allowed in this task.

**Interfaces:** Consumes all subsystem outputs and produces final machine-readable and human-readable evidence.

- [ ] **Step 1: Run static checks.** Run `git diff --check`, `python -m compileall -q vfe3`, and import probes for every click-to-run entry point. Expected: zero errors.
- [ ] **Step 2: Locate and use CUDA.** Query available Python interpreters for `torch.cuda.is_available()`. Run the focused CUDA-tagged geometry/cache/optimizer tests with `VFE3_TEST_DEVICE=cuda`. Record interpreter path, Torch/CUDA versions, device name, and exact test counts. If no CUDA-enabled interpreter exists, record `BLOCKED_ENVIRONMENT` rather than claiming GPU validation.
- [ ] **Step 3: Run the full suite with JUnit.** Run:

```powershell
python -m pytest --junitxml=C:\tmp\vfe3-curated-salvage-final-20260709.xml
```

Expected XML attributes: `failures=0`, `errors=0`; report `tests`, `skipped`, and derived passes exactly from the XML.

- [ ] **Step 4: Independently verify the ledger.** A reviewer must inspect every row against source and test output. No `OPEN` row may remain; no row may cite only a comment.
- [ ] **Step 5: Verify noninterference.** Compare the live checkout's pre-salvage status inventory with its current inventory. The repair must not have written any live file.
- [ ] **Step 6: Final docs commit.** Record exact JUnit/CUDA/static evidence in the ledger and daily note, then commit:

```powershell
git commit -m "docs(audit): close curated salvage ledger"
```

## Execution Handoff

Preferred option: subagent-driven execution. Each subsystem task receives a fresh implementer, a spec-compliance review, and a code-quality review before the next task. Alternate option: inline execution in this session through `superpowers:executing-plans`, with batch checkpoints. Do not merge automatically; present the verified branch and integration choices after the ledger is closed.
