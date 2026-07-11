# Curated audit salvage: next-session TODO

## Resume safely

The live desktop checkout was not used for these edits. Preserve its uncommitted configuration and WIP exactly as found. Start the next implementation session from a fresh worktree based on a freshly fetched `origin/main`; do not stash, reset, restore, or clean the live checkout.

This salvage run implemented and source-reviewed M1-M2, C1-C11, S1-S10, O1, O2, and O8. The working branch was `fix/curated-audit-salvage-20260709` in `C:\tmp\V3_Transformer_curated_audit_salvage_20260709`. The recent task commits are:

- S8: `9848358` and `c5055f1`.
- S9: `8ba67c8`.
- S10: `fb1c79a`.
- O1: `82bf21b` and `9360aeb`.
- O2: `ac48af7` and `3626918`.
- O8: `399e978` and `ca64a90`, followed by a documentation-only routing-comment cleanup.

Do not repeat those tasks. Their tests were authored before production edits but deliberately not executed. No pytest, lint, compile, import, CPU-runtime, or CUDA-runtime success claim exists for S6 onward.

## Decision required before O3

`.superpowers/sdd/O3-brief.md` requires float64 inversion for non-skew group elements and drifted skew elements, while the project hard constraint says float32 throughout. Resolve this explicitly before O3:

1. Approve a bounded float64 inverse island, validate the residual there, and cast the public result back to the input dtype; or
2. Revise O3/O5 and their tolerances to remain entirely float32, then prove the drifted inverse/cache cases on CPU and RTX 5090 CUDA.

Do not choose silently. O5 depends directly on this contract.

## Remaining implementation order

Use the saved briefs under `.superpowers/sdd/`. Continue regression-first source ordering, per-task source review, `git diff --check`, the shared dated edit log, and one local commit per task. Continue deferring execution until every remaining implementation task is complete.

1. **O3** — shared exact group inverse, reflection-safe cache admission, one shared-context inverse per rollout, and phi-cache precision forwarding. Planned subject: `fix(cache): share exact group inverses and reject reflection`.
2. **O4** — fail-closed omega configuration and constructible ablation arms. Planned subject: `fix(omega): fail closed and repair ablation construction`.
3. **O5** — compact block transport, inverse health, dirty-row persistence, and dirty-row-only reorthogonalization. Depends on O3/O4. Planned subject: `fix(omega): preserve compact blocks and project dirty rows`.
4. **O6** — visualization replay of the trained gauge parameterization, stored omega, and reflection state. Depends on O4/O5. Planned subject: `fix(viz): replay the trained gauge parameterization`.
5. **O7** — preserve direct links under trivial vertex gauge and thread `mean_per_head` through both omega-direct builders. Depends on the final O3/O5 transport contract. Planned subject: `fix(transport): preserve links and per-head omega contraction`.
6. **R1** — unify scaling estimators/weights, require four sizes for offset fits, persist confounds/provenance, and label Finding 15 as a structural ablation. Planned subject: `fix(scaling): unify estimators and persist confounds`.
7. **R2** — correct active-work accounting and derive pure-path predicates from registry metadata. Consume O8 group metadata. Planned subject: `fix(report): account for active work and exactness`.
8. **R3** — create one immutable converged diagnostic snapshot and reuse it across evaluation consumers. Depends on S10/O6. Planned subject: `perf(diagnostics): reuse one converged evaluation snapshot`.
9. **R4** — keep direct-link transports factored and cache the clamp Gram without host synchronization. Depends on O7 and the accepted C10 closure-cache contract. Planned subject: `perf(transport): keep direct links factored`.
10. **R5** — implement its written figure/config/memory gates and repair both offline `best_model.pt` replay seams in `vfe3/viz/report.py`. `generate_figures` and `vocab_comparison_figures` must unwrap `model_state`, validate `config_fingerprint`, and require semantic agreement with the migrated config before `load_state_dict`. Add regressions for both. Planned subject: `fix(figures): guard reusable generation and close failures`.
11. **R6** — explicit condition kind, paired-finite average-rank Spearman behavior, and honest NaNs for degenerate/zero-token statistics. Follow the mandatory signature order: defined float `eps` precedes `kind`. Planned subject: `fix(metrics): make shape and degeneracy explicit`.
12. **R7** — two-root-form JUnit parsing, machine-derived helper counts, strict randomized E-step integer bounds, signature cleanup, and American English. Modify the S9-created `tests/test_fixes_20260709_scripts.py`; do not recreate it. Planned subject: `fix(tooling): derive test counts and enforce public contracts`.
13. **R8** — move covariance-class ownership into complete transport-registration records after O7/R2/R4 settle the seam. Planned subject: `refactor(transport): register covariance metadata at the seam`.

## Consolidated validation

Run these only after O3-O7 and R1-R8 are implemented and source-reviewed. `pyproject.toml` already supplies `-q`; never add another `-q`. Pass counts must come from JUnit attributes, including both `testsuite` and `testsuites` roots:

```text
passes = tests - failures - errors - skipped
```

State/data matrix:

```powershell
python -m pytest tests/test_deterministic.py tests/test_checkpoint_resume.py tests/test_gauge_optim.py tests/test_fixes_20260709_data.py tests/test_fixes_20260709_scripts.py tests/test_run_artifacts.py tests/test_scaling_mup.py tests/test_multiseed.py tests/test_sigma_gate.py tests/test_ema.py tests/test_grad_accum.py --junitxml=C:\tmp\vfe3-curated-state-data.xml
```

Omega/policy matrix:

```powershell
python -m pytest tests/test_policy_registry.py tests/test_efe_scorer.py tests/test_belief_cache.py tests/test_generate.py tests/test_ring_task.py tests/test_regime_ii_link.py tests/test_omega_direct.py tests/test_omega_metropolis.py tests/test_config.py tests/test_round3_registry_guards.py tests/test_rope.py tests/test_viz.py --junitxml=C:\tmp\vfe3-curated-omega-policy.xml
```

Reporting/tooling matrix:

```powershell
python -m pytest tests/test_curated_audit_reporting_20260709.py tests/test_reporting_additions.py tests/test_report.py tests/test_metrics.py tests/test_numerics.py tests/test_viz.py tests/test_run_artifacts.py tests/test_fixes_20260709_scripts.py tests/test_run_naming.py tests/test_phase0_forward_beliefs.py --junitxml=C:\tmp\vfe3-curated-reporting-tooling.xml
```

Then locate a CUDA-enabled interpreter instead of assuming the default `python` has CUDA. Record its path, Python/Torch/CUDA versions, and `torch.cuda.get_device_name(0)`. Run the focused geometry, belief-cache, omega optimizer, and S10 GradScaler-overflow cases with `VFE3_TEST_DEVICE=cuda` and a dedicated JUnit XML. If no CUDA interpreter exists, record `BLOCKED_ENVIRONMENT`; do not claim GPU validation.

Finally run:

```powershell
git diff --check
python -m compileall -q vfe3
python -m pytest --junitxml=C:\tmp\vfe3-curated-salvage-final.xml
```

Run import probes for every click-to-run entry point. Keep the XML files until exact ledger counts and commit/test evidence are recorded, then clean them up.

## Whole-branch review and closure

Before aggregate execution, review the complete fetched-`origin/main`-to-HEAD diff. Resolve or explicitly classify the existing follow-ups:

- C8 near-equal/off-diagonal Frechet-log regression.
- C10 process-global closure-LRU concurrency.
- C11 module-qualified retraction-helper AST escape.
- S5 remediation text, mmap instrumentation, and loader-call-site drift coverage.
- S9 empty `model_state` rejection timing and the intentional tensor-identical legacy migration rule.
- S10 consumed-batch versus optimizer-update counters and mixed-schema history figures.
- O1 gradient-safe zero-probability masking if a future train-time differentiable EFE path is added.
- O8 strict rejection of Boolean lookalikes for numeric configuration fields, if the broader config policy is made type-strict.

Verify that no configured value changed, the pure path remains available, and the live checkout's final status inventory matches its pre-salvage inventory. Close every ledger row from executable source plus machine-readable evidence, then commit only closure documentation with `docs(audit): close curated salvage ledger`.

## Integration record

- Fetched `origin/main` before this session's merge: `90f2361`.
- O8 functional review-close commit: `ca64a90`.
- Feature branch: `fix/curated-audit-salvage-20260709`.
- Feature worktree: `C:\tmp\V3_Transformer_curated_audit_salvage_20260709`.
- Feature branch final pre-merge SHA: `<fill after TODO commit>`.
- Merge worktree/branch: `<fill after integration>`.
- Merge commit: `<fill after integration>`.
- Final `origin/main`: `<fill after integration>`.
- Final remote feature branch: `<fill after fast-forward>`.
