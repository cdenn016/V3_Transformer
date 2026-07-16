# E-step Fixed-Point and Phi-Numerics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add honest fixed-point, corpus-pass, BCH, and numerical-flatness diagnostics plus opt-in bounded phi optimization and exact group-product positional transport.

**Architecture:** Keep public runtime tensors float32 and move expensive reference calculations to validation/report boundaries. Preserve every existing default while adding a compatibility alias for `mm_exact`, a projected-M-step chart control, and a narrowly validated exact positional factorization on the flat tied-frame phi route.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib, pytest, JUnit XML, Git worktrees.

## Global Constraints

- Existing defaults and historical `mm_exact` configurations remain behavior-compatible.
- Public runtime tensors remain float32; sampled reference geometry calculations remain fp64 until scalar reduction.
- New diagnostics run only at log, validation, or report boundaries and cannot alter training state or RNG state.
- `phi_mstep_max_matrix_norm=None` and existing positional composition values execute no new behavior.
- `pos_phi_compose="group_product"` is supported only for phi parameterization, flat transport, and `s_frame_mode="tied"`.
- Every production change follows a witnessed failing test, then the minimum passing implementation.
- Pass counts come from JUnit XML and pytest is invoked without an extra `-q`.
- Update `docs/2026-07-15-edits.md`; do not create another dated edit record.

### Task 1: E-step updater naming and configuration contracts

**Files:**
- Modify: `vfe3/inference/e_step.py`
- Modify: `vfe3/config.py`
- Modify: `vfe3/model/block.py`
- Test: `tests/test_estep_fixed_point_reporting_20260715.py`

**Interfaces:**
- Consumes: the existing `e_step_update` dispatch and `mm_exact_update` callable.
- Produces: `canonical_e_step_update(name: str) -> str`, mapping `frozen_surrogate_exact` to the canonical `mm_exact` route; `phi_mstep_max_matrix_norm: Optional[float]` in `VFE3Config`; validated `pos_phi_compose="group_product"` route constraints.

- [x] **Step 1: Write failing alias and validation tests**

```python
def test_frozen_surrogate_exact_alias_matches_mm_exact_value_and_gradient():
    assert canonical_e_step_update("mm_exact") == "mm_exact"
    assert canonical_e_step_update("frozen_surrogate_exact") == "mm_exact"
    out_mm = run_e_step_update("mm_exact")
    out_alias = run_e_step_update("frozen_surrogate_exact")
    assert_tree_close(out_alias, out_mm)


def test_group_product_rejects_nonflat_or_independent_frame():
    with pytest.raises(ValueError, match="group_product"):
        VFE3Config(pos_phi="learned", pos_phi_compose="group_product",
                   transport_mode="regime_ii")
    with pytest.raises(ValueError, match="group_product"):
        VFE3Config(pos_phi="learned", pos_phi_compose="group_product",
                   s_frame_mode="phi_tilde")


def test_phi_mstep_max_matrix_norm_must_be_positive_or_none():
    VFE3Config(phi_mstep_max_matrix_norm=None)
    with pytest.raises(ValueError, match="phi_mstep_max_matrix_norm"):
        VFE3Config(phi_mstep_max_matrix_norm=0.0)
```

- [x] **Step 2: Run the tests and witness the missing alias and fields**

Run: `python -m pytest tests/test_estep_fixed_point_reporting_20260715.py --junitxml=C:\tmp\vfe3-estep-config-red-20260715.xml`

Expected: failures naming the missing registry alias or configuration fields.

- [x] **Step 3: Implement the compatibility contracts**

Add the small alias registry and canonicalize only at the dispatch boundary, add the optional chart bound, extend positional composition validation, and reject unsupported exact-factor routes with one explicit error message. Do not change the current defaults or the serialized spelling supplied by the user.

- [x] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_estep_fixed_point_reporting_20260715.py tests/test_config.py tests/test_e_step.py --junitxml=C:\tmp\vfe3-estep-config-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 5: Commit**

```powershell
git add vfe3/inference/e_step.py vfe3/config.py vfe3/model/block.py tests/test_estep_fixed_point_reporting_20260715.py
git commit -m "feat: clarify frozen-surrogate E-step contract"
```

### Task 2: Corpus-pass metrics and loss-curve boundaries

**Files:**
- Modify: `vfe3/train.py`
- Modify: `vfe3/run_artifacts.py`
- Modify: `vfe3/viz/figures.py`
- Test: `tests/test_training_epoch_metrics_20260715.py`

**Interfaces:**
- Consumes: `epoch`, `batches_consumed`, `len(train_loader)`, the per-step loss sequence.
- Produces: CSV fields `epoch`, `batch_in_epoch`, `steps_per_epoch`, and `corpus_pass`; `plot_trajectory(..., epoch_boundaries=...)`.

- [x] **Step 1: Write failing epoch and plotting tests**

```python
def test_training_rows_record_one_based_epoch_cursor_and_continuous_pass(tmp_path):
    rows = run_tiny_training(tmp_path, dataset_size_for_three_batches=True, n_steps=5)
    assert [(r["epoch"], r["batch_in_epoch"]) for r in rows] == [
        (1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    assert all(r["steps_per_epoch"] == 3 for r in rows)
    assert rows[-1]["corpus_pass"] == pytest.approx(5 / 3)


def test_loss_trajectory_draws_complete_corpus_pass_boundaries():
    fig = plot_trajectory([3.0, 2.0, 1.0, 0.5], steps=[1, 2, 3, 4],
                          epoch_boundaries=[2, 4])
    assert boundary_x_positions(fig.axes[0]) == [2.0, 4.0]
```

- [x] **Step 2: Run the tests and witness missing fields/argument**

Run: `python -m pytest tests/test_training_epoch_metrics_20260715.py --junitxml=C:\tmp\vfe3-epoch-red-20260715.xml`

Expected: failures for absent CSV fields and unsupported `epoch_boundaries`.

- [x] **Step 3: Implement epoch persistence and annotations**

Write one-based epoch and cursor fields into each CSV row, compute continuous corpus pass from the absolute step, and pass exact multiples of `len(train_loader)` to the loss figure. Draw subdued dashed boundary lines without changing trajectories when no boundaries are supplied.

- [x] **Step 4: Verify epoch behavior and resume regressions**

Run: `python -m pytest tests/test_training_epoch_metrics_20260715.py tests/test_train.py tests/test_checkpoint_resume.py tests/test_run_artifacts.py tests/test_viz.py --junitxml=C:\tmp\vfe3-epoch-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 5: Commit**

```powershell
git add vfe3/train.py vfe3/run_artifacts.py vfe3/viz/figures.py tests/test_training_epoch_metrics_20260715.py
git commit -m "feat: expose corpus-pass loss boundaries"
```

### Task 3: One-step-ahead fixed-point and inference-depth reporting

**Files:**
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/train.py`
- Modify: `vfe3/viz/report.py`
- Modify: `vfe3/viz/figures.py`
- Modify: `vfe3/run_artifacts.py`
- Test: `tests/test_estep_fixed_point_reporting_20260715.py`

**Interfaces:**
- Consumes: diagnostic `e_step_trace` beliefs at depths zero through `T+1`, fixed validation tokens, and existing attention/alpha evaluators.
- Produces: `estep_fp_kl`, `estep_fp_mu_rms`, `estep_fp_sigma_rms`, `estep_fp_phi_rms`, `estep_beta_js`, `estep_alpha_rms_delta`; `estep_depth_sensitivity.json`; `plot_estep_depth_sensitivity`.

- [x] **Step 1: Write failing fixed-point and depth-artifact tests**

```python
def test_one_step_ahead_residual_uses_q_t_to_q_t_plus_one():
    diag = fixed_point_diagnostics(model, tokens, n_steps=1)
    assert diag["estep_r_mu_last"] == pytest.approx(rms(diag.q1.mu - diag.q0.mu))
    assert diag["estep_fp_mu_rms"] == pytest.approx(rms(diag.q2.mu - diag.q1.mu))


def test_depth_sensitivity_marks_trained_depth_and_is_state_neutral(tmp_path):
    before = clone_state_and_rng(model)
    record = collect_estep_depth_sensitivity(model, loader, depths=[0, 1, 2])
    assert record["trained_depth"] == model.cfg.n_e_steps
    assert [p["depth"] for p in record["points"]] == [0, 1, 2]
    assert_state_and_rng_equal(before, model)
```

- [x] **Step 2: Run tests and witness absent fixed-point fields/functions**

Run: `python -m pytest tests/test_estep_fixed_point_reporting_20260715.py --junitxml=C:\tmp\vfe3-estep-report-red-20260715.xml`

Expected: failures for missing diagnostic fields and depth collector.

- [x] **Step 3: Implement state-neutral diagnostic replay**

Extend the validation replay to `T+1`, retain `q_T` as the decoded operating point, compute residuals only from detached tensors, and evaluate beta/alpha drift without mutating model state. Add the fixed subset depth collector and strict JSON writer.

- [x] **Step 4: Correct figure semantics**

Rename the existing configured-step series, mark the trained depth, add a log-scale one-step-ahead residual, and add the two-axis CE/free-energy depth plot. Historical traces without new fields must still render.

- [x] **Step 5: Run focused reporting tests**

Run: `python -m pytest tests/test_estep_fixed_point_reporting_20260715.py tests/test_reporting_additions.py tests/test_report.py tests/test_run_artifacts.py tests/test_viz.py --junitxml=C:\tmp\vfe3-estep-report-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 6: Commit**

```powershell
git add vfe3/model/model.py vfe3/train.py vfe3/viz/report.py vfe3/viz/figures.py vfe3/run_artifacts.py tests/test_estep_fixed_point_reporting_20260715.py
git commit -m "feat: report actual E-step fixed-point residuals"
```

### Task 4: Phi chart health and fp64 numerical-flatness references

**Files:**
- Modify: `vfe3/metrics.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/train.py`
- Modify: `vfe3/viz/figures.py`
- Modify: `vfe3/run_artifacts.py`
- Test: `tests/test_phi_numerics_buildout_20260715.py`

**Interfaces:**
- Produces: `phi_chart_statistics`, `bch_fidelity_statistics`, and `flatness_reference_statistics`; active-frame quantiles, clamp fraction/scale, condition quantiles, BCH fidelity/amplification, and fp32/fp64 absolute/relative closure metrics.

- [x] **Step 1: Write failing numerical-reference tests**

```python
def test_flatness_reference_reports_relative_and_fp64_residuals():
    stats = flatness_reference_statistics(ill_conditioned_flat_factors(), triples=fixed_triples())
    assert stats["numerical_holonomy_fp64_rel"] < stats["numerical_holonomy_fp32_rel"]
    assert stats["inverse_consistency_fp64"] < stats["inverse_consistency_fp32"]


def test_bch_fidelity_detects_large_chart_failure():
    stats = bch_fidelity_statistics(X_large, Y_large, order=4)
    assert stats["bch_relative_error_max"] > 0.5
    assert stats["bch_norm_amplification_max"] > 1.0
```

- [x] **Step 2: Run tests and witness missing metric APIs**

Run: `python -m pytest tests/test_phi_numerics_buildout_20260715.py --junitxml=C:\tmp\vfe3-phi-metrics-red-20260715.xml`

Expected: import or attribute failures for the new metric functions.

- [x] **Step 3: Implement sampled metric functions**

Use deterministic sampled triples. Preserve fp32 factors for runtime residuals, reconstruct sampled factors and products in fp64 for references, normalize by operand magnitude with `eps`, count nonfinite samples, and compute finite quantiles without hiding the nonfinite count.

- [x] **Step 4: Persist and relabel metrics**

Add active phi and condition quantiles to training/validation rows. Retain legacy holonomy columns, add explicitly named numerical closure columns, and select flatness versus curvature plot wording from `transport_mode`.

- [x] **Step 5: Run focused metric and visualization tests**

Run: `python -m pytest tests/test_phi_numerics_buildout_20260715.py tests/test_metrics.py tests/test_run_diagnostics_2026_06_13.py tests/test_reporting_additions.py tests/test_viz.py --junitxml=C:\tmp\vfe3-phi-metrics-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 6: Commit**

```powershell
git add vfe3/metrics.py vfe3/model/model.py vfe3/train.py vfe3/viz/figures.py vfe3/run_artifacts.py tests/test_phi_numerics_buildout_20260715.py
git commit -m "feat: distinguish numerical flatness from curvature"
```

### Task 5: Opt-in projected M-step phi chart

**Files:**
- Modify: `vfe3/gauge_optim.py`
- Modify: `vfe3/train.py`
- Modify: `vfe3/model/model.py`
- Test: `tests/test_phi_numerics_buildout_20260715.py`

**Interfaces:**
- Produces: `project_phi_parameter_rows_(model: VFEModel, max_matrix_norm: float) -> Dict[str, float]`, invoked once after a successful optimizer step.

- [x] **Step 1: Write failing projection tests**

```python
def test_project_phi_rows_bounds_embedded_matrix_norm_and_covers_tables():
    stats = project_phi_parameter_rows_(model_with_token_pos_and_model_phi(), 2.0)
    for table in live_phi_tables(model):
        assert embedded_row_norms(table, model.group.generators).max() <= 2.0 + 1e-6
    assert stats["phi_chart_projected_fraction"] > 0.0


def test_disabled_chart_projection_is_not_called(monkeypatch):
    model.cfg.phi_mstep_max_matrix_norm = None
    train_step(model, optimizer, scheduler, tokens, targets)
    assert projection_call_count(monkeypatch) == 0
```

- [x] **Step 2: Run tests and witness missing projection function**

Run: `python -m pytest tests/test_phi_numerics_buildout_20260715.py --junitxml=C:\tmp\vfe3-phi-project-red-20260715.xml`

Expected: missing function failure.

- [x] **Step 3: Implement chunked embedded-norm projection**

Enumerate trainable phi tables from the model, process large tables in row chunks, compute embedded Frobenius norms through the active generators, and scale only violating rows. Invoke after confirmed `did_step`; do not change optimizer moments or touch skipped steps.

- [x] **Step 4: Run optimizer and training regressions**

Run: `python -m pytest tests/test_phi_numerics_buildout_20260715.py tests/test_gauge_optim.py tests/test_train.py tests/test_phi_preconditioner.py --junitxml=C:\tmp\vfe3-phi-project-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 5: Commit**

```powershell
git add vfe3/gauge_optim.py vfe3/train.py vfe3/model/model.py tests/test_phi_numerics_buildout_20260715.py
git commit -m "feat: add opt-in projected phi M-step"
```

### Task 6: Exact positional group-product transport

**Files:**
- Modify: `vfe3/model/positional_phi.py`
- Modify: `vfe3/geometry/transport.py`
- Modify: `vfe3/inference/e_step.py`
- Modify: `vfe3/model/block.py`
- Modify: `vfe3/model/stack.py`
- Modify: `vfe3/model/model.py`
- Test: `tests/test_phi_numerics_buildout_20260715.py`

**Interfaces:**
- Produces: optional `right_phi` coordinate through transport builders; `U=exp(X)exp(Y)` and `U_inverse=exp(-Y)exp(-X)` in dense, factored, and compact block paths.

- [x] **Step 1: Write failing algebra and gradient tests**

```python
def test_group_product_vertex_and_inverse_match_direct_multiplication():
    built = build_factored_transport(X, group, right_phi=Y)
    assert_close(built.exp_phi, matrix_exp(embed(X)) @ matrix_exp(embed(Y)))
    assert_close(built.exp_neg_phi, matrix_exp(-embed(Y)) @ matrix_exp(-embed(X)))


def test_group_product_transport_gradients_reach_token_and_position():
    X.requires_grad_(True)
    Y.requires_grad_(True)
    loss = transport_mean(build_factored_transport(X, group, right_phi=Y), mu).square().mean()
    loss.backward()
    assert X.grad.abs().sum() > 0
    assert Y.grad.abs().sum() > 0
```

- [x] **Step 2: Run tests and witness unsupported `right_phi`**

Run: `python -m pytest tests/test_phi_numerics_buildout_20260715.py --junitxml=C:\tmp\vfe3-group-product-red-20260715.xml`

Expected: unexpected keyword or missing-route failure.

- [x] **Step 3: Implement exact vertex factor assembly**

Factor matrix-exponential construction into a helper returning positive and negative factors for dense or compact blocks. When `right_phi` is present, multiply positive factors in token-then-position order and inverse factors in reversed negative order. Preserve the no-`right_phi` path exactly.

- [x] **Step 4: Thread positional coordinates through the flat tied-frame route**

For `group_product`, keep token phi as the belief coordinate, build positional coordinates separately, and pass them through stack, block, E-step, shared-transport, diagnostics, and model-channel tied-frame calls. Existing BCH/euclidean modes continue composing coordinates before the stack.

- [x] **Step 5: Verify dense/factored/compact equivalence and model training**

Run: `python -m pytest tests/test_phi_numerics_buildout_20260715.py tests/test_transport.py tests/test_p1_compact_phi_block_transport_20260711.py tests/test_perf_equivalence.py tests/test_model.py --junitxml=C:\tmp\vfe3-group-product-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 6: Commit**

```powershell
git add vfe3/model/positional_phi.py vfe3/geometry/transport.py vfe3/inference/e_step.py vfe3/model/block.py vfe3/model/stack.py vfe3/model/model.py tests/test_phi_numerics_buildout_20260715.py
git commit -m "feat: add exact positional group-product transport"
```

### Task 7: Runnable ablations, documentation, and repository verification

**Files:**
- Modify: `ablation.py`
- Modify: `docs/2026-07-15-edits.md`
- Modify: `docs/superpowers/plans/2026-07-15-estep-fixed-point-phi-numerics.md`
- Test: `tests/test_estep_phi_ablation_routes_20260715.py`

**Interfaces:**
- Produces: `estep_depth_damping`, `phi_chart_control`, and `pos_phi_composition` sweep definitions.

- [x] **Step 1: Write failing ablation-route tests**

```python
def test_recommended_estep_and_phi_sweeps_are_registered_and_runnable():
    assert arm_labels("estep_depth_damping") == expected_depth_damping_labels
    assert arm_labels("phi_chart_control") == expected_phi_control_labels
    assert arm_labels("pos_phi_composition") == ["bch", "group_product", "none"]
    for name in ("estep_depth_damping", "phi_chart_control", "pos_phi_composition"):
        for arm in expand_sweep(name):
            VFE3Config(**resolved_config(arm))
```

- [x] **Step 2: Run tests and witness absent sweeps**

Run: `python -m pytest tests/test_estep_phi_ablation_routes_20260715.py --junitxml=C:\tmp\vfe3-ablation-routes-red-20260715.xml`

Expected: missing sweep-key failures.

- [x] **Step 3: Add explicit matched experiment arms**

Add fixed depth/damping, randomized depth, mass/learning-rate/natural-gradient/chart-bound, and BCH/group-product/positional-off arms. Keep the user’s active config dictionary values unchanged and leave the new sweeps opt-in.

- [x] **Step 4: Update the dated edit record and complete plan checkboxes**

Record exact files, semantics, TDD evidence, focused JUnit counts, full-suite JUnit counts, and any unavailable tooling. Do not claim long-run ablation results.

- [x] **Step 5: Run focused integration verification**

Run: `python -m pytest tests/test_estep_fixed_point_reporting_20260715.py tests/test_training_epoch_metrics_20260715.py tests/test_phi_numerics_buildout_20260715.py tests/test_estep_phi_ablation_routes_20260715.py tests/test_config.py tests/test_e_step.py tests/test_transport.py tests/test_train.py tests/test_run_artifacts.py tests/test_report.py tests/test_viz.py --junitxml=C:\tmp\vfe3-estep-phi-focused-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 6: Run syntax, whitespace, and full-suite verification**

Run: `python -m compileall -q vfe3 tests`

Run: `git diff --check`

Run: `python -m pytest --junitxml=C:\tmp\vfe3-estep-phi-full-20260715.xml`

Expected: every command exits zero; the JUnit root reports zero failures and zero errors.

- [x] **Step 7: Commit the completed buildout**

```powershell
git add ablation.py docs/2026-07-15-edits.md docs/superpowers/plans/2026-07-15-estep-fixed-point-phi-numerics.md tests/test_estep_phi_ablation_routes_20260715.py
git commit -m "feat: complete E-step and phi numerics buildout"
```

- [x] **Step 8: Execute the authorized Git lifecycle**

Fetch `origin`, ensure the task branch contains the current `origin/main`, push the task branch, fast-forward `origin/main`, safely advance the user’s local `main` without overwriting WIP, verify local/remote SHA parity where possible, remove the task-owned worktree, and delete the local task branch.
