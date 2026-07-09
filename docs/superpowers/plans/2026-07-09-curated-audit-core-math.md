# Curated Audit Core Mathematics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the implemented free-energy functional, inference gradients, hierarchy, transports, SPD retractions, and numerical guards mathematically consistent across every supported route.

**Architecture:** The scalar functional is the source of truth. Closed-form kernels, autograd oracles, phi objectives, line search, Metropolis scoring, and diagnostics are tested against it before geometry and hierarchy repairs build on those contracts.

**Tech Stack:** Python, PyTorch autograd, float64 numerical oracles, pytest, finite differences, Gaussian/Laplace families, SPD eigendecompositions and Cholesky factorizations.

## Global Constraints

- Follow the master plan and approved design; keep the affine-invariant SPD route as the pure default.
- Scope: Findings 4, 6-10, 16-19, 23, 33-34, 36, 49-55, 106, M1-M4, M6-M7, and M11. Task 10 supplies the pullback-cache addendum needed by reporting's Finding 14 closure.
- Preserve default-off byte identity for optional routes and do not change experiment config values.
- Apply masks to derivatives where the scalar clamp applies; do not alter the scalar objective merely to match an existing kernel.
- Float64 is permitted inside numerical oracles/solves; cast only the final public result to the input dtype.
- Use tiny dimensions (`K < 6`) and deterministic seeds for gradient and property tests.
- No extra pytest `-q`; record exact JUnit counts at plan completion.

---

## File Structure

- Modify `vfe3/free_energy.py`, `vfe3/gradients/kernels.py`, `vfe3/gradients/oracle.py`: one objective and gradient convention.
- Modify `vfe3/inference/e_step.py`, `vfe3/model/model.py`, `vfe3/model/stack.py`: truncation, reflection/RoPE phi objective, model channel, and actual prior capture.
- Modify `vfe3/geometry/retraction.py`, `vfe3/geometry/transport.py`, `vfe3/geometry/phi_preconditioner.py`, `vfe3/geometry/lie_ops.py`: SPD, caps, pullback, and BCH behavior.
- Modify `vfe3/families/laplace.py`, `vfe3/model/prior_bank.py`, `vfe3/numerics.py`, `vfe3/config.py`: stable divergences, exact decode, trust regions, and guards.
- Create `tests/test_curated_objective_math_20260709.py`, `tests/test_curated_inference_math_20260709.py`, `tests/test_curated_geometry_math_20260709.py`, and `tests/test_curated_math_contracts_20260709.py`.

### Task 1: One two-hop functional across scalar, kernel, oracle, and diagnostics

**Files:** Modify `vfe3/free_energy.py`, `vfe3/gradients/kernels.py`, `vfe3/gradients/oracle.py`, `vfe3/metrics.py`, `vfe3/model/model.py`; create `tests/test_curated_objective_math_20260709.py`.

**Interfaces:** `belief_gradients_autograd(..., lambda_twohop: float = 0.0, need_sigma_grad: bool = True)`; `free_energy_terms(..., lambda_twohop=0.0, coupling_energy=None, log_likelihood=None)`.

- [ ] **Step 1: Write failing oracle tests.** Construct a three-token diagonal Gaussian case with one zero-energy and one `>=kl_max` pair. Compare the scalar autograd derivative to the kernel and oracle:

```python
w2 = beta.detach() @ beta.detach()
scalar = base_F + lambda_twohop * (w2 * energy).sum()
g_ref, = torch.autograd.grad(scalar, mu_q)
assert torch.allclose(g_kernel, g_ref, atol=2e-5, rtol=2e-5)
assert torch.allclose(g_oracle, g_ref, atol=2e-5, rtol=2e-5)
```

Add `test_twohop_kernel_matches_scalar_autograd_with_zero_and_saturated_pairs` and `test_oracle_twohop_gradient_matches_scalar_free_energy`.
- [ ] **Step 2: Run** both tests; expect mismatches/unsupported oracle argument.
- [ ] **Step 3: Implement.** Use raw detached `beta @ beta` for hop weights. Apply `pair_mask` to the destination energy derivative, not to the intermediate attention factors. Thread `lambda_twohop` into the oracle scalar. Extend `free_energy_terms` and both diagnostic callers with the same coupling energy, two-hop term, and likelihood term.
- [ ] **Step 4: Add** `test_free_energy_terms_matches_scalar_with_twohop_and_value_gauge` and `test_diagnostics_threads_all_active_objective_terms`; run the four new tests plus `tests/test_fix_metrics_audit.py`, `tests/test_tier12_estep.py`, and `tests/test_tier12_attention.py`.
- [ ] **Step 5: Commit** `fix(objective): unify two-hop scalar and gradients`.

### Task 2: Decoupled-RoPE and reflected phi objective

**Files:** Modify `vfe3/inference/e_step.py`; modify `tests/test_phi_reflection.py`; extend `tests/test_curated_objective_math_20260709.py`.

**Interfaces:** `phi_alignment_loss(..., *, reflection: Optional[torch.Tensor] = None, ...)` builds score energy for attention and value energy for coupling when `RopeTransport.on_value is False`.

- [ ] **Step 1: Add failing tests** `test_phi_gradient_matches_scalar_f_under_decoupled_rope`, `test_phi_gradient_matches_scalar_f_with_mixed_reflection`, and `test_nonflat_oracle_transport_threads_reflection`.
- [ ] **Step 2: Run** the three tests; expect the phi gradient to match the all-positive/score-gauge objective instead of the scalar functional.
- [ ] **Step 3: Implement.** Thread `reflection` into `phi_alignment_loss`, its live call, and the non-flat `_omega_builder`. For decoupled RoPE compute:

```python
beta = attention_weights(score_energy, tau=tau, log_prior=log_prior)
coupling = (beta * value_energy).sum()
entropy = (tau * beta * (beta.clamp_min(log_eps).log() - log_pi)).sum()
loss = lambda_beta * (coupling + entropy) + mass
```

Do not detach `beta` in the phi objective. Use `omega.base` for value transport only when `on_value=False`.
- [ ] **Step 4: Run** the new tests plus `tests/test_phi_reflection.py tests/test_rope.py tests/test_e_step.py`; expect PASS.
- [ ] **Step 5: Commit** `fix(phi): match reflected and decoupled scalar objective`.

### Task 3: Correct last-k truncated oracle backpropagation

**Files:** Modify `vfe3/inference/e_step.py`, `vfe3/gradients/oracle.py`; create `tests/test_curated_inference_math_20260709.py`.

**Interfaces:** Boundary beliefs are fresh leaves; caller-supplied and internally hoisted transports are rebuilt from boundary phi.

- [ ] **Step 1: Add failing tests** `test_shared_prebuilt_transport_respects_truncation_boundary`, `test_oracle_last_k_restores_prior_gradient`, and `test_backprop_last_equal_total_matches_full_unroll`.
- [ ] **Step 2: Run** them; expect a missing prior gradient and a transport-gradient leak.
- [ ] **Step 3: Implement boundary leaves:**

```python
belief = belief._replace(
    mu=belief.mu.detach().requires_grad_(True),
    sigma=belief.sigma.detach().requires_grad_(True),
    phi=belief.phi.detach().requires_grad_(True),
)
```

Rebuild the flat transport unconditionally at the boundary when a hoisted/prebuilt transport exists. The oracle's `create_graph=True` route must use these live leaves and return attached gradients.
- [ ] **Step 4: Run** the new tests and existing truncation tests in `tests/test_e_step.py`; expect PASS.
- [ ] **Step 5: Commit** `fix(estep): preserve oracle gradients across truncation`.

### Task 4: Refined-s consistency and global model-channel controls

**Files:** Modify `vfe3/model/model.py`, `vfe3/inference/e_step.py`; extend `tests/test_curated_inference_math_20260709.py`.

**Interfaces:** `_fold_gamma_prior(..., s_belief: Optional[tuple[Tensor, Tensor]] = None)`; `_refine_s(..., rope: Optional[Tensor] = None, prebuilt_transport=None)` forwards all global E-step controls.

- [ ] **Step 1: Add failing tests** `test_gamma_attention_maps_uses_refined_s_belief`, `test_gamma_fold_uses_refined_s_belief`, `test_lambda_h_zero_state_dependent_s_refine_has_no_hyperprior_force`, `test_refine_s_forwards_global_estep_controls`, and `test_refine_s_rope_matches_direct_rotated_estep`.
- [ ] **Step 2: Run** these tests; expect raw-s gamma energy, hidden zero-gate force, and missing controls.
- [ ] **Step 3: Implement.** Capture `(s_mu1, s_sigma1)` and pass it into all gamma consumers. Before coefficient dispatch, set the hyperprior coefficient to zero when `cfg.lambda_h == 0.0`. Forward `e_step_update`, `mm_damping`, randomized min/max, truncation, halt tolerance, and RoPE into the s-channel call. Each channel draws randomized depth independently from the seeded generator and evaluates its own halt condition.
- [ ] **Step 4: Run** the focused tests plus `tests/test_model_channel_diagnostics_2026_06_13.py`; expect PASS.
- [ ] **Step 5: Commit** `fix(hierarchy): align refined model-channel inference`.

### Task 5: Capture the actual multilayer prior

**Files:** Modify `vfe3/model/stack.py`, `vfe3/model/model.py`; extend `tests/test_curated_inference_math_20260709.py`.

**Interfaces:** `vfe_stack(..., capture: Optional[dict] = None)` records `capture["final_block_prior"] = (mu_p, sigma_p)` immediately before the final block call.

- [ ] **Step 1: Add failing tests** `test_multilayer_capture_records_actual_final_block_prior`, `test_multilayer_mstep_self_coupling_matches_manual_recurrence`, and `test_single_layer_mstep_self_coupling_is_unchanged`.
- [ ] **Step 2: Run** them; expect the repeated-final-output approximation.
- [ ] **Step 3: Implement.** In the stack loop, write a cloned/reference-safe tuple before the final block. Remove the reconstruction loop in the M-step and use the captured tuple directly. Preserve the one-layer path.
- [ ] **Step 4: Run** the new tests plus existing M-step self-coupling tests; expect PASS.
- [ ] **Step 5: Commit** `fix(mstep): score against the actual final-block prior`.

### Task 6: Truthful fixed-surrogate contracts and frozen-sigma compute

**Files:** Modify `vfe3/config.py`, `vfe3/run_artifacts.py`, `vfe3/inference/e_step.py`, `vfe3/gradients/kernels.py`, `vfe3/gradients/oracle.py`; create `tests/test_curated_math_contracts_20260709.py`.

**Interfaces:** Pure-path report includes `fixed_covariance_surrogate`, `detached_precision_prior`, `detached_query_adaptive_tau`, and `state_dependent_alpha_majorizer`. Oracle/kernel accept `need_sigma_grad`.

- [ ] **Step 1: Add failing tests** for skip-sigma plus precision warning, surrogate report flags, MM-majorizer wording, and three spies proving sigma gradient/fusion/preconditioner work is skipped.
- [ ] **Step 2: Run** the new file; expect absent flags and unnecessary sigma calls.
- [ ] **Step 3: Implement.** Keep the intended detached gradients unchanged. Add warnings/report fields. Thread `need_sigma_grad=False` through the oracle; permit `mm_exact_update` to omit sigma fusion; bypass the sigma natural-gradient arm when its gradient is absent.
- [ ] **Step 4: Run** the new file plus precision, adaptive-tau, MM, and skip-sigma tests; expect PASS.
- [ ] **Step 5: Commit** `fix(contracts): expose surrogates and skip frozen sigma work`.

### Task 7: Family-aware halting, exact full-chunked decode, and Mahalanobis trust

**Files:** Modify `vfe3/inference/e_step.py`, `vfe3/model/prior_bank.py`, `vfe3/numerics.py`; create `tests/test_curated_geometry_math_20260709.py`.

**Interfaces:** Halting dispatches through `get_family`; full-chunked decode uses raw SPD values with jitter only after failure; full-covariance trust uses Cholesky whitening.

- [ ] **Step 1: Add failing tests** `test_halt_tol_full_covariance_uses_full_gaussian_kl`, `test_full_chunked_matches_dense_at_variance_floor_without_double_ridge`, `test_full_cov_query_invariants_use_raw_spd_on_round_zero`, `test_full_cov_box_binds_in_mahalanobis_units`, `test_full_cov_ball_bounds_mahalanobis_norm`, and `test_full_cov_failed_cholesky_falls_back_per_element`.
- [ ] **Step 2: Run** them; expect crash/value drift/diagonal approximation.
- [ ] **Step 3: Implement.** Dispatch the movement KL through the configured family. Remove unconditional `+eps` terms from chunked full-covariance algebra. For trust, solve `L w = delta_mu`, clamp/scale `w`, map back with `L`, and use the marginal path only where `safe_cholesky` reports failure.
- [ ] **Step 4: Run** the new tests plus `tests/test_fullcov_alpha_roadmap_2026_06_13.py tests/test_mu_trust_region.py`; expect PASS.
- [ ] **Step 5: Commit** `fix(fullcov): use exact KL geometry in inference and decode`.

### Task 8: Per-matrix damping and valid Log-Euclidean retraction

**Files:** Modify `vfe3/geometry/retraction.py`; extend `tests/test_curated_geometry_math_20260709.py`.

**Interfaces:** `_rel_gap_eps(A, ...) -> Tensor` has shape `(..., 1, 1)`; `_frechet_log_spd(sigma, tangent, *, eps) -> Tensor` supplies the chart tangent.

- [ ] **Step 1: Add failing tests** `test_rel_gap_eps_is_per_matrix`, `test_eigh_damped_gradient_is_batch_separable_across_scales`, `test_log_euclidean_full_retraction_has_identity_first_derivative`, and `test_log_euclidean_scalar_uses_h_over_sigma_chart_tangent`.
- [ ] **Step 2: Run** them; expect batch coupling and wrong derivative.
- [ ] **Step 3: Implement.** Compute `scale = A.detach().abs().amax(dim=(-2, -1), keepdim=True)`. In an eigenbasis, apply matrix-log divided differences to the ambient tangent, rotate back, and add that chart tangent to `log(Sigma)` before exponentiation.
- [ ] **Step 4: Run** new and existing retraction tests; expect PASS.
- [ ] **Step 5: Commit** `fix(spd): restore batch separation and Log-Euclidean tangent`.

### Task 9: Overflow-safe transport and Laplace routes

**Files:** Modify `vfe3/geometry/transport.py`, `vfe3/families/laplace.py`; extend `tests/test_curated_geometry_math_20260709.py`.

**Interfaces:** Soft-cap norms are finite before squaring; large skew exponentials auto-upcast; Laplace alpha-greater-than-one algebra stays in log space.

- [ ] **Step 1: Add failing tests** for all three Regime-II cap sites, large skew auto-upcast, small skew fp32 identity, finite Laplace large-separation gradients, quadrature agreement, and divergent-blend clamp policy.
- [ ] **Step 2: Run** them; expect inf/NaN or precision mismatch.
- [ ] **Step 3: Implement.** Compute cap norm in float64 or with `amax` scaling. For skew routes, force the float64 island above the existing norm threshold even in dimension mode. Reexpress the convergent Laplace integral with `torch.logaddexp` and mask divergent blends before dangerous exponentials.
- [ ] **Step 4: Run** new tests plus Regime-II, transport, and Laplace suites; expect PASS.
- [ ] **Step 5: Commit** `fix(numerics): stabilize transport caps and Laplace Renyi`.

### Task 10: Pullback precision, bounded closure cache, and BCH fail-closed validation

**Files:** Modify `vfe3/geometry/phi_preconditioner.py`, `vfe3/geometry/lie_ops.py`, `vfe3/config.py`; extend `tests/test_curated_geometry_math_20260709.py`.

**Interfaces:** Pullback metrics remain float64 through solve; closure cache is bounded and keyed by stable CPU value signature rather than tensor identity; nonclosed BCH configs reject at construction.

- [ ] **Step 1: Add failing tests** `test_pullback_preconditioner_matches_float64_ill_conditioned_reference`, `test_pullback_per_block_solve_stays_float64_until_final_cast`, `test_pullback_closure_cache_is_bounded_and_releases_cuda_basis`, and the three BCH config tests.
- [ ] **Step 2: Run** them; expect precision loss, cache growth, and accepted invalid config.
- [ ] **Step 3: Implement.** Return float64 metrics and cast only the solved natural gradient. Replace identity-keyed closure ownership with a bounded LRU keyed by shape/dtype and a CPU hash of basis values; cached entries must not hold caller CUDA tensors. Move the convergence scalar conversion after the series loop. Reject 3+-head cross-coupled explicit nonclosure when either BCH route is active.
- [ ] **Step 4: Run** new tests plus `tests/test_phi_preconditioner.py` and cross-coupling tests; expect PASS.
- [ ] **Step 5: Commit** `fix(pullback): retain precision and bound closure diagnostics`.

### Task 11: Bounded trainable variance exponentiation

**Files:** Modify `vfe3/numerics.py`, `vfe3/model/prior_bank.py`, `vfe3/model/model.py`; extend `tests/test_curated_geometry_math_20260709.py`.

**Interfaces:** `bounded_variance_from_log(log_sigma, *, eps=1e-6, max_log=80.0) -> Tensor` is used by all trainable prior/model/decode variance reads and does not reuse `sigma_max`.

- [ ] **Step 1: Add failing tests** `test_bounded_log_variance_exp_is_finite_with_finite_gradient`, `test_bounded_log_variance_is_identity_in_normal_range`, and `test_prior_model_and_decode_variance_reads_share_guard`.
- [ ] **Step 2: Run** them; expect overflow or direct `torch.exp` reads.
- [ ] **Step 3: Implement:**

```python
def bounded_variance_from_log(log_sigma: torch.Tensor, *, eps: float = 1e-6,
                              max_log: float = 80.0) -> torch.Tensor:
    return torch.exp(log_sigma.clamp(max=max_log)).clamp(min=eps)
```

Replace trainable-table exponentiation sites only. Emit the existing numerical warning when any detached value exceeds `max_log`; do not clamp belief-state retractions through this helper.
- [ ] **Step 4: Run** new tests plus prior-bank and hyperprior tests; expect PASS.
- [ ] **Step 5: Commit** `fix(variance): guard trainable log-scale exponentiation`.

## Core Plan Verification

- [ ] Run `python -m pytest tests/test_curated_objective_math_20260709.py tests/test_curated_inference_math_20260709.py tests/test_curated_geometry_math_20260709.py tests/test_curated_math_contracts_20260709.py tests/test_e_step.py tests/test_retraction.py tests/test_phi_reflection.py tests/test_model_channel_diagnostics_2026_06_13.py --junitxml=C:\tmp\vfe3-curated-core-math.xml`.
- [ ] Read JUnit attributes; update every assigned ledger row with its exact test, command, and commit.
- [ ] Run `git diff --check` and confirm no config value changed.
