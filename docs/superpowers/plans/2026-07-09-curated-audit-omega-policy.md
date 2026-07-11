# Curated Audit Omega and Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make omega-direct transport, cached inference, registry capabilities, active-inference policy scoring, and autoregressive generation coherent, fail-closed, and testable.

**Architecture:** Group-element inversion and capability metadata become shared seams used by live and cached paths. Unsupported chart combinations reject at construction, while policy scoring uses one context, one prior, stable probability products, and explicit preference semantics.

**Tech Stack:** Python, PyTorch, pytest, group matrices, cached inference, tiktoken, registered kernels/decoders/transports.

## Global Constraints

- Follow the master plan and design; preserve the phi/off pure path and all current config values.
- Scope: Findings 11, 20-21, 35, 47, 58-59, 61, 63-74, 82-84, 90-91, 94-100, M9, L1-L3, L5, and L7.
- Findings 62 and 92 are verification-only because they were fixed before this branch.
- Fail closed instead of inventing omega momentum, symplectic projection, or a new reflection proposal distribution.
- Old Findings 97 and 99 are in scope. The broader P1-P6 performance program is not.
- Cache support is an exactness claim; unsupported reflection routes return false or raise before cached scoring.
- No extra pytest `-q`; use JUnit for plan-level counts.

---

## File Structure

- Modify `vfe3/config.py`, `vfe3/geometry/groups.py`: capabilities and validation.
- Modify `vfe3/geometry/transport.py`, `vfe3/geometry/lie_ops.py`, `vfe3/gauge_optim.py`: inverse, compact blocks, diagnostics, and dirty rows.
- Modify `vfe3/inference/belief_cache.py`, `vfe3/inference/e_step.py`: exact cached/live transport contracts.
- Modify `vfe3/inference/policy.py`, `vfe3/inference/ring_task.py`, `vfe3/model/model.py`: policy and generation behavior.
- Modify `vfe3/model/prior_bank.py`, `vfe3/gradients/kernels.py`: decoder/kernel registry records.
- Modify `vfe3/viz/extract.py`, `ablation.py`: replay and structurally valid sweep arms.
- Extend the existing omega, cache, policy, registry, RoPE, and visualization test modules named below.

### Task 1: Policy validation, single prior, and stable probability products

**Files:** Modify `vfe3/config.py`, `vfe3/inference/policy.py`, `vfe3/inference/ring_task.py`; modify `tests/test_policy_registry.py`, `tests/test_efe_scorer.py`, `tests/test_ring_task.py`.

**Interfaces:** `logprob_control` accepts only horizon 1 and default EFE-term metadata; `policy_precision` is finite positive; hard support is represented explicitly rather than silently floored.

- [ ] **Step 1: Add failing tests** `test_logprob_control_rejects_ignored_config_fields`, NaN/Inf precision cases, `test_logprob_control_does_not_double_count_base_prior`, `test_efe_terms_handle_zero_probability_without_nan`, `test_policy_posterior_rejects_all_infinite_candidate_row`, and `test_decode_menu_rejects_invalid_sampling_inputs`.
- [ ] **Step 2: Run** the three focused test files; expect validation and NaN failures.
- [ ] **Step 3: Implement.** Reject non-unit horizon or nondefault score-term selections for `logprob_control`. Call `_policy_posterior(score, gamma, None)` in that scorer so the base prior is used once. Replace `q * q_log` with:

```python
q = q_log.exp()
q_log_term = torch.where(q > 0, q * q_log, torch.zeros_like(q))
```

Reject posterior rows with no finite candidate. Add explicit `support_floor: Optional[float] = None` to task preferences: `None` retains hard `-inf` support; a supplied finite value is normalized and reported as a finite-floor preference. Mirror generation's temperature/top-p/finiteness validation in `_decode_menu`.
- [ ] **Step 4: Run** `python -m pytest tests/test_policy_registry.py tests/test_efe_scorer.py tests/test_ring_task.py`; expect PASS.
- [ ] **Step 5: Commit** `fix(policy): validate controls and stabilize EFE scoring`.

### Task 2: One context contract and safe last-position generation

**Files:** Modify `vfe3/inference/policy.py`, `vfe3/model/model.py`; modify `tests/test_belief_cache.py`, `tests/test_generate.py`, `tests/test_efe_scorer.py`.

**Interfaces:** Overlength `efe_rollout` and policy menus fail before scoring; `forward_beliefs(..., decode_last: bool = False)` decodes `(B,1,V)` when requested.

- [ ] **Step 1: Add failing tests** `test_efe_rollout_rejects_context_plus_horizon_over_limit`, `test_policy_rejects_context_plus_candidate_over_limit`, `test_generate_decodes_only_last_position`, `test_generate_rejects_nonfinite_logit_rows`, and `test_policy_menu_rejects_nonfinite_base_logits`.
- [ ] **Step 2: Run** them; expect silent truncation/full decode/invalid sampling.
- [ ] **Step 3: Implement.** Check `context.shape[1] + candidate_length <= max_seq_len` before the menu prior or rollout; raise an accurate `ValueError` otherwise. Add `decode_last` to `forward_beliefs`; generation calls it with `return_logits=True, decode_last=True`. Before top-k or softmax, require at least one finite value and all retained values finite per row.
- [ ] **Step 4: Run** the focused tests plus `tests/test_generate.py`; expect PASS.
- [ ] **Step 5: Commit** `fix(generate): align policy context and decode only the last token`.

### Task 3: Shared group-element inverse and reflection-safe cache admission

**Files:** Modify `vfe3/geometry/transport.py`, `vfe3/inference/belief_cache.py`; modify `tests/test_belief_cache.py`, `tests/test_omega_direct.py`.

**Interfaces:** `group_element_inverse(omega: Tensor, group: GaugeGroup, *, residual_tol: float = 1e-4) -> Tensor` is used by live and cached paths. `cache_supported` rejects active phi reflection.

- [ ] **Step 1: Add failing tests** `test_cache_supported_rejects_phi_reflection`, `test_efe_rollout_reflection_fails_closed`, `test_cached_matches_full_with_norm_keyed_matrix_exp`, `test_omega_direct_cache_matches_full_after_skew_frame_drift`, and `test_cached_omega_inverts_shared_prefix_once`.
- [ ] **Step 2: Run** them; expect false cache admission, precision drift, inverse mismatch, and repeated inversions.
- [ ] **Step 3: Implement.** For skew groups compute `residual = ||U^T U-I||_F`; use transpose only for rows within tolerance and a float64 true inverse for drifted rows. Non-skew rows use float64 inverse. Call this helper from `build_transport_from_element` and cache. Forward `exp_fp64_mode` and threshold in phi cache operators. In rollout, invert the context once, expand it across candidates, and invert only appended rows.
- [ ] **Step 4: Run** cache and omega-direct tests; expect PASS.
- [ ] **Step 5: Commit** `fix(cache): share exact group inverses and reject reflection`.

### Task 4: Omega configuration and constructible ablation arms

**Files:** Modify `vfe3/config.py`, `ablation.py`, `vfe3/inference/e_step.py`, `vfe3/train.py`; modify `tests/test_omega_direct.py`, `tests/test_omega_metropolis.py`, `tests/test_config.py`.

**Interfaces:** Omega-direct rejects active positional phi, additive encoding, unsupported off/frozen semantics, and phi E-step updates. Single-block sweep arms use one head.

- [ ] **Step 1: Add failing tests** `test_ablation_omega_direct_arm_builds` that instantiates `VFEModel`, `test_multilayer_post_estep_transforms_preserve_omega_direct_frame`, `test_omega_direct_rejects_active_positional_phi`, `test_gauge_parameterization_sweep_disables_positional_phi_for_all_cells`, off/frozen reflection cases, additive encoder rejection, lower-level phi-update rejection, and knob type/range cases.
- [ ] **Step 2: Run** the focused files; expect construction and validation failures.
- [ ] **Step 3: Implement.** Set `n_heads=1` for `glk`, `so_k`, and `sp` omega arms; set sweep `requires["pos_phi"] = "none"`. Reject `omega_direct + pos_phi != "none"`, additive encoding, and `e_phi_lr>0` in both config and lower E-step API. Under `gauge_transport="off"`, require reflection off; reject unsupported omega-direct frozen/Metropolis semantics. Validate `type(omega_compact_storage) is bool` and nonnegative integer `omega_reorth_every`. Warn that gauge momentum/update rule do not apply to omega retraction SGD.
- [ ] **Step 4: Run** config, omega-direct, Metropolis, and two-layer preservation tests; expect PASS.
- [ ] **Step 5: Commit** `fix(omega): fail closed and repair ablation construction`.

### Task 5: Omega numerical health, compact transport, and dirty-row reorthogonalization

**Files:** Modify `vfe3/model/prior_bank.py`, `vfe3/geometry/transport.py`, `vfe3/geometry/lie_ops.py`, `vfe3/gauge_optim.py`, `vfe3/train.py`; modify `tests/test_omega_direct.py`, `tests/test_gauge_optim.py`.

**Interfaces:** `CompactBlockElement(blocks: Tensor, K: int, tied: bool = False)` carries `(...,H,d,d)` blocks. `CompactFactoredTransport(exp_blocks: Tensor, inv_blocks: Tensor, K: int, mean_per_head: bool = False)` is consumed by the transport contractions without materializing `(...,K,K)`. Optimizer state stores `omega_dirty: BoolTensor[V]`.

- [ ] **Step 1: Add failing tests** `test_skew_element_transport_uses_true_inverse_after_orthogonality_drift`, `test_sp_omega_membership_diagnostic_detects_drift`, `test_compact_transport_inverts_blocks_without_dense_K_matrix`, `test_full_and_compact_omega_transport_match`, `test_omega_reorthogonalizes_only_dirty_rows`, and `test_dirty_rows_clear_after_cadence`.
- [ ] **Step 2: Run** them; expect transpose drift, dense reconstruction, and full-table SVD.
- [ ] **Step 3: Implement compact blocks.** Return `CompactBlockElement` from `_omega_lookup` when storage is compact. Invert each `d x d` block into `CompactFactoredTransport` and make `transport_mean`/diagonal and full covariance transport contract over head blocks. Prohibit `_from_equal_diag_blocks` on this route; dense conversion remains an explicit compatibility method only.
- [ ] **Step 4: Implement diagnostics and dirty rows.** Record nonfinite/singular omega as an immediate error. Log condition or symplectic residual on diagnostic cadence. Label block-GL reflection as a block-0 probe; do not change proposals. Mark active updated vocabulary rows in `state[p]["omega_dirty"]`; project only those rows on cadence and clear their bits. Persist the mask through optimizer state.
- [ ] **Step 5: Run** omega-direct and optimizer suites; expect PASS.
- [ ] **Step 6: Commit** `fix(omega): preserve compact blocks and project dirty rows`.

### Task 6: Omega-aware visualization replay

**Files:** Modify `vfe3/viz/extract.py`; modify `tests/test_viz.py`.

**Interfaces:** `_iter_kwargs` and every replay pass `gauge_parameterization`, `omega`, and reflection as applicable.

- [ ] **Step 1: Add failing tests** `test_converged_state_omega_direct_uses_stored_frame`, `test_omega_direct_iterative_extractors_match_forward_transport`, and `test_phi_extractors_remain_unchanged`.
- [ ] **Step 2: Run** them; expect replay through the phi chart.
- [ ] **Step 3: Implement.** Forward `cfg.gauge_parameterization` through every `vfe_stack`/`e_step_iteration`; pass `out.omega` to each `_transport` rebuild. Preserve the phi defaults when omega is absent.
- [ ] **Step 4: Run** the three tests and existing viz extractors; expect PASS.
- [ ] **Step 5: Commit** `fix(viz): replay the trained gauge parameterization`.

### Task 7: Direct-link trivial-gauge semantics and per-head omega mean

**Files:** Modify `vfe3/geometry/transport.py`, `vfe3/inference/e_step.py`; modify `tests/test_regime_ii_link.py`, `tests/test_omega_direct.py`.

**Interfaces:** Trivial vertex gauge retains the direct edge link. `build_transport_from_element(..., mean_per_head: bool = False)` stores the flag on its factored result.

- [ ] **Step 1: Add failing tests** `test_charted_trivial_gauge_preserves_edge_link` and `test_element_transport_omega_direct_wires_mean_per_head` with dense allclose.
- [ ] **Step 2: Run** them; expect identity link/inert flag.
- [ ] **Step 3: Implement.** Remove `gauge_mode == "trivial"` from the charted edge-link early return. Add/forward `mean_per_head` at both omega-direct transport builders.
- [ ] **Step 4: Run** link and omega transport suites; expect PASS.
- [ ] **Step 5: Commit** `fix(transport): preserve links and per-head omega contraction`.

### Task 8: Decoder, kernel, group, and RoPE registry coherence

**Files:** Modify `vfe3/model/prior_bank.py`, `vfe3/model/model.py`, `vfe3/gradients/kernels.py`, `vfe3/geometry/groups.py`, `vfe3/config.py`; modify registry and RoPE tests.

**Interfaces:** `DecodeRegistration(callable, supports_full, supports_chunked, fused_ce)` is one record. `GaugeGroup` includes `omega_direct_capable: bool`. Kernel registration invalidates compiled cache. RoPE cache keys contain semantic fields.

- [ ] **Step 1: Add failing tests** for finite nonnegative constant alpha, stale decoder capabilities, custom chunked fused CE, compiled-kernel invalidation, invalid RoPE/ALiBi numerics, registered warning text, mutable rope-base cache, and group-registry omega capability.
- [ ] **Step 2: Run** `tests/test_config.py`, registry guard tests, kernel tests, and RoPE tests; expect failures.
- [ ] **Step 3: Implement.** Replace decoder side sets with one registration record and dispatch training CE through `fused_ce`. On `register_kernel`, `pop` the compiled entry. Validate `lambda_alpha`, `rope_base`, and ALiBi slope with `math.isfinite`; correct warning to a registered positional mode. Key RoPE cache on `(N, device, dtype, pos_rotation, rope_base)`. Move omega eligibility from a tuple into group registration metadata.
- [ ] **Step 4: Run** the focused registry/config suites; expect PASS.
- [ ] **Step 5: Commit** `fix(registry): make capabilities and caches coherent`.

## Omega/Policy Plan Verification

- [ ] Run `python -m pytest tests/test_policy_registry.py tests/test_efe_scorer.py tests/test_belief_cache.py tests/test_generate.py tests/test_ring_task.py tests/test_regime_ii_link.py tests/test_omega_direct.py tests/test_omega_metropolis.py tests/test_config.py tests/test_round3_registry_guards.py tests/test_rope.py tests/test_viz.py --junitxml=C:\tmp\vfe3-curated-omega-policy.xml`.
- [ ] Read JUnit attributes and update every assigned ledger row.
- [ ] Run `git diff --check`; confirm P1-P6 and hypothesis code are absent.
