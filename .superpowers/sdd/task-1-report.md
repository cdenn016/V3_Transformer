# Task 1 Report: Objective Fidelity and Accepted-Step State Transitions

## Scope

This task repairs first-panel findings B1, B2, B3, and B6 and second-panel findings S2-V1, S2-M1, S2-M2, and S2-M3. Work remained in the assigned worktree and branch. No Research-vault files, user configuration values, or controller-owned daily ledger were changed. The repository-wide default and slow suites were intentionally not run because the controller reserved those consolidated gates.

## RED evidence

The first focused command was:

```text
python -m pytest tests/test_objective_state_transitions_20260715.py tests/test_train.py::test_train_step_skips_on_nonfinite_grad_with_finite_loss --junitxml=C:\tmp\vfe3-task1-red-20260715.xml
```

It exited 1 with nine failures. The complete Metropolis score retained the belief-only negative decision; gamma energy was unchanged by RoPE; gamma-prior rows summed below one after support masking; the tiny-tail entropy derivative disagreed with the analytic coefficient; a rejected update advanced the scheduler; the inner oracle inherited the outer autocast dtype; `_maybe_metropolis_omega` had no `did_step` gate; diagnostics omitted the passive gamma-frame gradient state; and the existing rejected-gradient test still expected unconditional scheduler advancement.

After the initial repairs, the audit's successful-update persistence clause received its own RED command:

```text
python -m pytest tests/test_objective_state_transitions_20260715.py::test_successful_update_clock_persists_in_optimizer_state tests/test_objective_state_transitions_20260715.py::test_resumed_scheduler_uses_persisted_successful_update_clock
```

It exited 1 with two failures. The optimizer checkpoint carried no `successful_updates` field, and resume reconstructed LambdaLR with `last_epoch=4` from the outer cursor instead of `last_epoch=1` from two accepted updates.

## Repairs

The Metropolis scorer now composes the existing belief evaluator with one authoritative `_model_channel_free_energy` helper shared by the production forward loss. Active hyperprior, gamma coupling, gamma meta-entropy, tied-frame candidate dependence, configured two-hop terms, and captured RoPE controls are therefore scored without a second model-channel formula. Gamma transport now receives the same RoPE controls as the belief channel.

Gamma-to-beta prior mixing now masks forbidden beta support before row normalization. Split beta and gamma categorical entropy use a zero-safe `torch.special.xlogy` expression that leaves every positive tail probability unfloored; the exact zero branch uses a unit log operand to avoid an undefined derivative at hard zero.

`train_step` now derives one `did_step` decision and gates scheduler advancement, projection, barycenter, EMA, and downstream Metropolis behavior on it. The accepted-update count is persisted in optimizer parameter-group metadata, explicitly preserved by checkpoint loading, and used to reconstruct LambdaLR on resume. Legacy checkpoints fall back to the existing outer-step cursor. Metropolis cadence now consumes the zero-based accepted-update index and returns before state mutation or RNG use after a rejected update.

The outer E-step remains under configured autocast. Only the autograd oracle's inner objective and derivative construction re-enters with autocast disabled, casting its belief, prior, coefficient, log-prior, and representation-preserving transport inputs to fp32. Existing matrix-exp, SPD, decode, and cross-entropy islands remain narrow and unchanged. The default non-AMP path retains its prior dtype behavior. Diagnostics now expose `gamma_direct_frame_grad_active = 0.0`, accurately recording the intentionally passive direct gamma-frame route.

Neighbor tests were updated only where they encoded the superseded clamped-entropy derivative, compared grad-enabled inference with no-grad inference, or independently reconstructed the old belief-only Metropolis objective.

## GREEN evidence

Machine-readable JUnit results were:

| Gate | Tests | Passed | Failed | Errors | Skipped |
|---|---:|---:|---:|---:|---:|
| Focused final regressions plus existing rejected-gradient test | 11 | 11 | 0 | 0 | 0 |
| Targeted checkpoint compatibility | 3 | 3 | 0 | 0 | 0 |
| Existing training module | 41 | 41 | 0 | 0 | 0 |
| Free-energy, oracle, gamma, hierarchy, and RoPE neighbors | 138 | 136 | 0 | 0 | 2 |
| Metropolis, reflection, AMP, GradScaler, and training neighbors | 160 | 159 | 0 | 0 | 1 |

The final focused command was:

```text
python -m pytest tests/test_objective_state_transitions_20260715.py tests/test_train.py::test_train_step_skips_on_nonfinite_grad_with_finite_loss --junitxml=.superpowers/sdd/task-1-focused-final.xml
```

JUnit reported `tests=11`, `failures=0`, `errors=0`, and `skipped=0`. The targeted checkpoint gate reported `tests=3`, `failures=0`, `errors=0`, and `skipped=0`. The final `tests/test_train.py` gate reported `tests=41`, `failures=0`, `errors=0`, and `skipped=0`. The earlier neighboring JUnit gates reported `138/136 passed/2 skipped` and `160/159 passed/1 skipped`, both with zero failures and zero errors. `git diff --check` exited successfully.

## Files changed

- `vfe3/model/model.py`
- `vfe3/free_energy.py`
- `vfe3/gradients/oracle.py`
- `vfe3/train.py`
- `vfe3/run_artifacts.py`
- `tests/test_objective_state_transitions_20260715.py`
- `tests/test_train.py`
- `tests/test_phi_reflection_objective_parity_20260712.py`
- `tests/test_hierarchical_probabilistic_completeness_20260712.py`

## Review follow-up: narrow AMP boundary

The first implementation disabled autocast around transport construction, `_refine_s`, and the complete `vfe_stack`. A strengthened regression separately observes the outer `vfe3.model.block.e_step` entry and the inner `vfe3.gradients.oracle.pairwise_energy` call. Before the review fix, the exact command was:

```text
python -m pytest tests/test_objective_state_transitions_20260715.py::test_outer_estep_autocast_and_inner_oracle_fp32_boundary
```

The literal captured RED output included:

```text
F                                                                        [100%]
__________ test_outer_estep_autocast_and_inner_oracle_fp32_boundary ___________
>       assert outer_autocast and all(outer_autocast)
E       assert ([False] and False)
E        +  where False = all([False])
FAILED tests/test_objective_state_transitions_20260715.py::test_outer_estep_autocast_and_inner_oracle_fp32_boundary
1 failed, 1 warning in 0.74s
```

Removing the overbroad wrappers exposed the transport side of the same inner boundary rather than a reason to restore the wrappers. The next literal failing line was:

```text
>           return _VF.einsum(equation, operands)  # type: ignore[attr-defined]
E           RuntimeError: expected scalar type BFloat16 but found Float
FAILED tests/test_objective_state_transitions_20260715.py::test_outer_estep_autocast_and_inner_oracle_fp32_boundary
1 failed, 1 warning in 0.22s
```

The repair removes the three whole-E-step `_amp_off_context` wrappers and their eager casts. `_transport_to_float` now reconstructs dense, factored, compact-factored, direct-link, and RoPE transport representations with fp32 tensors only inside the oracle recursion. The strengthened test requires outer autocast to be enabled and inner pairwise objective construction to have autocast disabled with fp32 belief and prior tensors.

## Reproduced original RED evidence

The original raw XML had already been removed under the task-artifact cleanup contract. To avoid reconstructing output from memory, a detached temporary worktree was created at base commit `541fff6`, and the current regression file was copied into it. Source and reproduction copies had the identical SHA-256 `5C1D7B48803DB30D4ED8DD54907E8E60F43AA02F0AC28603BF6702CD96C2F229`. The task branch was not changed, and the temporary worktree was removed after capture.

The exact reproduction command was:

```text
python -m pytest tests/test_objective_state_transitions_20260715.py
```

The literal captured base output was:

```text
FFFFFFFFFF                                                               [100%]
E       assert -0.0005972981452941895 > 0.0
E       assert not True
E       AssertionError: Tensor-likes are not close!
E       Greatest absolute difference: 0.3333333134651184 at index (0, 0) (up to 1e-05 allowed)
E       AssertionError: Tensor-likes are not close!
E       Greatest absolute difference: 5.524901517761266e-14 at index (0, 1) (up to 1e-20 allowed)
E       assert 1 == 0
E       KeyError: 'successful_updates'
E       assert [4] == [1]
E       assert False
E       assert 'did_step' in mappingproxy(OrderedDict({'model': <Parameter "model: vfe3.model.model.VFEModel">, 'token_ids': <Parameter "token_ids: torch.Tensor">, 'step': <Parameter "step: int">, 'generator': <Parameter "generator: torch._C.Generator">}))
E       KeyError: 'gamma_direct_frame_grad_active'
FAILED tests/test_objective_state_transitions_20260715.py::test_metropolis_uses_complete_joint_objective_when_gamma_reverses_belief_decision
FAILED tests/test_objective_state_transitions_20260715.py::test_gamma_energy_changes_when_rope_transport_is_active
FAILED tests/test_objective_state_transitions_20260715.py::test_gamma_prior_mix_normalizes_after_applying_active_support
FAILED tests/test_objective_state_transitions_20260715.py::test_entropy_tail_derivative_matches_analytic_envelope_coefficient
FAILED tests/test_objective_state_transitions_20260715.py::test_scheduler_does_not_advance_after_rejected_update
FAILED tests/test_objective_state_transitions_20260715.py::test_successful_update_clock_persists_in_optimizer_state
FAILED tests/test_objective_state_transitions_20260715.py::test_resumed_scheduler_uses_persisted_successful_update_clock
FAILED tests/test_objective_state_transitions_20260715.py::test_outer_estep_autocast_and_inner_oracle_fp32_boundary
FAILED tests/test_objective_state_transitions_20260715.py::test_rejected_update_cannot_mutate_metropolis_state_or_rng
FAILED tests/test_objective_state_transitions_20260715.py::test_diagnostics_report_passive_direct_gamma_frame_gradient
10 failed, 2 warnings in 1.24s
```

All failure lines above are copied verbatim from the reproduction output.

## Review GREEN evidence

The focused Task 1 regressions and behavior-level AMP/autocast coverage were run together:

```text
python -m pytest tests/test_objective_state_transitions_20260715.py tests/test_amp.py tests/test_fp16_gradscaler.py tests/test_fix_model_audit.py tests/test_audit_fixes_2026_07_05.py tests/test_p1_compact_phi_block_transport_20260711.py --junitxml=C:\tmp\vfe3-task1-review-green-20260715.xml
```

The literal console summary was:

```text
........................................................................ [ 85%]
............                                                             [100%]
84 passed, 13 warnings in 1.16s
```

JUnit reported `tests=84`, `failures=0`, `errors=0`, `skipped=0`, and `time=1.158`.
