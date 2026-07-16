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

The inner E-step and oracle derivative construction now execute in autocast-disabled fp32 islands only when AMP is configured, while the outer forward/loss autocast remains active and the default non-AMP path retains its prior dtype behavior. Diagnostics now expose `gamma_direct_frame_grad_active = 0.0`, accurately recording the intentionally passive direct gamma-frame route.

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
