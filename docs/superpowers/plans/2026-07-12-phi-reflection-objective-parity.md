# Phi and Reflection Objective Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the phi coordinate substep and Metropolis reflection scorer evaluate the same active belief objective as the mean/covariance E-step when two-hop, precision-prior, or gamma-prior options are enabled.

**Architecture:** One model helper becomes the authoritative builder for the effective beta log-prior, including a fixed pre-stack precision fold and a candidate-dependent gamma fold. `phi_alignment_loss()` gains the existing detached `beta @ beta` two-hop block, while Metropolis preparation carries the exact pre-stack covariance plus enough model state to rebuild only the frame-dependent gamma contribution for current and trial frames and forwards `lambda_twohop` into `free_energy_value()`. Reflection STE remains a rejected reserved mode and is not implemented by this plan.

**Tech Stack:** Python 3, PyTorch autograd as the phi-gradient oracle, existing VFE3 attention/free-energy/transport helpers, pytest with JUnit XML.

## Global Constraints

`lambda_twohop=0.0` must preserve exact phi values and gradients. `omega_reflection="off"` and `phi_reflection="off"` must preserve the complete training path, RNG state, model state, and optimizer state. Two-hop weights remain detached on both beta factors and receive no independent entropy term. Precision-weighted attention remains detached from covariance exactly as deployed, and its fold is computed from the refined belief covariance entering `vfe_stack`, not from either converged or proposed Metropolis belief. Gamma-as-beta-prior remains detached from s-table gradients exactly as deployed, but its value must be recomputed from each trial frame when reflection changes that frame. The Metropolis current and trial scores must differ only by the proposed reflection after the shared fixed precision fold and all candidate-dependent folds are applied. No STE parameter, buffer, estimator, config value, or test is added; construction-time STE rejection remains unchanged. All focused tests run on CPU with `K < 6`; no full suite is run until the focused matrix is green. Every task updates `docs/2026-07-12-edits.md` and follows the isolated-worktree git lifecycle.

---

### Task 1: Centralize the effective beta-prior construction

**Files:**

- Create: `tests/test_phi_reflection_objective_parity_20260712.py`
- Modify: `vfe3/model/model.py:919-1003,2078-2161`
- Modify: `vfe3/contracts.py`
- Modify: `docs/2026-07-12-edits.md`

**Interfaces:**

- Produces: `EffectiveBetaPriorContext`, a typed record containing `token_ids`, `base_log_prior`, the fixed pre-stack `precision_sigma`, `model_phi`, and optional refined `s_belief`.
- Produces: `VFEModel._effective_beta_log_prior(belief, context) -> Optional[torch.Tensor]`.

```python
class EffectiveBetaPriorContext(NamedTuple):
    token_ids:      torch.Tensor
    base_log_prior: Optional[torch.Tensor]
    precision_sigma: torch.Tensor
    model_phi:      torch.Tensor
    s_mu:           Optional[torch.Tensor]
    s_sigma:        Optional[torch.Tensor]
```

Capture `precision_sigma=beliefs.sigma` after the optional s-refinement and immediately before `vfe_stack`, at the same seam used by the current inline precision fold. `_effective_beta_log_prior` always applies `_fold_precision_bias(context.base_log_prior, context.precision_sigma)` first; it never reads `belief.sigma` for that fold. If `gamma_as_beta_prior` is active, it supplies the candidate belief's omega/reflection only when `s_frame_mode="tied"`; an independent `phi_tilde` frame remains unchanged by a belief-frame reflection. The helper never mutates or caches a candidate-dependent tensor.

- [ ] **Step 1: Write red value/gradient tests.** Compare the helper to the current forward construction for no folds, precision only, gamma only, and both folds. Require exact equality and preserve the learnable T5 prior's graph while precision/gamma contributions stay detached according to their existing contracts.
- [ ] **Step 2: Write red fixed-precision and trial-frame tests.** Give the context and candidate deliberately different covariance tensors. Changing only `candidate.sigma` must leave the precision-only prior exactly equal, while changing only `context.precision_sigma` must change it. Under tied gamma, flip one phi or omega reflection in a cloned belief and require `_effective_beta_log_prior` to change; under `phi_tilde`, require it to remain fixed.
- [ ] **Step 3: Run the red tests.**

```powershell
$env:VFE3_TEST_DEVICE = "cpu"
python -m pytest tests/test_phi_reflection_objective_parity_20260712.py --junitxml=C:\tmp\vfe3-phi-reflect-task1-red.xml
```

Expected: the JUnit file records failures because the context and helper do not exist.

- [ ] **Step 4: Implement the typed context and helper.** Replace the inline forward fold sequence with one call and store the context in `DiagnosticSnapshot`/Metropolis preparation rather than reconstructing model state from token IDs later.
- [ ] **Step 5: Run the helper and existing attention-prior tests.**

```powershell
python -m pytest tests/test_phi_reflection_objective_parity_20260712.py tests/test_attention_prior.py tests/test_attention_prior_t5_windowed.py tests/test_gamma_coupling.py --junitxml=C:\tmp\vfe3-phi-reflect-task1-green.xml
```

Expected: `failures="0"` and `errors="0"`.

- [ ] **Step 6: Commit the prior boundary.**

```powershell
git add vfe3/model/model.py vfe3/contracts.py tests/test_phi_reflection_objective_parity_20260712.py docs/2026-07-12-edits.md
git commit -m "refactor: centralize effective beta priors"
```

### Task 2: Add two-hop coupling to the phi objective

**Files:**

- Modify: `vfe3/inference/e_step.py:505-604,607-970`
- Modify: `vfe3/model/block.py:45-68`
- Test: `tests/test_phi_reflection_objective_parity_20260712.py`
- Test: `tests/test_phi_retraction.py`
- Modify: `docs/2026-07-12-edits.md`

**Interfaces:**

- `phi_alignment_loss(...)` gains `lambda_twohop: float = 0.0` immediately after `lambda_beta` in the defined-float group.
- `e_step_iteration(...)` forwards the already configured `lambda_twohop` to `phi_alignment_loss`.

```python
value_energy = score_energy
has_decoupled_value = isinstance(omega, RopeTransport) and not omega.on_value
if has_decoupled_value:
    mu_tv = transport_mean(omega.base, mu)
    sigma_tv = transport_covariance(omega.base, sigma)
    value_energy = pairwise_energy(
        fam(mu, sigma), fam(mu_tv, sigma_tv),
        alpha=renyi_order,
        kl_max=kl_max,
        eps=eps,
        divergence_family=divergence_family,
        irrep_dims=group.irrep_dims,
    )
    base = free_energy(
        score_energy.new_zeros(score_energy.shape[:-1]),
        score_energy,
        score_energy.new_zeros(score_energy.shape[:-1]),
        tau=tau,
        lambda_beta=lambda_beta,
        include_attention_entropy=include_attention_entropy,
        log_prior=log_prior,
        coupling_energy=value_energy,
    )
elif include_attention_entropy:
    base = lambda_beta * reduced_free_energy(
        score_energy, tau=tau, log_prior=log_prior,
    ).sum()
else:
    beta = attention_weights(score_energy, tau=tau, log_prior=log_prior)
    base = lambda_beta * (beta * score_energy).sum()
if lambda_twohop != 0.0:
    beta = attention_weights(score_energy, tau=tau, log_prior=log_prior)
    hop = beta.detach() @ beta.detach()
    base = base + lambda_twohop * (hop * value_energy).sum()
return base + mass
```

The implementation must use the same score/value split as `free_energy()` when `rope_on_value=False`; it must not recompute beta from the value energy.

- [ ] **Step 1: Write the red phi-gradient oracle.** Build `K=3`, `N=3` flat diagonal beliefs with `lambda_twohop=0.2`; compare `autograd.grad(phi_alignment_loss, phi)` to autograd of `free_energy_value` with identical fixed hop weights. Repeat with decoupled RoPE value transport.
- [ ] **Step 2: Write the zero-weight identity test.** Call the old and extended forms at `lambda_twohop=0.0` and require exact scalar and gradient equality.
- [ ] **Step 3: Run the red tests.**

```powershell
python -m pytest tests/test_phi_reflection_objective_parity_20260712.py tests/test_phi_retraction.py --junitxml=C:\tmp\vfe3-phi-reflect-task2-red.xml
```

Expected: the nonzero two-hop oracle fails because `phi_alignment_loss` has no parameter or term.

- [ ] **Step 4: Implement and forward the term.** Reuse the already computed score energy, value energy, beta, and mass; add no new transport build and no host synchronization.
- [ ] **Step 5: Run phi, kernel, and oracle tests.**

```powershell
python -m pytest tests/test_phi_reflection_objective_parity_20260712.py tests/test_phi_retraction.py tests/test_gradients_oracle.py tests/test_gradients_kernels.py tests/test_curated_objective_math_20260709.py --junitxml=C:\tmp\vfe3-phi-reflect-task2-green.xml
```

Expected: `failures="0"` and `errors="0"`.

- [ ] **Step 6: Commit phi parity.**

```powershell
git add vfe3/inference/e_step.py vfe3/model/block.py tests/test_phi_reflection_objective_parity_20260712.py tests/test_phi_retraction.py docs/2026-07-12-edits.md
git commit -m "fix: include two-hop energy in phi updates"
```

### Task 3: Make Metropolis score the exact active fixed-belief objective

**Files:**

- Modify: `vfe3/contracts.py`
- Modify: `vfe3/inference/e_step.py:337-437`
- Modify: `vfe3/model/stack.py`
- Modify: `vfe3/model/model.py:1088-1159,1204-1341`
- Modify: `vfe3/train.py:676-708`
- Test: `tests/test_omega_metropolis.py`
- Test: `tests/test_phi_reflection.py`
- Test: `tests/test_phi_reflection_objective_parity_20260712.py`
- Modify: `docs/2026-07-12-edits.md`

**Interfaces:**

- Produces: `MetropolisObjectiveContext` with the fixed q/p state and `EffectiveBetaPriorContext` needed to rebuild candidate-dependent folds.
- `_metropolis_prepare(token_ids, *, mode) -> MetropolisObjectiveContext`.
- `_metropolis_free_energy(belief, context, *, mode) -> float`.

```python
class MetropolisObjectiveContext(NamedTuple):
    token_ids:  torch.Tensor
    mu_p:       torch.Tensor
    sigma_p:    torch.Tensor
    belief:     BeliefState
    tau:        "float | torch.Tensor"
    rope:       Optional[torch.Tensor]
    prior:      EffectiveBetaPriorContext
```

`_metropolis_prepare` stores the final block's converged/current `BeliefState` in `context.belief`; the proposal sweep initializes `f_cur` and the sequential current state from that exact object before constructing trials. Extend `MStepCapture` with `final_block_tau` and use the already captured `final_block_prior`, so the scorer reuses the exact query-adaptive tau and handoff-adjusted p that produced `cap["converged"]`, not the encode prior or a tau recomputed from converged sigma. It also captures the exact positional RoPE tensor used for this token length. For every current/trial evaluation, `_metropolis_free_energy` calls `_effective_beta_log_prior(candidate_belief, context.prior)` and calls `free_energy_value` with fixed `context.tau`, `context.mu_p/sigma_p`, that effective prior, `lambda_twohop`, active `transport_mode`, `connection_W/M/L`, cocycle/link/clamp controls, `transport_mean_per_head`, `context.rope`, `rope_on_cov`, `rope_on_value`, `exp_fp64_mode`, and `exp_fp64_norm_threshold`. Update `free_energy_value` so those two existing parameters are forwarded to `_transport` instead of accepted-and-ignored. Current and trial reuse the same fixed precision/tau/prior moments; only the tied-gamma frame contribution may depend on the proposal. They differ only in the proposed frame/reflection. No scorer-specific approximation warning remains for folded priors, two-hop, adaptive tau, or active transport/RoPE numerics.

- [ ] **Step 1: Write red exact-delta tests.** Parameterize over omega and phi reflection. For precision only, tied gamma only, two-hop only, query-adaptive tau, and all folds together, compare `_metropolis_delta_f` to two independent `free_energy_value` evaluations using manually rebuilt effective priors and the fixed captured tau. Make the final-block entry sigma differ from converged sigma and require the captured entry-derived tau. Add `n_layers=2` with nonzero mean/sigma handoff and require `context.mu_p/sigma_p` to equal `final_block_prior` and differ from encode `prior`. Assign flat/RoPE-on-cov and RoPE-decoupled-value cases to their valid reflection modes; assign `regime_ii`, `regime_ii_covariant`, `regime_ii_link`, and charted-link cases to phi reflection only because omega-direct Metropolis is flat-only. Use exact live connection/control tensors in the oracle. Add a phi norm above `exp_fp64_norm_threshold` and require the configured fp64 island to trigger identically in the active evaluator and Metropolis oracle. Set the captured precision covariance unequal to both current and trial covariance; require the two manual priors to share it while tied gamma changes with the proposal. Test independent `phi_tilde` invariance only with reflection off, because live config rejects phi_tilde plus either Metropolis mode. Add a caller-contract test that `_metropolis_prepare` returns one context whose `belief` tensor shapes match `(N,K)`, and that the first scorer call receives that same current belief before any proposal is applied. Use `V=6`, `K=4`, `N=3`.
- [ ] **Step 2: Write red acceptance-boundary tests.** With a fixed private generator, choose a proposal whose corrected delta changes the accept/reject result relative to the raw-prior scorer; assert the source row, returned belief, and RNG draw count match the corrected objective.
- [ ] **Step 3: Write off-path identity tests.** With both reflection modes off, monkeypatch the scorer to raise and prove the training helper never invokes it. With folds off, require the refactored scorer's delta to equal the existing raw-prior calculation exactly.
- [ ] **Step 4: Run the red reflection tests.**

```powershell
python -m pytest tests/test_omega_metropolis.py tests/test_phi_reflection.py tests/test_phi_reflection_objective_parity_20260712.py --junitxml=C:\tmp\vfe3-phi-reflect-task3-red.xml
```

Expected: folded-prior and two-hop delta comparisons fail against the current scorer.

- [ ] **Step 5: Implement the context and exact scorer.** Capture the base attention prior, exact pre-stack `beliefs.sigma`, model frame, refined s state, final-block handoff prior, final-block entry-derived tau, and RoPE from the same forward preparation. Reuse fixed precision/tau/prior moments and rebuild only the tied candidate-dependent gamma fold per proposal. First make `free_energy_value` forward `exp_fp64_mode` and `exp_fp64_norm_threshold` to `_transport`; then forward `lambda_twohop` and every active transport/RoPE/numerics control listed above from the scorer. Keep the sequential Markov chain and one private RNG draw per proposal unchanged.
- [ ] **Step 6: Remove only obsolete approximation text.** Keep documentation for the stale optimizer-moment caveat; state that precision, gamma, two-hop, adaptive tau, and active transport/RoPE now have objective parity.
- [ ] **Step 7: Run the reflection and resume tests.**

```powershell
python -m pytest tests/test_omega_metropolis.py tests/test_phi_reflection.py tests/test_phi_reflection_objective_parity_20260712.py tests/test_checkpoint_resume.py --junitxml=C:\tmp\vfe3-phi-reflect-task3-green.xml
```

Expected: `failures="0"`, `errors="0"`, exact delta parity, and unchanged private RNG continuation.

- [ ] **Step 8: Commit exact reflection scoring.**

```powershell
git add vfe3/contracts.py vfe3/inference/e_step.py vfe3/model/stack.py vfe3/model/model.py vfe3/train.py tests/test_omega_metropolis.py tests/test_phi_reflection.py tests/test_phi_reflection_objective_parity_20260712.py docs/2026-07-12-edits.md
git commit -m "fix: score reflections against the active objective"
```

### Task 4: Consolidate verification and document the scope boundary

**Files:**

- Modify: `README.md`
- Modify: `docs/2026-07-12-edits.md`
- Test: `tests/test_phi_reflection_objective_parity_20260712.py`

**Interfaces:**

- Documents objective parity for phi and Metropolis and explicitly reserves reflection STE outside this implementation.

- [ ] **Step 1: Add the final CPU combination matrix.** Cover phi updates with two-hop under flat and decoupled-value transport; omega Metropolis with every valid flat-mode fold; phi Metropolis with tied-gamma folds plus every valid nonflat transport; `phi_tilde` gamma invariance only on the reflection-off helper path; the triggered fp64 transport island; checkpointed private RNG continuation; and all-off exact identity. Every case uses `K<6`.
- [ ] **Step 2: Run focused verification.**

```powershell
$env:VFE3_TEST_DEVICE = "cpu"
python -m pytest tests/test_phi_reflection_objective_parity_20260712.py tests/test_phi_retraction.py tests/test_omega_metropolis.py tests/test_phi_reflection.py tests/test_gamma_coupling.py tests/test_checkpoint_resume.py --junitxml=C:\tmp\vfe3-phi-reflect-focused-20260712.xml
```

Expected: `failures="0"` and `errors="0"`.

- [ ] **Step 3: Run the full suite once.**

```powershell
python -m pytest --junitxml=C:\tmp\vfe3-phi-reflect-full-20260712.xml
```

Expected: no new failures or errors relative to the branch baseline; report exact JUnit counts.

- [ ] **Step 4: Add and run one RTX 5090 CUDA smoke.** Add `test_phi_reflection_objective_parity_cuda_smoke` to `tests/test_phi_reflection_objective_parity_20260712.py`, guarded by the repository's `VFE3_TEST_DEVICE` convention. Use `K=4`, `lambda_twohop=0.1`, tied gamma prior, precision weighting, and omega Metropolis; require finite scores, CUDA-resident objective tensors, a deterministic proposal sequence from the private generator, unchanged global CPU/CUDA RNG streams, and no device mismatch. Run it explicitly:

```powershell
$env:VFE3_TEST_DEVICE = "cuda"
python -m pytest tests/test_phi_reflection_objective_parity_20260712.py -k "phi_reflection_objective_parity_cuda_smoke" --junitxml=C:\tmp\vfe3-phi-reflect-cuda-20260712.xml
```

Read the XML and require `tests="1"`, `skipped="0"`, `failures="0"`, and `errors="0"` on the RTX 5090; a skipped smoke is not verification.
- [ ] **Step 5: Update documentation.** Describe the one objective used by mean/covariance/phi/reflection, the fixed-hop detach convention, fold recomputation, all-off identity, and the continuing construction-time rejection of STE.
- [ ] **Step 6: Complete git closeout.** Run `git add vfe3/contracts.py vfe3/inference/e_step.py vfe3/model/block.py vfe3/model/stack.py vfe3/model/model.py vfe3/train.py tests/test_phi_reflection_objective_parity_20260712.py tests/test_phi_retraction.py tests/test_omega_metropolis.py tests/test_phi_reflection.py README.md docs/2026-07-12-edits.md` as the exact final union. Inspect `git diff --check`, `git status --short`, and the staged diff; commit all intended files, push the task branch, merge into `main`, push `main`, safely fast-forward the user's checkout only if WIP is untouched, remove the temporary worktree, and report exact SHAs and JUnit counts.
