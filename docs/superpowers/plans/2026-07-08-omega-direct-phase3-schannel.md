# omega_direct Phase 3 — Implementation Plan (gamma / s-channel frame-fidelity)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Thread the stored frame `belief.omega` (`U_i`) through the gamma / model-coupling (s) channel so omega_direct transports it by `U_i U_j⁻¹` instead of `exp(phi)`, then remove the config gate that blocks `omega_direct + lambda_gamma>0 / s_e_step / gamma_as_beta_prior`. Spec: `docs/superpowers/specs/2026-07-08-omega-direct-phase3-schannel-design.md`. This unblocks the user's active training run (`omega_direct, lambda_gamma=0.75, s_e_step=True, gamma_as_beta_prior=True`).

**Architecture:** The s-channel reuses the belief frame (one `U` per token = `belief.omega`); the downstream funnel (`_gamma_energy`→`build_belief_transport`; `e_step`) is already omega-aware. The work is threading the existing tensor to the s-channel call sites that currently drop it. No new transport code, no new storage.

**Tech Stack:** Python, PyTorch. pytest.

## Global Constraints

- **Two must-not-change paths stay byte-identical:** (i) `gauge_parameterization="phi"` (every new `omega` arg is `None`, every `gp` falls to `"phi"`); (ii) shipped omega_direct BELIEF path with gamma OFF (`lambda_gamma=0, s_e_step=False, gamma_as_beta_prior=False`) — none of the s-channel methods are entered. Guard every forward with `... if X.omega is not None else None` (template `model.py:1344`).
- **Footgun:** `build_transport_from_element(None, group)` raises `AttributeError`. Never pass `omega=None` while forcing `gp="omega_direct"`; keep frame-present ⇔ `gp=="omega_direct"` coupled.
- **Gradient parity (load-bearing):** `_refine_s`'s `omega_s` stays **ATTACHED** (trains `omega_embed`, mirrors `phi_embed`). Forward gamma loss (`model.py:1102`) passes `belief.omega.detach()` (frame-inert, parity with `phi.detach()`). `_fold_gamma_prior` is under `no_grad` — pass `beliefs.omega` directly.
- **Frame source is `pb._omega_lookup(token_ids)`**, NOT `pb.omega_embed[token_ids]` (the former is compact-storage-aware + autograd-correct).
- **Remove the config gate LAST** (Task 5), after all threading + viz land.
- **Do NOT touch WIP:** `CLAUDE.md`, `scaling.py`, `scaling_analysis.py`, `train_vfe3.py`, `vfe3/geometry/transport.py`, `docs/2026-07-08-edits.md`, `docs/audits/`. (No Phase-3 file is WIP — all clean.)
- CLAUDE.md conventions; American English; CPU tests tiny (K<6). Pytest no extra `-q`; run focused tests SYNCHRONOUSLY.

## File Structure
- `vfe3/model/model.py` — thread `omega` through `_gamma_coupling_term`, `_gamma_coupling_terms`, `_fold_gamma_prior` (+ callers) and `_refine_s`.
- `vfe3/viz/extract.py` — thread `omega` into the three gamma folds; populate omega at `extract.py:46`.
- `vfe3/config.py` — remove the gate (Task 5).
- `ablation.py` — drop gamma-off overrides from omega cells + fix the pre-existing `gauge_group` arm (Task 6).
- `tests/test_config.py`, `tests/test_omega_direct.py` — invert gate tests + frame-fidelity property tests.

---

### Task 1: Thread `omega` through the forward gamma loss + diagnostic split (`lambda_gamma` path)

**Files:** Modify `vfe3/model/model.py` (`_gamma_coupling_term` ~1244, `_gamma_coupling_terms` ~1267, callers ~1102, ~1758). Test: `tests/test_omega_direct.py`.

**Interfaces:** Produces `_gamma_coupling_term(token_ids, phi, *, omega=None)` and `_gamma_coupling_terms(..., omega=None, s_belief=None)` that forward `omega` to `_gamma_energy` (already omega-capable). Byte-identical when omega is None.

- [ ] **Step 1: Failing test** — append a direct frame-fidelity test: call `_gamma_coupling_term` (or `_gamma_energy` directly) on a tiny omega_direct model with `belief.phi` held at zero and two DIFFERENT non-identity frames `U1=I`, `U2` (via `retract_omega`), assert the gamma energy differs (proves it uses `U`, not the fixed `exp(phi)=I`). Because the config gate still blocks omega_direct+gamma, build the model with gamma OFF but call the method directly passing `omega=U` (the method doesn't re-check the config). Mirror `test_appended_belief_step_omega_direct_uses_stored_frame` (test_omega_direct.py:354).
- [ ] **Step 2: Run** → expect FAIL (method drops omega → both frames give the same phi-cocycle result).
- [ ] **Step 3: Implement.** Add `omega: Optional[torch.Tensor] = None` (keyword-only, alongside `s_belief`) to `_gamma_coupling_term` (model.py:1244) and `_gamma_coupling_terms` (model.py:1267); forward `omega=omega` into their `_gamma_energy` calls (model.py:1258, 1290). At the forward caller (model.py:1102), pass `omega=belief.omega.detach() if belief.omega is not None else None` alongside `belief.phi.detach()`. At the diagnostic caller (model.py:1758), pass `omega=out.omega.unsqueeze(0) if out.omega is not None else None`.
- [ ] **Step 4: Run** the new test (pass) + `pytest tests/test_omega_direct.py tests/test_model.py` (byte-identity on phi/gamma-off paths).
- [ ] **Step 5: Commit** `git add vfe3/model/model.py tests/test_omega_direct.py`; `feat(omega_direct): frame-fidelity for the forward gamma loss + diagnostic split`.

---

### Task 2: Thread `omega` through `_fold_gamma_prior` + all callers (`gamma_as_beta_prior`, forward-value)

**Files:** Modify `vfe3/model/model.py` (`_fold_gamma_prior` ~1548, callers ~816, ~1328, ~1655, ~1934, ~2047). Test: `tests/test_omega_direct.py`.

**Interfaces:** `_fold_gamma_prior(log_prior, token_ids, phi, *, omega=None)` forwards `omega` to `_gamma_energy` (under `no_grad`). This fold shapes the forward VALUE (feeds `log_prior` → `vfe_stack`), so frame-fidelity here changes the prediction under omega_direct.

- [ ] **Step 1: Failing test** — direct frame-fidelity: call `_fold_gamma_prior` with `phi` fixed and two frames `U1=I`, `U2`, assert the returned `log_prior` differs (proves the fold uses `U`). Since it's `no_grad`, no gradient concern.
- [ ] **Step 2: Run** → FAIL (drops omega).
- [ ] **Step 3: Implement.** Add `omega: Optional[torch.Tensor] = None` to `_fold_gamma_prior` (model.py:1548); forward `omega=omega` into its `_gamma_energy` call (model.py:1576). At the forward hot-path caller (model.py:816), pass `omega=beliefs.omega`. At each diagnostic replay caller (model.py:1328 inside `gamma_attention_maps`, 1655, 1934, 2047), pass `omega=belief.omega.unsqueeze(0) if belief.omega is not None else None` (each `belief` there populates `omega` at construction).
- [ ] **Step 4: Run** new test (pass) + `pytest tests/test_omega_direct.py tests/test_model.py`.
- [ ] **Step 5: Commit** `git add vfe3/model/model.py tests/test_omega_direct.py`; `feat(omega_direct): frame-fidelity for _fold_gamma_prior (gamma_as_beta_prior forward value)`.

---

### Task 3: `_refine_s` frame-fidelity (`s_e_step` path — the disjunct the user trips)

**Files:** Modify `vfe3/model/model.py` (`_refine_s` ~594-674). Test: `tests/test_omega_direct.py`.

**Interfaces:** `_refine_s` re-derives `omega_s = pb._omega_lookup(token_ids)` (attached) under omega_direct, populates the s-belief's `omega`, and passes `gauge_parameterization` to `e_step`. No caller edits.

- [ ] **Step 1: Failing test** — call `_refine_s` (or a tiny model's s-refine) with `phi0` fixed and the model's `omega_embed` set to two different frames, assert the refined `s_mu`/`s_sigma` differ (proves the s E-step transports by `U`). Bypass the config gate by building gamma-off then calling `_refine_s` directly with `s_e_step` semantics, OR by temporarily setting cfg.gauge_parameterization on the built model (the method reads `cfg.gauge_parameterization`). Prefer a direct `_refine_s` call.
- [ ] **Step 2: Run** → FAIL (s-belief omega is None → uses exp(phi0)).
- [ ] **Step 3: Implement** (brief §A.4, all no-ops on the phi path). In `_refine_s`: after `s_mu, s_sigma = pb.encode_s(token_ids)` (~model.py:611), add
```python
    omega_s = pb._omega_lookup(token_ids) if cfg.gauge_parameterization == "omega_direct" else None
```
set `omega=omega_s` in the `BeliefState(mu=s_mu, sigma=s_sigma, phi=phi0, omega=omega_s)` (model.py:619), and add `gauge_parameterization=cfg.gauge_parameterization` to the `e_step(...)` call (kwargs block ~model.py:618-673). Keep `transport_mode="flat"`. Do NOT detach `omega_s`.
- [ ] **Step 4: Run** new test (pass) + `pytest tests/test_omega_direct.py tests/test_model.py`. Confirm the phi-path `_refine_s` is byte-identical (omega_s=None, gauge_parameterization="phi").
- [ ] **Step 5: Commit** `git add vfe3/model/model.py tests/test_omega_direct.py`; `feat(omega_direct): frame-fidelity for the s-channel E-step (_refine_s)`.

---

### Task 4: viz/extract.py — three gamma folds + populate omega at extract.py:46

**Files:** Modify `vfe3/viz/extract.py` (`~46`, `~54`, `~211`, `~283`). Test: none new required (diagnostics), or a light smoke.

**Interfaces:** The three `gamma_as_beta_prior` folds in extract.py use the belief frame; the `BeliefState` at extract.py:46 is populated with `omega` so its fold matches the forward.

- [ ] **Step 1: Implement.** At `extract.py:46`, add `omega=enc.omega[0] if enc.omega is not None else None` to the `BeliefState(...)` construction (matching model.py:1319-1320). At the three `_fold_gamma_prior` folds (extract.py:54, 211, 283), forward `omega=belief.omega` / `beliefs.omega` (guarded) — sites 211/283 already have `beliefs.omega` in scope; site 54 now has it from the populate.
- [ ] **Step 2: Verify** — `python -c "import vfe3.viz.extract"` and run any existing viz/extract test (`pytest tests/ -k extract` — skip if none). If there is an omega_direct diagnostics test, run it. This is a diagnostics-visibility change; the key correctness is that the replay frame matches the forward (covered once the gate is gone in Task 5's gauge-invariance test).
- [ ] **Step 3: Commit** `git add vfe3/viz/extract.py`; `feat(omega_direct): thread the frame through viz/extract gamma folds + populate omega at extract.py:46`.

---

### Task 5: Remove the config gate + invert gate tests + full-model frame-fidelity property tests (CAPSTONE — unblocks the user)

**Files:** Modify `vfe3/config.py` (delete ~941-955). Modify `tests/test_config.py` (invert the rejection test) and `tests/test_omega_direct.py` (add the end-to-end tests). Depends on Tasks 1-4.

**Interfaces:** `omega_direct` now constructs and runs with `lambda_gamma>0 / s_e_step / gamma_as_beta_prior`.

- [ ] **Step 1: Invert the gate tests.** `test_omega_direct_rejects_active_gamma_channel` (test_config.py:624-636, parametrized over `lambda_gamma=0.1` / `s_e_step=True` / `gamma_as_beta_prior=True,lambda_gamma=0.1`) → rewrite so all three now CONSTRUCT (no raise). Keep `test_omega_direct_pure_channel_off_constructs`, `test_omega_direct_accepts_all_eligible_groups`, and `test_free_energy_value_filtered_keys_rejects_omega_direct` (the e_step.py:330 gate STAYS) green.
- [ ] **Step 2: Run** → FAIL (gate still raises).
- [ ] **Step 3: Remove the gate.** Delete `config.py:941-955` (the `if self.lambda_gamma > 0.0 or self.s_e_step or self.gamma_as_beta_prior: raise ValueError(...)` block + its comment). Keep the other omega_direct guards (transport_mode, e_phi_lr, group, reflection).
- [ ] **Step 4: Add the end-to-end frame-fidelity tests** to `tests/test_omega_direct.py`:
  - **Finite forward at the user's target**: build `VFEModel` at `gauge_parameterization="omega_direct", lambda_gamma=0.75, s_e_step=True, gamma_as_beta_prior=True, prior_source="model_channel", family="gaussian_diagonal"` (tiny dims, gauge_group="glk" or "block_glk"), assert `torch.isfinite(m(tok)[0]).all()`. This is the user's config — it must run.
  - **Gauge invariance with gamma ON**: extend `test_omega_direct_full_model_gauge_invariance` (test_omega_direct.py:240) with `lambda_gamma>0, s_e_step=True`, co-transforming the s tables (`s_mu_embed`) and `r_mu` as well as `mu_embed`/`omega_embed` by `U→gU` (orthogonal g), asserting decode invariance to fp64. Certifies the s-channel now transports under the same `U`.
- [ ] **Step 5: Run** `pytest tests/test_config.py tests/test_omega_direct.py tests/test_model.py` (all green — the target config now works).
- [ ] **Step 6: Commit** `git add vfe3/config.py tests/test_config.py tests/test_omega_direct.py`; `feat(omega_direct): remove gamma-channel gate + end-to-end s-channel frame-fidelity tests`.

---

### Task 6: Ablation — drop gamma-off overrides + fix the pre-existing `gauge_group` arm bug

**Files:** Modify `ablation.py` (omega cells ~422-453; `gauge_group` arm ~387, ~391-392). Test: `tests/test_omega_direct.py` (the ablation test).

**Interfaces:** The omega ablation cells inherit BASELINE_CONFIG's gamma (0.75 / True) now that omega_direct supports it; the pre-existing `gauge_group` arm builds.

- [ ] **Step 1: Ablation omega cells.** Drop `"lambda_gamma": 0.0, "s_e_step": False` from all eight `gauge_parameterization` omega cells (ablation.py:422-453) so they inherit `BASELINE_CONFIG`'s `0.75/True` (apples-to-apples with the phi arm, both gamma-on). Refresh the stale rationale comment (ablation.py:402-406).
- [ ] **Step 2: Pre-existing `gauge_group` arm bug** (independent hygiene, brief §D). `tied_block_glk` cell (ablation.py:387): add `"phi_precond_mode": "killing"` (it inherits `killing_per_block`, rejected for tied). `so3_spin2x4` cell (ablation.py:391-392): add `"n_heads": 4` (irrep_spec `[("l2",4)]` yields 4 blocks; baseline n_heads=2 + causal_alibi trips the head-count check).
- [ ] **Step 3: Update the ablation test** (`test_ablation_omega_direct_arm_builds`, test_omega_direct.py:712): flip the `lambda_gamma == 0.0` / `s_e_step is False` assertions (test_omega_direct.py:739-740) to `== 0.75` / `is True` (or drop them); refresh the docstring. Optionally add a `gauge_group`-sweep build check covering the two fixed cells.
- [ ] **Step 4: Run** `pytest tests/test_omega_direct.py -k ablation` + `python -c "import ablation"` synchronously, then whole `pytest tests/test_omega_direct.py`.
- [ ] **Step 5: Commit** `git add ablation.py tests/test_omega_direct.py`; `feat(omega_direct): ablation gamma-on omega cells + fix pre-existing gauge_group arm cells`.

---

## Self-Review

**Spec coverage:** gamma loss → T1; _fold_gamma_prior → T2; _refine_s → T3; viz → T4; gate removal + end-to-end → T5; ablation + hygiene → T6. Deferred (STE, tower compaction, so_n seed, filtered-diagnostic, e_phi_lr) NOT in this plan.

**Byte-identity:** every task's new `omega` defaults None and every `gp` falls to `"phi"`; the gate stays up through T1-T4 (so omega_direct+gamma is still rejected — the threading is tested by direct method calls with a non-identity omega); T5 removes the gate + proves the target config runs; T6 adds arms.

**Gradient parity:** T1 detaches (gamma loss), T3 attaches (`_refine_s` trains omega_embed), T2 under no_grad. Flagged.

**Type consistency:** `omega` kwarg name uniform; `_omega_lookup` (not `omega_embed[...]`) in T3; `... if X.omega is not None else None` guards uniform.

## Execution Handoff

Subagent-driven, task-by-task, review + fix loop after each, final whole-branch review, then push + FF-merge to main (Phase-1/2 flow). Unblocks the user's `omega_direct + gamma` run at Task 5.
