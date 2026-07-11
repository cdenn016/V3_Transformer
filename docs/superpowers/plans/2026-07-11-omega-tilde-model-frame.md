# Omega-Tilde Model-Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-tied, opt-in `phi_tilde` model gauge frame that independently transports the model-channel Gaussians without changing the belief-frame path or `omega_direct`.

**Architecture:** `VFE3Config.s_frame_mode` selects a registry-backed model-frame resolver. The tied resolver returns the already composed belief frame; the `phi_tilde` resolver reads independently stored token and learned-position frame parameters, composes them with the existing positional-phi machinery, and passes that explicit tensor through all gamma and model-refinement consumers. The existing `BeliefState` and E-step transport kernels remain generic and unchanged.

**Tech Stack:** Python 3, PyTorch, pytest, existing VFE3 registries and Gaussian transport kernels.

## Global Constraints

The default is `s_frame_mode="tied"`, which creates no new parameters, consumes no random draws, and preserves the current state dictionary and values. `phi_tilde` is available only with `gauge_parameterization="phi"`, `s_e_step=True`, `prior_source="model_channel"`, `share_refine_s_transport=False`, `phi_reflection="off"`, and `pos_rotation="none"`. The feature does not modify `e_step_update`, `mm_exact`, `omega_direct`, the belief transport, or the model-channel scalar objective. Learned token and positional model frames start as detached clones of the corresponding belief-frame tables. Both click-to-run dictionaries expose the new controls, defaulted to tied/off.

### Task 1: Pin configuration, storage, initialization, and entrypoint contracts

**Files:**

- Create: `tests/test_omega_tilde_model_frame.py`
- Modify: `vfe3/config.py`
- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/model/model.py`
- Modify: `train_vfe3.py`
- Modify: `ablation.py`

**Interfaces:**

- Consumes: `VFE3Config`, `PriorBank.phi_embed`, `VFEModel.pos_phi_free`.
- Produces: `VFE3Config.s_frame_mode`, `VFE3Config.m_s_phi_lr`, `PriorBank.s_phi_embed`, `PriorBank.s_phi(token_ids)`, and optional `VFEModel.s_pos_phi_free`.

- [ ] Write tests proving tied mode has no `s_phi_embed` or `s_pos_phi_free` key and that `phi_tilde` creates independent, equal-valued cloned storage.
- [ ] Write parameterized rejection tests for `omega_direct`, inactive `s_e_step`, non-model-channel priors, shared transport, phi reflection, and gauge RoPE; write warning tests for a severed E-step or zero `m_s_phi_lr`.
- [ ] Write source/config tests proving `train_vfe3.py` and `ablation.py` expose `s_frame_mode="tied"` and `m_s_phi_lr=0.016`, with inactive ablation arms for both fields.
- [ ] Run `python -m pytest tests/test_omega_tilde_model_frame.py --junitxml=C:\tmp\vfe3-omega-tilde-red-config.xml` and confirm failures are caused by the absent fields/tables.
- [ ] Add the two config fields and validation without changing `e_step_update`.
- [ ] Create `s_phi_embed` only under `phi_tilde`, clone it from `phi_embed`, and add `PriorBank.s_phi`.
- [ ] Clone `s_pos_phi_free` from `pos_phi_free` only under `phi_tilde` plus learned positional phi.
- [ ] Add the default-off click dictionaries and inactive ablation arms.
- [ ] Rerun the focused file and require zero failures.

### Task 2: Resolve and route the complete model frame

**Files:**

- Create: `vfe3/model/model_frame.py`
- Modify: `vfe3/model/model.py`
- Test: `tests/test_omega_tilde_model_frame.py`

**Interfaces:**

- Consumes: `resolve_model_frame(mode, token_ids, belief_phi, *, token_phi, group, positional settings)`.
- Produces: `VFEModel._model_phi(token_ids, belief_phi) -> torch.Tensor` and explicit model-frame arguments for `_refine_s`, `_gamma_energy`, `_fold_gamma_prior`, gamma diagnostics, and snapshot replay.

- [ ] Write a red test that copied token and learned-position tables yield equal tied and `phi_tilde` effective frames and transports at construction, but different storage pointers.
- [ ] Write the load-bearing red test: perturb only `s_phi_embed`, require `_gamma_energy` and `_refine_s` to change, and require the belief transport to remain fixed.
- [ ] Write the converse red test: pass a fixed explicit model frame while perturbing the belief frame and require gamma/refinement values to remain unchanged.
- [ ] Run the three tests and confirm they fail because the model channel still reads belief phi.
- [ ] Implement the registry with `tied` and `phi_tilde` builders and a strict duplicate-registration guard.
- [ ] Resolve the effective model frame once at each orchestration boundary and thread it explicitly through model refinement, gamma prior folding, scored gamma, attention maps, diagnostics, and snapshot capture.
- [ ] Keep the scored `s_e_step=False` gamma-frame detach boundary unchanged; do not perform hidden parameter lookup inside `_gamma_energy`.
- [ ] Add `DiagnosticSnapshot.model_phi` as a detached clone used by snapshot-derived gamma diagnostics.
- [ ] Run the focused routing tests and require zero failures.

### Task 3: Train the independent model frame on its own optimizer clock

**Files:**

- Modify: `vfe3/train.py`
- Modify: `train_vfe3.py`
- Test: `tests/test_omega_tilde_model_frame.py`
- Test: `tests/test_train.py`

**Interfaces:**

- Consumes: `PriorBank.s_phi_embed`, optional `VFEModel.s_pos_phi_free`, `VFE3Config.m_s_phi_lr`.
- Produces: one model-frame optimizer group with role `phi`, gauge metadata on the natural-gradient path, and exact parameter coverage.

- [ ] Write a red test proving both independent-frame parameters are grouped exactly once at `m_s_phi_lr` with the expected weight-decay and gauge metadata.
- [ ] Write a red attached-E-step test proving nonzero finite `s_phi_embed.grad` and actual parameter motion after an optimizer step.
- [ ] Run the tests and confirm failure at the optimizer coverage guard or absent gradient.
- [ ] Add the model-frame group to `build_optimizer`, extend the clamp monitor to report belief/model token and positional tables separately, and include `m_s_phi_lr` in the click-run banner.
- [ ] Rerun the optimizer and gradient tests and require zero failures.

### Task 4: Geometry, regression, documentation, and integration

**Files:**

- Modify: `tests/test_omega_tilde_model_frame.py`
- Modify: `docs/2026-07-11-edits.md`

**Interfaces:**

- Consumes: the completed model-frame registry and routing.
- Produces: cocycle and shared-coordinate pushforward invariants plus the dated implementation record.

- [ ] Add tests for `tilde Omega_ij tilde Omega_jk = tilde Omega_ik`, tied-default state/value identity, checkpoint round-trip, and CPU forward/backward finiteness.
- [ ] Run `python -m pytest tests/test_omega_tilde_model_frame.py tests/test_live_s_model_channel.py tests/test_gamma_coupling.py tests/test_model_channel_diagnostics_2026_06_13.py tests/test_train.py --junitxml=C:\tmp\vfe3-omega-tilde-focused-20260711.xml`.
- [ ] Run `python -m pytest --junitxml=C:\tmp\vfe3-omega-tilde-full-20260711.xml`; compare any failures to unchanged `origin/main` failures and do not alter `mm_exact` or unrelated sweeps.
- [ ] Run a CUDA smoke with `VFE3_TEST_DEVICE=cuda` when CUDA is available, covering frame resolution, gamma energy, backward, and optimizer motion.
- [ ] Update `docs/2026-07-11-edits.md`, inspect `git diff --check`, status, and the staged diff.
- [ ] Commit all intended files, fetch `origin/main`, prove zero-behind fast-forward ancestry, push the feature branch and `main`, preserve the dirty live checkout, and report exact SHAs and machine-readable verification results.
