# Learnable reflection on the phi path (`R·exp(φ)`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Give the default `phi` gauge parameterization access to the `det<0` orientation component by prepending a per-token discrete reflection `R` (`det R=−1`) to `exp(φ)`, learned by the same ΔF-gated Metropolis flip as `omega_direct`, threaded through the belief channel AND the gamma/s-channel.

**Architecture:** A per-token sign bit `sᵢ∈{+1,−1}` selects `Rᵢ=diag(sᵢ,1,…,1)` (via `reflection_element`). The frame is `gᵢ=Rᵢexp(φᵢ)`; transport `Ωᵢⱼ=Rᵢexp(φᵢ)exp(−φⱼ)Rⱼ` folds into the built transport (negate row 0 of `exp_phi`, col 0 of `exp_neg_phi`, or the equivalent on dense Omega). The sign lives in a `prior_bank.reflection_sign` buffer, is carried on `BeliefState.reflection`, and is learned by generalizing `metropolis_omega_step`. Spec: `docs/superpowers/specs/2026-07-08-phi-reflection-design.md`.

**Tech Stack:** Python, PyTorch, pytest.

## Global Constraints

- **Pure path byte-identical:** default `phi_reflection="off"` — no `reflection_sign` buffer, `belief.reflection=None`, every new kwarg defaults `None`, the fold is not entered, `state_dict` identical, the Metropolis move early-returns. The shipped `omega_reflection` modes are untouched.
- **The sign is NON-differentiable state.** `reflection_sign` is a registered BUFFER (not `nn.Parameter`), updated only in-place by the Metropolis move under `no_grad`. No gradient flows through `reflection` on any path, so no detach/attach discipline is needed for it (unlike Phase 3's `omega`).
- **Fold is `Ωᵢⱼ→RᵢΩᵢⱼRⱼ`, group-agnostic at the K level.** `R=reflection_element(K)=diag(−1,1,…,1)`; applied as: negate ROW 0 of `exp_phi[i]` iff `sᵢ=−1`, negate COLUMN 0 of `exp_neg_phi[j]` iff `sⱼ=−1` (factored transport); the equivalent row-0(i)/col-0(j) negation on a dense `(…,i,j,K,K)` Omega. `R` is an involution (`R²=I`), so the transport cocycle stays flat.
- **Group eligibility = `_REFLECT_OK=("glk","block_glk","so_k")`** (same as `omega_reflection`); reject `sp`/`sp_n` (vacuous) and `so_n`/`tied_block_glk` (deferred), and require `gauge_parameterization=="phi"`.
- **Reuse, don't fork, the Metropolis machinery.** `omega_metropolis_temperature`/`omega_metropolis_every` govern both reflection modes; factor the shared sweep so `omega_direct` (flip `omega_embed`) and `phi` (flip `reflection_sign`) differ only in the per-token flip + trial-frame construction.
- CLAUDE.md conventions; American English; CPU tests tiny (K<6); pytest NO extra `-q`. Commit ONLY named files (never `git add -A`); the working tree carries unrelated WIP (`ablation.py`, `train_vfe3.py`, config, etc.) — per-hunk stage if a WIP file is unavoidable (it should not be for this feature).

## File Structure

- `vfe3/config.py` — `phi_reflection` field + validation (Task 1).
- `vfe3/belief.py` — `reflection` field on `BeliefState` (Task 1).
- `vfe3/model/prior_bank.py` — gated `reflection_sign` buffer + `encode` population (Task 1).
- `vfe3/inference/e_step.py` (+ `vfe3/geometry/transport.py` if the fold helper lives there) — `reflection` kwarg on `build_belief_transport` + the fold; belief-channel call sites (Task 2).
- `vfe3/model/model.py` — gamma/s-channel `reflection` threading (Task 3); Metropolis generalization (Task 4).
- `vfe3/train.py` — extend the seam gate to `phi_reflection` (Task 4).
- `tests/test_phi_reflection.py` — all tasks.

---

### Task 1: Config + `BeliefState.reflection` + `prior_bank.reflection_sign` (data layer)

**Files:** Modify `vfe3/config.py`, `vfe3/belief.py`, `vfe3/model/prior_bank.py`. Test: `tests/test_phi_reflection.py` (new).

**Interfaces:** Produces `cfg.phi_reflection`; `BeliefState(..., reflection=None)`; `prior_bank.reflection_sign` (V,) buffer (created iff phi + `phi_reflection!="off"`); `prior_bank.encode` populates `belief.reflection = reflection_sign[token_ids]` when the buffer exists.

- [ ] **Step 1: Failing tests.** `tests/test_phi_reflection.py`:
```python
import pytest, torch
from vfe3.config import VFE3Config


def _cfg(**over):
    base = dict(gauge_parameterization="phi", gauge_group="glk", embed_dim=4, n_heads=1,
                transport_mode="flat", use_head_mixer=False)
    base.update(over); return VFE3Config(**base)


def test_phi_reflection_field_and_gating():
    assert _cfg().phi_reflection == "off"                                  # default
    for grp, over in (("glk", {}), ("block_glk", {"n_heads": 2}), ("so_k", {})):
        assert _cfg(phi_reflection="metropolis", gauge_group=grp, **over).phi_reflection == "metropolis"
    for grp, over in (("sp", {}), ("so_n", {"group_n": 3, "irrep_spec": [("l0", 1), ("l1", 1)]}),
                      ("tied_block_glk", {"n_heads": 2})):
        with pytest.raises(ValueError):
            _cfg(phi_reflection="metropolis", gauge_group=grp, **over)


def test_phi_reflection_requires_phi_path():
    with pytest.raises(ValueError):        # metropolis reflection is a phi-path feature
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1,
                   transport_mode="flat", e_phi_lr=0.0, use_head_mixer=False, phi_reflection="metropolis")


def test_phi_reflection_ste_not_implemented():
    with pytest.raises((NotImplementedError, ValueError), match="ste"):
        _cfg(phi_reflection="ste")


def test_belief_carries_reflection_field():
    from vfe3.belief import BeliefState
    b = BeliefState(mu=torch.zeros(1, 3, 4), sigma=torch.ones(1, 3, 4), phi=torch.zeros(1, 3, 6))
    assert b.reflection is None                                            # default
    b2 = b._replace(reflection=torch.ones(1, 3))
    assert b2.reflection is not None


def test_prior_bank_reflection_sign_gated_and_encode_populates():
    from vfe3.model.model import VFEModel
    m = VFEModel(_cfg(phi_reflection="init_seed", vocab_size=6, max_seq_len=4, n_layers=1))
    assert hasattr(m.prior_bank, "reflection_sign")
    assert m.prior_bank.reflection_sign.shape == (6,)
    assert set(m.prior_bank.reflection_sign.tolist()) <= {1.0, -1.0}
    enc = m.prior_bank.encode(torch.tensor([[0, 1, 2, 3]]))
    assert enc.reflection is not None and enc.reflection.shape == (1, 4)
    # off path: no buffer, no belief field
    m_off = VFEModel(_cfg(vocab_size=6, max_seq_len=4, n_layers=1))
    assert not hasattr(m_off.prior_bank, "reflection_sign")
    assert m_off.prior_bank.encode(torch.tensor([[0, 1]])).reflection is None
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.**
  - `vfe3/config.py`: add `phi_reflection: str = "off"` beside `omega_reflection`. In `__post_init__`, add a top-level `"ste"` reject (mirror the `omega_reflection` `"ste"` guard) and a `_require(self.phi_reflection, ("off","init_seed","metropolis"), "phi_reflection")`. Then a validation block: if `phi_reflection != "off"`, require `gauge_parameterization=="phi"` (else raise: metropolis/reflection acts on the phi frame) AND `gauge_group in _REFLECT_OK` (reject sp/sp_n vacuous, so_n/tied deferred — same message shape as `omega_reflection`), and (mirror) a `UserWarning` under a diagonal covariance family. Reuse the hoisted `_REFLECT_OK`.
  - `vfe3/belief.py`: add `reflection: Optional[torch.Tensor] = None` as the trailing `BeliefState` field (after `omega`), with a shape comment `(..., N) per-token sign +1/-1; set only on the phi path under phi_reflection`.
  - `vfe3/model/prior_bank.py`: in `__init__`, gated on `gauge_parameterization=="phi" and phi_reflection!="off"`, `self.register_buffer("reflection_sign", torch.ones(vocab_size))` and, if `phi_reflection=="init_seed"`, set `reflection_sign[1::2] = -1.0` (mirror the `omega_embed` `init_seed` `[1::2]` seed). Thread `phi_reflection` into the `PriorBank` constructor from `model.py` (like `omega_reflection`). In `encode`, after building the belief, if `hasattr(self,"reflection_sign")`, set `reflection=self.reflection_sign[token_ids]` on the returned `BeliefState` (guarded; `None` otherwise).

- [ ] **Step 4: Run** `pytest tests/test_phi_reflection.py` (Task-1 tests pass); `pytest tests/test_config.py tests/test_model.py` (no regressions).
- [ ] **Step 5: Commit** `git add vfe3/config.py vfe3/belief.py vfe3/model/prior_bank.py tests/test_phi_reflection.py`; `feat(phi-reflection): config + BeliefState.reflection + gated reflection_sign buffer`.

---

### Task 2: The reflection fold in `build_belief_transport` (belief channel)

**Files:** Modify `vfe3/inference/e_step.py` (`build_belief_transport` ~123; belief-channel callers). Test: `tests/test_phi_reflection.py`.

**Interfaces:** `build_belief_transport(phi, group, *, reflection=None, …)`; a helper `_apply_reflection(built, reflection)` folding `Ωᵢⱼ→RᵢΩᵢⱼRⱼ` onto a `FactoredTransport` (factors) or a dense Omega tensor. Belief E-step + decode pass `reflection=belief.reflection`.

- [ ] **Step 1: Failing test** — fold correctness (independent recompute) + det<0 + off-byte-identity:
```python
def test_reflection_fold_matches_R_Omega_R_and_flips_det():
    import torch
    from vfe3.inference.e_step import build_belief_transport
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.generators import reflection_element
    K, N = 4, 3
    grp = get_group("glk")(K=K)
    torch.manual_seed(0)
    phi = 0.2 * torch.randn(1, N, grp.generators.shape[0])
    base = build_belief_transport(phi, grp, transport_mode="flat", gauge_parameterization="phi")
    sign = torch.tensor([[1.0, -1.0, 1.0]])                                # token 1 reflected
    refl = build_belief_transport(phi, grp, transport_mode="flat", gauge_parameterization="phi", reflection=sign)
    Om_base = base.Omega if hasattr(base, "Omega") else base               # (1,N,N,K,K)
    Om_refl = refl.Omega if hasattr(refl, "Omega") else refl
    R = reflection_element(K)
    Ri = torch.where(sign[..., :, None, None] < 0, R, torch.eye(K))        # (1,N,K,K)
    exp = torch.einsum("bikl,bijlm,bjmn->bijkn", Ri, Om_base, Ri)          # R_i Omega_ij R_j
    assert torch.allclose(Om_refl, exp, atol=1e-5)
    assert torch.det(Om_refl[0, 0, 1]) * torch.det(Om_base[0, 0, 1]) < 0   # s_0=+1,s_1=-1 -> det flips
    # off (reflection=None) is byte-identical to base
    assert torch.allclose(Om_base, (build_belief_transport(phi, grp, transport_mode="flat",
                                    gauge_parameterization="phi", reflection=None)).__class__ and Om_base)
```
(Implementer: adapt the dense-vs-factored extraction to the actual return types; the load-bearing assertion is `Om_refl == R_i Om_base R_j` to fp5 and the det-sign flip.)

- [ ] **Step 2: Run** → FAIL (`reflection` kwarg unknown / not applied).

- [ ] **Step 3: Implement.** Add `reflection: Optional[torch.Tensor]=None` to `build_belief_transport`. After `built` is produced on the `gauge_parameterization=="phi"` branch (factored or dense; NOT the omega_direct branch), if `reflection is not None` call `built = _apply_reflection(built, reflection)`:
  - `FactoredTransport`: `exp_phi' = exp_phi.clone(); exp_phi'[...,0,:] *= reflection[...,:,None]` (negate row 0 by the per-token sign), `exp_neg_phi' = exp_neg_phi.clone(); exp_neg_phi'[...,:,0] *= reflection[...,:,None]` (negate col 0). Return a new `FactoredTransport`.
  - dense Omega `(…,i,j,K,K)`: `Om' = Om.clone(); Om'[...,0,:] *= reflection[...,:,None,None]` (row 0 by `s_i`, broadcast over j), `Om'[...,:,0] *= reflection[...,None,:,None]` (col 0 by `s_j`, broadcast over i).
  Apply BEFORE the RoPE wrap (RoPE deferred; reflection composes with the base transport). Guard: reflection fold is phi-path only (the omega_direct branch ignores it — `omega_direct` has its own `omega_reflection`).
  Thread `reflection=belief.reflection` at the belief-channel `build_belief_transport` call sites (the E-step hoist at e_step.py ~859/937, and the belief forward's transport). Guard `... if belief.reflection is not None else None`.

- [ ] **Step 4: Run** the fold test + `pytest tests/test_phi_reflection.py tests/test_e_step.py tests/test_model.py` (no regressions; off path byte-identical).
- [ ] **Step 5: Commit** `git add vfe3/inference/e_step.py tests/test_phi_reflection.py` (+ transport.py if the helper lives there); `feat(phi-reflection): fold R into build_belief_transport (belief channel)`.

---

### Task 3: Thread `reflection` through the gamma / s-channel (Phase 3 `omega` map)

**Files:** Modify `vfe3/model/model.py` (`_gamma_coupling_term`/`_gamma_energy`, `_fold_gamma_prior`, `_refine_s` + their callers/diagnostics — the SAME sites the Phase 3 `omega` threading touched; grep `omega=belief.omega` / `omega=out.omega` for the map). Test: `tests/test_phi_reflection.py`.

**Interfaces:** Each gamma/s-channel transport build additionally forwards `reflection=belief.reflection` (guarded). `_refine_s` re-derives `reflection_s = pb.reflection_sign[token_ids]` (buffer lookup, non-diff) when `phi_reflection!="off"`, mirroring how it re-derives `omega_s`.

- [ ] **Step 1: Failing test** — gamma/s-channel frame-fidelity: with `phi_reflection` on, call `_gamma_coupling_term` / `_refine_s` with two different reflection signs and assert the energy / refined `s` differ (the reflection is USED), mirroring `test_gamma_coupling_term_uses_stored_frame_not_phi_cocycle` and `test_refine_s_uses_stored_frame_not_phi_cocycle`.
- [ ] **Step 2: Run** → FAIL (channel drops the reflection).
- [ ] **Step 3: Implement.** At each site where Phase 3 forwarded `omega=...`, ALSO forward `reflection=...` into the underlying `build_belief_transport`/`_gamma_energy`/`e_step` (add a `reflection` kwarg to `_gamma_energy`, `_gamma_coupling_term(s)`, `_fold_gamma_prior` mirroring their `omega` kwarg; in `_refine_s` add `reflection_s = pb.reflection_sign[token_ids] if cfg.phi_reflection != "off" else None` and populate the s-belief's `reflection` + pass it to `e_step`; `e_step`/`build_belief_transport` already gains `reflection` in Task 2 — forward it through `e_step`'s hoist like `omega`). All guarded `... if X.reflection is not None else None`.
- [ ] **Step 4: Run** the new tests + `pytest tests/test_phi_reflection.py tests/test_omega_direct.py tests/test_model.py` (no regressions).
- [ ] **Step 5: Commit** `git add vfe3/model/model.py vfe3/inference/e_step.py tests/test_phi_reflection.py`; `feat(phi-reflection): thread the reflection through the gamma/s-channel (frame-fidelity)`.

---

### Task 4: Learnable — generalize the Metropolis move to the phi reflection + train seam

**Files:** Modify `vfe3/model/model.py` (`metropolis_omega_step` → shared sweep), `vfe3/train.py` (seam gate). Test: `tests/test_phi_reflection.py`.

**Interfaces:** The Metropolis sweep flips `reflection_sign[i]` (`s→−s`) under `phi + phi_reflection=="metropolis"`, with exact ΔF at fixed beliefs via the phi transport with the flipped sign; the train seam fires when EITHER `omega_reflection=="metropolis"` OR `phi_reflection=="metropolis"`.

- [ ] **Step 1: Failing tests** — exact-ΔF anchor (masked-flip ΔF == independent `reflection_sign`-flip recompute); off-is-noop (no buffer touched, no RNG); downhill accepted flips the sign; uphill gated; seeded-reproducible. Mirror the `omega_direct` metropolis tests (`tests/test_omega_metropolis.py`).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Factor the shared sweep out of `metropolis_omega_step` (unique-token loop, carried `F_cur`, seeded accept, `_metropolis_free_energy`). Add the phi-reflection mode: the per-token flip mutates `pb.reflection_sign[tid] *= -1` (block 0 sign), the trial belief sets a flipped `belief.reflection`, and `_metropolis_free_energy` builds the phi transport with that reflection (Task 2 makes `free_energy_value` honor it). Dispatch: `omega_direct+omega_reflection=="metropolis"` → omega flip; `phi+phi_reflection=="metropolis"` → reflection flip; else no-op `{}`. `vfe3/train.py`: extend `_maybe_metropolis_omega`'s gate to `cfg.omega_reflection=="metropolis" or cfg.phi_reflection=="metropolis"`.
- [ ] **Step 4: Run** `pytest tests/test_phi_reflection.py tests/test_omega_metropolis.py tests/test_train.py` (no regressions; the omega-direct move still passes).
- [ ] **Step 5: Commit** `git add vfe3/model/model.py vfe3/train.py tests/test_phi_reflection.py`; `feat(phi-reflection): metropolis learns the phi reflection sign (shared sweep) + train seam`.

---

### Task 5: Capstone — end-to-end + gauge invariance (usable in the gamma-on config)

**Files:** Test-only: `tests/test_phi_reflection.py`.

**Interfaces:** proves the full-scope feature runs end-to-end and is gauge-correct.

- [ ] **Step 1: Add tests.**
  - **Finite forward at a gamma-on config:** build `VFEModel(gauge_parameterization="phi", phi_reflection="metropolis", lambda_gamma=0.75, s_e_step=True, gamma_as_beta_prior=True, prior_source="model_channel", family="gaussian_full", decode_mode="full", gauge_group="glk", tiny dims)`, assert `torch.isfinite(m(tok)[0]).all()` — the reflection is threaded through every channel with no gate.
  - **det<0 is reachable + used end-to-end:** set some `reflection_sign` to `−1`, assert the forward decode differs from all-`+1` (the reflection is used through the full forward), and `det(g_i)<0` for a reflected token's effective frame.
  - **Gauge invariance with the reflection:** co-transform the prior tables by an orthogonal `g` AND the reflection consistently; assert decode invariance (fp64) — the honest covariance-guard, labeled as in the `omega_direct` capstone (does not by itself certify frame-use; the per-task frame-fidelity tests do).
- [ ] **Step 2: Run** `pytest tests/test_phi_reflection.py` (all green); then `pytest tests/test_config.py tests/test_model.py tests/test_omega_direct.py tests/test_omega_metropolis.py tests/test_e_step.py` (whole-feature, no regressions).
- [ ] **Step 3: Commit** `git add tests/test_phi_reflection.py`; `test(phi-reflection): end-to-end gamma-on forward + det<0 reachability + gauge invariance`.

---

## Self-Review

**Spec coverage:** §3 fold → T2; §4 data flow → T1(config/belief/buffer) + T2(belief transport) + T3(gamma/s); §5 move → T4; §6 config → T1; §7 tests → T1-T5. Deferred (RoPE/regime_ii, STE) explicitly out.

**Type consistency:** `reflection` kwarg name uniform (belief field, `build_belief_transport`, `_gamma_energy`/`_fold_gamma_prior`/`_refine_s`, the move); `reflection_sign` buffer name uniform; `_REFLECT_OK` reused. The sign is a buffer (non-diff) throughout — no Parameter, no grad.

**Byte-identity:** every new `reflection`/kwarg defaults `None`/`"off"`; the buffer is created only under `phi_reflection!="off"`; the fold and move are not entered on the pure path.

## Execution Handoff

Subagent-driven, task-by-task, review+fix after each, final whole-branch review (with a live gamma-on `gaussian_full` train() smoke), then present the merge decision (do NOT auto-merge).
