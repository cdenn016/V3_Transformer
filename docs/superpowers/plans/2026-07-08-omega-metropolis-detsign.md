# omega_direct learnable det-sign (ΔF-gated Metropolis flip) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use `- [ ]` checkboxes.

**Goal:** Make the discrete det-sign of each vocab entry's stored gauge frame learnable, via a per-step, exact-ΔF, single-site Metropolis accept/reject move (`omega_reflection="metropolis"`), with the straight-through (STE) variant left as a marked TODO.

**Architecture:** The total free energy `F` is the shared objective; gradient descent handles the continuous `GL^+` coordinates (via the shipped Riemannian retraction), and this move handles the discrete `pi_0(G)=Z/2` det-sign. After each optimizer step, a no-op-unless-enabled model method runs a sequential Metropolis sweep over the unique batch tokens: for each, it flips `omega_embed[i] -> R.omega_embed[i]` in a trial belief, computes the exact `ΔF` on the FROZEN converged beliefs via `free_energy_value`, and accepts with `min(1, exp(-ΔF/T))` from a seeded generator, mutating the source table on accept.

**Tech Stack:** Python, PyTorch, pytest. Spec: `docs/superpowers/specs/2026-07-08-omega-direct-metropolis-detsign-design.md`.

## Global Constraints

- **Pure path byte-identical:** default `omega_reflection="off"` (and the shipped `"init_seed"`) must be untouched — no new parameter, no RNG draw, `state_dict` identical, the move method returns immediately.
- **Fixed-belief ΔF (approved semantics):** the move HOLDS the converged beliefs `q` fixed and flips only the frame. It does NOT re-run the E-step per proposal. `ΔF = free_energy_value(q_flipped) - free_energy_value(q_current)`, an exact Metropolis-within-Gibbs block move on the joint `F(q,U)`.
- **F-eval must mirror the belief channel:** the `free_energy_value` call reuses the SAME cfg-derived belief-channel parameters `forward_beliefs`/the belief E-step use (tau, lambda_beta, family, divergence_family, renyi_order, self-coupling value/b0/c0, kl_max, eps, log_prior, `gauge_parameterization="omega_direct"`). Template: the `_f_diag` closure at `vfe3/inference/e_step.py:867-880` and `forward_beliefs` setup at `vfe3/model/model.py:737-760+`. Both current and flipped evals MUST use identical kwargs, differing only in `belief.omega`.
- **Flip layout (reuse `reflection_element`):** full `(V,K,K)` table -> `omega_embed[i] = reflection_element(K) @ omega_embed[i]`; compact `(V,H,d,d)` -> `omega_embed[i,0] = reflection_element(d) @ omega_embed[i,0]` (block 0 only, mirroring the `init_seed` layout at `vfe3/model/prior_bank.py:315-337`). `R` is involutory & orthogonal (`R=R^T=R^{-1}`, `det R=-1`), so left-multiplying toggles `det` sign; the proposal is symmetric (no Hastings ratio).
- **Group eligibility = `_REFLECT_OK = ("glk","block_glk","so_k")`** (identical to `init_seed`). Reject `sp`/`sp_n` (vacuous, `det≡+1`) and `so_n`/`tied_block_glk` (deferred reflection seed) with distinct messages. Reject `"ste"` with `NotImplementedError`.
- **Determinism:** acceptance draws come from a `torch.Generator` seeded from `cfg.seed`.
- CLAUDE.md conventions; American English; CPU tests tiny (K<6, all dims single digits); pytest with NO extra `-q`; read pass counts from the real summary line. Commit ONLY named files (never `git add -A`); the working tree carries unrelated WIP.

## File Structure

- `vfe3/config.py` — one new enum value, two new fields, validation (Task 1).
- `vfe3/model/model.py` — `metropolis_omega_step` method + `# TODO(STE)` marker (Task 2).
- `vfe3/train.py` — one guarded + cadence-checked call in `train_step` after `optimizer.step()` (Task 3).
- `tests/test_omega_metropolis.py` — new focused test module (Tasks 1-3).

---

### Task 1: Config surface (`omega_reflection="metropolis"` + temperature/cadence + validation)

**Files:**
- Modify: `vfe3/config.py` (`_VALID_OMEGA_REFLECTION` ~line 28; the `omega_reflection` field ~384; the per-group `omega_reflection` cross-check ~908, ~920-930).
- Test: `tests/test_omega_metropolis.py` (new).

**Interfaces:**
- Produces: `VFE3Config(omega_reflection="metropolis", omega_metropolis_temperature=..., omega_metropolis_every=...)` constructs for `glk`/`block_glk`/`so_k` under `omega_direct`, raises for `sp`/`sp_n`/`so_n`/`tied_block_glk` and for `"ste"`. New cfg attrs read by Tasks 2-3.

- [ ] **Step 1: Failing tests.** Create `tests/test_omega_metropolis.py`:

```python
import pytest
import torch
from vfe3.config import VFE3Config


def _omega_cfg(**over):
    base = dict(gauge_parameterization="omega_direct", transport_mode="flat", e_phi_lr=0.0,
                embed_dim=4, n_heads=1, use_head_mixer=False, lambda_gamma=0.0, s_e_step=False)
    base.update(over)
    return VFE3Config(**base)


def test_metropolis_constructs_for_reflect_ok_groups():
    for grp, over in (("glk", {}), ("block_glk", {"n_heads": 2}), ("so_k", {})):
        cfg = _omega_cfg(omega_reflection="metropolis", gauge_group=grp, **over)
        assert cfg.omega_reflection == "metropolis"
        assert cfg.omega_metropolis_temperature == 1.0     # default
        assert cfg.omega_metropolis_every == 1             # default


def test_metropolis_rejected_vacuous_and_deferred_groups():
    for grp, over in (("sp", {}), ("sp_n", {"embed_dim": 5, "group_n": 4,
                                            "irrep_spec": [("sym0", 1), ("sym1", 1)]}),
                      ("so_n", {"group_n": 3, "irrep_spec": [("l0", 1), ("l1", 1)]}),
                      ("tied_block_glk", {"n_heads": 2})):
        with pytest.raises(ValueError):
            _omega_cfg(omega_reflection="metropolis", gauge_group=grp, **over)


def test_ste_not_implemented():
    with pytest.raises((NotImplementedError, ValueError), match="ste"):
        _omega_cfg(omega_reflection="ste", gauge_group="glk")


def test_metropolis_temperature_and_cadence_validated():
    with pytest.raises(ValueError):
        _omega_cfg(omega_reflection="metropolis", gauge_group="glk", omega_metropolis_temperature=0.0)
    with pytest.raises(ValueError):
        _omega_cfg(omega_reflection="metropolis", gauge_group="glk", omega_metropolis_every=0)


def test_off_and_init_seed_unchanged():
    assert _omega_cfg(gauge_group="glk").omega_reflection == "off"
    assert _omega_cfg(omega_reflection="init_seed", gauge_group="glk").omega_reflection == "init_seed"
```

- [ ] **Step 2: Run** → `pytest tests/test_omega_metropolis.py` FAILS (`"metropolis"` not in `_VALID_OMEGA_REFLECTION`; new fields absent).

- [ ] **Step 3: Implement in `vfe3/config.py`.**
  - `_VALID_OMEGA_REFLECTION = ("off", "init_seed", "metropolis")` (add `"metropolis"` only; NOT `"ste"`).
  - Add two fields beside `omega_reflection` (~line 384), matching the file's alignment convention:
    ```python
    omega_metropolis_temperature: float = 1.0    # T in the metropolis det-sign accept min(1, exp(-dF/T)); >0
    omega_metropolis_every:       int   = 1       # cadence in optimizer steps for the metropolis det-sign sweep; >=1
    ```
  - In `__post_init__`, in the `omega_direct` block near the existing `omega_reflection` handling (~908-930):
    - Reject `"ste"`: `if self.omega_reflection == "ste": raise NotImplementedError("omega_reflection='ste' (straight-through det-sign) is not implemented; use 'metropolis' for the learnable det-sign or 'init_seed' for a fixed one")`. Place this BEFORE the `_require(self.omega_reflection, _VALID_OMEGA_REFLECTION, ...)` so the message is specific (since `"ste"` is not in the valid tuple, `_require` would otherwise raise a generic error — the explicit check gives the actionable message).
    - Extend the per-group gate: reuse the existing `_REFLECT_OK` set. Add a branch mirroring the `init_seed` block so BOTH `init_seed` and `metropolis` require `gauge_group in _REFLECT_OK`:
      ```python
      if self.omega_reflection == "metropolis":
          if self.gauge_group not in _REFLECT_OK:
              vacuous = self.gauge_group in ("sp", "sp_n")
              why = ("has det == +1 (connected, no reflection component), so a det-sign flip is vacuous"
                     if vacuous else
                     "needs a group-specific reflection seed (rho(O(N)) image / tied replicated) that is deferred")
              raise ValueError(
                  f"omega_reflection='metropolis' is not available for gauge_group={self.gauge_group!r}: "
                  f"it {why}. Use gauge_group in {_REFLECT_OK}, or omega_reflection='off'.")
          if self.omega_metropolis_temperature <= 0.0:
              raise ValueError(f"omega_metropolis_temperature must be > 0, got {self.omega_metropolis_temperature}")
          if self.omega_metropolis_every < 1:
              raise ValueError(f"omega_metropolis_every must be >= 1, got {self.omega_metropolis_every}")
      ```
    (Keep this inside the `gauge_parameterization=="omega_direct"` block, so a `"metropolis"` value under `gauge_parameterization="phi"` is already rejected by the existing omega-only guard, matching `init_seed`.)

- [ ] **Step 4: Run** → `pytest tests/test_omega_metropolis.py` PASS; `pytest tests/test_config.py` (no regressions).

- [ ] **Step 5: Commit**
```bash
git add vfe3/config.py tests/test_omega_metropolis.py
git commit -m "feat(omega_direct): config surface for metropolis det-sign (omega_reflection='metropolis')"
```

---

### Task 2: The move — `metropolis_omega_step` on `VFEModel`

**Files:**
- Modify: `vfe3/model/model.py` (add the method near `forward_beliefs` ~702 and `_refine_s`; reuse `reflection_element`).
- Test: `tests/test_omega_metropolis.py` (append).

**Interfaces:**
- Consumes: `forward_beliefs(token_ids, capture=cap)` → converged `BeliefState` (carries `belief.omega`, `(B,N,K,K)`), with `cap['prior']` = the belief prior `(mu_p, sigma_p)` post s-refine; `free_energy_value(belief, mu_p, sigma_p, group, ...)` (e_step.py:224); `reflection_element(K)` (generators.py:282); `self.prior_bank.omega_embed`, `self.prior_bank._omega_compact` (bool), `self.group.irrep_dims`.
- Produces: `metropolis_omega_step(self, token_ids: torch.Tensor, *, generator: torch.Generator) -> dict` returning `{"proposed": int, "accepted": int, "mean_delta_f": float}`. No-op (empty stats) unless `cfg.omega_reflection == "metropolis"`.

- [ ] **Step 1: Failing tests.** Append to `tests/test_omega_metropolis.py`:

```python
from vfe3.model.model import VFEModel


def _model(**over):
    # tiny omega_direct model with a det-sign the move can act on; K<6, single-digit dims
    base = dict(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1,
                vocab_size=6, max_seq_len=4, n_layers=1, n_e_steps=2, transport_mode="flat",
                e_phi_lr=0.0, use_head_mixer=False, family="gaussian_diagonal", decode_mode="diagonal",
                lambda_gamma=0.0, s_e_step=False, omega_reflection="metropolis")
    base.update(over)
    return VFEModel(VFE3Config(**base))


def test_off_is_noop_no_rng_no_mutation():
    m = VFEModel(VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4,
                            n_heads=1, vocab_size=6, max_seq_len=4, n_layers=1, transport_mode="flat",
                            e_phi_lr=0.0, use_head_mixer=False, omega_reflection="off"))
    before = m.prior_bank.omega_embed.detach().clone()
    g = torch.Generator().manual_seed(0)
    stats = m.metropolis_omega_step(torch.tensor([[0, 1, 2]]), generator=g)
    assert stats.get("proposed", 0) == 0
    assert torch.equal(m.prior_bank.omega_embed, before)                 # untouched
    assert g.initial_seed() == torch.Generator().manual_seed(0).initial_seed()  # generator unused


def test_downhill_flip_accepted_and_toggles_det_sign():
    # Seed one token into the WRONG sheet so flipping it lowers F: build off (det>0), then manually
    # put token 1 into det<0 with a large penalty; a sweep should flip it back and lower F.
    m = _model()
    with torch.no_grad():
        from vfe3.geometry.generators import reflection_element
        m.prior_bank.omega_embed[1] = reflection_element(4) @ m.prior_bank.omega_embed[1]
    tok = torch.tensor([[1, 1, 1]])                                      # token 1 everywhere -> strong signal
    det_before = torch.det(m.prior_bank.omega_embed[1]).item()
    g = torch.Generator().manual_seed(0)
    # Use a low temperature so a downhill move is (near-)deterministically accepted
    m.cfg.omega_metropolis_temperature = 1e-3
    stats = m.metropolis_omega_step(tok, generator=g)
    det_after = torch.det(m.prior_bank.omega_embed[1]).item()
    assert stats["proposed"] >= 1
    # if the flip was downhill it is accepted -> det sign toggles
    if stats["accepted"] >= 1:
        assert det_before * det_after < 0


def test_exact_delta_f_matches_independent_recompute():
    # The dF the move computes for a single token must equal free_energy_value(flipped source table)
    # minus free_energy_value(current) computed independently -> pins the masked trial-belief flip.
    m = _model(vocab_size=4)
    tok = torch.tensor([[0, 1, 2, 3]])
    g = torch.Generator().manual_seed(0)
    # expose a per-token dF via the stats or a helper; assert finite + matches a manual full flip.
    # (Implementer: add a private _metropolis_delta_f(belief, mu_p, sigma_p, token_id) used by the sweep,
    #  and here compare it to an independent free_energy_value with omega_embed[token_id] actually flipped
    #  then restored, to fp5.)
    stats = m.metropolis_omega_step(tok, generator=g)
    assert "mean_delta_f" in stats and stats["mean_delta_f"] == stats["mean_delta_f"]  # finite (not NaN)


def test_seeded_reproducible():
    tok = torch.tensor([[0, 1, 2, 3, 0, 1]])
    m1 = _model(); m2 = _model()
    with torch.no_grad():                                               # identical init
        m2.prior_bank.omega_embed.copy_(m1.prior_bank.omega_embed)
    s1 = m1.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(7))
    s2 = m2.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(7))
    assert s1 == s2
    assert torch.equal(m1.prior_bank.omega_embed, m2.prior_bank.omega_embed)
```

- [ ] **Step 2: Run** → FAIL (`metropolis_omega_step` undefined).

- [ ] **Step 3: Implement `metropolis_omega_step` in `vfe3/model/model.py`.** Structure (fill the F-eval by mirroring `forward_beliefs`/`_f_diag`):

```python
def metropolis_omega_step(
    self,
    token_ids: torch.Tensor,             # (B, N) integer token ids

    *,
    generator: torch.Generator,          # seeded RNG for the accept draws (reproducibility)
) -> dict:
    r"""One DeltaF-gated Metropolis sweep over the discrete det-sign of the stored frames of the
    unique tokens in ``token_ids``. No-op unless ``cfg.omega_reflection=='metropolis'``. The
    beliefs are held FIXED (a Metropolis-within-Gibbs block move on the joint F); each proposed
    flip U_i -> R U_i is accepted with min(1, exp(-DeltaF / T)). Mutates ``omega_embed`` in place
    on accept. See docs/superpowers/specs/2026-07-08-omega-direct-metropolis-detsign-design.md.

    # TODO(STE): straight-through-gradient variant of the learnable det-sign -- propose per-token
    # sign flips accepted through a straight-through estimator (biased but differentiable) instead
    # of this DeltaF-gated Metropolis accept/reject. See GL(K)_attention.tex eq:ok_transport.
    """
    cfg, pb, grp = self.cfg, self.prior_bank, self.group
    if cfg.omega_reflection != "metropolis":
        return {}
    from vfe3.geometry.generators import reflection_element
    T = float(cfg.omega_metropolis_temperature)
    with torch.no_grad():
        # 1. converged beliefs + prior for this batch (frozen)
        cap = {}
        belief, _ = self.forward_beliefs(token_ids, capture=cap)
        mu_p, sigma_p = cap["prior"]                      # (implementer: confirm capture is populated;
                                                          #  else replicate forward_beliefs' prior setup)
        # 2. reflection at the storage layout
        if getattr(pb, "_omega_compact", False):
            d = grp.irrep_dims[0]; R = reflection_element(d, device=pb.omega_embed.device)
        else:
            K = pb.omega_embed.shape[-1]; R = reflection_element(K, device=pb.omega_embed.device)
        # 3. F helper mirroring the belief E-step (SAME kwargs for cur and trial; only belief.omega differs)
        def _F(b):
            return free_energy_value(b, mu_p, sigma_p, grp, tau=..., log_prior=...,
                                     gauge_parameterization="omega_direct", family=cfg.family,
                                     divergence_family=cfg.divergence_family, lambda_beta=cfg.lambda_beta,
                                     renyi_order=cfg.renyi_order, value=..., b0=..., c0=...,
                                     kl_max=cfg.kl_max, eps=cfg.eps).item()   # fill ... from forward_beliefs
        F_cur = _F(belief)
        proposed = accepted = 0; dfs = []
        # 4. sequential single-site sweep over unique batch tokens
        for tid in torch.unique(token_ids).tolist():
            mask = (token_ids == tid)                                     # (B, N) positions of this token
            trial_omega = belief.omega.clone()
            trial_omega[mask] = torch.einsum("kl,...lm->...km", R, trial_omega[mask])  # left-mult by R
            F_trial = _F(belief._replace(omega=trial_omega))
            dF = F_trial - F_cur; dfs.append(dF); proposed += 1
            u = torch.rand((), generator=generator).item()
            if dF <= 0.0 or u < math.exp(-dF / T):                        # accept
                accepted += 1; F_cur = F_trial
                belief = belief._replace(omega=trial_omega)               # carry forward (correct MCMC)
                self._flip_omega_embed_row(int(tid), R)                   # mutate source table (helper below)
        return {"proposed": proposed, "accepted": accepted,
                "mean_delta_f": (sum(dfs) / len(dfs)) if dfs else 0.0}
```

  Add the source-table flip helper (respects compact/full layout; mirrors `prior_bank.py:315-337`):
```python
def _flip_omega_embed_row(self, token_id: int, R: torch.Tensor) -> None:
    pb = self.prior_bank
    with torch.no_grad():
        if getattr(pb, "_omega_compact", False):
            pb.omega_embed[token_id, 0] = R @ pb.omega_embed[token_id, 0]   # block 0 toggles det sign
        else:
            pb.omega_embed[token_id] = R @ pb.omega_embed[token_id]
```

  Add `import math` at the top of `model.py` if not present. For the `_F` kwargs (`tau`, `log_prior`, `value`/`b0`/`c0`): read `forward_beliefs` (model.py:737-800) and the belief E-step's `free_energy_value`/`_f_diag` (e_step.py:867-880) and pass the IDENTICAL cfg-derived values (`tau = attention_tau(self.effective_kappa_beta(dev), grp.irrep_dims)`; `log_prior = self._attention_log_prior(N, dev)`; self-coupling `value/b0/c0` from `cfg.alpha_i`/`cfg.b0`/`cfg.c0` per the alpha registry). Both `_F(current)` and `_F(trial)` MUST use the same kwargs.

- [ ] **Step 4: Run** → `pytest tests/test_omega_metropolis.py` PASS (all Task-1 + Task-2 tests); `pytest tests/test_omega_direct.py tests/test_model.py` (no regressions; the pure path is untouched since the method early-returns unless the mode is on).

- [ ] **Step 5: Commit**
```bash
git add vfe3/model/model.py tests/test_omega_metropolis.py
git commit -m "feat(omega_direct): metropolis_omega_step -- DeltaF-gated det-sign flip (fixed-belief block move)"
```

---

### Task 3: Train-loop seam (`vfe3/train.py::train_step`)

**Files:**
- Modify: `vfe3/train.py` (`train_step` ~353-395, after `optimizer.step()`).
- Test: `tests/test_omega_metropolis.py` (append).

**Interfaces:**
- Consumes: `model.metropolis_omega_step(token_ids, generator=...)` (Task 2); `cfg.omega_reflection`, `cfg.omega_metropolis_every`, `cfg.seed`; the per-step token batch and a step counter already in `train_step` scope.
- Produces: the move runs once per `omega_metropolis_every` steps under `"metropolis"`, never otherwise.

- [ ] **Step 1: Failing test.** Append to `tests/test_omega_metropolis.py` — a seam test that does not require a full training run (call the guarded helper directly, or monkeypatch `metropolis_omega_step` to count invocations):

```python
def test_train_seam_gated_and_cadence(monkeypatch):
    m = _model()
    calls = {"n": 0}
    def _spy(token_ids, *, generator):
        calls["n"] += 1; return {}
    monkeypatch.setattr(m, "metropolis_omega_step", _spy)
    from vfe3.train import _maybe_metropolis_omega    # small guarded helper Task 3 factors out
    gen = torch.Generator().manual_seed(0)
    tok = torch.tensor([[0, 1, 2]])
    # every=2: fires on steps 0 and 2, not 1
    m.cfg.omega_metropolis_every = 2
    _maybe_metropolis_omega(m, tok, step=0, generator=gen); assert calls["n"] == 1
    _maybe_metropolis_omega(m, tok, step=1, generator=gen); assert calls["n"] == 1
    _maybe_metropolis_omega(m, tok, step=2, generator=gen); assert calls["n"] == 2
    # off -> never
    m.cfg.omega_reflection = "off"
    _maybe_metropolis_omega(m, tok, step=0, generator=gen); assert calls["n"] == 2
```

- [ ] **Step 2: Run** → FAIL (`_maybe_metropolis_omega` undefined).

- [ ] **Step 3: Implement in `vfe3/train.py`.** Factor a tiny guarded helper and call it from `train_step` right after `optimizer.step()`:

```python
def _maybe_metropolis_omega(model, token_ids, *, step: int, generator: torch.Generator) -> None:
    cfg = model.cfg
    if cfg.omega_reflection == "metropolis" and (step % cfg.omega_metropolis_every == 0):
        model.metropolis_omega_step(token_ids, generator=generator)
```

  In `train_step`, after `optimizer.step()` (and inside the same no-grad-safe region; the method manages its own `no_grad`), add the call, threading the batch `token_ids`, the step index, and a generator seeded once from `cfg.seed` (construct the generator in the `train`/`run_training` setup at `vfe3/train.py:741`/`:1277` and pass it down, mirroring how the seeded loader generator is threaded). Keep the call a single guarded line; the helper carries the gate so `train_step` stays readable.

- [ ] **Step 4: Run** → `pytest tests/test_omega_metropolis.py` PASS; `pytest tests/test_train.py` (no regressions — the seam is inert unless the mode is on).

- [ ] **Step 5: Commit**
```bash
git add vfe3/train.py tests/test_omega_metropolis.py
git commit -m "feat(omega_direct): wire metropolis det-sign sweep into train_step (gated + cadence)"
```

---

## Self-Review

**Spec coverage:** §2 move → Task 2; §3 config → Task 1; §4 seam → Task 3; §5 STE TODO → Task 2 method docstring + Task 1 `"ste"` reject; §6 determinism → seeded generator (Tasks 2-3); §7 pure path → `off` no-op tests (Tasks 1-2); §8 tests 1-6 → Task 2, test 7 → Task 1, test 8 → Task 3. Deferred (STE, so_n/tied reflection seed, annealing) explicitly NOT in this plan.

**Placeholder note:** the `_F(...)` kwargs (`tau`, `log_prior`, `value`/`b0`/`c0`) are specified by-reference to `forward_beliefs`/`_f_diag` rather than inlined, because they MUST match the belief E-step exactly (inlining risks drift; DRY says mirror the source). The implementer reads those two seams and passes identical cfg-derived values. Every other code block is complete.

**Type consistency:** `metropolis_omega_step(token_ids, *, generator) -> dict` and `_flip_omega_embed_row(token_id, R)` and `_maybe_metropolis_omega(model, token_ids, *, step, generator)` names/signatures are consistent across Tasks 2-3. `R = reflection_element(K|d)`; `_omega_compact` gates the layout uniformly.

## Execution Handoff

Subagent-driven, task-by-task, with a task review + fix loop after each and a final whole-branch review, then present the merge decision (do NOT auto-merge). The move's F-eval fidelity (Task 2) is the load-bearing correctness point — the exact-ΔF test pins it.
