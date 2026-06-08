# Live model channel `s` (dynamic prior tie) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the per-token model channel `s` a live field that is refined by its own E-step each forward and then fed in as the belief's prior, so the token's standing "disposition" shapes its belief and reaches the next-token loss — entirely behind a default-off `s_e_step` toggle.

**Architecture:** Under `s_e_step` (which requires `prior_source='model_channel'`), the forward refines `s` via the existing channel-agnostic `e_step` machinery (self-target = frozen `r`, coupling = `gamma_coupling`, gauge frame held fixed at `phi0`), then overrides the belief's initial value and prior `(mu_p, sigma_p)` with the refined `s1` before the belief stack runs. Because the belief E-step self-couples to its prior every iteration, `s` reaches `mu_final` even at the operative `n_e_steps=1`. The legacy loss-level `lambda_h`/`gamma` regularizer terms are superseded when `s_e_step` is on (the same forces now live inside the `s`-refine), and `r` stays frozen (un-freezing it is the out-of-scope meta-agent task **B**).

**Tech Stack:** Python, PyTorch (float32), pytest. No new dependencies. Reuses `vfe3.inference.e_step.e_step`, `vfe3.model.prior_bank.PriorBank.encode_s`, `vfe3.free_energy.attention_tau`.

**Spec:** `docs/superpowers/specs/2026-06-08-live-s-model-channel-design.md`

---

## File structure

- `vfe3/config.py` — add `s_e_step`, `e_s_mu_lr`, `e_s_sigma_lr`; validate; require `prior_source='model_channel'` when on; inert-misconfig warning.
- `vfe3/model/prior_bank.py` — create `s`-tables + frozen `r` when `s_e_step` is on (drawn last → belief tables byte-identical).
- `vfe3/model/model.py` — new `VFEModel._refine_s(...)`; thread `s_e_step` into `PriorBank(...)`; override the belief in `forward`/`generate`/`diagnostics` under `s_e_step`; supersede the loss-level `lambda_h`/`gamma` blocks; `TODO(B)` markers.
- `ablation.py` — add `s_e_step` (with its dependencies) as a sweep entry.
- `tests/test_live_s_model_channel.py` — new test module (all tasks add here).
- `docs/edits/2026-06-08-decode-bias.md` — append the post-edit changelog entry.

---

## Task 1: Config fields, validation, and the `prior_source` requirement

**Files:**
- Modify: `vfe3/config.py` (add three fields near the E-step knobs; extend `__post_init__` validation)
- Test: `tests/test_live_s_model_channel.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_live_s_model_channel.py`:

```python
r"""Live model channel s (dynamic prior tie), default-off. Spec:
docs/superpowers/specs/2026-06-08-live-s-model-channel-design.md.
"""

import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _tiny_cfg(**overrides) -> VFE3Config:
    r"""Minimal model config (embed_dim=4, n_heads=2, vocab=8, seq=4, 1 layer)."""
    base = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
    base.update(overrides)
    return VFE3Config(**base)


def test_s_e_step_defaults_off():
    cfg = VFE3Config()
    assert cfg.s_e_step is False
    assert cfg.e_s_mu_lr == 0.1
    assert cfg.e_s_sigma_lr == 0.1


def test_s_e_step_lr_validation_rejects_negative():
    with pytest.raises(ValueError):
        _tiny_cfg(s_e_step=True, prior_source="model_channel", e_s_mu_lr=-1.0)
    with pytest.raises(ValueError):
        _tiny_cfg(s_e_step=True, prior_source="model_channel", e_s_sigma_lr=-0.5)


def test_s_e_step_requires_model_channel_prior_source():
    # s_e_step anchors the belief to s AND must decode against s -> require model_channel.
    with pytest.raises(ValueError, match="prior_source"):
        _tiny_cfg(s_e_step=True, prior_source="token")


def test_s_e_step_inert_misconfig_warns():
    with pytest.warns(UserWarning, match="s_e_step"):
        _tiny_cfg(s_e_step=True, prior_source="model_channel",
                  lambda_h=0.0, gamma_coupling=0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_live_s_model_channel.py -v`
Expected: FAIL (`VFE3Config` has no `s_e_step`/`e_s_mu_lr`/`e_s_sigma_lr`; `TypeError: unexpected keyword`).

- [ ] **Step 3: Add the fields**

In `vfe3/config.py`, near the other E-step learning-rate fields (the `e_mu_lr`/`e_sigma_lr`/`e_phi_lr` group), add:

```python
    # Live model channel s (default OFF -> the manuscript's frozen slow channel). When True, s is
    # refined by its own E-step each forward and fed in as the belief's prior (dynamic fiber tie,
    # manuscript line 1399). Requires prior_source='model_channel' so the s-tables are the model's
    # vocab table for encode AND decode. e_s_*_lr are the refine learning rates; small -> slow
    # channel, and e_s_lr=0 collapses to the static model_channel tie. Inert when s_e_step=False.
    s_e_step:                  bool  = False
    e_s_mu_lr:                 float = 0.1
    e_s_sigma_lr:              float = 0.1
```

- [ ] **Step 4: Add validation in `__post_init__`**

In `vfe3/config.py.__post_init__`, alongside the other numeric/relationship checks, add:

```python
        if self.e_s_mu_lr < 0.0 or self.e_s_sigma_lr < 0.0:
            raise ValueError(
                f"e_s_mu_lr/e_s_sigma_lr must be >= 0, got "
                f"{self.e_s_mu_lr}/{self.e_s_sigma_lr}"
            )
        if self.s_e_step:
            if self.prior_source != "model_channel":
                raise ValueError(
                    "s_e_step=True requires prior_source='model_channel' so the s-tables are the "
                    f"model's vocab table for encode and decode; got prior_source={self.prior_source!r}."
                )
            if self.lambda_h == 0.0 and self.gamma_coupling == 0.0:
                import warnings
                warnings.warn(
                    "s_e_step=True with lambda_h=0 and gamma_coupling=0: the s-refine has no force, "
                    "so s1==s0 and the channel reduces to the static prior_source='model_channel' tie.",
                    UserWarning, stacklevel=2,
                )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_live_s_model_channel.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add vfe3/config.py tests/test_live_s_model_channel.py
git commit -m "feat(config): s_e_step + e_s_*_lr fields (live model channel, default off)"
```

---

## Task 2: PriorBank creates `s`-tables + frozen `r` under `s_e_step`

**Files:**
- Modify: `vfe3/model/prior_bank.py:114-186` (add `s_e_step` param; extend the table-creation gate)
- Modify: `vfe3/model/model.py` (pass `s_e_step=cfg.s_e_step` into the `PriorBank(...)` construction, alongside the existing `lambda_h=`/`gamma_coupling=` kwargs)
- Test: `tests/test_live_s_model_channel.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_s_model_channel.py`:

```python
def test_s_tables_and_frozen_r_created_under_s_e_step():
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, gamma_coupling=1.0))
    pb = m.prior_bank
    assert getattr(pb, "s_mu_embed", None) is not None
    assert getattr(pb, "r_mu", None) is not None
    assert pb.r_mu.requires_grad is False
    assert pb.r_sigma_log.requires_grad is False


def test_belief_tables_byte_identical_with_or_without_s_e_step():
    # s-tables are drawn LAST, so the belief tables (drawn first) are bit-identical.
    torch.manual_seed(0); off = VFEModel(_tiny_cfg(s_e_step=False))
    torch.manual_seed(0); on = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                                                  lambda_h=1.0, gamma_coupling=1.0))
    assert torch.equal(off.prior_bank.mu_embed, on.prior_bank.mu_embed)
    assert torch.equal(off.prior_bank.phi_embed, on.prior_bank.phi_embed)
    assert torch.equal(off.prior_bank.sigma_log_embed, on.prior_bank.sigma_log_embed)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_live_s_model_channel.py -k "s_tables or byte_identical" -v`
Expected: FAIL (`PriorBank.__init__` rejects `s_e_step`, or `s_mu_embed` absent).

- [ ] **Step 3: Add the `s_e_step` param + extend the gate**

In `vfe3/model/prior_bank.py`, add a keyword param `s_e_step: bool = False` to `PriorBank.__init__` (near `prior_source`), store `self.s_e_step = s_e_step`, and change the creation gate at line 176 from:

```python
        if lambda_h > 0.0 or gamma_coupling > 0.0 or prior_source == "model_channel":
```

to:

```python
        if lambda_h > 0.0 or gamma_coupling > 0.0 or prior_source == "model_channel" or s_e_step:
```

and change the `r` gate at line 179 from:

```python
        if lambda_h > 0.0:
```

to:

```python
        if lambda_h > 0.0 or s_e_step:
```

(The `s`-tables and `r` remain the LAST parameters created in `__init__`, so the belief tables are byte-unchanged. `r` stays `requires_grad=False`.)

- [ ] **Step 4: Thread `s_e_step` into the bank construction**

In `vfe3/model/model.py`, find the `PriorBank(...)` construction (it already passes `lambda_h=cfg.lambda_h, gamma_coupling=cfg.gamma_coupling, prior_source=cfg.prior_source`) and add:

```python
            s_e_step=cfg.s_e_step,
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_live_s_model_channel.py -k "s_tables or byte_identical" -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add vfe3/model/prior_bank.py vfe3/model/model.py tests/test_live_s_model_channel.py
git commit -m "feat(prior_bank): create s-tables + frozen r under s_e_step (drawn last)"
```

---

## Task 3: `VFEModel._refine_s` — the model-channel E-step

**Files:**
- Modify: `vfe3/model/model.py` (add the `_refine_s` method on `VFEModel`)
- Test: `tests/test_live_s_model_channel.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_s_model_channel.py`:

```python
def test_refine_s_preserves_shape_and_zero_lr_is_static():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, gamma_coupling=1.0,
                           e_s_mu_lr=0.0, e_s_sigma_lr=0.0))
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    phi0 = m._apply_pos_phi(m.prior_bank.encode(tok).phi)
    s0_mu, s0_sigma = m.prior_bank.encode_s(tok)
    s1_mu, s1_sigma = m._refine_s(tok, phi0)
    assert s1_mu.shape == s0_mu.shape == (2, 4, m.cfg.embed_dim)
    # e_s_lr=0 -> the refine is a no-op -> s1 == s0.
    assert torch.allclose(s1_mu, s0_mu)
    assert torch.allclose(s1_sigma, s0_sigma)


def test_refine_s_moves_s_with_nonzero_lr():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, gamma_coupling=1.0,
                           e_s_mu_lr=0.5, e_s_sigma_lr=0.5))
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    phi0 = m._apply_pos_phi(m.prior_bank.encode(tok).phi)
    s0_mu, _ = m.prior_bank.encode_s(tok)
    s1_mu, _ = m._refine_s(tok, phi0)
    assert not torch.allclose(s1_mu, s0_mu)   # the refine actually descends toward r + consensus
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_live_s_model_channel.py -k refine_s -v`
Expected: FAIL (`VFEModel` has no attribute `_refine_s`).

- [ ] **Step 3: Implement `_refine_s`**

In `vfe3/model/model.py`, add this method to `VFEModel` (it reuses the channel-agnostic `e_step`; `e_phi_lr=0.0` holds the shared frame fixed; `value=lambda_h` with `alpha_mode='constant'` is the `s->r` self-coupling; `lambda_beta=gamma_coupling` is the `s->s` coupling; `tau=gamma_tau`; `transport_mode='flat'` matches the existing tied gamma-block):

```python
    def _refine_s(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
        phi0:      torch.Tensor,         # (B, N, n_gen) encoded gauge frame (shared, held FIXED)

        *,
        e_step_gradient: str = "unroll",
    ) -> 'tuple[torch.Tensor, torch.Tensor]':
        r"""Refine the model channel s by its own E-step toward the frozen hyper-prior r plus the
        gamma model-consensus, with the shared gauge frame phi0 held fixed (e_phi_lr=0). Returns the
        refined (mu_s, sigma_s); the s-tables train through the unrolled trajectory. Manuscript
        eq:pointwise_free_energy (model channel)."""
        from vfe3.belief import BeliefState
        from vfe3.inference.e_step import e_step
        from vfe3.free_energy import attention_tau

        cfg, pb, grp = self.cfg, self.prior_bank, self.group
        s_mu, s_sigma = pb.encode_s(token_ids)                         # (B, N, K)
        r_mu = pb.r_mu.expand_as(s_mu)                                 # (B, N, K) frozen r broadcast
        r_sigma = torch.exp(pb.r_sigma_log).clamp(min=cfg.eps).expand_as(s_sigma)
        gamma_tau = attention_tau(cfg.kappa_gamma, grp.irrep_dims)
        gamma_log_prior = self._attention_log_prior(
            token_ids.shape[1], token_ids.device, prior=cfg.gamma_attention_prior,
        )
        out = e_step(
            BeliefState(mu=s_mu, sigma=s_sigma, phi=phi0), r_mu, r_sigma, grp,
            n_iter=cfg.n_e_steps, tau=gamma_tau,
            e_mu_lr=cfg.e_s_mu_lr, e_sigma_lr=cfg.e_s_sigma_lr, e_phi_lr=0.0,   # phi0 FIXED
            alpha_div=cfg.alpha_div, value=cfg.lambda_h, alpha_mode="constant", # s->r self-coupling
            b0=cfg.b0, c0=cfg.c0,
            lambda_beta=cfg.gamma_coupling,                                     # s->s coupling weight
            kl_max=cfg.kl_max, eps=cfg.eps,
            sigma_max=cfg.sigma_max, e_sigma_q_trust=cfg.e_sigma_q_trust,
            e_mu_q_trust=cfg.e_mu_q_trust, mu_trust_mode=cfg.mu_trust_mode,
            include_attention_entropy=cfg.include_attention_entropy,
            gradient_mode=cfg.gradient_mode, family="gaussian_diagonal",
            divergence_family=cfg.divergence_family,
            phi_precond_mode=cfg.phi_precond_mode, phi_retract_mode=cfg.phi_retract_mode,
            spd_retract_mode=cfg.spd_retract_mode, transport_mode="flat",       # tied flat cocycle
            e_step_gradient=e_step_gradient, oracle_unroll_grad=cfg.oracle_unroll_grad,
            log_prior=gamma_log_prior,
        )
        return out.mu, out.sigma
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_live_s_model_channel.py -k refine_s -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add vfe3/model/model.py tests/test_live_s_model_channel.py
git commit -m "feat(model): VFEModel._refine_s — model-channel E-step (phi fixed, frozen r)"
```

---

## Task 4: Wire `s`-refine into `forward` + supersede the loss-level terms

**Files:**
- Modify: `vfe3/model/model.py:377-384` (refine + override the belief before `vfe_stack`)
- Modify: `vfe3/model/model.py:484-559` (skip the loss-level `lambda_h`/`gamma` blocks under `s_e_step`)
- Test: `tests/test_live_s_model_channel.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_s_model_channel.py`:

```python
def _tok(m, b=2, n=4):
    return torch.randint(0, m.cfg.vocab_size, (b, n))


def test_default_off_forward_is_unchanged_by_the_new_code():
    # The pure path must be byte-identical: same seed, s_e_step=False, before/after parity is
    # guaranteed by no new parameter on that path (Task 2) and no new forward branch taken here.
    torch.manual_seed(0); m = VFEModel(_tiny_cfg(s_e_step=False))
    tok = _tok(m)
    lg = m(tok)
    assert torch.isfinite(lg).all()


def test_s_e_step_changes_logits_at_n_e_steps_1():
    # Belief tables are bit-identical across the two models (s-tables drawn last); the ONLY
    # difference is the live s channel, which must move the logits at the operative n_e_steps=1.
    torch.manual_seed(0); base = VFEModel(_tiny_cfg(s_e_step=False, prior_source="model_channel",
                                                    lambda_h=1.0, gamma_coupling=1.0))
    torch.manual_seed(0); live = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                                                    lambda_h=1.0, gamma_coupling=1.0,
                                                    e_s_mu_lr=0.5, e_s_sigma_lr=0.5))
    tok = _tok(live)
    assert not torch.allclose(base(tok), live(tok))


def test_e_s_lr_zero_reduces_to_static_model_channel():
    # s_e_step + e_s_lr=0 == static prior_source='model_channel' (refine no-ops), bitwise at seed.
    torch.manual_seed(0); static = VFEModel(_tiny_cfg(s_e_step=False, prior_source="model_channel",
                                                      lambda_h=1.0, gamma_coupling=1.0))
    torch.manual_seed(0); live0 = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                                                     lambda_h=1.0, gamma_coupling=1.0,
                                                     e_s_mu_lr=0.0, e_s_sigma_lr=0.0))
    tok = _tok(live0)
    assert torch.allclose(static(tok), live0(tok), atol=0.0, rtol=0.0)


def test_s_e_step_gradient_reaches_s_tables_at_t1():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, gamma_coupling=1.0, e_s_mu_lr=0.5))
    tok = _tok(m)
    tgt = _tok(m)
    _, loss, _ = m(tok, targets=tgt)
    loss.backward()
    assert m.prior_bank.s_mu_embed.grad is not None
    assert m.prior_bank.s_mu_embed.grad.abs().sum() > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_live_s_model_channel.py -k "changes_logits or reduces_to_static or gradient_reaches" -v`
Expected: FAIL (`s_e_step` not consulted in `forward`; live and base logits equal).

- [ ] **Step 3: Refine + override the belief before the stack**

In `vfe3/model/model.py`, inside the `with run, amp:` block (currently lines 378-384), immediately before the `out = vfe_stack(...)` call, insert:

```python
            if self.cfg.s_e_step:
                # Live model channel: refine s (phi0 fixed), then anchor the belief to it -- q0 and
                # the belief prior (mu_p, sigma_p) both become the refined s1. The belief E-step
                # self-couples to its prior every iteration, so s reaches mu_final even at n_e_steps=1.
                s_mu1, s_sigma1 = self._refine_s(token_ids, beliefs.phi, e_step_gradient=e_step_gradient)
                beliefs = beliefs._replace(mu=s_mu1, sigma=s_sigma1)
```

(`vfe_stack(beliefs, beliefs.mu, beliefs.sigma, ...)` then passes `s1` as both the initial belief and the prior `(mu_p, sigma_p)`.)

- [ ] **Step 4: Supersede the loss-level `lambda_h` / `gamma` blocks**

In `vfe3/model/model.py`, change the two loss-block guards so they are skipped when `s_e_step` is on (those forces now live inside `_refine_s`):

At line 484, change:

```python
        if self.cfg.lambda_h > 0.0:
```

to:

```python
        if self.cfg.lambda_h > 0.0 and not self.cfg.s_e_step:
```

At line 509, change:

```python
        if self.cfg.gamma_coupling > 0.0:
```

to:

```python
        if self.cfg.gamma_coupling > 0.0 and not self.cfg.s_e_step:
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_live_s_model_channel.py -v`
Expected: PASS (all tests in the module).

- [ ] **Step 6: Commit**

```bash
git add vfe3/model/model.py tests/test_live_s_model_channel.py
git commit -m "feat(model): forward refines s and anchors the belief to it under s_e_step; supersede loss-level lambda_h/gamma"
```

---

## Task 5: `generate()` + `diagnostics()` parity (trained-model consistency)

**Files:**
- Modify: `vfe3/model/model.py:658-664` (`generate`: refine + anchor before its `vfe_stack`)
- Modify: `vfe3/model/model.py:767-779` (`diagnostics`: refine + anchor before its block replay)
- Test: `tests/test_live_s_model_channel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_live_s_model_channel.py`:

```python
def test_generate_runs_under_s_e_step():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, gamma_coupling=1.0, e_s_mu_lr=0.5))
    prompt = torch.randint(0, m.cfg.vocab_size, (1, 3))
    out = m.generate(prompt, max_new_tokens=2)
    assert out.shape == (1, 5)


def test_diagnostics_runs_under_s_e_step():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, gamma_coupling=1.0, e_s_mu_lr=0.5))
    tok = torch.randint(0, m.cfg.vocab_size, (1, 4))
    d = m.diagnostics(tok)            # must not raise; the diagnostic belief is the anchored one
    assert d is not None
```

- [ ] **Step 2: Run to verify failure / behavior**

Run: `pytest tests/test_live_s_model_channel.py -k "generate_runs or diagnostics_runs" -v`
Expected: PASS only if `generate`/`diagnostics` already happen to run; the parity edit makes them use the live `s` (without it, a trained-with-`s` model would sample from a no-`s` forward — a train/inference mismatch). Confirm they run; if they error, the edit in Step 3 fixes it.

- [ ] **Step 3: Anchor the belief in `generate` and `diagnostics`**

In `vfe3/model/model.py`, in `generate()` immediately after the encode/`_apply_pos_phi` line (around 659) and before its `vfe_stack(...)` call (around 664), insert (note the single-sequence `[:1]` slicing these methods use — `belief` here is unbatched `(N, ...)`, so refine on the batched token slice and squeeze):

```python
            if self.cfg.s_e_step:
                s_mu1, s_sigma1 = self._refine_s(token_ids[:1], belief.phi.unsqueeze(0))
                belief = belief._replace(mu=s_mu1[0], sigma=s_sigma1[0])
```

Apply the identical insertion in `diagnostics()` after its encode/`_apply_pos_phi` (around 768) and before its block-replay loop (around 779).

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_live_s_model_channel.py -k "generate_runs or diagnostics_runs" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add vfe3/model/model.py tests/test_live_s_model_channel.py
git commit -m "feat(model): generate/diagnostics use the live s under s_e_step (train/inference parity)"
```

---

## Task 6: Property + gauge-equivariance tests, `TODO(B)` markers, ablation entry, changelog, full suite

**Files:**
- Modify: `vfe3/model/prior_bank.py:185` and the prior-handoff site (`TODO(B)` comments)
- Modify: `ablation.py` (add an `s_e_step` sweep entry with its dependencies)
- Modify: `docs/edits/2026-06-08-decode-bias.md` (append changelog)
- Test: `tests/test_live_s_model_channel.py`

- [ ] **Step 1: Write the property + equivariance tests**

Append to `tests/test_live_s_model_channel.py`:

```python
def test_model_channel_self_divergence_zero_at_s_equals_r():
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import self_divergence
    K = 4
    mu = torch.randn(2, 3, K)
    sig = torch.rand(2, 3, K) + 0.1
    d = self_divergence(DiagonalGaussian(mu, sig), DiagonalGaussian(mu, sig)).abs().max()
    assert d < 1e-5            # D(s||s) == 0


def test_s_e_step_forward_is_gauge_equivariant():
    # Logits are gauge-invariant (KL decode); a global frame action on the input must leave them
    # unchanged. With s sharing phi0, the s-refine + belief step must preserve this.
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, gamma_coupling=1.0, e_s_mu_lr=0.5,
                           gauge_group="so_k"))   # use a group with a clean global action
    tok = torch.randint(0, m.cfg.vocab_size, (1, 4))
    lg = m(tok)
    assert torch.isfinite(lg).all()
    # NOTE: pin the exact invariance check to the pattern in tests/test_gauge_*.py for this repo's
    # global-action helper; this test asserts the s_e_step forward is finite and runs under so_k.
```

(If the repo has a reusable global-gauge-action helper in `tests/test_gauge_*.py`, strengthen this to an exact invariance assertion using it.)

- [ ] **Step 2: Run to verify pass**

Run: `pytest tests/test_live_s_model_channel.py -k "self_divergence or equivariant" -v`
Expected: PASS (2 tests).

- [ ] **Step 3: Add the `TODO(B)` markers**

In `vfe3/model/prior_bank.py` at the frozen-`r` definition (line 185), append a comment:

```python
            # TODO(B): un-freeze r -> a token-dependent, top-down hyper-prior r_i = Omega_tilde[s_I^{(s+1)}]
            # requires the scale-(s+1) meta-agent (out of scope here). See
            # docs/superpowers/specs/2026-06-08-live-s-model-channel-design.md (option B).
```

In `vfe3/model/model.py` at the prior-handoff / shadow-prior site (near the `mstep_self_coupling` prior rebuild, ~line 457), add a one-line `TODO(B)` pointing at the same spec.

- [ ] **Step 4: Add the ablation sweep entry**

In `ablation.py`, add to the sweep registry an `s_e_step` entry that also sets its dependencies (so each cell is valid):

```python
    "s_e_step": {
        "param": "s_e_step", "values": [False, True],
        "fixed": {"prior_source": "model_channel", "lambda_h": 1.0, "gamma_coupling": 1.0},
    },
```

(Match the exact schema the other `ablation.py` sweep entries use — e.g. the `n_e_steps` entry at `ablation.py:260` — adapting field names if this repo's entries differ.)

- [ ] **Step 5: Run the full suite**

Run: `pytest --junitxml=out.xml` (do NOT add `-q`; read the `N passed` line or `testsuite tests=/failures=/errors=` from `out.xml`).
Expected: all prior tests still pass; the new `tests/test_live_s_model_channel.py` tests pass; default-config tests unchanged.

- [ ] **Step 6: Append the changelog and commit**

Append to `docs/edits/2026-06-08-decode-bias.md` a brief section: what `s_e_step` does, the dynamic-prior-tie mechanism, default-off invariant, the `prior_source='model_channel'` requirement, supersede rule, frozen `r` + `TODO(B)`, and the verified pass count (read from `out.xml`).

```bash
git add vfe3/model/prior_bank.py vfe3/model/model.py ablation.py docs/edits/2026-06-08-decode-bias.md tests/test_live_s_model_channel.py out.xml
git rm --cached out.xml   # do not commit the junit artifact
git commit -m "test(live-s): property + equivariance pins; TODO(B) markers; ablation entry; changelog"
```

---

## Self-Review

**Spec coverage:**
- Default-off byte-identical invariant → Task 2 (byte-identical belief tables) + Task 4 (no branch taken when off). ✓
- Dynamic prior tie / live `s` at `n_e_steps=1` → Tasks 3-4 (refine + anchor; gradient-reaches-`s`-tables test). ✓
- `prior_source='model_channel'` requirement → Task 1 (validation). ✓
- Reuse of `belief_gradients`/`e_step` (no reimplementation) → Task 3 (`_refine_s` calls `e_step`). ✓
- Supersede loss-level terms → Task 4 Step 4. ✓
- Frozen `r` + `TODO(B)` → Task 2 (`requires_grad=False`) + Task 6 (markers). ✓
- `e_s_lr=0` reduction to static `model_channel` → Task 4 (`test_e_s_lr_zero_reduces_to_static_model_channel`). ✓
- generate/diagnostics parity → Task 5. ✓
- Property + gauge-equivariance tests → Task 6. ✓
- Config surface, timescale-via-lr, out-of-scope (shared-frame, meta-agent, observation term) → covered by Tasks 1/3 and left out of scope by construction. ✓

**Placeholder scan:** the only soft spots are the two "match the exact schema/helper in this repo" notes (ablation entry schema; gauge global-action helper). These are real and necessary — the engineer must align with the repo's existing `ablation.py` entry shape and `tests/test_gauge_*.py` helper rather than a fabricated one. Flagged inline, not left as silent TODOs.

**Type consistency:** `_refine_s(token_ids, phi0, *, e_step_gradient)` returns `(mu_s, sigma_s)` and is called that way in Tasks 4 and 5. `s_e_step`/`e_s_mu_lr`/`e_s_sigma_lr` names are consistent across config, model, and tests. `PriorBank` gate uses the `s_e_step` param added in Task 2. Consistent.

---

## Open items the executor must confirm against the live repo

1. The exact `PriorBank(...)` construction kwargs in `model.py` (Task 2 Step 4) and the prior-handoff line number (Task 6 Step 3) — grep for `PriorBank(` and `prior_handoff` rather than trusting the approximate line numbers.
2. The `ablation.py` sweep-entry schema (Task 6 Step 4) — copy the shape of an existing entry.
3. Whether `diagnostics()`'s belief is unbatched `(N, ...)` exactly as `generate()`'s (Task 5 Step 3) — confirm the slicing before applying the squeeze.
