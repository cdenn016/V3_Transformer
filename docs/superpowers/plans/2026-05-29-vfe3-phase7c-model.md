# VFE_3.0 Phase 7c (Block + Stack + Model assembly) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:test-driven-development (RED→GREEN→COMMIT; **commit after every GREEN task**). V3-internal tests. No VFE_2.0 provenance.

**Goal:** Assemble the full forward model — `vfe3/model/block.py` (one VFE block = E-step iterations + optional norm), `vfe3/model/stack.py` (L blocks with the belief handoff `μ_q→μ_p`), `vfe3/model/model.py` (`VFEModel`: encode → stack inference → decode → cross-entropy). The load-bearing property: the **unrolled E-step keeps the training graph connected**, so `loss.backward()` delivers gradient to the encode/φ prior tables.

**Architecture:** Phase 7c (spec §4.6, §5). Data flow: `token_ids (B,N)` → `PriorBank.encode` → `BeliefState` (q=p) → for each batch element, `vfe_stack` runs `L` blocks (each `n_e_steps` E-step iterations via Phase-6 `e_step`) with the prior handoff between blocks → final belief → `PriorBank.decode` → logits `(B,N,V)` → CE vs `targets`. Training **unrolls** the E-step (the Phase-4 differentiable filtering kernel keeps `μ_new = μ − lr·natgrad(kernel(μ))` in the graph; CE backprops through every iteration to the priors). `config.detach_e_step=True` wraps inference in `no_grad` for the fixed-point/truncated alternative.

**Batching:** the Phase-4/6 gradient + E-step are unbatched `(N,K)`; the model **loops over the batch** around the stack (correct; true batched kernels are a deferred perf optimization). Decode + CE are batched.

**No-NN:** `VFEModel` is an `nn.Module` whose only parameters are the `PriorBank` tables (priors). Block/stack are parameter-free pure-VFE orchestration (plain functions). Zero `nn.Linear`/MLP/activations.

**Tech Stack:** Python 3, PyTorch (float32), pytest. No NN layers. No CLI. Device-agnostic.

**Reference spec:** §4.6, §5. Prereq: Phases 0–7b on `phase7-model`. Reuses `vfe3.inference.e_step` (`e_step`), `vfe3.model.prior_bank` (`PriorBank`), `vfe3.geometry.norms` (`get_norm`), `vfe3.geometry.groups` (`get_group`), `vfe3.attention_prior` (`attention_log_prior`), `vfe3.belief.BeliefState`, `vfe3.config.VFE3Config`.

**Design decisions (do not relitigate):**
1. **Unrolled training via the kernel.** The model default uses `gradient_mode='filtering'` (differentiable kernel) so the E-step iterations stay in the training graph; `loss.backward()` reaches `phi_embed`/`mu_embed`/`sigma_log_embed`. The oracle (`smoothing`) detaches and is validation-only; `detach_e_step` is the explicit fixed-point toggle.
2. **Belief handoff** (between blocks): `μ_p_next = (1−ρ)·μ_p + ρ·belief.μ` (`ρ=prior_handoff_rho`, 1.0=full flow); σ_p frozen at the embedding (default) or `prior_handoff_sigma`-damped; φ flows via the belief (not the prior).
3. **Group from config** via a small builder dispatch (`glk(K)`, `block_glk(K,n_heads)`, `so_k(K)`); built once, held on the model.
4. **Initial belief = the prior** (`q=p` at encode); the initial prior for block 0 is the encoded belief.

---

## Code Style / Provenance (MANDATORY)

Repo CLAUDE.md conventions. No "VFE_2.0"/"2.0"/"ported" in any shipped artifact; cite manuscript + math.

---

## File Structure

- **Create** `vfe3/model/block.py` — `vfe_block`.
- **Create** `vfe3/model/stack.py` — `vfe_stack`.
- **Create** `vfe3/model/model.py` — `VFEModel`, `build_group`.
- **Create** `tests/test_model.py`.

---

## Task 1 — `vfe_block` + `vfe_stack`

**Files:** `vfe3/model/block.py`, `vfe3/model/stack.py`; `tests/test_model.py`.

- [ ] **RED:** `tests/test_model.py`:

```python
import torch
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.model.block import vfe_block
from vfe3.model.stack import vfe_stack


def _belief(N=4, K=4, n_gen=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return BeliefState(
        mu=torch.randn(N, K, generator=g),
        sigma=torch.rand(N, K, generator=g) + 0.5,
        phi=0.1 * torch.randn(N, n_gen, generator=g),
    )


def test_block_runs_e_step_and_preserves_shapes():
    cfg = VFE3Config(embed_dim=4, n_heads=2, n_e_steps=2, e_mu_lr=0.05, e_phi_lr=0.0)
    grp = get_group("block_glk")(4, 2)
    b = _belief(K=4, n_gen=grp.generators.shape[0])
    out = vfe_block(b, b.mu, b.sigma, grp, cfg)
    assert out.mu.shape == b.mu.shape and (out.sigma > 0).all()


def test_stack_handoff_updates_prior_across_blocks():
    cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=3, n_e_steps=1,
                     e_mu_lr=0.05, e_phi_lr=0.0, prior_handoff_rho=1.0)
    grp = get_group("block_glk")(4, 2)
    b = _belief(K=4, n_gen=grp.generators.shape[0])
    out = vfe_stack(b, b.mu, b.sigma, grp, cfg)
    assert out.mu.shape == b.mu.shape and (out.sigma > 0).all()
    # with rho=1 and a nonzero E-step, the stack moves the belief off the input
    assert not torch.allclose(out.mu, b.mu, atol=1e-4)
```

- [ ] **GREEN:** `vfe3/model/block.py`:

```python
r"""A single VFE block for VFE_3.0: E-step belief inference + optional norm.

Parameter-free: all learnable capacity is the PriorBank's prior tables; the block
runs the iterative E-step (Phase 6) and an optional gauge-equivariant norm on the
mean. The belief handoff across blocks lives in stack.py.
"""

from typing import Optional

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.norms import get_norm
from vfe3.inference.e_step import e_step


def vfe_block(
    belief:    BeliefState,
    mu_p:      torch.Tensor,             # (N, K) prior means
    sigma_p:   torch.Tensor,             # (N, K) prior variances
    group:     GaugeGroup,
    cfg:       VFE3Config,

    *,
    log_prior: Optional[torch.Tensor] = None,
) -> BeliefState:
    r"""Run n_e_steps of the E-step from ``belief`` toward the prior, then optional norm."""
    out = e_step(
        belief, mu_p, sigma_p, group,
        n_iter=cfg.n_e_steps, tau=cfg.tau,
        e_mu_lr=cfg.e_mu_lr, e_sigma_lr=cfg.e_sigma_lr, e_phi_lr=cfg.e_phi_lr,
        alpha_div=cfg.alpha_div, value=cfg.alpha, kl_max=cfg.kl_max, eps=cfg.eps,
        sigma_max=cfg.sigma_max, e_sigma_q_trust=cfg.e_sigma_q_trust,
        include_attention_entropy=cfg.include_attention_entropy,
        gradient_mode=cfg.gradient_mode, family=cfg.family, alpha_mode=cfg.alpha_mode,
        phi_precond_mode=cfg.phi_precond_mode, log_prior=log_prior,
    )
    if cfg.norm_type_block != "none":
        norm = get_norm(cfg.norm_type_block)(cfg.embed_dim, eps=cfg.eps)
        out = BeliefState(mu=norm(out.mu, out.sigma), sigma=out.sigma, phi=out.phi)
    return out
```

`vfe3/model/stack.py`:

```python
r"""The VFE block stack for VFE_3.0: L blocks with the belief handoff mu_q -> mu_p.

After each block the updated belief becomes (a blend toward) the next block's prior:
mu_p_next = (1 - rho) mu_p + rho mu_q (rho = prior_handoff_rho); sigma_p frozen at the
embedding by default; phi flows through the belief, not the prior.
"""

from typing import Optional

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup
from vfe3.model.block import vfe_block


def vfe_stack(
    belief:    BeliefState,
    mu_p:      torch.Tensor,             # (N, K) initial prior means
    sigma_p:   torch.Tensor,             # (N, K) initial prior variances
    group:     GaugeGroup,
    cfg:       VFE3Config,

    *,
    log_prior: Optional[torch.Tensor] = None,
) -> BeliefState:
    r"""Run L = cfg.n_layers blocks, handing the belief mean off to the next prior."""
    rho = cfg.prior_handoff_rho
    rho_s = cfg.prior_handoff_sigma
    for _ in range(cfg.n_layers):
        belief = vfe_block(belief, mu_p, sigma_p, group, cfg, log_prior=log_prior)
        mu_p = (1.0 - rho) * mu_p + rho * belief.mu
        sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
    return belief
```

- [ ] Run → 2 passed. **COMMIT** `feat(model): vfe_block (E-step + norm) + vfe_stack (belief handoff)`.

---

## Task 2 — `VFEModel` (encode → stack → decode → CE)

**Files:** `vfe3/model/model.py`; `tests/test_model.py`.

- [ ] **RED:** append:

```python
from vfe3.model.model import VFEModel


def test_model_forward_shapes_and_loss():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=2,
                     n_e_steps=1, e_mu_lr=0.05, e_phi_lr=0.0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    logits = model(tokens)
    assert logits.shape == (3, 5, 20)
    targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = model(tokens, targets)
    assert loss.shape == () and torch.isfinite(loss)


def test_loss_backward_reaches_prior_tables():
    # THE crown jewel: the unrolled E-step keeps the training graph connected, so the
    # M-step gradient reaches the encode/phi prior parameters (not just decode).
    cfg = VFE3Config(vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                     n_e_steps=2, e_mu_lr=0.05, e_phi_lr=0.02, gradient_mode="filtering")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 15, (2, 4)); targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert model.prior_bank.phi_embed.grad is not None
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0          # gradient actually flows
    assert model.prior_bank.phi_embed.grad.abs().sum() > 0


def test_model_has_no_nn_layers():
    import torch.nn as nn
    cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3)
    model = VFEModel(cfg)
    for m in model.modules():
        assert not isinstance(m, (nn.Linear, nn.MultiheadAttention, nn.RNNBase, nn.Conv1d))
```

- [ ] **GREEN:** `vfe3/model/model.py`. Loop over batch around the stack (E-step unbatched); decode + CE batched; CE backprops through the unrolled E-step:

```python
r"""The full VFE_3.0 model: encode -> E-step inference -> decode -> cross-entropy.

No neural layers: the only parameters are the PriorBank's prior tables. The E-step
is unrolled into the training graph (the differentiable filtering kernel), so the CE
loss backpropagates through inference to the encode/phi priors. Batching loops over
the batch around the (unbatched) E-step; decode and CE are batched.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vfe3.attention_prior import attention_log_prior
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.model.prior_bank import PriorBank
from vfe3.model.stack import vfe_stack


def build_group(cfg: VFE3Config) -> GaugeGroup:
    r"""Construct the gauge group from config (dispatch on the builder signature)."""
    builder = get_group(cfg.gauge_group)
    if cfg.gauge_group == "block_glk":
        return builder(cfg.embed_dim, cfg.n_heads)
    return builder(cfg.embed_dim)


class VFEModel(nn.Module):
    """encode -> E-step stack -> decode -> CE. Parameters live only in the PriorBank."""

    def __init__(self, cfg: VFE3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.group = build_group(cfg)
        n_gen = self.group.generators.shape[0]
        self.prior_bank = PriorBank(
            cfg.vocab_size, cfg.embed_dim, n_gen,
            decode_tau=cfg.decode_tau, eps=cfg.eps,
        )

    def forward(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
        targets:   Optional[torch.Tensor] = None,   # (B, N) next-token ids (-100 = ignore)
    ) -> 'torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
        r"""Forward pass; returns logits, or (logits, loss, ce) when targets are given."""
        B, N = token_ids.shape
        beliefs = self.prior_bank.encode(token_ids)              # (B, N, K) ...
        log_prior = attention_log_prior(
            self.cfg.attention_prior, N, N, device=token_ids.device,
        )

        outs = []
        run = torch.no_grad() if self.cfg.detach_e_step else _nullcontext()
        with run:
            for b in range(B):
                belief_b = BeliefState(mu=beliefs.mu[b], sigma=beliefs.sigma[b], phi=beliefs.phi[b])
                out_b = vfe_stack(belief_b, belief_b.mu, belief_b.sigma, self.group, self.cfg, log_prior=log_prior)
                outs.append(out_b)
        mu_final = torch.stack([o.mu for o in outs], dim=0)      # (B, N, K)
        sigma_final = torch.stack([o.sigma for o in outs], dim=0)

        logits = self.prior_bank.decode(mu_final, sigma_final)   # (B, N, V)
        if targets is None:
            return logits

        ce = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=-100)
        loss = ce
        if self.cfg.mass_phi > 0.0:
            phi_all = torch.stack([o.phi for o in outs], dim=0)
            loss = loss + 0.5 * self.cfg.mass_phi * (phi_all ** 2).mean()
        return logits, loss, ce.detach()


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False
```

> If `detach_e_step=True` the whole stack runs under `no_grad`, so the encoded belief is severed where it enters the loop and `mu_final` reaches the priors ONLY through `decode`'s direct read of `mu_embed`/`sigma_log_embed` (encode is NOT a live gradient source, and `phi_embed` is frozen — phi appears only in encode and the phi step, never in decode). The phi natural-gradient step still runs (its own `enable_grad` island, so the blanket `no_grad` does not crash it), but it contributes no gradient to the priors in this regime. For the default (`detach_e_step=False`) the loop runs under grad and the unrolled graph reaches mu/sigma/phi priors via encode + the E-step + decode. (The GROUND-TRUTH parenthetical "decode+encode" is imprecise for the fixed-point regime: encode is severed by `no_grad`; only decode is a live path.)

- [ ] Run → 3 passed (esp. `test_loss_backward_reaches_prior_tables`). **COMMIT** `feat(model): VFEModel (encode -> unrolled E-step -> decode -> CE), grad reaches priors`.

---

## Task 3 — full suite + changelog + commit

- [ ] `python -m pytest -q` — expect prior 138 + new (~5) green.
- [ ] Append "## Phase 7c Block + Stack + Model — 2026-05-29 (continuation)" to the changelog (provenance-clean). Commit `docs(edits): 2026-05-29 phase 7c model changes log`.

---

## Self-Review

**Coverage:** block (E-step + norm) → T1; stack (handoff) → T1; model (encode→stack→decode→CE) → T2; the unrolled-graph M-step reaching the priors → T2 (crown jewel); no-NN → T2.
**Anchors:** `loss.backward()` populates `mu_embed.grad`/`phi_embed.grad` with nonzero mass (unrolled E-step connects the graph); no `nn.Linear`/MLP/activation modules; stack moves the belief; shapes.
**Modularity:** group/decode/encode/norm/attention-prior/gradient-mode all config-selected; `detach_e_step` toggles unroll vs fixed-point.
**Deferred (named):** true batched E-step kernels (perf); positional φ (BCH); head mixer; full-cov path; the data loader (7d) and training loop (7e).
