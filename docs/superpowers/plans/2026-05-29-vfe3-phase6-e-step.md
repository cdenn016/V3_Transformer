# VFE_3.0 Phase 6 (E-step: iterative belief update) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development (RED→GREEN→COMMIT). Tests are V3-internal (descent-direction + invariant + fixed-seed regression). V3 is self-contained; see the Provenance rule.

**Goal:** Build the E-step — the iterative belief-update loop that wires the free energy (Phase 3), the gradient layer (Phase 4), and the geometry retraction/preconditioner (Phases 2c–2e) into one descent on F over the Gaussian belief `(mu, sigma, phi)`. Decoupled learning rates (`e_mu_lr`, `e_sigma_lr`, `e_phi_lr`) and trust-region clamps. Correctness is pinned by **descent-direction** properties (the right objective per gradient mode) and a fixed-seed regression — not by asserting parallel-update monotonicity (which is false).

**Architecture:** Phase 6 (spec §4.5). One inner iteration, all positions in parallel (Jacobi), updates sequential within the iteration:
1. transport `Ω_ij = exp(φ_i)exp(−φ_j)` from the current `φ`;
2. `(grad_mu, grad_sigma)` from `gradients.belief_gradients` (Phase-4 envelope kernel for filtering+diagonal+KL+canonical, else the autograd oracle) — NOT a hand-rolled `∂β/∂μ` form (the manuscript's Algorithm-1 `Σ_j D ∂β/∂μ` term is the pre-envelope display; it drops at the canonical stationary β, Phase-3 gradient-gap = 0);
3. Fisher preconditioner `natural_gradient` → `(nat_mu, nat_sigma)`;
4. retract: `mu ← mu − e_mu_lr·nat_mu` (Euclidean + optional μ trust region); `sigma ← retract_spd_diagonal(sigma, −e_sigma_lr·nat_sigma)` (SPD retraction, keeps σ>0);
5. φ: `grad_phi = autograd(alignment_loss, φ)` where the alignment loss is the **canonical belief-coupling block** `Σ_ij[β_ij E_ij + τ β_ij log(β_ij/π_ij)]` evaluated at the just-updated `(mu, sigma)`; precondition (`precondition_phi_gradient`); `phi ← retract_phi(phi, −grad_phi, group, step_size=e_phi_lr)`.

The belief is a `BeliefState(mu, sigma, phi)` NamedTuple. Unbatched `(N, K)` beliefs (matching the Phase-4 gradient layer); batching is a Phase-7 concern. Deferred: Regime II (`connection_delta`), RoPE-on-μ, the head-mixer, and the M-step (prior/parameter learning) — named extension points.

**Descent objective per gradient mode (the crux — both verified):**
- `gradient_mode='filtering'` (query-side, mean-field default): the step descends **`F_filt`** — F with the keys frozen at their pre-step values (the key-detached objective, the same construction as the Phase-4 filtering oracle). It does NOT descend global F (updating a belief moves F through its key columns too, an omitted term). `F_filt(belief_after) < F_filt(belief_before)`.
- `gradient_mode='smoothing'` (full ∇F): the step descends **global F** (`F(after) < F(before)`), the Fisher natural gradient being a descent direction `⟨∇F, −G⁻¹∇F⟩<0` (G PD).
- The φ-step (μ,σ frozen) descends **global F** (the alignment loss is the full coupling block, both roles of φ via every `Ω_ij`).
- A **parallel** (all-position) filtering update is NOT guaranteed monotone in global F per iteration (Jacobi mean-field) — the E-step exposes the F-trajectory as a **diagnostic**, never asserted for parallel filtering.

**Tech Stack:** Python 3, PyTorch (float32), pytest. No NN. No CLI. Device-agnostic.

**Reference spec:** §4.5 + §8 Phase 6. Prereq: Phases 0–4 on `main` (branch `phase6-e-step`). Reuses `free_energy` (`pairwise_energy`, `self_divergence`, `attention_weights`, `free_energy`), `alpha_i` (`self_coupling_alpha`), `gradients.belief_gradients`, `geometry.retraction` (`natural_gradient`, `retract_spd_diagonal`, `retract_phi`), `geometry.phi_preconditioner` (`precondition_phi_gradient`), `geometry.transport` (`compute_transport_operators`, `transport_mean`, `transport_covariance`), `geometry.groups`.

**Manuscript theory:** Algorithm-1 E-step (`GL(K)_attention.tex` ~ll. 2033–2080): per inner iteration recompute transport + β, then natural-gradient μ update (`nat_mu = Σ ∇_μ F`), SPD-retraction σ update, Lie-retraction φ update; F-monotonicity holds for sequential coordinate ascent, NOT per-iteration for parallel mean-field (`Jordan 1999`, `Beal 2003`).

**Design decisions settled (do not relitigate):**
1. The E-step uses `gradients.belief_gradients` (envelope kernel / oracle), NOT a hand-rolled ∂β form. `gradient_mode` (`filtering` default / `smoothing`) flows straight through.
2. Descent tests are **tiny-step directional checks** with trust-region / σ-clamp **inactive** (so first-order descent isn't masked), each **isolated per block** (φ off for μσ-tests via `e_phi_lr=0`; μσ off for the φ-test). The filtering test compares `F_filt` (keys frozen); the smoothing and φ tests compare global F.
3. φ-alignment loss = the canonical belief-coupling block (with the entropy term) at the updated `(mu,sigma)`; autograd through β is the envelope φ-gradient.
4. `BeliefState(mu, sigma, phi)`; prior is `(mu_p, sigma_p)` (no φ). Unbatched `(N,K)`.

---

## Code Style (MANDATORY — repo CLAUDE.md)

Tensors first; then `float|Tensor`; undefined; defined scalars; `Optional`; `**kwargs`. Vertical alignment of names/types/`=`/trailing-`#`; type hints; docstrings carry the LaTeX/math; shape comments. Names match notation (`mu`, `sigma`, `phi`, `mu_p`, `e_mu_lr`, …).

## Provenance (MANDATORY — convention as of commit `114839c`)

No shipped artifact (docstring/comment/test name/test comment) may contain "VFE_2.0", "2.0", or "ported". Cite the manuscript + math only.

---

## File Structure

- **Create** `vfe3/belief.py` — `BeliefState` NamedTuple.
- **Create** `vfe3/inference/__init__.py` (empty), `vfe3/inference/e_step.py` — `free_energy_value`, `phi_alignment_loss`, `e_step_iteration`, `e_step`.
- **Create** `tests/test_e_step.py`.

---

## Task 1 — `BeliefState` + `free_energy_value`

**Files:** Create `vfe3/belief.py`, `vfe3/inference/__init__.py`, `vfe3/inference/e_step.py`; Test `tests/test_e_step.py`.

- [ ] **Step 1 (RED):** create `tests/test_e_step.py`:

```python
import torch

from vfe3.belief import BeliefState
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import free_energy_value


def _belief(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    n_gen = grp.generators.shape[0]
    b = BeliefState(
        mu=torch.randn(N, K, generator=g),
        sigma=torch.rand(N, K, generator=g) + 0.5,
        phi=0.1 * torch.randn(N, n_gen, generator=g),
    )
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    return b, mu_p, sigma_p, grp


def test_belief_state_fields():
    b, *_ = _belief()
    assert b.mu.shape == (3, 2) and b.sigma.shape == (3, 2)


def test_free_energy_value_is_finite_scalar():
    b, mu_p, sigma_p, grp = _belief()
    F = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5)
    assert F.shape == () and torch.isfinite(F)


def test_free_energy_filtering_equals_global_at_a_point():
    # F_filt and global F are the SAME NUMBER at a fixed belief (detach changes
    # gradients, not the value); they differ only as functions under a step.
    b, mu_p, sigma_p, grp = _belief()
    Fg = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5, keys=None)
    Ff = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5, keys=b)   # keys frozen at b
    assert torch.allclose(Fg, Ff, atol=1e-6)
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3 (GREEN):** create `vfe3/belief.py`:

```python
r"""The Gaussian belief tuple for VFE_3.0."""

from typing import NamedTuple

import torch


class BeliefState(NamedTuple):
    """A per-token Gaussian belief q_i = N(mu_i, Sigma_i) with gauge frame phi_i."""

    mu:    torch.Tensor             # (..., N, K) means
    sigma: torch.Tensor             # (..., N, K) diagonal variances (or (..., N, K, K) full)
    phi:   torch.Tensor             # (..., N, n_gen) gauge-frame coordinates
```

Create `vfe3/inference/__init__.py` (empty) and `vfe3/inference/e_step.py` with the header + `free_energy_value`:

```python
r"""The E-step for VFE_3.0: an iterative natural-gradient descent on F over the
Gaussian belief (mu, sigma, phi).

One inner iteration (all positions in parallel, updates sequential):
  transport Omega(phi) -> belief_gradients (envelope kernel / oracle) -> Fisher
  preconditioner -> retract mu (Euclidean) + sigma (SPD) -> phi (autograd of the
  canonical belief-coupling block -> precondition -> Lie retraction).
Decoupled learning rates and trust regions. Parallel mean-field updates are not
guaranteed monotone per iteration; F-descent holds as a DIRECTION property
(filtering descends F_filt; smoothing and the phi step descend global F).
"""

from typing import List, Optional, Tuple

import torch

from vfe3.alpha_i import self_coupling_alpha
from vfe3.belief import BeliefState
from vfe3.free_energy import attention_weights, free_energy, pairwise_energy, self_divergence
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.phi_preconditioner import precondition_phi_gradient
from vfe3.geometry.retraction import natural_gradient, retract_phi, retract_spd_diagonal
from vfe3.geometry.transport import compute_transport_operators, transport_covariance, transport_mean
from vfe3.gradients.kernels import belief_gradients


def _transport(
    phi:   torch.Tensor,             # (N, n_gen)
    group: GaugeGroup,
) -> torch.Tensor:                   # (N, N, K, K) Omega_ij
    r"""Build the pairwise transport Omega_ij = exp(phi_i) exp(-phi_j)."""
    return compute_transport_operators(phi.unsqueeze(0), group)["Omega"][0]


def free_energy_value(
    belief:                    BeliefState,
    mu_p:                      torch.Tensor,        # (N, K) prior means
    sigma_p:                   torch.Tensor,        # (N, K) prior variances
    group:                     GaugeGroup,

    *,
    tau:                       float = 1.0,
    alpha_div:                 float = 1.0,
    value:                     float = 1.0,
    b0:                        float = 1.0,
    c0:                        float = 1.0,
    kl_max:                    float = 100.0,
    eps:                       float = 1e-6,

    include_attention_entropy: bool = True,
    family:                    str  = "gaussian_diagonal",
    alpha_mode:                str  = "constant",

    log_prior:                 Optional[torch.Tensor] = None,
    keys:                      Optional[BeliefState]  = None,   # None -> global F; else keys frozen at `keys`
) -> torch.Tensor:                   # scalar F
    r"""Scalar free energy of a belief. ``keys=None`` -> global F (keys = the belief);
    ``keys`` given -> F with the transported keys frozen at ``keys`` (the F_filt objective)."""
    key_belief = belief if keys is None else keys
    omega = _transport(key_belief.phi, group)
    mu_t = transport_mean(omega.unsqueeze(0), key_belief.mu.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), key_belief.sigma.unsqueeze(0))[0]

    sd = self_divergence(belief.mu, belief.sigma, mu_p, sigma_p, alpha=alpha_div, kl_max=kl_max, eps=eps, family=family)
    alpha, reg = self_coupling_alpha(sd, value=value, mode=alpha_mode, b0=b0, c0=c0)
    energy = pairwise_energy(belief.mu, belief.sigma, mu_t, sigma_t, alpha=alpha_div, kl_max=kl_max, eps=eps, family=family)
    return free_energy(
        sd, energy, alpha, tau=tau, include_attention_entropy=include_attention_entropy,
        log_prior=log_prior, alpha_reg=(reg if alpha_mode != "constant" else None),
    )
```

> Note: in the `keys` (F_filt) form, only the *query/self* role uses `belief`; the transported second argument uses `key_belief` (frozen). At `keys=belief` the two coincide (global F). The value is identical at a fixed point; the objectives differ only under a step.

- [ ] **Step 4:** Run — expect 3 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/belief.py vfe3/inference/__init__.py vfe3/inference/e_step.py tests/test_e_step.py
git commit -m "feat(inference): BeliefState + free_energy_value (global / keys-frozen F_filt)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2 — `phi_alignment_loss` + `e_step_iteration`

**Files:** Modify `vfe3/inference/e_step.py`; Test `tests/test_e_step.py`.

- [ ] **Step 1 (RED):** append tests:

```python
from vfe3.inference.e_step import e_step_iteration


def test_iteration_keeps_sigma_positive_and_shapes():
    b, mu_p, sigma_p, grp = _belief()
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5,
                           e_mu_lr=0.05, e_sigma_lr=0.05, e_phi_lr=0.05)
    assert (out.sigma > 0).all()
    assert out.mu.shape == b.mu.shape and out.phi.shape == b.phi.shape


def test_decoupled_learning_rates_freeze_components():
    b, mu_p, sigma_p, grp = _belief()
    # e_phi_lr=0 -> phi unchanged
    o1 = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=0.05, e_sigma_lr=0.05, e_phi_lr=0.0)
    assert torch.allclose(o1.phi, b.phi, atol=1e-7)
    # e_mu_lr=0 -> mu unchanged
    o2 = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=0.0, e_sigma_lr=0.05, e_phi_lr=0.0)
    assert torch.allclose(o2.mu, b.mu, atol=1e-7)
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3 (GREEN):** append to `vfe3/inference/e_step.py`:

```python
def phi_alignment_loss(
    mu:        torch.Tensor,             # (N, K)
    sigma:     torch.Tensor,             # (N, K)
    phi:       torch.Tensor,             # (N, n_gen) -- the differentiated variable
    group:     GaugeGroup,

    *,
    tau:       float = 1.0,
    alpha_div: float = 1.0,
    kl_max:    float = 100.0,
    eps:       float = 1e-6,
    family:    str   = "gaussian_diagonal",

    include_attention_entropy: bool = True,
    log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Canonical belief-coupling block as a function of phi (mu, sigma fixed):

        L(phi) = Sum_ij [ beta_ij E_ij + tau beta_ij log(beta_ij/pi_ij) ],
        E_ij = D(q_i || Omega_ij(phi) q_j),  beta = softmax_j(log_prior - E/tau).
    Both roles of phi flow (Omega_ij depends on phi_i and phi_j); autograd gives the
    envelope phi-gradient.
    """
    omega = _transport(phi, group)
    mu_t = transport_mean(omega.unsqueeze(0), mu.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), sigma.unsqueeze(0))[0]
    energy = pairwise_energy(mu, sigma, mu_t, sigma_t, alpha=alpha_div, kl_max=kl_max, eps=eps, family=family)
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)
    L = (beta * energy).sum()
    if include_attention_entropy:
        pi = torch.softmax(log_prior, dim=-1) if log_prior is not None else torch.full_like(beta, 1.0 / beta.shape[-1])
        L = L + tau * (beta * (torch.log(beta.clamp(min=1e-12)) - torch.log(pi.clamp(min=1e-12)))).sum()
    return L


def e_step_iteration(
    belief:                    BeliefState,
    mu_p:                      torch.Tensor,        # (N, K)
    sigma_p:                   torch.Tensor,        # (N, K)
    group:                     GaugeGroup,

    *,
    tau:                       float = 1.0,
    e_mu_lr:                   float = 0.1,
    e_sigma_lr:                float = 0.1,
    e_phi_lr:                  float = 0.1,
    alpha_div:                 float = 1.0,
    value:                     float = 1.0,
    b0:                        float = 1.0,
    c0:                        float = 1.0,
    kl_max:                    float = 100.0,
    eps:                       float = 1e-6,
    sigma_max:                 float = 5.0,

    include_attention_entropy: bool = True,
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    alpha_mode:                str  = "constant",
    phi_precond_mode:          str  = "none",

    e_sigma_q_trust:           float = 5.0,
    log_prior:                 Optional[torch.Tensor] = None,
) -> BeliefState:
    r"""One inner E-step iteration: mu, sigma (Fisher natgrad + SPD retraction) then phi
    (autograd of the alignment block + preconditioner + Lie retraction)."""
    omega = _transport(belief.phi, group)
    grad_mu, grad_sigma = belief_gradients(
        belief.mu, belief.sigma, mu_p, sigma_p, omega,
        tau=tau, alpha_div=alpha_div, value=value, b0=b0, c0=c0, kl_max=kl_max, eps=eps,
        include_attention_entropy=include_attention_entropy, gradient_mode=gradient_mode,
        family=family, alpha_mode=alpha_mode, log_prior=log_prior,
    )
    nat_mu, nat_sigma = natural_gradient(grad_mu, grad_sigma, belief.sigma, eps=eps)

    mu = belief.mu - e_mu_lr * nat_mu
    sigma = retract_spd_diagonal(
        belief.sigma, -e_sigma_lr * nat_sigma, trust_region=e_sigma_q_trust, eps=eps, sigma_max=sigma_max,
    )

    phi = belief.phi
    if e_phi_lr > 0.0:
        phi_g = belief.phi.detach().clone().requires_grad_(True)
        L = phi_alignment_loss(
            mu, sigma, phi_g, group, tau=tau, alpha_div=alpha_div, kl_max=kl_max, eps=eps,
            family=family, include_attention_entropy=include_attention_entropy, log_prior=log_prior,
        )
        grad_phi = torch.autograd.grad(L, phi_g)[0]
        grad_phi = precondition_phi_gradient(grad_phi, belief.phi, group.generators, mode=phi_precond_mode)
        phi = retract_phi(belief.phi, -grad_phi, group, step_size=e_phi_lr)

    return BeliefState(mu=mu, sigma=sigma, phi=phi)
```

- [ ] **Step 4:** Run — expect 2 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/inference/e_step.py tests/test_e_step.py
git commit -m "feat(inference): e_step_iteration (Fisher natgrad mu/sigma + autograd-phi update)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3 — descent properties (crown jewels)

**Files:** Test `tests/test_e_step.py`.

- [ ] **Step 1 (RED):** append. Tiny LR, clamps inactive, isolated per block; the **right objective** per mode:

```python
def test_filtering_step_descends_F_filt():
    # filtering (query-side) gradient descends F with KEYS FROZEN at the pre-step belief.
    b, mu_p, sigma_p, grp = _belief()
    F_before = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5, keys=b)
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=1e-3, e_sigma_lr=1e-3,
                           e_phi_lr=0.0, gradient_mode="filtering", e_sigma_q_trust=0.0)
    F_after = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5, keys=b)   # SAME frozen keys b
    assert F_after < F_before


def test_smoothing_step_descends_global_F():
    b, mu_p, sigma_p, grp = _belief()
    F_before = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5)            # global (keys=belief)
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=1e-3, e_sigma_lr=1e-3,
                           e_phi_lr=0.0, gradient_mode="smoothing", e_sigma_q_trust=0.0)
    F_after = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5)
    assert F_after < F_before


def test_phi_step_descends_global_F_with_beliefs_frozen():
    b, mu_p, sigma_p, grp = _belief()
    F_before = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5)
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=0.0, e_sigma_lr=0.0,
                           e_phi_lr=1e-3)
    F_after = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5)
    assert F_after < F_before
```

- [ ] **Step 2:** Run — expect 3 passed (no new implementation; exercises Task 1–2). If `test_filtering_step_descends_F_filt` fails, the `keys`-frozen objective is mismatched between before/after (must use the SAME pre-step `keys=b`); if smoothing fails, the gradient mode isn't reaching the oracle. Do not weaken the asserts.

- [ ] **Step 3 (COMMIT):**
```
git add tests/test_e_step.py
git commit -m "test(inference): descent directions (filtering->F_filt, smoothing/phi->global F)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4 — `e_step` loop + trajectory + fixed-seed regression

**Files:** Modify `vfe3/inference/e_step.py`; Test `tests/test_e_step.py`.

- [ ] **Step 1 (RED):** append:

```python
from vfe3.inference.e_step import e_step


def test_e_step_runs_n_iter_and_returns_trajectory():
    b, mu_p, sigma_p, grp = _belief()
    out, traj = e_step(b, mu_p, sigma_p, grp, tau=1.5, n_iter=5,
                       e_mu_lr=1e-2, e_sigma_lr=1e-2, e_phi_lr=1e-2, return_trajectory=True)
    assert len(traj) == 6                                  # F before + after each of 5 iters
    assert (out.sigma > 0).all()


def test_smoothing_loop_decreases_F_overall():
    b, mu_p, sigma_p, grp = _belief()
    out, traj = e_step(b, mu_p, sigma_p, grp, tau=1.5, n_iter=10,
                       e_mu_lr=2e-3, e_sigma_lr=2e-3, e_phi_lr=2e-3,
                       gradient_mode="smoothing", e_sigma_q_trust=0.0, return_trajectory=True)
    assert traj[-1] < traj[0]                              # smoothing descends global F


def test_fixed_seed_regression():
    b, mu_p, sigma_p, grp = _belief(seed=7)
    out = e_step(b, mu_p, sigma_p, grp, tau=1.5, n_iter=3,
                 e_mu_lr=1e-2, e_sigma_lr=1e-2, e_phi_lr=1e-2)
    # snapshot pins determinism (regenerate ONCE from a trusted green run, then freeze)
    assert torch.isfinite(out.mu).all() and torch.isfinite(out.sigma).all() and torch.isfinite(out.phi).all()
    checksum = float(out.mu.sum() + out.sigma.sum() + out.phi.sum())
    assert abs(checksum - EXPECTED_CHECKSUM) < 1e-3        # set EXPECTED_CHECKSUM from the first green run
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3 (GREEN):** append to `vfe3/inference/e_step.py`:

```python
def e_step(
    belief:            BeliefState,
    mu_p:              torch.Tensor,        # (N, K)
    sigma_p:           torch.Tensor,        # (N, K)
    group:             GaugeGroup,

    *,
    n_iter:            int   = 1,
    tau:               float = 1.0,
    e_mu_lr:           float = 0.1,
    e_sigma_lr:        float = 0.1,
    e_phi_lr:          float = 0.1,
    return_trajectory: bool  = False,

    log_prior:         Optional[torch.Tensor] = None,
    **kwargs,
) -> 'BeliefState | Tuple[BeliefState, List[float]]':
    r"""Iterate ``e_step_iteration`` ``n_iter`` times (parallel mean-field). Optionally
    returns the global-F trajectory (a DIAGNOSTIC; parallel updates are not guaranteed
    monotone per iteration)."""
    traj: List[float] = []
    if return_trajectory:
        traj.append(float(free_energy_value(belief, mu_p, sigma_p, group, tau=tau, log_prior=log_prior, **kwargs)))
    for _ in range(n_iter):
        belief = e_step_iteration(
            belief, mu_p, sigma_p, group, tau=tau,
            e_mu_lr=e_mu_lr, e_sigma_lr=e_sigma_lr, e_phi_lr=e_phi_lr, log_prior=log_prior, **kwargs,
        )
        if return_trajectory:
            traj.append(float(free_energy_value(belief, mu_p, sigma_p, group, tau=tau, log_prior=log_prior, **kwargs)))
    return (belief, traj) if return_trajectory else belief
```

> `free_energy_value` and `e_step_iteration` share the same `**kwargs` knobs (alpha_div, value, b0, c0, kl_max, eps, family, alpha_mode, include_attention_entropy, gradient_mode where applicable); pass them through. `free_energy_value` ignores `gradient_mode`/`e_*`/`phi_precond_mode`/`sigma_max`/`e_sigma_q_trust` — filter or accept-and-ignore via `**kwargs` so the shared call site does not error. (Implementer: add `**kwargs` sinks where needed; keep behavior identical.)

- [ ] **Step 4:** Run the regression once, read the printed checksum, set `EXPECTED_CHECKSUM`, re-run — expect 3 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/inference/e_step.py tests/test_e_step.py
git commit -m "feat(inference): e_step loop + F-trajectory diagnostic + fixed-seed regression

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5 — full suite + changelog + final

- [ ] **Step 1:** `python -m pytest -q` — expect all prior (113) + new (~11) green, no regressions.
- [ ] **Step 2:** Append "## Phase 6 E-step — 2026-05-29 (continuation)" to `docs/edits/2026-05-29-phase2b-transport.md` (match format; provenance-clean: Files, per-function Changes, the descent-objective-per-mode note, Analytic anchors, Test results, Commits). Commit `docs(edits): 2026-05-29 phase 6 e-step changes log`.
- [ ] **Step 3:** Final commit of phase files only if anything uncommitted (never the user's CLAUDE.md / untracked dirs).

---

## Self-Review

**Spec coverage (§4.5 + §8 Phase 6):** iterative belief loop reading `gradients` + `geometry` → Tasks 2,4. Decoupled LRs + trust region → Task 2. F-descent property → Task 3. Fixed-seed regression → Task 4.

**Hand-derived anchors:**
- filtering step → `F_filt` decreases (keys frozen, same `keys=b` before/after); smoothing step → global F decreases; φ step (μ,σ frozen) → global F decreases. Tiny LR, clamps inactive.
- σ stays positive; decoupled LRs freeze their components; deterministic fixed-seed checksum.
- smoothing loop decreases F overall; the trajectory is a diagnostic (parallel mean-field not asserted monotone).

**Deferred (named):** Regime II (`connection_delta`); RoPE-on-μ; head-mixer; the **M-step** (prior/parameter learning); full-covariance E-step path (diagonal first); batching (Phase 7); the key-side (smoothing) hand kernel (smoothing uses the oracle).

**Type/name consistency:** `BeliefState(mu, sigma, phi)`; `free_energy_value(belief, mu_p, sigma_p, group, *, …, keys)`; `phi_alignment_loss(mu, sigma, phi, group, *, …)`; `e_step_iteration(belief, mu_p, sigma_p, group, *, e_mu_lr, e_sigma_lr, e_phi_lr, gradient_mode, …)`; `e_step(belief, mu_p, sigma_p, group, *, n_iter, …, return_trajectory)`. E-step calls `belief_gradients` (envelope), not a hand-rolled ∂β; descent objective matches the gradient mode.
