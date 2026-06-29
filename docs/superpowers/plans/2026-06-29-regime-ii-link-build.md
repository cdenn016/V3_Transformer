# Regime II Direct-Link Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `transport_mode="regime_ii_link"` as a dense all-pairs, model-owned direct-link connection for the belief channel, with a preserved flat pure path and source-backed tests.

**Architecture:** The new mode registers a belief-independent transport builder that returns direct group-valued links `L_ij` as `TransportDict["Omega"]`. `connection_L` is a zero-initialized model parameter of shape `(max_seq_len, max_seq_len, n_gen)`, sliced to the active sequence length and exponentiated after self-edge masking and an embedded-matrix soft cap. The mode threads through the existing q-channel E-step, diagnostics, optimizer, reports, and visualization paths, while the s-channel and prefix cache remain flat-only.

**Tech Stack:** Python, PyTorch, pytest, existing `vfe3` transport registry, existing Gaussian belief families, existing free-energy and E-step kernels.

## Global Constraints

The source specification is `docs/research/2026-06-29-regime-ii-direct-link-spec.md`.

The implementation must preserve `transport_mode="flat"` as the default theoretically pure path, with no `connection_L` attribute and no added parameters.

The direct-link flat limit is identity links, not the nonzero Regime I vertex-frame cocycle. Compare to current flat transport only when `phi == 0` or an explicit charted compatibility mode is later designed.

The canonical direct-link builder must not register `needs_mu=True` or `needs_sigma=True`, because `connection_L` is model-owned and belief-independent.

The q-channel may use `regime_ii_link`; the s-channel remains `transport_mode="flat"` in v1.

Because the canonical direct link is an edge-owned variable and does not make the loss depend on the vertex-frame `phi`, v1 must reject `transport_mode="regime_ii_link"` with `e_phi_lr > 0.0` unless a later implementation adds an explicit zero-gradient phi substep. Do not hide that incompatibility with `allow_unused=True` in `torch.autograd.grad`.

`gauge_transport="off"` must continue rejecting every non-flat `transport_mode`, including `regime_ii_link`.

No link exponentials, covariance sandwiches, Cholesky solves, or log-determinants may run in bf16 or fp16.

Do not add CLI parsing. Tests and entry points must follow existing click-to-run/config patterns.

Do not run pytest with an extra `-q`; `pyproject.toml` already sets quiet mode. Report pass counts only from pytest summary output or junit XML.

Update the single daily edits log `docs/2026-06-29-edits.md` when implementation changes are made.

## Phase 0: Documentation Discovery

**Documents and source anchors already inspected:**

`docs/research/2026-06-29-regime-ii-direct-link-spec.md` defines the direct-link ontology, identity-link limit, covariance law, dense all-pairs rule, optimizer/reporting seams, cache restriction, and required tests.

Research wiki pages consulted: `VFE Transformer Program`, `Lattice gauge theory`, `Gauge transformation`, `Parallel transport`, `Holonomy`, `GL(K) gauge-equivariant attention`, `Evidence lower bound (ELBO)`, `Variational EM`, and `Gauge equivariant CNN`.

Executable source anchors: `vfe3/geometry/transport.py:109-131`, `vfe3/geometry/transport.py:212-340`, `vfe3/geometry/transport.py:372-507`, `vfe3/geometry/transport.py:510-588`, `vfe3/inference/e_step.py:39-150`, `vfe3/gradients/kernels.py:164-202`, `vfe3/model/model.py:215-253`, `vfe3/model/model.py:700-752`, `vfe3/model/model.py:1456-1466`, `vfe3/train.py:137-146`, `vfe3/train.py:867-880`, `vfe3/inference/belief_cache.py:56-72`, `vfe3/viz/extract.py:60-101`, `vfe3/viz/extract.py:453-512`, and `vfe3/metrics.py:696-832`.

**Allowed APIs and patterns:**

Use `register_transport` / `get_transport` from `vfe3.geometry.transport`. Register `regime_ii_link` without `needs_mu` or `needs_sigma`.

Use `stable_matrix_exp_pair(..., only_forward=True, block_dims=..., exp_dim=...)` for direct-link exponentials.

Use `transport_mean` and `transport_covariance` downstream. They already implement `Omega mu` and `Omega Sigma Omega.T`; do not add another covariance transport formula.

Use `uses_kernel_route` as the source of truth for kernel eligibility. Direct-link mode should stay eligible under the canonical fixed-transport case, then prove equivalence to the autograd oracle.

Use optimizer grouping from `connection_W` and `connection_M`: `lr=cfg.m_phi_lr`, `role="phi"`, and `cfg.connection_weight_decay` when set.

Use `cache_supported(cfg)` as-is for production cache rejection; add tests proving `regime_ii_link` is rejected.

**Anti-pattern guards:**

Do not implement `Omega = exp(phi_i) exp(delta_ij) exp(-phi_j)` for `regime_ii_link`. That is the existing middle-factor chart, not the direct-link ontology.

Do not compare zero direct links to nonzero flat Regime I transport. Identity direct links and pure-gauge cocycles are gauge-equivalent, not byte-identical coordinates.

Do not auto-enable `oracle_unroll_grad=True` merely because `transport_mode="regime_ii_link"`. The existing auto-enable rule is for state-dependent non-flat modes.

Do not silently drop `connection_L` in `_transport`, `build_belief_transport`, `vfe_block`, `vfe_stack`, `forward_beliefs`, diagnostics, or extraction helpers.

Do not claim `||H-I||_F` is gauge-invariant for noncompact `GL(K)`. Keep it as a fixed-gauge diagnostic and add a conjugacy-invariant loop statistic.

## File Structure

Create `tests/test_regime_ii_link.py` for direct-link transport, covariance, curvature, gradient, model, optimizer, and cache acceptance tests.

Modify `vfe3/geometry/transport.py` to add the registered direct-link builder and small helpers for link diagnostics if they naturally belong beside the builder.

Modify `vfe3/config.py` to add `link_alpha` and `link_soft_cap` fields, validation, warnings, and non-flat predicates.

Modify `vfe3/inference/e_step.py`, `vfe3/model/block.py`, `vfe3/model/stack.py`, and `vfe3/model/model.py` to thread `connection_L`, `link_alpha`, and `link_soft_cap`.

Modify `vfe3/train.py` to add optimizer grouping and metrics CSV forwarding for link diagnostics.

Modify `vfe3/metrics.py` only if adding a reusable conjugacy-invariant direct-link loop diagnostic.

Modify `vfe3/viz/extract.py` and `vfe3/run_artifacts.py` so reports and figures see the active direct-link connection.

Modify existing tests in `tests/test_config.py`, `tests/test_belief_cache.py`, `tests/test_train.py`, `tests/test_run_diagnostics_2026_06_13.py`, and report/artifact tests only where they already own the relevant behavior.

Modify `docs/2026-06-29-edits.md` once after code changes are complete.

## Task 1: Direct-Link Acceptance Tests

**Files:**

Create: `tests/test_regime_ii_link.py`

Modify: `tests/test_config.py`

Modify: `tests/test_belief_cache.py`

**Interfaces:**

Consumes: existing `get_transport`, `compute_transport_operators`, `build_belief_transport`, `VFE3Config`, `VFEModel`, `build_optimizer`, `free_energy_value`, and `cache_supported`.

Produces: failing tests for `regime_ii_link`, `connection_L`, `link_alpha`, `link_soft_cap`, `e_phi_lr` incompatibility, full-covariance noncompact safety, model wiring, optimizer grouping, gradient flow, kernel eligibility, and cache rejection.

- [ ] **Step 1: Add transport-core tests**

Add helpers and tests for registration, no state-routing metadata, identity-link fallback, `link_alpha=0`, dict shape, and active-length slicing.

```python
import math

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import _TRANSPORT_NEEDS_MU, _TRANSPORT_NEEDS_SIGMA, get_transport


def _grp(k: int = 4, n_heads: int = 2):
    return get_group("block_glk")(k, n_heads)


def _identity_omega(batch: int, n_tok: int, k: int, *, device, dtype):
    eye = torch.eye(k, device=device, dtype=dtype)
    return eye.expand(batch, n_tok, n_tok, k, k).contiguous()


def _small_inputs(seed: int = 0, *, batch: int = 1, n_tok: int = 4, k: int = 4, n_heads: int = 2):
    gen = torch.Generator().manual_seed(seed)
    grp = _grp(k, n_heads)
    phi = 0.1 * torch.randn(batch, n_tok, grp.generators.shape[0], generator=gen)
    connection_l = 0.15 * torch.randn(n_tok + 2, n_tok + 2, grp.generators.shape[0], generator=gen)
    return phi, connection_l, grp


def test_regime_ii_link_is_registered():
    assert callable(get_transport("regime_ii_link"))


def test_regime_ii_link_has_no_state_routing_metadata():
    assert "regime_ii_link" not in _TRANSPORT_NEEDS_MU
    assert "regime_ii_link" not in _TRANSPORT_NEEDS_SIGMA


def test_regime_ii_link_connection_none_returns_identity_links():
    phi, _connection_l, grp = _small_inputs(seed=1)
    omega = get_transport("regime_ii_link")(phi, grp, connection_L=None)["Omega"]
    expected = _identity_omega(phi.shape[0], phi.shape[1], grp.generators.shape[-1],
                               device=phi.device, dtype=phi.dtype)
    assert torch.equal(omega, expected)


def test_regime_ii_link_zero_alpha_ignores_nonzero_table():
    phi, connection_l, grp = _small_inputs(seed=2)
    omega = get_transport("regime_ii_link")(
        phi, grp, connection_L=connection_l, link_alpha=0.0
    )["Omega"]
    expected = _identity_omega(phi.shape[0], phi.shape[1], grp.generators.shape[-1],
                               device=phi.device, dtype=phi.dtype)
    assert torch.allclose(omega, expected, atol=1e-6, rtol=0.0)
```

- [ ] **Step 2: Add self-edge, curvature, covariance, and kernel-route tests**

Extend the same file with tests for self-edge identity, nonzero loop holonomy, covariance congruence, full-Gaussian active link covariance, and kernel-route intent. The covariance test may transform assembled `Omega` directly; production does not need a matrix-log projection from arbitrary transformed links back into coordinates.

- [ ] **Step 3: Add model, optimizer, gradient, config, and cache tests**

Add `_tiny_cfg(...)`, then test that `VFEModel` creates `connection_L` only in direct-link mode, optimizer groups it exactly once, nonzero links change forward loss, `loss.backward()` produces finite `connection_L.grad`, config rejects invalid `link_alpha` and `link_soft_cap`, config rejects `transport_mode="regime_ii_link"` with `e_phi_lr > 0.0`, full covariance with noncompact direct links warns or rejects until the float64 sandwich and hard non-PD policy exist, and `cache_supported` rejects the mode.

- [ ] **Step 4: Verify expected failures**

Run:

```powershell
python -m pytest tests/test_regime_ii_link.py --junitxml=C:\tmp\v3-regime-ii-link-task1.xml
```

Expected: failures for missing `regime_ii_link`, missing config fields, and missing `connection_L`.

- [ ] **Step 5: Commit tests**

```powershell
git add tests/test_regime_ii_link.py tests/test_config.py tests/test_belief_cache.py
git commit -m "test: add Regime II direct-link acceptance coverage"
```

## Task 2: Transport Builder and Config Fields

**Files:**

Modify: `vfe3/geometry/transport.py`

Modify: `vfe3/config.py`

Test: `tests/test_regime_ii_link.py`

**Interfaces:**

Consumes: existing `GaugeGroup`, `TransportDict`, `build_factored_transport`, and `stable_matrix_exp_pair`.

Produces: registered `regime_ii_link` builder, config fields `link_alpha` and `link_soft_cap`, and validation.

- [ ] **Step 1: Add config fields and validation**

Add fields near `transport_mode` and `cocycle_relaxation`.

```python
    link_alpha:              float = 1.0
    link_soft_cap:           float = 6.0
```

Add validation beside the `cocycle_relaxation` guard.

```python
        if not (0.0 <= self.link_alpha <= 1.0):
            raise ValueError(f"link_alpha must be in [0,1], got {self.link_alpha}")
        if not (self.link_soft_cap > 0.0 and self.link_soft_cap == self.link_soft_cap):
            raise ValueError(f"link_soft_cap must be positive and finite, got {self.link_soft_cap}")
        if self.transport_mode == "regime_ii_link" and self.e_phi_lr > 0.0:
            raise ValueError(
                "transport_mode='regime_ii_link' is edge-owned and independent of vertex-frame phi; "
                "set e_phi_lr=0.0 or implement an explicit zero-gradient phi substep."
            )
```

Include `regime_ii_link` in full-covariance and non-flat warning predicates, but leave the auto-oracle rule limited to `("regime_ii", "regime_ii_covariant")`. Add a direct test that `family="gaussian_full"` with a noncompact group such as `block_glk` either emits the warning or raises a clear `ValueError` until the float64 covariance sandwich and hard non-PD policy are implemented.

- [ ] **Step 2: Add the direct-link builder**

Register the builder in `vfe3/geometry/transport.py`.

```python
@register_transport("regime_ii_link")
def _build_regime_ii_link(
    phi:                torch.Tensor,             # (B, N, n_gen) gauge frames
    group:              GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:         str                    = "learned",
    link_alpha:         float                  = 1.0,
    link_soft_cap:      float                  = 6.0,
    connection_L:       Optional[torch.Tensor] = None,
    **kwargs,
) -> TransportDict:
    r"""Dense all-pairs direct-link Regime II transport."""
    fac = build_factored_transport(phi, group, gauge_mode=gauge_mode)
    exp_phi, exp_neg_phi = fac.exp_phi, fac.exp_neg_phi
    B, N = phi.shape[0], phi.shape[1]
    K = group.generators.shape[-1]
    dtype = phi.dtype
    device = phi.device

    eye_K = torch.eye(K, device=device, dtype=dtype)
    if connection_L is None or link_alpha == 0.0:
        omega = eye_K.expand(B, N, N, K, K)
        return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}

    if connection_L.dim() != 3 or connection_L.shape[0] < N or connection_L.shape[1] < N:
        raise ValueError(
            "regime_ii_link requires connection_L with shape "
            f"(max_seq_len, max_seq_len, n_gen) covering active N={N}; "
            f"got {tuple(connection_L.shape)}."
        )
    if connection_L.shape[-1] != group.generators.shape[0]:
        raise ValueError(
            f"connection_L last dim must equal n_gen={group.generators.shape[0]}, "
            f"got {connection_L.shape[-1]}."
        )

    with torch.amp.autocast("cuda", enabled=False):
        link_coord = (link_alpha * connection_L[:N, :N, :]).to(device=device, dtype=torch.float32)
        eye_N = torch.eye(N, dtype=torch.bool, device=device)
        link_coord = link_coord.masked_fill(eye_N.view(N, N, 1), 0.0)
        generators = group.generators.to(device=device, dtype=torch.float32)
        link_mat = torch.einsum("ija,akl->ijkl", link_coord, generators)
        fro_sq = link_mat.pow(2).sum(dim=(-2, -1), keepdim=True)
        link_mat = link_mat * torch.rsqrt(1.0 + fro_sq / (link_soft_cap * link_soft_cap))

    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_link, _ = stable_matrix_exp_pair(
        link_mat,
        skew_symmetric=group.skew_symmetric,
        only_forward=True,
        block_dims=block_dims,
        exp_dim=(max(block_dims) if block_dims is not None else None),
    )
    omega = exp_link.to(dtype).unsqueeze(0).expand(B, N, N, K, K)
    # Do not call contiguous here: the batch dimension must remain an expanded view until
    # a small-case test or a row-chunked downstream energy path explicitly materializes it.
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}
```

- [ ] **Step 3: Run core tests**

```powershell
python -m pytest tests/test_regime_ii_link.py::test_regime_ii_link_is_registered tests/test_regime_ii_link.py::test_regime_ii_link_has_no_state_routing_metadata tests/test_regime_ii_link.py::test_regime_ii_link_connection_none_returns_identity_links tests/test_regime_ii_link.py::test_regime_ii_link_zero_alpha_ignores_nonzero_table --junitxml=C:\tmp\v3-regime-ii-link-task2.xml
```

- [ ] **Step 4: Commit transport and config**

```powershell
git add vfe3/geometry/transport.py vfe3/config.py tests/test_regime_ii_link.py
git commit -m "feat: add Regime II direct-link transport"
```

## Task 3: E-Step, Stack, Model, and Optimizer Wiring

**Files:**

Modify: `vfe3/inference/e_step.py`

Modify: `vfe3/model/block.py`

Modify: `vfe3/model/stack.py`

Modify: `vfe3/model/model.py`

Modify: `vfe3/train.py`

Test: `tests/test_regime_ii_link.py`

**Interfaces:**

Consumes: `_build_regime_ii_link` and config fields.

Produces: `connection_L` creation, q-channel forwarding, active diagnostics forwarding, and optimizer coverage.

- [ ] **Step 1: Thread link arguments through E-step APIs**

Add these keyword arguments to `_transport`, `build_belief_transport`, `free_energy_value`, `phi_alignment_loss`, `e_step_iteration`, and `e_step`.

```python
    connection_L:       Optional[torch.Tensor] = None,
    link_alpha:         float                  = 1.0,
    link_soft_cap:      float                  = 6.0,
```

In `build_belief_transport`, add a direct-link dispatch branch.

```python
        elif transport_mode == "regime_ii_link":
            transport_kw = dict(
                connection_L=connection_L,
                link_alpha=link_alpha,
                link_soft_cap=link_soft_cap,
            )
```

- [ ] **Step 2: Thread link arguments through block and stack**

Add `connection_L: Optional[torch.Tensor] = None` to `vfe_block` and `vfe_stack`, then forward `connection_L`, `cfg.link_alpha`, and `cfg.link_soft_cap` to `e_step`.

- [ ] **Step 3: Create and forward `connection_L` in `VFEModel`**

Add parameter creation beside the existing Regime II connection parameters.

```python
        if cfg.transport_mode == "regime_ii_link":
            n_gen = self.group.generators.shape[0]
            self.connection_L = nn.Parameter(torch.zeros(cfg.max_seq_len, cfg.max_seq_len, n_gen))
```

Forward `connection_L` through `forward_beliefs`, diagnostics, attention maps, and per-layer diagnostics using `getattr(self, "connection_L", None)`.

Add a `lambda_gamma` / `s_e_step` regression showing that the direct-link mode remains belief-channel only and does not silently switch model-channel gamma transport away from the current flat, out-of-scope path.

- [ ] **Step 4: Add optimizer grouping**

In `build_optimizer`, add a group beside `connection_W` and `connection_M`.

```python
    if getattr(model, "connection_L", None) is not None:
        l_group = {"params": [model.connection_L], "lr": cfg.m_phi_lr, "role": "phi"}
        if cfg.connection_weight_decay is not None:
            l_group["weight_decay"] = cfg.connection_weight_decay
        groups.append(l_group)
```

- [ ] **Step 5: Run model and optimizer tests**

```powershell
python -m pytest tests/test_regime_ii_link.py::test_model_regime_ii_link_creates_connection_l_zero_init tests/test_regime_ii_link.py::test_model_flat_has_no_connection_l tests/test_regime_ii_link.py::test_model_regime_ii_link_nonzero_l_changes_forward tests/test_regime_ii_link.py::test_model_regime_ii_link_gradient_flows_to_l tests/test_regime_ii_link.py::test_build_optimizer_groups_connection_l_once --junitxml=C:\tmp\v3-regime-ii-link-task3.xml
```

- [ ] **Step 6: Commit wiring**

```powershell
git add vfe3/inference/e_step.py vfe3/model/block.py vfe3/model/stack.py vfe3/model/model.py vfe3/train.py tests/test_regime_ii_link.py
git commit -m "feat: wire Regime II direct links through q-channel"
```

## Task 4: Variational Gradient and Kernel Verification

**Files:**

Modify: `tests/test_regime_ii_link.py`

Modify: `vfe3/gradients/kernels.py` only if a source-of-truth predicate helper is introduced.

**Interfaces:**

Consumes: fixed direct-link transport and existing kernel/oracle gradient paths.

Produces: proof that `regime_ii_link` stays kernel-eligible under canonical diagonal/KL/filtering/entropy-on settings and that gradients to `connection_L` match finite differences.

- [ ] **Step 1: Add finite-difference gradient test for `connection_L`**

Add a `dF/dconnection_L` finite-difference test by adapting `tests/test_regime_ii.py::test_regime_ii_df_dw_matches_fd`. Probe selected off-diagonal `(i, j, a)` entries only. Diagonal entries should remain masked and may have zero gradient by design.

- [ ] **Step 2: Add kernel-vs-oracle fixed-link test**

Adapt `tests/test_gradients_kernels.py::test_kernel_matches_oracle_multihead_canonical` so the fixed transport is built with `transport_mode="regime_ii_link"`. Keep the canonical kernel conditions: `gradient_mode="filtering"`, `family="gaussian_diagonal"`, `divergence_family="renyi"`, `renyi_order=1.0`, and `include_attention_entropy=True`.

- [ ] **Step 3: Run gradient verification**

```powershell
python -m pytest tests/test_regime_ii_link.py::test_regime_ii_link_df_dconnection_l_matches_fd tests/test_regime_ii_link.py::test_regime_ii_link_kernel_matches_autograd_oracle_for_fixed_transport --junitxml=C:\tmp\v3-regime-ii-link-task4.xml
```

- [ ] **Step 4: Commit gradient verification**

```powershell
git add tests/test_regime_ii_link.py vfe3/gradients/kernels.py
git commit -m "test: verify direct-link gradients"
```

## Task 5: Diagnostics, Metrics, and Reporting

**Files:**

Modify: `vfe3/model/model.py`

Modify: `vfe3/metrics.py`

Modify: `vfe3/train.py`

Modify: `vfe3/run_artifacts.py`

Modify: `tests/test_regime_ii_link.py`

Modify: `tests/test_run_diagnostics_2026_06_13.py`

**Interfaces:**

Consumes: active `Omega` from `regime_ii_link`.

Produces: `connection_l_norm`, `connection_l_offdiag_norm`, `link_self_residual`, `link_cond_p95`, and a conjugacy-invariant sampled loop statistic.

- [ ] **Step 1: Add a sampled loop spectrum metric**

Add a helper to `vfe3/metrics.py` near existing holonomy functions.

```python
def holonomy_log_spectrum_sampled(
    omega:      torch.Tensor,             # (N, N, K, K)

    *,
    n_samples: int = 64,
    seed:      int = 0,
    eps:       float = 1e-8,
) -> Dict[str, float]:
    r"""Conjugacy-invariant loop statistic from eigenvalue distances to one."""
    N = omega.shape[0]
    if N < 3:
        return {"mean": 0.0, "max": 0.0}
    gen = torch.Generator(device=omega.device).manual_seed(seed)
    vals = []
    for _ in range(n_samples):
        idx = torch.randperm(N, generator=gen, device=omega.device)[:3]
        i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
        H = omega[i, j].double() @ omega[j, k].double() @ omega[k, i].double()
        eig = torch.linalg.eigvals(H.to(torch.complex128))
        vals.append(torch.log(eig.abs().clamp_min(eps)).abs().mean().real)
    stacked = torch.stack(vals)
    return {"mean": float(stacked.mean()), "max": float(stacked.max())}
```

- [ ] **Step 2: Add link diagnostics in `VFEModel.diagnostics`**

Add link-specific diagnostics after existing connection norms. If full edge condition numbers are too expensive, sample them and name the key `link_cond_sample_p95`.

- [ ] **Step 3: Forward link diagnostics into metrics CSV and reports**

Add `connection_l_norm`, `connection_l_offdiag_norm`, `link_self_residual`, `link_log_spectrum`, and `link_cond_p95` to the train-row allow-list and run-artifacts report fields that already surface gauge geometry.

- [ ] **Step 4: Add diagnostics tests**

Adapt `tests/test_regime_ii_covariant.py::test_diagnostics_holonomy_reflects_connection_m` and `tests/test_run_diagnostics_2026_06_13.py::test_conditional_break_columns_present_only_under_toggle` for `connection_L`. Add an explicit per-layer diagnostics test so the current `diagnostics_per_layer()` API or its live equivalent sees the active direct-link transport rather than replaying only `connection_W` / `connection_M` paths.

- [ ] **Step 5: Run diagnostics tests**

```powershell
python -m pytest tests/test_regime_ii_link.py::test_diagnostics_holonomy_reflects_connection_l tests/test_run_diagnostics_2026_06_13.py --junitxml=C:\tmp\v3-regime-ii-link-task5.xml
```

- [ ] **Step 6: Commit diagnostics and reporting**

```powershell
git add vfe3/model/model.py vfe3/metrics.py vfe3/train.py vfe3/run_artifacts.py tests/test_regime_ii_link.py tests/test_run_diagnostics_2026_06_13.py
git commit -m "feat: report Regime II direct-link diagnostics"
```

## Task 6: Visualization Extraction and Cache Routing

**Files:**

Modify: `vfe3/viz/extract.py`

Modify: `tests/test_regime_ii_link.py`

Modify: `tests/test_belief_cache.py`

**Interfaces:**

Consumes: model-owned `connection_L`.

Produces: converged-state and health extractors that reflect the active direct-link transport, and cache tests that prove fallback behavior.

- [ ] **Step 1: Update extractor kwargs**

In `_iter_kwargs` and `_fe_kwargs`, forward link parameters.

```python
        connection_W=getattr(model, "connection_W", None),
        connection_M=getattr(model, "connection_M", None),
        connection_L=getattr(model, "connection_L", None),
        link_alpha=cfg.link_alpha,
        link_soft_cap=cfg.link_soft_cap,
```

Replace hard-coded `("regime_ii", "regime_ii_covariant")` checks with `_TRANSPORT_NEEDS_MU` and `_TRANSPORT_NEEDS_SIGMA` where those checks describe state-routing metadata.

- [ ] **Step 2: Add converged-state test**

Adapt `tests/test_regime_ii_covariant.py::test_converged_state_omega_reflects_connection_m` for `connection_L`.

- [ ] **Step 3: Extend cache guard test**

In `tests/test_belief_cache.py`, add `transport_mode="regime_ii_link"` to the unsupported cases.

- [ ] **Step 4: Run extractor and cache tests**

```powershell
python -m pytest tests/test_regime_ii_link.py::test_converged_state_omega_reflects_connection_l tests/test_belief_cache.py --junitxml=C:\tmp\v3-regime-ii-link-task6.xml
```

- [ ] **Step 5: Commit extraction and cache tests**

```powershell
git add vfe3/viz/extract.py tests/test_regime_ii_link.py tests/test_belief_cache.py
git commit -m "fix: route direct links through diagnostics extractors"
```

## Task 7: Numerical and Performance Guardrails

**Files:**

Modify: `vfe3/config.py`

Modify: `vfe3/geometry/transport.py`

Modify: `tests/test_regime_ii_link.py`

Modify: `tests/test_config.py`

Optional Modify: `tests/test_amp.py`

**Interfaces:**

Consumes: direct-link dense builder.

Produces: explicit supported operating point, full-covariance warning, AMP protection, self-edge post-step protection, and a concrete non-materializing large-shape transport route.

- [ ] **Step 1: Extend full-covariance warning to direct-link mode**

Include `regime_ii_link` in the noncompact full-covariance warning predicate and keep `family="gaussian_diagonal"` as the first supported operating point.

- [ ] **Step 2: Add AMP and self-edge tests**

Add a self-edge post-optimizer-step test and a CUDA-only AMP test proving direct-link matrix exponentials enter `stable_matrix_exp_pair` or the direct-link exponential helper in fp32 or float64 under autocast. A finite-output smoke test is not enough; use a narrow monkeypatch, debug hook, or captured dtype assertion. Skip the AMP test when CUDA is unavailable.

- [ ] **Step 3: Add dense-memory estimator and CUDA guard**

Add a small helper for the dense `Omega` byte estimate and test the spec example. Then implement one concrete large-shape route: either a row-chunked downstream energy path that feeds `transport_mean` / `transport_covariance` without ever materializing full batched `(B,N,N,K,K)`, or a typed transport container with explicit row/block materialization consumed by those functions. The CUDA availability-guarded memory test may record `torch.cuda.max_memory_allocated()` instead of asserting an exact byte count, but it must fail if production direct-link code eagerly materializes an avoidable full batched fp32 transport tensor before chunking or the container takes over.

The small correctness path may still expose an `Omega` object with logical shape `(B,N,N,K,K)`, but batch expansion must remain a view and production-size code must enter the row-chunked or container path before any operation forces the full batched tensor contiguous.

```python
def _direct_link_dense_bytes(batch: int, n_tok: int, k: int, dtype: torch.dtype) -> int:
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    return batch * n_tok * n_tok * k * k * bytes_per
```

Test:

```python
def test_regime_ii_link_dense_memory_estimator_large_case():
    from vfe3.geometry.transport import _direct_link_dense_bytes

    assert _direct_link_dense_bytes(64, 128, 64, torch.float32) == 64 * 128 * 128 * 64 * 64 * 4
```

- [ ] **Step 4: Run numerical guard tests**

```powershell
python -m pytest tests/test_regime_ii_link.py tests/test_config.py --junitxml=C:\tmp\v3-regime-ii-link-task7.xml
```

- [ ] **Step 5: Commit numerical guards**

```powershell
git add vfe3/config.py vfe3/geometry/transport.py tests/test_regime_ii_link.py tests/test_config.py tests/test_amp.py
git commit -m "test: guard Regime II direct-link numerics"
```

## Task 8: Final Verification and Documentation

**Files:**

Modify: `docs/2026-06-29-edits.md`

Read: pytest junit XML outputs

**Interfaces:**

Consumes: all prior tasks.

Produces: machine-readable verification and daily edit note.

- [ ] **Step 1: Run targeted test suite**

```powershell
python -m pytest tests/test_regime_ii_link.py tests/test_regime_ii.py tests/test_regime_ii_covariant.py tests/test_transport.py tests/test_config.py tests/test_belief_cache.py tests/test_train.py tests/test_run_diagnostics_2026_06_13.py tests/test_run_artifacts.py tests/test_report.py --junitxml=C:\tmp\v3-regime-ii-link-targeted.xml
```

Expected: `failures="0"` and `errors="0"` in the XML. `skipped` may be nonzero on CPU-only machines.

- [ ] **Step 2: Run full suite if targeted tests pass**

```powershell
python -m pytest --junitxml=C:\tmp\v3-regime-ii-link-full.xml
```

Expected: `failures="0"` and `errors="0"` in the XML. Report the pass count from XML attributes or pytest summary only.

- [ ] **Step 3: Grep for routing omissions**

```powershell
rg -n "connection_W|getattr\(model, \"connection_W\"|transport_mode == \"regime_ii\"|regime_ii_covariant" vfe3 tests
```

Inspect every remaining literal. Expected: literals that refer specifically to old modes stay; generic active-transport paths use registry metadata or also forward `connection_L`.

- [ ] **Step 4: Update daily edit log**

Append one section to `docs/2026-06-29-edits.md` describing the direct-link implementation and the junit XML artifacts.

- [ ] **Step 5: Commit documentation and final state**

```powershell
git add docs/2026-06-29-edits.md
git commit -m "docs: record Regime II direct-link build"
git status --short --branch
```

Expected: branch is clean after the final commit.

## Final Review Checklist

Every direct-link test compares the identity limit to identity `Omega`, not to the nonzero flat cocycle.

`regime_ii_link` is absent from `_TRANSPORT_NEEDS_MU` and `_TRANSPORT_NEEDS_SIGMA`.

`regime_ii_link` is not added to the config auto-oracle rule.

`connection_L` is created only when `cfg.transport_mode == "regime_ii_link"`.

`connection_L` is grouped exactly once by `build_optimizer`.

The q-channel forwards `connection_L`, `link_alpha`, and `link_soft_cap`.

The s-channel remains flat.

`cache_supported(cfg)` rejects `regime_ii_link`.

`transport_mode="regime_ii_link"` rejects `e_phi_lr > 0.0` or explicitly implements a zero-gradient phi substep.

Full covariance with noncompact direct links warns or rejects until the float64 sandwich and hard non-PD policy exist.

Diagnostics, per-layer diagnostics, and visualization extractors see active direct-link holonomy.

The CUDA memory guard shows the large operating point avoids avoidable full batched transport materialization.

No AMP path runs link matrix exponentials in bf16 or fp16.

The final answer must report only verified test counts from junit XML or pytest summary output.
