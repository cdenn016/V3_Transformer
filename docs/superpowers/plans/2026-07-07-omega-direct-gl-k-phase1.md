# omega_direct GL(K) Group-Element Gauge Parameterization — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gauge_parameterization="omega_direct"` a live, default-OFF parameterization that stores the per-token gauge frame as a group element `U_i ∈ GL(K)` (reaching the full group, det < 0 included) instead of the Lie-algebra coordinate `phi_i`, with an exp-free forward transport and a principled Lie-group (Lie-exp default, Cayley opt-in) natural-gradient optimizer.

**Architecture:** The belief tuple gains a per-token element field `omega` sourced from a new gated `omega_embed` prior table (identity init; optional det < 0 reflection seeding). The transport funnel dispatches on a new `gauge_parameterization` axis: when `"omega_direct"`, it fills the existing `FactoredTransport` slots directly (`exp_phi := U_i`, `exp_neg_phi := U_j⁻¹`) so every downstream mean/covariance consumer is reused unchanged, and the same cocycle `Ω_ij = U_i U_j⁻¹` preserves strict gauge equivariance. The `omega_embed` table is optimized on the group manifold by a new arm of `GaugeNaturalGradAdamW` that retracts `U ← U·retr(−η·Gram⁻¹ proj_g(Uᵀ E))`. The pure `phi`/exp path stays the default and is byte-identical when the axis is off.

**Tech Stack:** Python, PyTorch (float32 working dtype, float64 islands for matrix inversion), pytest. No new dependencies.

## Global Constraints

- NO neural networks on the pure path; `omega_direct` is a pure-path-preserving alternative *chart* (the same cocycle `U_i U_j⁻¹`), not a sanctioned NN exception. It preserves strict gauge equivariance and flat holonomy exactly.
- NO CLI arg parsing; config is edited then run (click-to-run).
- float32 throughout; matrix inversion runs in a float64 island then casts back.
- Default `gauge_parameterization="phi"` MUST stay byte-identical (the mandatory theoretically-pure path). Every new table/field is created ONLY on the `omega_direct` path so the default `state_dict` is byte-identical.
- Scope is `gauge_group ∈ {glk, block_glk}`, `transport_mode="flat"`, `e_phi_lr=0.0` (no E-step frame refinement in Phase 1). All other combinations with `omega_direct` are rejected at config validation with a clear message.
- Function signature convention (CLAUDE.md): tensors first, then keyword-only, vertically aligned names/types/`=`, shape comments, LaTeX in docstrings for non-trivial formulas. Variable names match paper notation (`U`, `omega`, `xi`).
- American English spelling everywhere.
- CPU-bound tests MUST be tiny: `K < 6`, all other dims single-digit. NEVER build a production-scale model in a test.
- Pytest: do NOT pass `-q` (already in `addopts`; `-qq` hides the pass count). Read the `N passed` line or `--junitxml`.
- Post-edit policy: maintain one dated edits doc `docs/2026-07-07-edits.md`, updated (not recreated) as tasks land.

---

## File Structure

**Modified:**
- `vfe3/config.py` — flip the `omega_direct` reject into per-group gates; add `omega_retract_mode`, `omega_reflection` fields + validation.
- `vfe3/belief.py` — add trailing `omega: Optional[torch.Tensor] = None` field.
- `vfe3/geometry/generators.py` — add `reflection_element(K, ...)` (the only det < 0 element builder in the repo).
- `vfe3/geometry/transport.py` — add `build_transport_from_element(omega, group)` (exp-free, fp64 inverse island).
- `vfe3/geometry/lie_ops.py` — add `retract_omega(U, xi, generators, *, mode)` (Lie-exp / Cayley group retraction).
- `vfe3/model/prior_bank.py` — add gated `omega_embed` table + `gauge_parameterization`/`irrep_dims`/`omega_reflection` ctor args; populate `belief.omega` in `_encode_per_token`.
- `vfe3/inference/e_step.py` — thread `gauge_parameterization` + `omega` through `build_belief_transport` / `_transport` / `e_step_iteration` / `e_step` / `free_energy_value`; add the omega dispatch branch.
- `vfe3/model/model.py` — thread `gauge_parameterization` into the transport-build boundaries; pass `irrep_dims`/`gauge_parameterization` to `PriorBank`; thread `omega` through `_gamma_energy` + its 4 callers; diagnostics `_transport` sites.
- `vfe3/gauge_optim.py` — add an `"omega"` param-group branch doing the group retraction.
- `vfe3/train.py` — route `omega_embed` into a `{"omega": True}` group and select `GaugeNaturalGradAdamW` when `omega_direct`.
- `vfe3/inference/belief_cache.py` — element-frame KV rebuild + `cache_supported` clause.
- `ablation.py`, `train_vfe3.py`, `scaling.py` — remove from `NON_SWEPT_FIELDS`, add sweep arm, refresh stale comments.

**Created:**
- `tests/test_omega_direct.py` — the Phase-1 test suite.

## Interfaces (locked names/signatures every task references)

```
# belief.py
BeliefState(mu, sigma, phi, s=None, r=None, omega=None)          # omega: (..., N, K, K) per-token block-diagonal U_i

# generators.py
reflection_element(K: int, *, dtype=torch.float32, device=None) -> torch.Tensor   # (K, K) diag(-1,1,...,1), det=-1

# transport.py
build_transport_from_element(
    omega: torch.Tensor,          # (B, N, K, K) per-token U_i (block-diagonal for block_glk)
    group: GaugeGroup,
) -> 'FactoredTransport | dict'   # FactoredTransport for fused (equal-block) groups; {'exp_phi','exp_neg_phi','Omega'} dict otherwise

# lie_ops.py
retract_omega(
    U:          torch.Tensor,     # (..., K, K) current element
    xi:         torch.Tensor,     # (..., n_gen) descent step in algebra coords (already -lr scaled)
    generators: torch.Tensor,     # (n_gen, K, K)
    *, mode: str = "lie_exp",     # 'lie_exp' | 'cayley'
) -> torch.Tensor                 # (..., K, K) U @ retr(xi)

# prior_bank.py PriorBank.__init__ gains kwargs:
#   gauge_parameterization: str = "phi", irrep_dims: Optional[List[int]] = None, omega_reflection: str = "off"
# encode() (per_token) returns BeliefState(..., omega=<(B,N,K,K)> or None)

# e_step.py build_belief_transport / _transport / e_step_iteration / e_step / free_energy_value gain:
#   gauge_parameterization: str = "phi", omega: Optional[torch.Tensor] = None

# config.py new fields (E-step/M-step group):
#   omega_retract_mode: str = "lie_exp"     # 'lie_exp' | 'cayley'
#   omega_reflection:   str = "off"         # 'off' | 'init_seed'
_VALID_OMEGA_RETRACT   = ("lie_exp", "cayley")
_VALID_OMEGA_REFLECTION = ("off", "init_seed")
```

---

### Task 1: Config — make `omega_direct` live (per-group gates), add fields, invert the rejection test

**Files:**
- Modify: `vfe3/config.py:22` (add valid tuples), `:64-66` (field comment), `:378-380` (new fields), `:893-912` (delete reject, add gates), `:2154` (`_require`, unchanged)
- Test: `tests/test_config.py:224-229` (invert)

**Interfaces:**
- Produces: a `VFE3Config` that accepts `gauge_parameterization="omega_direct"` for `gauge_group ∈ {glk, block_glk}` with `transport_mode="flat"`, `e_phi_lr=0.0`, and rejects it otherwise; new fields `omega_retract_mode`, `omega_reflection`.

- [ ] **Step 1: Write the failing tests** (replace the current rejection test at `tests/test_config.py:224-229`)

```python
# tests/test_config.py  — replace test_config_rejects_omega_direct_gauge_parameterization
def test_config_accepts_omega_direct_on_gl_groups():
    """omega_direct is now a live element-valued chart on the GL groups (glk, block_glk)."""
    for grp, over in (("glk", {}), ("block_glk", {"n_heads": 2})):
        cfg = VFE3Config(gauge_parameterization="omega_direct", gauge_group=grp,
                         embed_dim=4, n_heads=over.get("n_heads", 1), transport_mode="flat", e_phi_lr=0.0)
        assert cfg.gauge_parameterization == "omega_direct"
    assert VFE3Config(gauge_parameterization="phi").gauge_parameterization == "phi"


def test_config_rejects_omega_direct_off_scope():
    with pytest.raises(ValueError):                       # so_k has no det<0 element-store scope in Phase 1
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="so_k", embed_dim=4, n_heads=1)
    with pytest.raises(ValueError):                       # E-step frame refinement not supported yet
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1, e_phi_lr=0.1)
    with pytest.raises(ValueError):                       # only the flat regime in Phase 1
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1,
                   transport_mode="regime_ii")


def test_config_omega_retract_and_reflection_validated():
    with pytest.raises(ValueError):
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1,
                   omega_retract_mode="bogus")
    assert VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1,
                      omega_retract_mode="cayley").omega_retract_mode == "cayley"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_config.py -k "omega_direct or omega_retract" -x`
Expected: FAIL (`omega_direct` still raises `NotImplementedError`; new fields do not exist).

- [ ] **Step 3: Add the valid tuples** at `vfe3/config.py:26` (after `_VALID_GAUGE_TRANSPORT`)

```python
_VALID_OMEGA_RETRACT       = ("lie_exp", "cayley")
_VALID_OMEGA_REFLECTION    = ("off", "init_seed")
```

- [ ] **Step 4: Add the two fields** at `vfe3/config.py:380` (after `spd_retract_mode`, in the E-step/M-step group)

```python
    omega_retract_mode:        str   = "lie_exp"  # omega_direct group-manifold retraction: 'lie_exp' | 'cayley'
    omega_reflection:          str   = "off"      # omega_direct det<0 seeding: 'off' (det>0 only) | 'init_seed'
```

- [ ] **Step 5: Rewrite the field comment** at `vfe3/config.py:66`

```python
    gauge_parameterization:    str   = "phi"          # 'phi' (exp(phi.G), the pure default) | 'omega_direct' (store U in GL(K); glk/block_glk, transport_mode='flat', e_phi_lr=0)
```

- [ ] **Step 6: Delete the reject and add the Phase-1 gates** — replace `vfe3/config.py:901-912` (keep 893-900) with:

```python
        # 'omega_direct' stores the per-token frame as a GL(K) group element U_i (belief.omega),
        # sourced from prior_bank.omega_embed, transport Omega_ij = U_i U_j^{-1}. Phase 1 scope:
        # the non-compact GL groups (glk, block_glk), the flat regime, and no E-step frame
        # refinement. Everything outside that scope is rejected with a clear message.
        if self.gauge_parameterization == "omega_direct":
            if self.gauge_group not in ("glk", "block_glk"):
                raise ValueError(
                    f"gauge_parameterization='omega_direct' is Phase-1-scoped to gauge_group in "
                    f"('glk', 'block_glk') (the non-compact GL groups where det<0 reach is meaningful); "
                    f"got gauge_group={self.gauge_group!r}."
                )
            if self.transport_mode != "flat":
                raise ValueError(
                    f"gauge_parameterization='omega_direct' requires transport_mode='flat' in Phase 1; "
                    f"got transport_mode={self.transport_mode!r}."
                )
            if self.e_phi_lr != 0.0:
                raise ValueError(
                    f"gauge_parameterization='omega_direct' does not support E-step frame refinement "
                    f"(e_phi_lr>0) in Phase 1; got e_phi_lr={self.e_phi_lr}. Set e_phi_lr=0.0."
                )
        _require(self.omega_retract_mode, _VALID_OMEGA_RETRACT, "omega_retract_mode")
        _require(self.omega_reflection, _VALID_OMEGA_REFLECTION, "omega_reflection")
```

- [ ] **Step 7: Run to verify pass**

Run: `pytest tests/test_config.py -k "omega_direct or omega_retract" -x`
Expected: PASS. Then `pytest tests/test_config.py` — expect the full file green (no other test referenced the old rejection).

- [ ] **Step 8: Commit**

```bash
git add vfe3/config.py tests/test_config.py
git commit -m "feat(omega_direct): flip config reject into Phase-1 per-group gates + retract/reflection fields"
```

---

### Task 2: Belief field — add `omega` to `BeliefState`

**Files:**
- Modify: `vfe3/belief.py:22-31`
- Test: `tests/test_omega_direct.py` (create)

**Interfaces:**
- Produces: `BeliefState(..., omega=None)` — a trailing optional `(..., N, K, K)` element field, byte-identical for existing constructions.

- [ ] **Step 1: Write the failing test** (create `tests/test_omega_direct.py`)

```python
import pytest
import torch

from vfe3.belief import BeliefState


def test_beliefstate_omega_field_optional_and_addressable():
    mu = torch.zeros(1, 3, 4); sigma = torch.ones(1, 3, 4); phi = torch.zeros(1, 3, 5)
    b = BeliefState(mu=mu, sigma=sigma, phi=phi)
    assert b.omega is None                                   # default: phi path untouched
    U = torch.eye(4).expand(1, 3, 4, 4)
    b2 = b._replace(omega=U)
    assert torch.equal(b2.omega, U)
    assert b2.mu is mu and b2.phi is phi                     # other fields preserved
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_omega_direct.py::test_beliefstate_omega_field_optional_and_addressable -x`
Expected: FAIL (`TypeError: __new__() got an unexpected keyword argument 'omega'`).

- [ ] **Step 3: Add the field** at `vfe3/belief.py:31` (after `r`)

```python
    omega: Optional[torch.Tensor] = None      # optional GL(K) frame U_i (..., N, K, K); set only on gauge_parameterization='omega_direct'
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_omega_direct.py::test_beliefstate_omega_field_optional_and_addressable -x`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vfe3/belief.py tests/test_omega_direct.py
git commit -m "feat(omega_direct): add optional omega frame field to BeliefState"
```

---

### Task 3: Reflection element builder (the only det < 0 constructor in the repo)

**Files:**
- Modify: `vfe3/geometry/generators.py` (append a module-level function)
- Test: `tests/test_omega_direct.py`

**Interfaces:**
- Produces: `reflection_element(K, *, dtype, device) -> (K, K)` `= diag(-1, 1, ..., 1)`, `det = -1`, orthogonal, block-diagonal (the `-1` sits in block 0 so it is a valid element of `GL(d_0) × ...`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_omega_direct.py
from vfe3.geometry.generators import reflection_element

def test_reflection_element_is_det_negative_orthogonal():
    R = reflection_element(4)
    assert R.shape == (4, 4)
    assert torch.det(R) < 0                                  # reaches the other GL component
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(4), atol=1e-7)   # reflection: R R^T = I
    assert torch.allclose(R @ R, torch.eye(4), atol=1e-7)   # involutory
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_omega_direct.py::test_reflection_element_is_det_negative_orthogonal -x`
Expected: FAIL (`ImportError: cannot import name 'reflection_element'`).

- [ ] **Step 3: Implement** — append to `vfe3/geometry/generators.py`

```python
def reflection_element(
    K:      int,

    *,
    dtype:  torch.dtype                     = torch.float32,
    device: 'torch.device | str | None'     = None,
) -> torch.Tensor:                          # (K, K) det<0 reflection
    r"""Canonical orientation-reversing reflection diag(-1, 1, ..., 1) in O(K) <= GL(K).

    The single -1 gives det R = -1, so R lies in the det<0 component of GL(K) that no matrix
    exponential (det exp = e^{tr} > 0) can reach. R is orthogonal and involutory (R = R^{-1} = R^T),
    and diagonal, hence block-diagonal for any irrep_dims -- it is a valid element of GL(d_0) x ...
    (the -1 sits in block 0). Used to seed the det<0 component at init for gauge_parameterization=
    'omega_direct' (the discrete component cannot be reached by the continuous VFE minimization; see
    docs/superpowers/specs/2026-07-07-omega-direct-gauge-parameterization-design.md sec 3.4).
    """
    diag = torch.ones(K, dtype=torch.float64)
    diag[0] = -1.0
    return torch.diag(diag).to(dtype).to(device)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_omega_direct.py::test_reflection_element_is_det_negative_orthogonal -x`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vfe3/geometry/generators.py tests/test_omega_direct.py
git commit -m "feat(omega_direct): add reflection_element (canonical det<0 reflection)"
```

---

### Task 4: Element transport builder (exp-free forward)

**Files:**
- Modify: `vfe3/geometry/transport.py` (add function after `compute_transport_operators`, ~`:984`)
- Test: `tests/test_omega_direct.py`

**Interfaces:**
- Consumes: `FactoredTransport` (`transport.py:23`), `GaugeGroup`.
- Produces: `build_transport_from_element(omega, group) -> FactoredTransport | dict` filling `exp_phi := U_i`, `exp_neg_phi := U_j⁻¹` (fp64 inverse island); a `FactoredTransport` for equal-block groups (block_glk), a `{'exp_phi','exp_neg_phi','Omega'}` dict otherwise (glk). Downstream `transport_mean` / `transport_covariance` consume both unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_omega_direct.py
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (build_transport_from_element, compute_transport_operators,
                                      transport_mean, FactoredTransport)
from vfe3.geometry.generators import reflection_element

def test_element_transport_cocycle_and_identity():
    grp = get_group("glk")(K=4)
    g = torch.Generator().manual_seed(7)
    # Random near-identity invertible frames per token.
    U = torch.eye(4) + 0.1 * torch.randn(1, 3, 4, 4, generator=g)
    built = build_transport_from_element(U, grp)
    omega = built["Omega"]                                    # glk -> dict path
    # cocycle: Omega_ij Omega_jk = Omega_ik  (U_i U_j^{-1} U_j U_k^{-1} = U_i U_k^{-1})
    lhs = omega[0, 0, 1] @ omega[0, 1, 2]
    assert torch.allclose(lhs, omega[0, 0, 2], atol=1e-4)
    # identity frames -> Omega = I
    I = torch.eye(4).expand(1, 3, 4, 4)
    omega_I = build_transport_from_element(I, grp)["Omega"]
    assert torch.allclose(omega_I, torch.eye(4).expand(1, 3, 3, 4, 4), atol=1e-6)

def test_element_transport_matches_phi_path_when_U_equals_exp_phi():
    grp = get_group("glk")(K=3)
    g = torch.Generator().manual_seed(3)
    phi = 0.2 * torch.randn(1, 3, grp.generators.shape[0], generator=g)
    ref = compute_transport_operators(phi, grp, gauge_mode="learned")
    U = ref["exp_phi"]                                        # U_i := exp(phi_i)
    got = build_transport_from_element(U, grp)["Omega"]
    assert torch.allclose(got, ref["Omega"], atol=1e-5)      # same cocycle, exp-free assembly

def test_element_transport_reaches_det_negative():
    grp = get_group("glk")(K=4)
    R = reflection_element(4)
    U = torch.stack([torch.eye(4), R], dim=0).unsqueeze(0)   # (1, 2, 4, 4): token0 det>0, token1 det<0
    omega = build_transport_from_element(U, grp)["Omega"]
    assert torch.det(omega[0, 0, 1]) < 0                     # I @ R^{-1} has det < 0

def test_element_transport_block_glk_is_factored():
    grp = get_group("block_glk")(K=4, n_heads=2)             # irrep_dims [2,2]
    U = torch.eye(4).expand(1, 3, 4, 4).contiguous()
    built = build_transport_from_element(U, grp)
    assert isinstance(built, FactoredTransport)
    mu = torch.randn(1, 3, 4)
    mt = transport_mean(built, mu)                           # (1,3,3,4) via the factored fast path
    assert mt.shape == (1, 3, 3, 4)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_omega_direct.py -k element_transport -x`
Expected: FAIL (`ImportError: cannot import name 'build_transport_from_element'`).

- [ ] **Step 3: Implement** — add to `vfe3/geometry/transport.py` after `compute_transport_operators` (~`:984`)

```python
def build_transport_from_element(
    omega:  torch.Tensor,             # (B, N, K, K) per-token GL(K) element U_i (block-diagonal for block_glk)
    group:  GaugeGroup,
) -> 'FactoredTransport | TransportDict':
    r"""Exp-free flat cocycle from a stored group element: Omega_ij = U_i U_j^{-1}.

    The 'omega_direct' parameterization stores the frame as the element U_i itself rather than the
    Lie-algebra coordinate phi_i, so the transport is assembled WITHOUT any matrix exponential --
    only the inverse U_j^{-1}. The FactoredTransport / builder-dict slots exp_phi / exp_neg_phi are
    filled with U_i and U_j^{-1} directly; every downstream consumer (transport_mean,
    transport_covariance, RoPE) reads only those two slots, so nothing else changes.

    U_j^{-1} is computed in a float64 island (the congruence Omega Sigma Omega^T squares cond(U); the
    inverse degrades as det U -> 0, which the free-energy barrier keeps a trained U away from). For
    the equal-block groups (block_glk) a FactoredTransport is returned so the per-head fast paths run;
    for a single block (glk) the dense {'exp_phi','exp_neg_phi','Omega'} dict is returned (matching
    compute_transport_operators' return shape).
    """
    with torch.amp.autocast(omega.device.type, enabled=False):    # inverse never in bf16/fp16
        u_inv = torch.linalg.inv(omega.double()).to(omega.dtype)  # (B, N, K, K)
    block_dims = group.irrep_dims
    if len(block_dims) > 1 and len(set(block_dims)) == 1:
        return FactoredTransport(exp_phi=omega, exp_neg_phi=u_inv, irrep_dims=list(block_dims))
    Omega = torch.einsum("...ikl,...jlm->...ijkm", omega, u_inv)   # (B, N, N, K, K)
    return {"exp_phi": omega, "exp_neg_phi": u_inv, "Omega": Omega}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_omega_direct.py -k element_transport -x`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add vfe3/geometry/transport.py tests/test_omega_direct.py
git commit -m "feat(omega_direct): exp-free element transport Omega_ij = U_i U_j^{-1}"
```

---

### Task 5: E-step dispatch — thread `gauge_parameterization` + `omega` through the funnel

**Files:**
- Modify: `vfe3/inference/e_step.py` — `build_belief_transport` (`:117-190`), `_transport` (`:41-98`), `e_step_iteration` (`:420`), `e_step` (`:745`), `free_energy_value` (`:206`), and the internal call sites (`:402, 512, 532, 816, 886`, `:291`).
- Test: `tests/test_omega_direct.py`

**Interfaces:**
- Consumes: `build_transport_from_element` (Task 4).
- Produces: `build_belief_transport(..., gauge_parameterization="phi", omega=None)` returns the element transport when `gauge_parameterization="omega_direct"`, else the unchanged phi path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_omega_direct.py
from vfe3.inference.e_step import build_belief_transport

def test_build_belief_transport_omega_direct_branch():
    grp = get_group("glk")(K=3)
    phi = torch.zeros(1, 3, grp.generators.shape[0])          # ignored on the omega path
    U = torch.eye(3) + 0.1 * torch.randn(1, 3, 3, 3, generator=torch.Generator().manual_seed(1))
    built = build_belief_transport(phi, grp, gauge_parameterization="omega_direct", omega=U)
    ref = build_transport_from_element(U, grp)["Omega"]
    assert torch.allclose(built["Omega"], ref, atol=1e-6)
    # default axis unchanged: phi path returns its usual object
    phi_out = build_belief_transport(phi, grp)                # default 'phi' path, phi=0 -> Omega = I
    eye = torch.eye(3).expand(1, 3, 3, 3, 3)
    assert torch.allclose(phi_out, eye, atol=1e-6)            # glk single-block returns a dense Omega tensor
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_omega_direct.py::test_build_belief_transport_omega_direct_branch -x`
Expected: FAIL (`build_belief_transport() got an unexpected keyword argument 'gauge_parameterization'`).

- [ ] **Step 3: Import the new builder** — add to the `from vfe3.geometry.transport import (...)` block at `vfe3/inference/e_step.py:26-37`

```python
    build_transport_from_element,
```

- [ ] **Step 4: Add the two params + dispatch branch to `build_belief_transport`** — add to its signature (after `phi`/`group`, in the keyword-only block near `:122`)

```python
    gauge_parameterization: str = "phi",                   # 'phi' (exp path) | 'omega_direct' (stored element)
    omega:              Optional[torch.Tensor] = None,     # (B, N, K, K) per-token U_i (omega_direct only)
```

and insert the dispatch as the FIRST branch of the fused/dense decision at `:170` (before `if _can_fuse_flat(...)`):

```python
    if gauge_parameterization == "omega_direct":
        built = build_transport_from_element(omega, group)
    elif _can_fuse_flat(transport_mode, group):
```

(the existing `_can_fuse_flat` block becomes the `elif`; the `else: _transport(...)` and the RoPE wrap at `:188-190` are unchanged, so an omega base is RoPE-wrappable too).

- [ ] **Step 5: Thread the axis through the callers.** In `e_step_iteration` (`:420`) add `gauge_parameterization: str = "phi"` to its signature, and at its two `build_belief_transport` sites (`:512`, `:532`) pass `gauge_parameterization=gauge_parameterization, omega=belief.omega`. In `e_step` (`:745`), read the axis out of kwargs like `transport_mode_kw` does (`:805`): `gauge_param_kw = kwargs.get("gauge_parameterization", "phi")`, forward `gauge_parameterization=gauge_param_kw` into each `e_step_iteration` call, and at the two hoist sites (`:816`, `:886`) pass `gauge_parameterization=gauge_param_kw, omega=belief.omega`. In `free_energy_value` add the param and pass `gauge_parameterization`/`omega=belief.omega` into its `_transport` call (`:291`) — see Task 5 note below on `_transport`.

- [ ] **Step 6: Add the params to `_transport`** (`:41`) — add `gauge_parameterization: str = "phi"` and `omega: Optional[torch.Tensor] = None`, and at the top of its body branch to the element builder before the `build = get_transport(...)` dispatch:

```python
    if gauge_parameterization == "omega_direct":
        built = build_transport_from_element(omega, group)
        return built["Omega"] if isinstance(built, dict) else built.to_dense_omega()
```

(the diagnostic `_transport` returns a dense Omega, so materialize it; the factored container's `to_dense_omega()` is byte-identical to the einsum.)

- [ ] **Step 7: Run to verify pass + regressions**

Run: `pytest tests/test_omega_direct.py::test_build_belief_transport_omega_direct_branch tests/test_round3_geometry_wiring.py -x`
Expected: PASS (the new branch works; the registry-routing regression suite still green).

- [ ] **Step 8: Commit**

```bash
git add vfe3/inference/e_step.py tests/test_omega_direct.py
git commit -m "feat(omega_direct): dispatch the element transport through the e_step funnel"
```

---

### Task 6: PriorBank — gated `omega_embed` table + encode populates `belief.omega`

**Files:**
- Modify: `vfe3/model/prior_bank.py:131-182` (ctor args + table), `:894-911` (`_encode_per_token`)
- Modify: `vfe3/model/model.py:144-163` (pass new ctor args)
- Test: `tests/test_omega_direct.py`

**Interfaces:**
- Consumes: `reflection_element` (Task 3).
- Produces: `PriorBank(..., gauge_parameterization=, irrep_dims=, omega_reflection=)`; when `omega_direct`, a `(V, K, K)` `omega_embed` (identity init, block-diagonal for block_glk since identity is diagonal); `encode()` returns `BeliefState(..., omega=(B,N,K,K))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_omega_direct.py
from vfe3.model.prior_bank import PriorBank

def test_prior_bank_omega_table_gated_and_encodes_identity():
    # default (phi) path: no omega table, no omega on the belief
    pb_phi = PriorBank(vocab_size=6, K=4, n_gen=16)
    assert not hasattr(pb_phi, "omega_embed")
    assert pb_phi.encode(torch.zeros(1, 3, dtype=torch.long)).omega is None
    # omega_direct path: table exists, identity init, belief carries (B,N,K,K)
    pb = PriorBank(vocab_size=6, K=4, n_gen=16, gauge_parameterization="omega_direct", irrep_dims=[4])
    assert pb.omega_embed.shape == (6, 4, 4)
    assert torch.allclose(pb.omega_embed, torch.eye(4).expand(6, 4, 4), atol=1e-7)
    b = pb.encode(torch.zeros(1, 3, dtype=torch.long))
    assert b.omega.shape == (1, 3, 4, 4)
    assert torch.allclose(b.omega, torch.eye(4).expand(1, 3, 4, 4), atol=1e-7)

def test_prior_bank_omega_reflection_seeds_det_negative():
    pb = PriorBank(vocab_size=6, K=4, n_gen=16, gauge_parameterization="omega_direct",
                   irrep_dims=[4], omega_reflection="init_seed")
    dets = torch.det(pb.omega_embed)
    assert (dets < 0).any()                                  # some tokens seeded into det<0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_omega_direct.py -k prior_bank -x`
Expected: FAIL (`__init__() got an unexpected keyword argument 'gauge_parameterization'`).

- [ ] **Step 3: Add ctor args** to `PriorBank.__init__` at `vfe3/model/prior_bank.py:157` (end of the keyword-only block, before `) -> None:`)

```python
        gauge_parameterization: str                 = "phi",
        irrep_dims:             Optional[List[int]] = None,
        omega_reflection:       str                 = "off",
```

- [ ] **Step 4: Stash + create the gated table.** Add to the scalar-stash block (`:176`): `self.gauge_parameterization = gauge_parameterization`. Then create the table at the END of `__init__` (after the untied-decode gate, so RNG order is preserved), following the gated-table idiom:

```python
        # omega_direct: a per-token GL(K) group element table (identity init -> step-0 == trivial gauge).
        # Created ONLY on the omega_direct path so the default state_dict is byte-identical. Block-
        # diagonal by construction for block_glk (identity is diagonal; the group retraction keeps it so).
        if gauge_parameterization == "omega_direct":
            eye_K = torch.eye(K)
            self.omega_embed = nn.Parameter(eye_K.expand(vocab_size, K, K).clone())
            if omega_reflection == "init_seed":
                from vfe3.geometry.generators import reflection_element
                R = reflection_element(K)
                with torch.no_grad():                        # seed every OTHER token into the det<0 sheet
                    self.omega_embed[1::2] = R
```

- [ ] **Step 5: Populate `omega` in `_encode_per_token`** (`vfe3/model/prior_bank.py:894-911`) — before the `return`, add the lookup and pass it:

```python
    omega = pb.omega_embed[token_ids] if getattr(pb, "gauge_parameterization", "phi") == "omega_direct" else None
    sigma = sigma_diag if pb.diagonal_covariance else torch.diag_embed(sigma_diag)
    return BeliefState(mu=mu, sigma=sigma, phi=phi, omega=omega)
```

- [ ] **Step 6: Pass the new args at the construction site** — `vfe3/model/model.py:145-158`, add to the `PriorBank(...)` call:

```python
            gauge_parameterization=cfg.gauge_parameterization,
            irrep_dims=list(self.group.irrep_dims),
            omega_reflection=cfg.omega_reflection,
```

- [ ] **Step 7: Run to verify pass**

Run: `pytest tests/test_omega_direct.py -k prior_bank -x`
Expected: PASS. Then `pytest tests/test_prior_bank.py` (or the bank's suite) — expect green (default path untouched: no `omega_embed`, `omega=None`).

- [ ] **Step 8: Commit**

```bash
git add vfe3/model/prior_bank.py vfe3/model/model.py tests/test_omega_direct.py
git commit -m "feat(omega_direct): gated omega_embed table + encode populates belief.omega"
```

---

### Task 7: Model forward threading — end-to-end `omega_direct` forward

**Files:**
- Modify: `vfe3/model/model.py` — forward E-step knob bag (thread `gauge_parameterization`), the share build (`:771`), `_gamma_energy` (`:1178-1204`) + its 4 callers (`:1235, 1267, 1317, 1549`), diagnostics `_transport` sites (`:1656, 1930, 2039`).
- Test: `tests/test_omega_direct.py`

**Interfaces:**
- Produces: a `VFEModel` whose full forward runs under `gauge_parameterization="omega_direct"` and, at identity init, produces logits byte-equal to the trivial-gauge (`Omega = I`) forward.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_omega_direct.py
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

def _cfg(**over):
    base = dict(vocab_size=6, embed_dim=4, n_heads=1, max_seq_len=4, n_layers=1, n_e_steps=2,
                gauge_group="glk", family="gaussian_full", transport_mode="flat",
                pos_rotation="none", use_head_mixer=False, use_prior_bank=True, decode_mode="full",
                e_phi_lr=0.0)
    base.update(over); return VFE3Config(**base)

def test_full_model_forward_omega_direct_finite_and_matches_identity_gauge():
    tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
    torch.manual_seed(0); m_od = VFEModel(_cfg(gauge_parameterization="omega_direct"))
    with torch.no_grad():
        logits_od = m_od(tok)[0]
    assert torch.isfinite(logits_od).all()
    # identity-init omega_direct == phi path with frames zeroed (both give Omega = I)
    torch.manual_seed(0); m_phi = VFEModel(_cfg(gauge_parameterization="phi"))
    with torch.no_grad():
        m_phi.prior_bank.phi_embed.zero_()
        if hasattr(m_phi, "pos_phi_free"):
            m_phi.pos_phi_free.zero_()
        logits_phi = m_phi(tok)[0]
    assert torch.allclose(logits_od, logits_phi, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_omega_direct.py::test_full_model_forward_omega_direct_finite_and_matches_identity_gauge -x`
Expected: FAIL (the forward E-step does not yet thread `gauge_parameterization`, so `belief.omega` is ignored and the phi path runs on an all-zero `phi` — actually equal here, but the assertion may pass spuriously; if it does, force `omega_reflection="init_seed"` in a second arm to prove the omega frame is actually consumed — see Step 5).

- [ ] **Step 3: Thread `gauge_parameterization` into the forward E-step knob bag.** Locate the forward's `e_step(...)` / `vfe_stack(...)` invocation in `VFEModel.forward` (the block that builds the loose-kwarg bag from `self.cfg`, near the share build at `model.py:757-816`) and add `gauge_parameterization=self.cfg.gauge_parameterization` to the kwarg bag passed down to `e_step`. (Because `e_step` reads it out of `**kwargs`, Task 5 Step 5, this is a one-line addition to the bag.)

- [ ] **Step 4: Add `gauge_parameterization` to the share build** at `model.py:771` and the `_gamma_energy` build at `:1204`:

```python
                    gauge_parameterization=self.cfg.gauge_parameterization,
```

and for `_gamma_energy` (`:1204`), also pass `omega=omega_frame` where `omega_frame` is threaded in via the signature change below.

- [ ] **Step 5: Thread `omega` through `_gamma_energy`.** Change its signature (`model.py:1178`) to accept the element frame:

```python
    def _gamma_energy(self, token_ids, phi, *, omega=None, s_belief=None):
```

pass `omega=omega` into its `build_belief_transport` at `:1204`, and update the 4 callers (`:1235, 1267, 1317, 1549`) to forward the belief's omega alongside phi (e.g. `self._gamma_energy(token_ids, phi, omega=beliefs.omega)`; at `:1317` use `out.omega.unsqueeze(0)` mirroring the `out.phi.unsqueeze(0)` there, guarding `None`). Where a caller has no belief omega in scope, pass `omega=None` (phi path).

- [ ] **Step 6: Thread the axis into the diagnostics `_transport` sites** (`model.py:1656, 1930, 2039`) — add `gauge_parameterization=cfg.gauge_parameterization, omega=<belief>.omega` to each (`out.omega`, `belief.omega`, `belief.omega` respectively). These are eval-only; they make det < 0 visible in diagnostics.

- [ ] **Step 7: Add the non-vacuity arm to the test** (prove the frame is consumed)

```python
# tests/test_omega_direct.py  (extend the test)
def test_omega_direct_reflection_changes_logits():
    tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
    torch.manual_seed(0); m0 = VFEModel(_cfg(gauge_parameterization="omega_direct"))
    torch.manual_seed(0); m1 = VFEModel(_cfg(gauge_parameterization="omega_direct", omega_reflection="init_seed"))
    with torch.no_grad():
        d = (m0(tok)[0] - m1(tok)[0]).abs().max()
    assert d > 1e-4                                          # the stored det<0 frame actually feeds the forward
```

- [ ] **Step 8: Run to verify pass**

Run: `pytest tests/test_omega_direct.py -k "full_model_forward or reflection_changes" -x`
Expected: PASS (identity parity + reflection has bite).

- [ ] **Step 9: Commit**

```bash
git add vfe3/model/model.py tests/test_omega_direct.py
git commit -m "feat(omega_direct): thread the element frame through the full model forward"
```

---

### Task 8: Group-manifold optimizer — learn `omega_embed` by Lie-exp / Cayley retraction

**Files:**
- Modify: `vfe3/geometry/lie_ops.py` (add `retract_omega`), `vfe3/gauge_optim.py:143-207` (add the `"omega"` branch), `vfe3/train.py:136-267` (route `omega_embed`, select the optimizer)
- Test: `tests/test_omega_direct.py`

**Interfaces:**
- Consumes: `embed_phi`, `extract_phi`, `gram_pinv` (`lie_ops.py`).
- Produces: `retract_omega(U, xi, generators, *, mode)`; a `GaugeNaturalGradAdamW` `"omega"` group branch that steps `U ← U·retr(−lr·Gram⁻¹ proj_g(Uᵀ E))` per active row; `build_optimizer` routes `omega_embed` and selects `GaugeNaturalGradAdamW` when `omega_direct`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_omega_direct.py
from vfe3.geometry.lie_ops import retract_omega
from vfe3.geometry.generators import generate_glk

def test_retract_omega_stays_in_component_and_group():
    G = generate_glk(3)                                       # (9,3,3)
    U = torch.eye(3).expand(4, 3, 3).contiguous()
    xi = 0.05 * torch.randn(4, 9, generator=torch.Generator().manual_seed(0))
    for mode in ("lie_exp", "cayley"):
        Un = retract_omega(U, xi, G, mode=mode)
        assert Un.shape == (4, 3, 3)
        assert (torch.det(Un) > 0).all()                     # retraction preserves the det>0 component
        # a det<0 base stays det<0 (component preserved)
        Rneg = U.clone(); Rneg[:, 0, 0] = -1.0
        assert (torch.det(retract_omega(Rneg, xi, G, mode=mode)) < 0).all()

def test_gauge_optim_omega_step_moves_active_rows_only():
    from vfe3.gauge_optim import GaugeNaturalGradAdamW
    G = generate_glk(3)
    U = torch.nn.Parameter(torch.eye(3).expand(5, 3, 3).contiguous())
    opt = GaugeNaturalGradAdamW([{"params": [U], "lr": 0.1, "omega": True, "weight_decay": 0.0}],
                                G, [3], gauge_momentum=0.0)
    U.grad = torch.zeros_like(U)
    U.grad[2] = torch.randn(3, 3, generator=torch.Generator().manual_seed(1))   # only row 2 active
    before = U.data.clone()
    opt.step()
    assert torch.allclose(U.data[0], before[0])              # inactive rows untouched
    assert not torch.allclose(U.data[2], before[2])          # active row moved
    assert torch.det(U.data[2]) > 0                           # still in GL+(3)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_omega_direct.py -k "retract_omega or gauge_optim_omega" -x`
Expected: FAIL (`ImportError: cannot import name 'retract_omega'`).

- [ ] **Step 3: Implement `retract_omega`** — append to `vfe3/geometry/lie_ops.py`

```python
def retract_omega(
    U:          torch.Tensor,             # (..., K, K) current group element
    xi:         torch.Tensor,             # (..., n_gen) descent step in algebra coords (already -lr scaled)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    mode:       str = "lie_exp",
) -> torch.Tensor:                        # (..., K, K) retracted element U @ retr(xi)
    r"""Group-manifold retraction of a stored GL(K) element: U_new = U @ retr(xi . G).

    'lie_exp' (default, principled): retr = exp, follows the one-parameter subgroup; exp of the small
    near-identity step xi . G is well-conditioned and stays in U's det component (det exp = e^{tr} > 0).
    'cayley' (exp-free): retr(A) = (I - A/2)^{-1}(I + A/2), a second-order retraction, also component-
    preserving. Because xi is valued in the algebra span (block-diagonal for block_glk), retr(xi) is
    block-diagonal and U @ retr(xi) keeps U's block structure exactly.
    """
    A = embed_phi(xi, generators)                             # (..., K, K) algebra matrix
    if mode == "lie_exp":
        step = torch.linalg.matrix_exp(A)
    elif mode == "cayley":
        K = A.shape[-1]
        eye = torch.eye(K, dtype=A.dtype, device=A.device)
        step = torch.linalg.solve(eye - 0.5 * A, eye + 0.5 * A)
    else:
        raise ValueError(f"omega retract mode must be 'lie_exp' or 'cayley', got {mode!r}")
    return U @ step
```

- [ ] **Step 4: Add the `"omega"` branch to `GaugeNaturalGradAdamW.step`** — inside the `for group in self.param_groups:` loop at `vfe3/gauge_optim.py:145`, before the existing `if not group.get("gauge", False): continue`, add:

```python
            if group.get("omega", False):
                lr   = group["lr"]
                Gd   = self._generators.to(device=group["params"][0].device, dtype=group["params"][0].dtype)
                mode = getattr(self, "_omega_retract_mode", "lie_exp")
                from vfe3.geometry.lie_ops import extract_phi, gram_pinv, retract_omega
                gp = gram_pinv(Gd)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    E   = p.grad                                       # (V, K, K) dF/dU
                    U   = p.data
                    act = E.reshape(E.shape[0], -1).abs().sum(dim=-1) > 0
                    if not bool(act.any()):
                        p.grad = None
                        continue
                    Ua, Ea = U[act], E[act]
                    # natural-gradient tangent xi = Gram^{-1} proj_g(U^T E) (extract_phi does exactly this)
                    xi = extract_phi(torch.einsum("...lk,...lm->...km", Ua, Ea), Gd, gram_pinv_=gp)
                    U[act] = retract_omega(Ua, -lr * xi, Gd, mode=mode)
                    p.grad = None
                continue
```

Store the mode in `__init__` (add to `gauge_optim.py:75-86` signature `omega_retract_mode: str = "lie_exp"`, and `self._omega_retract_mode = omega_retract_mode` in the body).

- [ ] **Step 5: Route `omega_embed` + select the optimizer in `build_optimizer`.** In `vfe3/train.py`, where the phi group is built (`:136-140`), add an omega group when the axis is on, and force `GaugeNaturalGradAdamW`:

```python
    omega_direct = cfg.gauge_parameterization == "omega_direct"
    if omega_direct:
        groups.append({"params": [pb.omega_embed], "lr": cfg.m_phi_lr,
                       "weight_decay": 0.0, "role": "phi", "omega": True})
```

(insert after the base `groups = [...]` list at `:148-152`), and in the optimizer selection (`:250-267`) change the guard to `if nat or omega_direct:` and pass `omega_retract_mode=cfg.omega_retract_mode` to the `GaugeNaturalGradAdamW(...)` constructor. The exact-coverage assertion (`:242-248`) then finds `omega_embed` grouped.

- [ ] **Step 6: Run to verify pass**

Run: `pytest tests/test_omega_direct.py -k "retract_omega or gauge_optim_omega" -x`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add vfe3/geometry/lie_ops.py vfe3/gauge_optim.py vfe3/train.py tests/test_omega_direct.py
git commit -m "feat(omega_direct): group-manifold natural-gradient optimizer (lie_exp/cayley)"
```

---

### Task 9: End-to-end gauge-invariance + det < 0 property tests

**Files:**
- Test: `tests/test_omega_direct.py`

**Interfaces:** none new — this pins the theory contract on the assembled system.

- [ ] **Step 1: Write the test** (mirror the t8 two-arm structure, `tests/test_gauge_groups.py:188-235`, on the omega path)

```python
# tests/test_omega_direct.py
def test_omega_direct_full_model_gauge_invariance():
    """A global gauge transform of the tied prior tables leaves omega_direct decode logits invariant
    (fp64), and the linear-decode arm has bite (fp32) -- the same t8 contract as the phi path."""
    # omega_direct is glk-scoped here; family=gaussian_full + decode_mode=full represent the general
    # g in GL(4), and Sigma=I makes the congruence g Sigma g^T representable. Mirrors t8 (test_gauge_groups.py:188).
    def delta(dbl, **over):
        torch.manual_seed(0); m = VFEModel(_cfg(gauge_parameterization="omega_direct", **over))
        with torch.no_grad():
            m.prior_bank.omega_embed.copy_(torch.eye(4).expand(6, 4, 4))       # frames -> identity
            m.prior_bank.sigma_log_embed.zero_()                              # Sigma = I
            if hasattr(m, "pos_phi_free"):
                m.pos_phi_free.zero_()
        if dbl: m = m.double()
        m.eval()
        gen = m.group.generators.to(torch.float64 if dbl else torch.float32)
        c = 0.2 * torch.randn(gen.shape[0], generator=torch.Generator().manual_seed(1)).to(gen.dtype)
        g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", c, gen))         # random g in GL(4)
        tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            l0 = m(tok)[0].clone()
            m.prior_bank.mu_embed.copy_(torch.einsum("kl,vl->vk", g, m.prior_bank.mu_embed))
            # co-transform the stored frame: U -> g U (the cocycle U_i U_j^{-1} is g-invariant)
            m.prior_bank.omega_embed.copy_(torch.einsum("kl,vlm->vkm", g, m.prior_bank.omega_embed))
            l1 = m(tok)[0].clone()
        return float((l0 - l1).abs().max())
    assert delta(dbl=True) < 1e-5
    assert delta(dbl=False, use_prior_bank=False) > 1e-4       # linear decode does not co-transform -> bite
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_omega_direct.py::test_omega_direct_full_model_gauge_invariance -x`
Expected: PASS (pure path invariant; broken arm has bite).

- [ ] **Step 3: Commit**

```bash
git add tests/test_omega_direct.py
git commit -m "test(omega_direct): end-to-end gauge-invariance + det<0 reachability"
```

---

### Task 10: Off-funnel threading — KV cache + `cache_supported`

**Files:**
- Modify: `vfe3/inference/belief_cache.py:56-87` (`cache_supported`), `:104-119` (element-frame rebuild)
- Test: `tests/test_omega_direct.py`

**Interfaces:** produces a KV-cache rebuild that uses the stored element (`beliefs.omega`) when `omega_direct`, and a `cache_supported` clause that admits it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_omega_direct.py
from vfe3.inference.belief_cache import cache_supported

def test_cache_supported_admits_omega_direct_flat():
    assert cache_supported(_cfg(gauge_parameterization="omega_direct")) is True
    assert cache_supported(_cfg(gauge_parameterization="phi")) is True
```

- [ ] **Step 2: Run to verify failure/behavior**

Run: `pytest tests/test_omega_direct.py::test_cache_supported_admits_omega_direct_flat -x`
Expected: FAIL only if `cache_supported` currently excludes it; if it already returns True (it keys on `transport_mode=="flat"`), instead make the rebuild correct (Step 3) and keep this as a guard test.

- [ ] **Step 3: Element-frame KV rebuild** — at `vfe3/inference/belief_cache.py:114-117`, branch on the axis (mirroring the mixed-frame einsum with stored elements):

```python
    if cfg.gauge_parameterization == "omega_direct":
        U_q = beliefs.omega[:, N:]                                             # (B', L, K, K) query frames
        with torch.amp.autocast(U_q.device.type, enabled=False):
            U_k_inv = torch.linalg.inv(beliefs.omega.double()).to(U_q.dtype)   # (B', M, K, K)
        omega = torch.einsum("bikl,bjlm->bijkm", U_q, U_k_inv)                 # (B', L, M, K, K)
    else:
        exp_q     = compute_transport_operators(phi_q, group)["exp_phi"]
        exp_neg_k = compute_transport_operators(phi_k, group)["exp_neg_phi"]
        omega     = torch.einsum("bikl,bjlm->bijkm", exp_q, exp_neg_k)
```

(the `N` slice offset mirrors the existing `phi_q = beliefs.phi[:, N:]` at `:109-111`.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_omega_direct.py -k cache_supported -x`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vfe3/inference/belief_cache.py tests/test_omega_direct.py
git commit -m "feat(omega_direct): element-frame KV-cache rebuild + cache_supported clause"
```

---

### Task 11: Ablation surface + stale entry-point comments

**Files:**
- Modify: `ablation.py:1031-1045` (remove from `NON_SWEPT_FIELDS`, fix comment), `:375-399` region (add sweep arm), `:139`
- Test: `tests/test_omega_direct.py` (a light config-build check)
- NOTE: `train_vfe3.py:128` and `scaling.py:148` carry the same stale comment but have the user's uncommitted WIP; per CLAUDE.md do NOT touch/commit WIP files — their comment refresh is deferred to a later, WIP-clear session.

**Interfaces:** makes `gauge_parameterization` a sweepable ablation axis.

- [ ] **Step 1: Remove `"gauge_parameterization"` from `NON_SWEPT_FIELDS`** (`ablation.py:1043`) and rewrite the reason comment (`:1033`) to note it is now live (drop the "config-rejected" line).

- [ ] **Step 2: Add a categorical sweep arm** in the `SWEEPS` dict (near the `gauge_group` entry, `ablation.py:383`), mirroring that template with a `requires`:

```python
    "gauge_parameterization": {
        "description": "gauge frame chart: phi (exp coords) vs omega_direct (stored GL(K) element)",
        "configs": [
            {"label": "phi",          "gauge_parameterization": "phi"},
            {"label": "omega_direct", "gauge_parameterization": "omega_direct", "gauge_group": "glk",
             "use_head_mixer": False},
        ],
        "requires": {"transport_mode": "flat"},
    },
```

- [ ] **Step 3: Refresh the stale entry-point comment** — `ablation.py:139` currently reads `# "phi" | "omega_direct" (omega_direct: live-rejected, no belief source)`. Replace with:

```python
    gauge_parameterization    = "phi",        # "phi" | "omega_direct" (stored GL(K) element; glk/block_glk, flat)
```

(Do NOT edit `train_vfe3.py:128` / `scaling.py:148` — they carry uncommitted WIP; deferred.)

- [ ] **Step 4: Light check** (the sweep config builds)

```python
# tests/test_omega_direct.py
def test_ablation_omega_direct_arm_builds():
    cfg = _cfg(gauge_parameterization="omega_direct", gauge_group="glk", use_head_mixer=False)
    assert cfg.gauge_parameterization == "omega_direct"
```

- [ ] **Step 5: Run + full-suite regression**

Run: `pytest tests/test_omega_direct.py` then `pytest --junitxml=out.xml` and read `testsuite tests=/failures=/errors=`.
Expected: `tests/test_omega_direct.py` fully green; no regressions in the existing suite (default `phi` path byte-identical).

- [ ] **Step 6: Update the edits doc + commit**

Append the Phase-1 summary to `docs/2026-07-07-edits.md` (create if absent, one doc per day), then:

```bash
git add ablation.py tests/test_omega_direct.py docs/2026-07-07-edits.md
git commit -m "feat(omega_direct): sweepable ablation axis + refresh stale entry-point comment"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-07-omega-direct-gauge-parameterization-design.md`):
- Chart change / cocycle `U_i U_j⁻¹` → Task 4. Strict equivariance → Task 9. Same param count (glk) → Task 6 (block_glk stored full `(V,K,K)`; compact storage flagged Phase 2).
- Manifold optimizer (Lie-exp default, Cayley opt-in, natural gradient `Gram⁻¹ proj_g(Uᵀ E)`) → Tasks 8 (`retract_omega`, optimizer branch). Forward exp-free, fp64 inverse island → Task 4.
- Discrete det-sign init (reflection) → Tasks 3, 6. STE (learnable sign) → deferred to spec Phase 2, not in this plan (honestly out of scope; the plan delivers init-time det < 0).
- Config surface + default-OFF pure path → Task 1; per-group reach → Task 1 gates. Dispatch seam → Task 5. Belief field → Task 2. Off-funnel (KV cache) → Task 10; viz/extract diagnostics threading is deferred (eval-only, listed in the spec but not load-bearing for training/decoding — flagged as a follow-up, not silently dropped). Tests + ablation → Tasks 9, 11.

**Placeholder scan:** every code step carries the actual code; test commands carry exact `pytest` invocations and expected PASS/FAIL. No "TBD"/"add validation"/"similar to Task N".

**Type consistency:** `build_transport_from_element(omega, group)` (Task 4) is consumed with those exact args in Tasks 5/10. `retract_omega(U, xi, generators, *, mode)` (Task 8) matches its optimizer call. `omega_embed` shape `(V,K,K)` is consistent across Tasks 6/8/9. `BeliefState(..., omega=None)` field name `omega` is uniform across Tasks 2/5/6/7/9/10. Config fields `omega_retract_mode` / `omega_reflection` names match across Tasks 1/6/8.

**Known scope honesty (not silent):** (1) `block_glk` stores the frame as a full `(V,K,K)` table with off-blocks frozen at zero by the algebra-projected retraction — correct but not storage-compact; compaction is a Phase-2 optimization. (2) STE (learnable det-sign) and E-step frame refinement (`e_phi_lr>0`) are rejected at config in Phase 1 and deferred. (3) `viz/extract.py` diagnostic threading is deferred (eval-only).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-07-omega-direct-gl-k-phase1.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
