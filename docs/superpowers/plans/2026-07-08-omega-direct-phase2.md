# omega_direct Phase 2 — Implementation Plan (other groups + compact block storage)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Extend `gauge_parameterization="omega_direct"` from `glk`/`block_glk` to `tied_block_glk`, `sp`, `sp_n`, `so_k`, `so_n`; add the exact transpose inverse for the orthogonal (skew) groups with orthogonality-drift control; add fail-closed per-group reflection-seed cross-checks; and add opt-in compact block-diagonal storage. Spec: `docs/superpowers/specs/2026-07-08-omega-direct-phase2-design.md`.

**Architecture:** The frame is a per-token group element `U_i`; transport is the same cocycle `Ω_ij = U_i U_j⁻¹`; the optimizer's generator-basis projection is already group-agnostic. Per group the only differences are the inverse (`Uᵀ` for skew groups), the reflection seed (fail-closed), and orthogonality drift (SO groups). Compact storage stores `(V,H,d,d)`/`(V,d,d)` and assembles the block-diagonal `(B,N,K,K)` at encode; the belief still carries the assembled element so the downstream transport stack is untouched.

**Tech Stack:** Python, PyTorch (float32; fp64 island for the non-compact inverse). pytest.

## Global Constraints

- **Both the pure `phi` default AND the shipped `omega_direct` `glk`/`block_glk` path stay byte-identical.** Every new branch is gated on `group.skew_symmetric` / the group name / the default-OFF `omega_compact_storage`. `glk`/`block_glk` are `skew_symmetric=False` and keep the exact fp64 `inv`; compaction is opt-in default-OFF.
- No NN on the pure path. `omega_direct` remains a pure-path-preserving chart.
- float32; the non-compact inverse (`glk`/`block_glk`/`tied_block_glk`/`sp`/`sp_n`) stays in the fp64 island; skew groups (`so_k`/`so_n`) use the exact transpose (no island).
- **Do NOT touch the user's WIP files:** `CLAUDE.md`, `scaling.py`, `scaling_analysis.py`, `train_vfe3.py`, `vfe3/geometry/transport.py`'s uncommitted `TRANSPORT_CLAMP_MAX_NORM 15→20` edit region (line ~741 — do NOT stage that line; you WILL edit `transport.py` for the transpose branch at ~1004-1010, which is a different region — `git add` the file only if the WIP line is not staged; prefer `git add -p` or confirm `git diff --cached` shows only your hunk). Also never touch `docs/2026-07-08-edits.md` / `docs/audits/`.
- Function-signature convention (CLAUDE.md): tensors first, aligned columns, shape comments, LaTeX in docstrings, American English.
- CPU tests tiny: `K < 6`, single-digit dims. Groups built via `get_group("...")(K=..., n_heads=..., group_n=..., irrep_spec=...)`.
- Pytest: no extra `-q`; read the `N passed` line. Run focused tests SYNCHRONOUSLY; do NOT background the full suite.
- Per-group tiny configs: `so_k` `get_group("so_k")(K=4)`; `sp` `get_group("sp")(K=4)`; `tied_block_glk` `get_group("tied_block_glk")(K=4, n_heads=2)`; `so_n` `get_group("so_n")(K=4, group_n=3, irrep_spec=[("l0",1),("l1",1)])`; `sp_n` `get_group("sp_n")(K=5, group_n=4, irrep_spec=[("sym0",1),("sym1",1)])`. Element frames in tests: build a valid in-group `U` via `retract_omega(eye_expand, small_xi, grp.generators)` — NEVER `eye + eps*randn` (that leaves the group for so/sp).

## File Structure

- `vfe3/geometry/transport.py` — `build_transport_from_element`: branch the inverse on `group.skew_symmetric`.
- `vfe3/config.py` — widen the `omega_direct` group whitelist; add the fail-closed per-group reflection cross-check.
- `vfe3/model/prior_bank.py` — per-group reflection seed dispatch; retain `irrep_dims`; (Task 5) compact table + assembly.
- `vfe3/gauge_optim.py` — (Task 4) periodic re-orthogonalization for skew groups; (Task 5) per-block retraction for the compact table.
- `vfe3/geometry/lie_ops.py` — reuse `_from_equal_diag_blocks` scatter (Task 5); maybe a re-orthogonalization helper (Task 4).
- `ablation.py` — per-group omega_direct sweep arms.
- `tests/test_omega_direct.py` — per-group + compact-storage tests.

## Interfaces (locked)

```
# config.py — new field (near omega_retract_mode / omega_reflection)
omega_compact_storage: bool = False      # opt-in (V,H,d,d)/(V,d,d) block storage for equal-block groups
omega_reorth_every:    int  = 0          # SO-group re-orthogonalization cadence (M-steps); 0 = off/default

# transport.py build_transport_from_element(omega, group): u_inv = omega.transpose(-1,-2) if group.skew_symmetric else fp64-inv
# prior_bank.py PriorBank: retains irrep_dims; per-group seed dispatch; compact table under omega_compact_storage
# gauge_optim.py: "omega" branch re-orthogonalizes skew-group rows every omega_reorth_every; per-block retraction when compact
```

---

### Task 1: Exact transpose inverse for skew (orthogonal) groups

**Files:** Modify `vfe3/geometry/transport.py` (`build_transport_from_element`, ~1004-1010). Test: `tests/test_omega_direct.py`.

**Interfaces:** Produces `build_transport_from_element` that uses `Uᵀ` as the inverse when `group.skew_symmetric`, the fp64 `inv` otherwise. Byte-identical for `skew_symmetric=False` groups (all shipped omega_direct groups).

- [ ] **Step 1: Write the failing test** (append to `tests/test_omega_direct.py`). Build a valid orthogonal `U` for `so_k`, assert the builder's `exp_neg_phi` equals `Uᵀ` exactly and the cocycle holds:

```python
def test_element_transport_skew_group_uses_transpose_inverse():
    grp = get_group("so_k")(K=4)                              # skew_symmetric=True -> U in O(4)
    g = torch.Generator().manual_seed(11)
    xi = 0.2 * torch.randn(1, 3, grp.generators.shape[0], generator=g)
    from vfe3.geometry.lie_ops import retract_omega
    U = retract_omega(torch.eye(4).expand(1, 3, 4, 4).contiguous(), xi, grp.generators)  # in SO(4)
    built = build_transport_from_element(U, grp)              # single block -> dict
    assert torch.allclose(built["exp_neg_phi"], U.transpose(-1, -2), atol=0)   # EXACT transpose, no inv
    # cocycle telescopes
    om = built["Omega"]
    assert torch.allclose(om[0, 0, 1] @ om[0, 1, 2], om[0, 0, 2], atol=1e-5)
    # Omega is exactly orthogonal (isometry)
    eye = torch.eye(4).expand(1, 3, 3, 4, 4)
    assert torch.allclose(torch.einsum("...kl,...ml->...km", om, om), eye, atol=1e-4)

def test_element_transport_nonskew_still_uses_inv_byte_identical():
    grp = get_group("glk")(K=3)                               # skew_symmetric=False -> unchanged inv path
    U = (torch.eye(3) + 0.1 * torch.randn(1, 2, 3, 3, generator=torch.Generator().manual_seed(1)))
    built = build_transport_from_element(U, grp)
    ref_inv = torch.linalg.inv(U.double()).to(U.dtype)
    assert torch.allclose(built["exp_neg_phi"], ref_inv, atol=0)   # exact same fp64-inv path as shipped
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_omega_direct.py -k "skew_group_uses_transpose or nonskew_still_uses_inv" -x`. Expected: the skew test FAILS (current code always uses `inv`, so `exp_neg_phi != Uᵀ` exactly).

- [ ] **Step 3: Implement.** In `vfe3/geometry/transport.py` `build_transport_from_element` (read the current ~1004-1010 first), replace the unconditional inverse with a skew branch:

```python
    if group.skew_symmetric:
        u_inv = omega.transpose(-1, -2)                       # U^{-1} = U^T exactly (orthogonal group), free
    else:
        with torch.amp.autocast(omega.device.type, enabled=False):   # fp64 island (non-compact inverse)
            u_inv = torch.linalg.inv(omega.double()).to(omega.dtype)
```

(Keep the rest — the FactoredTransport-vs-dict branch and the Omega einsum — unchanged.)

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_omega_direct.py -k "skew_group_uses_transpose or nonskew_still_uses_inv"` (both pass), then `pytest tests/test_omega_direct.py` (whole file green — the shipped glk/block_glk tests still pass, confirming byte-identity).

- [ ] **Step 5: Commit** — `git add vfe3/geometry/transport.py tests/test_omega_direct.py` (confirm `git diff --cached` shows ONLY your ~1004-1010 hunk in transport.py, NOT the WIP clamp line ~741). `git commit -m "feat(omega_direct): exact transpose inverse for skew (orthogonal) groups"`.

---

### Task 2: Config gate widening + fail-closed per-group reflection cross-check

**Files:** Modify `vfe3/config.py` (the `omega_direct` block: whitelist ~910, and a new clause after the reflection static-tuple validation ~942). Modify `vfe3/model/prior_bank.py` (retain `irrep_dims` at ~160 — needed by later tasks and to know the group). Test: `tests/test_config.py`.

**Interfaces:** Produces a config that accepts `omega_direct` for all seven omega-eligible groups and raises a clear `ValueError` for an invalid per-group reflection seed. Adds `omega_reorth_every: int = 0` and `omega_compact_storage: bool = False` fields (used by Tasks 4/5).

- [ ] **Step 1: Write failing tests** (append to `tests/test_config.py`):

```python
def test_omega_direct_accepts_all_eligible_groups():
    common = dict(gauge_parameterization="omega_direct", transport_mode="flat", e_phi_lr=0.0,
                  lambda_gamma=0.0, s_e_step=False)
    for grp, over in (("glk", {}), ("block_glk", {"n_heads": 2}), ("tied_block_glk", {"n_heads": 2}),
                      ("sp", {}), ("so_k", {}),
                      ("so_n", {"group_n": 3, "irrep_spec": [("l0", 1), ("l1", 1)]}),
                      ("sp_n", {"embed_dim": 5, "group_n": 4, "irrep_spec": [("sym0", 1), ("sym1", 1)]})):
        kw = dict(embed_dim=over.pop("embed_dim", 4), n_heads=over.pop("n_heads", 1),
                  gauge_group=grp, use_head_mixer=False, **over, **common)
        assert VFE3Config(**kw).gauge_parameterization == "omega_direct"

def test_omega_direct_reflection_cross_check_fail_closed():
    import pytest
    base = dict(gauge_parameterization="omega_direct", transport_mode="flat", e_phi_lr=0.0,
                lambda_gamma=0.0, s_e_step=False, omega_reflection="init_seed", use_head_mixer=False)
    # reject init_seed where it is vacuous / group-incorrect
    with pytest.raises(ValueError):    # sp: no det<0 component
        VFE3Config(embed_dim=4, n_heads=1, gauge_group="sp", **base)
    with pytest.raises(ValueError):    # so_n: needs rho(O(N)) image, deferred
        VFE3Config(embed_dim=4, n_heads=1, gauge_group="so_n", group_n=3,
                   irrep_spec=[("l0", 1), ("l1", 1)], **base)
    with pytest.raises(ValueError):    # tied_block_glk: ambient seed breaks the tie, deferred
        VFE3Config(embed_dim=4, n_heads=2, gauge_group="tied_block_glk", **base)
    # accept init_seed where it is group-correct
    assert VFE3Config(embed_dim=4, n_heads=1, gauge_group="so_k", **base).omega_reflection == "init_seed"
    assert VFE3Config(embed_dim=4, n_heads=1, gauge_group="glk", **base).omega_reflection == "init_seed"
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_config.py -k "eligible_groups or cross_check" -x`. Expected FAIL (gate rejects the new groups).

- [ ] **Step 3: Add the two fields** near `omega_retract_mode`/`omega_reflection` in `config.py`:

```python
    omega_compact_storage:     bool  = False     # opt-in compact (V,H,d,d)/(V,d,d) block storage (equal-block groups)
    omega_reorth_every:        int   = 0         # SO-group re-orthogonalization cadence in M-steps (0 = off)
```

- [ ] **Step 4: Widen the whitelist** in the `if self.gauge_parameterization == "omega_direct":` block (read the current ~910 first). Replace the `gauge_group not in ("glk","block_glk")` check with the full omega-eligible set:

```python
            _OMEGA_GROUPS = ("glk", "block_glk", "tied_block_glk", "so_k", "so_n", "sp", "sp_n")
            if self.gauge_group not in _OMEGA_GROUPS:
                raise ValueError(
                    f"gauge_parameterization='omega_direct' supports gauge_group in {_OMEGA_GROUPS}; "
                    f"got {self.gauge_group!r}."
                )
```

Keep the existing `transport_mode!='flat'`, `e_phi_lr!=0`, and gamma/s-channel rejections unchanged.

- [ ] **Step 5: Add the fail-closed per-group reflection cross-check** AFTER the `omega_reflection` static-tuple `_require` (~942), still inside the `omega_direct` block:

```python
            if self.omega_reflection == "init_seed":
                # det<0 seeding is group-specific; reject where the ambient diag(-1,1,...) seed is not a
                # valid, tie-respecting, det<0 element of the structure group (deferred cases -> Phase 2b).
                _REFLECT_OK = ("glk", "block_glk", "so_k")   # so_k: valid O(K) seed; gl: ambient seed valid
                if self.gauge_group not in _REFLECT_OK:
                    raise ValueError(
                        f"omega_reflection='init_seed' is not available for gauge_group="
                        f"{self.gauge_group!r} in Phase 2: sp/sp_n have no det<0 component, so_n needs the "
                        f"rho(O(N)) image seed, and tied_block_glk needs a tied replicated seed (all deferred). "
                        f"Use omega_reflection='off', or gauge_group in {_REFLECT_OK}."
                    )
```

- [ ] **Step 6: Retain `irrep_dims` in `PriorBank`.** Read `prior_bank.py:160` — the ctor accepts `irrep_dims` but drops it. Add `self.irrep_dims = irrep_dims` in the stash block (needed by Tasks 4/5). No behavior change (it was accepted-but-unused).

- [ ] **Step 7: Run to verify pass** — `pytest tests/test_config.py -k "eligible_groups or cross_check"` (pass), then `pytest tests/test_config.py` (whole file green — shipped gates unchanged).

- [ ] **Step 8: Commit** — `git add vfe3/config.py vfe3/model/prior_bank.py tests/test_config.py`; `git commit -m "feat(omega_direct): admit all eligible groups + fail-closed per-group reflection cross-check"`.

---

### Task 3: Full-table extension for `sp`, `sp_n`, `tied_block_glk` (REAL value, no optimizer change)

**Files:** Test-only + verify. These groups are `skew_symmetric=False`, reuse the existing `(V,K,K)` table, the fp64-inv transport (Task 1 leaves non-skew unchanged), and the group-agnostic optimizer — so NO source change is needed beyond Task 2's gate. This task PROVES the extension works end-to-end and pins the group-membership invariants. Test: `tests/test_omega_direct.py`.

**Interfaces:** Confirms `omega_direct` builds and runs a full-model forward for `sp`, `sp_n`, `tied_block_glk`, and that the transport respects each group's structure.

- [ ] **Step 1: Write the tests** (append). Transport-level (symplectic membership) + a full-model forward smoke:

```python
def test_omega_direct_sp_symplectic_membership_and_cocycle():
    from vfe3.geometry.lie_ops import retract_omega
    grp = get_group("sp")(K=4)                                 # Sp(4,R); n_gen = m(2m+1)=10
    g = torch.Generator().manual_seed(3)
    xi = 0.15 * torch.randn(1, 3, grp.generators.shape[0], generator=g)
    U = retract_omega(torch.eye(4).expand(1, 3, 4, 4).contiguous(), xi, grp.generators)  # in Sp(4,R)
    # symplectic form J = [[0,I],[-I,0]] preserved: U^T J U = J
    m = 2; J = torch.zeros(4, 4); J[:m, m:] = torch.eye(m); J[m:, :m] = -torch.eye(m)
    UtJU = U.transpose(-1, -2) @ J @ U
    assert torch.allclose(UtJU, J.expand_as(UtJU), atol=1e-4)
    om = build_transport_from_element(U, grp)["Omega"]
    assert torch.allclose(om[0, 0, 1] @ om[0, 1, 2], om[0, 0, 2], atol=1e-4)   # cocycle

def test_omega_direct_full_model_forward_sp_spn_tied():
    tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
    for over in (dict(gauge_group="sp", embed_dim=4, n_heads=1),
                 dict(gauge_group="sp_n", embed_dim=5, n_heads=1, group_n=4,
                      irrep_spec=[("sym0", 1), ("sym1", 1)]),
                 dict(gauge_group="tied_block_glk", embed_dim=4, n_heads=2)):
        torch.manual_seed(0)
        m = VFEModel(_cfg(gauge_parameterization="omega_direct", use_head_mixer=False, **over))
        with torch.no_grad():
            logits = m(tok)[0]
        assert torch.isfinite(logits).all()
```

(Note: `_cfg` is the helper in the file; it defaults `family="gaussian_full"`, `decode_mode="full"`. For `sp_n` `embed_dim=5` the irrep_spec must sum to 5 — `sym0`(dim1)+`sym1`(dim4)=5; verify against `get_group("sp_n")`.)

- [ ] **Step 2: Run** — `pytest tests/test_omega_direct.py -k "sp_symplectic or forward_sp_spn_tied" -x`. If a config/build error surfaces (e.g. an irrep_spec dim mismatch), fix the TEST config (not source) to a valid tiny spec. If a real source gap surfaces (transport/optimizer chokes on a group), STOP and report BLOCKED with the diagnosis.

- [ ] **Step 3: Commit** — `git add tests/test_omega_direct.py`; `git commit -m "test(omega_direct): sp/sp_n/tied_block_glk full-table extension (symplectic membership + forward)"`.

---

### Task 4: Full-table extension for `so_k`, `so_n` + orthogonality-drift control

**Files:** Modify `vfe3/gauge_optim.py` (add periodic re-orthogonalization of skew-group `omega_embed` rows in the `"omega"` branch, gated on `omega_reorth_every>0`). Optionally add a small helper in `vfe3/geometry/lie_ops.py`. Test: `tests/test_omega_direct.py`.

**Interfaces:** `omega_direct` builds/runs for `so_k`/`so_n`; `so_k` accepts `init_seed` (det<0 into `O(K)`); periodic polar re-orthogonalization keeps `U` on `O(K)` under fp32 drift.

- [ ] **Step 1: Write the tests** (append):

```python
def test_omega_direct_so_k_orthogonal_and_reflection_reach():
    from vfe3.geometry.lie_ops import retract_omega
    grp = get_group("so_k")(K=4)
    U = retract_omega(torch.eye(4).expand(1, 3, 4, 4).contiguous(),
                      0.2 * torch.randn(1, 3, grp.generators.shape[0], generator=torch.Generator().manual_seed(4)),
                      grp.generators)
    assert torch.allclose(U @ U.transpose(-1, -2), torch.eye(4).expand_as(U @ U.transpose(-1,-2)), atol=1e-4)
    assert (torch.det(U) > 0).all()                            # retraction stays in SO(4)
    # init_seed reaches det<0 (O(4)\SO(4)) via reflection_element
    from vfe3.geometry.generators import reflection_element
    R = reflection_element(4)
    assert torch.det(R @ U[0, 0]) < 0

def test_omega_reorth_projects_drifted_element_back_to_O_K():
    # a slightly non-orthogonal element (fp32-drift analog) is re-orthogonalized by the helper/optimizer path
    U = torch.eye(4) + 0.05 * torch.randn(4, 4, generator=torch.Generator().manual_seed(5))
    # polar factor: U = Q P, Q orthogonal
    from vfe3.gauge_optim import _polar_orthogonalize   # (or wherever the plan places it)
    Q = _polar_orthogonalize(U.unsqueeze(0))[0]
    assert torch.allclose(Q @ Q.transpose(-1, -2), torch.eye(4), atol=1e-5)
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_omega_direct.py -k "so_k_orthogonal or reorth_projects" -x`. Expected: the reorth test FAILS (`_polar_orthogonalize` missing); the so_k test may pass already (Task 1/2 admit it).

- [ ] **Step 3: Implement the polar re-orthogonalization helper** in `vfe3/gauge_optim.py` (or `lie_ops.py`):

```python
def _polar_orthogonalize(U: torch.Tensor) -> torch.Tensor:   # (..., K, K) -> nearest orthogonal (polar factor)
    r"""Nearest orthogonal matrix to ``U`` via the polar decomposition Q = U (U^T U)^{-1/2}.

    Uses the SVD polar factor Q = W V^T (U = W S V^T), the exact minimizer of ||U - Q||_F over O(K).
    Run in a float64 island; keeps a drifted skew-group frame exactly on O(K) so U^T stays the inverse.
    """
    with torch.amp.autocast(U.device.type, enabled=False):
        W, _, Vh = torch.linalg.svd(U.double(), full_matrices=False)
        return (W @ Vh).to(U.dtype)
```

- [ ] **Step 4: Wire it into the `"omega"` optimizer branch** (read `gauge_optim.py:148-173`). After the retraction step, when the group is skew and `omega_reorth_every>0` and the step counter hits the cadence, project the `omega_embed` rows back to `O(K)` under `no_grad`. Pass `group.skew_symmetric` and `omega_reorth_every` into the optimizer (via the constructor, like `omega_retract_mode`). Gate: default `omega_reorth_every=0` → no-op → byte-identical for non-SO / default.

- [ ] **Step 5: Run to verify pass** — `pytest tests/test_omega_direct.py -k "so_k_orthogonal or reorth_projects"` (pass), then the whole `pytest tests/test_omega_direct.py` and `pytest tests/test_gauge_optim.py` (default optimizer unchanged).

- [ ] **Step 6: Commit** — `git add vfe3/gauge_optim.py tests/test_omega_direct.py` (+ `lie_ops.py` if the helper landed there); `git commit -m "feat(omega_direct): so_k/so_n extension + polar re-orthogonalization drift control"`.

---

### Task 5: Opt-in compact block storage for `block_glk`, `tied_block_glk`

**Files:** Modify `vfe3/model/prior_bank.py` (compact table under `omega_compact_storage`; assemble block-diagonal `(B,N,K,K)` at encode via `lie_ops._from_equal_diag_blocks`). Modify `vfe3/gauge_optim.py` (per-block retraction when the omega table is compact). Test: `tests/test_omega_direct.py`.

**Interfaces:** With `omega_compact_storage=True`, `block_glk` stores `omega_embed` as `(V,H,d,d)` and `tied_block_glk` as `(V,d,d)`; encode assembles the identical block-diagonal element; param count matches `phi_embed`; the optimizer retracts per block. Default-OFF → the shipped `(V,K,K)` path is byte-identical.

- [ ] **Step 1: Write the tests** (append):

```python
def test_omega_compact_storage_param_parity_and_assembly():
    pb_full = PriorBank(vocab_size=6, K=4, n_gen=8, gauge_parameterization="omega_direct", irrep_dims=[2, 2])
    pb_cmp  = PriorBank(vocab_size=6, K=4, n_gen=8, gauge_parameterization="omega_direct", irrep_dims=[2, 2],
                        omega_compact_storage=True)
    assert pb_full.omega_embed.shape == (6, 4, 4)             # full (V,K,K)
    assert pb_cmp.omega_embed.shape == (6, 2, 2, 2)           # compact (V,H,d,d)
    assert pb_cmp.omega_embed.numel() == 6 * 8                # == V * n_gen (matches phi_embed)
    # identity init assembles to the block-diagonal identity element
    tok = torch.zeros(1, 3, dtype=torch.long)
    om = pb_cmp.encode(tok).omega
    assert om.shape == (1, 3, 4, 4)
    assert torch.allclose(om, torch.eye(4).expand(1, 3, 4, 4), atol=1e-7)
    # off-blocks are exactly zero for a non-identity compact frame
    with torch.no_grad():
        pb_cmp.omega_embed[0, 0] = torch.tensor([[1.2, 0.3], [0.0, 0.9]])
    om2 = pb_cmp.encode(torch.zeros(1, 1, dtype=torch.long)).omega[0, 0]
    assert torch.allclose(om2[:2, 2:], torch.zeros(2, 2)) and torch.allclose(om2[2:, :2], torch.zeros(2, 2))

def test_omega_compact_tied_shares_one_block():
    pb = PriorBank(vocab_size=6, K=4, n_gen=4, gauge_parameterization="omega_direct", irrep_dims=[2, 2],
                   omega_compact_storage=True, gauge_group_is_tied=True)   # (name the tied flag as implemented)
    assert pb.omega_embed.shape == (6, 2, 2)                  # (V,d,d) one shared block
    om = pb.encode(torch.zeros(1, 1, dtype=torch.long)).omega[0, 0]
    assert torch.allclose(om[:2, :2], om[2:, 2:], atol=1e-7)  # same block in both heads
```

(The implementer must decide how `PriorBank` learns tied-vs-untied: pass a boolean derived from the group name at the construction site in `model.py`, since the bank does not hold the group. Read `model.py`'s `PriorBank(...)` call site and thread a `tied` flag or the group name.)

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_omega_direct.py -k "compact_storage or compact_tied" -x`. Expected FAIL (`omega_compact_storage` kwarg unknown).

- [ ] **Step 3: Implement the compact table + assembly** in `prior_bank.py`. When `omega_compact_storage and len(set(irrep_dims))==1 and len(irrep_dims)>1`: store `(V, H, d, d)` (untied) or `(V, d, d)` (tied) identity table; in `_encode_per_token`, assemble the block-diagonal `(B,N,K,K)` via `lie_ops._from_equal_diag_blocks` (untied: scatter the H blocks; tied: broadcast the one block into H diagonal slots), then set `belief.omega`. Single-block / tower / `omega_compact_storage=False` keep the current full `(V,K,K)` path unchanged.

- [ ] **Step 4: Implement the per-block optimizer retraction.** In the `"omega"` branch, when `p.data.dim()==4` (compact `(V,H,d,d)`): reshape `(A,H,d,d)->(A*H,d,d)`, run `extract_phi`/`retract_omega` against a `gl(d)` generator basis (`generate_glk(d)`), reshape back. Full `(V,K,K)` tables keep the existing K×K path. The optimizer must be given the `gl(d)` basis (build it once from `d = irrep_dims[0]`).

- [ ] **Step 5: Run to verify pass** — `pytest tests/test_omega_direct.py -k "compact"` (pass); add and run a natural-gradient-step-equivalence test (a single optimizer step on the compact table equals the full-table step restricted to blocks, atol ~1e-5); then whole `pytest tests/test_omega_direct.py`.

- [ ] **Step 6: Commit** — `git add vfe3/model/prior_bank.py vfe3/gauge_optim.py vfe3/model/model.py tests/test_omega_direct.py`; `git commit -m "feat(omega_direct): opt-in compact (V,H,d,d)/(V,d,d) block storage + per-block retraction"`.

---

### Task 6: Ablation fan-out over the omega-eligible groups

**Files:** Modify `ablation.py` (the `gauge_parameterization` sweep arm ~405-413, reusing the `gauge_group` sweep's `group_n`/`irrep_spec`/`phi_precond_mode` payloads ~383-398). Test: `tests/test_omega_direct.py`.

**Interfaces:** The `gauge_parameterization`/omega sweep enumerates a phi baseline and per-group omega_direct arms that all construct against `BASELINE_CONFIG` (gamma channel off, per Phase 1).

- [ ] **Step 1: Extend the sweep arm** — add per-group omega_direct cells (`glk`, `block_glk`, `tied_block_glk`, `sp`, `sp_n`, `so_k`, plus `so_n`/`sp_n` towers using the `gauge_group` arm's `group_n`/`irrep_spec` payloads), each carrying `gauge_parameterization="omega_direct", lambda_gamma=0.0, s_e_step=False, use_head_mixer=False` and the group's own params. Keep `"requires": {"transport_mode": "flat"}`.

- [ ] **Step 2: Rewrite `test_ablation_omega_direct_arm_builds`** (in `tests/test_omega_direct.py`) to iterate ALL cells via `ablation.make_run_overrides("gauge_parameterization")` → `ablation._cell_cfg_dict(overrides, seed=0, max_steps=1)` → `VFE3Config(**cfg_dict)`, asserting each cell constructs (not the old two-label set). Run `pytest tests/test_omega_direct.py -k ablation` and `python -c "import ablation"` synchronously.

- [ ] **Step 3: Commit** — `git add ablation.py tests/test_omega_direct.py`; `git commit -m "feat(omega_direct): per-group ablation fan-out over omega-eligible groups"`.

---

## Self-Review

**Spec coverage:** transpose inverse → T1; gate + reflection cross-check → T2; sp/sp_n/tied extension → T3; so_k/so_n + drift → T4; compact storage → T5; ablation → T6. Deferred (STE, tower compaction, so_n ρ(O(N)) seed, tied reflection seed) explicitly NOT in this plan.

**Byte-identity:** every task preserves the phi default AND the shipped omega_direct glk/block_glk path (T1 skew branch leaves non-skew untouched; T2 only adds accept/reject for non-shipped groups; T3 test-only; T4/T5 gated on default-OFF `omega_reorth_every`/`omega_compact_storage`; T6 adds arms).

**WIP guard:** T1 edits transport.py at ~1004-1010 — the plan explicitly warns to keep the WIP clamp line (~741) out of `git diff --cached`.

**Type consistency:** `build_transport_from_element(omega, group)` (T1), `omega_compact_storage`/`omega_reorth_every` fields (T2/T4/T5), `_polar_orthogonalize` (T4), `_from_equal_diag_blocks` reuse (T5) — names consistent across tasks.

## Execution Handoff

Subagent-driven, task-by-task, with a task review + fix loop after each and a final whole-branch review, then push + FF-merge to main (per the established Phase-1 flow). The STE remains design-only pending the user's two-tier confirmation (spec §4).
