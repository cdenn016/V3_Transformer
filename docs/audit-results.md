# Deep Audit — 2026-07-26

Branch `audit/2026-07-26-deep-audit`, forked from `main` at `12a273a`. Read-only investigation; no
source was modified. The user's uncommitted `train_vfe3.py` WIP was left untouched throughout.

## Scope

Whole repository: `vfe3/**` (66 files), the 11 root drivers, and `tests/`. Extra scrutiny on code
that landed 2026-07-25 — the mechanism diagnostics in `vfe3/run_artifacts.py`, their figures, the
`s_e_step_n_iter` seam, the `gaussian_diagonal_exact` / `gaussian_frame_diagonal` families, and the
`BeliefParams.coupling_energy` dispatch seam.

## Investigators dispatched

**Base five** (wave 1, parallel): `code-reviewer`, `debugger`, `refactoring-specialist`,
`performance-engineer`, `python-pro`.

**Domain experts** (wave 2, parallel — the theory gate is met: `CLAUDE.md` declares gauge
equivariance, SPD geometry, KL/free-energy derivations, and pure-path invariants). The whole
relevant `audit-*` pool ran, because this is a whole-repo audit and every lens is on-scope by
construction: `audit-gauge-theorist`, `audit-geometer`, `audit-info-geometer`,
`audit-numerical-analyst`, `audit-transformer-ml`, `audit-variational`,
`audit-implementation-engineer`.

Waves were run sequentially (5 then 7) to stay inside the parallel cap; agents within a wave ran
concurrently.

## Test suite (current, machine-read)

| lane | command | result |
|---|---|---|
| CPU | `pytest -m "not cuda"` under `C:/Python314/python.exe`, JUnit XML | `tests=4269 failures=4 errors=0 skipped=32` |

The 4 failures are **pre-existing and stale on `main`**, not caused by this audit or by the
2026-07-25 work: `origin/main:vfe3/config.py:241` has `rope_on_value: bool = False` while
`tests/test_rope.py::test_rope_on_value_defaults_true` asserts `True`
(`test_gauge_optim` ×2, `test_rope` ×2). See M-07 for the underlying code inconsistency.

A targeted expert re-run of `tests/test_exact_congruence_family.py`,
`tests/test_frame_gaussian_family.py`, `tests/test_divergence.py` gave JUnit
`tests=73 failures=0 errors=0 skipped=0`.

## Severity roll-up (pre-verification)

| severity | count |
|---|---|
| critical | 2 (one contested down to high by three other lenses) |
| high | 12 |
| medium | 14 |
| low | 12 |

---

# Findings

## Cluster A — `gaussian_frame_diagonal` discards the transport

**Reported independently by four lenses** (`code-reviewer`, `audit-geometer`, `audit-gauge-theorist`,
`audit-implementation-engineer`), with severity split critical/high.

### A-01 — Both transport seams ignore `omega` unconditionally
**Location:** `vfe3/families/frame_gaussian.py:101-130`
**Severity:** critical (geometer) / high (three other lenses) — contested, see challenge tier
**Evidence:**
```python
    def transport_location(cls, mu, omega) -> torch.Tensor:
        return _broadcast_over_queries(mu)
    def transport_dispersion(cls, dispersion, omega, *, diagonal_out=None) -> torch.Tensor:
        return _broadcast_over_queries(dispersion)
```
The frame cancellation `Sigma_i = U_i diag(s_i) U_i^T` holds **only** for the coboundary
`Omega_ij = U_i U_j^{-1}`. `transport.py:721` builds `regime_ii` as
`exp(phi_i G) exp(delta_ij G) exp(-phi_j G) = U_i D_ij U_j^{-1}`, which is not a coboundary.

Measured at the family seam (K=4, block_glk 2x2, N=3):
```
FLAT   max |U_i^-1 Omega_ij U_j - I| = 2.38e-07
REG2   max |U_i^-1 Omega_ij U_j - I| = 3.37
regime_ii  loc err=3.79e+00  diag(var) err=3.15e+01  max|offdiag of receiver-frame Sigma_t|=2.04e+01
```
Pair energy is bit-identical between flat and every Regime-II mode under this family, while
`gaussian_diagonal` differs by 81.2:
```
regime_ii              gaussian_frame_diagonal  max|E(nonflat)-E(flat)| = 0.000000e+00
regime_ii_covariant    gaussian_frame_diagonal  max|E(nonflat)-E(flat)| = 0.000000e+00
regime_ii_link         gaussian_frame_diagonal  max|E(nonflat)-E(flat)| = 0.000000e+00
regime_ii_link_charted gaussian_frame_diagonal  max|E(nonflat)-E(flat)| = 0.000000e+00
regime_ii              gaussian_diagonal        max|E(nonflat)-E(flat)| = 8.121053e+01
```
End-to-end on a tiny `VFEModel` with `connection_W = 0.8*N(0,1)`:
```
gaussian_diagonal        max|logits(regime_ii) - logits(flat)| = 4.79e-03   d loss/d connection_W = 1.1e-04
gaussian_frame_diagonal  max|logits(regime_ii) - logits(flat)| = 0.00e+00   d loss/d connection_W = None
```
`connection_W.grad is None` — the learned connection is severed from the graph and can never train,
while the builder still pays O(N^2) per-edge matrix exponentials per E-step iteration. The discarded
operator carries genuine curvature: triangle holonomy `1.1e+00` (`regime_ii`), `2.6e+04`
(`regime_ii_covariant`), `2.3e+00` (`regime_ii_link`), `3.2e+00` (`regime_ii_link_charted`) against
`3.6e-07` for `flat`.

The sibling family guards exactly this contract at `exact_congruence.py:179`
(`if not _certifies_same_frame_flat_cocycle(omega): raise TypeError(...)`). `frame_gaussian.py` has
no guard, and `grep -n "frame_diagonal" vfe3/config.py` returns nothing — all four Regime-II modes,
`pos_rotation='rope'`, and `phi_reflection` construct and run cleanly with zero warnings.

**Fix:** Gate both seams on `_certifies_same_frame_flat_cocycle(omega)` and raise as
`exact_congruence._pullback_query` does, or reject the non-coboundary transports in
`VFE3Config.__post_init__` keyed off a family-declared `requires_coboundary_transport` capability
rather than a literal family name.

### A-02 — Gauge-RoPE and `phi_reflection` are exact no-ops under this family
**Location:** `vfe3/families/frame_gaussian.py:112`, `:130`
**Severity:** high
**Evidence:**
```
gaussian_diagonal        rope on_cov=False: max|E_rope - E_norope| = 2.721035e+00
gaussian_diagonal        rope on_cov=True : max|E_rope - E_norope| = 1.446683e+00
gaussian_frame_diagonal  rope on_cov=False: max|E_rope - E_norope| = 0.000000e+00
gaussian_frame_diagonal  rope on_cov=True : max|E_rope - E_norope| = 0.000000e+00
```
Distinct from A-01: with `rope_full_gauge=True` the rotated operator *is* a coboundary (R is
orthogonal), so the identity transport is mathematically self-consistent — the defect is that the
positional mechanism the user selected contributes exactly zero, silently, while `pure_path_report`
still records `pos_rotation: "rope"` as active.
**Fix:** Warn or reject in `VFE3Config` when `pos_rotation != 'none'` or `phi_reflection != 'off'`
is combined with this family.

### A-03 — `sandwich_absmax` overflow probe is inert for this family
**Location:** `vfe3/model/model.py:3001-3003`
**Severity:** medium
**Evidence:** The probe dispatches through the configured family, so under
`gaussian_frame_diagonal` it reports `max|Sigma|` rather than `max|Omega Sigma Omega^T|`:
```
gaussian_diagonal          sandwich_absmax = 97.975517
gaussian_diagonal_exact    sandwich_absmax = 97.975517
gaussian_frame_diagonal    sandwich_absmax =  1.335873   (raw max|sigma| = 1.335873)
```
**Fix:** Build the probe from `vfe3.geometry.transport.transport_covariance` directly; it measures
the transport, not the family's energy.

### A-04 — Wasted `omega` construction on the hot path
**Location:** `vfe3/families/frame_gaussian.py:101-130`; callers at `vfe3/inference/e_step.py:948`,
`:751`, `vfe3/model/model.py:1870`
**Severity:** medium
**Evidence:** Every caller builds the full transport unconditionally before dispatching to seams
that ignore it. Confirmed real by `tests/test_frame_gaussian_family.py:142-170`
(`phi_embed.grad is None`).
**Fix:** Short-circuit `build_belief_transport` for families that declare they do not consume it.

---

## Cluster B — the 2026-07-25 mechanism diagnostics

### B-01 — `collect_estep_character` may attribute the model channel and the layer stack to the belief E-step
**Location:** `vfe3/run_artifacts.py:3222-3290`
**Severity:** high — **pending verification; see challenge tier**
**Evidence:** `_mm_spy` appends a record on *every* `mm_exact_update` call in the forward,
undifferentiated, and the headline numbers read `recorded[0]` and `recorded[-1]`:
```python
        recorded.append({... "mu_p": mu_p.detach(), "mu_star": mu_star.detach(), ...})
...
            first, last = recorded[0], recorded[-1]
            displacement = last["mu_star"] - first["mu_p"]
            prior_mass, pair_mass = float(first["prior_mass"]), float(first["pair_mass"])
```
`model.py:1077-1120` runs the model channel *before* the belief stack and `model.py:922` threads
`e_step_update=cfg.e_step_update` into it, so under `s_e_step=True` `recorded[0]` is claimed to be
the s-channel fusion. At `n_layers>1` `recorded[0]`/`recorded[-1]` would span the whole stack.
`recompute_max_abs_err` cannot detect this because each individual replica is still exact.

**Impact if confirmed:** the pair-precision shares reported on 2026-07-25 for the model-channel
checkpoints (0.190 at K=20, 0.298 at K=300) would be the s-channel's, not the belief channel's, and
the "share rises with width" reading with them. The token-prior figure (0.109) is unaffected —
`s_e_step=False` there, and the call sites were traced directly on that checkpoint.
**Fix:** Tag each recorded entry with its originating channel/layer and slice the belief-channel
window before computing displacement and precision shares.

### B-02 — `collect_beta_channel_decomposition` ablates only one of two live `attention_weights` bindings
**Location:** `vfe3/run_artifacts.py:3371` (patch site) vs `vfe3/inference/e_step.py:23,792,799`
**Severity:** high
**Evidence:** The probe patches only `kernels_module.attention_weights`. `e_step.py:23` imports the
same function independently, and `phi_alignment_loss` calls that unpatched binding at `:799`
whenever `lambda_twohop != 0.0` on the phi substep (`e_phi_lr > 0.0`, `e_step.py:1136`), regardless
of `e_step_update`. Empirically reproduced on a tiny CPU model
(`e_phi_lr=0.05`, `lambda_twohop=0.3`): **6 calls** to the real unablated function during the
probe's three arms, while the record reports `available: True`.

The commit that introduced this fixed the identical aliasing-import bug class for
`mm_exact_update` in the same change, reasoning that "kernels.py calls attention_weights on both
belief routes" — true, and insufficient, because `e_step.py` also calls it directly.
Not reachable at `e_phi_lr=0`/`lambda_twohop=0`, so the 2026-07-25 M6 measurements stand.
**Fix:** Patch `vfe3.free_energy.attention_weights` at the definition site so every aliasing import
resolves through it, or add `e_step_module` to a multi-holder list as `collect_estep_character` does.

### B-03 — `collect_estep_depth_sensitivity` pins the model channel to `n_e_steps`, not to the trained `s_e_step_n_iter`
**Location:** `vfe3/run_artifacts.py:3084`
**Severity:** high
**Evidence:** Two investigators contradicted each other here; `audit-implementation-engineer`
adjudicated from source with a repro. **Claim A (`python-pro`) is correct.**
```
3050	    trained_depth = int(model.cfg.n_e_steps)
3051	    trained_s_depth = model.cfg.s_e_step_n_iter
...
3084	            points.append(_score(depth, trained_depth))
```
`_score` writes its second argument straight into the field (`:3060`). The trained model-channel
depth is `n_iter=(cfg.n_e_steps if cfg.s_e_step_n_iter is None else cfg.s_e_step_n_iter)`
(`model.py:868`), and `s_e_step_n_iter != n_e_steps` is a legal, independently validated config
(`config.py:982-986`). Repro at `n_e_steps=2, s_e_step_n_iter=5`:
```
points  belief_depth: [0, 1, 2]
points  s_depth     : [2, 2, 2]     <-- should all be 5
```
The two collectors written the next day use the correct idiom (`:3271`, `:3477`). Both the `ce` and
`free_energy_per_token` columns are affected (`viz/extract.py:196-199` also calls `_refine_s`), and
the artifact's published `interpretation` string is false for such a run.
**Fix:** `points.append(_score(depth, trained_s_depth if trained_s_depth is not None else trained_depth))`

### B-04 — `emit_expensive_diagnostics` is nested under `generate_figures`; the cheap tier does not always run
**Location:** `vfe3/run_artifacts.py:2874`, `:2890-2917`
**Severity:** medium
**Evidence:** `figures_enabled = bool(getattr(cfg, "generate_figures", True))` at `:2872` wraps the
entire mechanism block. `generate_figures=False` therefore suppresses `estep_depth_sensitivity`,
`estep_character`, `beta_channel_decomposition`, `phi_numerics`, and the whole
`_log_mechanism_diagnostics` banner. `scaling.py:404` and `scaling.py:816` both set it, so the
scaling entry point emits none of them. `VFE3Config(emit_expensive_diagnostics=True,
generate_figures=False)` constructs with no warning and the toggle is silently inert.
**Fix:** Hoist the cheap block out from under `figures_enabled` (gate only on
`depth_loader is not None`) and add the inert combination to the `_inert` warning list.

### B-05 — AMP recursion double-records and can fire a false "probe is stale" warning
**Location:** `vfe3/gradients/kernels.py:571-573` with `vfe3/run_artifacts.py:3222`
**Severity:** medium
**Evidence:** The kernel's reduced-precision island recurses through the **module global**, which
the probe has replaced. Under `amp_dtype='bf16'`/`'fp16'` each real call yields two `recorded`
entries: an inner fp32 one and an outer one whose replica is recomputed from bf16
`mu/sigma/mu_p/sigma_p` against fp32 `pair_prec`/`pair_mean`. That outer error enters `max_err` and
can trip `"E-step character replica error ... is NONZERO"` on a correct probe.
Independently, `audit-performance-engineer` verified the recursion does **not** double compute work
(`torch.is_autocast_enabled` is correctly `False` inside the nested `enabled=False` context).
**Fix:** Return `original_mm(...)` unmodified from `_mm_spy` when autocast is enabled, letting only
the inner fp32 invocation record.

### B-06 — Diagnostic spy retains a `(B, N, K)` tensor pair per kernel call
**Location:** `vfe3/run_artifacts.py:3251-3257`
**Severity:** medium
**Evidence:** `recorded.clear()` runs once per depth, so within a forward the list holds
`2 x n_layers x (n_e_steps + s_depth)` live device tensors while only `recorded[0]["mu_p"]` and
`recorded[-1]["mu_star"]` are read. At `depths=(1,2,3,5,8)` on a production batch this is a
multi-hundred-MB to multi-GB device allocation at end of training with optimizer state resident; an
OOM is swallowed by the outer `except Exception`.
**Fix:** Keep two rolling slots rather than appending every call's tensors.

### B-07 — `_depth_tokens` can be referenced unbound
**Location:** `vfe3/run_artifacts.py:2875-2910`
**Severity:** low
**Evidence:** Bound inside the first `try` after a statement that can raise, then consumed outside
it. A bare-tensor batch makes all three probes log
`"failed (cannot access local variable '_depth_tokens')"` instead of the real cause.
**Fix:** Hoist batch extraction into its own `try` and guard on `_depth_tokens is not None`.

### B-08 — Eval-time E-step halting can silently cap the probes' depth axis
**Location:** `vfe3/inference/e_step.py:1431` with `vfe3/run_artifacts.py:3269-3290`
**Severity:** low
**Evidence:** Halting is active precisely in `eval`, which the probes set. Rows are labeled with the
*requested* depth, so with `cfg.e_step_halt_tol` set every depth past the halt point yields an
identical trajectory — reading as "depth buys only magnitude" when it is the tolerance. Neither
probe record emits `e_step_halt_tol` or a realized-iteration count.
**Fix:** Record the realized iteration count per row and echo `cfg.e_step_halt_tol` into both records.

### B-09 — RNG save/restore present in two sibling probes, absent in the two new ones
**Location:** `vfe3/run_artifacts.py:3264`, `:3390` vs `:3055`, `:3459`
**Severity:** low
**Evidence:** `collect_estep_depth_sensitivity` and `collect_context_sensitivity` snapshot and
restore global RNG; `collect_estep_character` and `collect_beta_channel_decomposition` run 5 and 3
full forwards with no such bracket. No RNG consumer was found on the eval forward path, so this is
latent, but it is an undocumented divergence from the contract the adjacent probes assert.
**Fix:** Add the same bracket to both new probes.

### B-10 (CLEAN NEGATIVE) — the monkeypatch restoration is sound
**Location:** `vfe3/run_artifacts.py:3163-3327`
This was an explicit audit question. `holders` (`:3211`) and `original_contract` (`:3213`) are
computed **before** the `try` opens with no side effects; the four patch assignments occur **inside**
it (`:3266-3268`); the `finally` (`:3291-3297`) reassigns from the pre-captured originals — an
idempotent no-op if nothing was patched yet, a full restore if it was. Because `finally` executes
before an exception propagates, and every restore statement precedes the one call
(`model.train(was_training)`) that could itself raise, **no leaked binding is reachable through any
Python-level exception**. Same conclusion for `collect_beta_channel_decomposition`'s single-holder
patch and `collect_context_sensitivity`'s cfg/training/RNG restore. `finalize_run`'s outer
`try/except Exception` only ever sees an already-clean model: no double-restore, no masked
restoration, no scenario where the outer catch fires with patched bindings installed.

---

## Cluster C — numerical correctness

### C-01 — The diagonal-congruence guard tests sign, but the error reaches 44% while still positive
**Location:** `vfe3/geometry/transport.py:1269-1306`; call sites `:2406`, `:2501`
**Severity:** high
**Evidence:** The guard's only test is `float(out.amin()) >= 0.0`. Measured against a float64
`sum_l Omega_kl^2 sigma_l` reference:
```
   cond(U)    min f32 out  escalated?   max rel err vs f64 ref
     1e+02         0.6971       False               5.8182e-05
     1e+03         0.5076       False               2.1279e-02
     3e+03         0.4621       False               1.1737e-01
     1e+04         0.3010       False               4.3778e-01   <-- 44% wrong, still positive
     1e+05       -29.1700        True               4.3708e-03
```
Sign violation lags the error by a decade of conditioning. The regime is reachable: `retract_phi`
caps `||phi||` at `max_norm=5.0` for non-skew groups (`retraction.py:795`) with an orthonormal
`block_glk` basis, so `cond(U) <= exp(10) = 2.2e4` and `cond(Omega_ij)` is the product.
**Fix:** Key escalation on a computed accuracy/conditioning bound, not on the sign.

### C-02 — ...and the sign it tests lives on values the next call discards
**Location:** same
**Severity:** medium
**Evidence:** The mechanism producing a negative entry of an exactly-PSD congruence is total
cancellation at the **self** pair, where `Omega_ii = I` exactly so the true value is `O(sigma)`
while intermediates are `O(cond(U)^2 sigma)`. Measured, 400 draws:
```
draws with a negative fp32 variance (escalation would fire): 92/400
  ... of those, negatives confined to the SELF pair i==j:    92/92
```
Those self-pair values are then overwritten by `_restore_certified_self_links_`
(`transport.py:2152`, `:2168`), so the trigger is measured on discarded data. With escalation
silent, off-diagonal relative error reached `4.913e-03` at `cond(Omega)=6.04e6` over 600 draws.

### C-03 — ...and the check costs an unconditional host sync on the production hot path
**Location:** same
**Severity:** critical (as reported by `audit-performance-engineer`) — see challenge tier
**Evidence:** `float(out.amin())` forces a device-to-host sync on **every** diagonal-covariance
build: at least once per E-step inner iteration per layer for the belief channel, again for the
model channel under `s_e_step`. Unlike the sibling clamp-surrogate warning, only the warning
*message* is latched (`_DIAG_CONGRUENCE_NEGATIVE_WARNED`), not the reduction itself.

> **C-01 / C-02 / C-03 are one defect seen from three angles and admit one fix:** gate escalation on
> a conditioning proxy computed once per build, and drop the per-call `amin` sync. Deleting the sync
> alone (C-03 in isolation) would make a real correctness hole cheaper rather than fixing it.

### C-04 — `_direct_link_diagonal_covariance` returns negative variances to the caller
**Location:** `vfe3/geometry/transport.py:2294-2315`, dispatched at `:2137-2139`
**Severity:** high
**Evidence:** The third diagonal-congruence route uses the same mixed-sign regrouping the
2026-07-25 F3 work hardened on the other two, but has no autocast island, no float64 escalation, and
no `clamp(min=0.0)`, and `transport_covariance` returns it without `_restore_certified_self_links_`.
Measured (fp32, N=3, K=3, 200 draws, `||phi||_F` in [4,8]):
```
draws with a NEGATIVE transported variance reaching the caller: 1/200
most negative value: -3.3456e+00   (at pair (2,2), cond(U)=1.93e4)
max relative error vs fp64 sum-of-squares reference: 5.501e+00
```
A negative reaches the divergence kernel's `clamp(min=eps)=1e-6`, inverting that key's precision
weight by ~6 orders and saturating `E_ij` at `kl_max`.
**Fix:** Give this route the same autocast island, escalation call, and `clamp(min=0.0)` its two
siblings have.

### C-05 — Dense `Omega` and the dense diagonal congruence sit outside the fp32 island under AMP
**Location:** `vfe3/geometry/transport.py:1620`, `:2174-2176`
**Severity:** high
**Evidence:** Vertex factors are pinned to fp32 (`:1425`) and then combined by a bare
autocast-eligible einsum. Measured (`glk`, K=4, N=3, CPU bf16 autocast):
```
  exp_phi dtype     : torch.float32
  dense Omega dtype : torch.bfloat16
  ||Omega_ii - I||_inf (bf16-autocast): 3.906e-03
  ||Omega_ii - I||_inf (fp32):          1.192e-07
dense diagonal congruence under bf16 autocast: max rel err vs fp64 = 2.412e-03
```
`Omega_ii = I` is the identity that makes `E_ii = 0` and keeps the self-pair out of `pair_mask`; at
3.9e-3 it is no longer exact. Reachable: `_can_fuse_flat` (`e_step.py:199`) returns `False` for
single-block groups, so flat + `amp_dtype='bf16'` builds this dense `Omega`. Both compact/factored
siblings carry `autocast(enabled=False)`; the dense route does not.
**Fix:** Wrap the dense `Omega` einsum and the dense diagonal-congruence einsum in the same
`autocast(enabled=False)` island the factored routes use.

### C-06 — Exact-congruence dense route discards the fp32 promotion it just paid for
**Location:** `vfe3/families/exact_congruence.py:164-167`
**Severity:** high
**Evidence:**
```python
        inverse = torch.linalg.inv(
            omega.to(_linalg_dtype(omega.dtype))).to(omega.dtype)
        s_tilde = torch.einsum("...ijkl,...il->...ijk", inverse ** 2, sigma_q)
```
The `.to(omega.dtype)` rounds the fp32 inverse back before `inverse ** 2` squares it — the opposite
of the sibling `_slogdet` (`:84`), which returns at fp32 by design. Measured (N=3, K=4, bf16):
```
s_tilde rel-err: cast-back-to-bf16 9.395e-03   keep-fp32 2.202e-07
energy error vs fp64: up to 2.352e-01 nats (E in [3.2, 30.2])
```
The fast route is clean. This is the dense/reference route only, which is why the float64 exactness
pins never saw it.
**Fix:** Drop the `.to(omega.dtype)` and keep the inverse at `_linalg_dtype(omega.dtype)`.

### C-07 — `TRANSPORT_CLAMP_MAX_NORM = 20` sits ~2x past where the fp32 pair stops being a group inverse
**Location:** `vfe3/geometry/transport.py:1257`, `:1476`, `:1830`
**Severity:** medium
**Evidence:** `stable_matrix_exp_pair` calls `_checked_group_inverse` with no tolerance, so the
identity-residual check is skipped (`residual_tol: Optional[float] = None` at `:1650`; only
`group_element_inverse` passes `1e-4`). Measured on the STORED fp32 pair, gl(4), 25 draws per norm:
```
  ||M||_F=  5.0  median=5.275e-07  frac exceeding 1e-4:  0/25
  ||M||_F= 10.0  median=2.801e-05  frac exceeding 1e-4:  4/25
  ||M||_F= 16.0  median=1.061e-03  frac exceeding 1e-4: 18/25
  ||M||_F= 20.0  median=2.081e-02  p90=2.808e+00  frac exceeding 1e-4: 25/25
```
`matrix_exp` itself is accurate to ~1e-6 at every norm up to 20 — the clamp is calibrated on the
wrong quantity.
**Fix:** Pass `residual_tol` from the pair builders, or lower the clamp toward ~10.

### C-08 — Full SPD retractions hard-raise on one nonfinite batch element
**Location:** `vfe3/geometry/retraction.py:104`, reached from `:505` and `:622`
**Severity:** medium
**Evidence:** `torch.linalg.eigh` is the raising variant, batched. One `+inf` or `NaN` entry in
batch element 0 raises `_LinAlgError` for `retract_spd_full` and `retract_logeuclidean_full`,
aborting the whole training step. The codebase's own `numerics.safe_cholesky` uses `cholesky_ex`
plus an `ok` mask for exactly this reason.
**Fix:** Sanitize or mask nonfinite rows before `_eigh_damped`, keeping healthy elements alive.

### C-09 — Diagonal SPD retraction returns NaN, not a bounded step, on an overflowing tangent
**Location:** `vfe3/geometry/retraction.py:454-460`, `:695-708`
**Severity:** medium
**Evidence:** `inf * 0` is NaN and `clamp` propagates it, so the trust region converts an
overflowing tangent into NaN instead of a step of length `trust_region`:
```
retract_spd_diagonal, delta=[1, inf, 1] -> tensor([[0.5000, nan, 0.5000]])
   delta_sigma=1e+33  whitened=1.0e+39  out=[nan, nan, nan]
```
The fp32 overflow threshold is `delta_sigma ~ 1e33` when sigma sits at the `eps=1e-6` floor — the
regime a diverging run enters.
**Fix:** Compute the rescale on a `nan_to_num`-ed norm, or clamp `whitened` before forming it.

### C-10 — Congruence floors at exactly `0.0`, admitting a point outside the open SPD cone
**Location:** `vfe3/geometry/transport.py:2407`, `:2519`
**Severity:** medium
**Evidence:** `out.clamp(min=0.0)`. For invertible `Omega` and `D > 0`, congruence preserves inertia
(Sylvester), so every diagonal entry is strictly positive; an output of exactly `0` is a boundary
point, not a representable point of the cone. It then hits `clamp(min=eps)` in the divergence kernel
and inverts that key's precision weight rather than reporting the failure. The retraction path
floors at `eps > 0` and certifies (`retraction.py:539-543`).
**Fix:** Floor at the same strictly positive `eps` the SPD retraction certifies against.

### C-11 — `safe_cholesky` is the only `numerics` helper without the sibling dtype-promotion policy
**Location:** `vfe3/numerics.py:198-199`
**Severity:** low
**Evidence:** `safe_spd_inverse` and `condition_number` both promote; `safe_cholesky` factors the
input dtype directly and raises `NotImplementedError` on bf16/fp16, breaking its own "never raises"
contract. It is on the full-covariance KL path, the covariance table round trip, and the full decode.
**Fix:** Add the same `compute_dtype` promotion and restore the caller's dtype on return.

### C-12 — Finite BCH is divergent in the reachable `||phi||` regime, and higher order is measurably worse
**Location:** `vfe3/geometry/lie_ops.py:263-286`, `:600-643`; `vfe3/config.py:222`
**Severity:** medium
**Evidence:** The Dynkin series is correct — compact and dense agree bit-for-bit and convergence
exponents match `order+2` exactly at small argument. But in the asymmetric regime the code runs:
```
  ||X||   ||Y||        ord1        ord2        ord3        ord4
    1.0    0.10   7.562e-04   2.555e-05   9.508e-06   4.142e-07
    3.0    0.10   4.519e-03   3.931e-04   4.152e-04   9.666e-05   <-- ord3 worse than ord2
    5.0    0.10   7.994e-03   1.259e-03   1.378e-03   6.066e-04   <-- ord3 worse than ord2
```
The classical radius `||X||+||Y|| < log 2` is exceeded, so `O(||X||^{order+2})` is an asymptotic
statement that does not describe this operating point. `_fail_if_bch_residual_exceeds` exists
(`:311-350`) but `bch_residual_max` defaults to `None`, so nothing measures it.
**Fix:** Bound `||phi||` for BCH-composed routes to the convergence radius, or default
`bch_residual_max` to a finite value so an out-of-radius composition fails closed.

---

## Cluster D — variational / information-geometric

### D-01 — `mm_exact` is not a descent method on the frozen-keys F under any registered attention prior
**Location:** mechanism at `vfe3/gradients/kernels.py:649`, `:664-671`; false contract at
`tests/test_tier12_estep.py:106`
**Severity:** high
**Evidence:** The test builds its own prior that hard-masks every diagonal entry, then asserts
descent. No registered prior produces that mask — `causal` (the default) allows `j <= i`, and even
`causal_noself` keeps `(0,0)`. Measured, 600 draws per prior, `mm_damping=1.0`:
```
  causal                                       increased 34/600  worst +45.585  (22.986 -> 68.571)
  causal_noself                                increased  5/600  worst +33.793
  test's hand-built prior                      increased  0/600
```
Mechanism: `E_ii = 0` structurally (measured `diag(E) max = 0.000e+00`), the strict
`pair_mask = ((energy > 0) & (energy < kl_max))` zeroes it, so `w = lambda_beta * (beta * pair_mask)`
drops the row's largest weight — `beta_ii` measures `mean=0.4086 min=0.1668 max=1.0000`, and on
causal row 0 `pair_prec = 0` makes `mu_star = mu_p` unconditionally. The `config.py:709-711`
concession ("carries NO majorization / monotone-F-descent") is accurate; the test is the only place
claiming otherwise, and it is registered as a pinned contract in `tests/pytest_policy.py:51`.
**Fix:** Rename/re-scope the test to the self-key-masked hypothesis it actually exercises and add a
regression recording the counterexample under `beta_attention_prior='causal'`.

### D-02 — `mm_exact_update` bypasses the family coupling seam and its own route predicate
**Location:** `vfe3/gradients/kernels.py:598-628` (energy at `:618`)
**Severity:** high
**Evidence:** The sibling `belief_gradients` calls `uses_kernel_route(...)` at `:377` and hands a
non-covered config to the oracle. `mm_exact_update` takes the same `family`/`divergence_family`
arguments, never calls that predicate, and builds its grid from `pairwise_energy` instead of
`fam.coupling_energy`:
```
[2] mm_exact_update: gaussian_diagonal vs gaussian_diagonal_exact -> identical mu*? True  max|diff| = 0.0
[3] belief_gradients gaussian_diagonal vs _exact -> identical grad_mu? False max|diff| = 0.2936
  mm_exact_update(divergence_family='jeffreys')          -> ran   uses_kernel_route=False
  mm_exact_update(divergence_family='bhattacharyya')     -> ran   uses_kernel_route=False
  mm_exact_update(divergence_family='squared_hellinger') -> ran   uses_kernel_route=False
```
So `gaussian_diagonal_exact` silently gets the truncated energy, and a non-Rényi divergence drives
beta and `pair_mask` and is then fused with hardcoded diagonal-Gaussian-KL precision expressions.
There is no `renyi_order` parameter at all; `alpha=1.0` is hardcoded. Both live call sites gate
first (`e_step.py:994`, `config.py:2547`), so no shipped config is wrong today.
**Fix:** Call `uses_kernel_route(...)` internally and raise, or route the grid through
`fam.coupling_energy`.

### D-03 — The purity ledger omits the E-step-update axis, so an `mm_exact` run reports `on_pure_path: True`
**Location:** `vfe3/run_artifacts.py:3895-3967`
**Severity:** medium
**Evidence:** `pure_flags` has no `e_step_update` key and `on_pure_path = all(pure_flags.values())`;
`config_toggles` has neither `e_step_update` nor `mm_damping`. The only surviving trace,
`state_dependent_alpha_majorizer`, is `False` at the pure `lambda_alpha_mode='constant'`. So a run
whose E-step descends the self-pair-masked surrogate of D-01 is published as on the theoretically
pure path with no record of the substitution.
**Fix:** Add `e_step_update == "gradient"` to `pure_flags` and both fields to `config_toggles`.

### D-04 — `attention_entropy_cov_gap`'s stated premise is contradicted by the executable path
**Location:** `vfe3/viz/extract.py:1151`
**Severity:** medium
**Evidence:** The instrument is framed as measuring what production is missing. Measured:
```
max|kernel_mu - oracle_canonical_mu|        = 4.768e-07
max|oracle_canon_mu - oracle_surrogate_mu|  = 7.516e-01
```
and `uses_kernel_route(...)` is `True` at the default config, `False` when
`include_attention_entropy=False`. So the kernel is served only on the canonical path and reproduces
autograd-of-F-with-entropy to 5e-7; the surrogate is served exclusively by the oracle. SymPy confirms
the row-Lagrangian residual at softmax is `[0,0,0]`, so the envelope derivative is exactly
`sum_j beta* dE/dtheta` — what the kernel computes. `cov_gap` is a valid canonical-minus-surrogate
*objective* gap, not an error production incurs.
**Fix:** Restate the premise as the objective gap and drop the claim that the kernel descends the
surrogate.

### D-05 — Exact-congruence fast route accepts an energy partition disagreeing with the transport partition
**Location:** `vfe3/families/exact_congruence.py:130-136`
**Severity:** medium
**Evidence:** `_vertex_log_abs_det` checks only the block *count*, never the block *sizes*. With
transport blocks `[2,2]` and energy `irrep_dims=[1,3]` the guard passes and per-block determinants
are paired with mismatched coordinate slices:
```
  max |returned - true| = 11.545606981774455
```
The dense route rejects the identical input, so the two routes disagree on the guard. Not reachable
from `compute_transport_operators` (which always sets `irrep_dims = group.irrep_dims`), but it
returns a silently wrong divergence against this family's stated fail-closed contract.
**Fix:** Compare `list(irrep_dims)` against the transport block sizes elementwise.

### D-06 — Exact-congruence dense reference route assumes a block-diagonal `Omega` without checking
**Location:** `vfe3/families/exact_congruence.py:164-177`
**Severity:** medium
**Evidence:** The inverse is taken on the full `K x K` operator while per-block determinants come
from the diagonal blocks only; that composition is the correct block marginal only when `Omega` is
block-diagonal. With a deliberately non-block-diagonal dense `Omega` and `irrep_dims=[2,2]`:
```
  exact family vs true block-marginal KL, max|err| = 4.888e+02  (raised? no)
```
Live builders keep `Omega` block-diagonal whenever `len(irrep_dims) > 1`, so this is latent.
**Fix:** Assert the off-block mass is zero, or fall back to the full-`K` single-block energy.

### D-07 — `reuse_pairwise_kl_stats` is not value-preserving on the grid that gates `pair_mask`
**Location:** `vfe3/gradients/pairwise_stats.py:96-121`
**Severity:** low
**Evidence:** The three coordinate reductions promote to `float64` and cast back, whereas
`DiagonalGaussian.renyi_closed_form` reduces at `float32`:
```
irrep=[4]:    max|E_stats - E_generic| = 7.629e-06   bitwise equal = False   mask mismatches = 0 / 36
irrep=[2, 2]: max|E_stats - E_generic| = 7.629e-06   bitwise equal = False   mask mismatches = 0 / 72
```
No mask flip was observed, but the toggle moves the energy that `beta` and `pair_mask` are computed
from, so it is a value toggle, not a pure hoist.
**Fix:** Reduce in `float32` to match, or document it as value-changing and pin the mask-agreement bound.

### D-08 — `mass_phi` denotes two incommensurate penalties, and the logged F carries neither
**Location:** `vfe3/inference/e_step.py:771` vs `vfe3/model/model.py:1602`
**Severity:** low
**Evidence:** `0.5*mass_phi*(phi**2).sum()` in the E-step objective against
`0.5*cfg.mass_phi*(belief.phi**2).mean()` in the M-step loss — one coefficient denoting penalties
differing by `phi.numel()`. `ablation.py:1611` sweeps it over 7 values. `free_energy_value` declares
`mass_phi` "accepted-and-ignored", so at `mass_phi > 0` the logged F and the Metropolis frame scorer
evaluate a functional the phi substep does not descend.
**Fix:** Pick one reduction and honor `mass_phi` in `free_energy_value` for the global F.

### D-09 — Reported `attention_entropy` uses a floored log while the scored total uses `xlogy`
**Location:** `vfe3/metrics.py:402`
**Severity:** low
**Fix:** Use `xlogy` so the raw and scored entropy are one expression.

---

## Cluster E — attention, positional, and config

### E-01 — Non-causal `gamma_attention_prior` leaks future tokens into the causally masked belief prior
**Location:** `vfe3/model/model.py:2364-2377`
**Severity:** high
**Evidence:** `gamma` is normalized over the *full* key row before beta's mask is applied, and the
mixture is renormalized afterward — a convex mixture is not proportional to a single exponential, so
gamma's row normalizer `Z_i` (summing over `j > i`) does not cancel. Measured on `log_prior[i<4,j<4]`
(all strictly in the past), two sequences differing only in the last token:
```
gamma_attention_prior=causal                  max|d log_prior| = 0.000e+00
gamma_attention_prior=causal_alibi_noself     max|d log_prior| = 0.000e+00
gamma_attention_prior=uniform                 max|d log_prior| = 2.962e-05
gamma_attention_prior=alibi                   max|d log_prior| = 4.292e-06
gamma_attention_prior=windowed                max|d log_prior| = 2.962e-05
```
End-to-end `max|dlogits|` at positions `< t`:
```
gamma_as_beta_prior + gamma causal    0.000e+00
gamma_as_beta_prior + gamma UNIFORM   1.490e-08
gamma_as_beta_prior + gamma alibi     2.794e-09
```
`config.py:1513` validates `gamma_attention_prior` only against the registry; nothing requires its
support to contain beta's. `gamma_as_beta_prior=True` is live in `train_vfe3.py:421`, and only the
causal `gamma_attention_prior` there keeps the run clean.
**Fix:** Reject in `__post_init__` (or pre-mask `e_s`/`gamma_log_prior` with beta's support before
the gamma softmax) whenever `gamma_as_beta_prior=True` and gamma's allowed set is not a subset of
beta's.

### E-02 — `gaussian_diagonal_exact` + gauge-RoPE is config-accepted and hard-crashes on first forward
**Location:** `vfe3/geometry/transport.py:1960-1965`, `vfe3/families/exact_congruence.py:115-126`
**Severity:** medium
**Evidence:** `_certifies_same_frame_flat_cocycle` recurses into `RopeTransport` and returns `True`,
so `_pullback_query` takes the fast route — but `_vertex_log_abs_det` has no `RopeTransport` branch
and raises `TypeError: no vertex factors on transport 'RopeTransport'`. The route is mathematically
available: `rope.py:76-91` builds `R` block-diagonal from 2x2 rotations, so `R` is exactly orthogonal
and `|det Omega^RoPE| = |det Omega|`.
**Fix:** Unwrap `RopeTransport` to its base inside `_vertex_log_abs_det`, or make the predicate
return `False` for it so the family's own documented error fires.

### E-03 — `gaussian_diagonal_exact` scope violations surface mid-forward, not at config validation
**Location:** `vfe3/families/exact_congruence.py:260-267`; validator gap at `vfe3/config.py:1447`
**Severity:** medium
**Evidence:** `grep -n "diagonal_exact" vfe3/config.py` returns nothing outside two exclusion tuples.
```
exact + renyi 0.5            ACCEPTED (config)  -> FWD ValueError
exact + squared_hellinger    ACCEPTED (config)  -> FWD ValueError
exact + regime_ii_link       ACCEPTED (config)  -> FWD TypeError
exact + pos_rotation='rope'  ACCEPTED (config)  -> FWD TypeError
```
Every neighboring family/mode pairing is gated in `config.py`; this one is not. In `ablation.py`,
`run_single` catches config errors only around `VFE3Config(**cfg_dict)` (`:2574-2580`), so a
mid-forward raise falls to the outer `except Exception` (`:4366`) and the cell is misfiled as
`error_kind="train"`.
**Fix:** Mirror the family's runtime preconditions into `__post_init__` via a family-declared
capability record.

### E-04 — Registry seam broken: hardcoded family tuples reject a family both call sites support
**Location:** `vfe3/config.py:1590`, `:2112`
**Severity:** medium
**Evidence:** Both are fail-closed with a named message (verified, not silent), but wrong on the
merits: `ExactCongruenceDiagonalGaussian(DiagonalGaussian)` overrides only `block`,
`broadcast_over_keys`, `stack`, and `coupling_energy`; its self-divergence, moments and decode are
inherited unchanged and `cov_kind = "diagonal"`, so `PriorBank.barycenter_r_`'s diagonal moment match
(`prior_bank.py:790`, `:804-809`) is exactly right for it. Every neighboring validator in the file is
registry-driven; these two literals are the exceptions, so registering a Gaussian family requires
editing `config.py`, against the add-by-registering contract.
**Fix:** Replace both literals with a family-registry capability query.

### E-05 — The executable gauge-invariance verifier cannot cover either 2026-07 family
**Location:** `vfe3/geometry/groups.py:395-404`, `:156-162`
**Severity:** medium (elevated from the reporter's "low": this is why A-01 shipped undetected)
**Evidence:** `check_admissible` — the project's only executable invariance certificate — hard-codes
three family names and raises `NotImplementedError` otherwise. Every group declares
`invariant_families = ("gaussian", "gaussian_full")`, and the registry contains no `"gaussian"`:
```
registered families: ('gaussian_diagonal', 'gaussian_diagonal_exact', 'gaussian_frame_diagonal', 'gaussian_full', 'laplace_diagonal')
  group block_glk      invariant_families = ('gaussian', 'gaussian_full')
gaussian_diagonal_exact    family_group_invariant=False  on_gauge_pure_path=False
```
So `"gaussian"` is dead data, `gaussian_diagonal_exact` — whose coupling energy *is* the exact
GL(K)-invariant congruence divergence — is reported non-invariant, and neither family added this
month is covered by any invariance test. `invariant_for` has zero consumers.
**Fix:** Give each registered family a pushforward hook so `check_admissible` can exercise it; drop
the dead `"gaussian"` key; wire `invariant_for` in or delete it.

### E-06 — `learnable_kappa_beta` on a single-irrep-block group makes `tau` rank-1 and corrupts shapes
**Location:** `vfe3/free_energy.py:38-40`, `vfe3/config.py:1488-1495`
**Severity:** medium
**Evidence:** `log_kappa_beta` is built with shape `(len(irrep_dims),) = (1,)` on `glk`/`so_k`;
`_broadcast_tau` unconditionally treats a 1-d tau as per-head, adding a leading axis that propagates
into `grad_sigma`:
```
tau: tensor([2.]) torch.Size([1])
  energy (5, 5)    -> beta (1, 5, 5)   (scalar-tau beta (5, 5))
attention_maps raised: ValueError Size of label 'j' for operand 1 (4) does not match previous terms (5)
```
The equivalent explicit config is rejected (`config.py:1463`) while `learnable_kappa_beta=True` only
warns (`:1490`).
**Fix:** Collapse a length-1 kappa vector to a 0-d scalar, or reject the toggle on single-block groups.

### E-07 — `rope_on_value` carries three contradicting defaults
**Location:** `vfe3/config.py:241`, `vfe3/geometry/transport.py:246`, `vfe3/model/block.py:93`
**Severity:** medium
**Evidence:**
```
VFE3Config.rope_on_value default     = False
RopeTransport.on_value default       = True
vfe_block(rope_on_value=...) default = True
```
(`stack.py:51` and `e_step.py:305,478,696,864,1260` also default `True`.) Any direct caller
forwarding `rope=` without `rope_on_value=` silently gets the coupled gauge, the opposite of config.
This is the code-level substrate of the 4 stale rope test failures. Separately, gauge-RoPE plus the
shipped `e_step_update='mm_exact'` is mutually exclusive at config time
(`uses_kernel_route(..., decoupled_value_gauge=True)` is `False`, `kernels.py:307`), so every RoPE
run must also change the update rule.
**Fix:** Make the parameter defaults match `VFE3Config.rope_on_value`, or make them required
keyword arguments wherever `rope` is passed.

### E-08 — `metrics.py::gauge_equivariance_residual` still inlines the pre-migration composition
**Location:** `vfe3/metrics.py:1572-1627`
**Severity:** medium
**Evidence:** The `coupling_energy` docstring claims to be "the single seam every consumer of the
coupling grid dispatches through", naming the attention-map diagnostics. The 2026-07-25 migration
rewired five call sites; this one still does
`transport_mean(...)` / `transport_dispersion(...)` / `pairwise_energy(...)` inline. Currently
byte-identical (`FullGaussian` does not override `coupling_energy`), so a doc/code consistency
defect rather than a behavioral bug — but a future specialization would silently not be honored.
**Fix:** Replace the inline composition with `full.coupling_energy(...)`, or narrow the docstring's
claim.

### E-09 — The "inert configuration setting(s)" warning list omits four live inert combinations
**Location:** `vfe3/config.py:2671-2734`
**Severity:** low
**Evidence:** No warning anywhere for `emit_expensive_diagnostics` with `generate_figures=False`,
`query_tau_c` with `query_adaptive_tau=False`, `s_e_step_n_iter` with `s_e_step=False`, or
`decode_mode` on the linear-decode path (`untie_decode_bank` does warn, `:2643-2649`).
**Fix:** Add the four guards using the existing `_changed(...)` helper.

### E-10 — `mm_exact_update` reads `omega.on_value` but never uses the un-rotated base
**Location:** `vfe3/gradients/kernels.py:595-663`
**Severity:** low
**Evidence:** `decoupled_value` gates only the pair-statistics hoist; the fusion is built entirely
from the rotated transport, unlike `oracle.py:216-221` and `e_step.py:645-649`, `:776-788` which
rebuild from `omega.base`. Live callers are protected by `e_step.py:994-1004`.
**Fix:** Raise inside `mm_exact_update` under a decoupled gauge, mirroring the caller-side guard.

### E-11 — `attention_weights` returns NaN for an all-masked row with no guard
**Location:** `vfe3/free_energy.py:318-321`
**Severity:** low
**Evidence:** Verified latent, not live: a sweep of all nine registered priors for `N = 1..6` at the
tightest `window=0` produced no all-masked row on any route, and `attention_window` is validated
`>= 1` at config time. A new registered prior or a cross-attention `n_query > n_key` call site would
trip it silently.
**Fix:** Guard the row support, or assert at least one finite logit per row.

### E-12 — Gauge-RoPE is exactly the identity on 1-dimensional irrep blocks
**Location:** `vfe3/geometry/rope.py:81-90`, `vfe3/config.py:1965`
**Severity:** low
**Evidence:** `n_pairs = d // 2` is `0` for `d = 1`. On an `so_n` tower `irrep_dims=[1,3,5,7]`, head
0 receives zero positional signal; every odd block leaves its last coordinate un-rotated. Inherent
to RoPE (rotations are defined on coordinate pairs), so this needs surfacing rather than fixing.
**Fix:** Warn at config time when `pos_rotation='rope'` and any `irrep_dims` entry is odd.

### E-13 — UMAP worker `Popen` bypasses the repo's process-containment helper
**Location:** `vfe3/viz/figures.py:187-194`
**Severity:** medium (pre-existing; carried over as M18 from `docs/audits/audit-2026-07-24.md`)
**Evidence:** `process_utils.run_process_tree` exists to "contain every descendant in one disposable
process tree" and is used by `run_artifacts.py:592/685`, `ablation.py:5510`, `make_figures.py:188`,
`run_cpu_tests.py:171`. This one long-lived worker, which spawns numba, opts out.
**Fix:** Route through `process_utils`, or add a `Popen`-shaped entry point there.

### E-14 — `apply_mu_trust_region(mode='box')` breaks GL(K) congruence equivariance
**Location:** `vfe3/numerics.py:146`, `:160`
**Severity:** low
**Evidence:** Under `Sigma -> g Sigma g^T`, `chol(g Sigma g^T) = g L Q` for orthogonal `Q`, so the
whitened vector rotates and a per-coordinate box does not commute:
```
mode=box    rel equivariance error = 6.244e-01
mode=ball   rel equivariance error = 2.012e-16
```
Default `e_mu_q_trust=None` leaves the guard off, so a pure path exists; the docstring names `box`
"the recommended mode".
**Fix:** Note that `ball` is the equivariant mode and `box` a non-equivariant baseline.

### E-15 — Orphaned import left by the 2026-07-25 extractor fix
**Location:** `vfe3/viz/extract.py:1236`
**Severity:** low
**Evidence:** `from vfe3.families.gaussian import DiagonalGaussian` is the file's only reference to
that name (confirmed by pyflakes and grep); the commit replaced every use with `fam(...)`.
**Fix:** Delete the line.

### E-16 — Three new figure plotters have no return-type annotation
**Location:** `vfe3/viz/figures.py:1165`, `:1211`, `:1243`
**Severity:** low
**Evidence:** All three return a `Figure` via `_save(fig, path)` but declare no return type, against
the project's "type hints on every signature" mandate. Consistent with the pre-existing module-wide
gap, so a reproduction rather than a regression.
**Fix:** Annotate the return type on all three.

### E-17 — `n_e_steps` validated `>= 1` but set to `0` by a probe with no re-validation
**Location:** `vfe3/config.py:980-981` vs `vfe3/run_artifacts.py:3047-3048`, `:2885`
**Severity:** medium
**Evidence:** `collect_estep_depth_sensitivity` assigns `cfg.n_e_steps = 0`, a value the dataclass's
own `__post_init__` rejects, by direct attribute assignment. Mechanically harmless today
(`range(0)` is a legal no-op) but the field's enforced domain is no longer true of every live
instance.
**Fix:** Either loosen the validated domain to `>= 0` or drive the loop count without mutating the
validated field.

### E-18 — `Optional[float]` annotations that are never `None`, feeding an unguarded format
**Location:** `vfe3/run_artifacts.py:3383`, `:3470`; consumed at `:3151-3158`
**Severity:** low
**Evidence:** Both dicts are declared `Dict[str, Optional[float]]` but every populating path assigns
a concrete `float`. `_log_mechanism_diagnostics`'s `f"{value:.4f}"` has no `None` check, so a future
edit matching the declared type would raise at the end of a training run.
**Fix:** Narrow the annotations to `Dict[str, float]`, or add a `None`-skip in the formatter.

---

# Clean negatives (verified, not assumed)

These are audit results in their own right and several were explicit asks.

**Diagnostics.** The `collect_estep_character` monkeypatch restoration is exception-safe with no
reachable leak path (B-10). `collect_beta_channel_decomposition`'s `no_prior` arm is a valid
ablation: `torch.where(isneginf(log_prior), log_prior, zeros)` verified to preserve the mask while
flattening the graded shape, with `full` reproducing the unpatched CE. The cheap/expensive tier
ordering is correctly calibrated — 8 bounded forwards for the cheap pair against up to 15
`e_step_belief_trace` calls for the gated one. All three collectors return genuinely
JSON-serializable dicts on every branch with `Optional` fields consistently `None` rather than absent.

**Causality.** No causal leak on any live route. Perturbing token `t` and measuring `max|dlogits|`
at positions `< t` gave exactly `0.000e+00` for `causal`, `causal_noself`, `causal_alibi`,
`causal_alibi_noself`, `causal_windowed` and `t5_relative_bias`, under both `e_step_update` values,
and additionally under `precision_weighted_attention`, `query_adaptive_tau`, `lambda_twohop=0.3`,
`use_head_mixer`, mahalanobis norms, `pos_rotation='rope'` (both `rope_on_value` settings), a causal
`gamma_as_beta_prior` fold, and an unequal `so_n` tower. Non-causal controls leaked `8.5e-03` to
`8.6e-03`, confirming probe sensitivity. Live beta row sums in `[0.9999999, 1.0000001]`, future mass
exactly `0.000e+00`, no-self diagonal mass exactly `0.000e+00`. The pair mask never re-opens a masked
entry; the two-hop `beta @ beta` stays lower-triangular. (E-01 is the one config-gated exception.)

**Free energy.** The implemented `F` equals its envelope exactly (`F_canonical = 5.93036556`,
`-tau*sum logZ = 5.93036556`, difference `0.000e+00`). The canonical/surrogate toggle is genuinely
selectable and routes the gradient to the oracle when off. SymPy: with the entropy term the row
Lagrangian's residual at softmax is `[0,0,0]` with Hessian `diag(lb*tau/b_j) > 0`; without it no
interior stationary point exists and the minimizer is the vertex `delta_argmin(E)`. The gradient gap
residual `(dG - dF) + Cov_beta(E,dE)/tau` simplifies to `0`.

**mm_exact stationarity.** The fusion is the exact stationary point of the objective its docstring
states: rebuilding `F_hat` by hand at frozen beta/keys/masks and differentiating at `(mu*, sigma*)`
gives `||grad_mu||_inf = 2.4e-07`, `||grad_sigma||_inf = 7.5e-08`. `mm_damping` is applied as
documented (natural-coordinate blend, `e_step.py:1036-1039`). The non-majorizer concession is
accurate — the dropped self-pairs are non-negative and vanish at the current point, making the
surrogate a minorant.

**Kernel/oracle parity.** `belief_gradients` matches `belief_gradients_autograd` to `<= 4.8e-07`
across 14 combinations (baseline, `lambda_beta=0.4`, `lambda_twohop=0.3`, both, both
state-dependent alphas, finite and `-inf` log-priors, `irrep_dims=[2,2]`, per-head tau,
`query_adaptive_tau` single-block and per-head, twohop x per-head, twohop x causal). An independent
fp64 central-difference check of `belief_gradients` gave `grad_mu` max abs diff `1.568e-09`,
`grad_sigma` `1.582e-09`. The saturation mask is consistent on both routes: `torch.clamp` backward
passes gradient at both exact bounds, so the strict kernel mask differs from autograd only on a
measure-zero set, and is inert there because `dD = 0` at `D = 0` and `E_ii = 0` exactly.

**E/M separation.** `forward_beliefs` takes only `token_ids`; `targets` never enters belief
production. `free_energy`'s `log_likelihood` has **no production caller** —
`belief_gradients_autograd` never passes it, `e_step`'s `free_energy_value` never passes it, and
`train.py:973-976` records that the reported total omits it. The deployed loop is two objectives,
i.e. structural EM, and the source states the non-monotonicity of the parallel Jacobi update rather
than overclaiming. `gradient_mode='smoothing'` (the exact `dF`) exists as the pure path; `filtering`
differs from it by `||diff||/||dF|| = 0.186`, `cos = 0.9832`.

**No hyper-prior / gamma double-count.** `_model_channel_free_energy` is the sole loss-side adder
and is gated on `not self.cfg.s_e_step`; `_refine_s` is called only under `if self.cfg.s_e_step:`.
The coupling row and meta-entropy row come from disjoint expressions.

**Two-hop objective parity.** `phi_alignment_loss` and the scored scalar are the same function of
phi: `|L_phi - F_scalar| = 4.768e-07` and `max|dL/dphi - dF/dphi| = 4.768e-07` at both
`lambda_twohop = 0.0` and `0.4`.

**Stack prior handoff is gauge-coherent.** `mu_p = (1-rho)*mu_p + rho*belief.mu` (`stack.py:147`) is
correct as written: `mu_p_i` and `mu_q_i` live in the same fiber over token `i`, so no transport
belongs there — transport couples distinct tokens, which `pairwise_energy` handles. It is a cascade
of L per-layer variational problems, each a proper `F_l` with `p_{l+1} = q_l*`, and the code claims
nothing more. (This supersedes the "degenerate one-token-per-agent shadow" characterization in
`docs/2026-07-25-shadow-prior-investigation.md`.) The real structural note is that at the default
`prior_handoff_sigma=0.0` the prior variance never advances with depth.

**Exact-congruence pullback identity.** SymPy, `K=3`, fully symbolic dense `Omega` with symbolic
diagonal `s_i`, `s_j`: `LHS - RHS = 0` and `code_energy - RHS = 0` after exact rational substitution.
Numerically against a brute-force dense-KL oracle: dense reference `8.5e-14`; `FactoredTransport`
fast route `2.5e-14` (`[4]`), `8.9e-15` (`[2,2]`); `CompactFactoredTransport` `2.8e-14`; self-pairs
`E_ii = 0.0` exactly; non-negativity holds. fp32 degrades gracefully to `1.1e-4` relative at
`cond(U) = 1.5e6`. The KL-specificity guard fires correctly for `alpha=0.5`, `jeffreys`, RoPE-wrapped
and uncertified cocycles.

**`coupling_energy` default body.** Diffing the migration against every rewired site, the
`diagonal_out` resolution, batch lift/drop, and the `omega.base` value-gauge second call are
preserved one-for-one. `DiagonalGaussian` overrides neither transport seam, so the base seam resolves
to the same primitives the hand kernel calls directly; the kernel route is gated to
`family == "gaussian_diagonal"`, so a family needing different composition is excluded from the
shortcut.

**`alpha* = c0/(b0 + D)`.** Exactly the stationary point of the stated log-barrier:
`d/dalpha[alpha*D + b0*alpha - c0*log alpha] = 0` gives it, with `d^2/dalpha^2 = c0/alpha^2 > 0`, and
the envelope residual `D + R'(alpha*) = 0` is why no product-rule correction is needed. Per-coord vs
per-position broadcast is guarded at construction.

**Divergence registry.** Across `{renyi, squared_hellinger, bhattacharyya, jeffreys} x
{gaussian_diagonal, gaussian_full, laplace_diagonal}` at `alpha in {0.25, 0.5, 1.0, 2.0}`, fp64:
`D(p||p) = 0` and `D >= 0` everywhere, zero violations. Renyi -> KL at the correct first order.
Generic Bregman path agrees with the pinned closed form to `<= 6.4e-13`.

**Fisher / natural gradient.** Diagonal `(sigma*grad_mu, 2 sigma^2*grad_sigma)` and full
`(Sigma grad_mu, 2 Sigma grad_sigma Sigma)` are exact inverses of the Gaussian Fisher; Laplace
`b^2*grad` is correct on both blocks. The preconditioner dispatches through the family seam, not a
hardcoded Gaussian. `killing_metric` is the genuine Cartan-involution form; `pullback_metric` uses
the correct right-trivialized `dexp` kernel.

**SPD geometry.** `retract_spd_full` is the exact affine-invariant exponential map (max abs err
`2.22e-15`; round trip `5.22e-15`) and is exactly congruence-equivariant on the pure path
(`1.03e-15` orthogonal, `4.24e-16` scale 3, `3.88e-16` scale 0.02); the `sigma_max=10.0` ceiling
breaks equivariance exactly as an eigenvalue bound must, and the unbounded pure path exists.
`_certify_public_spd` does not collapse well-conditioned covariances. `retract_logeuclidean_full`
has identity first derivative. `_blockwise_matrix_exp` is exact for block-diagonal input. The
diagonal truncation `diag(Omega Sigma Omega^T)` is exactly the Frobenius orthogonal projection onto
the diagonal cone and coincides with the Amari m-projection `argmin_D KL(N(0,S)||N(0,D))` — a
defensible information-geometric choice, though not the metric projection under the affine-invariant
or log-Euclidean metric the retractions use (it inflates transported variances ~25-30%).

**Gauge structure.** Every covariance route forms `Omega Sigma Omega^T`, never `Omega Sigma` or
`Omega Sigma Omega^{-1}`. `flat` is exactly flat: `|Omega_ii - I| = 1.2e-07`, triangle holonomy
`3.6e-07`, `|Omega_ij Omega_ji - I| = 2.4e-07` at fp32. `_certifies_same_frame_flat_cocycle` sets
`True` only where the coboundary genuinely holds; `_apply_reflection` correctly clears the flag on a
key/query mismatch; `_transport_qk`'s mixed frames are never certified; `DirectLinkTransport` never
certifies. Head mixer and CG coupling are exactly equivariant on labeled towers (`4.4e-16` to
`8.9e-16`) — the module docstring's "diagonal gauges only" caveat is over-conservative — and are
byte-identical no-ops when disabled while still passing gradient. `irreps.py` verifies the Lie
bracket at build time and raises on residual. `MahalanobisNorm` rescales `mu` by a
congruence-invariant scalar and leaves `Sigma` untouched, so it is exactly equivariant.

**Attention mechanics.** `attention_tau([1,3,5,7]) = [1.0, 1.732, 2.236, 2.646]` — `kappa*sqrt(d_h)`
per irrep block everywhere, with the scaling inside the exponent and aligned to the correct axis.
The Gibbs envelope identity held to `2.4e-07` under a causal `-inf` prior. RoPE algebra is correct:
`RR^T - I` at `1.2e-07`, off-block entries exactly `0`, relative-position identity
`R_i R_j^T = R(theta_{i-j})` to `1.5e-08`, frequency schedule matching the source. ALiBi head-count
alignment is guarded at both config and model construction.

**Pure-path existence — the central `CLAUDE.md` audit question.** A fully pure config
(`use_prior_bank=True`, no mixer, no CG, `transport_mode='flat'`, `layernorm_affine=False`,
norms `'none'`, `query_adaptive_tau=False`, no learnable kappa/T5, `pos_phi='none'`,
`learnable_r=False`, `gradient_mode='filtering'`, `family='gaussian_diagonal'`, `renyi` at order
1.0, `include_attention_entropy=True`) constructs with **zero warnings**, takes the closed-form
kernel route, forwards and backwards, and carries exactly four learned tensors — `mu_embed`,
`sigma_log_embed`, `phi_embed`, `decode_log_scale` — all `float32`, all with nonzero gradient. No
mixer or connection parameter exists on that path. The gauge-pure `none` and `mahalanobis` norms are
both registered and are the shipped defaults.

**No neural networks.** No `nn.Linear`, `nn.Conv`, `nn.Sequential`, activation, or `nn.LayerNorm`
module appears anywhere in `vfe3/` — every grep hit is a comment.

**Registry reachability.** All five registered families forward and backward under both decode
routes. `divergence_family`, `gauge_group`, `transport_mode`, `spd_retract_mode`, `phi_retract_mode`,
`phi_precond_mode`, `lambda_alpha_mode`, `lambda_h_mode`, `e_step_update`, `decode_mode`,
`encode_mode`, `pos_phi`, `pos_rotation`, both attention priors, and both norm seams are validated
against live registries rather than literals. The two exceptions are E-04.

**Config hygiene.** All 164 `VFE3Config` fields have a non-comment consumer outside `config.py`. No
`allow_unused` anywhere in `vfe3/`, so no broken-gradient path is masked. Float32 discipline holds on
the active path; the `float64` occurrences are construction-time exact-integer generator builds or
opt-in islands.

**Ablation infrastructure.** `STRICT_CODE_IDENTITY=False` provably cannot delete or invalidate a
finished cohort: the two constant sentinels make the terminal snapshot always equal the invocation
snapshot, `code_identity_error` is always `None`, and `_invalidate_code_drifted_cells` is
unreachable. `validate_sweeps` runs over all declared sweeps before any training, field names are
checked against the dataclass, and a cell whose config is rejected returns `error_kind="config"`
rather than aborting the sweep.

**Security.** Every `torch.load` passes `weights_only` explicitly; the sole `weights_only=False`
(`run_artifacts.py:1795`) is reachable only after the safe load raises *and* only under
`cfg.trust_resume_checkpoint`, with the risk named in the raised error. No `pickle`, `yaml.load`,
`eval`, `exec`, `os.system`, or `shell=True` anywhere. No `argparse`/`sys.argv` in project code. No
hardcoded user paths or secrets.

**Migration completeness.** The `coupling_energy` migration is complete across `oracle.py`,
`e_step.py`, `model.py`, and `viz/extract.py` with no orphaned imports (E-08 and E-15 are the two
exceptions). The three new plotters are registered via `@register_figure` and consumed by
`_render_saved_probe_figures` exactly as the two pre-existing probe figures are, with matching JSON
keys. Both new families register correctly on any import path.

---

# Verifier verdicts

An independent `general-purpose` verifier re-read every cited line. It assessed 26 findings
(all critical/high, plus prioritized mediums and the B-10 clean negative).

**Result: 26 CONFIRMED, 0 REFUTED, 0 INCONCLUSIVE, 0 contradictions.** Every pair it checked is
jointly satisfiable; C-01/C-02/C-03 are three simultaneously-true properties of one code region.

Verifier corrections to this report:
- **C-03 re-severitied critical → high.** The sync is a throughput cost on a path whose
  *correctness* problem is C-01; critical should be reserved for the correctness half. It also
  found the sync is once **per irrep block**, not per build — stronger than reported.
- **C-01 independently re-measured**: 172% relative error with the guard silent at `cond(U)=1.4e4`
  (the report said 44% at 1e4). Shape confirmed, magnitude larger.
- **E-05's "`invariant_for` has zero consumers" is FALSE.** It has three test consumers
  (`tests/test_admissibility_verifier.py:38`, `tests/test_gauge_groups.py:146`, `:184`). Zero
  *production* consumers is correct.
- **B-01 settled by direct call-order instrumentation**, not inference: at
  `n_layers=2, s_e_step=True`, the sequence is
  `['S_CHANNEL','S_CHANNEL','S_CHANNEL','belief','belief','belief','belief']`.

# Adversarial challenge

Four duels on the highest-impact confirmed findings. The remaining nine confirmed-high findings
(A-02, B-02, B-03, C-04, C-05, C-06, D-02, E-01) were **bounded out of this wave by the ~6-duel
cap, not dropped**; B-03 in particular is triple-confirmed with a repro and needs no duel to stand.

| # | Finding | Red (attack) | Blue (defend) | Verdict | Reason |
|---|---|---|---|---|---|
| A-01 | frame family discards transport | `connection_W` is zero-init (`transport.py:521`) and at W=0 `regime_ii` reduces to flat (`:729-731`); gradient severed so W never leaves 0; probe shows `loss flat == loss regime_ii` **bitwise** | reachable via `ablation.py:249` + `:1086-1096`; provenance writes `flat_transport: False` for a bit-flat run; the arm's purpose is `connection_W` trainability | **DOWNGRADED critical → high** | No incorrect numerics are produced — the composite is exactly the pure Regime-I model (`transport.py:521`, `:729-731`). But `run_artifacts.py:3897`/`:3925` record `regime_ii` as active for a bit-flat run, so a sweep arm silently cannot do its job. Above a dead toggle, below critical. |
| A-02 | RoPE/reflection inert under frame family | for a flat base, `R_iΩ_ijR_j^T = (R_iU_i)(R_jU_j)^{-1}` is still an exact coboundary, so the cancellation is **correct** and `E_rope − E_norope = 0` is the right answer | (not duelled) | **RE-SCOPED — partly misdiagnosed** | Correct behavior at `rope_full_gauge=True`. Open at `rope_full_gauge=False`, where the geometer measured mean transporting under `R_iΩ_ijR_j^T` while covariance transports under bare `Ω_ij` (`transport.py:2074-2081`) — no single frame works. Needs its own check; neither side established it. |
| B-01 | character probe misattributes channel | attacked every escape route (`_pair_contract` unconditional at `kernels.py:659-663`; no zero-iteration escape; `points` cannot disambiguate) — all close. **Conceded.** | s prior is `r_mu` broadcast (`model.py:850-852`), not `s1`; the two fusions carry different priors, pair weights and alphas; `recompute_max_abs_err` is per-call algebra with no power over channel identity | **UPHELD at high, impact narrowed** | Repro: shipped `pair_precision_share=0.7468` vs belief-only `0.4810` — 1.55x error on one forward. |
| C-01 | congruence guard tests sign only | recorded `metrics.csv` across archived runs: `vertex_cond` max over all steps **= 796**, against evidence measured at 1e4–1.4e4; worst-entry error ≤ 1e-3 at every logged conditioning, below the bf16 floor this project already accepted as benign; `retract_phi`'s `max_norm=5.0` is gated on `e_phi_lr > 0` (`e_step.py:1136`) and runs record `e_phi_lr: 0.0`, so the reachability bound was unreachable | guard demonstrably fired in ordinary K=300 eval (`min=-329.324`); `_restore_certified_self_links_` writes only the diagonal (`transport.py:1986-1990`) so attention pairs are never restored; `guard_energy_klmax_frac = 0.0000` so `kl_max` absorbs nothing | **DOWNGRADED high → medium, with an open obligation** | Both agree the defect is real, unguarded and monotone in conditioning. Neither established the operating point that decides severity: the skeptic's recorded metrics cover **K=20 only**; the defender's guard-firing evidence covers K=300 but never quantifies the silent band there. |
| D-01 | mm_exact not F-descent; test contract false | shipped prior is `causal_alibi_noself` not `causal` → 1/600 not 34/600; nothing consumes monotone F (`e_step.py:1503-1513` halts on belief *move*); `train.py:1041-1042` already ships `estep_f_nondecreasing_frac` as an expected metric | **Conceded.** Could not reproduce 34/600 or +45.585; measured **0/600** at shipped `mm_damping=0.75` and 0/600 from the anchored start | **DOWNGRADED high → low** | Row-0 behavior is not a defect: with only the self-key allowed `E_00 = 0` carries no information, so `q_0 = p_0` is the correct restricted minimizer, and the gradient route shares the mask (`kernels.py:477`). What survives is a test whose premise no registered prior satisfies. |

## Unresolved disagreement (recorded, not adjudicated)

**Does the batch axis close or widen the C-01 gap?** The defender measured at `B=8, N=64, d=6` and
concluded "the batch axis does not close the gap." The skeptic measured at `B=4, N=128, d=10` and
found the global `amin()` fires 12/12 at production size where a small probe fires 1/12, i.e. small
probes *understate* the guard. Different geometries, opposite conclusions, no basis in what either
produced to prefer one.

## Corrections to this report established by the challenge tier

- The A-01 reachability framing (`cond(U) <= exp(10)` from `retract_phi`) cites a guard that is
  **dead on the production path** (`e_phi_lr: 0.0`).
- D-01's headline figures (34/600, `+45.585`, `beta_ii` mean 0.4086) **did not reproduce**. The
  defender measured 8/600 under `causal` and 1/600 under the shipped prior at `eta=1.0`.
- `tests/pytest_policy.py:41` is a **CUDA device-parity cohort**, not an invariant registry.
  Describing `:50` as a pinned mathematical contract overstated it.
- A-02's RoPE evidence leg is a misdiagnosis for the flat base (see table).

# Surviving punch list

Ranked by decision impact. **Nothing here had been fixed at the time this report was written; the
audit itself modified no source.** Remediation since then is tracked in `docs/2026-07-26-edits.md`:
items 1-9 (B-01, A-01, B-02, B-03, E-01, C-04, C-05, C-06, D-02) and item 12
(E-02/E-03/E-04/E-05) are fixed on `fix/2026-07-26-audit-remediation`; item 10 (C-01/C-02/C-03) is
held on its stated open obligation, and item 11 (B-04) was investigated and deliberately not applied.

1. **[high] B-01 — `collect_estep_character` measures the wrong channel.** `run_artifacts.py:3251`,
   `:3279-3283`. `recorded[0]` is the s-channel fusion under `s_e_step=True`, and its `mu_p` is the
   frozen centroid `r` (`model.py:850-852`), not the belief's prior. Tag each record with its
   originating channel/layer and slice the belief window. **This invalidates published measurements
   — see the data-integrity note below.**
2. **[high] A-01 — `gaussian_frame_diagonal` silently discards non-coboundary transports.**
   `frame_gaussian.py:112`, `:130`. Gate both seams on `_certifies_same_frame_flat_cocycle` and
   raise as `exact_congruence.py:179` does, or reject the pairing in `__post_init__` via a
   family-declared capability. Also fix the provenance so an inert `regime_ii` is not recorded as
   active.
3. **[high] B-02 — beta probe ablates one of two live `attention_weights` bindings.**
   `run_artifacts.py:3392` vs `e_step.py:23`, `:799`. Patch `vfe3.free_energy.attention_weights` at
   the definition site.
4. **[high] B-03 — depth probe pins the model channel to `n_e_steps`.** `run_artifacts.py:3084`.
   One-line fix; triple-confirmed with a repro.
5. **[high] E-01 — non-causal `gamma_attention_prior` leaks future tokens.** `model.py:2364-2377`.
   Config-gated (the shipped gamma prior is causal), but unguarded. Reject in `__post_init__` when
   gamma's support is not a subset of beta's.
6. **[high] C-04 — `_direct_link_diagonal_covariance` returns negative variances.**
   `transport.py:2294-2315`. Give it the autocast island, escalation, and `clamp(min=0.0)` its two
   siblings have.
7. **[high] C-05 — dense `Omega` outside the fp32 island under AMP.** `transport.py:1620`, `:2176`.
   Breaks `Omega_ii = I` (3.9e-3) and therefore `E_ii = 0`.
8. **[high] C-06 — exact-congruence dense route discards its fp32 promotion.**
   `exact_congruence.py:164-167`. Drop the `.to(omega.dtype)` before the squaring.
9. **[high] D-02 — `mm_exact_update` bypasses the family seam and its own route predicate.**
   `kernels.py:598-628`. Live callers gate, so this is an unguarded public seam.
10. **[medium] C-01/C-02/C-03 — the congruence guard, three ways.** `transport.py:1269-1306`. One
    fix: gate escalation on a conditioning proxy computed once per build; drop the per-call `amin`
    sync. **Open obligation before acting:** instrument `cond(U)` on a K=300 run. If it stays under
    ~1e3 the whole cluster is cosmetic; the archived K=20 runs top out at 796.
11. **[medium] B-04 — the cheap diagnostic tier is gated under `generate_figures`**, so
    `scaling.py` emits none of it and `emit_expensive_diagnostics=True` is silently inert there.
12. **[medium] E-02/E-03/E-04/E-05 — config-validation and registry-seam gaps** around the two
    2026-07 families, including the invariance verifier that cannot cover either of them.

Deferred as low or pre-existing: A-03, A-04, B-05..B-09, C-07..C-12, D-03..D-09, E-06..E-18.

# Data-integrity consequence of B-01

Measurements published on 2026-07-25 for `s_e_step=True` checkpoints came from the defective probe.
Status after the duel:

| published quantity | status |
|---|---|
| pair-precision share 0.190 (K=20), 0.298 (K=300) | **INVALID** — s-channel, not belief channel. Both duel sides agree these "die under every reading." Repro: shipped 0.7468 vs belief-only 0.4810 on one forward. |
| "attention share rises with width" | **INVALID** — derived from the above. |
| displacement 0.147 / 0.227, `cos_dir_vs_step1` 0.982 / 0.962, step-1 share 73% / 70% | **UNCERTAIN, narrowed.** At the published `n_layers=1`, `recorded[-1]` *is* the belief's last iterate; only `first` is cross-channel. The skeptic's repro shows an r-anchored measure is flat/decreasing across depth with cos ~1.0, whereas belief-only *rises* — and the published rows show the belief-only signature (+36% growth, cos 0.982). So they are either approximately right (if `r` sits near `s1`) or internally inconsistent; they are **not** simply "the s-channel's." Requires re-measurement. |
| token-prior figures: share 0.109, displacement 0.521, cos 0.971 | **VALID.** `s_e_step=False`, `n_layers=1`, single channel; call sites traced directly on that checkpoint. |
| the registered prediction "share should rise above 0.30" being refuted | **STANDS** on its own terms (0.109 is not above 0.30), but the *explanation* offered for it — a fall from 0.190 — compared a belief-channel number against an s-channel number and does not survive. |

Downstream artifacts requiring correction: `docs/2026-07-25-state-of-knowledge.md` §6, and the
research-wiki notes `2026-07-25-estep-character-and-channel-decomposition`,
`2026-07-25-token-prior-estep-character-and-diagnostics`, plus the `Precision weighting` and
`VFE Transformer Program` pages.

**RESOLVED 2026-07-26 (partly).** The corrected probe was replayed on the same checkpoints under the
production protocol (first validation batch, eight sequences, depths 1/2/3/5/8); raw record in
`docs/2026-07-26-b01-remeasurement.json`, `recompute_max_abs_err = 0.0` on both.

- **K=300** (`55.41_wikitext-103_K300_...`) re-measured: pair share **0.213** (published 0.298),
  prior share **0.787**, displacement **0.297** at depth 1 and **0.317** at depth 8, step-1 share
  **94%** (published 70%), `cos` **+0.965**. The "UNCERTAIN" displacement row resolves here: the
  corrected trajectory is FLATTER across depth (+6.8% from depth 1 to 8) than the published +42%,
  so the E-step is more nearly one-shot, not less.
- **K=20 shares are UNDEFINED, not merely invalid.** The only surviving 2026-07-25 K=20
  `s_e_step=True` checkpoint runs `e_step_update='gradient'`, and the pair/prior split decomposes
  the `mm_exact` fusion that route never computes. No `estep_character.json` was ever persisted, so
  `0.190` cannot be traced to any artifact on disk. It is withdrawn, not corrected. Displacement
  **0.501 -> 0.571** and `cos` **+0.920** ARE recoverable and were measured.
- **"Attention share rises with width" cannot be repaired** — one endpoint is 0.213 and the other is
  gone. Re-establishing it needs a fresh K=20 `mm_exact` + `s_e_step=True` run.

The re-measurement also exposed a defect in the fixed probe itself: `collect_estep_character`
documented that off the `mm_exact` route "the displacement and cosine are still reported", but its
displacement window was populated only inside the `mm_exact` spy, so a gradient-route checkpoint
returned `None` for every field. `vfe_block` receives the belief prior and returns the converged
belief on every route, so the window is now route-independent, with the `mm_exact` window kept
primary where it exists (it reads `mu_star` before the block's optional norm). Two regressions pin it.

Separately, the variational expert established that
`docs/2026-07-25-shadow-prior-investigation.md`'s characterization of the layer-stack prior handoff
as "a degenerate one-token-per-agent shadow" is **wrong**: `mu_p_i` and `mu_q_i` live in the same
fiber over token `i`, so no transport belongs there.

