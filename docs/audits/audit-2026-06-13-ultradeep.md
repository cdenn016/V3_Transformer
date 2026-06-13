All findings verified against actual code. The line numbers, mechanisms, and code match the JSON claims. I have sufficient verification to synthesize the report.

# VFE_3.0 Deep Audit — Synthesized Findings Report (2026-06-13, second sweep)

## Executive Summary

This sweep adjudicated 25 surviving candidate findings against the working tree. After false-positive removal and severity adjudication, the verified counts by **effective severity** are:

| Severity | Count |
|---|---|
| Critical | 0 |
| High | 0 |
| Medium | 4 |
| Low | 20 |

**No critical or high finding survived.** The one candidate "high" (`s_e_step=True` + `gaussian_full` crash) was adjudicated down to **medium** with explicit decisive evidence: the s-channel is diagonal-by-design, a correct diagonal pure path exists and runs (loss 2.78, nonzero gradient to `s_mu_embed`), so the "no pure path" hook is false; what remains is a missing config guard that fail-loud crashes an *invalid* combination.

### Top issues that most deserve action

1. **(Medium) `s_e_step=True` + full-covariance family silently accepted, then hard-crashes** deep in a kernel with an opaque shape error (`vfe3/model/model.py:437`, `:514`; `vfe3/config.py:805-820`; crash at `vfe3/families/gaussian.py:230`). Real, reproduced, fixable by a one-line `__post_init__` guard or a `diag_embed`.
2. **(Medium) Diagonal/full-cov Rényi has a float32 catastrophic-cancellation band** just outside the KL switch (`vfe3/families/gaussian.py:99-104`, `:273-275`), returning a ~1% wrong divergence for any configured `alpha_div` in roughly `|alpha-1| ~ 1e-3..1e-4`. This is the only finding that silently corrupts a *value* on a normally-reachable, intended path.
3. **(Medium) Regime-II edge factor is the clamped surrogate (wrong operator) for non-orthonormal generator bases** (`so_n`/`sp_n`/`so_k`/`sp`) (`vfe3/geometry/transport.py:223-243`). The `delta_soft_cap` caps the *coordinate* norm, but the exactness guarantee needs the *Frobenius* norm of the embedded matrix; for the `so_n` tower (Gram diag up to 12) the embedded edge factor reaches `||delta·G||_F ≈ 41.6 > max_norm=15` at the cap, so `stable_matrix_exp_pair` silently rescales and autograd optimizes a clamped surrogate.
4. **(Medium) `gaussian_full` + non-compact `block_glk` full-covariance sandwich loses all fp32 digits** at the default retraction `max_norm=5`, with no guard (`vfe3/geometry/transport.py:390-397`, `:547-549`). A valid toggle combination is numerically broken at the documented default while none of the docstring's listed mitigations are applied.

### Verdict on the owner's primary concern — does a theoretically-pure path EXIST?

For each major theoretical structure, the status of the *pure* (mathematically correct) path under appropriate toggles:

| Structure | Pure path status | Note |
|---|---|---|
| Free-energy assembly (belief channel, self/coupling) | **CONFIRMED PRESENT** | `sum`-reduced canonical blocks; envelope identity verified to ~5e-7. |
| Attention-entropy term (`tau·beta·log(beta/pi)`) | **CONFIRMED PRESENT** | `include_attention_entropy=True` gives the exact envelope phi-gradient via `reduced_free_energy`; softmax stationarity sympy-verified independent of `lambda_beta`. The surrogate branch is the documented opt-in. |
| Gauge equivariance (flat `block_glk`, CG coupling, Mahalanobis full-cov) | **CONFIRMED PRESENT** | Flat transport exact for every group; CG coupling exact to ~2e-7; full-cov Mahalanobis gauge-invariant to ~1.5e-9. Diagonal Mahalanobis is subgroup-invariant only, matching its declared admissibility. |
| SPD / Riemannian geometry (retraction, sandwich) | **CONFIRMED PRESENT** but **NUMERICALLY DEGRADED on one valid combo** | Affine retraction matches exact `exp_Sigma`; compact `so_n`/`sp_n` and `gaussian_diagonal` are well-conditioned. The `gaussian_full` + `block_glk` sandwich is fp32-broken at the default `max_norm=5` (Medium #4). |
| Natural gradient / Fisher | **CONFIRMED PRESENT** | Diagonal `nat_mu=sigma·grad_mu`, `nat_sigma=2 sigma^2·grad_sigma` are the exact metric inverse. Killing preconditioner is opt-in; `mode='none'` is the canonical pure path and is active. |
| KL / f-divergence | **CONFIRMED PRESENT** but **NUMERICALLY DEGRADED in a narrow alpha band** | KL and Rényi far from 1 are exact to ~5e-7; new f-divergences reuse the pinned Rényi. The cancellation band (Medium #2) is a narrow defect of the otherwise-pure path. |
| Hyper-prior (`lambda_h·KL(s||r)`, gamma block) | **CONFIRMED PRESENT (diagonal)** | Scored exactly under `s_e_step=False`. The **full-covariance** s-channel does NOT exist (Medium #1) but the s/r tables are diagonal by construction, so there is no full-cov s object to approximate — the appropriate pure toggle is the diagonal family, which is correct. |
| E-step fixed point | **CONFIRMED PRESENT** | Kernel == filtering oracle to ~2e-7; EM separation and no-label-leakage verified. |

No pure path is **MISSING** or **BROKEN** in the strict sense. Two pure paths are **numerically degraded** on specific valid toggle combinations (Medium #2 alpha band, Medium #4 full-cov sandwich), and one toggle combination (full-cov s-channel) is unsupported-and-uncaught rather than wrong.

---

## Critical

None.

## High

None. The single candidate (`s_e_step` + `gaussian_full`) was adjudicated to **medium** — see below.

---

## Medium

### M1. `s_e_step=True` + full-covariance family is silently accepted, then hard-crashes in the E-step

**Location:** `vfe3/model/model.py:437` (hardcoded `family="gaussian_diagonal"` in `_refine_s`), `vfe3/model/model.py:514` (belief sigma overwritten with diagonal `(B,N,K)`), `vfe3/model/prior_bank.py:219-221` (`encode_s` returns diagonal sigma unconditionally), `vfe3/config.py:805-820` (guard checks `prior_source`/`lambda_h`/`gamma_coupling`, never `family`). Crash surfaces at `vfe3/families/gaussian.py:230`.

**Severity:** Medium — **adjudication verdict: severity_downgraded (from High)**. The crash and missing guard are real and reproduced; the "no pure path" framing is false because the s/r tables are diagonal by construction.

**Math/theory at stake:** The hyper-prior term `lambda_h·KL(s_i||r)` and the gamma model-coupling block. Under `gaussian_full`/`diagonal_covariance=False` the belief sigma is `(B,N,K,K)`, but `_refine_s` produces a diagonal `(B,N,K)` and overwrites the full belief covariance with it (`model.py:514`), which the full vfe_stack then cannot consume.

**Evidence (verified):** `model.py:437` reads `family="gaussian_diagonal"` (hardcoded, ignores `cfg.family`); `model.py:644`/`:654` confirm the s-channel terms are gated on `not self.cfg.s_e_step`. The adjudicator built the exact config (`family='gaussian_full', diagonal_covariance=False, s_e_step=True, prior_source='model_channel', lambda_h=0.5`), `__post_init__` accepted it, and the forward crashed with `RuntimeError: size of tensor a (6) must match b (8)` at `gaussian.py:230`. Decisive counter-evidence to the "no pure path" claim: `prior_bank.py:179-180`/`:191-192` show the s table is `(V,K)` and r is `(K,)` — diagonal by construction; a source-wide search found no `(V,K,K)` s-table; and the diagonal pure path (`family='gaussian_diagonal', s_e_step=True`) runs to a finite loss with nonzero gradient to `s_mu_embed`.

**Fix (minimal):** In `config.__post_init__`, reject `s_e_step=True and not family_is_diagonal` with a clear message; OR thread `family=cfg.family` into `_refine_s` and `torch.diag_embed` the encoded variances when `diagonal_covariance=False` (mirroring `prior_bank.py:475`, which already diag-embeds on the `encode()` path).

---

### M2. Diagonal/full-cov Rényi closed form has a float32 catastrophic-cancellation band just outside the KL switch

**Location:** `vfe3/families/gaussian.py:99-104` (diagonal) and the full-cov twin `gaussian.py:273-275`. KL-switch threshold at `gaussian.py:82` (`abs(alpha-1.0) < 1e-6`).

**Severity:** Medium. (Lenses: info-geometer; consistent with the prior 2026-06-10 `renyi_per_coord` saturation note 17726.)

**Math/theory at stake:** For `alpha` just outside `1e-6` the three log terms in `logdet_per_dim = (1-alpha)log(sigma_q)+alpha·log(sigma_t)-log(sigma_blend)` nearly cancel and are then divided by a tiny `(alpha-1)`, amplifying float32 round-off. The `1e-6` switch is right for float64 but the float32 cancellation band extends to roughly `|alpha-1| ~ 1e-3..1e-4`.

**Evidence (verified):** `gaussian.py:104` is the literal `logdet_per_dim.sum(-1) / (alpha - 1.0)` quotient inside the `else` (non-KL) branch entered for any `alpha` failing the `:82` threshold. Reported probe (seed 8, K=4): `KL(alpha=1)=6.793636` vs `renyi(alpha=1.000002)=6.725648` — a -0.068 (≈1%) non-monotone deviation. `alpha_div` is a user config value routed verbatim to this kernel.

**Fix (minimal):** Widen the KL-limit detection (e.g. `abs(alpha-1.0) < 1e-3`), or for alpha inside the band evaluate the divergence by a first-order series expansion in `(alpha-1)` around the KL value instead of the literal quotient.

---

### M3. Regime-II edge factor is the clamped surrogate (wrong operator) for non-orthonormal generator bases

**Location:** `vfe3/geometry/transport.py:223-230` (`delta_soft_cap=12.0` coordinate-norm cap), `:232` (matrix embed), `:240-243` (exp call), `:254`/`:265-270` (`stable_matrix_exp_pair max_norm=15`, "SAFEGUARD, NOT THE EXACT OPERATOR").

**Severity:** Medium. (Lens: geometer; corroborated by the gauge lens's holonomy-observability finding M-adjacent L below, and prior observation 18501.)

**Math/theory at stake:** The docstring exactness guarantee (`transport.py:184-188`) holds only when `||delta·G||_F = ||delta||_2`, i.e. orthoNORMAL bases (Gram=I, `glk`/`block_glk`). For orthogonal-but-not-orthonormal towers (`so_n` l1+l2 Gram diag up to 12, `so_k`=2, `sp`=1..2) the soft cap on the coordinate norm does not bound the embedded matrix norm: at the cap `||delta||_2=12` the `so_n` embedded factor reaches `||delta·G||_F = sqrt(12)·12 ≈ 41.6 > 15`. `stable_matrix_exp_pair` then rescales (no-grad) and returns `exp(15·M/||M||_F)`, NOT `exp(delta·G)` — wrong singular values/det, and autograd optimizes through the clamp the soft cap was added to prevent.

**Evidence (verified):** `transport.py:229-230` is the coordinate-norm cap `delta * rsqrt(1 + sq/cap^2)` with `sq = delta.pow(2).sum(-1)` (no Gram weighting); `:232` embeds via `einsum('bija,akl->bijkl', delta, generators)`; `:240` calls `stable_matrix_exp_pair(..., max_norm=15 default)`; `:265-270` documents the clamp is non-exact. Reachable under `transport_mode='regime_ii'` + `gauge_group in {so_n, sp_n, so_k, sp}` once any edge delta approaches the cap.

**Fix (minimal):** Cap delta in the matrix-Frobenius norm: scale by `rsqrt(1 + (delta^T Gram delta)/cap^2)` using the group's underlying Gram, OR set `delta_soft_cap < max_norm / sqrt(max(diag Gram))` per group so the embedded factor provably stays below the hard clamp.

**Pure-path note:** The flat default is unaffected and exact for every group (worst-case flat `||phi·G||_F ≈ 10.9 < 15` even for the `so_n` tower). This is a defect only on the opt-in regime_ii NN-exception path.

---

### M4. `gaussian_full` + non-compact `block_glk` full-covariance sandwich loses all fp32 digits at the default `max_norm=5`

**Location:** `vfe3/geometry/transport.py:390-397` (docstring states `cond(Omega) ~ exp(2||phi||)` and "No guard is imposed here"), `:547-549` (full-cov sandwich `einsum('...ijkl,...jlm,...ijnm->...ijkn', omega, sigma, omega)`).

**Severity:** Medium. (Lens: numerical.)

**Math/theory at stake:** The congruence sandwich `Omega Sigma Omega^T` squares the conditioning of `Omega`. At `d_head=20`, `||phi||_F=5` (the retraction `max_norm`) with a symmetric eigenvalue split, `cond(exp(M)) = exp(2a) ≈ 1177`, `cond(Omega)` up to ~1.4e6, and the sandwich squares to ~1e12. At fp32 (eps~1.2e-7) any sandwich conditioned beyond ~1e7 loses every significant digit.

**Evidence (verified):** `transport.py:390-397` is the conditioning docstring with the explicit "No guard is imposed here"; `:547-549` is the full-cov sandwich einsum. Reachable under `family=gaussian_full`, `gauge_group=block_glk`, `transport_mode=flat` — all valid toggles — with no runtime guard.

**Fix (minimal):** Lower the default GL `max_norm` when `family=gaussian_full` (≈2.5 keeps the squared sandwich under ~1e4), OR compute the full-cov sandwich in float64, OR emit the existing `condition_number` monitor on the sandwich for full-cov + non-compact.

**Pure-path note:** Well-conditioned pure paths exist (compact `so_n`/`sp_n` give orthogonal Omega cond=1; `gaussian_diagonal` squares conditioning only once). The gap is the full-cov + GL combo being broken at the documented default.

---

## Low

These are latent (off any live path), documentation/observability gaps, or robustness margins on inactive pure paths. None corrupts a value on a live, intended path. Grouped by theme.

### Documentation / overclaim (code correct, docstring wrong)

**L1. `MahalanobisNorm` docstring claims GL(K) invariance the diagonal branch does not provide.** `vfe3/geometry/norms.py:31-37` (docstring), `:56-57` (diagonal `s2 = sum(mu^2/sigma)`). Full-cov branch (`:62-65`) is gauge-invariant to ~1.5e-9; diagonal is invariant only under the diagonal subgroup (counter-example 8.84→10.56 under non-diagonal g). Matches the family's own `check_admissible` declaring `gaussian_diagonal` non-invariant. *Fix:* qualify the docstring. (Lens: geometer.)

**L2. `pos_phi_project_slk` natural-grad gate comments are false / dead branch.** `vfe3/train.py:99-102`, `vfe3/model/positional_phi.py:133-134`, `vfe3/geometry/lie_ops.py:427-448`. `pos_phi_free` is always created at full width `(max_seq_len, n_gen)` (`model.py:263-264`) and `project_phi_to_slk` preserves width, so the documented "AdamW fallback for the reduced chart" branch is dead — under `m_phi_natural_grad=True` the slk-projected table is always natural-grad-stepped. *Fix:* delete the misleading shape-gate comments, or gate on `cfg.pos_phi_project_slk` if AdamW fallback is actually intended. (Lens: impl.)

**L3. `train.py` weight-decay comment overstates effect on dead belief tables.** `vfe3/train.py:139-144` with `vfe3/model/prior_bank.py:231,236`. Under `prior_source='model_channel'`, `mu_embed`/`sigma_log_embed` are never read (grad=None confirmed); AdamW skips None-grad params entirely, so the "shared weight_decay still decays the dead table harmlessly" comment is wrong (no decay fires). Net effect benign. *Fix:* correct the comment, or skip grouping the rerouted tables. (Lens: impl.)

### Coherence / equivariance bookkeeping (guard missing, pure path present)

**L4. RopeTransport means-only path produces a non-congruence belief with no guard.** `vfe3/geometry/transport.py:52-71, 528-534`; mean uses `R_i Omega R_j^T` (`:492-497`) while covariance under means-only uses bare base Omega. A downstream Mahalanobis/affine-invariant consumer then reads a gauge-inconsistent quantity; incoherence lives only in a docstring. Coherent path (`rope_on_cov=True`) exists. *Fix:* in `config.__post_init__`, require `rope_on_cov=True` when rope is active and a Mahalanobis-style `block_norm` is wired. (Lens: gauge.)

**L5. `compose_bch` closure diagnostic is size-gated off for exactly the bases that can be non-closed.** `vfe3/geometry/lie_ops.py:306-313`; `extract_phi` (`:304`) silently projects out the out-of-span BCH component. The 2026-06-13 edit size-gates `warn_if_basis_not_closed`, so a large non-closed cross_coupled `block_glk` (`close_basis=False`) skips the scan and the truncation is silent. The closed pure path (`close_basis=True` → `close_under_brackets`, residual 0.0) exists but is opt-in/default-off. *Fix:* gate the scan on a structural test (is the basis a pure block-diagonal direct sum?) rather than element count, or default `close_basis=True` when `cross_couplings` is set. (Lens: gauge.)

**L6. `check_admissible` verifier is never invoked at construction.** `vfe3/geometry/groups.py:198-274` (`check_admissible`), `:79-81` (`invariant_for`). Only consumers are tests. The executable congruence check works (`block_glk`/`gaussian`=True, `gaussian_diagonal`=False) but is wired to no model-build/config-validation site. Declared invariants are in fact correct for shipped groups. *Fix:* call `check_admissible(group, cfg.family)` once under a default-on validation toggle. (Lens: gauge.)

### Numerical robustness on inactive / latent paths

**L7. `renyi()` emits the `alpha>1` non-PD-blend warning even when the closed form takes the KL branch.** `vfe3/families/base.py:234-238`. For `alpha in (1.0, 1.0+1e-6)` the closed form uses plain KL (no blend) yet the caller still gets the RuntimeWarning. Cosmetic; no value affected. *Fix:* gate the warning on `alpha > 1.0 + 1e-6`. (Lens: info-geometer.)

**L8. Generic Bregman/Rényi-from-A path ignores eps and uses raising cholesky.** `vfe3/families/base.py:171-211` (`_renyi_from_log_partition`, eps "intentionally unused"), full-cov `log_partition_at` at `gaussian.py:195-201` (bare `torch.linalg.cholesky`). Both families register `renyi_closed_form`, so this generic path is dead in production (reached only by direct/test calls; agrees with closed form to <5e-7 well-conditioned). *Fix:* route the generic full-cov `log_partition_at` through `safe_cholesky`, or document it as a well-conditioned-only pinning oracle. (Lens: info-geometer.)

**L9. `FullGaussian.entropy()` discards `safe_cholesky`'s ok mask, returns finite-but-wrong on non-PD Sigma.** `vfe3/families/gaussian.py:207-210`; ok mask bound to `_`. Probe: `entropy()` of `-5*I` returns -104.85 (finite) while ok=False. Zero callers in `vfe3/` today (latent), but part of the public `BeliefParams` interface. *Fix:* `L, ok = safe_cholesky(...)` then `torch.where(ok, ..., NaN)`. (Lens: numerical.)

**L10. `FullGaussian.log_partition_at` uses bare `torch.linalg.cholesky` that raises on documented-reachable ill-conditioning.** `vfe3/families/gaussian.py:195-201`; `natural()` documents cond up to ~5e6. Dead today (only consumer `_renyi_from_log_partition` is dead). *Fix:* route through `safe_cholesky` and mask to NaN. (Lens: numerical.) — Note L8/L10 are the same generic-A-path robustness issue from two angles.

**L11. `_eigh_damped` backward biases gradients up to 50% on eigenvalue gaps near the variance floor.** `vfe3/geometry/retraction.py:71-72, 81-86`; `gap_eps=1e-8` is a fixed constant untied to the `[eps=1e-6, sigma_max=5]` spectrum (`:148,168`). Probe: gap 1e-4 → 50% rel error. Exact stock `eigh` backward is 100% NaN on the `Sigma=I` default init, so the damped form is the only finite path with no toggle back. *Fix:* scale `gap_eps` to the spectrum (e.g. `(rel_tol·max|lambda|)^2`) or document the accuracy floor and pin a small-gap test. (Lens: numerical.)

**L12. `build_killing_preconditioner` lifts any sub-tol eigenvalue to `center_reg`.** `vfe3/geometry/phi_preconditioner.py:130-132`; magnitude-only test (`tol=1e-6`) replaces any true Killing eigenvalue below 1e-6 with `reg=2K`, silently corrupting the natural-gradient component along it. Inert in active config (`mode='none'` is the pure path, `e_phi_lr=0`). *Fix:* lift only directions identified as the center (project against per-block trace functionals), or assert the lifted-direction count equals the expected nullspace dimension. (Lens: numerical.)

**L13. `safe_spd_inverse` / `condition_number` monitor raise on a diagonal `(...,K)` input.** `vfe3/numerics.py:96-119, 135-142, 207-210`; both assume `(...,K,K)`. The active family is `gaussian_diagonal`; the registered monitor would raise instead of returning `max(sigma)/min(sigma)`. Monitors are opt-in, never on the hot path. *Fix:* add a rank-keyed diagonal branch, or document as full-cov only. (Lens: numerical.)

**L14. `stable_matrix_exp_pair` clamp has no opt-in activation monitor on the regime_ii edge path.** `vfe3/geometry/transport.py:250-299` (clamp), `:240-243` (regime_ii caller), `:269-270` ("per-call runtime monitor intentionally omitted"). A raised `delta_soft_cap` or a non-orthonormal basis (see M3) silently enters the clamped surrogate with no observability. *Fix:* add a default-off, non-syncing running-flag clamp-activation counter reduced once per epoch. (Lens: gauge.) — Same root mechanism as M3; this is the observability half.

### Metrics mirror (latent, no production caller)

**L15. Registry `effective_rank` metric mishandles full covariance.** `vfe3/metrics.py:890-893`; `effective_rank(sigma)` treats matrix rows as a spectrum and clamps negative off-diagonals to 0. Probe: registry 1.959 vs correct eigenvalue-based 2.498. Sidestepped by `model.diagnostics` (inline `eigvalsh`); no production caller. *Fix:* `return float(effective_rank(_spectrum(sigma)).mean())`. (Lens: transformer.) Confirmed at `metrics.py:891-893`.

**L16. Registry `free_energy_terms` metric defaults `tau=1.0`, dropping the group-aware temperature.** `vfe3/metrics.py:919-923`; the wrapper has no way to know `d_head`, so its `total` is correct only at `tau=1` (kappa=1,K=1). Live diagnostics path (`model.py:915`) passes the real `attention_tau`. No production caller. *Fix:* require `tau` in the wrapper (no default), or document that callers must pass `attention_tau(...)`. (Lens: transformer.) Confirmed at `metrics.py:920-921` (`tau=1.0` default).

### Free-energy convention / transparency (correct up to scale or by-design)

**L17. Hyper-prior `KL(s||r)` and gamma blocks are dropped from the scored loss under `s_e_step=True`.** `vfe3/model/model.py:644,654` (both gated `and not self.cfg.s_e_step`). Under `s_e_step=True` the canonical F terms enter only as an E-step descent direction, never scored; s-tables get M-step gradient only through refined-s → belief → CE. This is internally consistent EM (no double-count), but the assembled objective is not literally `F = ... + lambda_h·KL(s||r) + gamma-block`. Scored pure path exists via `s_e_step=False`. *Fix:* document/assert this is intended, or add the detached block values to logged diagnostics. (Lens: variational.) Confirmed `model.py:644,654`.

**L18. Hyper-prior/gamma blocks use `mean()` while the belief channel uses `sum()`.** `vfe3/model/model.py:699,746,750` (`.mean()`) vs `vfe3/free_energy.py:367,369,381` (`.sum()`). The s-channel blocks carry a `1/(B·N)` relative scale vs the sum-reduced belief block, so `lambda_h`/`gamma_coupling` mean something different relative to the belief block than the canonical sum-of-F_i suggests. Correct up to an overall scale. *Fix:* pick one reduction convention and document the calibration. (Lens: variational.)

**L19. phi step uses the entropy-suppressed surrogate gradient when `include_attention_entropy=False`.** `vfe3/inference/e_step.py:304-307`. The surrogate also differentiates through beta, so its phi-gradient differs from the canonical envelope by the documented `-tau^-1 Cov_beta(KL, grad KL)`. **This is the documented surrogate; the pure path (`include_attention_entropy=True`) exists and is verified correct** (envelope identity to ≤5e-7, softmax stationarity sympy-verified). *Fix:* none required; optionally note the surrogate's non-envelope gradient in the branch. (Lens: variational.)

### Clebsch-Gordan / positional (robustness margin, by-design)

**L20. CGCoupling antisymmetric-self-pair prune uses a fixed 1e-10 threshold untied to the CG solve's atol-scaled gate.** `vfe3/model/cg_coupling.py:82-85` vs `cg.py:125` (`verify_gate = max(1e-7, 10*atol)`). A slot antisymmetric only to ~1e-9 (thin-gap/loosened-atol solve) would fail the 1e-10 prune and be kept as a near-dead live path. CG coupling itself is the exact equivariant pure path (~2e-7). *Fix:* scale the prune threshold to atol (`max(1e-10, 10*atol)`). (Lens: gauge.)

**L21. gauge-RoPE leaves the odd-block leftover coordinate un-rotated.** `vfe3/geometry/rope.py:69-77`; an odd-dim block's last coordinate keeps identity (zero positional content). This is standard RoPE convention and `config.py:907` already rejects `pos_rotation='rope'` for `so_n`/`sp_n`; unreachable for those groups, and `block_glk` `d_head` is even at the active config. *Fix:* none for correctness; document in the rope docstring that odd-dim blocks carry one positionally-inert channel so a future un-gating does not assume full coverage. (Lens: transformer.)

---

## Prioritized Fix Punch-List

### Correctness / purity (do first)

1. **M1 — `s_e_step` + full family guard** (`config.py:805-820`): reject `s_e_step=True and not family_is_diagonal`, OR `diag_embed` the s-sigma in `_refine_s` when `diagonal_covariance=False`. Converts an opaque deep-kernel crash into a clear fail-fast (or a working diagonal-embedded full path).
2. **M2 — Rényi cancellation band** (`gaussian.py:99-104`, `:273-275`): widen the KL switch to `~1e-3` or series-expand in `(alpha-1)`. The only finding that silently corrupts a *value* on a normally-reachable path.
3. **M3 — Regime-II Frobenius cap** (`transport.py:223-230`): cap delta in the matrix-Frobenius norm via the group Gram, or set `delta_soft_cap` per-group below `max_norm/sqrt(max diag Gram)`. Restores exact-operator regime_ii for `so_n`/`sp_n`/`so_k`/`sp`.
4. **M4 — full-cov sandwich conditioning** (`transport.py:390-397`, `:547-549`): lower GL `max_norm` (≈2.5) when `family=gaussian_full`, or float64 the sandwich, or emit the condition-number monitor for full-cov + non-compact.
5. **L4/L5/L6 — equivariance/admissibility guards**: require `rope_on_cov=True` with Mahalanobis (L4); structural (not size) closure-scan gate or default `close_basis=True` under cross_couplings (L5); invoke `check_admissible` at build under a default-on toggle (L6).
6. **L1/L2/L3 — fix overclaiming docstrings/comments** (Mahalanobis diagonal invariance; pos_phi slk gate; AdamW weight-decay-on-None). Per the audit mandate that comments must not lie.

### Robustness / perf (do next)

7. **L9/L10/L8 — `safe_cholesky` everywhere in the family layer**: bind the ok mask in `FullGaussian.entropy()` (L9); route `log_partition_at` and the generic A-path through `safe_cholesky` (L8/L10). Hardens latent public-interface and generic-family paths.
8. **L11/L12 — eigen-adjoint and Killing conditioning**: scale `_eigh_damped` `gap_eps` to the spectrum and pin a small-gap test (L11); lift only true center directions in the Killing preconditioner (L12).
9. **L14 — clamp-activation observability** (paired with M3): default-off non-syncing counter so a silently-active Frobenius clamp on the transport/edge exp is detectable.
10. **L13 — diagonal branch in the SPD monitors** so the family-agnostic health probes work on `gaussian_diagonal`.
11. **L15/L16 — metrics-registry parity**: route `effective_rank` through `_spectrum`; require `tau` in `free_energy_terms`. Latent (no production caller) but should mirror the live diagnostics.
12. **L7/L20/L21 — cosmetic/threshold**: gate the `alpha>1` warning on `>1+1e-6` (L7); tie the CG antisymmetry prune to atol (L20); document the RoPE odd-block inert channel (L21).

### Transparency (assert intent; no code-behavior change required)

13. **L17/L18 — free-energy convention**: assert/document that under `s_e_step=True` the hyper-prior and gamma blocks are realized as an E-step descent direction (not scored terms), and unify the `mean()`/`sum()` reduction across F blocks (or document the `lambda_h`/`gamma_coupling` calibration).
14. **L19 — no action**: the entropy-suppressed phi-gradient surrogate is the documented opt-in; the pure path is verified correct.

Relevant files: `C:\Users\chris and christine\Desktop\V3_Transformer\vfe3\model\model.py`, `...\vfe3\families\gaussian.py`, `...\vfe3\geometry\transport.py`, `...\vfe3\geometry\retraction.py`, `...\vfe3\geometry\phi_preconditioner.py`, `...\vfe3\geometry\norms.py`, `...\vfe3\geometry\lie_ops.py`, `...\vfe3\geometry\rope.py`, `...\vfe3\families\base.py`, `...\vfe3\numerics.py`, `...\vfe3\metrics.py`, `...\vfe3\config.py`, `...\vfe3\model\prior_bank.py`, `...\vfe3\model\cg_coupling.py`, `...\vfe3\inference\e_step.py`, `...\vfe3\train.py`, `...\vfe3\free_energy.py`.