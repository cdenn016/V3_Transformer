# VFE_3.0 Overnight Deep Audit — 2026-06-14

Branch: `audit/overnight-deep-audit-2026-06-14`. Report-only (no source changed).

## Executive summary

A six-lens parallel expert audit (differential geometry, gauge theory, information
geometry, variational free energy / inference, model + config wiring, numerics + runtime)
swept the entire `vfe3/` package end-to-end, tracing actual control/data flow and verifying
every substantive claim numerically (CPU torch, `import vfe3`) and, where applicable,
symbolically (sympy). Each high-severity candidate was adversarially re-checked against the
running code before inclusion; several agent-reported items were **downgraded or corrected
during adjudication** (noted inline).

**Verdict: the codebase is in strong shape.** No critical or high finding survived
adjudication. The theoretically pure, no-NN, gauge-equivariant, canonical-free-energy paths
**all exist, are reachable under their intended toggles, and were verified correct** (envelope
identity to ~3e-7, softmax-beta stationarity to ~2e-7, hand-kernel == filtering oracle to
~1e-7, gauge equivariance exact for the structure-group images, SPD retraction matches the
exact exponential). The findings below are documentation-honesty mismatches, latent
robustness gaps off the live path, and interpretive items. This is consistent with the
codebase having already absorbed multiple prior audit cycles (the 2026-06-13 ultradeep
mediums — Rényi cancellation band, regime-II Frobenius cap, full-cov sandwich float64 island
— are already fixed in this tree).

### Verified baseline test numbers

Quoted from the machine-readable run (`/tmp/baseline.xml`, full suite, no extra `-q`):

```
testsuite name="pytest" errors="0" failures="3" skipped="9" tests="964"
```

`tests=964 failures=3 errors=0 skipped=9` (plus `1 xpassed`). **All 3 failures are an
environment-only missing dependency**, not code defects: `tests/test_viz.py` imports
`sklearn` (`ModuleNotFoundError: No module named 'sklearn'`). After `pip install
scikit-learn`, that file reports `27 passed, 3 skipped`. With the dependency present the
suite is fully green (≈954 passed effective, 9 skipped, 1 xpassed, 0 real failures). Torch,
numpy, matplotlib, and sklearn were not preinstalled in this container and had to be added to
run the suite (CPU-only; `torch.cuda.is_available()` is False here).

### Findings by severity

| Severity | Count |
|---|---|
| Critical | 0 |
| High | 0 |
| Medium | 1 |
| Low | 11 |
| Informational / verified-correct | (catalogued below) |

### Top items deserving attention

1. **(Medium) Decode boundary is silently fixed at α=1 KL while the class docstring claims it
   tracks the configured divergence family / α** (`vfe3/model/prior_bank.py:16-24`). The
   firmest violation of the CLAUDE.md "comments must not lie" mandate, with a real (if
   interpretive) objective inconsistency on a reachable non-default config.
2. **(Low) `model._amp_context` comment states fp16 is rejected at construction — it is
   accepted and the fp16 training path is live** (`vfe3/model/model.py:399-401`).
3. **(Low) The generic Rényi-from-A path lacks the fp32 cancellation guard the Gaussian
   closed forms have** (`vfe3/families/base.py:244-248`) — unreachable for shipped families,
   but it is the would-be *pure* path for a future custom exponential family.
4. **(Low) `squared_hellinger` saturates to exactly `1.0` in float32 at the default
   `kl_max=100`, contradicting its `[0,1)` docstring** (`vfe3/families/base.py`).

### Pure-path existence verdict (the owner's primary concern)

| Structure | Pure path | Status |
|---|---|---|
| Canonical free energy (self + belief-coupling + attention entropy) | `include_attention_entropy=True`, `lambda_beta=1` | **PRESENT & CORRECT** (envelope identity ~3e-7; softmax stationarity ~2e-7) |
| Hyper-prior `λ_h KL(s‖r)` + γ model-coupling blocks | `lambda_h>0` / `gamma_coupling>0` | **PRESENT & CORRECT** (canonical envelope form; split across `free_energy.py` and `model.py`) |
| Gauge equivariance | `pos_rotation='none'`, no head mixer, no CG, flat transport | **PRESENT & CORRECT** (exact for every shipped group image) |
| SPD / Riemannian retraction + sandwich | `spd_affine` / `log_euclidean` | **PRESENT & CORRECT** (matches exact `exp_Σ`) |
| Natural gradient / Fisher | `phi_precond_mode='none'` | **PRESENT & CORRECT** (diagonal metric inverse exact) |
| KL / f-divergence | `divergence_family='renyi'`, `alpha_div=1` | **PRESENT & CORRECT** (closed forms ~5e-7; band-guarded near α=1) |
| Full-covariance belief | `family='gaussian_full'` | **PRESENT & TRAINS** (prior tables receive nonzero gradient; see L3 re the unroll-estimator downgrade) |
| E-step fixed point | filtering kernel / oracle | **PRESENT & CORRECT** (kernel == oracle ~1e-7; EM separation, no label leakage) |

No pure path is missing, broken, or unreachable. One pure path (full-cov) silently uses a
straight-through-equivalent gradient instead of the requested full unroll unless
`oracle_unroll_grad=True`, but it **does train** and the downgrade is warned at runtime (L3).

---

## Medium

### M1. Decode boundary is hardcoded to α=1 KL on the Gaussian family, while the docstring claims it tracks the configured divergence family / α

**Location:** `vfe3/model/prior_bank.py:16-24` (class docstring), `:297-326` (`reference_decode`),
`:709-730` (`_decode_full`), and the fused `_decode_diagonal` kernel.

**Severity:** Medium. The docstring-vs-code mismatch is definitive (the firm part); the
behavioral question of whether decode *should* honor the coupling divergence is interpretive.

**Evidence (quoted):** The class docstring asserts the decode tracks the configured seam:

```
Divergence-agnostic, scope clarified: ``reference_decode`` is the literal seam path --
it calls ``divergence.kl`` and so tracks whatever divergence family/alpha the seam is
configured for.
...
and ``reference_decode`` already covers any registered divergence for verification.
```

But the code calls `kl` (which `divergence.py:3` documents as "Renyi alpha-divergence is the
primitive; KL is its alpha = 1 special case" — i.e. a fixed-α=1 functional) on a hardcoded
diagonal family:

```python
# prior_bank.py:324-325
diag = get_family("gaussian_diagonal")
kl_v = kl(diag(mu_q_b, sigma_q_b), diag(mu_v, sigma_v), kl_max=float("inf"))
```

`_decode_full` does the same with `full(...)` (`:730`). Neither reads `cfg.alpha_div` nor
`cfg.divergence_family`. `kl` is **not** a family/α-tracking seam — there is no α or
`divergence_family` argument to `kl()` at all.

**Why it is wrong:** `alpha_div` and `divergence_family` are advertised as live global seams
(`config.py` docstrings). The E-step honors them (an `alpha_div=1.5` run emits the
non-convex-regime Rényi warning from the coupling kernel), but `decode()` always emits the
α=1 KL readout regardless. The docstring's "tracks whatever divergence family/alpha" and
"covers any registered divergence" are false: `reference_decode` only ever computes α=1 KL on
`gaussian_diagonal`. A model configured with `divergence_family='squared_hellinger'` or
`alpha_div≠1` and `use_prior_bank=True` trains its inference under one divergence and reads
out logits under a different one, silently. (The model.py class docstring at `:18` half-
discloses the *kernel* is "a hand-specialized alpha=1 gaussian_diagonal shortcut," which makes
the `reference_decode` claim at `:16-17` the misleading one.)

**Suggested fix:** Either (a) thread `alpha=cfg.alpha_div, divergence_family=...` through
`reference_decode` via a `divergence_for_alpha`-style seam call and guard the fused kernels to
reject/fall-back when `alpha_div≠1`; or (b) correct the class docstring to state the decode is
fixed at α=1 KL and add a `__post_init__` warning when
`use_prior_bank=True and (alpha_div≠1 or divergence_family≠'renyi')`, so the
decode/E-step divergence mismatch is not silent.

---

## Low

### Documentation / comment honesty (code correct, comment wrong)

**L1. `model._amp_context` comment claims fp16 is rejected at construction; it is accepted and
the fp16 training path is live.** `vfe3/model/model.py:399-401`:

```python
# config.py rejects amp_dtype='fp16' at construction (deferred: needs a GradScaler), so 'bf16' is the only
# reachable non-None value, but map both and raise on anything else.
```

Contradicted by `vfe3/config.py:1330` (`_require(self.amp_dtype, (None, "bf16", "fp16"), "amp_dtype")` —
fp16 accepted) and `vfe3/train.py:566` (`torch.amp.GradScaler(device=..., enabled=(cfg.amp_dtype == "fp16"))`
— the fp16 GradScaler path is fully wired and pinned by `tests/test_amp.py`). The code in
`_amp_context` is correct (it maps fp16 too); only the comment lies. *Fix:* correct the comment.

**L2. RoPE-as-gauge framing is exact only for single-block groups.** `vfe3/geometry/rope.py:1-9`,
`:57-80`; `vfe3/geometry/transport.py:53-66`. The docstring frames the rotary rotation as a gauge
element of the structure group (`U_i = R(θ_i) exp(φ_i)`). For `glk`/`so_k`/`l1`-of-`so(3)` blocks
`R` lies inside the group image and the product is literal. For higher `so_n`/`sp_n` irrep blocks
(dim `2p+1 > N`) the coordinate-pair rotation is a generic `SO(d)` element **outside** `ρ(SO(N))`
(verified: `so(3)` `l2` `(0,1)`-pair generator has residual 8.0e-1 off `span(ρ(so(3)))`). The code
is internally consistent (RoPE enters only as the outer similarity `R_i Ω_ij R_j^T`, never assumed
in `ρ(G)`; orthogonality / block-diagonality / relative-position invariance all verified to fp32),
so this is interpretive framing, not a bug. *Fix:* narrow the docstring to say RoPE is a
block-diagonal orthogonal similarity that coincides with a structure-group gauge element only on
blocks where the irrep image is full `SO(d)`.

**L3. The `'unroll'` E-step estimator silently degrades to a straight-through-equivalent gradient
for every non-kernel family (including the pure `gaussian_full` path) unless
`oracle_unroll_grad=True`.** `vfe3/inference/e_step.py:395-414` (the runtime warning), `:428`
(`create_graph=(oracle_unroll_grad and e_step_gradient=='unroll')`). `uses_kernel_route` requires
`filtering + gaussian_diagonal + renyi + alpha_div==1 + include_attention_entropy + flat`; any other
(valid, pure) family routes to the autograd oracle, which returns a **detached tangent** at the
default `oracle_unroll_grad=False`.

*Adjudication (downgraded from the model/config agent's "Medium / under-trains"):* I built a
`gaussian_full` model and ran forward+backward — the prior tables **do** receive nonzero gradient
(`mu_embed.grad` |·|-sum 0.335, `sigma_log_embed.grad` 0.0047), and the truncation warning fires
once. The detached tangent only severs the *second-order* through-inference term; the additive
chain (`mu = mu_prev − detached_Δ`) keeps `d mu_final/d table = I` live, so the estimator degrades
to straight-through (a documented, valid estimator) rather than to no gradient. The pure full-cov
path therefore **trains correctly**; it just does not use the *exact* unrolled estimator the user
selected. *Fix:* none required for correctness; consider defaulting `oracle_unroll_grad=True` for
non-kernel families, or elevating the warning, so `'unroll'` means unroll.

**L4. `lambda_h_i` module docstring writes the hyper-prior term as `KL(s_i‖r)` while CLAUDE.md /
the manuscript hierarchy write `KL(s_i‖h)`.** `vfe3/lambda_h_i.py:2,5,12`. `r` (the centroid table)
and `h` (the hyper-prior centroid in the `h→s→p→q` hierarchy) are the same object; this is notation
drift, not a math error (delegation to `self_coupling_alpha` verified: `c0_h/(b0_h+KL)`, `R_h`,
`exp(log_lambda_h)` all match). *Fix:* reconcile the `r`/`h` symbol in a doc pass.

### Numerical robustness — latent / off the live path

**L5. The generic Bregman/Rényi-from-A path has no fp32 cancellation guard near α=1.**
`vfe3/families/base.py:244-248`:

```python
else:
    blend = tuple(alpha * a + (1.0 - alpha) * b for a, b in zip(tq, tp))
    div = (cls.log_partition_at(blend)
           - alpha * cls.log_partition_at(tq)
           - (1.0 - alpha) * cls.log_partition_at(tp)) / (alpha - 1.0)
```

Probe (forcing the generic path on a diagonal Gaussian): α=1.000005 → rel err **1.09e-2**;
α=1.0001 → 8.85e-4 — the exact catastrophic-cancellation pathology `_RENYI_KL_BAND` was added to
fix, but that float64-island fix lives only in the Gaussian closed forms (`gaussian.py`), not here.
**Unreachable for the two shipped Gaussian families** (both register `renyi_closed_form`), so it is
Low — but it is the would-be *pure* path for any future custom exponential family that supplies only
`log_partition_at`, and CLAUDE.md requires the generic pure path to be correct. *Fix:* mirror the
band logic generically — inside `|α−1| < _RENYI_KL_BAND` (excluding the `<1e-6` KL switch), cast
`tq`/`tp`/`blend` to double for the three `log_partition_at` calls and cast back.

**L6. `squared_hellinger` saturates to exactly `1.0` in float32 at the default `kl_max=100`,
contradicting its `[0,1)` docstring.** `vfe3/families/base.py` (`# (...) squared Hellinger H^2(q||p) in [0, 1)`;
`return 1.0 - torch.exp(-0.5 * d_half)`). Independently reproduced: far-apart Gaussians,
`kl_max=100` → `H² = 1.0` exactly (`1 − exp(−50)` underflows the fp32 mantissa). Opt-in,
non-default divergence; nothing in the audited slices keys on a strict `H² < 1`. *Fix:* soften the
comment to `[0, 1]` (note the fp32 saturation), or only matters if a strict `<1` is required.

**L7. Kernel variance floor `eps` (default 1e-6) is far higher than `natural()`'s `1e-12`,
producing materially wrong KL for sub-1e-6 variances.** `vfe3/families/gaussian.py:88-90,145-147`
(`.clamp(min=eps)`, eps 1e-6) vs `:59` (`natural()` clamps at 1e-12). Probe: a true `σ=1e-9`
(below the kernel clamp) gives closed-form KL 6.41 vs true-f64 KL 9.86 (35% error, silent).
**Not reachable on the default pipeline** — all σ are floored at `cfg.eps=1e-6` before the kernel
and callers thread `eps=cfg.eps` (verified across `prior_bank.py`, `retraction.py`,
`e_step.py:244`, `kernels.py:128`) — so this is an internal-consistency / defense-in-depth concern.
*Fix:* align the `natural()` floor and the kernel `eps` to a single source, or document that the
kernel `eps` must equal the pipeline σ floor.

**L8. `safe_spd_inverse` poisons well-conditioned batch elements when any one element defeats all
jitter levels.** `vfe3/numerics.py:96-119`: `torch.linalg.cholesky` is batched and raises for the
whole call if any element is non-PD at every ridge, falling the entire batch to `torch.linalg.pinv`
— the whole-batch poisoning its sibling `safe_cholesky` was rewritten with `cholesky_ex` to avoid.
Reproduced: a 2-element batch with one indefinite element loses the exact Cholesky inverse on the
good element. **Zero non-test callers** (`Grep` shows only `tests/`), so it is latent. *Fix:* mirror
`safe_cholesky` — `cholesky_ex`, retry only failed elements, `pinv` only the still-failed mask.

**L9. `build_killing_preconditioner` magnitude cut `|λ| < tol` could lift a genuine small Killing
eigenvalue for a custom basis.** `vfe3/geometry/phi_preconditioner.py:135-137`
(`evals = torch.where(evals.abs() < tol, reg, evals)`). The cut isolates the intended center only
because the shipped groups have a clean spectral gap (verified: glk K=4 → one ~0 eigenvalue, rest
at +8, nothing near `tol=1e-6`). Opt-in path (`mode='none'` is the pure default); documented
assumption. *Fix:* none for shipped groups.

### Robustness margins / observability on inactive paths (documented tradeoffs)

**L10. Diagonal congruence sandwich runs at fp32 while the full-cov sandwich gets a float64 island.**
`vfe3/geometry/transport.py:556-558` (diagonal fp32 einsum) vs `:569-571` / `_factored_full_covariance`
(float64 island). For non-compact groups the diagonal sandwich also squares `cond(Ω) ~ exp(2‖φ‖)`,
but only the full path got the 2026-06-13 float64 island (M4). The docstring explicitly scopes the
island to the full path and names the diagonal default as the unchanged hot path with the bound-`φ`
/ compact-group mitigation, so this is a knowing perf/accuracy tradeoff, not a defect. *Fix:* none
required.

**L11. `attention_weights` / `log_partition` lack a head-arity shape guard.**
`vfe3/free_energy.py:276-279`, `:298-304`. A single-block `(N,N)` energy combined with a per-head
`(H,N,N)` prior silently broadcasts to `(H,N,N)` (reproduced) instead of erroring. **Not reachable
through the model** — the constructor pins `n_heads==1` for single-block groups (`model.py:95-104`)
and squeezes a `(1,N,N)` prior to `(N,N)` (`model.py:362-364`) — but these are public seams in the
file. *Fix (optional, defensive):* assert `log_prior.dim() == energy.dim()` (or matching head axis)
before combining.

### Verified-correct exceptions and by-design splits (no action)

- **Head mixer** matches CLAUDE.md exception (2) exactly: exactly equivariant under the tied gauge
  (residual ~1e-14), strictly broken under untied `block_glk` (~0.84), bit-exact identity at zero
  init. (`vfe3/model/head_mixer.py`)
- **CG coupling** is exactly equivariant for arbitrary nonzero weights, means-only, zero-init
  byte-identical at step 0 (residual ~1.8e-15). (`vfe3/model/cg_coupling.py`)
- **Irrep towers** are genuine representations — the build-time bracket-homomorphism assert is real
  and dimensions match the closed forms (`so(3)` `l0..l3` = 1,3,5,7; `so(5) l2` = 14). (`vfe3/geometry/irreps.py`)
- **`free_energy()` omits the hyper-prior and γ blocks**, which are assembled in canonical envelope
  form in `model.py:707-737` (reusing `pairwise_energy` + `reduced_free_energy`). The canonical F is
  split across the belief fiber (`free_energy.py`) and the model fiber (`model.py`) — a modularity
  choice, not a gap. The docstring is honest ("extension points, absent from this default path").
- **Observation-likelihood term** is a gated stub, honestly documented as deliberately inert (no
  live caller; non-vacuous only under a future top-down shadow prior). (`vfe3/free_energy.py:340,394-395`)
- **`reference_decode` `kl_max=inf`**, EM separation (γ tied transport detached; `connection_W`
  detached in the φ step), saturation masks, `safe_cholesky` ok-mask masking, full-cov `natural()`
  using `linalg.solve` (pinned regression), weight-decay grouping / centroid exemptions, grad-accum
  mean-normalization, AMP fp32 decode island, resume/checkpoint state, `TokenWindows` shift-by-1
  (no leakage), and the metrics-registry `tau` / `effective_rank` / `holonomy_deviation` wrappers
  were all traced and verified correct.

---

## Adjudication notes (adversarial verification record)

- **Decode docstring (M1):** the model/config agent cited lines `300-303`; those are the method
  signature. The actual overclaim is in the **class docstring at `prior_bank.py:16-24`** — verified
  by reading the file and confirming `_decode_diagonal`/`_decode_full`/`reference_decode` all call
  `kl` (fixed α=1) on a hardcoded Gaussian family. Finding stands with corrected location/quote.
- **Full-cov gradient (L3):** the model/config agent rated this "Medium / under-trains." I built the
  model and confirmed the prior tables receive nonzero gradient; the estimator degrades to
  straight-through (valid, warned), not to no-gradient. **Downgraded to Low.**
- **fp16 comment (L1), squared_hellinger saturation (L6), baseline test numbers:** independently
  re-verified by direct probe / config read, not taken on the agents' word.
- Items the prior 2026-06-13 ultradeep audit flagged (Rényi cancellation band, regime-II Frobenius
  cap, full-cov sandwich float64 island) were checked in this tree and confirmed **already fixed**,
  so they are not re-reported.

## Prioritized punch-list

1. **M1** — fix the `prior_bank.py:16-24` decode docstring (and/or thread `alpha_div`/
   `divergence_family` into decode, or add a `__post_init__` warning for `use_prior_bank=True` with
   a non-KL/non-α=1 seam). Highest-value: a comment that lies plus a silent objective inconsistency.
2. **L1, L4** — correct the false fp16 comment and the `r`/`h` notation drift (comment honesty).
3. **L5** — give the generic Rényi-from-A path the same fp32 band guard as the closed forms (pure
   path correctness for future families).
4. **L6, L7, L8** — squared_hellinger `[0,1]` doc / fp32 note; align the kernel `eps` and `natural()`
   floor; make `safe_spd_inverse` per-element (latent hardening).
5. **L2, L3, L9, L10, L11** — interpretive / documented-tradeoff / latent: narrow the RoPE-as-gauge
   docstring; consider `oracle_unroll_grad=True` default for non-kernel families; optional shape
   guard in `attention_weights`. No correctness action required.
