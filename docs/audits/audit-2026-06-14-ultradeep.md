# Deep Audit — 2026-06-14 (ultra-deep, overnight)

Multi-agent deep audit of the VFE_3.0 codebase after a heavy build day. Run as a 3-phase
background workflow: **12 investigators** (base 5 + expert 7) sweep in parallel → **sharded
independent verifiers** re-read source for every finding (CONFIRMED / REFUTED / INCONCLUSIVE)
→ **adversarial challenge** (skeptic vs defender vs adjudicator) on every confirmed
critical/high + escalated expert-medium. 46 agents, ~4.6M agent tokens.

## Headline

- **0 critical, 0 surviving-high.** Every one of the 8 challenged findings *survived* (none
  dropped) but **all 8 were downgraded** (6 high→medium, 2 medium→low): the highs were all on
  **opt-in, non-default paths**, and the pure Gaussian default path is correct.
- **49 raw findings → 32 CONFIRMED, 17 REFUTED, 0 inconclusive.**
- The dominant theme is one cluster: the **new `DiagonalLaplace` family** (shipped today) has
  correct E-step / divergence math but **two silent inconsistent seams** under
  `use_prior_bank=True` — the **decode boundary** and the **natural-gradient Fisher metric**
  both hardcode Gaussian, with no warning. Plus a **one-line T5-bucket config-validation gap**.
- The **gauge-theory expert returned ZERO findings** — flat cocycle, congruence sandwich, head
  mixer, CG coupling, regime_ii W=0 reduction, RoPE cocycle preservation all verified
  *executably* correct. The geometry, variational, transformer, and numerics cores are largely
  clean; the confirmed defects concentrate at the **family-seam** and **diagnostic/figure**
  edges, not the training objective.

## Scope

Whole repo (135 Python files), prioritizing today's heavy surface: `families/laplace.py` (new),
`families/base.py`, `config.py` (+594), `model/model.py` (+660), `model/prior_bank.py` (+310),
`geometry/transport.py` (+233), `geometry/lie_ops.py`, `inference/e_step.py`, `gradients/*`,
`metrics.py` (+169, Wilson holonomy), `lambda_h_i.py` (new), `free_energy.py`, plus the
entry-point config dicts.

## Investigators dispatched

- **Base (5):** code-reviewer, debugger, refactoring-specialist, performance-engineer, python-pro.
- **Experts (7, theory-invariant gate met):** numerical-analyst, gauge-theorist, geometer,
  info-geometer, variational, transformer-ml, implementation-engineer. The math/theory lenses
  verified derivations symbolically (sympy) rather than asserting them.
- **Verifiers:** general-purpose agents, sharded by source file (source-only, no investigator reasoning).
- **Challenge:** audit-skeptic + audit-defender + general-purpose adjudicator per finding.

## Surviving punch list (de-duplicated, ranked)

Four investigators independently flagged the same Laplace-decode defect (CR-2 / PY-2 / INFO-2 /
IMPL-1, with IMPL-3 the docstring-level proof); two flagged the s-channel hardcode (DBG-002 /
IMPL-2). De-duplicated, the actionable list is:

1. **[medium] Laplace belief decoded with Gaussian KL, silently** — `family='laplace_diagonal'`
   + `use_prior_bank=True` runs a genuine Laplace E-step (`oracle.py:120`/`e_step.py:248`
   `get_family(cfg.family)`) but every decode kernel hardcodes Gaussian
   (`prior_bank.py:337` `get_family('gaussian_diagonal')`, `:685-701` inline Gaussian KL,
   `:742` `gaussian_full`) and reads neither `cfg.family` nor `cfg.divergence_family`. The
   correct Laplace KL exists (`laplace.py:128-129`) but no decode kernel calls it, and the only
   related guard (`config.py:1161`) warns on the renyi/alpha axis, never the family axis. The
   converged belief is correct; only its projection to logits uses the wrong metric (argmax
   flips on ~27-42% of sampled beliefs). *Fix:* add a `__post_init__` warning mirroring
   `config.py:1161` when `use_prior_bank=True` and the family is non-Gaussian, **or** add a
   Laplace decode kernel. (Challenge: downgraded high→medium — opt-in, non-crashing, pure
   Gaussian default correct.)

2. **[medium] Laplace natural-gradient uses the Gaussian Fisher metric** (INFO-1) —
   `natural_gradient` (`retraction.py:370-371`) hardcodes the Gaussian Fisher (`nat_mu=σ·grad`,
   `nat_sigma=2σ²·grad`) and is called unconditionally at `e_step.py:455` with no family key.
   The true diagonal-Laplace Fisher is `I_μ=I_b=1/b²` (sympy-confirmed), so the correct
   preconditioner is `b²·grad` on both coordinates — the mean is off by a state-dependent `1/b`
   (2× at b=0.5, diverging as the belief sharpens), a wrong *direction* on the product manifold,
   not a rescalable LR. It is a strictly-positive preconditioner (sign-preserving, every
   stationary point preserved, `grad=0→step=0`) and F itself uses the correct Laplace divergence,
   so only the trajectory/convergence-quality is wrong, not the converged objective. No
   family-keyed natural-gradient path exists. *Fix:* make `natural_gradient` a family-keyed
   Fisher hook on `BeliefParams`. (Challenge: downgraded high→medium.)

3. **[medium] T5 relative-bias bucket math is unvalidated** (TFM-1) — `attention_prior.py:219`
   divides by `math.log(max_distance / (num_buckets//2))` with no guard that
   `t5_max_distance > t5_num_buckets//2` (`config.py:778-784` checks only `>=1` / `>=2`). A
   config-valid combo crashes (`md == nb//2` → `log(1)=0` → garbage `.long()` index →
   `IndexError`) **or silently wraps** (`md < nb//2` → negative bucket index reads the wrong end
   of the bias table → wrong relative-position prior). Defaults (32/128, `causal`) are safe.
   *Fix:* one-line `config.py` guard `t5_max_distance > t5_num_buckets//2` (mirrors the existing
   `num_buckets<2` guard). (Challenge: downgraded high→medium — non-default prior, inverted
   config, loud dominant failure.)

4. **[medium] `t5_bias` freezes silently under `straight_through`** (DBG-006) — the freeze
   warning at `model.py:333` fires only for `effective_e_step_gradient=='detach'`, but
   `straight_through` also severs the gradient to `t5_bias` (both the mu/sigma tangent at
   `e_step.py:467` and the phi step are detached). Unlike its siblings (log_alpha,
   connection_W, log_lambda_*, pos_phi_free — all warned at `config.py:1267`),
   `t5_learnable_bias` is warned at *neither* the model nor config level. *Fix:* extend the
   model.py:333 condition (or the config.py:1267 predicate) to cover `straight_through`.

5. **[medium] s-channel E-step hardcodes Gaussian** (DBG-002 / IMPL-2) — `_refine_s`
   (`model.py:531`) passes `family='gaussian_diagonal'` literally, but `config.py:894` gates
   `s_e_step` on diagonality only, so `s_e_step=True` + `family='laplace_diagonal'` passes and
   refines the s-channel as Gaussian while the belief is Laplace. The challenge established the
   model channel is *uniformly* DiagonalGaussian by design (`model.py:855,862,929,940`), so the
   Laplace belief runs its own Laplace E-step against a Gaussian anchor (a well-posed mixed
   prior/posterior, no NaN) — the residue is a **missing warning** on the double-opt-in combo.
   (Challenge: IMPL-2 downgraded to low; DBG-002 verified medium.) *Fix:* warn, or restrict
   `s_e_step` to `gaussian_diagonal`.

6. **[medium] `_refine_s` drops RoPE for the s-channel** (REF-2) — `_refine_s` never forwards
   `rope`/`rope_on_cov`/`rope_on_value`, so with `s_e_step=True` + `pos_rotation='rope'` the
   s-channel E-step is RoPE-blind while the belief E-step is RoPE-transported (an inconsistency,
   both toggles default OFF). *Fix:* thread the rope args (or document the omission as
   `transport_mode='flat'` already is).

7. **[medium] Full-cov sandwich re-casts `oh.double()` H times** (PERF-06) —
   `_factored_full_covariance` (`transport.py:673-678`) loops over all H×H block pairs and
   re-casts the per-head `oh` to fp64 inside the inner loop (O(H²) casts where O(H) suffices).
   Only on the `gaussian_full` path; correctness fine, bandwidth wasteful. *Fix:* hoist
   `oh.double()` into a per-head `blocks_d64` list before the inner loop.

All remaining confirmed findings are **low** (cosmetic-but-real, latent, diagnostic-figure, or
perf micro-opts) — see the full table below. There is **no critical defect and no broken pure
default path.**

## Verifier verdicts (CONFIRMED, by area)

| # | Investigator | Finding | Final severity | Source |
|---|---|---|---|---|
| CR-1 | code-reviewer | `t5_learnable_bias=True` inert with no warning when no t5 channel | low | config.py:297; model.py:331 |
| CR-2 | code-reviewer | Laplace + use_prior_bank decodes as Gaussian (dup) | medium\* | prior_bank.py:337,685 |
| CR-3 | code-reviewer | Wilson per-head split mis-partitions unequal towers (unreachable in prod) | low | metrics.py:672-673 |
| DBG-002 | debugger | `_refine_s` Gaussian hardcode unguarded for laplace+s_e_step | medium | model.py:531; config.py:894 |
| DBG-004 | debugger | per-coord guard doesn't check family exposes `renyi_per_coord` (latent) | low | config.py:828 |
| DBG-006 | debugger | `t5_bias` freeze under `straight_through` not warned | medium | model.py:333 |
| DBG-007 | debugger | `gamma_attention_maps` omits `capture` (latent) | low | model.py:1037-1045 |
| REF-2 | refactoring | `_refine_s` drops rope for s-channel | medium | model.py:507-547 |
| REF-3 | refactoring | `docs/audit.md` untracked stale artifact | low | docs/audit.md |
| REF-4 | refactoring | per-head Wilson return data unused | low | model.py:1277 |
| REF-6 | refactoring | uncommitted entry-point edits vs edits-doc | low | ablation.py, train_vfe3.py |
| REF-7 | refactoring | config error message lists only `gaussian_diagonal` | low | config.py:829-833 |
| REF-8 | refactoring | `attention_entropy` dict key is KL(β‖π), not Shannon | low | metrics.py:150 |
| PERF-02 | performance | chunked-decode `lse_chunks` list retention (micro) | low | prior_bank.py:404-432 |
| PERF-03 | performance | diagnostics re-run full E-step (off hot path) | low | model.py:1112,1037 |
| PERF-04 | performance | `generate()` re-runs full forward per token (documented) | low | model.py:1086-1109 |
| PERF-05 | performance | Laplace fp64 upcast unconditional (intentional) | low | laplace.py:130-167 |
| PERF-06 | performance | full-cov sandwich re-casts `oh.double()` H times | medium | transport.py:673-678 |
| PERF-08 | performance | `phi_alignment_loss` double transport build | low | e_step.py:500-513 |
| PY-2 | python-pro | Laplace + use_prior_bank decode hardcoded Gaussian (dup) | medium\* | prior_bank.py:337,742 |
| PY-6 | python-pro | decorator `_wrap` inner fns lack return annotation | low | base.py:153,183 |
| PY-8 | python-pro | `cov_kind` getattr-None fallback silently degrades (latent) | low | free_energy.py:107 |
| NUM-1 | numerical | holonomy/Wilson metrics fp32 (no float64 island) | low\* | metrics.py:617,665,723 |
| NUM-2 | numerical | `spd_geodesic_distance` full-cov self-dist ≠ 0 at high cond (figure) | low | metrics.py:284-288 |
| GEO-1 | geometer | SPD trust-region: per-coord box (diag) vs Frobenius ball (full) | low | retraction.py:128,176 |
| INFO-1 | info-geometer | Laplace rides the Gaussian Fisher metric | medium\* | retraction.py:370; e_step.py:455 |
| INFO-2 | info-geometer | Laplace KL decode boundary hardcoded Gaussian (dup) | medium\* | prior_bank.py:337,685 |
| VAR-1 | variational | `e_step(return_trajectory=True, mass_phi=…)` → TypeError (diagnostic) | low | e_step.py:570-572 |
| TFM-1 | transformer | T5 bucket math crash / silent wrong on `md ≤ nb//2` | medium\* | attention_prior.py:219 |
| IMPL-1 | impl-engineer | Laplace decode hardcoded Gaussian, no guard (dup) | medium\* | prior_bank.py:337,742 |
| IMPL-2 | impl-engineer | `s_e_step` hardcodes Gaussian for s-channel (dup) | low\* | model.py:531 |
| IMPL-3 | impl-engineer | `reference_decode` docstring claims divergence-agnostic (test-only) | low | prior_bank.py:318-339 |

\* = severity set by the adversarial challenge tier (see below).

## Adversarial challenge (8 findings; all survived, all downgraded)

| Finding | Skeptic | Verdict | Final | Reason |
|---|---|---|---|---|
| CR-2 | severity inflated | DOWNGRADED | medium | reachable pure path via use_prior_bank=False linear; decode is a by-design fixed-α=1 seam; missing-warning, not missing pure path |
| PY-2 | needs two non-defaults | DOWNGRADED | medium | default Gaussian path correct; missing-family-warning class, not a broken default |
| INFO-1 | preconditioner not objective | DOWNGRADED | medium | sign-preserving, stationary points kept, F uses correct divergence; trajectory-only on opt-in family |
| INFO-2 | same root as M2 audit item | DOWNGRADED | medium | rank-correlated wrong readout (Spearman ~0.98), non-default experimental family |
| TFM-1 | inverted config, loud failure | DOWNGRADED | medium | non-default prior, safe defaults, dominant mode is a loud IndexError; one-line guard |
| IMPL-1 | doubly opt-in, readout-only | DOWNGRADED | medium | silent + no Laplace decode kernel exists, but no crash/NaN, corrupts readout not belief |
| NUM-1 | unreachable operating point | DOWNGRADED | low | retraction clamps ‖φ‖≤5.0, so real cond(Ω)~1e4-1e5 (not 1e7-1e10); ~1e-2 artifact only in curvature_field/holonomy_deviation |
| IMPL-2 | uniform Gaussian channel by design | DOWNGRADED | low | s-channel is DiagonalGaussian everywhere; well-posed mixed prior; ergonomics/missing-warning |

## Refuted (17 — not defects)

`DBG-001` (Laplace α>1 sign: the formula is correct; `csum≤0` *is* the genuine non-integrability →
kl_max), `DBG-003` (straight_through+log_lambda_h freeze *is* warned at config.py:1271),
`DBG-005`/`DBG-008`/`DBG-009`/`DBG-010` (self-refuting or unreachable-by-contract),
`REF-1`/`REF-5` (per-instance dispatch works; reduce to a stale comment), `PERF-01` (regime_ii
dense Ω is mathematically non-separable — no factored form exists to bypass), `PERF-07` (conflates
two functions; the diagnostic path is vectorized), `PY-1`/`PY-5` (the alleged list-`b0/c0`
TypeError cannot occur — `_as_coeff` materializes the tensor at every call site),
`PY-3`/`PY-4`/`PY-7`/`PY-9`/`PY-10` (premises factually wrong or latent-only). The verifier broke
investigator consensus repeatedly — e.g. several "high" Laplace-α>1 / dtype claims were refuted by
direct symbolic + numeric checks.

## Clean-bill verifications (executable evidence)

- **Gauge theory (zero findings):** flat cocycle composition exact to fp32 (~2.4e-7); Wilson W/K=1
  on flat; congruence sandwich Ω Σ Ωᵀ byte-identical factored-vs-dense and demonstrably ≠ Ω@Σ;
  head mixer exactly equivariant under the tied gauge; CG coupling equivariant for nonzero weights;
  regime_ii → flat at W=0 (~1.2e-7); RoPE preserves the cocycle; per-block τ keys off true
  `irrep_dims`; Laplace transport honestly documented as permutation/sign-exact only.
- **Divergence math:** Laplace KL + Rényi closed forms symbolically exact vs quadrature (~1e-10 to
  ~1e-17); the `sinhc` singularity reformulation is an exact identity, stable in value *and*
  autograd through the branch; α>1 `csum≤0`→NaN→kl_max is the correct convergence policy.
- **Optimizer coverage:** the `build_optimizer` exact-coverage assertion groups every learnable
  param (output_proj, head_mixer, cg path_weights, pos_phi_free, s/r tables, connection_W,
  log_alpha, log_lambda_beta, log_lambda_h, t5_bias) — the t5_bias regression class is fixed.
- **Pure paths exist and are reachable** for every documented toggle (use_prior_bank, head_mixer,
  regime_ii→flat, t5_learnable_bias, learnable_* scalars, pos_phi none/learned, use_cg_coupling),
  validated against the live registries.

## Test suite

- **Command:** `python -m pytest` (Python 3.14.4, torch 2.11.0+cpu, numpy 2.4.4, CUDA=False).
- **Result:** the full single-process suite **does not complete** — it dies with a **native
  Windows access violation (SIGSEGV, exit 139)** at ~89% (during `tests/test_train.py`),
  reproducibly, inside `_decode_linear` (`prior_bank.py:798`, the `mu_q @ Wᵀ` linear-decode
  matmul reached via `use_prior_bank=False` at the real `vocab_size=50257`).
- **Diagnosis — environment, not a logic regression:**
  - `tests/test_train.py` **passes 21 passed / 1 xpassed in isolation** (exit 0). The crash only
    appears in the *full* run → a **cumulative native-memory fault**: ~900 tests of accumulated
    torch/BLAS state, then the large `V=50257` matmul tips it over.
  - The crashing line is a trivial, unchanged matmul; an access violation there is a native/BLAS
    fault, not a code defect.
  - **Not** the duplicate-OpenMP abort: persisted with `KMP_DUPLICATE_LIB_OK=TRUE`.
  - torch is the **CPU-only** wheel on **Python 3.14.4** (very new) — the RTX 5090 is unused.
- **Chunked aggregate (5 fresh processes, round-robin file split):** _pending — to be filled in_.
- **Recommendations:** pin a supported interpreter (≤ 3.13) and install a **CUDA torch build** for
  the 5090; add a `conftest.py` / CI shim that runs the suite in per-file or `-p xdist`/`--forked`
  isolation so a cumulative native fault cannot abort the whole run.

## Notes

- Findings concentrate at the **Laplace family seams** (decode + Fisher) and **diagnostic/figure
  precision** — i.e. the freshest opt-in surface, exactly where a heavy build day would leave gaps.
  The core training objective, gauge geometry, and divergence math are sound.
- No fixes were applied. Awaiting authorization on the punch list.

## Resolution (2026-06-15)

All 7 surviving mediums + the CR-1 low were fixed on 2026-06-15 — see `docs/edits/2026-06-15-edits.md`.
Fix 2 (Laplace Fisher) got the real family-keyed natural-gradient hook (the one finding with no existing
pure path); Fixes 1/4/5 + CR-1 got footgun warnings; Fix 3 a config guard; Fix 6 a doc note; Fix 7 the
perf hoist. New regression pins in `tests/test_audit_fixes_2026_06_15.py` (14). The Laplace Fisher
(`I_mu = I_b = 1/b^2`) was re-derived symbolically before coding. Not done: a true Laplace decode kernel
(Fix 1's heavier alternative) — the warning suffices because the pure readout paths already exist.
