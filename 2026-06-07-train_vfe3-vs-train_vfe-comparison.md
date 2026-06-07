# V3 `train_vfe3.py` vs VFE_2.0 `train_vfe.py` — implementation comparison

Date: 2026-06-07
Scope: the V3 `vfe3` package behind `train_vfe3.py` versus the V2 `transformer/vfe`
package behind `transformer/vfe/train_vfe.py`, excluding the `coupled_fep/` and
`pure_fep/` subpackages (per request).
Method: an 11-dimension fan-out workflow — one domain-expert agent per dimension reading
the actual code paths in both repos and anchoring every claim to `path:line`, pipelined
into an adversarial verification stage that attacked every high-significance claim
(false-equivalence claims attacked as hard as claimed differences). 145 comparison
points; 51 high-significance claims verified; 10 overturned by verification. Raw
structured output: `docs/comparisons/_raw_workflow_output_2026-06-07.json`.

## Bottom line

V3 (`vfe3/`) is a faithful but leaner, registry-driven clean-room reimplementation of V2
(`transformer/vfe/` + `transformer/core/`). The objects the golden tests pin — the
forward logits and the cross-entropy / PPL derived from them, the canonical free energy
`F`, the flat gauge transport `Ω_ij = exp(φ_i)exp(-φ_j)`, the `gl(K)` and block-`gl(d)`
generator bases, the diagonal KL-to-prior decode, and the Fisher natural-gradient E-step
update order — match V2 by construction. The defining change is architectural philosophy,
not mathematics: V2 hard-codes valid options through `typing.Literal` plus `__post_init__`
membership lists and consumes them inside monolithic free functions and methods, whereas
V3 puts a config-selected registry behind every seam (divergence, family, alpha,
group, transport, retraction, decode, positional) so a variant is added by registration
rather than by editing call sites. V2 is several times larger per module
(`e_step.py` 3196, `trainer.py` 2244, `config.py` 1543 lines) and spreads its heavy
numerics and analytic gradients across a sibling `transformer/core/` package; V3
consolidates the same machinery into compact, single-responsibility modules
(`inference/e_step.py` 457, `train.py` 535, `config.py` 697).

The interesting content of the comparison is a set of ten verified divergences where a
first-pass "same" or "renamed" label did not survive adversarial reading. Most are
numerical clamp or scaling differences that do not move the golden-pinned logits but do
change the optimizer's loss object, or change belief gradients only on opt-in paths and
in extreme-value clamp regimes this code reading did not establish actually occur. None
of the 145 points found a contradiction in the core canonical-`F` math; the divergences
live at the numerical-conditioning and training-dynamics layer.

## Architectural relationship

The seam philosophy is the largest structural difference and was confirmed across two
independent dimensions. V3 validates every seam against a live registry — for example
`_require(value, tuple(sorted(_REGISTRY)), name)` at `config.py:310` (divergence),
`:332` (gauge_group), `:347` (transport), `:446` (alpha_mode), `:512` (retraction) — so a
newly registered functional, family, group, or retraction is legal without touching the
validator. V2 encodes the same option spaces as `typing.Literal` annotations plus
hard-coded lists in `__post_init__` (`config.py:361`/`:802` for `gauge_group`, `:1122`
for `phi_preconditioner`), where adding a variant requires editing both the type and the
runtime list. The three "modularity seam" registries V3 makes first-class — divergence
functionals (`divergence.py`), exponential families (`families/`, a `BeliefParams` ABC
with diagonal and full Gaussian subclasses), and alpha self-coupling forms
(`alpha_i.py`) — correspond in V2 to inline scalar/bool flags (`alpha_divergence` float,
`diagonal_covariance` bool, `E_learnable_alpha` bool) consumed by hard-coded free
functions (`diag_kl`/`diag_renyi` in `_numerics.py`) and a method inside the E-step.

A consolidation point that bears on the whole comparison: V2's actual heavy numerics and
analytic gradients do not live in `transformer/vfe/` but in the sibling
`transformer/core/` package (`vfe_gradients.py` at line ~1791+, `vfe_utils.py`,
`kl_computation.py`). V2's `transformer/vfe/_numerics.py` is thin (308 lines) precisely
because the SPD/Cholesky stabilization and the closed-form gradient kernels live in
`core/`. Several claims that a primitive is "V3-only" are therefore really "V3
consolidates into `vfe3/` what V2 scattered across `transformer/core/`"; the verification
stage reached into `core/` to find V2's true implementation and corrected those labels
accordingly. The nominal scope (`transformer/vfe/`) understates V2's machinery, and this
report flags each place that mattered.

## What is pinned (confirmed equivalences)

These high-significance equivalences were attacked by a verifier instructed to find a
hidden difference and survived:

| Pinned object | V3 | V2 |
|---|---|---|
| Per-head softmax temperature `τ = κ·√(d_head)` | `free_energy.py:32` `attention_tau` | `config.py:324` per-head temperature |
| Belief-coupling block weight (`λ_β` ≡ `λ_align`) over (coupling + entropy) | `config.py:132`, `free_energy.py:296,301` | `config.py:252`, `e_step.py:903/941/1020` |
| Head decomposition (`gauge_group`+`n_heads` ≡ `irrep_spec`) | `config.py:55,58` | `config.py:728`, `train_vfe.py:29,406` |
| `gl(K)` and block-`gl(d_head)` generator bases (byte-identical); `close_under_brackets` (line-identical) | `geometry/generators.py`, `geometry/closure.py` | `math_utils.generators`, `model._build_generators` |
| Flat transport `Ω_ij = exp(φ_i)exp(-φ_j)` | `geometry/transport.py` | `non_flat.py` flat path |
| Diagonal SPD retraction core (`Σ·exp(clamp(Δ/Σ))` then clamp `[ε, σ_max]`) | `geometry/retraction.py` | `core` retraction |
| E-step update order (μ, σ retracted, then φ via autograd + Killing + Lie retraction) and Fisher natural gradient | `inference/e_step.py` | `e_step.py` forward |
| `β` as the softmax stationary point of `F` (not an ad-hoc softmax) | `free_energy.py` | `e_step.py` |
| Token-weighted CE and `PPL = exp(min(ce, 20))` with `ignore_index=-100`, reported CE detached pre-scaling | `train.py:279-289`, `model.py:539` | `trainer.py:1377/1491`, `model.py:308` |

The canonical free energy itself — `α·KL(q‖p)` self-coupling, `λ·Σ β·KL` belief coupling,
and the attention-entropy term `τ·β·log(β/π)` with uniform `π = 1/N` — is present and
term-for-term matched, including the `λ` folding over coupling and entropy together. The
`γ` model-channel coupling and `λ_h` hyper-prior terms are absent from both `F` paths
(documented extension points on each side), so that omission is itself matched.

## Verified divergences

### Default-active divergences (affect the shipped configuration)

Optimizer loss scaling. V2 scales the gradient-bearing CE by `1/√(embed_dim)`:
`config.py:327` ships `normalize_ce_by_dim = True` and `model.py:316-318` applies
`loss_scale = 1.0/(embed_dim**0.5)`. V3 has no such scaling and no such knob; its
optimizer loss is the raw CE (`model.py`). Because V2 scales the entire optimizer target
uniformly (`trainer.py:1713-1714`: `(CE + mass_phi + aux)/√K`), the effect is largely
equivalent to a learning-rate redefinition, and the two repos already run different
M-step learning rates — so this is not by itself a trajectory-level divergence, but it
means raw loss numbers are not comparable and LR settings are not directly transferable
between repos. The reported CE/PPL/BPC are taken from the unscaled detached CE on both
sides, so the headline metrics remain comparable. (V2's own trainer comments disagree on
the default; the dataclass code at `config.py:327` is authoritative: `True`.)

Saturation clamps on divergences. V3 routes the self-coupling and the pairwise
attention energy through a fixed trust-region clamp `safe_kl_clamp` to `[0, 100]` with
NaN/`+inf` → 100 (`free_energy.py:289`, and the energy clamp in the attention assembly).
V2's corresponding paths are either unclamped (`_build_self_coupling_term` at
`e_step.py:1488` calls `diag_kl` with no ceiling; the transported-energy path at
`e_step.py:889-895` applies no ceiling before the softmax) or use a dimension-scaled
ceiling `max(100, 5K)` on the flat core-attention path rather than a fixed 100. A
saturating clamp is not an additive logit shift, so it does not cancel in
`β = softmax_j(log π - E_ij/τ)`: once any divergence crosses the ceiling — large gauge
frames, early training, distant tokens, exactly when transport matters — V3 saturates at
100 while V2 keeps sharpening, producing different `β` and different `(μ, σ, φ)`
gradients. The same fixed-100-versus-`max(100, 5K)` mismatch appears in the analytic
gradient kernel: V3 masks the self term by `1[0 < D(q‖p) < kl_max]`
(`gradients/kernels.py:115,127`) with `kl_max = 100` flat (`base.py:26`), while V2's self
term carries no saturation mask and never clamps the self-divergence
(`core/vfe_gradients.py:848-849,1259-1260`), and V2's only saturation point is
`K`-dependent (`core/vfe_utils.py:63-64`). The two kernels therefore disagree at
saturation and differ in the clamp constant for `K > 20`.

Gauge optimizer geometry. V2 ships a `RiemannianAdamW` optimizer that
Killing-preconditions and `GL⁺(K)`-retracts the gauge parameters (natural-gradient on the
group); V3's `train.py` uses plain Euclidean AdamW for `φ`/`ω` with no Riemannian
preconditioning. This is a genuine optimizer-dynamics difference on the default path, not
a refactor — the gauge variables descend in a different metric.

Effective rank and BPC reporting. V3's `effective_rank` is the spectral participation
ratio `(Σλ)²/Σλ²` (`metrics.py:21-36`); V2's is the exponential entropy
`exp(-Σ p log p)` (`cross_coupling_metrics.py:309`). Both equal `K` for a flat spectrum
but disagree for any concentrated spectrum, so the two repos' effective-rank figures are
not directly comparable. Separately, V3's `bpc = ce/ln 2` (`train.py:290`) is
bits-per-token, while V2's `training/bpc.py:69-79` multiplies by `tokens_per_char` to
report true bits-per-character; V3 caches no character count, so its BPC differs from
V2's by the compression ratio (several-fold for sub-word tokenizers) and cannot be made
honest without the sidecar. CE and PPL are unaffected and remain pinned.

Head-mixer placement. V2 applies the learned head mixer per block (`L` times,
interleaved with the E-step); V3 applies it once after the whole stack. At the default
`n_layers = 1` the two coincide; for `L > 1` they diverge.

### Opt-in-only divergences (dormant under the default toggles)

State-dependent alpha gradient. Under `alpha_mode = "state_dependent"` (opt-in; the
default is `learnable`/`constant`), the two repos descend different belief gradients. Both
share the closed form `α* = c0/(b0 + D)` (`alpha_i.py:83` ≡ `e_step.py:660`), but V3
explicitly adds the regularizer `R = b0·α − c0·log α` (`alpha_i.py:57`) into `F`
(`free_energy.py:291`) and lets the envelope cancel the product-rule term, whereas V2
never adds `R` and instead hand-applies the product-rule correction
`−(α²/c0)·D·dD` (`core/vfe_gradients.py:81`). The belief gradients differ by exactly
`(α²/c0)·D·dD`, nonzero whenever `D > 0`; the `(c0, b0)` hyperparameter gradients differ
too. They coincide only in the constant-alpha default. This same finding surfaced
independently in the E-step dimension ("envelope vs explicit product-rule"), a useful
cross-check.

Non-flat connection. With `transport_mode = "regime_ii"` / `use_non_flat_transport`
(both default off), the edge-relaxed cocycle skeleton `exp(φ_i)exp(δ_ij·G)exp(-φ_j)` is
shared, but the operator that produces `δ` differs materially. V2 feeds an
antisymmetrized, block-masked, `1/d_h`-scaled, `s_max·tanh`-gated bilinear with a
per-edge Frobenius clamp (`non_flat.py:266-267,298,252,299,308-312`); V3 feeds the raw
`connection_W` with a single `cocycle_relaxation` scalar and no antisymmetrization, mask,
per-generator scaling, or per-edge clamp (`transport.py:184`). They agree only at the
trivial `W = 0` fixed point, so holonomy diagnostics read different curvature when the
connection is enabled.

Full-covariance SPD ceiling. On the full-covariance path
(`diagonal_covariance = False`, opt-in), V2 clamps the output covariance spectrum to
`[ε, σ_max²] = [1e-6, 25]` on the theory that `σ_max` bounds the standard deviation
(`core/vfe_utils.py:710`); V3 clamps to `[ε, σ_max] = [1e-6, 5]` on the reinterpretation
that the eigenvalues are variances (`retraction.py:168`). For any covariance with a
post-retraction eigenvalue in `(5, 25]` the two return different SPD matrices at identical
default parameters. The registered wrapper default `trust_region` also differs (V3 5.0 vs
V2 2.0), though the bare functions match at 2.0.

Gauge-RoPE. With `pos_rotation = "rope"` (default `none`), V3 implements a true
block-diagonal transport sandwich `Ω^RoPE = R_i Ω_ij R_j^T`, whereas V2's RoPE is a
mu-only pre-KL rotation — different objects when enabled.

### Behaviorally inert formula difference

Decode raw-logit constant. V3's fused diagonal KL decode retains the v-independent
per-position term `K + Σ_k log σ_q` (returning exactly `−KL/τ`), and centers means by
their vocabulary mean before the matmul as a float32-cancellation preconditioner; V2 drops
that constant. The raw logit tensors therefore differ by a per-position additive shift,
but softmax, cross-entropy, and argmax are identical, so there is no behavioral effect.
V3's own `tests/test_prior_bank.py:46` pins the kept-term form against its own reference,
not against V2.

### A scope-boundary correction

The first-pass claim that V3's escalating-jitter `safe_cholesky` / SPD-inverse primitives
are "V3-only / divergent" was overturned to "refactored": V2 has the same
`cholesky_ex` + per-element `info` + escalating-jitter + ok-mask technique live in
`core/vfe_utils.py:_safe_spd_inv` (`:242,:253,:270`, production-reachable from
`vfe_gradients.py:643`) and a factor-returning analogue in
`core/kl_computation.py:_cholesky_with_fallback`. V3 consolidates three scattered V2 sites
into one reusable `vfe3/numerics.py` primitive (with a genuine refinement: round-0 zero
jitter so the factor is byte-identical to plain Cholesky on SPD inputs). The "absent in
`_numerics.py`" reading is literally true but misleading about V2 as a whole.

## Capability ledger

Present on only one side (not divergences in shared code, but real capability gaps):

| V2-only | V3-only |
|---|---|
| `RiemannianAdamW` Killing-preconditioned gauge optimizer | Registry seams (add-by-registration) across every config axis |
| SO(N) higher/multi-irrep machinery (SO3 spin-`l` tesseral, wedge2, sym2-traceless, block assembly) | `tied_block_glk` group; unified gram-pinv Lie-coordinate path |
| Cross-head super-block merge/reorder plumbing and its metrics (`cross_coupling_metrics.py`) and visualizations (`cross_coupling_viz.py`) | Straight-through / detach E-step backward-estimator control |
| Decode mixture-of-Gaussians (`decode_n_components`, learnable weights) | Log-Euclidean SPD retraction; pullback `φ` preconditioner |
| `gauge_fixed_priors` shared-base orbit prior `μ_v = A_v μ_0`, `Σ_v = A_v diag(s) A_vᵀ` (a `NotImplementedError` stub in V3) | Additive attention log-prior `B_ij` (uniform/causal/alibi) folded into `β` logits and `π` |
| E-step safety mechanisms: σ condition-clamp (`κ ≤ 10`), isotropic enforcement, `e_mu_q_trust`, `e_grad_clip`, nat-grad-norm cap | Centralized `numerics.py` conditioning module + production autograd-of-`F` oracle with runtime fallback |
| Geometry-faithful precomputed-distance UMAP (Bhattacharyya / log-Euclidean / `GL⁺` Ω-geodesic) + unsupervised auto-`k` clustering; full tokenization pipeline (`datasets.py` 3138 lines) | Group-dispatched gauge invariants (non-vacuous for SO/Sp), gauge-equivariance certificate, linguistic-taxonomy UMAP coloring; `amp_dtype`/`grad_accum_steps`/`min_lr_frac`/synthetic fallback; thin cache-reader data loader |
| `USE_FULL_COV` / `USE_VFE1_PRESET` opt-in preset blocks; `omega_direct` parameterization | `diagonal_chunked` decode (memory win) |

## Per-dimension verdict summary

| Dimension | High claims | Overturned | One-line relationship |
|---|---|---|---|
| Entry + config | 4 | 0 | Faithful leaner re-impl; registry seams vs Literal gates is the one structural divergence |
| Train loop / M-step | 2 | 0 | Same AdamW + warmup-cosine + best-by-PPL skeleton; V3 drops RiemannianAdamW, collapses 8 LR groups to 3 |
| Model forward | 4 | 1 | Same inference pipeline; optimizer loss object and head-mixer placement diverge |
| E-step | 6 | 0 | Same per-iteration math/order/natural-gradient; alpha-envelope vs product-rule and missing safety clamps diverge |
| Free energy | 8 | 2 | Canonical `F` term-for-term matched; clamps and the surrogate split differ |
| Gauge / Lie | 4 | 0 | Generator bases byte-identical; V3 adds `tied_block_glk`, drops SO(N) higher irreps |
| Transport / SPD / RoPE | 6 | 2 | Flat transport + SPD core matched; non-flat parameterization, full-cov ceiling, RoPE diverge |
| Prior / decode | 3 | 1 | Diagonal KL decode equivalent up to softmax-invariant shift; V3 drops orbit prior + mixture |
| Registries | 6 | 2 | Same closed-form numerics; registry-vs-flag architecture, state-dependent-alpha gradient diverge |
| Numerics / gradients | 3 | 2 | Same math; V2's primitives live in `core/`, so "V3-only" is mostly consolidation |
| Data / metrics / viz | 5 | 0 | Thin loader vs full pipeline; BPC units and effective-rank definitions diverge |

## Methodology and caveats

Every claim above traces to a `path:line`-anchored finding in the persisted workflow
output, and the most consequential default-path claim (the `1/√K` optimizer-loss scaling)
was additionally confirmed by direct grep of both repos. The adversarial verification
stage, which attacked false-equivalence claims as hard as claimed differences, is what
surfaced all ten overturned labels — without it, the report would have under-reported the
clamp, scaling, and gradient divergences as "same."

Three caveats bound the result. First, scope: V2's true gradient and SPD-conditioning
implementation spans `transformer/core/`, outside the nominal `transformer/vfe/` boundary;
the agents reached into `core/` when a V2 symbol was missing from `vfe/`, but a reader
should treat "V2's `_numerics.py` is thin" as an artifact of where V2 puts the code, not
evidence that V2 lacks the machinery. Second, the comparison reads code, not runs: it
establishes structural and formula-level (non)equivalence and identifies where numerics
diverge, but it does not execute a golden test, so claims of bit-equivalence on the pinned
objects rest on code reading rather than a numerical diff. Third, two consequential
findings — the optimizer-loss scaling and the state-dependent-alpha gradient — had their
verifier note that an advisor channel was rate-limited; their conclusions rest on the
literal quoted code, which I re-confirmed for the scaling claim.
