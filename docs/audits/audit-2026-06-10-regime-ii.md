# Deep Audit — 2026-06-10 — regime_ii transport path

> **Status update (same day):** all confirmed findings fixed; see the evening-pass section of
> `docs/edits/2026-06-10-edits.md` for the change-by-change record. F8d was investigated and
> resolved as semantically unsound (Omega depends on (mu, phi), which advance past every build,
> so the proposed reuse would serve a stale operator); the refuted F8b needed no action. The
> projection-matrix investigation below was deliberately NOT acted on, per the user.

## Scope

The `transport_mode='regime_ii'` path end to end: `vfe3/geometry/transport.py`
(`_build_regime_ii`, `stable_matrix_exp_pair`), `vfe3/inference/e_step.py` (transport plumbing,
phi step, free_energy_value), `vfe3/gradients/{kernels,oracle}.py`, `vfe3/model/{model,block,stack}.py`,
`vfe3/config.py` guards, `vfe3/train.py` optimizer wiring, `vfe3/metrics.py` and `vfe3/viz/extract.py`
diagnostics, `tests/test_regime_ii.py`. Per user instruction the adversarial challenge tier was
**skipped**; verdicts below are the independent verifier's. Per CLAUDE.md audit policy, config
defaults were not judged — only the path's correctness when active. The user-accepted gauge break
at nonzero W was treated as settled and not re-reported.

Additionally investigated (user question): is regime_ii the natural entry point for standard
transformer projection matrices, and why does test perplexity bottom out at 60? See the final
section.

## Investigators Dispatched

- Base five: code-reviewer, debugger, refactoring-specialist, performance-engineer, python-pro.
- Expert tier (theory gate met): audit-gauge-theorist, audit-geometer, audit-numerical-analyst,
  audit-transformer-ml, audit-variational, audit-implementation-engineer — selected because the
  path touches transport/gauge composition, SPD sandwiches, a matrix-exp hot loop, the attention
  energy, the E-step descent objective, and many toggle interactions.
- One dedicated general-purpose investigator for the projection-matrix / 60-ppl question.
- Verifier: general-purpose, fresh context, source-only rules.

## Verifier Verdicts (deduplicated findings)

| # | Finding | Verdict | Severity (verified) | Source |
|---|---------|---------|---------------------|--------|
| F1 | Analytic kernel route omits dOmega/dmu under regime_ii; `uses_kernel_route` never inspects `transport_mode`, so the canonical operating point descends a frozen-Omega gradient (executed: cosine ≈ 0.48 vs true grad at synthetic scale) | CONFIRMED-mechanism / INCONCLUSIVE-magnitude | **high** | kernels.py:142-161, :133, :215; transport.py:186 |
| F2 | Oracle with use_live=False differentiates a detached clone against a pre-built Omega (also omits dOmega/dmu); oracle signature cannot build regime_ii Omega, so no ground-truth dF/dW or dF/dmu check exists; tests pin gradient existence only | CONFIRMED | **high** (test gap) | oracle.py:51-78, :87-92; e_step.py:395; test_regime_ii.py:179-199 |
| F3 | delta = c·mu_i^T W^a mu_j is unbounded (quadratic in ‖mu‖, generators unit-Frobenius); sole guard is the per-edge Frobenius clamp returning exp(15·M/‖M‖) — a different operator — silently, with no monitor; past the clamp the cocycle_relaxation homotopy is inert and autograd optimizes the clamped surrogate (feedback loop via per-iteration rebuild) | CONFIRMED-mechanism / INCONCLUSIVE-magnitude | **high** (critical iff the clamp engages at trained scales — unproven; executed magnitudes used ×50-inflated test means) | transport.py:184-194, :232-234; e_step.py:359-363; generators.py:67-70 |
| F4 | Omega_ii ≠ I: no i==j exclusion in the delta einsum; nonzero self-energy E_ii enters the unmasked softmax (flat path gives E_ii = 0 exactly) | CONFIRMED | **high** | transport.py:186-197; free_energy.py:252-263 |
| F5 | Omega_ij·Omega_ji ≠ I for general W (reciprocity broken) | CONFIRMED (math); no source consumer relies on reciprocity — debugger's HIGH rejected | low / document | transport.py:186; metrics.py:439-447 |
| F6 | `_transport` never forwards gauge_mode; trivial-gauge early return unreachable via this path | CONFIRMED — but no `gauge_mode` config field exists, no production caller runs trivial | low (dead knob) | e_step.py:37-66; transport.py:134,181 |
| F7 | Model channel hardcoded flat under regime_ii: gamma coupling (documented) and `_refine_s` (no intent marker) | CONFIRMED | medium | model.py:711, :439 |
| F8a | `_build_regime_ii` calls `compute_transport_operators`, materializing a dense flat (B,N,N,K,K) Omega that is discarded; `build_factored_transport` already exists | CONFIRMED | medium (perf) | transport.py:171-172, :355, :359-391 |
| F8b | `.to(dtype).contiguous()` double copy | REFUTED — `.contiguous()` on a contiguous tensor returns self | n/a | transport.py:234, :243 |
| F8c | float64 upcast keys on full K (`d = matrix.shape[-1]`, threshold 20), not block size — block_glk at K≥20 runs the per-block exp in float64 even when d_head < 20; perf "critical" and numerics "never engages" both corrected | CONFIRMED (perf mechanism for K≥20) | medium (perf) | transport.py:236-246 |
| F8d | Repeated full Omega rebuilds: 2 per e_step_iteration when e_phi_lr>0 (mu step + phi step), +1 per trajectory sample, +1 in diagnostics(), +1 per layer in attention_maps(), +1 per iteration in e_step_belief_trace | CONFIRMED | medium (perf) | e_step.py:285-286, :359-363, :493-512; model.py:864-869, :989-994; extract.py:240-246 |
| F8e | `numerical_health` builds flat transport regardless of cfg.transport_mode — health panel describes the wrong model under regime_ii | CONFIRMED | medium (diagnostics correctness) | extract.py:313; e_step.py:42 |
| F9 | connection_W's optimizer group has lr only; inherits global AdamW weight decay (working tree: 0.02 via train_vfe3.py:202) with no dedicated frame-norm ceiling analogous to phi_weight_decay | CONFIRMED (0.05-vs-0.02 contradiction resolved: config default 0.05, working tree 0.02) | medium | train.py:111-112, :156; config.py:342 |
| F10 | regime_ii × gaussian_full: indefinite transported Sigma is masked NaN→kl_max silently in the full-family KL; no config guard for the combination (impl-engineer executed: NaN connection_W.grad under oracle_unroll_grad=True) | CONFIRMED (line cite corrected to gaussian.py:239-278) | medium-high | gaussian.py:239-278; numerics.py:58-81; config.py:917-1008 |
| F11 | Zero-TENSOR W (the actual init) takes the generic einsum path; "byte-flat at init" holds only to fp32 tolerance (tests pin atol 1e-6) | CONFIRMED | low | transport.py:177-181; test_regime_ii.py:51,62,70 |
| F12 | `gauge_equivariance_residual` co-transforms the given Omega jointly, so it is **blind** to the regime_ii builder-level break — it false-certifies rather than (as the investigator claimed) blowing up; no regime_ii awareness in metric or caller | PARTIAL: blindness CONFIRMED, residual claim REFUTED | low-medium | metrics.py:826-831; viz/report.py:191-196 |
| F13 | (a) W^a reads full-K means — cross-head content coupling; (b) delta is position-blind (RoPE wraps outside the full Omega; mu fed raw); (c) means-only RoPE (rope_full_gauge=False) feeds the KL a rotated mean with un-rotated covariance | CONFIRMED (all three) | low (design-adjacent; (c) has an intent marker) | transport.py:186, :74-79, :414-419, :454-456 |
| F14 | "accepted-and-ignored" inline comments on free_energy_value params contradict the body, which honors them | CONFIRMED — documentation defect only | low | e_step.py:168-174 vs :200-218 |
| F15 | phi step evaluated at (mu_new, sigma_new, phi_old) | CONFIRMED factually — sequential Gauss-Seidel coordinate descent, standard | low / no action | e_step.py:421-455 |
| F16 | Registered metric "holonomy_deviation" routes to the deterministic row-major estimator the module itself supersedes; diagnostics() uses the sampled one — same key, two estimators | CONFIRMED | low | metrics.py:890-893; model.py:925 |
| F17 | Logged F (true regime_ii F) is not the functional the descent direction is a gradient of | CONFIRMED — derivative of F1/F2, merged there | (merged) | e_step.py:199-209 |
| F18 | Signature convention: defined float `cocycle_relaxation` placed after Optional params | CONFIRMED | low | transport.py:133-138; e_step.py:41-45, :89-95 |

## Surviving Punch List (verified, ranked; challenge tier skipped per user request)

1. **[high] E-step gradient is wrong under regime_ii** — `kernels.py:142-161` /
   `oracle.py:87-92` — the kernel route and the non-live oracle both treat Omega as constant in
   mu, so under regime_ii the beliefs descend a frozen-Omega functional while `free_energy_value`
   logs the true F. Fix: route `transport_mode='regime_ii'` to a live-omega autograd path (build
   Omega from the same live mu leaf inside the differentiated F), or add the dOmega/dmu term to
   the kernel; then add the missing FD/oracle pin for dF/dmu and dF/dW (closes F2 and F17).
2. **[high] Unbounded delta meets a silent operator-substituting clamp** — `transport.py:184-194,
   232-234` — bound delta upstream (normalize the bilinear by ‖mu_i‖‖mu_j‖, or tanh-saturate at a
   documented scale) so the clamp never defines the trained operator; add a cheap off-hot-path
   max|delta| / clamp-activation readout to `numerical_health` (also fixes the inert-homotopy and
   biased-gradient corollaries).
3. **[high] Spurious self-energy: Omega_ii ≠ I** — `transport.py:186-197` — zero the diagonal of
   delta before the matrix exp so E_ii = 0 exactly, matching the flat path's self-coupling.
4. **[medium-high] regime_ii × gaussian_full silently saturates to kl_max** —
   `gaussian.py:239-278` — add a config-time warning for the combination (mirroring the existing
   regime_ii estimator warnings) or floor the transported Sigma's eigenvalues before the
   divergence.
5. **[medium] Model channel on a different connection** — `model.py:439, :711` — thread
   cfg.transport_mode (or an explicit documented flat-only marker) into `_refine_s`; gamma already
   carries the marker.
6. **[medium] connection_W lacks a dedicated norm ceiling** — `train.py:111-112` — give it its own
   weight-decay group (the analogue of phi_weight_decay), since W's growth drives item 2.
7. **[medium] Diagnostics describe the wrong model** — `extract.py:313` (flat under regime_ii),
   `metrics.py:890-893` (biased holonomy estimator behind the registry key), `metrics.py:826-831`
   (equivariance certificate blind to the builder-level break — relabel or test Omega(g·mu)
   against g·Omega(mu)·g^{-1}).

## Speedups (user-flagged priority)

1. **Stop materializing the discarded flat Omega** (F8a): replace `compute_transport_operators`
   with `build_factored_transport` inside `_build_regime_ii` — removes a dense (B,N,N,K,K)
   allocation plus its einsum from every build. One-line change, exactness preserved.
2. **Key the float64 island on the exp block size, not full K** (F8c): `stable_matrix_exp_pair`
   upcasts when `matrix.shape[-1] >= 20`, so block_glk at K=20 runs its d_head=10 per-block exps
   in float64 — on consumer CUDA that is ~1/64 fp32 throughput on the single most expensive op in
   the path. When `block_dims` is given, dispatch the dtype per block dimension.
3. **Reuse the last E-step Omega** (F8d): diagnostics(), attention_maps(), the trajectory
   diagnostic, and e_step_belief_trace all rebuild a full regime_ii Omega from tensors the E-step
   just used. Return the last-built Omega (side channel or optional out-param) and accept a
   prebuilt Omega in `free_energy_value`.
4. **Document e_phi_lr=0 as a regime_ii cost halver**: the phi step's second Omega build per
   iteration disappears when the phi step is off.
5. **Causal-mask waste**: delta and the per-edge exps are computed for all N² ordered pairs while
   the causal prior zeroes j>i — a triangular-only build would halve the edge-exp count (left as
   an opportunity; the masked pairs were verified not to leak into F).

The refuted speedup: the `.contiguous()` after `.to(up_dtype)` is a no-op, not a copy (F8b).

## Test Suite

- Command: `python -m pytest --junitxml=audit_test_results.xml` (temp file removed after reading)
- Result (junitxml): tests=829, failures=1, errors=0 — i.e. 827 passed, 1 xpassed, 1 failed.
- The failure is `tests/test_config.py::test_config_phi_retract_mode_validated`, caused by the
  uncommitted working-tree change of the default `phi_retract_mode` to `"bch"` (test pins
  `"euclidean"`). Unrelated to regime_ii; resolve by updating the pin or reverting the default
  before merge.

## Investigation: is regime_ii where standard transformer projections enter, and why 60 ppl?

Findings of the dedicated investigator, verified against the working tree (regime_ii is OFF in
both entry points; the 60-ppl operating point is wikitext-103, gpt2 BPE V=50257, N=128, K=20,
H=2, L=1, T=2 E-steps, e_phi_lr=0, bias-free rank-20 linear decode, ~13.09M params of which 92%
are vocab tables and exactly 4 parameters — the head mixer — mix content globally).

**The correspondence.** delta_ij^a = mu_i^T W^a mu_j is algebraically the same species as a QK
logit mu_i^T W_Q^T W_K mu_j, and connection_W carries ~80,000 parameters at the active shapes —
100× a standard per-head QK budget. But the correspondence is interpretive, not exact, for three
verified reasons. First, the bilinear never appears as the logit: expanding the KL energy to
first order in delta, the logit perturbation arrives multiplied by a data-dependent prefactor
whose mean-channel part is proportional to the residual (Omega_flat·mu_j − mu_i) — it vanishes to
leading order exactly at the transported-mean-matched pairs where attention currently peaks. The
net logit is quartic in mu, not bilinear. Second, logits and values cannot be decoupled: the same
exp(delta·G) sits inside the Omega that transports the value, so any logit-shaping deformation
moves the values by the corresponding group action — the central W_QK-independent-of-W_V freedom
of a transformer has no counterpart; this is the genuine gauge-theoretic content of regime_ii (it
deforms the connection, and the connection plays both roles). Third, the flat path already
carries a constrained bilinear logit through the Mahalanobis term (mu_i^T diag(s)^{-1}
exp(phi_i)exp(−phi_j) mu_j with the 10M-parameter phi table behind it) — QK capacity is not
absent, it is constrained-and-tied.

**What has no counterpart at all.** No FFN analogue exists (zero learned per-token nonlinearity
between embedding and readout; a standard transformer puts ~2/3 of non-embedding parameters
there); W_O's analogue is the 4-parameter head mixer; W_V is restricted to the group orbit
(orthogonal-only for so_n towers — no scaling or shear of values). The residual/LayerNorm
analogues (belief carry, alpha self-coupling, trust regions) are present and are not the deficit.

**The 60-ppl ranking (evidence-ordered).** (1) Decode/width ceiling: a bias-free rank-20 softmax
factorization over 50,257 classes — the softmax-bottleneck regime at an extreme; the
embed_dim sweep already in ablation.py is the direct test. (2) Missing interaction/FFN
learnables, per the previous paragraphs. (3) Decode-mode choice: the prior-bank KL decode is an
effective rank-≈2K+1 readout with a built-in per-vocab bias; the current readout is the less
expressive of the two and lacks even decode_bias. (4) Optimization is the weakest explanation:
the active config takes the kernel route, gradients reach all tables, and no freeze footgun bites
it. ln 60 ≈ 4.09 nats/token sits in strong-n-gram/weak-LSTM territory on this corpus
(word-level conversion ≈ 90–170 ppl depending on the tokens-per-word ratio; LSTM baseline 48.7
word-level, Transformer-XL 18.3).

**Verdict.** regime_ii is the sanctioned slot where a learned bilinear pair map enters, and it is
the honest input-dependent generalization (closer to a hypernetwork than to static W_Q/W_K) —
but as a QK surrogate it is indirect, as a route to parity it is insufficient alone (FFN, W_O,
and decode rank are untouched by it), and it is the most expensive lever in the codebase (dense
(B,N,N,K,K) tensors, ~B·N²·T per-edge matrix exps per forward, the factored fast path
forfeited). The mathematically cleanest QK entry is not regime_ii but a learned content-dependent
attention prior pi_ij = softmax(causal + mu_p,i^T A mu_p,j / sqrt(d)) built from embedding means
through the existing attention-prior registry seam: the free energy keeps its exact form, beta
remains the F-stationary softmax, the entropy term is unchanged, and the cost is 400–800
parameters rather than a dense transport — with the same accepted impurity class as the head
mixer. Predicted ppl levers, in order: embed_dim up; decode_bias / prior-bank KL decode; depth
(n_layers / n_e_steps); the bilinear attention prior; regime_ii last. Unverified items are
flagged in the investigator's full memo (no training or profiling was run; baseline numbers for a
conventional 13M transformer are practice estimates, not citations).
