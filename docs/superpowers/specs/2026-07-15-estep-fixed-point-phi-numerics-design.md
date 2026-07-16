# E-step Fixed-Point and Phi-Numerics Buildout Design

**Date:** 2026-07-15

**Status:** Implemented and verified on 2026-07-15.

## Purpose

This buildout makes the inference-depth semantics measurable, prevents the current reports from calling a one-step movement a fixed-point certificate, identifies the numerical source of flat-path holonomy, bounds opt-in noncompact phi charts during the M-step, and provides an exact group-product alternative to truncated BCH positional composition. Existing defaults and historical run configurations remain behavior-compatible.

The design also explains and surfaces corpus-pass discontinuities in the training loss curve. For the investigated WikiText-103 run, 116,840,318 training tokens, sequence length 128, and batch size 16 produce 912,814 windows and 57,050 optimizer steps per epoch after `drop_last`. The visible loss drops at approximately 57,050, 114,100, and 171,150 are corpus-exposure boundaries, not additional M-steps, learning-rate restarts, or E-step events.

## Confirmed inference semantics

One training step runs the configured number of E-step refinements inside the forward pass and then performs one backward pass and one optimizer update. With `n_e_steps=5`, the forward graph contains five belief refinements. It does not contain five optimizer or M-step updates. Gradient accumulation may split the batch into microbatches, but the accumulated gradients still produce one optimizer update.

The `mm_exact` kernel computes the closed-form minimizer of a locally frozen surrogate. If the current belief is denoted by $q_t$, the kernel computes

$$
T(q_t)=\operatorname*{argmin}_{q'}\widetilde F(q';q_t).
$$

The full fixed point instead satisfies

$$
q_\star=T(q_\star).
$$

Attention, transported key beliefs, state-dependent alpha coefficients, and clamp masks are recomputed from the updated state. The live fixed-point equation is therefore a coupled nonlinear root problem and has no general algebraic closed form. A closed form exists in the restricted problem where those coefficients are frozen; that restricted solve is the existing `mm_exact` update.

A decoder trained at depth one is optimized against the distribution of $q_1$. Replacing $q_1$ with $q_5$ only during evaluation creates an inference-depth distribution shift. Fixed depth five must be present during training, or randomized-depth training must include five, before depth-five evaluation can be interpreted as a matched operating point.

## Compatibility and naming

The existing `e_step_update="mm_exact"` key remains accepted and unchanged. A clearer registry alias, `e_step_update="frozen_surrogate_exact"`, dispatches to the same kernel. Configuration warnings, reports, and documentation describe the update as an exact frozen-surrogate minimizer, not a one-step fixed-point solution. Historical JSON and checkpoint fingerprints continue to accept `mm_exact`.

No default training behavior changes. New diagnostics run only at existing validation/report boundaries. New chart controls and exact positional composition are opt-in.

## E-step convergence instrumentation

Validation diagnostics run the configured $T$ E-step iterations and then one additional no-gradient iteration. The additional iteration is diagnostic only and cannot affect the decoded validation loss, model state, optimizer state, or random number stream.

The persisted metrics are:

- `estep_fp_kl`, the mean configured-family divergence from $q_T$ to $q_{T+1}$.
- `estep_fp_mu_rms`, `estep_fp_sigma_rms`, and `estep_fp_phi_rms`.
- `estep_target_gap`, the distance from $q_T$ to the undamped frozen-surrogate target when the selected updater exposes such a target; otherwise the value is unavailable rather than fabricated.
- `estep_beta_js`, the mean row-wise Jensen-Shannon divergence between attention at $q_T$ and $q_{T+1}$.
- `estep_alpha_rms_delta`, the RMS change in the active alpha coefficients.

The existing `estep_r_*_last` fields remain for compatibility but are labeled as the final configured-step movement. They are not called fixed-point residuals. The convergence figure shows the configured depth with a vertical marker and includes the one-step-ahead residual on a logarithmic scale.

A final-report depth-sensitivity artifact evaluates a fixed validation subset at depths `[0, 1, 2, 3, 5, 8]`, clipped to supported positive limits where necessary. It stores CE and variational free energy in `estep_depth_sensitivity.json` and plots both against depth in `estep_depth_sensitivity.png`. The trained depth is marked explicitly, and the artifact states that same-weight depth sensitivity is not a retrained ablation.

## Corpus-pass instrumentation

Every training CSV row records:

- `epoch`, a one-based corpus-pass index.
- `batch_in_epoch`, a one-based batch cursor inside the current loader permutation.
- `steps_per_epoch`, the exact `len(train_loader)` value.
- `corpus_pass`, the continuous ratio `step / steps_per_epoch`.

The loss plot receives `steps_per_epoch` and draws subdued vertical lines at complete corpus passes. A caption states that the loader reshuffles each pass and that an abrupt training-loss improvement can occur when examples begin their next exposure. Validation curves remain the generalization reference. These fields also make checkpoint-resume cursor behavior auditable.

## Phi and flatness instrumentation

Active-frame diagnostics record coordinate-norm median, p95, p99, maximum, fraction above the transport exponential cap, minimum applied radial scale, and vertex-condition median, p95, p99, and maximum. Raw parameter-table health is collected only during final reporting, not every log step, and records top offending token IDs and positional indices without scanning the full vocabulary on the hot path.

When BCH positional composition is active, the validation diagnostic samples positions and records the relative fidelity error

$$
\frac{\lVert\exp(\operatorname{BCH}_m(X,Y))-\exp(X)\exp(Y)\rVert_F}
{\max(\lVert\exp(X)\exp(Y)\rVert_F,\epsilon)},
$$

plus the coordinate-norm amplification ratio. It persists median, p95, p99, and maximum values.

Flat-transport diagnostics are renamed semantically. The legacy holonomy fields remain in CSV for compatibility, while new fields explicitly report runtime-precision numerical closure:

- `numerical_holonomy_fp32_abs` and `numerical_holonomy_fp32_rel`.
- `numerical_cocycle_fp32_abs` and `numerical_cocycle_fp32_rel`.
- `inverse_consistency_fp32`.
- Sampled fp64 reference counterparts for all three quantities.

On a flat path, figures use “numerical flatness residual” or “numerical closure residual.” Curvature wording is reserved for nonflat transport modes. The legacy `holonomy.png` filename is retained, but its title and axis labels follow the active transport regime.

## Floating-point policy

The project remains float32 at its public runtime boundaries. Matrix exponentials may use the existing internal fp64 island and cast their factors back to the working dtype. The buildout does not retain all pairwise transports, belief applications, or training activations in fp64.

Reference geometry diagnostics reconstruct sampled factors, products, inverses, cocycles, and holonomies in fp64 until the final scalar reduction. This establishes whether a residual is numerical without changing training. The existing block-GL path already upcasts each 30 by 30 exponential under the investigated configuration; the new reference path measures the precision lost after factors return to float32.

Precision is not used as a substitute for chart control. Neither fp64 nor a larger exponential clamp repairs a fourth-order BCH input whose norm lies outside its useful chart.

## Opt-in M-step chart bound

A new optional configuration field, `phi_mstep_max_matrix_norm`, defaults to `None`. A positive value activates projected Adam after every successful optimizer update. Each row in every trainable phi-coordinate table is embedded with the active group generators, measured by embedded Frobenius norm, and radially projected only when it exceeds the configured bound.

The projection covers token phi, learned positional phi, and independent model-frame phi tables when present. It is chunked across large vocabulary tables so it does not allocate a vocabulary-by-$K$-by-$K$ tensor. Skipped or overflowed optimizer steps do not project. The default `None` path executes no extra operations and remains byte-identical.

This is projected optimization, not an assertion that the bounded chart is globally equivalent to the unbounded GL parameterization. The selected bound is an experimental control and is reported in provenance. Adam moment buffers are preserved; chart-hit fractions reveal whether the optimizer persistently presses against the boundary.

## Exact factorized positional composition

A new `pos_phi_compose="group_product"` mode represents the active vertex as

$$
U_i=\exp(X_i)\exp(Y_i),
$$

where $X_i$ is the token-frame coordinate and $Y_i$ is the positional coordinate. Its inverse is assembled in the correct reversed order,

$$
U_i^{-1}=\exp(-Y_i)\exp(-X_i).
$$

Flat transport is then $\Omega_{ij}=U_iU_j^{-1}$ without forming a BCH approximation or a matrix logarithm. Dense, factored, and compact equal-block transport builders accept an optional right-side positional coordinate and assemble these vertex factors directly. Gradients reach both token and positional coordinates.

The initial supported route is deliberately narrow: phi gauge parameterization, flat transport, and `s_frame_mode="tied"`. Configuration validation rejects `group_product` with nonflat transport, `omega_direct`, or an independent model frame until those semantics receive a separate design. Reflection is applied outside the vertex-factor builder as it is today. The E-step may update token phi while the positional factor remains the configured fixed or learned table.

The existing `bch` and `euclidean` modes retain their current coordinate-composition semantics. No existing configuration is migrated automatically.

## Experiment definitions

The ablation registry gains three explicit, runnable experiment groups:

1. `estep_depth_damping`: matched retraining at fixed depths 1, 3, and 5, including damping 0.75 and 1.0 at depth 5, plus randomized training depth 1 through 5 evaluated at depth 5 under both damping values. Each arm records its trained depth regime.
2. `phi_chart_control`: unbounded AdamW, `mass_phi=0.01`, lower phi learning rate, pullback natural-gradient, and finite `phi_mstep_max_matrix_norm=5` arms.
3. `pos_phi_composition`: BCH, exact group product, and positional-off controls under the otherwise matched flat phi route.

The implementation makes these arms runnable but does not claim results from unexecuted long training jobs.

## Error handling and provenance

Unsupported semantic combinations fail during configuration validation. Reference diagnostics are best-effort at report time: an unavailable metric is serialized with an explicit reason and does not inherit a stale previous value. Any nonfinite sampled factor is counted and reported rather than silently removed from quantiles.

Run provenance records the canonical E-step update label, configured and sampled inference depth, chart bound, positional composition mode, fp64 diagnostic policy, and exact steps per epoch.

## Testing and acceptance criteria

Implementation follows test-first red-green cycles. Tests must establish:

- Five E-steps produce one optimizer update and backpropagate through the configured unroll.
- The one-step-ahead residual distinguishes configured movement from a fixed-point residual.
- The `frozen_surrogate_exact` alias is value- and gradient-identical to `mm_exact`.
- Epoch fields and boundary positions remain correct across normal iteration and checkpoint resume.
- Default metrics and plotting remain compatible with historical CSV files.
- fp64 reference closure is smaller than runtime float32 closure on an ill-conditioned flat example.
- Chart projection respects the embedded matrix norm, covers all live phi tables, and is a no-op when disabled.
- Exact group-product factors equal direct matrix multiplication, use the reversed inverse order, preserve flat cocycle closure in fp64, and send gradients to both factors.
- Configuration rejects every unsupported `group_product` route.
- Default-off paths preserve existing regression values.

Focused tests run before the full suite. Final pass counts come from JUnit XML. The implementation also updates the single dated post-edit record for 2026-07-15.
