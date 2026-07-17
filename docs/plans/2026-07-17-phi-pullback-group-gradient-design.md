# Outer-Objective Phi Pullback Group-Descent Design

**Date:** 2026-07-17

**Status:** Approved in conversation and awaiting review of this written specification.

## Purpose

This change replaces the optional stateful phi M-step with a stateless, manuscript-consistent local group descent while retaining the current supervised outer training objective. The default optimizer path remains ordinary AdamW and must remain behaviorally unchanged.

When selected, the new path consumes the coordinate covector produced by the existing outer loss, computes a ridge-regularized Frobenius-pullback direction, converts that chart vector to the left-trivialized velocity required by the group equation, and applies a right-multiplication retraction. It does not implement the canonical fixed-state variational-free-energy frame M-step. That larger objective change is marked explicitly as a code TODO at the completed scalar-objective assembly seam.

The current heavy-ball and coordinatewise-Adam phi implementations are removed. They are not retained as legacy optimization modes because neither realizes the approved geometric update.

## Confirmed defect

With `m_phi_natural_grad=False`, the current code places phi tables in ordinary AdamW. That default path is not defective and remains the default.

With `m_phi_natural_grad=True`, `GaugeNaturalGradAdamW.step` first solves a local pullback system but then applies one of two coordinate-space moment rules. The heavy-ball branch accumulates chart vectors across changing tangent spaces without vector transport and adds the resulting buffer directly to phi. The Adam branch applies coordinatewise second-moment normalization after metric preconditioning, so the result is no longer the pullback Riemannian-gradient direction. Both branches omit the conversion from a chart tangent to the left-trivialized Lie-algebra velocity, and both update phi additively rather than through the required right group retraction.

The implementation is also described too broadly. The selected metric is the pullback of the ambient Frobenius metric through the matrix exponential. It is not the Fisher information metric of the Gaussian belief family and is not the predictive categorical Fisher metric of the outer cross-entropy model.

The live click-run configuration currently leaves the optional path disabled. This repair therefore changes only explicitly selected geometric phi updates.

## Objective contract

The first implementation uses the current outer scalar. In the ordinary configuration this scalar begins as cross-entropy and may include explicitly enabled outer regularizers. With unrolled inference, its phi covector is the total derivative through the finite E-step trajectory. Denoting that scalar by $L_{\mathrm{outer}}$, the consumed covector is

$$
g_\phi = \frac{\partial L_{\mathrm{outer}}}{\partial \phi}.
$$

The optimizer is objective-agnostic. Let $g_\phi^{\mathrm{raw}}$ denote the raw differential above. With gradient clipping disabled, `p.grad` equals $g_\phi^{\mathrm{raw}}$. With clipping enabled, the established training order first unscales and clips the differential, producing a processed covector $\bar g_\phi$ that the geometry kernel consumes without further normalization. The optimizer does not reconstruct cross-entropy, beliefs, attention weights, or variational state.

After all optional additions to `loss` and immediately before the model returns the completed outer scalar in `vfe3/model/model.py`, the implementation adds this comment:

```python
# TODO(canonical-vfe-phi-mstep): `loss` is the outer supervised objective (CE plus
# enabled outer regularizers), so the pullback/group phi step consumes its covector.
# Add a separately selected fixed-returned-state VFE frame objective, declare whether
# beta/gamma are frozen or envelope-eliminated, and keep the optimizer objective-agnostic.
```

The later canonical route will require a separately selected scalar or gradient context that holds the returned E-step state fixed. Exact-envelope language is justified only when stationarity has been established. The later design must declare whether beta and gamma are frozen as coordinate variables or eliminated and recomputed at a simplex stationary point. It is outside this implementation. A Gaussian-Fisher or predictive-Fisher metric is also outside scope.

The outer cross-entropy uses a fixed vocabulary readout and is therefore gauge-fixed. A local group retraction does not make its frame trajectory gauge-invariant. This implementation is manuscript-consistent in its local frame geometry, not a gauge-invariant or canonical-VFE frame flow.

## Geometric update

For one stored frame factor, write

$$
X(\phi)=\sum_a \phi^aG_a,
\qquad
U(\phi)=\exp X(\phi).
$$

Let $J_\phi=D\exp_{X(\phi)}$ restricted to the registered generator span. The ambient-Frobenius pullback metric is

$$
\mathcal{G}_{ab}(\phi)
=\left\langle J_\phi[G_a],J_\phi[G_b]\right\rangle_F.
$$

Let $B_{ab}=\langle G_a,G_b\rangle_F$ be the generator Gram matrix. The ridge-regularized chart direction is

$$
v_\phi
=\left(\mathcal{G}_{\mathrm{pull}}(\phi)+\lambda_{\mathrm{rel}}B\right)^{-1}\bar g_\phi.
$$

Using $B$ rather than a coordinate identity makes the ridge scale covariant under a change of generator basis. The damping term is part of the executable numerical contract. Documentation and diagnostics call the result a ridge-regularized pullback direction rather than the Riemannian gradient of the undamped metric.

The chart vector is converted to the left-trivialized velocity

$$
\xi
=\operatorname{dexp}^{L}_{\phi}(v_\phi)
=\frac{1-e^{-\operatorname{ad}_{\phi}}}{\operatorname{ad}_{\phi}}v_\phi.
$$

The required group step is right multiplication,

$$
U^+=U\exp(-\eta\xi).
$$

Because this path continues to store phi rather than U, the updated chart coordinate is recovered locally with the existing registered BCH composition,

$$
\phi^+=\operatorname{BCH}_4(\phi,-\eta\xi).
$$

`compose_bch(phi, delta, ...)` has the correct operand order because it approximates `log(exp(phi) exp(delta))`. Reversing the operands would implement a different, left-multiplication update.

The order-four BCH recovery is a local retraction, not a globally exact logarithm of the finite product. A global real logarithm is unavailable for arbitrary elements of the positive-determinant component of GL. Exact finite group storage remains the role of the existing `omega_direct` parameterization.

## Stored-factor product geometry

The trainable token frame, learned positional frame, independent model token frame, and independent model positional frame are separate stored group factors. The new update equips their product with the block product of the per-factor Frobenius-pullback metrics. Each factor is stepped from its own autograd covector.

The forward pass may compose token and positional factors before building transport. Autograd already carries that composition's chain rule into the covector of each stored factor. The selected product metric does not claim to equal the pullback metric of the complete composed transport map, and it does not introduce a joint token-position metric solve.

## Configuration

The boolean `m_phi_natural_grad` is replaced by

```python
m_phi_update_mode: str = "adamw"
```

The registered values are `"adamw"` and `"pullback_group"`. A small phi M-step policy registry owns those values and their optimizer-group metadata so a future update rule is added by registration rather than another call-site conditional.

`"adamw"` preserves the existing default optimizer topology and semantics. `"pullback_group"` selects the stateless update defined above. The term `natural_grad` is removed from the public phi M-step control because it obscures the distinction between the frame pullback metric and a statistical Fisher metric.

`"pullback_group"` is valid only with the phi-coordinate gauge parameterization. Combining it with `gauge_parameterization="omega_direct"` is a configuration error because the token frame is already stored and stepped as a group element there. `"adamw"` plus `omega_direct` preserves the existing mixed optimizer, including its `optimizer_extra`, dirty-row mask, and retraction-cadence state.

The fields `m_gauge_momentum` and `m_gauge_update_rule` are removed from the runtime configuration, click-run dictionaries, validation, artifacts, and tests. Their heavy-ball and Adam implementations and optimizer-state slots are deleted.

`phi_precond_mode` remains because it also controls the E-step registry. When `m_phi_update_mode="pullback_group"`, configuration validation requires `phi_precond_mode` to select a supported pullback implementation. A non-pullback mode is an error rather than a warning. The M-step group retraction uses BCH directly and does not reuse `phi_retract_mode`, which remains an E-step control.

Phase one accepts only the built-in `glk` and untied `block_glk` registrations under the phi-coordinate parameterization. The full `glk` representation must have $K\le12$; every local `block_glk` irrep must have dimension at most 12. These bounds match the structure-constant kernel's executable ceiling and fail at configuration time rather than during an optimizer step. The accepted routes provide the tested full-GL elementary bases, bracket closure, and the required full or per-block pullback implementation. `tied_block_glk`, compact orthogonal groups, symplectic groups, irrep towers, cross-coupled bases, and custom registrations remain rejected until they receive group-specific radii and oracle coverage. A group whose basis cannot certify closure fails rather than projecting an out-of-span trivialized velocity silently. `pos_phi_project_slk=True` likewise requires a metric constructed on the projected positional factor; until that projected pullback is implemented and tested, that combination fails closed instead of applying the raw full-GL factor metric to a different forward map.

The new path has an explicit positive `m_phi_group_trust_radius`, initially `0.1`. It bounds the embedded Frobenius norm of the left-trivialized velocity placed in the right-multiplication factor after multiplication by the learning rate. The existing `phi_mstep_max_matrix_norm`, when configured, supplies an additional candidate chart bound. When it is absent, the pullback-group route uses the established noncompact chart radius of `5.0`; this conditional bound does not affect the AdamW route.

The selected mode requires an explicit `transport_chart_max_norm` larger than the per-factor chart radius and smaller than the transport kernel's detached hard-clamp radius. Per-factor bounds cannot certify the norm of a BCH-composed token-position chart. The existing transport validity check therefore remains the final fail-closed guard on the effective forward chart: the forward pass either remains below `transport_chart_max_norm` or raises before detached clamp scaling. The new path never treats a transport scale below one as an accepted approximation.

## Serialized configuration migration

A new typed `SerializedConfigMigration` result carries the effective `VFE3Config`, the raw verified mapping, the set of consumed retired keys, and a `legacy_stateful_phi_optimizer` provenance flag. A lower-level migration function returns that result. The existing `config_from_serialized` convenience API continues to return only `.config` for weight-only and visualization consumers, while resume and artifact preflight code call the typed API when provenance matters.

If a serialized payload has no `m_phi_update_mode`, both historical boolean values produce the non-geometric compatibility mode `"adamw"`; `m_phi_natural_grad=True` additionally sets `legacy_stateful_phi_optimizer=True`. Historical `True` is not silently upgraded to the new algorithm because its configuration may violate the new chart, group, or transport-validity requirements. Weight-only consumers can therefore construct the model without enabling the new optimizer path, while a user restarting training from those weights must select `"pullback_group"` explicitly in a current configuration. The retired momentum and update-rule fields are consumed and omitted from the effective schema. If old and new controls are both present, their compatibility and provenance are checked explicitly rather than choosing by field order.

Stored artifact fingerprints are first verified against the raw historical mapping that produced them. Only after that integrity check does semantic migration occur. Strict schema guards admit the three specifically retired keys from this change while continuing to reject genuinely unknown fields; they do not recompute an old digest from the migrated mapping.

Resume configuration migration runs before optimizer-state topology validation so an old stateful phi checkpoint receives the intended incompatibility error. Compatibility is tested separately for legacy `False` plus phi storage, legacy `False` plus `omega_direct`, and legacy `True`. The first retains plain AdamW resume, the second retains the existing mixed omega optimizer and its extra state, and the third follows the rejection contract below.

A legacy `True` checkpoint contains state from the removed heavy-ball or Adam phi algorithm. Exact optimizer resume from such a checkpoint fails with a message that the stateful phi optimizer is incompatible and that training may restart from model weights under the new update. Weight-only evaluation and initialization remain supported. Legacy moment tensors are never silently interpreted as state for the stateless group step.

## Geometry kernel

`vfe3/geometry/phi_preconditioner.py` gains one pure factorwise operation that returns both $v_\phi$ and $\xi$. It refactors the current structure-constant and series machinery so the right differential used by the metric and the left-trivialized differential used by the retraction share one convention and convergence check.

The existing right-differential series is

$$
\Psi_R(A)=\sum_{k=0}^{M-1}\frac{A^k}{(k+1)!}.
$$

The left-trivialized series is

$$
\Psi_L(A)=\sum_{k=0}^{M-1}\frac{(-1)^kA^k}{(k+1)!}.
$$

For per-block direct-sum groups, both operations run on the same local block bases as the current `pullback_metric_per_block`. They must not reconstruct a full structure-constant tensor and thereby undo the per-block memory bound.

The strict M-step helper is a new registered operation. Shared internal algebra utilities may be factored only if the existing E-step `precondition_phi_gradient` dispatcher preserves its current values, warning behavior, damping, and dtype contract exactly. Seeded byte-identity regressions cover every existing E-step preconditioner route touched by the refactor; the new Gram-relative ridge, adaptive series rejection, and regularity gates do not leak into the E-step.

The production path uses fixed, test-pinned numerical constants rather than new experimental knobs: minimum series order $M_{\min}=40$, maximum series order $M_{\max}=128$, relative series-tail tolerance $10^{-12}$, relative Gram damping $\lambda_{\mathrm{rel}}=10^{-6}$, minimum undamped generalized metric eigenvalue $10^{-8}$, maximum accepted damped generalized condition number $10^6$, and scaled linear-solve residual tolerance $10^{-10}$. These constants are implementation semantics and do not become serialized configuration fields in this repair.

The current last-term heuristic is not a sufficient truncation certificate. For $A=\operatorname{ad}_\phi$, the kernel selects whichever of the induced one-norm or infinity-norm is smaller and then uses that selected subordinate, submultiplicative norm consistently for the entire tail calculation. Let its value on $A$ be $\alpha$. At a candidate order $M$, the implementation constructs

$$
t_M=\frac{\alpha^M}{(M+1)!},
\qquad
r_M=\frac{\alpha}{M+2}.
$$

When $r_M<1$, the remaining tail of either differential series is bounded by $t_M/(1-r_M)$. Starting at order 40, the kernel increases the order in increments of eight until this bound is finite and no larger than $10^{-12}$ times the larger of one and the accumulated operator norm. Failure at order 128 rejects the candidate. Float64 augmented-matrix-exponential or autograd Frechet oracles verify this certificate at the chart boundary for symmetric, nonnormal, Jordan-like, and traceless diagonal GL(5) cases.

The undamped metric is compared with the generator Gram matrix through its generalized eigenvalues. A minimum relative eigenvalue below $10^{-8}$ rejects the point as outside the certified regular chart rather than allowing ridge damping to hide a singular `dexp`. Tests include the GL(2) rotation generator at angle $\pi$, whose embedded norm is below five but whose adjoint spectrum reaches the $2\pi i$ singular locus.

The damped system is

$$
A_\lambda=\operatorname{sym}(\mathcal{G}_{\mathrm{pull}})
+10^{-6}B.
$$

It is solved with `cholesky_ex` and `cholesky_solve`; the implementation never forms an inverse. A damped generalized condition number above $10^6$ rejects the candidate because a small backward residual alone cannot certify the direction's forward accuracy. Every accepted solution satisfies

$$
\frac{\lVert A_\lambda v-\bar g\rVert_2}
{\lVert A_\lambda\rVert_2\lVert v\rVert_2+\lVert\bar g\rVert_2}
\le 10^{-10}.
$$

Phi, the processed covector, both differential series, the generalized-eigenvalue check, the damped solve, the trivialized velocity, trust scaling, BCH candidate, and group-product residual validation remain in a float64 autocast-disabled island. The candidate is cast once to the parameter dtype only when the staged update commits. Failure of any series, regularity, factorization, residual, or finiteness check rejects the optimizer step before mutation. A later performance change may replace the certified series with a Frechet-derivative kernel only behind the same oracle tests.

## Optimizer architecture

`GaugeNaturalGradAdamW` is renamed `GaugeManifoldAdamW`. It remains a mixed optimizer: ordinary parameter groups use base AdamW, stored direct-group elements retain their existing `omega_direct` retraction, and phi groups selected by `"pullback_group"` use the new stateless local retraction.

The optimizer stages every candidate phi update before mutating any table. For each selected parameter, it flattens leading dimensions to rows, identifies rows with a nonzero current gradient, computes $v_\phi$ and $\xi$ only for those rows, applies the trust-region scale, forms the BCH candidate, and validates it. Never-active and currently inactive rows remain byte-identical because no optimizer moment can move them.

All phi candidates across token, positional, and independent model-frame groups are validated before any are committed. A nonfinite metric, solve, direction, BCH result, chart-bound violation, or failed group-product residual rejects the complete phi manifold step. The gradient is consumed only when the candidate is committed, preventing base AdamW from applying a second update. Since `pullback_group` and `omega_direct` are mutually exclusive, no optimizer step can partially commit an omega candidate before a later phi failure; the existing omega-only atomic staging remains intact.

No phi optimizer state is created. Weight decay remains zero for the selected group path; any intended frame penalty belongs in the outer objective. Ordinary AdamW groups and `omega_direct` state remain unchanged.

## Local retraction safeguards

The trust-region scale acts on the embedded algebra matrix of the proposed right factor, not merely on an unweighted coordinate norm. This preserves correctness for nonorthonormal registered generator bases.

The candidate chart bound uses the same embedded-matrix Frobenius norm contract as transport and post-M-step projection. The selected phi path enforces that bound inside the optimizer, so `train_step` does not subsequently apply the generic radial projector to the same candidate. The AdamW path retains the existing optional post-step projection behavior.

The order-four BCH residual is a mandatory acceptance condition for `pullback_group`, because a small right factor alone does not bound terms containing high powers of the current nonnormal chart. In the float64 island the optimizer compares the candidate with the exact right group product and requires relative Frobenius residual at most $10^{-6}$. A failing row halves its right-factor scale and retries, up to ten backtracking reductions. If no scale passes, the complete phi manifold step is rejected. The existing general `bch_residual_max` setting remains available to impose a stricter bound; it cannot weaken the mandatory threshold.

## Diagnostics

Existing pullback condition and gradient-angle diagnostics remain available, but their labels are updated to identify a regularized chart direction. The selected path additionally reports the mean and minimum trust-region scale on logged steps, the number of active phi rows, and the maximum accepted embedded chart norm. A recorded transport scale below one is a failure of the new path, not a normal clipping statistic.

Diagnostics remain gated to logging steps and must not add host synchronization to silent training steps.

## Test-driven acceptance

Implementation begins with failing focused tests. The minimum acceptance matrix is as follows.

1. A float64 autograd or augmented-matrix Frechet oracle constructs $J$, $J^TJ$, both trivialized differentials, and the damped chart solve for symmetric, random nonnormal, Jordan-like, and traceless diagonal GL blocks at chart norms 0, 1, 3, and 5. The traceless diagonal GL(5) norm-five fixture must pass the adaptive tail certificate. Relative differential and metric error must be at most $10^{-9}$.
2. The scaled damped-solve residual must be at most $10^{-10}$, the damped generalized condition number must be at most $10^6$, and relative direction error against the oracle must be at most $10^{-8}$. A fixture above the condition bound is rejected rather than accepted from its residual alone.
3. The identity $U\xi=D\exp_\phi[v]$ has relative Frobenius residual at most $10^{-9}$ for random and nonnormal frames. A sign or left-versus-right convention error must fail this test.
4. A noncommuting GL(2) case verifies the exact right-product order, mandatory BCH backtracking, double-precision residual at most $10^{-6}$, and committed-float32 residual at most $5\times10^{-6}$ at current chart norms 3 and 5. The reversed product must fail the oracle comparison.
5. The GL(2) angle-$\pi$ adjoint-singularity fixture is rejected before damping. Separate damping-dominated and near-threshold fixtures report both undamped and damped generalized condition numbers and never emit NaN or infinity.
6. With stochastic behavior, autocast, optional outer regularizers, and gradient clipping disabled, a normalized random-direction central-difference sweep with $h\in\{10^{-2},3\times10^{-3},10^{-3}\}$ shows a stable error plateau and at most $5\times10^{-3}$ relative disagreement between returned cross-entropy and the raw phi gradient. A separate clipping test verifies that the processed covector is passed unchanged to the geometry kernel.
7. Under the phi-coordinate parameterization, the default `"adamw"` configuration returns the existing plain AdamW topology and preserves a seeded one-step result and state layout. The separate `omega_direct` routing remains unchanged.
8. The selected route covers token, learned positional, independent model token, and independent model positional frame tables. Only rows with a nonzero current gradient move. `pos_phi_project_slk=True` and `omega_direct` combinations fail with their specific compatibility messages.
9. No heavy-ball, Adam phi moment, row clock, or other phi-specific optimizer state exists after stepping, saving, or resuming the new route.
10. Oversized gradients are trust-scaled. Every committed factor candidate stays within its effective chart radius, and the generic post-step projector is not called. An in-bound composed forward remains unclamped; an out-of-bound composed token-position chart raises through `transport_chart_max_norm` before the detached hard clamp.
11. A staged failure in a later phi parameter group leaves every earlier phi table unchanged. Nonfinite loss, GradScaler overflow, and skipped steps likewise leave all phi rows unchanged.
12. Float32 model gradients, autocast, gradient accumulation, clipping, and GradScaler compose with the float64 geometry island without a double update or an unscaled covector. Only the final committed chart tensor is cast back to float32.
13. A new stateless pullback-group checkpoint resumes step-exactly. Migrated legacy `False` plus phi storage retains exact AdamW resume; legacy `False` plus `omega_direct` retains exact mixed-optimizer resume and omega extra state; legacy stateful `True` optimizer resume fails with the specified message while weight-only loading succeeds.
14. Raw historical artifact fingerprints verify before semantic migration. Only the three retired phi keys bypass strict unknown-field rejection, and conflicting old/new update controls fail.
15. Configuration tests pin the new registry, removal of both retired runtime fields, the phase-one `glk`/`block_glk` allowlist and dimension ceilings, pullback-mode pairing, closure requirement, trust-radius validation, explicit transport validity bound, and every declared incompatibility. Separate seeded regressions prove that every existing E-step preconditioner path touched by internal factoring remains byte-identical.

The focused numerical regime is the investigated K10 `block_glk` model with two GL(5) blocks. The discarded K20 experiment is not used as empirical justification or as the primary acceptance benchmark. Dimension-generic unit tests may still exercise small representative sizes.

## Performance verification

The new path remains opt-in because a position-dependent pullback solve is computationally expensive. On the RTX 5090, a dedicated benchmark records median and p95 time, peak allocated memory, active-row count, and geometry-kernel time for 128, 512, and 2,048 active rows in the two-GL(5)-block regime.

The comparison uses the same seeded phi rows and covectors for the existing metric-only kernel and the complete metric-plus-trivialization-plus-retraction kernel. Each case uses 20 untimed warmup iterations followed by at least 100 paired, alternating measured iterations on identical inputs. CUDA events delimit only the measured regions, device synchronization occurs outside those regions, and peak memory statistics reset for each case. The machine-readable record includes raw paired samples, bootstrap confidence intervals for median and p95 differences, environment identity, and the configured tolerances.

The additional left-differential, backtracking, and retraction work is reviewed against a 20 percent p95-overhead boundary using the paired confidence interval rather than a single noisy ratio. Exceeding that boundary requires profiling and revision before the path is enabled in a production run. Performance measurements do not become brittle CPU pytest thresholds.

## Expected implementation surface

The implementation is expected to modify `vfe3/config.py`, `vfe3/geometry/phi_preconditioner.py`, `vfe3/gauge_optim.py`, `vfe3/train.py`, `vfe3/model/model.py`, `vfe3/run_artifacts.py`, and the three click-run configuration files. Mandatory migration coverage includes `tests/test_gauge_optim.py`, `tests/test_phi_preconditioner.py`, `tests/test_checkpoint_resume.py`, `tests/test_fp16_gradscaler.py`, `tests/test_exp8_buildout.py`, `tests/test_fix_config_audit.py`, `tests/test_phi_weight_decay.py`, `tests/test_hyperprior.py`, `tests/test_omega_tilde_model_frame.py`, and `tests/test_b5_finite_config_controls_20260716.py`, plus focused artifact and outer-objective tests. The dated `docs/2026-07-17-edits.md` record is updated rather than duplicated.

The implementation does not change the default training objective, the E-step equations, the attention metric, the Gaussian natural-gradient kernels, the `omega_direct` group update, any experiment sweep value unrelated to the retired controls, or the user's current `m_phi_update_mode="adamw"` behavior.

## Terminology

The implementation and documentation use `ridge-regularized Frobenius-pullback direction`, `certified regular chart`, and `local group retraction`. They reserve `Frobenius-pullback Riemannian gradient` for the undamped inverse at a regular point. They do not call the executable ridge direction Gaussian Fisher, predictive Fisher, affine-invariant, bi-invariant, gauge-invariant, or a canonical VFE M-step.

The mathematical contract follows the frame-sector sequence in `Research/manuscripts/PIFB2.tex`, equations `eq:gauge_natural_gradient_def` through `eq:gauge_group_retraction`: solve in the exponential chart, convert through the left-trivialized differential, and multiply the group update on the right.
