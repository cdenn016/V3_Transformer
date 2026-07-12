# Backprop-Free Gauge-VFE Language Model: Revised Investigation and Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` only after the route-selection gate in Phase 1 has passed. Do not
> begin any model-code phase speculatively.

**Date:** 2026-07-11. **Revised:** 2026-07-12.

**Status:** Theory-gated plan. No implementation exists. Phases 0 and 1 are investigations; they
may terminate the project before model code is added.

**Goal:** Determine whether V3 can assign next-token credit without differentiating through the
inference trajectory, and implement that trainer only if its update rule agrees with the settled
cross-entropy gradient under explicit mathematical and numerical gates.

**Architecture:** One application of the free-phase sweep remains byte-identical to deployed V3
inference. The plan separately represents the configured finite production iterate and the
continued-to-equilibrium state required by a fixed-point theorem, then gates their discrepancy.
Predictive credit may proceed through one of two routes: a genuine joint-energy route supporting
equilibrium propagation, or a separately derived fixed-point vector-field route. The existing V3
filtering map is not presumed to satisfy either route. Decode parameters retain direct analytic
cross-entropy updates; all other parameter families remain frozen until their selected credit rule
passes a finite-difference oracle.

**Tech stack:** Python, PyTorch float32, CUDA on the RTX 5090 for scale experiments, analytic
kernels in production, and autograd only in tiny test oracles.

## 1. Revision record and scope

The original plan was produced by a 14-agent adversarial investigation and committed as
`f1674f222cec4df09d86419ad29985d7b81ce402`, with Git author and committer `cdenn016` and
`Claude Fable 5` recorded as co-author. The July 12 review retained its staged empirical method,
analytic decode updates, collapse diagnostics, and frame falsifier, but found that its central
learning rule was not implementation-ready.

This revision corrects six points. First, selectively removing the decode-temperature factor from
the target attraction changes the stated cross-entropy objective. Second, the shipped `mm_exact`
filtering cascade has not been shown to be stationary descent on one joint scalar energy, so the
ordinary scalar equilibrium-propagation contrast does not automatically apply. Third, the
negative nudge is a signed curvature contribution rather than an ordinary convex precision-fusion
pair. Fourth, shared initialization does not by itself prove cancellation of truncation error.
Fifth, the detach-versus-unroll comparison is a calibration measurement rather than an upper
bound. Sixth, VFE2 already contains a substantial `coupled_fep` experiment that directly bears on
the proposed causal diagnosis and must be audited before a new trainer is designed.

The plan does not modify the deployed forward, existing backprop trainer, or pure configuration
path. It does not claim EM monotonicity, predictive-coding equivalence to backpropagation, or an
EqProp gradient theorem unless the corresponding gate below passes.

## 2. Evidence ledger

| Claim | Current status | Required disposition |
|---|---|---|
| Target-blind `pure_fep` omits the through-fixed-point CE term | Code-supported | Retain as a structural diagnosis, not as a complete causal explanation of any PPL plateau |
| An artifact records exactly 25,000 PPL | Unsupported; the original plan itself says none was found | Remove as a factual premise; reconstruct actual runs in Phase 0 |
| The post-fix plateau is caused only by missing through-state credit | Plausible, not isolated | Compare clean-EM, backprop oracle, and prior target-aware coupled paths under matched data and scale |
| Symmetric nudging removes first-order finite-nudge bias | Established for suitable energy-based EqProp systems | Apply only after the joint-energy gate; do not generalize it to arbitrary truncation drift |
| Current V3 `mm_exact` supports scalar EqProp | Unproved | Test integrability and construct an explicit scalar energy or reject this route |
| Selective target scaling remains exact CE | False for `tau_eff != 1` | Use the exact CE derivative and handle scale with preconditioning |
| Negative nudge remains convex precision fusion | False without an additional curvature bound | Prove phase existence or use the one-sided fallback |
| Detach-versus-unroll gap upper-bounds the proposed method | False | Report it only as a matched calibration measurement |
| VFE2 has no prior target-aware local-learning experiment | False | Audit `transformer/vfe/coupled_fep/` and its historical honest, leak, and filter results |
| DFA and predictive-coding PPL values are direct V3 gates | False across differing datasets and tokenizers | Treat literature values as context; use only matched local baselines as gates |

## 3. Non-negotiable observation algebra

For vocabulary energy

$$
E_v(q)=\mathrm{KL}(q\|p_v),
$$

the deployed decode cross-entropy is

$$
C(q,y)=\frac{E_y(q)}{\tau_{\mathrm{eff}}}
 + \log\sum_v \exp\left(-\frac{E_v(q)}{\tau_{\mathrm{eff}}}\right).
$$

Its exact belief derivative is

$$
\nabla_q C
=\frac{1}{\tau_{\mathrm{eff}}}
 \left(\nabla_q E_y-\sum_v P_v\nabla_q E_v\right),
\qquad
P_v=\mathrm{softmax}_v\left(-E_v/\tau_{\mathrm{eff}}\right).
$$

The target attraction and softmax repulsion therefore carry the same
`lambda_obs / tau_eff` coefficient. Production code must not remove the temperature factor from
only one term. Dimension and token-frequency scaling are handled by a belief-space natural metric,
per-family preconditioner, trust region, or step size without changing this derivative.

The nudged objective, where an energy route exists, is exactly

$$
E_{\lambda}(z,\theta)
=E_{\mathrm{joint}}(z,\theta)+\lambda C(z,y;\theta).
$$

`E_joint` is the same scalar whose gradient generates the certified free dynamics and whose partial
derivatives enter every Route-E contrast. A different surrogate may not generate the phase states
or the reported statistics.

At `lambda < 0`, the CE curvature is signed. In the metric used by Route E, define the normalized
curvature margin

$$
m_{\lambda}
=\frac{\lambda_{\min}(M^{-1/2}\nabla_z^2E_{\lambda}M^{-1/2})}
       {\max(\|M^{-1/2}\nabla_z^2E_{\lambda}M^{-1/2}\|_2,1)}.
$$

A negative phase is admissible only if `m_lambda >= 1e-6` in the float64 oracle and the production
float32 conservative bound is at least `1e-4`, or if an independently certified update-map bound
has spectral radius `rho(DT_lambda) <= 0.99`. Covariance eigenvalues must also remain above the
configured floor. These are acceptance inequalities, not telemetry. If that gate fails while both
positive phases remain admissible, then for a family with
`partial C / partial theta = 0` define

$$
h_{\lambda}
=\frac{
 \partial_{\theta}E_{\mathrm{joint}}(z_{+\lambda},\theta)
 -\partial_{\theta}E_{\mathrm{joint}}(z_0,\theta)
}{\lambda},
\qquad
\widehat g_{\mathrm{one}}=2h_{\lambda/2}-h_{\lambda}.
$$

This one-sided Richardson estimator is the only permitted signed-phase fallback. It requires its
own phase-existence and finite-difference convergence gate. If either positive phase fails, the
M-step is skipped; the code may not silently fall back to unextrapolated nudged statistics. A
family that appears directly in `C` also receives the analytic direct term at `z_0`, as specified
under Route E.

## 4. Route-selection theorem gate

Let `z` contain every live belief state at every layer, and let

$$
R(z,\theta)=z-T(z,\theta)
$$

be the residual of one exact production inference sweep, including masks, detached keys,
attention recomputation, clamps, damping, and cross-layer prior handoff. Phase 1 must select one of
the following outcomes.

### Route E: joint energy and equilibrium propagation

Route E is the preferred strict backprop-free route. It requires an explicit scalar
`E_joint(z, theta)` whose stationary equations match the proposed free and nudged updates. Let
`z_deploy` be the state after the configured finite number of production sweeps and `z_0` the state
obtained by continuing the same map to the registered stationarity tolerance. These are different
objects unless a parity gate proves otherwise. At tiny dimensions, either the residual itself or
the covector obtained through an explicitly defined
positive-definite mobility or metric must have a symmetric state Jacobian to numerical tolerance
away from nondifferentiable boundaries. The same scalar energy must decrease under each accepted
update, and the settled symmetric contrast must converge to the finite-difference derivative of
settled CE as `lambda -> 0`.

For a family with no direct appearance in `C`, passing Route E permits the through-state contrast

$$
\widehat g_{\theta}
=\frac{1}{2\lambda}
 \left[
 \frac{\partial E_{\mathrm{joint}}}{\partial\theta}(z_{+\lambda},\theta)
 -\frac{\partial E_{\mathrm{joint}}}{\partial\theta}(z_{-\lambda},\theta)
 \right].
$$

For every parameter that appears directly in `C`, including decode-bank parameters and the decode
scale, the total update also includes the analytic direct term
`partial C(z_0, theta) / partial theta`. Attention temperatures that affect CE only through the
settled state have no such direct term. Each family’s oracle must test the same decomposition used
in production.

Failure means this formula is prohibited on the current dynamics. A new energy-derived inference
variant may be proposed, but it must remain an opt-in registry variant and preserve the deployed
pure path.

### Route V: explicit vector-field fixed-point rule

If the residual is not integrable, the plan may derive a rule for the actual vector field. The exact
settled CE derivative satisfies

$$
R_z^T v=C_z^T,
\qquad
\frac{dC}{d\theta}=C_{\theta}-v^T R_{\theta}.
$$

This route may use analytic Jacobian-vector operators and an iterative linear solve, but no autograd
or retained backward graph in production. Because the transpose solve is an implicit credit
operator, it must be reported as fixed-point implicit differentiation rather than ordinary EqProp.
A modified nonconservative-EP rule is acceptable only with its own derivation and finite-difference
oracle. Whether this interpretation satisfies the project's stricter meaning of “backprop-free” is
a user decision at the Phase 1 gate.

### Stop outcome

If neither route agrees with finite differences, non-decode families remain frozen. The project may
continue only as the readout-only M0 control. It must not ship a single-phase nudged-statistics rule
under an EqProp or exact-gradient label.

## 5. Planned file boundaries

The following boundaries apply only after their owning phase passes.

| Path | Responsibility |
|---|---|
| `vfe3/fep/residual.py` | Pack the multilayer state and evaluate the exact production-sweep residual `R(z, theta)` |
| `vfe3/fep/observe.py` | Exact decode CE, belief derivatives, direct decode-row derivatives, and scale derivatives |
| `vfe3/fep/energy.py` | Route-E joint energy and analytic partial derivatives; absent if Route E fails |
| `vfe3/fep/vector_field.py` | Route-V analytic Jacobian-vector operators and solver; absent if Route V is rejected |
| `vfe3/fep/phases.py` | Free/nudged settling, shared-state bookkeeping, residual gates, and phase-existence checks |
| `vfe3/fep/stats.py` | Streamed sufficient statistics and uncertainty estimates without parameter mutation |
| `vfe3/fep/rules.py` | Registered, family-specific updates after oracle approval |
| `vfe3/fep/trainer.py` | Backprop-free training loop with no `.backward()`, optimizer, scaler, or `.grad` reads |
| `train_fep.py` | Root click-to-run config and entry point, parallel to `train_vfe3.py` |
| `tests/fep/` | Algebra, residual, route-selection, phase, update, and end-to-end oracle tests |

Contrary to the original plan, this work cannot be expressed with zero call-site edits. Config
validation, E-step dispatch, multilayer capture, phase resweeps, checkpoint state, run provenance,
and the training entry point all require explicit integration. Every edit must preserve existing
defaults and registry-selected pure paths.

## 6. Phase 0: reconstruct the evidence before designing the cure

**Files:** Create the tracked directory
`docs/investigations/backprop-free-fep-phase0-2026-07-12/` containing `report.md`, `manifest.json`,
`metrics.jsonl`, and one exact config JSON per experimental arm. Raw checkpoints and transient logs
live under the valid Windows naming convention
`C:\tmp\vfe3-fep-phase0-20260712-{short_sha}`, with `{short_sha}` replaced by the source commit’s
seven hexadecimal characters when the phase starts. Record their hashes and extracted metrics in
the tracked manifest, then delete those task-owned temporary files before the phase is complete.

1. Record the exact Git commits, configs, tokenizer, dataset split, seeds, parameter counts, and
   reachable training modes for VFE2 `pure_fep`, VFE2 `coupled_fep`, and V3.
2. Recover the VFE2 `coupled_fep` design and audit history. Distinguish `honest`, the target-leaking
   negative control, and `filter`, which uses present-token reconstruction plus learned
   roll-forward. Do not infer a result from a design document when no run artifact exists.
3. Reproduce tiny matched runs for reachable modes: VFE2 clean-EM, VFE2 backprop oracle, VFE2
   coupled honest, VFE2 coupled filter, the matched VFE2 coupled leak negative control, V3 detached
   inference, and V3 unrolled/backprop estimator. The leak mode is diagnostic only and cannot
   qualify as a language-model result.
4. Compute unigram, bigram, and 5-gram baselines on the identical V3 BPE token stream. Retain the
   published word-level 152.7 value only as historical context.
5. Report actual train, validation, and test CE/PPL with artifact paths. Do not repeat the exact
   25,000-PPL narrative unless a machine-readable artifact establishes it.
6. Measure the detached-versus-unrolled difference in CE, per-family gradient cosine, gradient norm,
   and depth attenuation. Label it “calibration,” not “upper bound.”

**Gate P0:** A reviewer can reproduce every reported number from a named config and artifact. If
the prior target-aware paths already falsify the structural diagnosis, revise the model hypothesis
before Phase 1.

## 7. Phase 1: exact algebra, residual map, and route selection

### Task 1.1: Pin the observation derivatives

**Files:** Create `vfe3/fep/observe.py` and
`tests/fep/test_observation_gradients.py`.

The tests must compare analytic derivatives of CE with respect to `mu_q`, `sigma_q`, decode means,
decode log-variances, and inverse temperature against an autograd oracle at `K=2..4`. Include
`tau_eff != 1`, saturated-but-unclamped logits, repeated vocabulary rows, and ignored targets. Add
a regression asserting that selectively dropping `1 / tau_eff` from target attraction fails.

Run:

```powershell
python -m pytest tests/fep/test_observation_gradients.py --junitxml=C:\tmp\vfe3-fep-observe.xml
```

**Gate 1.1:** Zero failures and errors in JUnit XML; relative error below `1e-5` in float64 oracle
tests and below `2e-4` in float32 tests.

### Task 1.2: Define the production residual

**Files:** Create `vfe3/fep/residual.py`, `tests/fep/test_residual_map.py`, and
`tests/fep/test_train_deploy_equilibrium_gap.py`; modify capture seams in `vfe3/model/model.py` and
`vfe3/model/stack.py` only as required to expose state without changing default values.

The residual must reproduce one production sweep exactly, including frozen-key filtering,
strict-pair masks, attention recomputation, damping, clamps, and layer handoffs. A byte-parity test
must compare one residual-map sweep against one production sweep for every supported `mm_exact`
configuration selected for this project. Separately compute `z_deploy` at the configured finite
iteration budget and `z_0` after continued settling. Record their per-token belief KL, CE gap, and
stationarity residual.

**Gate 1.2:** Either deployment uses the same residual-based settling rule as training, behind an
opt-in FEP deployment mode, or the finite deployment state satisfies mean belief
`KL(z_deploy || z_0) <= 1e-3` and absolute CE gap `<= 0.01` nats per token. Failure means the
settled objective is a different model and blocks Route E for finite-step deployment.

### Task 1.3: Test integrability rather than assuming it

**Files:** Create `tests/fep/test_integrability_gate.py` and
`docs/investigations/backprop-free-fep-route-gate-2026-07-12.md`.

At differentiable tiny states, define `G(z) = M(z)^{-1} R(z)`, where `M` is the explicitly proposed
positive-definite mobility; use `M = I` if no mobility is proposed. Compute the full Jacobian of
`G` with the test-only autograd oracle and report

$$
\epsilon_{\mathrm{curl}}
=\frac{\|G_z-G_z^T\|_F}{\max(\|G_z\|_F,10^{-12})}.
$$

Test the complete multilayer map as well as isolated self, pair, attention, and prior-handoff
blocks. Masks and clamps must be held in a locally constant regime. A large antisymmetric component
rejects ordinary scalar EqProp for that map.

**Gate 1.3:** Route E requires `epsilon_curl <= 1e-6` in float64, a positive-definite `M` when one
is used, and an explicitly evaluated joint energy whose gradient is `G`. Otherwise select Route V
or stop.

### Task 1.4E: Validate Route E

**Files:** Create `vfe3/fep/energy.py`, `tests/fep/test_energy_descent.py`,
`tests/fep/test_energy_contrast_oracle.py`, and `tests/fep/test_nudge_stability.py`.

Pin analytic energy partials against autograd, require accepted state updates to lower the same
energy, and compare the symmetric contrast to central finite differences of CE evaluated at the
same settled state `z_0`. For the negative phase, report the smallest local curvature eigenvalue and
demonstrate `m_lambda >= 1e-6` in the float64 oracle, the registered production bound of at least
`1e-4` or `rho(DT_lambda) <= 0.99`, and covariance positivity throughout settling. The same test
file must exercise the one-sided fallback by choosing a case where the negative phase is rejected
while the `lambda` and `lambda/2` positive phases pass, then compare
`2 h_(lambda/2) - h_lambda` against the same settled-CE finite difference.

**Gate 1.4E:** For symmetric nudging, as `lambda` is halved over at least three values, gradient
error decreases at the expected second-order rate until numerical error dominates; cosine exceeds
`0.99`; relative norm error is below `0.05`; every phase meets its residual and positivity
thresholds. The one-sided Richardson arm must independently show second-order convergence, cosine
above `0.99`, and relative norm error below `0.05`. Otherwise no fallback is registered.

### Task 1.4V: Validate Route V

**Files:** Create `vfe3/fep/vector_field.py`, `tests/fep/test_vector_field_oracle.py`, and
`tests/fep/test_adjoint_solver.py`.

Pin analytic `R_z^T v` and `R_theta^T v` products against test-only autograd, then compare the full
implicit derivative against central finite differences of settled CE. Record solver residual,
iteration count, conditioning estimate, and memory. This task is an alternative to Task 1.4E, not a
second estimator to blend with it.

**Gate 1.4V:** Gradient cosine exceeds `0.99`, relative norm error is below `0.05`, and the linear
solver reaches relative residual below `1e-5` without retained trajectory storage. User approval is
required before treating this route as satisfying the project’s backprop-free objective.

## 8. Phase 2: readout-only control

**Files:** Create `vfe3/fep/stats.py`, the decode-only portion of `vfe3/fep/rules.py`,
`vfe3/fep/trainer.py`, `train_fep.py`, `tests/fep/test_decode_rules.py`, and
`tests/fep/test_train_fep_readout.py`.

Train only the untied decode bank and inverse temperature from free-phase statistics. Use per-row
damped Gauss-Newton or another oracle-pinned preconditioner, per-row count gates, and trust regions.
Encode, frame, positional, attention-prior, and temperature families not explicitly covered by the
decode rule remain frozen. The trainer must contain no `.backward()`, optimizer, scaler, or `.grad`
read.

**Gate M0:** Beat the measured BPE unigram and bigram controls under matched tokens and budget.
Failure after oracle parity rejects the proposed readout machinery; it does not by itself prove that
streamed statistics are mathematically wrong.

## 9. Phase 3: selected fixed-point credit rule

**Files:** Create `vfe3/fep/phases.py`; extend `vfe3/fep/rules.py`; modify `vfe3/config.py`,
`vfe3/inference/e_step.py`, `vfe3/model/block.py`, `vfe3/model/stack.py`, `vfe3/run_artifacts.py`,
`vfe3/fep/trainer.py`, and `train_fep.py`; create `tests/fep/test_phase_parity.py`,
`tests/fep/test_phase_stability.py`, `tests/fep/test_fep_checkpoint.py`, and one family-specific
oracle test per enabled update.

Implement only the route selected in Phase 1. Every family begins disabled and is enabled after its
own settled-CE finite-difference test passes. Route E uses exact CE nudging with common
`lambda_obs / tau_eff` scaling; Route V uses only the approved residual rule. If a negative phase
fails its curvature, contraction, or covariance gate, use the registered one-sided Richardson rule
only when both positive phases pass their gates. If they do not, skip the M-step and record the
reason.

Stream estimates with uncertainty and stepwise decay only after per-minibatch oracle agreement has
been established. A high cosine between two biased estimates is not sufficient; finite-difference
agreement remains binding.

**Gate M1-credit:** On a tiny language model, enabled encode updates agree with settled-CE finite
differences and improve held-out CE relative to the frozen-encode M0 control across at least three
seeds. No absolute PPL target substitutes for this causal comparison.

## 10. Phase 4: single-layer geometry and frame falsification

Enable encode tables, `phi`, positional frames, T5 prior tables, and temperature rules one family at
a time. Frame updates use the shipped preconditioner only as a preconditioner; they do not claim it
is the full `GL(K)` metric. Procrustes remains opt-in and requires the explicitly tested
per-block-isotropic covariance premise.

**Gate M1-LM:** Beat the locally trained 5-gram baseline on the identical BPE stream within the
pre-registered budget. **Gate M2-phi:** learned-frame versus frozen-random-frame runs, with identical
decode and seeds, must improve test PPL by at least 10 percent. Otherwise report frame learning as
dead weight for this configuration and retain the honest frozen-frame path.

## 11. Phase 5: depth

Depth begins only after the single-layer estimator passes. Specify which layer contributes to each
shared table, expose the rule in config, and test early-versus-late row conflict. Observation terms
are injected only where the selected route’s joint objective or residual derivation permits them.
Top-down resweeps are counted explicitly in compute and may not be described as ordinary forwards.

**Gate M3:** At matched width, data, seeds, and total compute, four layers improve test PPL by at
least 15 percent over one layer. Failure first triggers a measured sweep over settling tolerance and
resweep count. Persistent failure rejects the depth mechanism rather than merely raising the budget.

## 12. Phase 6: WikiText-103 scale

Run `K=32..64` on the RTX 5090 only after all prior gates. Compare against matched-token, matched-K,
matched-data local backprop V3, readout-only M0, and frozen-frame controls. DFA and predictive-coding
papers remain literature context, not numeric acceptance bands.

Report actual cost as

$$
T_{\mathrm{free}}+T_{+}+T_{-}+T_{\mathrm{resweep}}+T_{\mathrm{solve}},
$$

or the corresponding one-sided form. Do not summarize this as “two to three forwards” unless a
profiler establishes that ratio against the exact matched V3 training step.

**Gate M4:** The selected route improves test PPL over M0 and remains within two times the matched
backprop V3 PPL across three seeds, without violating residual, covariance, saturation, or update-SNR
gates. Absolute literature PPL is reported separately.

## 13. Diagnostics and abort conditions

Every run records phase residuals, accepted and skipped M-steps, signed-phase existence checks,
covariance minima, clamp and saturation counts, free-to-nudged belief KL by layer, estimator error
against periodic tiny-oracle probes, per-family update norm and SNR, bank dispersion and effective
rank, decode logit spread, and train/validation/test CE and PPL.

Abort rather than silently freeze when the bank collapses, a covariance becomes nonpositive, a phase
cannot meet its residual threshold, an enabled family remains below the pre-registered SNR floor, or
the estimator repeatedly disagrees with finite differences. A fallback that changes the objective or
credit rule creates a new named experimental arm and requires a new gate.

## 14. Verification and completion requirements

Each implementation task follows test-first development. Focused commands omit extra `-q` because
`pyproject.toml` already supplies it. Pass counts come from JUnit XML. Before any phase is called
complete, run its focused tests, the relevant existing regression surface, `git diff --check`, the
plan’s placeholder and banned-language scan, and a staged-diff review. Full-suite verification is
reserved for phases that change production code.

The final implementation, if reached, must preserve a mathematically pure default path, use float32
in production, keep extreme computation opt-in, update the dated post-edit document, and complete
the repository’s branch, push, merge, and cleanup lifecycle.

## 15. Literature calibration

The applicable primary references are Scellier and Bengio (2017) for energy-based equilibrium
propagation; Laborieux et al. (2021) for symmetric finite-nudge bias reduction under EqProp
assumptions; Scellier et al. (2018) for vector-field generalization and its symmetry-dependent
gradient error; Millidge, Tschantz, and Buckley (2020) for predictive coding on translated
computation graphs; Launay et al. (2020) for DFA on transformer language modeling; and Pinchetti et
al. (2022) for predictive-coding training of transformers on conditional language models.

These papers do not establish the July 11 rule for V3’s detached filtering cascade. Claims about the
absence of any published backprop-free transformer language model must be scoped to the precise
autoregressive, from-scratch, dataset, and tokenizer setting rather than stated absolutely.

## 16. Wiki and manuscript handling

The research wiki currently describes the superseded scalar-contrast proposal. After this revised
plan is reviewed and only with user confirmation, update `[[Nudged two-phase EM]]`, the associated
2026-07-11 run source, and the VFE Transformer Program page. Manuscript claims wait for Phase 1
route selection and may describe no gradient equivalence before the corresponding proof and oracle
gates pass.
