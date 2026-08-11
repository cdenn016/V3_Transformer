# V3 Transformer three-wave deep audit

**Audit date:** 2026-08-10–11
**Pinned source:** `b033a19d4a75630a88924e57efc25581364fbc6e` (`origin/main` at audit start)
**Audit worktree:** `codex/deep-audit-full-gaussian-20260810`
**Primary focus:** active and supported `gaussian_full` paths, with the current diagonal-Gaussian
51.886 test-PPL artifact used only as unmatched historical context.

> This is an evidence-ranked audit, not an automatic remediation. The user's live WIP and active
> CUDA training process were not modified. GPU tests and benchmarks were withheld while that job
> remained resident.

## Executive summary

Three waves of ten investigator lanes produced 59 carried candidates. Strict combined adjudication
closed **24 findings as EVIDENCE_VERIFIED** (2 high, 20 medium, 2 low), **refuted 1**, and retained
**34 as INCONCLUSIVE** rather than promoting source-only analysis. There are no critical findings.

The two high findings form one provenance-integrity chain: a non-strict process can relabel imported
code with the current disk identity, and the multi-seed publisher can then admit a drifted requested
run into a complete aggregate. The most important full-Gaussian correctness items are medium: a
silent fp32 spectral-exponential Jacobian defect on the active AIRM path, nonfinite/failed-SPD
accounting seams, and a CUDA fused-Adam resume path that fails before restoration.

The confirmed 51.886 diagonal result is not a controlled comparator for the retained full-Gaussian
results. Besides K=300/460k/model-channel versus K=20/15k/token-bank confounds, it uses stride-16
overlapping evaluation while the full runs use stride 128. Both score every transition once, but with
very different context distributions. No same-checkpoint stride effect or replicated family effect
was measured. The active K32 full-Gaussian run remained incomplete and is not reported as an endpoint.

For an MLP, the exact Transformer-faithful seam is inside each VFE block after the converged E-step,
optional W_O-like mixing/CG, and current attention-side norm, followed by an MLP residual and a new
FFN-side norm before the next-layer prior handoff. This is an explicit hybrid augmentation, not a
term derived by the current canonical free energy. A canonical-frame full-moment construction and an
in-E-step unary-energy alternative are specified below.

## Ranked findings

The ordering below combines severity, active/observed reachability, scientific-integrity impact, and
mechanical evidence. `Supported optional` means the configuration is valid but not used by the active
K32 run; `conditional` means the path is active only after the named numerical event.

| Rank | Severity / ID | Reachability | Evidence-backed finding | Required fix |
|---:|---|---|---|---|
| 1 | **HIGH — W1-PROCESS-IDENTITY** | Active when source changes during a non-strict process | `_verified_process_code_identity` replaces the import-time generation with the current disk digest, allowing mixed-generation output to be labeled as one current source (`vfe3/run_artifacts.py:2296-2314`; probe). | Preserve and publish import-time and current identities separately; fail closed on mismatch. |
| 2 | **HIGH — W1-MULTISEED-DRIFT** | Aggregate publication; real drifted artifact reproduced | The provenance join omits drift status and can publish a complete requested-seed aggregate from a source-drifted run (`multiseed_analysis.py:271-322`; end-to-end probe). | Bind each seed's start/end source identity and withhold the whole aggregate on drift. |
| 3 | **MEDIUM — W3-SPECTRAL-EXP-GRAD** | Active AIRM path; near-degenerate conditional | In fp32, distinct eigenvalues can have equal rounded exponentials, selecting a zero divided difference instead of the Fréchet limit and silently deleting off-diagonal cotangents (`vfe3/geometry/retraction.py:226-250`). | Use a value-aware repeated branch/stable `expm1` or higher-precision divided difference; retain the Jacobian probe. |
| 4 | **MEDIUM — W1-DECODE-NONFINITE** | Active full-chunked path; nonfinite-input conditional | A single NaN/Inf covariance can poison the streamed scalar CE rather than being excluded row-locally (probe; `vfe3/model/prior_bank.py:1790-1933`). | Validate/mask nonfinite query invariants before reductions and publish the excluded count. |
| 5 | **MEDIUM — W2-CUDA-RESUME** | Supported active CUDA/fused-Adam recovery | Safe load maps the bundle to CPU, then preflight rejects fused Adam's CPU `step` tensor against live CUDA parameters before `load_state_dict` can transfer it (`vfe3/run_artifacts.py:1120-1137,1838-1850,1972-2029`). | Validate portable scalar state first, load/transfer, then validate realized devices. |
| 6 | **MEDIUM — W2-OPT-NAMES** | Resume after permitted config/source reorder | Optimizer moments are bound by integer position, not parameter name; equal-shaped reordering silently cross-loads moments (mechanical reorder probe; `vfe3/run_artifacts.py:1081-1170`). | Persist canonical parameter names/shapes per slot and reject any nonidentical mapping. |
| 7 | **MEDIUM — W3-EXCLUDED-TOKEN-DENOM** | Active full decode; total-Cholesky-failure conditional | Decoder-excluded non-PD positions are counted again by evaluation/accumulation denominators; an all-failed validation batch reports CE 0/PPL 1 (`prior_bank.py:1831,1894,1925-1933`; `train.py:664-692,950-976`). | Return the actual scored-token count and make zero-scored evaluation undefined/failing. |
| 8 | **MEDIUM — W3-MM-PARTIAL-CHOLESKY** | Supported optional `gaussian_full + mm_exact` | MM fusion discards `safe_cholesky` masks and consumes failed partial factors; an `eta=1` fixture silently returns identity instead of the target (`vfe3/gradients/kernels.py:930-939,1114-1123`). | Honor every success mask and explicitly retain/fail/count invalid rows before any solve. |
| 9 | **MEDIUM — W3-EMISSION-TRACE** | Supported optional full-Gaussian emission | A diagonal precision vector broadcasts over the dense covariance and counts off-diagonal entries; probe result 4.375 versus correct 4.0 (`vfe3/inference/e_step.py:697-704`). | Contract with `diag(Sigma)` or `diag_embed(precision)` consistently in value and gradient paths. |
| 10 | **MEDIUM — W2-LOGEUC-NONFINITE** | Supported optional log-Euclidean retraction | NaN/Inf tangent input collapses to `eps I` without incrementing the nonfinite counter (probe; `vfe3/geometry/retraction.py:785-807`). | Fail/retain per row or count the fallback explicitly; never convert an unobserved failure into a plausible covariance. |
| 11 | **MEDIUM — W3-LOGEUC-REPEATED-GRAD** | Supported optional log-Euclidean retraction | At a repeated spectrum the base-point backward zeros off-diagonal gradients even when the forward map is identity (`vfe3/geometry/retraction.py:115-121,785-807`). | Use the within-eigenspace Fréchet limit instead of zero. |
| 12 | **MEDIUM — W1-GLOBAL-POLICY** | Multi-model same-process conditional | Constructing a second model changes process-global full-covariance precision policy used by the first model (mechanical cross-model probe). | Make the policy immutable per model/call and pass it explicitly to numerical kernels. |
| 13 | **MEDIUM — W1-PARAM-GRID** | Checked-in parameter-match sweep | The grid retains only K=60 within 2% and raises because two widths are required; this is the sole failure in the 253-test CPU lane (`ablation.py:2042-2120`). | Restore two admissible realized widths or revise the target/tolerance with a documented design. |
| 14 | **MEDIUM — W2-MULTISEED-EXTRAS** | Multi-seed publication | Requested-seed joining silently ignores unexpected observed runs and exposes no contradictory-cohort field (`multiseed_analysis.py:425-497`; probe). | Reject or explicitly publish unexpected/duplicate seeds and derive all top-level counts from the accepted cohort. |
| 15 | **MEDIUM — W2-UNPAIRED-PARETO** | Validation/scaling reports | Validation PPL and wall time are filtered independently and averaged over different seed sets (`vfe3/viz/sweep_adapters.py:66-89`; probe). | Inner-join by seed and report both paired and missing counts. |
| 16 | **MEDIUM — W3-EVAL-STRIDE-COMPARISON** | Observed artifact protocol mismatch | The 51.886 run uses stride 16 (mean context 120.474), while full runs use stride 128 (64.494); both score 280,541 transitions once, but are different conditioned metrics (`vfe3/data/datasets.py:573-626`; `vfe3/train.py:947-976`). | Re-evaluate selected checkpoints under one declared stride; report the other protocol separately. |
| 17 | **MEDIUM — W3-REJECTED-UPDATES** | Observed completed/full-active runs | Two K20 full runs rejected 18.53% and 25.59% of optimizer attempts; the guard is safe, but configured steps are unequal accepted budgets (`vfe3/train.py:789-833`; artifacts). | Persist live cumulative counts and compare arms by accepted updates with a preregistered stability threshold. |
| 18 | **MEDIUM — W3-SHARP-ROWS** | Observed in 4/5 completed full runs plus active K32 | Both heads trip the minimum-row entropy alarm; the additive run is a counterexample. This proves sharp rows, not whole-head collapse (`vfe3/metrics.py:840-875`; artifacts). | Persist row quantiles/fractions and test paired relationships with CE and effective rank. |
| 19 | **MEDIUM — W3-FLOP-COV-BLIND** | Active compute reporting | Matched K20 diagonal/full artifacts receive identical analytic FLOPs despite a recorded 3.674x wall-time ratio; the proxy has no covariance-family term (`vfe3/run_artifacts.py:2762-2783`). | Add full-covariance decoder/retraction terms and use profiled FLOPs plus wall time for compute claims. |
| 20 | **MEDIUM — W3-START-PROVENANCE** | Active long/incomplete runs | Start Git/code/data-order identity is held in memory until finalization; a crash can leave config/metrics/best weights without recoverable start provenance (`vfe3/run_artifacts.py:1354-1375,1439-1445,3005-3016`). | Atomically write an immutable start manifest before the first optimizer step. |
| 21 | **MEDIUM — W3-READOUT-BLINDNESS** | Active full-chunked decoder | With diagonal vocabulary priors and zero z-loss, class differences read `mu` and `diag(Sigma)` only; `-logdet(Sigma)` is class-common and cancels. Correlations remain only indirectly useful through inference (derivation D1; probe). | Document the limit or add class-dependent full-covariance/readout structure that can directly use correlations. |
| 22 | **MEDIUM — W3-BLOCK-SPD** | Active K32/H2 block-GL route | The executed covariance remains in `SPD(16) × SPD(16)` (272 degrees of freedom), not unrestricted `SPD(32)` (528); 256 cross-head degrees are absent (derivation D2; source trace). | Report the product manifold explicitly or extend energies/transport/readout to model cross-head covariance. |
| 23 | **LOW — W1-NATURAL-DTYPE** | Direct mixed-dtype API use | `FullGaussian.natural()` fails when mean and covariance dtypes differ (probe). | Promote/cast both operands under one explicit precision policy. |
| 24 | **LOW — W2-NEGATIVE-MOMENT** | Corrupt/adversarial checkpoint | Checkpoint preflight accepts finite negative Adam second moments (probe; `vfe3/run_artifacts.py:1010-1149`). | Require elementwise nonnegative `exp_avg_sq`/`max_exp_avg_sq` before loading. |

## Rejected and downgraded candidates

`W2-CHUNK-GATE` is **REFUTED**. Current code budgets whole vocabulary size `V`, and targeted tests
prove fp64 whole-vocabulary sizing plus chunk-size invariance
(`vfe3/model/prior_bank.py:2241-2259`; `tests/test_family_chunked_workspace_20260807.py:274-300,
353-371`). Preserve that regression coverage.

Thirty-four candidates remain **INCONCLUSIVE**, not findings. They retain useful remediation/test
ideas, but lack current domain-eligible evidence for the full runtime claim:

`W1-BEST-PUBLISH`, `W1-RELATIVE-JITTER`, `W1-VIZ-AXES`, `W1-ALPHA-MASK`, `W1-BREGMAN-DUAL`,
`W1-SPD-PERF`; `W2-BEST-BINDING`, `W2-RUNDIR-REUSE`, `W2-RESUME-ACCOUNTING`,
`W2-LUCKIEST-SEED`, `W2-ROUTE-COLLAPSE`, `W2-BOOTSTRAP-PAIRING`, `W2-FOREST-CHECKPOINT`,
`W2-FINALIZE-DECODE`, `W2-GUARD-DENOM`, `W2-CHANNEL-SPD-LOG`, `W2-STALE-REPORT`,
`W2-T5-ORPHAN`, `W2-BANNER-INERT`, `W2-PLAIN-ITERABLE-SELECTION`; `W3-LOGEUC-PURITY`,
`W3-RIGHT-ROPE-PURITY`, `W3-JITTER-GAUGE`, `W3-GAUGE-FIGURE`, `W3-EMISSION-BATCH`,
`W3-ENTROPY-FLOOR`, `W3-MULTILAYER-PRIOR-DIAG`, `W3-HYBRID-F-TOTAL`,
`W3-ADAPTIVE-TAU-DIAG`, `W3-PURITY-READOUT`, `W3-TEST-EXHAUSTION`,
`W3-COV-SWEEP-OBJECTIVE`, `W3-TOKEN-BOOTSTRAP-DEPENDENCE`, and `W3-MEAN-PPL-RANK`.

Notable severity corrections after challenge:

- CUDA resume is medium: opt-in and fail-before-mutation, despite blocking recovery.
- The active spectral-exponential defect is medium: occurrence frequency/impact is not measured.
- Evaluation stride is medium: the conditioning mismatch is large, but no individual endpoint is
  invalid and no same-checkpoint CE delta was measured.
- Repeated exposure to one test digest is a medium **inconclusive** confirmatory-status risk, not
  evidence of test-driven selection. The code selects checkpoints by validation and PPL renaming is
  automatic.

Policy-accepted or clean candidates included fp32 escalation policy, hard KL/SPD caps,
filtering-versus-smoothing, active RoPE insertion, model-channel marginal projection, and the generic
editable offline config loader. No current evidence justified relitigating those contracts.

## Full-Gaussian experiment evidence

The retained artifacts do **not** support a causal diagonal-versus-full conclusion. The headline
51.886 diagonal endpoint is authentic, but it differs from the retained full-Gaussian endpoints in
model width, training budget, prior route, source revision, and evaluation protocol.

| Representative run | Family / width | Endpoint | Eval stride | Accepted / attempted | Source identity |
|---|---|---:|---:|---:|---|
| `data/51.89...` | diagonal K300 | test PPL 51.886 | 16 | unavailable / 460k | dirty-stable `47d75a8` |
| `data/138.35_diag-optimized` | diagonal K20, model-channel | test 138.353 | 128 | unavailable / 15k | dirty-stable `47d75a8` |
| `full_gaussian_N=128/149.50...` | diagonal K20, token bank | test 149.503 | 128 | 15,000 / 15,000 | dirty-stable `f489fe7` |
| `full_gaussian_N=128/132.78...` | full K20, token bank | test 132.777 | 128 | 12,221 / 15,000 | dirty-stable `f489fe7` |
| `full_gaussian_N=128/132.70...` | full K20 + mixer | test 132.698 | 128 | 11,161 / 15,000 | **drifted** `f489fe7` |
| `141.20_ropefull-rope-on-value` | full K20 | test 141.198 | 128 | 15,000 / 15,000 | dirty-stable `b033a19` |
| active `20260810-221345...` | full K32 | val 122.069 at step 9,000 | 128 | incomplete | no terminal provenance |

The most important comparison defect is `eval_stride`. The 51.886 run has 15,307 overlapping
validation windows and masks the first 112 targets in each window, so almost every scored transition
has 112–127 tokens of context. The stride-128 runs have 1,915 disjoint windows and repeatedly score
context lengths 0–127. They use the same raw split hashes but estimate different context-conditioned
perplexities. The individual numbers remain valid under their own protocols; comparing 51.886
directly with 132–141 does not.

Full-Gaussian stability also differs materially across retained runs. The 132.78 and 132.70 K20 runs
rejected 2,779/15,000 (18.53%) and 3,839/15,000 (25.59%) optimizer attempts. The guard correctly kept
nonfinite updates out of AdamW, but configured step counts are not equal accepted-update budgets.
Four of five inspected completed full-Gaussian configurations, plus the active K32 run, also triggered
the minimum-row entropy-collapse alarm in both heads. This means at least one exceptionally sharp row
per head—not whole-head collapse. The additive full run is a useful counterexample: no flagged heads
and a minimum normalized row entropy of 0.597.

The active K32 run was read only. Its validation PPL improved from 172.269 at step 1,500 to 122.069 at
step 9,000, but it had no final test, summary, or provenance artifact and is not reported as a result.
Across the retained full-Gaussian cohort every inspected endpoint is seed 6, so there is no replicated
performance trend.

Before interpreting family effects, re-evaluate selected checkpoints under one declared stride
(preferably 16 as the primary long-context protocol), then run a clean pinned paired-seed study with
equal **accepted** updates. A minimal family-by-MLP design is diagonal/full × MLP off/on at K32/H2,
N=128, at least five paired initialization/data-seed blocks (ten preferred), fixed data order,
validation-only selection, one locked test evaluation, and paired test-CE analysis. Report exact skip
counts, row-entropy quantiles, effective rank,
off-diagonal covariance norm, clamp fractions, wall time, and peak memory.

## Where an MLP would go

The canonical model should keep `MLP off` by default: the present variational derivation does not
derive a Transformer feed-forward network. If the goal is the closest standard-Transformer hybrid,
the MLP belongs **inside every VFE block**, after the derived E-step update, optional W_O-like head
mixer/CG transform, and the current attention-side block norm, but before `vfe_stack` hands the
belief to the next layer:

```mermaid
flowchart LR
    A["incoming Gaussian belief"] --> B["E-step: derived q star"]
    B --> C["capture canonical converged belief"]
    C --> D["optional head mixer / CG"]
    D --> E["attention-side block norm"]
    E --> F["new positionwise MLP + residual"]
    F --> G["new FFN-side norm"]
    G --> H["next-layer state and prior handoff"]
```

The exact production seam is after the current block norm at `vfe3/model/block.py:203-204` and
before the return at line 205; the resulting belief then reaches the handoff at
`vfe3/model/stack.py:153-154`. Inserting the MLP before the existing single norm is a
useful one-norm ablation, but it is not the usual two-sublayer post-norm Transformer. Inserting it only
after the stack (`vfe3/model/model.py:1307-1313`) makes it a readout/capacity control, not a per-layer
FFN and not a test of deep rank-collapse prevention. The modules should normally be untied across
layers; implement them as a per-layer `ModuleList` selected by `vfe_stack` and passed into
`vfe_block`. Reusing one module would instead implement tied recurrent computation.

### Derivation boundary

The E-step already has an identity-plus-correction form for the mean and an SPD retraction for the
covariance (`vfe3/inference/e_step.py:1175-1200`). The latest manuscript explicitly distinguishes
that residual resemblance and the Boltzmann-gated linear message from a learned expansion/activation/
contraction FFN (`Research/manuscripts/GL(K)_attention.tex:1790-1819,2089-2159,2204-2206`). Thus a
standard MLP is an explicit hybrid augmentation. The canonical free-energy claim ends at the captured
pre-transform `q*` (`vfe3/model/block.py:168-174`); neither the post-E-step MLP nor its output inherits
an E-step descent guarantee.

### Gauge-compatible construction

A fixed-coordinate MLP

\[
\mu_i^+=\mu_i+W_2\rho(W_1\mu_i+b_1)+b_2
\]

does not commute with an arbitrary local `GL(K)` action. It is a valid conventional baseline only if
it is labeled non-gauge and its unchanged covariance is labeled a mean-only approximation.

For the flat factored-transport route, reuse the *authoritative* forward/inverse factors already
exposed by `VFEModel._canonical_frame_context` (`vfe3/model/model.py:795-818`), rather than
re-exponentiating `phi`:

\[
a_i=U_i^{-1}\mu_i,\qquad C_i=U_i^{-1}\Sigma_iU_i^{-\top},
\]
\[
a_i^+=a_i+W_2\rho(W_1a_i+b_1)+b_2,\qquad \mu_i^+=U_i a_i^+.
\]

Because `a_i` and `C_i` are invariant under `U_i -> g_i U_i`, the pushed-forward mean transforms
covariantly. There are three honest covariance contracts:

1. `canonical_mean_only`: retain `C_i`; gauge-compatible but deliberately omit nonlinear uncertainty
   propagation.
2. `canonical_delta_moment`: for the residual map `a_i^+=a_i+m_i(a_i)`, use its full Jacobian
   `J_i = ∂a_i^+ / ∂a_i = I + ∂m_i / ∂a_i` and set
   `C_i^+ = J_i C_i J_i^T + Q_i`, with positive-definite `Q_i`, then push forward by `U_i`.
3. `canonical_congruence`: have invariant features produce an invertible `A_i` (for example
   `A_i=exp(B_i)`) and use `C_i^+=A_i C_i A_i^T`; this guarantees SPD, though it is not necessarily
   the exact nonlinear pushforward.

The latter two are the meaningful full-Gaussian research arms. If a dense canonical FFN mixes heads,
its covariance map also leaves the presently preserved product `SPD(d)^H` submanifold; either use a
headwise/block-diagonal FFN or deliberately upgrade downstream energies and decoding to consume the
new cross-head covariance. Gauge-pure arms should also avoid ordinary LayerNorm, whose fixed-coordinate
operation is not `GL(K)` equivariant; use no norm or the exact full-covariance Mahalanobis norm.

### Theory-first alternative

If the objective is to keep the learned nonlinearity *inside* variational inference, use a scalar
token-local unary potential rather than a post-hoc vector MLP:

\[
\mathcal F_\theta(q)=\mathcal F(q)+\lambda_u\sum_i
\mathbb E_{z_i\sim q_i}[V_\theta(z_i,x_i)].
\]

The resulting mean and covariance gradients then flow through the existing natural-gradient/SPD
update. A particularly controlled version lets an invariant-feature MLP output a canonical Gaussian
target and adds `KL(q_i || r_theta,i)` as a unary factor. This must be wired consistently through
`free_energy_value`, the oracle, hand/MM gradients, trajectory diagnostics, and any frame objective;
adding it to only one update route would create a new objective-parity defect.

### Recommended ablation

Compare: off; post-stack coordinate MLP; per-block coordinate mean-only; per-block canonical
mean-only; per-block canonical full-moment; and in-E-step unary factor. Use zero-initialized residual
output, untied layers, mixer off, parameter-matched widths, identical token budgets, at least five
paired seeds (ten preferred), and both `L=1` and `L>=4`. Measure PPL, effective rank by depth,
pre-E-step/post-E-step/post-FFN
free energy separately, random-gauge mean/covariance error, covariance conditioning, skipped steps,
throughput, and peak memory. This directly tests the manuscript's stated possibility that a
positionwise FFN brakes deep rank collapse (`GL(K)_attention.tex:2432-2437`) without mislabeling the
hybrid arm as derived.

Implementation guards are part of the design: an in-stack MLP placed inside the `detach` E-step
estimator's `no_grad` scope would be frozen, so that configuration must be rejected or the post-E-step
module must be reevaluated with gradients. A post-block MLP may increase free energy, a mean-only map
does not propagate uncertainty, a Jacobian map can become ill-conditioned, and a single shared MLP
changes the hypothesis from standard Transformer depth to tied recurrent inference.

## Verification and test evidence

All numerical test lanes were CPU-only with `CUDA_VISIBLE_DEVICES=-1`; the active RTX 5090 training
job was not disturbed.

| Evidence lane | Machine-readable result | Interpretation |
|---|---|---|
| Full-Gaussian targeted suite | 231 tests; 231 passed; 0 failed/error/skipped | Broad full-covariance numerics, transport, decoder, fallback, MM, precision, and RoPE coverage |
| Audit candidate probes | 22 tests; 22 passed; 0 failed/error/skipped | Reproduces the current behaviors asserted in `.verification/results/test_candidate_probes_20260810.py` |
| Fast CPU reliability selection | 253 tests; 252 passed; 1 failed; 0 error/skipped | The one failure is the stale parameter-match grid, not a blanket suite failure |

The failing test is
`tests.test_2026_07_15_driver_reliability_remediation::test_default_parameter_match_grid_retains_two_realized_30m_widths`:
the checked grid admits only K=60 within its 2% tolerance, so the sweep's requirement for two widths
raises before launch. Counts come from JUnit XML, not console progress.

The final probe source SHA-256 is
`80ebd1d5609bddc178b1df21312d313bff2924263ef5fce3e2152eeaffaf27a1`; its fresh JUnit SHA-256 is
`22b56dd4fd9784619be0c207728a0243d95fe70ec1c34b5529ac219354328124`. Stale intermediate XML and
probe cache artifacts were removed after exact-path validation; only the current final XML is used.

In the isolated audit worktree, `.verification/deep-audit-2026-08-10-ledger.json` contains all 59
terminal claims and passes the installed verification gate against the pinned artifact revision.

No CUDA suite or GPU performance benchmark was run while the user's training job remained resident.
CUDA-specific resume behavior was instead closed with source tracing, a real fused-Adam checkpoint,
and a deterministic non-CPU-device preflight probe; this establishes the preflight contradiction,
not an end-to-end live-GPU resume measurement.

## Three-wave coverage

Thirty bounded investigator assignments were executed in three waves of ten under a four-slot runtime;
they were not all simultaneous. Adversarial challenge and final adjudication lanes were additional.

| Wave | Ten investigator lenses |
|---|---|
| 1 | active config/reachability; full-Gaussian decoder; SPD geometry; gauge transport; information geometry; variational/free-energy parity; data/provenance; experiment artifacts; performance; adversarial source review |
| 2 | CUDA checkpoint/resume; optimizer binding; active config trace; multiseed/data joins; sweep statistics; scaling inference; report/visualization integrity; finalization memory; full-covariance channels; gauge clean-room review |
| 3 | SPD spectral calculus; gauge/purity proofs; information geometry; variational diagnostics; numerical failure paths; Transformer/MLP placement; run-artifact interpretation; theoretical purity; statistics/reproducibility; adversarial test coverage |

The source review used an isolated clean worktree at `b033a19`; run artifacts were read from the live
`vfe3_runs` tree with each run's own identity retained. The user's dirty live checkout, config edits,
and resident CUDA training process were not modified. The latest Research-vault manuscript and wiki
were consulted for theory context, but current code and artifacts—not project prose—were treated as
authority for runtime claims.

## Remediation order

1. **Repair the experimental decision boundary first.** Standardize evaluation stride, compare CE
   under the same loader, use equal accepted-update budgets, freeze the 2×2 family/MLP protocol, and
   reserve a new untouched confirmatory holdout or produce a timestamped decision ledger.
2. **Make long-run recovery and identity trustworthy.** Fix fused-CUDA optimizer-step preflight,
   bind optimizer slots to parameter names, persist immutable start provenance, retain imported
   process identity under drift, and reject drifted/unexpected members in aggregate publication.
3. **Close active/conditional full-Gaussian correctness seams.** Fix the fp32 Loewner divided
   difference, propagate the decoder's scored-token count into training/evaluation, reject or safely
   fall back from failed MM Cholesky factors, and make nonfinite decoder behavior row-local.
4. **Correct the verified model-family reporting now.** Distinguish `SPD(d)^H` from unrestricted
   `SPD(K)` and state that diagonal vocabulary priors make CE directly blind to posterior
   correlations. Mechanically close the remaining purity/readout-label candidates before changing
   those labels.
5. **Repair the verified statistical/reporting defects, then test the rest.** Pair wall time with the
   same seeds as validation loss and make analytic cost covariance-aware. Mechanically test the
   inconclusive CE-ranking, clustered-bootstrap, lucky-seed, and stale-output claims; repair them only
   if confirmed.
6. **Only then add an MLP.** Start with `off` as the compatibility default, implement the per-block
   post-attention-norm residual FFN as an explicit hybrid, and separately test canonical full-moment
   and unary-energy variants. Require default-off bit identity, optimizer/checkpoint wiring, random
   gauge tests, SPD preservation, and deep rank-collapse ablations before interpreting performance.

## Scope limits and open obligations

- The K32 full-Gaussian run was still active and has no terminal test/provenance result in this audit.
- No GPU benchmark, CUDA suite, or live CUDA resume was run while the resident training job was active.
- The fp32 spectral-exponential Jacobian defect is mechanically real on the active path, but live
  tangent-spectrum frequency and loss/PPL impact remain unmeasured.
- The stride-16/stride-128 conditioning distributions are mechanically different; no same-checkpoint
  dual-stride evaluation measured the resulting CE delta.
- Repeated visibility of one test digest is established, but test-driven adaptive configuration
  selection is not. That research-process claim remains inconclusive absent a timestamped decision
  ledger or other primary record.
- The retained full-Gaussian endpoints are all seed 6. Single-run values are descriptive, not a
  replicated family effect.
- Findings marked supported/optional or inactive are source-reachable defects, not claims that they
  occurred in the active K32 run.
- The MLP analysis is prospective architecture design. No MLP was implemented or tested.
- This audit intentionally made no production-source fix; remediation should be a separate,
  reviewable change set after the user selects scope.

## Non-provenance remediation closure — 2026-08-11

After the audit, the user explicitly deprioritized provenance-integrity work and authorized the
other fixes. The remediation was performed on the isolated branch
`codex/nonprovenance-remediation-20260811`, from pinned base `b033a19`, with the final code change at
`7c1d705`. The dirty live checkout and its resident full-Gaussian CUDA process were not modified or
inspected. This appendix updates remediation status only; the original ranked audit above remains the
revision-bound record of what was found at `b033a19`.

### Ranked fixes completed

The following 15 mechanically verified non-provenance findings were repaired in original audit-rank
order. `W2-CUDA-RESUME` is implemented and mechanically verified through portable CPU/meta-device
preflight, load, realized validation, and rollback; an actual CUDA resume remains an explicit open
integration obligation.

| Audit rank | Finding | Remediated behavior | Code commits |
|---:|---|---|---|
| 3 | `W3-SPECTRAL-EXP-GRAD` | Stable Fréchet/Loewner limits preserve near-degenerate off-diagonal cotangents. | `de660b8` |
| 4 | `W1-DECODE-NONFINITE` | Full decoders sanitize before arithmetic, exclude invalid rows, count them once, and produce zero invalid-row gradients, including projected dense frames and head-block-only failures. | `177879a`, `89a1820`, `7c1d705` |
| 5 | `W2-CUDA-RESUME` | Snapshot preflight is device-portable; load uses live optimizer metadata; realized placement validation is transactional with rollback. | `def8c5e`, `d1fa349` |
| 6 | `W2-OPT-NAMES` | Schema-v2 manifests bind checkpoint-local slot IDs to canonical parameter names, shapes, and exact group order under a digest. | `def8c5e`, `d1fa349` |
| 7 | `W3-EXCLUDED-TOKEN-DENOM` | Decode CE returns the actual scored-token count; evaluation and accumulation use that denominator and skip zero-count updates. | `177879a` |
| 8 | `W3-MM-PARTIAL-CHOLESKY` | MM exact combines factorization masks, sanitizes before solves, retains the whole old row on any failure, and publishes one fallback count. | `c2d70ab` |
| 9 | `W3-EMISSION-TRACE` | Full-Gaussian emission uses `tr(diag(d) Sigma)` rather than broadcasting across off-diagonal entries. | `c2d70ab` |
| 10 | `W2-LOGEUC-NONFINITE` | Invalid log-Euclidean rows are counted and frozen with finite, identity pullbacks instead of becoming plausible unreported covariances. | `563621f` |
| 11 | `W3-LOGEUC-REPEATED-GRAD` | Repeated-spectrum base-point backward uses the Fréchet limit and preserves identity-map off-diagonal gradients. | `de660b8` |
| 12 | `W1-GLOBAL-POLICY` | Full-Gaussian precision policy is owned and propagated by each model/PriorBank instance; legacy standalone behavior remains a fallback. | `3d66110`, `fe01e56`, `3c7efb2` |
| 13 | `W1-PARAM-GRID` | The exact 90-cell raw grid now deterministically admits K45/H5 and K60/H10 near 30M parameters without expanding the selected training count. | `3a8f188` |
| 14 | `W2-MULTISEED-EXTRAS` | Requested, accepted, observed, unexpected, duplicate, and unidentified cohorts are explicit; invalid panels withhold aggregates and figures; config homogeneity applies only after cell classification. | `3a8f188`, `33256f6` |
| 15 | `W2-UNPAIRED-PARETO` | Validation quality and wall time are inner-joined by exact unique seed with paired/missing metadata and no mixed denominators. | `3a8f188` |
| 23 | `W1-NATURAL-DTYPE` | Mixed mean/covariance inputs are promoted consistently without changing matching-dtype alias behavior. | `3d66110` |
| 24 | `W2-NEGATIVE-MOMENT` | Adam second-moment slots must be finite and elementwise nonnegative before restoration. | `def8c5e` |

Two compatibility-only commits (`97d386e`, `c17176c`) keep legacy test doubles and genuinely
pre-manifest fixtures aligned with their public contracts; they do not weaken production validation.

### Adversarial review outcomes

Independent review was not ceremonial. It found and forced repairs for five defects in intermediate
implementations:

- log-Euclidean invalid rows had a finite forward but could still emit NaN base-point gradients;
- a `FullGaussian` subclass override could receive a new private policy keyword despite the legacy
  plugin contract;
- checkpoint names/shapes were initially not cryptographically bound to serialized slot-ID order;
- decoder-final head-block invalidity and projected dense-frame invalidity were not both reproduced
  by later exclusion/backward consumers; and
- multiseed config homogeneity initially ran before failed/missing cells were classified.

Each was repaired and independently re-reviewed. The final Task 4 review approved `7c1d705` with a
specific 1/1 regression and an 88/88 canonical/fallback/dispatch lane. The final Task 6 reviews
approved `33256f6` with independent 19/19 and 57/57 lanes; its earlier broad lane was 277/277.

### Mechanical closure

The final revision-wide CPU selection was derived mechanically from every test file changed between
`b033a19..7c1d705`. Its JUnit artifact
`.verification/final-modified-tests-current.xml` reports **1,029 tests, 0 failures, 0 errors, and 3
expected skips**. Task-specific RED, GREEN, compatibility, and adversarial JUnit files remain under
`.verification/` in the isolated worktree.

An earlier all-repository diagnostic collected 5,143 tests and reported 37 failures, 0 errors, and
41 skips. It was not used as closure evidence: most failures were caused by a concurrently active
verification marker, three compatibility fixtures were subsequently repaired, and the remaining
13-node residual selection was reproduced exactly at both the pinned base and then-current branch
(380 tests and the same 13 failures in each). The current changed-test closure above is the
revision-bound passing lane.

All numerical remediation checks were CPU-only using `C:/Python314/python.exe`,
`CUDA_VISIBLE_DEVICES=-1`, and `VFE3_TEST_DEVICE=cpu`. No live-GPU resume, CUDA benchmark, or GPU
process inspection was performed. Therefore the real-device half of `W2-CUDA-RESUME` remains
`INCONCLUSIVE`; the portable preflight/load/rollback contract is `EVIDENCE_VERIFIED`.

### Deliberately deferred

- `W1-PROCESS-IDENTITY`, `W1-MULTISEED-DRIFT`, and `W3-START-PROVENANCE` were not targeted, per the
  user's explicit provenance decision.
- The evaluation-stride, rejected-update, sharp-row, analytic-FLOP, readout-blindness, and block-SPD
  findings are experiment-protocol or interpretation obligations rather than silent fixes to apply
  during this code-remediation branch.
- Inconclusive candidates remain inconclusive; none was promoted into a repair without new eligible
  evidence.
- The MLP remains a prospective explicit hybrid. Its exact placement, gauge-compatible variants,
  covariance Jacobian (`I + dm/da`), detach-mode guard, and ablation design are documented in
  **Where an MLP would go** above; no MLP code was added.
