# Deep Codebase Audit and WikiText-103 Performance Investigation — 2026-07-09

## Scope, snapshot, and noninterference

This audit examined the executable repository at origin/main commit e504f1c5ad5d277f653534cfc7fb63fd3b1bee61. The source inspection and validation ran in the isolated worktree C:\tmp\V3_Transformer_deep_audit_20260709 on branch audit/deep-audit-20260709. The live checkout remained on fix/ultradeep-audit-findings-20260709 with its modified ablation.py and train_vfe3.py files untouched. No source code or experiment configuration was changed.

The existing investigation at docs/audits/ultradeep-audit-findings-investigation-2026-07-09.md was treated as an exclusion ledger. Its 107 rows were not relabeled or repeated as new findings. The audit instead searched for executable defects and bottlenecks outside that set, then checked every candidate for overlap. Two candidates were found to deepen existing Findings 14 and 87; they are recorded as addenda rather than assigned new identifiers.

The repository inventory contained 272 tracked or nonignored files, including 194 Python files, 53 Python files under vfe3, 129 Python test files, and 49,566 Python source lines. Five specialist passes covered mathematical and numerical behavior, debugging and training semantics, Python and configuration contracts, structural and dead-path analysis, and performance plus archived-result interpretation. A separate verifier reread the source and existing 107-row report without inheriting the investigators' reasoning. Comments were used only to locate seams; every accepted finding is based on executed paths, source expressions, or focused probes.

The result archive examined was C:\Users\chris and christine\Desktop\data. It currently contains the 66.48 and 69.38 WikiText-103 K=160 runs discussed below. No new training was launched.

## Verdict

The audit confirmed 19 findings not represented by the 107-row investigation: 11 medium and 8 low. There is no new high or critical finding. The highest-priority defects are an oracle truncation path that removes the requested last-k inference graph, a state-dependent hyperprior that remains active at lambda_h=0, omission of phi reflections from two differentiated transport paths, batch-coupled eigengap damping, and inconsistent implementation of the deterministic-run contract.

| ID | Severity | Reachability | Finding |
| --- | --- | --- | --- |
| M1 | medium | oracle route plus e_steps_backprop_last > 0 | Truncated-backprop boundary leaves freeze the requested last-k E-step |
| M2 | medium | s_e_step plus state-dependent lambda_h mode at lambda_h=0 | The documented hyperprior off switch still exerts force |
| M3 | medium | phi reflection plus live phi update or non-flat oracle | Reflections are omitted from the phi objective and non-flat oracle transport |
| M4 | medium | batched full covariance with spectral retraction | Eigengap damping couples unrelated batch elements |
| M5 | medium | canonical and scaling entry points; mixed ablation runs | The deterministic setting has entry-point and process-lifetime drift |
| M6 | medium | full covariance plus log_euclidean | The retraction consumes an affine-Fisher tangent as a log-chart tangent |
| M7 | medium | s_e_step plus nondefault E-step controls | The model-channel E-step ignores six global controls |
| M8 | medium | resumed reflection or reorthogonalization runs | Auxiliary Metropolis RNG and omega cadence state are not restored |
| M9 | medium | cached EFE rollout plus phi reflection | The cache claims support but omits the reflection frame |
| M10 | medium | generate_efe.py with best_model.pt plus external config_from | Model state is not bound to the configuration used to interpret it |
| M11 | medium | n_layers > 1 plus mstep_self_coupling_weight > 0 | The M-step reconstructs a prior trajectory that never occurred |
| L1 | low | constant lambda_alpha | Negative and nonfinite self-coupling coefficients are accepted |
| L2 | low | decoder registration or override | Decoder capability metadata and training dispatch can diverge |
| L3 | low | kernel override after compiled lookup | The old compiled kernel remains active |
| L4 | low | stochastic generate_efe.py comparison | Base and policy arms use unpaired random streams |
| L5 | low | RoPE or ALiBi with invalid numeric inputs | Validation fails open, and RoPE guidance names an unregistered mode |
| L6 | low | skipped update with EMA enabled | EMA advances even though the live model did not update |
| L7 | low | post-construction mutation of RoPE config | The RoPE tensor cache ignores value-setting fields |
| L8 | low | grad_accum_steps > 1 with E-step diagnostics | Reported E-step gradient norms come only from the final microbatch |

The discrete severity labels above round the verifier's “medium-low” reachability classifications upward to medium so enabled-path semantic defects are not hidden among reporting-only lows. None is active in every archived best-run step. M5 affects interpretation of that run's reproducibility contract; the other medium findings are configuration-dependent.

## Validation status

The full suite was run without adding a second quiet flag and recorded to JUnit XML. The machine-readable suite attributes were tests=1692, failures=9, errors=0, skipped=1, time=746.524 seconds. Therefore 1,682 tests passed and 1 test skipped. The checkout is not green.

The nine failures form four default-contract clusters: two deterministic-default tests, three Phase-0 forward checksums, one checkpoint-interval default test, and three run-label tests. A commit bisection showed that tests/test_phase0_forward_beliefs.py passes 11 of 11 at 7ffb68e and fails three checks after 1b917d0. The cause is the committed default seed changing from 0 to 6. At the current source, explicitly setting seed=0 reproduces the frozen bank checksum -0.7388787269592285 and linear checksum 0.13538306095870212 exactly; seed=6 gives -0.7379648685455322 and 0.1337966330065683. The recent transport-clamp change is not the cause. These are stale golden/default tests, not evidence that seed 6 is an invalid experimental choice.

Focused probes were used where the suite has no assertion. The reflection-aware scalar free-energy gradient differed from the current phi substep by 0.0149650276 in maximum absolute component, while the all-positive reflection case matched within 8.94e-8. A truncated two-iteration oracle run produced no mu-table gradient despite a nonzero output-projection gradient. State-dependent lambda_h moved s_mu by 0.0045795441 at lambda_h=lambda_gamma=0, whereas constant mode moved it by zero. The batched eigengap probe changed one sample's gradient by 94.0651 when an unrelated larger matrix was appended. Invalid rope_base=0 reached a ZeroDivisionError, and alibi_slope=NaN produced nonfinite output. Registry probes returned the replacement eager kernel alongside the old compiled kernel and retained stale full/chunked decoder flags after override.

No CUDA training or long profiler run was started. Performance quantities below either follow exactly from tensor shapes and file sizes or come from the archived RTX 5090 runs.

## Detailed medium findings

### M1. Truncated oracle backprop freezes the requested last-k inference graph

At vfe3/inference/e_step.py:991-999 the truncation boundary replaces mu, sigma, and phi by detached tensors. They are not made gradient-bearing leaves. The oracle at vfe3/gradients/oracle.py:94-103 enables its live graph only when create_graph is requested and both mu and sigma already require gradients. The detached boundary therefore forces the fallback leaf path, and vfe3/gradients/oracle.py:147-150 returns detached gradients.

The defect is reachable whenever e_steps_backprop_last is positive and the E-step uses an oracle route, including smoothing, full covariance, entropy suppression, non-KL divergence, Regime II, or decoupled RoPE. A two-iteration probe with only the last iteration requested for backpropagation produced a mu_embed gradient norm of 0.2469561696 under full unroll, no mu_embed gradient under truncation, and a nonzero output-projection gradient of 0.0106131611. The decoder trains, but the requested post-boundary dependence of inference on the prior tables and connections does not.

This is separate from existing Finding 6, which concerns a shared prebuilt transport leaking across the truncation boundary. The direct repair is to create fresh boundary leaves with requires_grad enabled and to preserve the oracle's differentiable path for the requested last iterations.

### M2. State-dependent lambda_h remains active at lambda_h=0

The model-channel refinement at vfe3/model/model.py:631-659 forwards value=cfg.lambda_h together with cfg.lambda_h_mode. The state-dependent coefficient delegates to c0_h/(b0_h + D) and does not use value. The executable result conflicts with the warnings at vfe3/config.py:1417-1422 and 1468-1477, which state that lambda_h=0 and lambda_gamma=0 leave the s refinement without force and that a nonconstant lambda_h mode has no effect when lambda_h=0.

The focused probe measured zero s_mu movement in constant mode and 0.0045795441 maximum movement in state-dependent mode at the same nominal off settings. The remedy is an explicit channel gate before registry dispatch, or construction-time rejection of state-dependent modes when the channel weight is zero.

### M3. Phi reflections are omitted from two differentiated transport paths

The reflected frame is g_i = R_i exp(phi_i), so its flat transport is R_i exp(phi_i) exp(-phi_j) R_j. The scalar free-energy path passes belief.reflection at vfe3/inference/e_step.py:371-379, and the ordinary flat mu/sigma path passes it at 631-640. In contrast, phi_alignment_loss has no reflection argument at 448-478, its transport construction at 499-504 omits the reflection, and the live phi substep at 816-834 cannot supply it. The non-flat oracle closure at 605-623 also omits the reflection.

The root probe found that the current phi loss and an all-positive-reflection scalar F agree to 8.94e-8 in gradient, while a mixed-reflection scalar F differs by 0.0149650276. This defect persists with pos_rotation=none and is distinct from existing Finding 49's decoupled-RoPE score/value envelope mismatch. Reflection must be threaded through phi_alignment_loss and the non-flat oracle builder, followed by a scalar-F gradient regression.

### M4. Spectral damping couples unrelated full-covariance examples

The relative eigengap safeguard at vfe3/geometry/retraction.py:108-123 computes scale = A.detach().abs().amax() without matrix axes. One scalar over the entire batch then sets the divided-difference damping for every matrix. A spectral function is batch-separable; appending an unrelated covariance must not change the derivative of a covariance already present.

The isolated and batched probes produced gap values near 1e-12 and 1e-10, respectively, and changed the first matrix's gradient by 94.0651, or 0.76994 relative difference. The scale must be computed per matrix over the final two axes and broadcast only within that matrix's eigengap grid.

### M5. Deterministic-run semantics differ by entry point and process history

VFE3Config currently defaults deterministic=True at vfe3/config.py:567. The ablation runner calls its comprehensive seeding helper, but train_vfe3.py:540-581 and scaling.py:631-634, 704-718 only seed Torch and never apply cfg.deterministic to deterministic algorithms, cuDNN, or CUBLAS behavior. Their saved configuration can therefore claim deterministic=True without implementing the same policy.

The helper itself is one-way over a long-lived process. Calling its false branch after the true branch leaves torch.are_deterministic_algorithms_enabled() true, cuDNN deterministic true, and benchmark false. Mixed ablation cells can inherit the prior cell's global policy. Deterministic setup should be centralized, called by every entry point, and made reversible for process-local state. The intended default is an experiment choice and is not itself the finding.

### M6. log_euclidean is not a retraction for the tangent supplied by the E-step

The full-Gaussian family supplies the affine-Fisher covariance tangent H = 2 Sigma G Sigma through vfe3/geometry/retraction.py:368-401. The registered log_euclidean path at 317-364 then computes exp(log Sigma + H), treating that ambient tangent as a log-chart displacement. A log-Euclidean retraction of an ambient tangent requires D log_Sigma[H] in the chart. The current map fails the first-order retraction condition away from the identity.

In one dimension with Sigma=4 and H=1, the proper first-order chart displacement is H/Sigma=1/4, giving 4 exp(1/4)=5.136, while the implementation gives 4 exp(1)=10.873. The source discloses that the convention is noncanonical, but the mathematical operation is still not a retraction for the tangent the E-step supplies. The pure spd_affine path remains available. The opt-in variant should apply the Fréchet derivative of the matrix logarithm or form the log-Euclidean Riemannian gradient before exponentiation.

### M7. The model-channel E-step ignores six global controls

The belief channel forwards e_step_update, mm_damping, randomize_e_steps, e_steps_min, e_steps_max, e_steps_backprop_last, and e_step_halt_tol through vfe3/model/block.py:94-104. The model-channel call at vfe3/model/model.py:631-687 omits them. With s_e_step enabled, q can therefore use MM while s silently uses gradient updates; randomized depth, truncation, and halting likewise apply to q alone.

This is distinct from existing Findings 6, 12, and 23, which cover a shared transport boundary, integer validation, and missing model-channel RoPE. The settings should either be forwarded to both channels or exposed as explicitly separate q and s controls.

### M8. Resume omits auxiliary reflection and reorthogonalization state

The checkpoint bundle at vfe3/run_artifacts.py:277-290 stores global CPU/CUDA RNG, model, optimizer, scaler, and EMA state. The reflection generator is created after checkpoint loading at vfe3/train.py:876-882 from the initial seed, so a resumed Metropolis sequence restarts rather than continues. GaugeNaturalGradAdamW._omega_step is a plain attribute initialized to zero at vfe3/gauge_optim.py:122 and is not part of optimizer.state_dict; it controls reorthogonalization cadence at 277-278.

The defect is confined to resumed phi/omega reflection runs and skew omega-direct runs with periodic reorthogonalization. Persist both the private generator state and cadence counter in the checkpoint bundle or optimizer extra state.

### M9. Belief-cache capability drift gives wrong reflected EFE scores

The capability predicate at vfe3/inference/belief_cache.py:56-87 admits the phi path without checking phi_reflection. The cached update at 109-159 rebuilds transport from phi alone and drops the reflection from the returned state. The full E-step folds the stored reflection into transport.

An init_seed probe reported cache_supported=True but a maximum cached-versus-full log-probability difference of 0.0051743984 and allclose=False. This is distinct from existing Finding 58's matrix-exponential precision key and Findings 71 and 95's omega-direct inversion work. Until reflection is represented in the cache state and transport, the capability predicate must reject it.

### M10. best_model.pt is not bound to its interpretation config

RunArtifacts writes best_model.pt as a pure state_dict. generate_efe.py:60-72 and 92-93 can load that state while borrowing config_from from another run. Strict state loading verifies names and shapes, not shape-preserving semantic fields. A state from n_e_steps=1 loaded strictly into an otherwise shape-compatible n_e_steps=3 model without missing or unexpected keys.

This is a narrow provenance defect rather than unsafe deserialization. Save the best state with its configuration and a semantic fingerprint, and reject an external configuration unless the fingerprint matches.

### M11. Multi-layer M-step self-coupling uses a prior trajectory that never occurred

The real stack at vfe3/model/stack.py:75-99 folds each intermediate layer output into the next prior. The M-step path at vfe3/model/model.py:1307-1323 starts from the encoded prior but repeats the final output in every fold. For more than one layer, this replaces all intermediate beliefs by the last belief. It is exact only for one layer, zero handoff, or identical intermediate outputs.

The source labels this as an approximation, which lowers severity, but the configured term is otherwise described as self-divergence against the block prior. Capture the actual input prior to the final block during the forward pass and score against it.

## Low findings

| ID | Executable evidence and consequence | Recommended correction |
| --- | --- | --- |
| L1 | vfe3/config.py has no finite/nonnegative guard for constant lambda_alpha. VFE3Config(lambda_alpha=-1) constructs and returns alpha=-1; NaN also passes. Negative alpha rewards self-divergence. | Require a finite, nonnegative value for constant mode. |
| L2 | register_decode at vfe3/model/prior_bank.py:85-105 never removes old full/chunked flags on override, while vfe3/model/model.py:1220-1253 hardcodes known fused CE functions. A custom chunked decoder can use one kernel at inference and diagonal KL CE in training. | Store callable and capabilities, including fused CE, in one registry record and clear old metadata on override. |
| L3 | register_kernel at vfe3/gradients/kernels.py:22-36 replaces the eager callable but does not invalidate _COMPILED_KERNELS at 214-234. A process probe returned “new” from eager lookup and “old” from compiled lookup. | Pop the compiled cache entry on every registration or override. |
| L4 | generate_efe.py:118-128 generates the stochastic base arm, advances global RNG, constructs the policy model, and then samples the policy arm. A changed continuation is not a paired policy comparison. | Snapshot and restore CPU/CUDA RNG around each arm, or use greedy/matched replicate evaluation. |
| L5 | The RoPE warning at vfe3/config.py:1573-1582 recommends pos_phi=sinusoidal, but only none, learned, and frozen are registered. rope_base=0 constructs and later divides by zero; alibi_slope=NaN constructs and yields nonfinite output. | Name a registered mode and require finite rope_base>0 plus finite nonnegative slopes. |
| L6 | After a nonfinite update is skipped, vfe3/train.py:942-943 still calls ema.update, moving the shadow toward unchanged live weights and incrementing its history. Scheduler advancement is deliberately defined per loop iteration and is not counted as a defect. | Return a did_step flag, including scaler overflow, and gate EMA updates on accepted parameter changes. |
| L7 | The RoPE cache at vfe3/model/model.py:517-529 keys only on sequence length, device, and dtype. Mutating rope_base after construction returned the old tensor; a fresh model differed by 1.9505632. | Freeze semantic config or include pos_rotation and rope_base in the key and invalidate on mutation. |
| L8 | A single E-step diagnostic dictionary is overwritten by every microbatch at vfe3/train.py:445-451 and copied after the loop at 506-512. With accumulation, estep_grad_norm fields describe only the last microbatch. | Accumulate a defined mean, RMS, or maximum and name the statistic. |

## Addenda to findings already under repair

The pullback-per-block path deepens existing Finding 14. vfe3/geometry/phi_preconditioner.py:392-398 creates fresh contiguous sub-bases, and the bracket-closure diagnostic at vfe3/geometry/lie_ops.py:127-190 retains each tensor by identity in a process-global cache. A three-call, two-block probe grew the cache from 0 to 2 to 4 to 6 entries. On CUDA this retains device tensors even after garbage collection and empty_cache. The repair for Finding 14 should therefore remove both host synchronization and unbounded ownership, for example by a value-stable basis signature or a bounded weak cache.

The failed-cell behavior deepens existing Finding 87. RunArtifacts writes matching config.json before training; if training then raises, ablation.py:1839-1864 writes error_kind=train. _cell_is_current at 1710-1755 compares configuration, dataset, and token cap but never rejects the error marker. A default resume therefore prints CACHED and skips the crashed cell indefinitely. The probe returned failed_cell_is_current=True. Finding 87's resume predicate should require a successful terminal state in addition to matching inputs and requested artifacts.

## Verified runtime and memory bottlenecks

### P1. The default block-GL “factored” path stores dense zero structure

The elementary block basis is allocated as n_gen by K by K, phi is expanded through a dense contraction, and FactoredTransport stores exp(phi) and exp(-phi) as full B by N by K by K tensors. At B=32, N=128, K=160, H=8, d=20, each factor occupies 419,430,400 bytes in float32. A packed B by N by H by d by d factor occupies 52,428,800 bytes, exactly eight times less. Carrying the packed block representation through BCH, exponentiation, and transport is the largest exact structural opportunity. Dense conversion should occur only for a consumer that cannot operate blockwise.

### P2. Lookup-only vocabulary tables use dense gradients and dense AdamW state

The archived phi table has 160,822,400 parameters, or 79.82 percent of the 201,488,242-parameter model. A batch contains at most 4,096 token positions, only 8.15 percent of the 50,257-row vocabulary before duplicate tokens, yet raw Parameter indexing feeds ordinary AdamW. Dense moment updates and weight decay touch the whole table each step. A row-sparse or lazy optimizer could remove most of that work, but it is not automatically value-equivalent: absent-row decay and moment time must be applied analytically when a row is next touched.

### P3. The canonical diagonal-KL route recomputes dominant sufficient statistics

The pairwise energy forms variance ratios, mean differences, inverse variances, and logs. The filtering gradient kernel then reclamps and recomputes the same inverse variance and mean-difference tensors over B by N by N by K. An energy-and-sufficient-statistics kernel can return the energy, masks, inverse variance, and difference once for reuse after beta is formed. Equivalence tests must allow only the expected float32 reassociation.

### P4. Stored phi can remain far outside the exact exponential radius

vfe3/geometry/transport.py:803-815 computes radial scaling under no_grad for the forward exponential, while plain AdamW leaves the stored phi table unconstrained. The current source uses a radius of 20; the archived runs used the then-active radius of 15. Their raw logs show the 66.48 run above 15 on 382 of 600 logged rows with maximum 26.40, while the 69.38 run was above 15 on 529 rows with maximum 2807.89. These single-seed, different-code runs do not establish a perplexity effect, but they establish sustained surrogate operation. A bounded parameterization or post-update group retraction should be evaluated as a separate mechanism, with the pure unclamped/exact path retained where numerically admissible.

### P5. Record validation minima synchronously rewrite an 806 MB state

maybe_save_best writes model.state_dict inline on every new validation minimum. Each archived best file is 805,957,877 bytes. The 66.48 history set 38 record minima across 40 evaluations and the 69.38 history set 40, implying approximately 30.63 GB and 32.24 GB of synchronous best-state writes. Keeping an in-memory CPU best snapshot and serializing it once at finalization, or using a staged asynchronous writer with bounded ownership, would remove the stalls while periodic resumable checkpoints retain crash recovery.

### P6. Logged throughput mixes training with callbacks

The throughput window starts before the training loop, includes model.diagnostics, and is reset before validation, sampling, figures, and best-state writing. Those callbacks then enter the next window. tokens_per_s is therefore neither clean kernel throughput nor full end-to-end throughput. CUDA-event step timing and a separate wall-clock pipeline rate would make optimization claims measurable.

The verifier did not independently confirm the proposed pairwise stack-copy byte estimate, so it is not promoted as a finding here.

## What the archived WikiText-103 runs establish

| Run | Test CE | Test PPL | Test CE without E-step | Best validation PPL | Wall time | Record minima |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 66.48 linear-mix | 4.1969537452 | 66.4834965849 | 6.7959748235 | 65.3666060419 | 45,383.58 s | 38/40 |
| 69.38 KL-max-fix | 4.2395489367 | 69.3765515645 | 43.1158981463 | 67.7297929541 | 45,341.51 s | 40/40 |

The 66.48 run is the current best of the two. It used K=160, eight block-GL heads of dimension 20, one layer, one E-step, sequence length 128, batch size 32, seed 6, 201,488,242 parameters, and 245,760,000 training tokens. Its last five validation PPL values were 67.4963, 66.3602, 65.7369, 65.5225, and 65.3666, so the fixed-budget trajectory had not visibly plateaued. The request excludes “just add E-steps/layers/toggles”; the proposals below therefore target memory, data presentation, objective control, or nonparametric prediction.

The no-E-step gap is large: 2.5990210783 nats for the best run and 38.8763492105 nats for the KL-max run. That demonstrates how much prediction depends on refinement, but the second value also shows that a brittle starting belief can be rescued by one step while still producing worse final PPL. It does not show that more E-steps would close the generalization gap.

The 66.48 artifact reports rare/mid/frequent CE of 6.2436, 4.2301, and 1.8452. Existing Finding 27 establishes that these strata are sample-local rather than corpus-frequency strata, so they are descriptive and cannot support a causal rare-token claim. Any frequency-targeted experiment must first compute strata from the training corpus.

The two runs are not a controlled kl_max comparison. kl_max changed from 100 to 1280, sigma_max changed from 100 to 1000, and the provenance records different dirty SHAs, 0ddef1857c39ab0321deb4c909419e3dfeb3b6e2 and 46013def69b4caa51851c1696b86ac86499ec9ee. Both are single-seed runs. The 2.893-PPL difference cannot be attributed to one setting.

## New performance hypotheses

The literature search was run on 2026-07-09 over primary papers in the ACL Anthology, arXiv, PMLR, and OpenReview. Inclusion required a mechanism that could be expressed without a neural submodule and that was not already a current toggle or a restatement of docs/2026-07-05-improvement-ideas.md. Neural results are used as mechanism evidence, not as predicted effect sizes for this model. Every proposal remains exploratory until matched-seed evidence exists.

### H1. Sequential continuous cache, followed only if useful by a kNN belief datastore

A validation-only continuous cache is the cheapest quality test. Store prior context states and observed next tokens, reset at document boundaries, score the current state against past states, and interpolate the induced empirical next-token distribution with the VFE distribution. This is a nonparametric observation factor, not a neural module. Continuous caches have improved language-model adaptation to recent history, and kNN-LMs have improved long-tail prediction without retraining in conventional models: [Grave et al. 2016](https://arxiv.org/abs/1612.04426) and [Khandelwal et al. 2019](https://arxiv.org/abs/1911.00172).

Raw Euclidean distance between belief coordinates must not be called gauge-invariant. The first test should use function-space distance between decoder distributions or transported symmetrized KL with retained frames. Tune cache weight and temperature on sequential validation only, then evaluate test once. The null hypothesis is that the validation optimum sets cache weight to zero. Stop if paired per-token bootstrap analysis shows less than 0.01-nat test-CE reduction or if any gain depends on cross-document carryover. Build a sampled training datastore only after the continuous cache passes.

### H2. Rotate training-window origins across epochs

TokenWindows sets stride=seq_len by default and start=idx*stride. Both archived epochs therefore present every corpus location at the same within-window position and truncate the same left context. Rotate the stream origin across epochs, for example offsets 0 and N/2, while keeping token count, sequence length, batch count, optimizer steps, and all model settings fixed. This changes data presentation, not capacity or the free-energy functional. Work on recurrence and shorter-input curricula supplies related evidence that fixed segmentation and context fragmentation matter: [Dai et al. 2019](https://aclanthology.org/P19-1285/) and [Press et al. 2021](https://aclanthology.org/2021.acl-long.427/).

Run three matched seeds for 10,000 to 15,000 steps and report CE by target position within the window. The null is equal validation CE under fixed and rotating origins. Stop if the mean gain is below 0.005 nat and position-conditioned CE does not improve.

### H3. Aggregate primal-dual control of the hyperprior KL budget

The existing state-dependent lambda_h is a pointwise analytic envelope, and kl_max is a numerical cap. Neither enforces an aggregate information budget. Replace manual weighting on the experimental path by a constraint such as mean KL(s_i || h) <= C and update a nonnegative dual multiplier from an unclamped, numerically stable diagnostic. GECO provides primary precedent for replacing fragile variational weights by explicit constraints: [Rezende and Viola 2018](https://arxiv.org/abs/1810.00597).

The best run's period of hyperprior saturation and the worse result after a much looser cap motivate adaptive control; they do not prove it. The null is equality with fixed lambda_h at matched tokens and wall time. Stop if the multiplier repeatedly hits its bounds, the budget error remains above 10 percent after warmup, or matched-seed validation CE fails to improve by 0.01 to 0.02 nat. The canonical fixed-weight path must remain available.

### H4. Segment-recurrent gauge memory, gated by a position-CE diagnostic

Before building memory, measure validation CE by within-window target position while controlling for corpus token frequency. If early-window targets are not at least 0.02 nat worse than late-window targets, stop. If the gap exists, retain detached past beliefs as fixed boundary conditions, retain their frames, use rectangular causal coupling from current queries to past and current keys, and reset at document boundaries. Appending mu in its local coordinates without frame transport is not gauge-principled.

Test memory lengths 0, 128, and 512 with local N fixed at 128. The null is no improvement over memory 0. Stop if memory 128 gains less than 0.01 nat or if stale-memory diagnostics worsen monotonically. Transformer-XL is mechanism precedent for segment recurrence, not an expected-PPL estimate for this architecture: [Dai et al. 2019](https://aclanthology.org/P19-1285/).

### H5. Byte-ngram hierarchical Gaussian priors

Share statistical strength across GPT-2 BPE byte strings by representing token mu/sigma or model-channel natural parameters as a hierarchy of byte-ngram Gaussian components plus a token residual. Natural-parameter barycenters preserve a probabilistic reading. Do not add or average phi coordinates: additive n-gram frame composition is not group-safe. Subword sharing has a strong rare-representation precedent in [Bojanowski et al. 2017](https://aclanthology.org/Q17-1010/), but its language-model effect here is unknown.

The cheapest gate is checkpoint surgery: fit n-gram components on frequent-token rows, shrink held-out rare rows toward their composition, tune one shrinkage coefficient on validation, and evaluate corpus-count-stratified CE. The null is an optimum at zero shrinkage. Stop unless rare-token CE improves without worsening total CE; only then run joint training.

### H6. Horizon-specific geometric auxiliary prediction

Naively training one distribution against several future tokens is wrong because it fits a mixture target rather than separate horizon conditionals. A valid no-neural-network version uses horizon-specific registered algebraic transports and separate linear or Gaussian observation heads, consumes no future inputs, and discards auxiliary heads at inference. Multi-token prediction supplies mechanism evidence in conventional models: [Gloeckle et al. 2024](https://arxiv.org/abs/2404.19737).

Start with horizon two, a small auxiliary weight, and three matched short runs. The null is no reduction in primary one-step validation CE. Stop if primary CE is more than 0.02 nat worse by 10,000 steps or if only the auxiliary head improves. The archived best trajectory was still descending at its final step, so any slowdown in primary optimization is a real risk.

### H7. Gauge-aligned manifold averaging

This is lower priority and is not the existing coordinate EMA toggle. Align late checkpoints within only those transformations proven to preserve the live diagonal-family and linear-decode function, average Euclidean means, average SPD variables geometrically, and average frames on their group. Riemannian iterate averaging provides general precedent: [Tripuraneni et al. 2018](https://proceedings.mlr.press/v75/tripuraneni18a.html). Symmetry alignment before weight averaging is supported by conventional-model work such as [Ainsworth et al. 2023](https://openreview.net/forum?id=CQsmMYmlP5T), but transfer to this gauge model is an inference.

The archive has no periodic checkpoints, and the best checkpoint is the final descending iterate, so this requires a new checkpoint trajectory. First test the validation-loss interpolation barrier between late checkpoints before and after alignment. Stop if alignment improves the barrier by less than 0.005 nat; do not attempt cross-seed averaging until same-trajectory alignment passes.

### Rejected proposal: exact global right-gauge recentering on the live phi route

A common right multiplication preserves U_i U_j^{-1} for stored group elements, but the best run stores exp(phi_i). A product exp(phi_i)A need not have a real logarithm in the represented algebra. BCH projection is not exact, and the live learned positional phi, diagonal family, head mixer, weight decay, and linear decode further obstruct a blanket quotient claim. This proposal should not be implemented for the archived route without a machine-precision logits-and-loss invariance test. An exact quotient may be available for a restricted symmetry or omega-direct storage, but that is a different experiment.

## Recommended experiment order

First, run three diagnostics that require no training: sequential continuous-cache validation, CE by target position within the 128-token window, and corpus-frequency-stratified CE. These decide whether H1, H4, and H5 have an empirical target. Second, run rotating-window origins because it is low-cost, parameter-free, and leaves the functional unchanged. Third, test the primal-dual controller with matched seeds. Segment memory, byte-ngram joint training, multi-horizon prediction, and manifold averaging should proceed only after their gates pass.

The runtime work should begin independently with packed block factors and deferred best-state serialization. The former has an exact eightfold factor-storage argument; the latter targets approximately 31 to 32 GB of repeated writes observed in each archived run. Sparse table optimization has larger potential but changes absent-row AdamW semantics and needs a dedicated equivalence definition.

## Limitations and handoff

This is a frozen-source audit of e504f1c plus read-only analysis of two older dirty-run archives. Claude/Fable is actively repairing the prior 107 findings on the live branch, so no source fix was attempted here. The new findings should be reconciled against that work before implementation, especially the addenda to Findings 14 and 87.

The full suite remains red because nine tests encode pre-change defaults. No new training, CUDA benchmark, or multi-seed experiment was run. Every proposed quality gain is a hypothesis, not a forecast. A matched-seed preregistration should record the primary CE/PPL endpoint, token and wall-time budgets, stop rules above, and the exact git state before any result is treated as causal.
