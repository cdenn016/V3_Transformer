# Second-Panel Multiphase Ultradeep Codebase Audit — 2026-07-15

## Executive verdict

This second audit examined the whole repository at `origin/main` commit `2ebf0c84b6f49a1edfa855c9a750f858e9ee5aef` from the isolated branch `codex/second-ultradeep-audit-20260715`. That commit differs from the first audit's code snapshot only by the committed first-audit documentation. The live checkout and its pre-existing configuration and documentation WIP were not modified.

Ten distinct expert definitions from `C:\Users\chris and christine\.claude\agents` were executed as ten independent lens passes in capacity-limited parallel waves: `audit-gauge-theorist`, `audit-geometer`, `audit-info-geometer`, `audit-variational`, `audit-numerical-analyst`, `audit-transformer-ml`, `audit-implementation-engineer`, `debate-expert-ml-engineer`, `debate-expert-code-quality`, and `data-engineer`. Each pass was instructed to ignore all earlier audit reports and re-derive claims from executable source and fresh probes. The runtime/config investigator returned a clean negative. The other nine passes produced 31 raw claims.

A newly spawned blind verifier received the claim ledger and source locations, but not the investigators' reasoning. It re-read the executable paths, reproduced the load-bearing probes, and froze every verdict before opening the first audit. The raw result was **30 CONFIRMED, 1 REFUTED, and 0 INCONCLUSIVE**. Two confirmed claims described the same fixed-ridge covariance defect, leaving **29 unique confirmed findings: 5 high, 21 medium, and 3 low**. No production source or test was changed.

The default suite passed with **2,896 passed and 31 skipped**. The slow-enabled suite passed with **2,914 passed and 13 skipped**. Both runs collected 2,927 tests and reported zero failures and zero errors in JUnit XML.

## Scope and audit phases

The audit covered all 247 tracked Python files: the 66 production modules under `vfe3`, click-run entry points, active and opt-in configurations, registries, geometry and divergence kernels, inference and cache paths, training and optimizers, data/cache loaders, resumability and artifacts, reporting and visualization, and tests. Comments and docstrings were treated as intent rather than proof.

The work proceeded in seven phases. First, `origin/main` was fetched and pinned in a fresh worktree. Second, the default JUnit baseline established the clean snapshot. Third, the Research vault supplied the program's declared VFE, gauge, information-geometric, SPD, attention, and pure-path boundaries. Fourth, the ten specialist definitions performed blind source and probe passes. Fifth, a fresh verifier adjudicated all claims without access to the first audit. Sixth, the frozen verdicts were compared against `docs/audits/ultradeep-audit-codebase-2026-07-15.md`. Seventh, the slow suite, syntax parse, dependency check, and environment checks were completed.

The Research context came from `wiki/projects/VFE Transformer Program.md` and the linked gauge-equivariance, SPD-manifold, information-geometry, variational-free-energy, attention, and transformer-scaling theme pages. Those pages were contextual constraints. Current executable code and reproduced probes are the evidence for every retained finding.

## Verified findings

| ID | Domain | Finding and scope | Severity | Executable source | Required repair |
|---|---|---|---|---|---|
| S2-G1 | Gauge geometry | The fixed `eps * I` ridge in the route labeled `regime_ii_covariant` is basis-dependent under nonorthogonal `GL(K)` congruence, including the full-Gaussian path recorded as exact. | High | `vfe3/geometry/transport.py:402-458,685-690,788-803`; `vfe3/run_artifacts.py:1773-1775,1837-1844` | Factor valid SPD inputs without a ridge; treat jitter recovery as an explicitly approximate fallback and correct the artifact flag. |
| S2-G2 | SPD geometry | A finite `sigma_max` applies an ordinary spectral ceiling after the AIRM exponential, so the bounded route ceases to be congruence-equivariant when the cap binds. The `sigma_max=None` pure route remains available. | Medium | `vfe3/geometry/retraction.py:212-220,247-269`; `ablation.py:781-801,1030-1035` | Name the capped route as a projected approximation and use `sigma_max=None` whenever exact AIRM congruence is required. |
| S2-G3 | Gauge chart | Matrix-exponential input clamping changes `exp(phi)` to the exponential of a rescaled algebra element. Logging can warn later, but there is no hard validity gate at the altered call. | Medium | `vfe3/geometry/transport.py:1086-1098`; `vfe3/train.py:56-114,1216-1218`; `train_vfe3.py:138,211-215` | Enforce a post-step chart bound below the clamp or fail closed when the transport clamp activates. |
| S2-G4 | Positional geometry | The active learned positional route uses a finite fourth-order BCH composition without a locality or residual gate, although exact group multiplication exists as a separate path. | Medium | `vfe3/model/positional_phi.py:124-163`; `vfe3/geometry/lie_ops.py:263-286,482-506`; `vfe3/model/model.py:710-737` | Use exact group product for learned position or enforce a chart-radius and BCH-residual acceptance gate. |
| S2-I1 | Statistical family | Head mixing propagates diagonal `sigma` as a variance. For Laplace beliefs that slot is scale `b`, so the mixed dispersion has the wrong homogeneity and moment semantics. | Medium | `vfe3/model/head_mixer.py:159-187`; `vfe3/families/laplace.py:60-68,87-101` | Dispatch dispersion mixing through the family registry and use the Laplace moment-matched scale. |
| S2-I2 | Information geometry | `MahalanobisNorm` weights diagonal Laplace means by `1/b` instead of the Laplace Fisher weight `1/b^2`. | Medium | `vfe3/geometry/norms.py:75-107`; `vfe3/families/laplace.py:124-146`; `vfe3/model/model.py:263-274` | Make normalization family-aware and use the correct Fisher block for each family. |
| S2-I3 | Information geometry | The mean trust region perturbs every diagonal family with `sqrt(sigma)` scaling; Laplace scale requires normalization by `b`, not `sqrt(b)`. | Medium | `vfe3/numerics.py:110-140`; `vfe3/inference/e_step.py:977-990` | Route trust-region whitening through the configured belief family. |
| S2-I4 | Attention reliability | Precision-weighted attention treats the Laplace scale sum as a covariance trace. The factorized Laplace covariance trace is `2 * sum(b^2)`. | Medium | `vfe3/model/model.py:54-77,2370-2401` | Compute the reliability statistic through a family-specific registry hook. |
| S2-I5 | Numeric configuration | `kl_max` and `renyi_order` accept NaN or infinity, allowing nonfinite bounds or meaningless saturated divergences. | Medium | `vfe3/config.py:841-846,943-953`; `vfe3/families/base.py:19-27` | Require finite, strictly positive values at configuration construction. |
| S2-I6 | Diagnostics | Gaussian Fisher and covariance-spectrum formulas are emitted unchanged for Laplace runs, producing mislabeled diagnostics without changing optimization. | Low | `vfe3/model/model.py:3077-3096`; `vfe3/metrics.py:519-561`; `vfe3/viz/extract.py:305` | Add family-aware Fisher and covariance-spectrum diagnostic hooks. |
| S2-V1 | Free-energy gradient | Flooring `log(beta)` in the split entropy expression breaks exact softmax stationarity below the floor and makes the differentiated scalar disagree with the analytic envelope coefficient in tiny tails. | Low | `vfe3/free_energy.py:423-455,653-661`; `vfe3/gradients/oracle.py:153-160` | Use the coherent reduced free energy or a zero-safe `xlogy(beta, beta)` expression without flooring positive probabilities. |
| S2-N1 | Numerical accuracy | The float32 diagonal-Gaussian pairwise statistic cancels small positive KL values to zero or inflates them before deriving the active pair mask. | Medium | `vfe3/gradients/pairwise_stats.py:87-117`; `vfe3/gradients/kernels.py:507-530`; `train_vfe3.py:363,406` | Use a cancellation-stable series form or a narrow float64 reduction before forming the mask. |
| S2-N2 | Spectral derivatives | The absolute `1e-12` eigengap floor suppresses valid full-SPD eigenvector derivatives by orders of magnitude near the covariance floor. | High | `vfe3/geometry/retraction.py:77-124,247,260,267`; `ablation.py:781-801,1030-1035` | Replace the absolute floor with a scale- and dtype-aware guard plus a correct exact-degeneracy limit. |
| S2-T1 | Architecture contract | The checked-in active route uses a raw affine vocabulary projection with 1,055,397 trainable parameters at `V=50257,K=20`. This contradicts the repository's hard statement that all capacity comes from iterative VFE minimization, despite avoiding the `nn.Linear` class. | High | `train_vfe3.py:117-124`; `vfe3/model/model.py:1522`; `vfe3/model/prior_bank.py:368-390,1710-1734` | Restore the active prior-bank geometric decoder or explicitly revise the hard architecture contract and segregate the affine route as a non-pure baseline. |
| S2-T2 | Gauge equivariance | The active trainable head mixer does not commute with independent per-head `block_glk` gauge actions. | Medium | `train_vfe3.py:118-119,166`; `vfe3/model/head_mixer.py:29-35,143-187`; `vfe3/model/block.py:155-157` | Disable the mixer on the gauge-pure route or restrict it to an intertwiner compatible with the selected group action. |
| S2-M1 | Optimizer schedule | GradScaler overflow or another rejected optimizer update still advances the learning-rate scheduler. | Medium | `vfe3/train.py:623-655,699-700` | Advance the scheduler only when `did_step` is true and persist the successful-update count. |
| S2-M2 | Mixed precision | Inner E-step derivatives are materialized under autocast before the outer loss is scaled, so the outer GradScaler cannot recover inner underflow or overflow. | Medium | `vfe3/model/model.py:1000-1016`; `vfe3/gradients/oracle.py:153-162`; `vfe3/inference/e_step.py:1043`; `vfe3/train.py:488-501` | Construct inner objectives and derivatives in explicit fp32 islands or independently scale those differentiations. |
| S2-M3 | State transition | Opt-in Metropolis frame transitions can mutate state and consume RNG after an outer optimizer step was rejected. | Medium | `vfe3/train.py:1223-1236`; `vfe3/model/model.py:1414-1444` | Gate proposals on `did_step` or define and record their cadence as independent transitions. |
| S2-C1 | GPU scheduling | The active no-grad head-mixer shortcut calls `.item()` on every forward, forcing a device-host synchronization during evaluation, generation, and diagnostic replay. | Medium | `vfe3/model/head_mixer.py:147-168`; `vfe3/model/block.py:155-157` | Remove the CUDA identity shortcut or cache identity through parameter-version state without a tensor-to-Python conversion. |
| S2-C2 | Optimizer cost | Direct-omega validation performs repeated scalar synchronizations and float64 `slogdet` before and after each update on an optimizer hot path. | Medium | `vfe3/gauge_optim.py:202-213,508-545` | Consolidate checks into one device status transfer and move determinant validation to a sparse diagnostic cadence. |
| S2-C3 | Dead code | Four objective-related private methods have test-only callers. The duplicate surfaces are real, but current production-objective divergence was not demonstrated. | Low | `vfe3/model/model.py:1302,1837,1924,1952` | Delete the test-only surfaces or make production delegate through one authoritative implementation. |
| S2-C4 | GPU scheduling | Optional EMA executes a Python scalar finiteness decision for each parameter on every successful step, causing one accelerator synchronization per parameter. | Medium | `vfe3/ema.py:83-95`; `vfe3/train.py:1235-1236` | Aggregate finiteness on device and transfer one combined status per EMA update. |
| S2-D1 | Resume integrity | A checkpoint restores cursor and RNG state without binding them to the corpus, tokenizer, data cap, or content identity, so resume can splice semantically different data at one cursor. | High | `vfe3/run_artifacts.py:500-520,539-554,619-690`; `vfe3/train.py:1154-1184` | Persist an exact train-source contract and reject resume when the live loader differs. |
| S2-D2 | Cache integrity | Binary token caches trust sidecar `n_tokens` without checking exact file length, so stale metadata can silently truncate or misrepresent the logical corpus. | High | `vfe3/data/datasets.py:217-229,248-251` | Validate positive logical length and exact byte-size equality before mapping or reporting the cache. |
| S2-D3 | Evaluation coverage | Sequence construction omits the final partial stride region, leaving `(T - 1) % seq_len` validation or test transitions unscored. | Medium | `vfe3/data/datasets.py:345-375` | Emit a padded final window and mask padding targets so every real held-out transition is scored once. |
| S2-D4 | Metric semantics | When character normalization is unavailable, evaluation and artifacts still publish a field named `bpc`, making the fallback regime indistinguishable from character-normalized bits. | Medium | `train_vfe3.py:573-578,630-640`; `vfe3/train.py:728-776`; `vfe3/run_artifacts.py:1245-1250,1366-1375` | Persist normalization availability and publish `bpc=null` with a separately named bits-per-token value. |
| S2-D5 | Cache invalidation | Tokens-per-character memoization omits content or metadata identity, so an in-place cache replacement can reuse stale normalization. | Medium | `vfe3/data/datasets.py:303-342` | Key the memo on the cache-source identity or content digest. |
| S2-D6 | Memory | Character normalization materializes the complete token split as a Python list and one decoded string. | Medium | `vfe3/data/datasets.py:333-341`; `train_vfe3.py:573-578,630-640` | Count characters incrementally in tokenizer-safe bounded chunks or persist a validated count. |
| S2-D7 | Memory | Run finalization copies the complete training stream to CPU int64 before `bincount`, producing a corpus-scale memory spike. | Medium | `vfe3/run_artifacts.py:1065-1073,1395-1399` | Accumulate unigram counts over bounded chunks. |

## Refuted and duplicate claims

The proposed width-sweep initialization-temperature defect was refuted as stated. The generic `embed_dim` sweep inherits `use_prior_bank=False`, so it uses the affine decoder and does not consume `decode_tau`. A related scaling issue may exist on the separate prior-bank `tied_block_glk_wide` arm, but that was outside the claim and is not retained.

The gauge theorist and differential geometer independently reported the same fixed-ridge `regime_ii_covariant` defect. Both raw claims were confirmed, then deduplicated into S2-G1.

## Comparison with the first audit

The comparison occurred only after all second-panel verdicts were frozen.

S2-D6 independently corroborates first-audit P8: `tokens_per_char` materializes and decodes an entire split. S2-I5 independently refines first-audit B5 by identifying the same missing finiteness contract specifically at `kl_max` and `renyi_order`. S2-C2 extends first-audit P2: the direct-omega path not only rebuilds a large Gram pseudoinverse but also performs repeated synchronized determinant validation. S2-D7 extends first-audit P3 from repeated corpus hashing to another corpus-scale finalization pass. S2-C3 lies in the objective-fidelity area exposed by first-audit B1 and B2, but is a narrower maintenance defect rather than another demonstrated objective error.

S2-T1 contradicts the first audit's statement that production source contained no neural-layer violation. The first pass treated the absence of `nn.Linear` as dispositive. The second pass applied the full repository constraint: the active learned `W,b` vocabulary projection adds conventional affine capacity outside iterative VFE minimization. The code deliberately labels this route an ablation and retains a prior-bank pure path, but `train_vfe3.py` selects the affine route in the checked-in active configuration. The contradiction is therefore retained as a high-severity architecture-contract finding.

All other retained findings are new relative to the first report. The two exact independent rediscoveries were B5 through S2-I5 and P8 through S2-D6.

## High-severity punch list

1. **S2-G1 — repair the route labeled gauge-covariant.** Remove the coordinate-fixed ridge from valid SPD feature calculations, add nonorthogonal `GL(K)` regression coverage, and stop recording exactness for jitter-recovered results.
2. **S2-N2 — restore the intended full-SPD spectral derivative.** Replace the absolute eigengap damping floor with a scale-aware guard and test near-floor repeated spectra against a trusted derivative.
3. **S2-T1 — reconcile the active affine decoder with the hard architecture contract.** Select the prior-bank geometric decoder for the pure active profile or formally narrow the contract and label the affine route as a non-pure baseline.
4. **S2-D1 — bind resumability to data identity.** Store dataset, tokenizer, cap, cache identity, and content digest with the cursor and reject any mismatch before resuming.
5. **S2-D2 — validate binary cache bytes against metadata.** Reject missing, truncated, extended, nonpositive, or dtype-inconsistent cache files before constructing a memmap.

## Targeted verification evidence

| Probe | Reproduced result |
|---|---|
| Fixed-ridge gauge features | A common nonorthogonal push changed two supposedly invariant features by approximately `0.499999` and `0.499998`. |
| Capped AIRM congruence | Relative residual `0.999999` with the cap active versus `9.31e-16` with `sigma_max=None`. |
| Laplace head-mixer dispersion | Implementation returned `[5, 3]`; moment-matched Laplace scale was `[3.60555, 3]`. |
| Entropy-tail gradient | At `beta_tail=1.8467e-14`, differentiated scalar gradient was `-3.6782e-14` versus envelope coefficient `+1.8467e-14`. |
| Float32 pairwise KL | Ratio `1.0001` returned zero versus float64 `4.99933e-8`; ratio `1.0003` was inflated by `3.77x`. |
| Full-SPD eigengap damping | Gradient sensitivity multipliers were `0.5`, `0.0099`, and `0.0001` at gaps `1e-6`, `1e-7`, and `1e-8`. |
| Independent-head mixer | Maximum equivariance commutator residual was `2.4`. |

## Verification and limitations

The default command was `python -m pytest -x --junitxml=C:\tmp\vfe3-second-audit-baseline-20260715.xml`. JUnit reported `tests=2927`, `failures=0`, `errors=0`, and `skipped=31` in 130.315 seconds; the console reported 2,896 passed and 31 skipped.

The extended command was `python -m pytest --runslow -x --junitxml=C:\tmp\vfe3-second-audit-runslow-20260715.xml`. JUnit reported `tests=2927`, `failures=0`, `errors=0`, and `skipped=13` in 421.525 seconds; the console reported 2,914 passed and 13 skipped.

An in-memory AST parse reported `tracked_python_files=247`, `syntax_ok=247`, and `syntax_errors=0`. `python -m pip check` reported no broken requirements. Ruff was unavailable (`No module named ruff`). The interpreter is `torch 2.11.0+cpu`; CUDA was unavailable, so CUDA synchronization findings are source-verified scheduling defects rather than measured GPU timings. No long training run was launched.

Several findings are deliberately scoped to reachable opt-in paths: S2-G1, S2-G2, S2-N2, S2-M3, S2-C2, and S2-C4. The default or pure alternative remains available where stated. That scope lowers neither a false exactness claim nor a data-integrity defect, but it matters when ordering remediation.

The Research vault currently describes the covariance-feature route as gauge-covariant and records the affine output route as retained. S2-G1 and S2-T1 warrant a later qualification in the vault if the user approves an ingest. This audit did not modify the vault.

## Recommended remediation order

The first repair tranche should address S2-G1, S2-N2, S2-T1, S2-D1, and S2-D2 because they violate a declared invariant, replace a core derivative, contradict the architecture contract, or threaten data continuity. The second tranche should address the Laplace-family cluster S2-I1 through S2-I6 together, using family-dispatched statistics and diagnostics. The third tranche should address active-path numerical and state behavior: S2-G3, S2-G4, S2-V1, S2-N1, S2-T2, S2-M1 through S2-M3, and S2-D3 through S2-D5. The remaining synchronized hot paths, dead surfaces, and bounded-memory reporting work can then be grouped by subsystem.

No fixes should be combined with this audit commit. Each repair tranche should preserve the mathematically pure route, add focused regression tests for the exact failure, and finish with machine-readable default and slow-suite verification.
