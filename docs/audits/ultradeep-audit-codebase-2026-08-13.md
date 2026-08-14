# Ultradeep codebase audit — 2026-08-13

## Executive decision

The audited revision is **not release-ready**. The closure ledger contains 49 evidence-verified findings: 7 high, 30 medium, and 12 low. Two additional candidates were refuted. Eleven remain inconclusive because their only honest closure requires either an idle-CUDA measurement or performance profiling on the active hardware.

No critical vulnerability or conventional secret, injection, unsafe-deserialization, arbitrary-process, or destructive-file defect survived verification. The dominant risks are scientific provenance, reusable configuration regressions, overbroad gauge-purity certificates, and finite-precision violations of the covariance/gauge invariants.

## Scope and artifact identity

- Repository: `V3_Transformer`
- Audited source: `origin/main`
- Git revision: `714e3c5be458ef489a93e720468efa8f637a838b`
- Artifact identity: `git:714e3c5be458ef489a93e720468efa8f637a838b:sha256:19ba7a95480dbaf44b42aedde84ca107b81d1be30b7f8f84c4a19fa3268c2c79`
- Isolated audit worktree: `C:\tmp\V3_Transformer_ultradeep_audit_20260813`
- Audit branch: `codex/ultradeep-audit-20260813`
- Live checkout policy: read-only except for this report; existing user WIP was not stashed, reset, reverted, or edited.

This was a code and scientific-correctness audit, not a remediation pass. Findings bind only to the revision and artifact identity above.

## Audit method

The review used staged, independent lanes rather than one monolithic model judgment:

1. Five base investigators covered code quality/security, runtime defects, dead code/config drift, performance, and Python/API contracts.
2. Nine specialist lenses covered gauge theory, differential geometry, information geometry, variational/free-energy semantics, transformer/ML behavior, numerical analysis, implementation/config wiring, general ML methodology, and philosophy-of-science claim discipline.
3. Three independent verification views reassessed the deduplicated candidate set from source, mathematical, and adversarial perspectives. A fourth view re-read every then-high candidate.
4. A separate adjudicator closed the evidence ledger. Independent skeptic/defender challenges then stress-tested every finding that remained high in the final severity calibration.
5. Current machine-readable JUnit artifacts were used for test totals. Agent agreement alone was not treated as evidence.

The Research wiki was consulted for the program's gauge-purity, exactness, and experimental-claim boundaries. It was not modified.

## Mechanical test evidence

All CPU lanes used `C:\Python314\python.exe` with CUDA hidden. The test suite is not clean.

| Lane | Total | Passed | Failed | Errors | Skipped | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Targeted audit seam | 421 | 400 | 21 | 0 | 0 | `.verification/evidence/targeted-2026-08-13.xml` |
| CPU-fast | 5,215 | 4,936 | 242 | 0 | 37 | `.verification/evidence/cpu-fast-2026-08-13.xml` |
| CPU-slow | 3 | 2 | 1 | 0 | 0 | `.verification/evidence/cpu-slow-2026-08-13.xml` |

The 242 CPU-fast failures are not 242 independent defects. A machine parse found that 207 failure records contain the `pos_phi_compose='group_product'` incompatibility guard. The remaining targeted failures include two stale default assertions, the `decode_tau=0.01` default notice, and other consequences of the same reusable-default changes. The single CPU-slow failure is a stale exact-dictionary assertion after four numerical-policy provenance fields were added.

CUDA tests and benchmarks were deliberately not run. Before any CUDA claim, `C:\anaconda\python.exe` was checked and the RTX 5090 was observed at about 92% utilization with roughly 15.5 GiB resident and an active Anaconda Python process. Disturbing that workload was outside scope.

## High-severity punch list

These seven findings survived evidence closure and adversarial challenge at high severity.

| ID | Finding and impact | Decisive source | Challenge |
|---|---|---|---|
| A01 | Mixed-process source drift is detected but, by default, the artifact is relabeled with the newer disk digest even though imported executable modules came from older bytes. This breaks revision binding for long-lived processes and can misattribute checkpoints/results. | `vfe3/run_artifacts.py:2550-2587` | UPHELD |
| A02 | Multiseed/scaling provenance consumers do not reject persisted `source_identity_status='drifted'` or consistently bind to final source identity. Drifted runs can enter verified cohorts or reuse decisions. | `multiseed_analysis.py:271-327`; `scaling.py:958-1054` | UPHELD |
| A03 | The reusable `pos_phi_compose='group_product'` default is narrower than registered omega-direct, nonflat, alternate-frame, and legacy-migration routes. It makes broad accepted/config-migration paths unconstructible and is the common text in 207 CPU-fast failures. | `vfe3/config.py:330,1256,4010` | UPHELD |
| M04 | `on_gauge_pure_path` can be true for registry-declared gauge-fixed or noncovariant transports. The durable purity certificate therefore overstates executable equivariance. | `vfe3/run_artifacts.py:4486`; `vfe3/geometry/transport.py:822,1241` | UPHELD |
| M13 | Active finite-only FP32 congruence escalation returns finite but indefinite transported covariances because escalation checks nonfiniteness, not the SPD invariant. Downstream jitter/clamping contains NaNs but silently changes geometry and attention energy. | `vfe3/geometry/transport.py:1438,2799`; `train_vfe3.py:496` | UPHELD |
| M14 | Active GaugeGate computes the Mahalanobis invariant with an unchecked FP32 linear solve. Within accepted covariance bounds, conditioning can make gauge-related representations receive different learned gates, breaking the layer's intended equivariance. | `vfe3/model/block_mlp.py:162-200`; `train_vfe3.py:159` | UPHELD |
| M30 | Missing Regime-II runtime exactness evidence (`None`) is promoted to an exact certificate. A checked-in ablation schedules this route and calls empty-history reporting, so the artifact reverses evidence polarity even though no runtime certificate exists. The executed transport itself uses group products; no additive-transport defect is claimed. | `vfe3/run_artifacts.py:4433-4446`; `ablation.py:969,2697` | UPHELD |

### Required remediation order

1. Make executable-code identity fail closed or preserve the actually imported identity; then make every cohort/reuse consumer reject drifted/final-identity mismatches.
2. Restore reusable configuration defaults that construct the registered public surface and migrate historical configs without importing experiment-local choices.
3. Replace the gauge-purity boolean with a conjunction of executable invariants: transport covariance class, normalization, spectral/trust policies, causal prior scope, adaptive temperature, and runtime exactness evidence.
4. Add condition/residual/SPD-aware precision escalation to congruence and GaugeGate solves; retain the higher-precision result where required.
5. Only after those roots are fixed, update affected tests and run the full CPU plus idle-GPU lanes.

## Medium-severity findings

| ID | Verified finding | Primary location |
|---|---|---|
| A04 | `decode_tau=0.01` leaked from an experiment into the reusable default, warns while inert by default, changes omitted prior-bank behavior by 100x, and alters legacy replay. It was downgraded because the active launcher supplies explicit settings. | `vfe3/config.py:614,2746`; `vfe3/model/prior_bank.py:1453,1618` |
| A05 | Covariance and Rényi-order ablations lost the shared `family_chunked` prerequisite and abort registered sweep validation. | `ablation.py:1361,1440,5989` |
| A06 | Prerequisite repair silently changes `pos_phi_compose` for incompatible arms while preserving a one-factor ablation label. | `ablation.py:2551-2611` |
| A07 | Scaling structural signatures omit BlockMLP identity and can pool equal-parameter but computationally different mechanisms. | `scaling_analysis.py:132,433,840` |
| A11 | Mutating a retained config after model construction can make pure-path metadata disagree with the executable BlockMLP. | `vfe3/model/model.py:268,365`; `vfe3/run_artifacts.py:4409` |
| A12 | BlockMLP builders expose incompatible direct-call contracts, including a misleading frame-free canonical call. | `vfe3/model/block_mlp.py:86,122,264,300` |
| A13 | Direct construction with an invalid covariance string selects contradictory silent fallbacks across BlockMLP modes. | `vfe3/model/block_mlp.py:73,115,141,229` |
| A14 | `CanonicalFrameContext` does not validate that its matrices are mutual inverses before canonical propagation trusts them. | `vfe3/contracts.py:11`; `vfe3/model/block_mlp.py:272` |
| A15 | Direct `DecodeRegistration` construction accepts contradictory full-covariance capability metadata. | `vfe3/model/prior_bank.py:421` |
| A16 | Old checkpoints missing `decode_ce_checkpoint` inherit current `auto` semantics instead of historical `always`, without a drift record. | `vfe3/config.py:3918`; `vfe3/run_artifacts.py:2007` |
| A17 | A timed-out UMAP subprocess remains in the worker pool and can poison later requests. | `vfe3/viz/figures.py:220,270`; `vfe3/viz/report.py:788` |
| A18 | UMAP cleanup performs an unbounded wait after kill and can hang finalization. | `vfe3/viz/figures.py:360` |
| M01 | CG `delta_full` can have a singular Jacobian and map an SPD covariance to the PSD boundary without a positive floor. | `vfe3/model/cg_coupling.py:264` |
| M02 | Gauge-pure reporting ignores a fixed-coordinate spectral cap that is not GL-congruence equivariant. | `vfe3/run_artifacts.py:4492`; `vfe3/geometry/retraction.py:784` |
| M03 | `spd_retraction_exact` ignores trust projection and exponential clipping, so it can label a modified retraction exact. | `vfe3/run_artifacts.py:4450`; `vfe3/geometry/retraction.py:764` |
| M05 | Gauge-pure reporting accepts fixed-coordinate LayerNorm, which is not GL-equivariant. | `vfe3/run_artifacts.py:4486`; `vfe3/geometry/norms.py:124` |
| M06 | Strict checkpoint loading accepts reflection values outside `{−1,+1}`, destroying identity and cocycle laws. | `vfe3/model/prior_bank.py:1160`; `vfe3/inference/e_step.py:278` |
| M08 | Accepted Rényi orders above one can yield all-`-inf` family-decoder logits and NaN cross-entropy. It is an opt-in configuration and the training loop skips the nonfinite step, so challenge downgraded it from high. | `vfe3/families/gaussian.py:358`; `vfe3/model/prior_bank.py:2430` |
| M09 | The active full decoder is directly blind to posterior off-diagonal correlations; they only produce class-independent logit shifts. | `vfe3/model/prior_bank.py:1968-2019` |
| M10 | `fp32_escalate` accepts successful but inaccurate full-Gaussian factorizations because success, not error/conditioning, controls escalation. | `vfe3/families/gaussian.py:890,914`; `train_vfe3.py:488` |
| M11 | Active FP32 expanded-decoder algebra can reverse a near-tied vocabulary ranking. | `vfe3/model/prior_bank.py:310,1965`; `train_vfe3.py:425` |
| M12 | Laplace divergence and natural-gradient APIs unconditionally downcast float64 operands. | `vfe3/families/laplace.py:223,241` |
| M15 | BlockMLP diagnostics mix a pre-MLP self term with post-MLP coupling/entropy, so the displayed total belongs to no single belief state. | `vfe3/model/model.py:3151-3218,3889` |
| M16 | Target blindness does not imply near-zero or nonnegative cross-arm Pearson correlation, but the analysis presents that as a structural-EM signature. | `scaling_analysis.py:1263,1391` |
| M17 | Generic divergence figures hard-label non-KL registered objectives as KL and nats. | `vfe3/viz/figures.py:933,2949,4393` |
| M18 | A finite one-step, no-halt iterate is labeled converged/fixed-point/descent without stationarity or monotonicity evidence. | `vfe3/run_artifacts.py:3229`; `vfe3/viz/figures.py:1264,3214` |
| M20 | Noncausal content-sensitive priors can expose future tokens during next-token training while purity reporting remains true. The route is configuration-specific rather than active in the pinned launcher, so final severity is medium. | `vfe3/attention_prior.py:75`; `vfe3/run_artifacts.py:4462` |
| M24 | Full-Gaussian held-out CE/PPL excludes non-PD positions but final artifacts omit expected/scored/excluded token counts. | `vfe3/model/prior_bank.py:1987,2030`; `vfe3/run_artifacts.py:3168` |
| M26 | Multiseed reports claim shared data order for accepted `data_seed=None` cohorts even though order follows varying model seeds. | `multiseed_analysis.py:308,859,1114` |
| M29 | Gauge-pure reporting ignores query-adaptive temperature even though its trace rule is GL-breaking. The option is warning-labeled, separately reported, and disabled in checked-in launchers, so challenge reduced the contradictory purity summary from high. | `vfe3/config.py:3508`; `vfe3/run_artifacts.py:4486,4562` |

## Low-severity findings

| ID | Verified finding | Primary location |
|---|---|---|
| A08 | `NON_SWEPT_FIELDS` is dead and its rationale omits live encoder/divergence alternatives. | `ablation.py:1901` |
| A09 | The reachable ablation table omits BlockMLP gauge classification while a duplicate helper includes it. | `ablation.py:5040,5699` |
| A10 | `_sweep_is_complete` is an unused, weaker status-only predicate beside the provenance-aware cohort gate. | `ablation.py:5067` |
| A19 | `run_process_tree` annotates `CompletedProcess[str]` although default binary mode returns bytes. | `vfe3/process_utils.py:176` |
| M07 | Durable metadata omits that block-GL reflection flips only block 0 and accesses two of `2^H` components. | `vfe3/run_artifacts.py:4548` |
| M22 | `decode_last` is not bitwise/numerically identical to slicing the full decoder's last position. | `vfe3/model/model.py:990` |
| M25 | Default exploratory workflows expose test PPL despite validation-only selection language. | `multiseed_analysis.py:808`; `train_vfe3.py:653,793` |
| M27 | Launchers label `min_lr_frac=0.01` as scheduler floor OFF even though the endpoint multiplier is 0.01. | `train_vfe3.py:392`; `vfe3/train.py:527` |
| M28 | Reported PPL silently saturates at `exp(20)`, making very high-loss runs indistinguishable by the named metric. | `vfe3/train.py:973,1038`; `vfe3/run_artifacts.py:3171` |
| M31 | Adding a BlockMLP mode requires synchronized hard-coded edits across construction, validation, accounting, and reporting. | `vfe3/model/block_mlp.py:300`; `vfe3/config.py:2960` |
| M32 | The checked-in scaling temperature is inert under the selected linear decoder. | `scaling.py:181`; `vfe3/model/prior_bank.py:2997` |
| T01 | The CPU-slow provenance test's exact dictionary is stale after production added four numerical-policy keys. | `tests/test_run_diagnostics_2026_06_13.py:441` |

## Refuted candidates

| ID | Closure | Decisive counterevidence |
|---|---|---|
| M21 | REFUTED — the `infer_L` baseline does not activate the BlockMLP, so the proposed parameter-scaling contamination is absent. | `scaling.py:659,764` |
| M23 | REFUTED — the shared T5-style relative-position bias is an intentional tied bias, not evidence of a head/block correctness defect. | `vfe3/attention_prior.py:316`; `vfe3/model/model.py:562` |

## Inconclusive obligations

These are not findings and must not be described as verified regressions.

| ID | Open obligation |
|---|---|
| P01 | Measure retained saved-tensor memory for active decode checkpoint modes on an idle CUDA device. |
| P02 | Profile accelerator synchronization attributable to full-Gaussian Python boolean decisions. |
| P03 | Benchmark batched Cholesky retry amplification when only one matrix fails. |
| P04 | Measure the active cost of inverse-conditioning reductions unused by `fp32_escalate`. |
| P05 | Profile unconditional float64 SPD certificate factorizations on the active CUDA shape. |
| P06 | Measure per-parameter finite-gradient reduction/launch overhead. |
| P07 | Measure the full coupling-grid diagnostic that is subsequently replaced by zeros. |
| P08 | Confirm and profile the three repeated entry-attention computations during evaluation. |
| P09 | Profile diagnostic scalar-transfer serialization. |
| P10 | Benchmark serialized per-head GaugeGate solves/residual updates and compare a batched equivalent with identical semantics. |
| M19 | On an idle GPU, run cadence-parity experiments that vary only periodic generation and compare RNG/model-state/training trajectories. Current CPU evidence proves train-mode dropout and CPU-only RNG restoration, but not the full CUDA trajectory impact. |

## Challenge and severity calibration

The challenge tier did not use vote counts. The binding severity followed reachability, invariant impact, containment, and current source evidence.

- UPHELD at high: A01, A02, A03, M04, M13, M14, M30.
- DOWNGRADED to medium: A04, M08, M10, M20, M29. These remain real defects, but are either inactive in the pinned launcher, explicitly opt-in, contained by fail-visible behavior, separately disclosed in artifacts, or better characterized as bounded numerical/reporting errors than active invariant-destroying paths.
- REFUTED: M21 and M23.
- INCONCLUSIVE: P01-P10 and M19.

## Root-cause map

The findings should be remediated as clusters, not as 49 isolated tickets:

1. **Provenance identity:** A01, A02, A16, A26.
2. **Reusable-default/config migration regression:** A03, A04, A05, A06, T01.
3. **Gauge/exactness certificate is not executable-invariant complete:** A11, M02-M05, M20, M29, M30.
4. **Finite-precision policy checks success rather than mathematical validity:** M10-M14, plus M06 checkpoint-domain validation.
5. **Metrics/reporting mix states or hide denominators:** M15-M18, M24-M28.
6. **Registry/contract fragmentation:** A08-A15, M31-M32.
7. **Process/finalization robustness:** A17-A19.

## Closure statement

The verified ledger is `.verification/ledger.json`. Its terminal distribution is 49 `EVIDENCE_VERIFIED`, 2 `REFUTED`, and 11 `INCONCLUSIVE`. The ledger is revision-bound to the artifact identity above and must be invalidated after any affected source/configuration change.

No CUDA coverage claim is made. No remediation claim is made. The next safe action is a root-cause remediation branch followed by the same CPU lanes and, only when the GPU is idle, the CUDA/cadence obligations.

---

## Remediation disposition — 2026-08-14

This appendix preserves the audit above as the historical assessment of
`714e3c5be458ef489a93e720468efa8f637a838b`. It records the bounded remediation
authorized by the approved design and does not rewrite the original evidence.

### Authority and revision boundary

- Remediation branch: `codex/ultradeep-remediation-20260813`.
- Tested source/test revision: `659487ccd8ad9dd3c7b6afa042a4db04f578c04b`.
- Tested source tree: `1603bec3a6a26fe5ebf1948ceb7fc77c1c0607de`.
- Documentation revision: the commit containing this appendix. Its exact SHA is recorded after
  commit in `.superpowers/sdd/2026-08-13-ultradeep-audit-remediation/final-fix-report.md`, the
  final closure ledger, and the docs-only transfer record because a commit cannot contain its own
  SHA.
- The isolated remediation worktree was used throughout. No merge, push, or change to the user's
  live checkout was performed.

Evidence run at the tested source revision remains eligible after the documentation commit only
through a mechanical proof that the transfer is documentation-only. The final closure ledger
records that proof and binds its claims to the containing documentation revision.

### Scope disposition

`A01` and `A02` are **OWNER_WAIVED**, not fixed and not refuted. Per the binding owner policy,
source-identity drift remains informational and no load, resume, artifact-acceptance, cohort,
analysis, or reuse rejection gate was added.

Approved punch-list items 3 through 16 are **FIXED**. This disposition covers the exact finding
groups assigned to Tasks 1 through 7 in the approved plan: `A03`-`A19` as enumerated there;
`M01`-`M08`, `M10`-`M20`, `M22`, `M24`-`M32` as enumerated there; `T01`; and the real-CUDA `M19`
cadence obligation. It does not silently expand the approved scope to an original audit entry that
the design did not assign to those tasks.

The original `M21` and `M23` dispositions remain **REFUTED**. `P01` through `P10` remain
out-of-scope profiling obligations and therefore **INCONCLUSIVE**; this remediation makes no
unmeasured performance claim. `M19` moves from **INCONCLUSIVE** to **EVIDENCE_VERIFIED** through
the bounded real-CUDA cadence comparison described below.

### Approved items 3-16 coverage map

The approved design operationalizes numbered items 3-16 through the exact finding groups in the
tracked plan. This table binds every assigned group to its implementation/review range and to the
fresh final-ledger claim that closes it. `FIXED` is an audit disposition; the ledger uses the
formal state `EVIDENCE_VERIFIED`.

| Delivery | Exact approved finding IDs | Implementation / closure revision | Independent review | Fresh final-ledger claim |
|---|---|---|---|---|
| Task 1 | `A03`, `A04`, `A05`, `A06`, `A16` | `953cfb5..4eb24c9` | `task-1-review.md`: Approved | `FINAL-TASK1-APPROVED-SCOPE` |
| Task 2 | `A07`-`A15`, `A19`, `M31`, `M32` | `4eb24c9..383f17e` | `task-2-review.md`: Approved | `FINAL-TASK2-APPROVED-SCOPE` |
| Task 3 | `M02`-`M05`, `M07`, `M20`, `M29`, `M30` | `383f17e..8888069` | `task-3-review.md`: Approved | `FINAL-TASK3-APPROVED-SCOPE` |
| Task 4 | `M13`, `M14` | `8888069..59a6ca2` | `task-4-review.md`: Approved | `FINAL-TASK4-APPROVED-SCOPE` |
| Task 5 | `M01`, `M06`, `M08`, `M10`-`M12`, `M22` | `59a6ca2..50540b5` | `task-5-review.md`: Approved | `FINAL-TASK5-APPROVED-SCOPE` |
| Task 6 | `M15`-`M18`, `M24`-`M28`, `T01` | `50540b5..5927100`; final barrier repair `659487c` | `task-6-review.md`: Approved; final scoped review | `FINAL-TASK6-APPROVED-SCOPE`; `FINAL-TARGET-ACCOUNTING-HOT-PATH` |
| Task 7 | `A17`, `A18`, `M19` | `5927100..06cc097`; cleanup compatibility repair `f8ab5f8`; CUDA close at `659487c` | `task-7-review.md`: Approved; final scoped review | `FINAL-TASK7-APPROVED-SCOPE`; `FINAL-M19-CADENCE-PARITY` |
| Tasks 8-9 control plane | Comparative CPU/static/CUDA/status closure; no new finding scope | code/test `f8ab5f8` and `659487c`; docs-only descendants | `task-8-review.md`; `broad-final-review.md`; final scoped review | `FINAL-CPU-NO-NEW-ROOTS`; `FINAL-REPOSITORY-CPU-ALL-GREEN`; `FINAL-CANONICAL-CUDA-24`; `FINAL-DURABLE-DOCUMENTATION-STATUS` |

Task-local reports and reviews live under
`.superpowers/sdd/2026-08-13-ultradeep-audit-remediation/`. The fresh final ledger lives under
`.verification/remediation-2026-08-14/`; its artifact revision and the final docs-only transfer
record supply the binding to the checked-out documentation HEAD.

### Revision-bound closure evidence

All CPU lanes used `C:/Python314/python.exe` with CUDA hidden and `VFE3_TEST_DEVICE=cpu`. CUDA
claims use `C:/anaconda/python.exe`, `VFE3_TEST_DEVICE=cuda`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` after a fresh idle-GPU gate.

| Lane | Revision | Total | Passed outcomes | Failed | Errors | Skipped | SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| Final-fix focused accounting | `659487c` | 4 | 4 | 0 | 0 | 0 | `FF20122C110A9FDAB8C02F6C7FAFF581E49F80CD5F9D45410B965F973C2473FA` |
| Task 6 accounting/adjacent inventory | `659487c` | 381 | 378 | 0 | 0 | 3 | `6C6C46DFF070CFE13E66EF1FC67A9472B1801C0461E1708CAEFD9AC338A46554` |
| CPU-fast policy | `659487c` | 5,473 | 5,430 | 6 | 0 | 37 | `A029AD52CBFBFF8F2E8AC12E21C917E2ADC0670E2CE3C99139CF8BA3224E5F81` |
| CPU-slow policy | `659487c` | 3 | 2 | 1 | 0 | 0 | `2E26D02E156179A61D337BBCB1CDADC382AA46B49AC31E587A5F4F44395B132B` |
| Canonical CUDA marker | `659487c` | 24 | 24 | 0 | 0 | 0 | `9521566FA62B6E2BD75BEF3BDBF66D69947FC3BA260AA4CFA9A4E2D3DAB659F5` |

The CPU-fast `Passed outcomes` entry includes 5,418 ordinary pytest passes plus 12 passing
subtests. Repository-wide CPU green is therefore **REFUTED**, not verified: CPU-fast retains six
failures and CPU-slow retains one. The machine-readable base comparison establishes that all seven
remaining failure nodes have the same outcome and first failure root as the direct audit base,
with zero new or changed roots attributable to remediation. The accounting-specific source lane is
green.

The final-fix removes the unconditional exact-count device-to-host materialization from silent
training steps. It retains independent expected-target derivation, device-side scored/excluded
partition validation, exact detailed invalid-partition errors, metrics-step integer counts, and
strict `int64` behavior above `2**53`. Exact counts are never combined with the floating transfer.

For `M19`, two deterministic real-CUDA arms each ran three identical training steps and differed
only by one periodic-generation call after step 1. Exact equality held for model state, optimizer
state, scheduler state, CPU RNG, every CUDA RNG state, step records (including target counts),
training-dropout activity, and module modes. The comparison artifact SHA-256 is
`3CAA83EEB2DD7F02CEB724A75A491A40F991F489A2BF42C20D3AD241EB169B24`.

### Warnings and limitations

- The six CPU-fast and one CPU-slow failures are persistent base failures. They prevent any claim
  that all repository CPU tests are green even though they introduce no new remediation root.
- CUDA stderr was empty. The cadence run emitted only the known Triton warnings that
  `cuobjdump.exe` and `nvdisasm.exe` were unavailable; its required comparisons still exited zero.
- The existing resident Spyder/Anaconda GPU processes were identified read-only and preserved.
- No broad training, stress, three-seed, or 200-step workload was run.
- `P01` through `P10` remain open and require separately authorized profiling.
  They are pre-existing, out-of-scope performance-measurement candidates, not approved
  punch-list items 3 through 16 and not blockers to this remediation closure.
