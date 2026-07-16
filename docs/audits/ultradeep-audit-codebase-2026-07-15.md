# Multiphase Ultradeep Codebase Audit — 2026-07-15

## Executive verdict

This audit examined the whole repository at `origin/main` commit `c5c1f1c1660bb8b73205a977ac6a423325a9ce73` from the isolated branch `codex/ultradeep-audit-20260715`. The live checkout and its pre-existing WIP were not modified. The scope comprised 247 tracked Python files, including 66 production modules under `vfe3`, all click-run drivers, configuration and registry seams, inference and geometry kernels, training, checkpoints and artifacts, reporting, visualization, benchmarks, and tests.

Five independent investigators produced 32 candidate findings. A fresh verifier re-read every cited executable path and reproduced the load-bearing runtime claims. The final result is **32 CONFIRMED, 0 REFUTED, and 0 INCONCLUSIVE**: 10 high-severity, 19 medium-severity, and 3 low-severity findings. No source fixes were made.

The default suite passed with **2,896 passed and 31 skipped**. The slow-enabled suite passed with **2,914 passed and 13 skipped**. Both runs collected 2,927 tests and reported 0 failures and 0 errors in JUnit XML. The residual skips are environment or artifact gated and are enumerated below.

## Scope and phases

The audit used six phases. First, `origin/main` was fetched and pinned in an isolated worktree. Second, the production architecture and configuration/registry surfaces were mapped from executable code. Third, the Research vault was consulted for the program's declared VFE, gauge, SPD, and pure-path boundaries. Fourth, five investigators separately audited quality/security, bugs and theory/gradients, dead code and reachability, performance, and type/serialization contracts. Fifth, a fresh verifier adjudicated all 32 claims from source and targeted CPU probes. Sixth, the default and slow integration suites, an in-memory syntax pass, and dependency checks were run.

Theory-concordance context came from the Research vault pages `wiki/projects/VFE Transformer Program.md`, `wiki/concepts/GL(K) gauge-equivariant attention.md`, `wiki/themes/Variational free energy and predictive coding.md`, `wiki/themes/Gauge equivariance and geometric deep learning.md`, and `wiki/themes/SPD-manifold geometry and Riemannian optimization.md`. Those pages were context, not proof: every retained finding is supported by current executable source and, where needed, a fresh probe.

The latest post-remediation series added approximately 8,700 Python lines across 52 files after commit `25ccf3a`. The audit therefore treated July 13 remediation claims as regression targets rather than inherited facts.

## Investigator findings

### 1. Quality, security, and reliability

The investigator reported nine candidates: run-directory collision and shared temporary-file races; incomplete deterministic-runtime setup in the ring experiment; survivor-only scaling and multiseed analysis; checkpoint schema reinterpretation in sigma-gate measurement; non-atomic figure/sidecar publication; acceptance of zero or duplicate multiseed requests; unseeded non-greedy EFE comparisons; an undeclared direct NetworkX dependency; and process-wide suppression of duplicate OpenMP runtime failures. The verifier confirmed all nine, downgrading the NetworkX item to low because the current lock obtains it transitively and the affected functions are public but not registered report figures.

### 2. Bugs, gradients, and mathematical fidelity

The investigator reported six candidates: Metropolis scoring omitted active gamma/meta and hyperprior-sensitive terms; scored gamma energy omitted RoPE; beta/gamma prior mixing normalized on incompatible supports; approximate transpose inversion broke exact cocycles below a loose residual threshold; several objective controls accepted nonfinite values; and gamma energy had no direct frame-optimization route. The verifier confirmed all six. The final item is classified as a documented low-severity design limitation rather than a hidden autograd defect.

### 3. Dead code, reachability, and configuration wiring

The investigator reported five candidates: the scaling driver converted failures into unconditional completion; 17 advertised ablation cells retained an incompatible `mm_exact` baseline; three further ablation cells omitted paired prerequisites; six scaling cells combined tied groups with an incompatible per-block pullback preconditioner; and omega retraction bypassed the repository's required registry seam. Exhaustive construction confirmed the 17, 3, and 6 invalid-cell counts, and the verifier confirmed all five findings.

### 4. Performance and memory behavior

The investigator reported eight candidates: dense diagnostic transports defeated the compact path; full `omega_direct` optimization rebuilt a large Gram pseudoinverse each step; ablation finalization rehashed unchanged corpora; opt-in phi projection scanned every frame row each step; evaluation synchronized the host twice per batch; fixed-point diagnostics replayed inference; reporting repeated full inference over one token bank; and `tokens_per_char` defeated bounded-memory loading. The verifier confirmed all eight. The new phi-projection norm calculation itself passed the investigator's numerical review: certified diagonal Gram bases use the exact coordinate norm, uncertified groups retain exact dense embedding, and chunk sizing bounds temporaries.

### 5. Type, cache, and serialization contracts

The investigator reported four candidates: an E-step alias bypassed cache eligibility; group-product positional composition lost its right frame factor in cached rollout; boolean-like strings survived deserialization and reversed truthiness; and controlled embedding comparison omitted sequence partition fields. Independent probes confirmed all four.

## Verifier verdicts

| ID | Domain | Finding | Verdict | Severity | Executable source |
|---|---|---|---|---|---|
| Q1 | Reliability | Concurrent runs can collide in one artifact directory and race on fixed temporary filenames. | CONFIRMED | High | `train_vfe3.py:505-516,582-588`; `vfe3/run_artifacts.py:294-327` |
| Q2 | Reliability | The ring experiment does not apply or persist the repository's deterministic-runtime contract. | CONFIRMED | High | `vfe3/inference/ring_task.py:165-176`; `vfe3/runtime.py:12-30`; `efe_ring_experiment.py:344-358,393-423` |
| Q3 | Reliability | Scaling and multiseed analysis silently select surviving readable finite results. | CONFIRMED | Medium | `scaling_analysis.py:74-96,244-247,460-464`; `multiseed_analysis.py:36-40,112-129` |
| Q4 | Reliability | Sigma-gate measurement silently reinterprets older checkpoint configs through the current schema. | CONFIRMED | Medium | `sigma_gate_measure.py:73-87` |
| Q5 | Reliability | A sidecar-publication failure can delete a successfully written final figure. | CONFIRMED | Medium | `vfe3/viz/report.py:475-488`; `vfe3/viz/figures.py:2313-2330` |
| Q6 | Reliability | Multiseed resolution accepts zero runs and duplicate seeds. | CONFIRMED | Medium | `train_vfe3.py:655-675` |
| Q7 | Reliability | Non-greedy EFE generation has no explicit recorded generation seed. | CONFIRMED | Medium | `generate_efe.py:43-67,207-226`; `vfe3/model/model.py:2272,2361` |
| Q8 | Packaging | Public attention-graph helpers import NetworkX although the visualization extra omits it. | CONFIRMED | Low | `pyproject.toml:15-27`; `vfe3/viz/figures.py:339-377` |
| Q9 | Reliability | Entry points globally enable `KMP_DUPLICATE_LIB_OK`. | CONFIRMED | Medium | `train_vfe3.py:23-24`; `scaling.py:30`; `ablation.py:45`; `make_figures.py:16`; `scaling_analysis.py:25`; `compare_vocab_figures.py:19` |
| B1 | Objective fidelity | Metropolis frame acceptance omits active gamma/meta and hyperprior-sensitive terms. | CONFIRMED | High | `vfe3/model/model.py:1195-1257,1427-1438,1655-1703` |
| B2 | Objective fidelity | Scored gamma energy omits active RoPE transport. | CONFIRMED | Medium | `vfe3/model/model.py:792-903,979,1692-1699,1849-1922` |
| B3 | Probability contract | Beta/gamma prior mixing normalizes before applying beta support. | CONFIRMED | Medium | `vfe3/model/model.py:2403-2450`; `vfe3/config.py:346,370` |
| B4 | Gauge geometry | Near-orthogonal transpose inversion can violate the exact inverse/cocycle condition. | CONFIRMED | Medium | `vfe3/geometry/transport.py:1298-1344` |
| B5 | Numerical contract | Several objective/update controls accept NaN or infinity and can poison loss. | CONFIRMED | Medium | `vfe3/config.py:952-953,1465-1476,1651-1653,2606-2607` |
| B6 | Gradient scope | Gamma energy has no direct model-frame optimization route. | CONFIRMED | Low | `vfe3/model/model.py:841,1692-1699` |
| D1 | Experiment driver | Scaling failures can end with `ALL ROUTES COMPLETE` and a successful process status. | CONFIRMED | High | `scaling.py:729-751,861-875` |
| D2 | Experiment reachability | Seventeen advertised ablation cells are invalid under baseline `mm_exact`. | CONFIRMED | High | `ablation.py:402-406,663-677,1076-1089,1667-1685`; `vfe3/config.py:2537-2554` |
| D3 | Experiment reachability | Three advertised ablation cells violate independent paired constraints. | CONFIRMED | Medium | `ablation.py:593-607,680-687`; `vfe3/config.py:1024-1043,1361-1375` |
| D4 | Experiment reachability | Six advertised scaling cells combine tied groups with incompatible per-block preconditioning. | CONFIRMED | Medium | `scaling.py:480-490,551-576`; `vfe3/config.py:1024-1031` |
| D5 | Modularity | Lie-group retraction is hardcoded rather than registry selected. | CONFIRMED | Medium | `vfe3/geometry/lie_ops.py:782-828`; `vfe3/config.py:28,1145` |
| P1 | Memory | Diagnostics materialize dense pairwise transports despite an active compact path. | CONFIRMED | High | `vfe3/model/model.py:2528-2577,2606-2632,2777-2803`; `vfe3/train.py:1204-1223` |
| P2 | Compute | Full direct-omega mode pseudoinverts a `K^2 x K^2` Gram matrix each step. | CONFIRMED | High | `vfe3/gauge_optim.py:483-546`; `vfe3/geometry/lie_ops.py:30-45` |
| P3 | I/O | Every finalized ablation run rehashes reused train/validation/test corpora. | CONFIRMED | High | `vfe3/run_artifacts.py:850-908,1338-1348,1680-1686`; `ablation.py:1691-1728` |
| P4 | Compute | Opt-in phi projection scans every embedding row after every successful step. | CONFIRMED | Medium | `vfe3/gauge_optim.py:59-86,108-183`; `vfe3/train.py:663-698` |
| P5 | GPU scheduling | Validation forces host synchronization inside every batch. | CONFIRMED | Medium | `vfe3/train.py:729-776` |
| P6 | Compute | Fixed-point reporting replays inference after a diagnostic snapshot already exists. | CONFIRMED | Medium | `vfe3/train.py:827-903`; `vfe3/viz/extract.py:476-541` |
| P7 | Compute | Report generation performs several independent full passes over one token bank. | CONFIRMED | Medium | `vfe3/viz/report.py:189,224-249`; `vfe3/viz/extract.py:228-314,327-392,1209-1275` |
| P8 | Memory | `tokens_per_char` materializes and decodes an entire memory-mapped split. | CONFIRMED | Medium | `vfe3/data/datasets.py:306-342` |
| T1 | Cache correctness | An E-step alias canonicalizes to unsupported exact MM after cache eligibility admits it. | CONFIRMED | High | `vfe3/inference/belief_cache.py:60-92,173-183`; `vfe3/inference/e_step.py:47-65,761-869` |
| T2 | Cache correctness | Cached group-product rollout drops the right positional frame factor. | CONFIRMED | High | `vfe3/inference/belief_cache.py:60-92,211-217`; `vfe3/model/model.py:968-972` |
| T3 | Serialization | Boolean-like strings can pass validation and reverse behavior through truthiness. | CONFIRMED | Low | `vfe3/config.py:2508-2518,2718-2738`; `vfe3/model/model.py:232` |
| T4 | Comparison contract | Embedding-comparison validation omits sequence count and length. | CONFIRMED | Medium | `vfe3/viz/embedding_comparison.py:277-318,339-359,372-424` |

## Confirmed high-severity punch list

1. **B1 — make Metropolis acceptance evaluate the complete proposal-sensitive objective.** The current scorer can accept a frame proposal that improves the belief block while worsening the actual assembled objective after gamma/meta and hyperprior-sensitive terms are included (`vfe3/model/model.py:1195-1257,1655-1703`).
2. **T1 — canonicalize cache-sensitive E-step modes before eligibility.** `frozen_surrogate_exact` reaches the gradient cache although its effective route is exact MM, producing different beliefs (`vfe3/inference/belief_cache.py:60-92`; `vfe3/inference/e_step.py:47-65`).
3. **T2 — preserve or reject right-frame positional state in cached rollout.** The group-product route carries position in `right_phi`, which the cache drops (`vfe3/inference/belief_cache.py:211-217`; `vfe3/model/model.py:968-972`).
4. **D2 — repair the 17 invalid advertised ablation cells and construction-validate every fully merged arm.** These geometry/divergence cells retain exact MM although they require the autograd route (`ablation.py:402-406,1667-1685`; `vfe3/config.py:2537-2554`).
5. **D1 — propagate scaling failures to persistent artifacts and process status.** Requested routes can all fail while the driver prints `ALL ROUTES COMPLETE` and exits successfully (`scaling.py:729-751,861-875`).
6. **Q1 — reserve run directories atomically and use unique temporary paths.** Two same-second launches with the same settings select one path and fixed temporary filenames (`train_vfe3.py:505-516`; `vfe3/run_artifacts.py:294-327`).
7. **Q2 — apply and persist the deterministic runtime contract in the ring experiment.** The configuration says deterministic, but the route only calls `torch.manual_seed` and does not record effective backend state (`vfe3/inference/ring_task.py:165-176`; `vfe3/runtime.py:12-30`).
8. **P1 — retain compact transport in diagnostics.** At `N=128,K=100`, one dense float32 transport is 625 MiB before covariance and energy intermediates (`vfe3/model/model.py:2528-2577`; `vfe3/train.py:1204-1223`).
9. **P2 — cache or factorize the direct-omega generator metric.** At `K=100`, the per-step Gram has 100 million entries before pseudoinversion (`vfe3/gauge_optim.py:483-546`; `vfe3/geometry/lie_ops.py:30-45`).
10. **P3 — compute immutable corpus hashes once per split and reuse them across cells.** Finalization currently rereads and canonicalizes identical corpus tensors for every ablation run (`vfe3/run_artifacts.py:850-908`; `ablation.py:1691-1728`).

## Targeted verification evidence

| Probe | Exact result |
|---|---|
| Q1 run-path collision | Two calls in one second returned the same `vfe3_runs\\20260715-211313_synthetic_K4_block_glk_linear_s6` path. |
| Q2 deterministic state | `algorithms=False`, `cudnn_deterministic=False`, `cudnn_benchmark=True`, no `CUBLAS_WORKSPACE_CONFIG`, while config `deterministic=True`. |
| B1 proposal objective | `belief_delta=-0.0005837082863`, `gamma_delta=+0.0694887042`, `joint_delta=+0.06890499592`; belief-only and joint decisions disagree. |
| B3 support mixing | Pre-mask row sums were `[0.625292, 0.750448, 0.874916, 1.0]`, proving normalization included mass outside beta support. |
| B4 inverse/cocycle | Transpose inverse error about `6.02e-05`; true inverse error about `7.02e-08`. |
| B5 nonfinite config | NaN/Inf controls constructed successfully; `lambda_beta=inf` produced nonfinite loss. |
| D2/D3 ablation expansion | 20 invalid cells: 17 exact-MM route mismatches and 3 independent prerequisite failures. |
| D4 scaling expansion | 6 invalid tied-group/per-block-preconditioner cells. |
| P1 memory arithmetic | `655,360,000` bytes = `625.0 MiB` for one `N=128,K=100` float32 pairwise transport. |
| P2 Gram arithmetic | `K=100`, `n_gen=10,000`, `100,000,000` Gram entries. |
| T1 cache equivalence | Cache reported supported; cached/full maximum belief difference was at least `0.0012455`. |
| T2 group-product cache | Cache reported supported with nonzero `right_phi`; cached/full maximum difference was `0.00097084`. |
| T3 truthiness | Serialized `'False'` remained a string and evaluated true. |
| T4 partition contract | Validation accepted incompatible `(10,10)` and `(11,5)` sequence partitions. |

The 17 D2 cells were `transport_mode={regime_ii, regime_ii_covariant}`, `cocycle_relaxation={0.0, 0.5, 1.0}`, `pos_rotation=rope`, `rope_base={10, 100, 1000}`, `rope_full_gauge={means_only, full_gauge}`, `covariance=full`, and `renyi_order={0.5, 0.8, 1.2, 1.5, 2.0}`. The three D3 cells were `gauge_group=tied_block_glk`, `gauge_group=so3_spin2x4`, and `cross_couplings=pair_0_1`. The six D4 cells were `blocks_K48_tied_2x/{K48_GL3, K48_GL6, K48_GL8, K48_GL12, K48_GL24}` and `group/K64_tied_h8`.

## Overlap and scope notes

Q3 and D1 form one causal chain but are distinct defects: the driver suppresses or loses failed cells, and the analysis then silently fits survivors. Q1 and Q6 interact because duplicate seeds raise collision probability, but Q1 also applies to independent concurrent processes. B2 and B6 both concern gamma/frame fidelity; B2 changes the scored value, while B6 records an intentionally passive direct gradient route. P1, P6, and P7 are separate repeated-computation sites. T1 and T2 are independent cache failures.

The audit found no neural-layer violation in production source. The pure-path controls remain present, and the new phi-projection norm optimization retained exactness on both certified and fallback group bases. These negative checks do not offset the confirmed route-specific defects above.

## Verification and limitations

The default command was `python -m pytest -x --junitxml=C:\\tmp\\vfe3-ultradeep-audit-baseline-20260715.xml`. JUnit reported `tests=2927`, `failures=0`, `errors=0`, `skipped=31`, and the console reported `2896 passed, 31 skipped` in 127.53 seconds.

The extended command was `python -m pytest --runslow -x --junitxml=C:\\tmp\\vfe3-ultradeep-audit-runslow-20260715.xml`. JUnit reported `tests=2927`, `failures=0`, `errors=0`, `skipped=13`, and the console reported `2914 passed, 13 skipped` in 419.91 seconds. The residual skips comprise six tests tied to a removed closure-ledger document, six CUDA-only smokes, and one external baseline/feature-bundle identity probe.

The interpreter is `torch 2.11.0+cpu`; `torch.cuda.is_available()` returned false, so no CUDA result is inferred. An in-memory AST parse reported `tracked_python_files=247`, `syntax_ok=247`, and `syntax_errors=0`. `python -m pip check` reported no broken requirements. Ruff was unavailable (`No module named ruff`). A repository-wide `compileall` was not accepted as evidence because permission-denied writes into test-created `__pycache__` directories made that command incomplete; the non-writing AST parse replaced it.

## Recommended remediation order

The first repair tranche should address B1, T1, T2, D2, and D1 because they can change model behavior or invalidate advertised experiments. The second tranche should address Q1, Q2, P1, P2, and P3 because they threaten artifact integrity, reproducibility, or feasible scaling. The remaining medium and low findings can then be grouped by subsystem: gamma/objective fidelity, configuration finiteness and experiment construction, reporting atomicity and comparison contracts, and offline/runtime performance.

No fixes should be combined with this audit commit. Each remediation tranche should add focused regression tests, preserve a mathematically pure path under explicit toggles, and finish with machine-readable default and slow-suite verification.
