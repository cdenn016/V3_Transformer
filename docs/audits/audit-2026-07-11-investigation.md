# Post-Merge Investigation of `audit-2026-07-11.md`

## Scope and evidence boundary

This investigation re-checked the 22 findings in `docs/audits/audit-2026-07-11.md`
against executable source after PR #164 merged. The audit report was committed as `9770a85`
with parent `d78bf16` and merged to `main` as `10c146b`. The investigators worked from a
clean worktree at `10c146b`. Before this report was written, `origin/main` advanced to
`fc6fecd`; that intervening range changed only `README.md`, the dated edit note, and two
README planning documents. None of the source or test files cited below changed.

The review used source and Git history, not code comments as behavioral proof. The Research
vault pages `[[VFE Transformer Program]]`, `[[Killing form]]`, `[[Fisher information metric]]`,
and `[[SPD-manifold geometry and Riemannian optimization]]` supplied theory context. The user
directed that tests run only after code changes. No implementation code was changed, so no test,
probe, or benchmark result is part of this investigation. An initially started baseline was
stopped, its partial JUnit file was deleted, and none of its output was used.

## Audit provenance corrections

The original scope line does not identify the committed snapshot. Its `dafdc45` statement may
describe the branch's historical fork point, but the report commit's actual parent is `d78bf16`.
The range from `dafdc45` to `d78bf16` includes the gamma mean-transport change in
`vfe3/model/model.py`, its regression, and the dated edit note. The report should therefore
identify `d78bf16` as the committed code snapshot and reserve `dafdc45` for the branch-creation
claim, which Git parentage alone cannot prove or refute.

The F1 challenge mixed committed source with uncommitted live configuration. In committed
source, `compact_phi_block_transport` appears only as the `False` default at
`vfe3/config.py:706`; `train_vfe3.py`, `ablation.py`, and `scaling.py` do not override it.
The challenge's cited `train_vfe3.py:400` assignment exists only in the user's dirty live
checkout, where `compact_phi_block_transport=True` is intentional WIP. At the merged snapshot,
line 400 is an `exp_fp64_mode` setting. The WIP train and ablation configurations bypass F1,
but committed train, ablation, and scaling configurations do not. That distinction was not
recorded in the audit scope and changed the reachability argument.

The original six pytest failures do not establish a code baseline failure. The missing
`docs/audits/curated-audit-closure-ledger-2026-07-09.md` exists in `dafdc45`, `d78bf16`,
`9770a85`, and merged `main`; its absence was an uncommitted live-checkout deletion. The report
correctly called the deletion pre-existing, but the run was not isolated from user WIP and no
retained JUnit artifact was available for this investigation. The numeric test result was not
reused or re-run.

## Adjudicated disposition

Using the worst affected committed workload, the 22 original findings resolve to two High,
one Medium, eleven Low, seven Informational, and one Refuted. F1 is split by workload: High on
the active scaling route and Medium on the K=20 train/ablation baseline. No theory, objective,
or numerical-correctness defect was found on the normal float32 path.

| ID | Verdict | Final severity | Investigation result |
|---|---|---|---|
| F1 | Confirmed; original challenge premise refuted | High on active scaling; Medium at K=20 | The dense BCH path copies and hashes the full CUDA basis before every cache lookup. Committed click-run configurations do not enable the compact bypass. |
| F2 | Confirmed as a test-oracle gap | Medium | Production decode is correct, but the golden comparison uses a local encoder-table twin instead of `PriorBank.reference_decode`, so off-default decode accessors lack an exact KL pin. |
| F3 | Confirmed | Low | Bare `Callable` annotations are typing hygiene. No type checker is configured and bad registrations fail at invocation. |
| F4 | Confirmed; original mechanism understated | High on `killing_per_block` | Fresh contiguous sub-bases miss the strong-reference cache on every active preconditioner call, producing O(steps) device-memory retention within one valid run. |
| F5 | Split | Informational | The registries are unused by built-in production diagnostics, but remain executable standalone extension APIs. No runtime defect follows. |
| F6 | Split | Informational | `coordinate_dim` is an unused internal capability but a valid abstract family-interface contract. Adding a synthetic call would not improve correctness. |
| F7 | Confirmed | Low | `_as_coeff` accepts tuples while its annotation and docstring mention only lists. Runtime behavior is correct. |
| F8 | Confirmed | Low | `data_state` has a precise tensor-and-integer schema but is annotated as `Dict[str, object]`. Internal callers honor the schema. |
| F9 | Confirmed behavior; overclassified as a defect | Low roadmap item | Plain generation recomputes encode and E-step per token but decodes only the last position, warns about cost, and is explicitly a correct-first implementation. |
| F10 | Confirmed | Low | A plotting exception after figure creation can leak one Matplotlib figure because the helper closes only a successfully returned figure. |
| F11 | Split | Low | The code and dashboards consistently use one-half of the mean-block Fisher trace; the name and explanatory prose conflate that KL quadratic coefficient with the full Fisher trace. |
| F12 | Split | Low | `retract_spd_full` restores the input dtype on return but computes float64 inputs at float32 accuracy. The shipped pipeline is intentionally float32. |
| F13 | Confirmed, with a broader guard gap | Low | The helper has no local positivity guard, and `query_tau_c < 0` validation also admits NaN and positive infinity. The option is off in the click-run configuration. |
| F14 | Confirmed | Low | Direct invalid callers can bypass the `mm_damping` contract; routed configuration enforces a positive convex blend, under which `lam_new` cannot be zero. |
| F15 | Split | Low; numerical impact unmeasured | Compact BCH follows autocast on an opt-in live-phi path, but dense BCH lacks an fp32 island too. Positional BCH executes before autocast. Any correction must cover both paths. |
| F16 | Confirmed | Informational | `self_coupling_alpha` is an unused import in `vfe3/viz/extract.py`. |
| F17 | Confirmed | Informational | `coverage_lines` is an unused import in `scaling.py`. |
| F18 | Confirmed | Informational | `math`, `os`, and `typing.List` are unused imports in the three cited modules. |
| F19 | Confirmed | Low | Live diagnostic out-parameters and figure return lists have imprecise annotations but correct behavior. |
| F20 | Confirmed behavior; not a defect | Informational | The field annotations describe normalized tuple pairs; JSON list pairs are intentionally accepted and normalized during deserialization. |
| F21 | Refuted | None | `"gaussian"` is an intentional semantic alias used by `check_admissible` at `vfe3/geometry/groups.py:270-315`, not a stale family-registry entry. Removing it would break that API. |
| F22 | Confirmed cost; overclassified as a defect | Informational | The first transport serves the belief substep; the second is attached to a fresh `phi_g` leaf after the mean/covariance update. This preserves the current Gauss-Seidel and autograd-island contract. |

## Material finding F4: per-call `killing_per_block` retention

`build_killing_preconditioner_per_block` constructs each local basis with advanced indexing and
`.contiguous()` at `vfe3/geometry/phi_preconditioner.py:209-216`. Every invocation therefore
creates fresh storage. `build_killing_preconditioner` keys that storage by `data_ptr` at
`phi_preconditioner.py:134-136` and stores a strong `(generators, inverse)` pair at
`phi_preconditioner.py:149`. Retention prevents pointer reuse, so later fresh sub-bases cannot
hit the old entries. The global dictionary has no bound or clear operation.

The named `gauge_mstep_optim` ablation selects `m_phi_natural_grad=True` and
`phi_precond_mode="killing_per_block"` at `ablation.py:736-746`. `GaugeNaturalGradAdamW.step`
calls the preconditioner once for each gauge parameter with any active gradient row at
`vfe3/gauge_optim.py:458-476`. The K=20, H=2 baseline has two d=10 blocks. Per call, each block
retains a `(100,10,10)` basis and `(100,100)` inverse, or 160,000 float32 bytes across both
blocks. If both `phi_embed` and learned `pos_phi_free` remain active, the source-derived upper
bound is 320,000 bytes per accepted step, about 4.47 GiB over the configured 15,000-step cell,
before allocator and dictionary overhead. This is an upper bound rather than a measured
allocation because source alone cannot prove that both parameter gradients are active on every
step. Both adversarial reviewers upheld High: the sweep is opt-in, but it is valid and directly
selectable, and the retention grows with active steps inside one run. `gc.collect()` and
`torch.cuda.empty_cache()` cannot free tensors still owned by the global dictionary.

The correct repair boundary is the stable parent basis, not ablation cleanup. Cache one completed
per-block inverse by parent-basis identity plus block metadata, or precompute it once in the
optimizer and pass it through `inv_metric`. Merely clearing between arms would leave the
within-run growth intact.

## Material finding F1: synchronous basis hashing

`_basis_value_signature` executes `generators.detach().contiguous().cpu()` and SHA-256 at
`vfe3/geometry/lie_ops.py:140-146`. `warn_if_basis_not_closed` computes that signature before
consulting `_BRACKET_CLOSURE_RES` at `lie_ops.py:197-199`; cache hits therefore avoid only the
bracket scan, not the device synchronization, host copy, byte materialization, or hash.
`compose_bch` invokes the diagnostic at `lie_ops.py:546-547`.

Committed train, ablation, and scaling select learned positional BCH while inheriting
`compact_phi_block_transport=False`. With the tied model frame, there is one positional
composition per forward. At K=20, H=2, the generator basis is about 0.305 MiB, so Medium is the
appropriate K=20 severity without a measured timing result. The active scaling route uses
K=60,80,100 with H=4, producing source-derived basis transfers of about 12.36, 39.06, and
95.37 MiB per forward. The route specifies three seeds and 60,000 steps, so the repeated copy,
byte conversion, and hash constitute a High-severity performance defect on that committed
workload even though they do not change numerical values. The live uncommitted train and
ablation WIP enable the compact early return and bypass this path; committed scaling does not.

The repair should attach bracket-closure metadata to the immutable `GaugeGroup`, or use a bounded
identity/version-aware cache whose lookup precedes any value hash. The closure result depends on
the fixed basis, so recomputing a content signature in every forward is unnecessary.

## Material finding F2: off-default golden-oracle gap

`PriorBank.reference_decode` correctly reads `_decode_mu_table()` and
`_decode_sigma_log_table()` at `vfe3/model/prior_bank.py:671-674`. The local test oracle in
`tests/test_prior_bank.py:9-19` instead reads `mu_embed` and bare
`exp(sigma_log_embed)`. It is therefore a valid oracle only for the default tied token-prior bank
while its values stay away from the bounded-variance floor. Existing tests cover untied-table
initialization, perturbation isolation, gradient landing, model-channel routing, and static source
guards, so this is not an active production-scoring defect. What remains absent is an exact
seam-value comparison for `untie_decode_bank=True` and `prior_source="model_channel"`.

The smallest durable correction is to call `pb.reference_decode(...)` from the golden test and
parameterize it over tied, untied, and model-channel banks, then delete the duplicated local twin.

## Independent verifier record

A fresh verifier independently re-read all 22 findings at `10c146b` without tests or runtime
probes. It returned 19 Confirmed and three Split verdicts, elevated F1 and F4 to High, retained F2
at Medium, and agreed that the remaining findings were Low or Informational. Its F21 conclusion
treated `"gaussian"` as an inert stale value because it checked only registered family names. The
contract specialist's subsequent executable call-path trace showed that `check_admissible`
deliberately accepts and routes the alias at `vfe3/geometry/groups.py:270-315`; the final
adjudication therefore refutes F21. The F1 and F4 severity disagreements then went to the
adversarial challenges below.

## Adversarial challenge record

| Finding | Skeptic | Defender | Adjudication |
|---|---|---|---|
| F4 per-call cache retention | Upheld High; default-off status does not defeat a directly selectable valid arm whose retained storage grows with active steps. | Upheld High; the call chain, fresh storage, strong ownership, and source-derived several-GiB exposure are complete. | **UPHELD HIGH**, scoped to `killing_per_block`; actual OOM on a particular run remains unmeasured. |
| F1 basis copy and hash | Downgraded to Medium for K=20 because source alone does not establish wall-time share or failed runs. | Upheld High for the active scaling route because the source requires 12.36-95.37 MiB synchronized copies per forward across 60,000-step, three-seed cells. | **SPLIT:** High on committed scaling, Medium on K=20 train/ablation. |

## Recommended fix order

The first repair should be F4 because it is an unbounded lifetime defect on a valid path and can
consume several GiB within one cell. F1 follows because it is unconditional on the committed
scaling route and has a narrow cache-ownership solution. F2 is third because it restores the
project's golden-regression claim without changing production behavior. The remaining low items
can be batched by concern: F10 figure cleanup and F13/F14 boundary validation are behavioral
hardening; F7/F8/F11/F19 are type or naming corrections; F15 needs an explicit dense-and-compact
AMP policy before any edit. F5, F6, F9, F12, F16-F18, F20, F21, and F22 should not block work.

No fixes were applied in this investigation.
