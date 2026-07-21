# Ultra-deep eight-agent codebase audit, 2026-07-20

Audit date: 2026-07-20

Audited revision: `7a9d7d86a02d80d78d7158d542bdf8fd84a28904`, the fetched
`origin/main` revision at audit start.

Audit worktree: isolated branch `codex/ultradeep-audit-20260720`, created from
the fetched remote revision. The user's live checkout and its pre-existing
working changes were not altered during investigation.

Scope: all Python source and executable root drivers in the repository,
including configuration, data loading, model construction, inference,
optimization, geometry, variational objectives, evaluation, diagnostics,
artifact integrity, ablation, scaling, visualization, and process boundaries.
The audit did not modify production code and did not run a full training job.

## Executive conclusion

The audit closed one High, seventeen Medium, and three Low findings. Nine
additional candidates were refuted after exact source-path review because the
code already discloses their approximation status, rejects them before
execution, or preserves an exact opt-in path. No Critical finding was
established.

The High finding is on the active click-to-run route. Periodic validation
diagnostics decode and clone a dense float32 `(256, 128, 50257)` logits tensor.
The original and clone alone require 13,174,571,008 bytes, or 12.2698 GiB,
before the snapshot's beliefs, attention maps, cross-entropy workspace, model,
and allocator state are counted. This allocation is attempted seventy times
over the configured 105,000-step run. The broad diagnostic exception handler
can preserve training and checkpoint selection after a CUDA allocation error,
but then all held-out structural diagnostics for that evaluation are replaced
with NaNs. The finding is High because the device-scale allocation and loss of
the run's scientific diagnostic record are deterministic consequences of the
active path. This audit did not claim or reproduce an out-of-memory process
termination.

The most consequential Medium findings concern provenance and numerical
integrity. An uncapped binary token cache remains backed by a mutable memmap
after its identity is frozen; selected-checkpoint finalization bypasses the
available code and validation-data identity checks; resume accepts NaN and
infinite tensors; invalid fused-decoder class IDs receive a finite loss;
empty evaluation is reported as CE zero and PPL one; fractional E-step
truncation silently removes the entire E-step graph; and the full-Gaussian
float32 self-KL route can generate a large false covariance gradient at a
condition number admitted by the configured SPD bounds. Other findings cover
registry incompleteness, experimental completeness, dormant counted
parameters, unconditional duplicate held-out evaluation, and exact-type
validation gaps.

The complete CPU suite is not green at the audited revision. Its JUnit record
contains 3,904 cases, 10 failures, 0 errors, and 37 skips. The CUDA-marked lane
contains 22 cases, all passing. The audit does not infer that the ten failing
tests are caused by the findings below; several are stale contract or fixture
expectations, while two reach a non-SPD Cholesky failure. They remain an open
baseline obligation.

## Method and evidence standard

Eight distinct expert lanes ran in parallel waves, with four simultaneous
agents at a time because the runtime exposes four total concurrency slots.
The lanes covered code quality and security, runtime debugging, configuration
and refactoring seams, performance, Python API contracts, numerical analysis,
gauge and differential geometry, and variational and information-geometric
semantics. Each investigator read executable branches and active
configuration. Comments and docstrings were used only to locate a claimed
contract; they were not accepted as behavioral evidence.

A ninth, independent verifier re-read all thirty candidate claims at the exact
audited revision. It reproduced selected defects without changing files and
classified every candidate as sustained or refuted. Each initially High
candidate then received a claim-specific adversarial challenge. The challenge
downgraded the memmap, resume, and full-Gaussian findings to Medium because
their current reachability or demonstrated end-to-end blast radius did not
support High. The diagnostic allocation remained High after both skeptic and
defender review. A deterministic verification ledger was validated in closure
mode after the report revision was committed.

The audit also consulted the Research vault's `VFE Transformer Program`,
`Gauge equivariance and geometric deep learning`, `Information geometry and
natural gradient`, `SPD-manifold geometry and Riemannian optimization`,
`Variational free energy and predictive coding`, `GL(K) gauge-equivariant
attention`, and `Meta-entropy` pages. Those pages supplied program context,
not authority for code behavior. Mathematical closure below comes from exact
derivation and executable source inspection.

## High finding

### H1. Active validation diagnostics have a 12.2698-GiB logits-only copy pair

Location: `train_vfe3.py:80-88,340`, `vfe3/train.py:921-926,1450-1452,1556-1566`,
and `vfe3/model/model.py:158-164,2571-2576,2617-2619`.

Severity: High.

The active configuration uses vocabulary size 50,257, sequence length 128,
validation batch size 256, 105,000 steps, and an evaluation interval of 1,500.
With artifacts enabled, every evaluation invokes `_val_diagnostics`, which
takes the full first validation batch and calls `build_diagnostic_snapshot`.
That snapshot requests full logits and passes them through `_freeze_tensor`,
whose float32 branch executes `detach().clone()`.

The exact lower-bound arithmetic is
`256 * 128 * 50257 * 4 = 6,587,285,504` bytes for one dense tensor and
13,174,571,008 bytes for the original plus clone. A current small-shape probe
confirmed that `_freeze_tensor` returns distinct storage. A current cache probe
also confirmed that the first validation batch has shape `(256,128)`. The full
run schedules `105000 / 1500 = 70` such attempts. If the allocation succeeds,
the run repeatedly pays the device-scale transient. If it raises, the handler
at `vfe3/train.py:1562-1566` catches the exception and resets all held-out
diagnostics to NaN before continuing to best-model saving. The handler lowers
the process-termination risk but does not preserve the diagnostic evidence.

Recommended remediation: make validation diagnostics operate on an explicit,
small diagnostic batch or position slice before decoding; compute positional
cross-entropy in vocabulary chunks; and omit dense logits from the durable
snapshot unless a caller explicitly requests them. Add a CUDA regression that
records peak allocated bytes for the active diagnostic shape or a bounded
surrogate whose allocation formula is asserted.

## Medium findings

### M1. Uncapped binary-cache data can drift after its recorded identity is frozen

Location: `vfe3/data/datasets.py:326,339-364,531,560`.

The `.bin` loader returns a tensor backed by a live read-only NumPy memmap. The
identity guard hashes immediately before and after loading, then stores that
identity on `TokenWindows`; it does not bind future reads to immutable bytes.
A current synthetic-cache probe loaded a first window `[0,1]`, retained source
SHA-256 `cd9a54ed1f18bf97db08914e280ea7349e11ca2c4885a4d8052552ceba84208d`,
mutated token zero in place after loading, and then read `[5,1]` from the
already-created dataset while its recorded identity remained unchanged.

The active WikiText-103 cache selects the `.pt` branch, and the repository has
no ordinary post-load cache writer. Those reachability facts calibrate this to
Medium. Recommended remediation: hold an owned immutable view for identity-
bound runs, or revalidate content at defined run boundaries and fail before
publishing artifacts if the selected source changes.

### M2. A symlinked binary cache can split metadata identity from interpretation

Location: `vfe3/data/datasets.py:235-252,322-326`.

`cache_source_identity` resolves the binary payload and parses the metadata
beside the resolved target, but hashes the metadata path beside the unresolved
link. `load_cached_tokens` parses the unresolved link-side metadata. A link
across directories can therefore identify target-side metadata while loading
the same bytes under different link-side metadata. Recommended remediation:
resolve the payload once and derive every payload, metadata, provenance, and
loader path from that one resolved parent, rejecting split ownership.

### M3. Final selected-model evaluation bypasses code and data identity validation

Location: `vfe3/run_artifacts.py:752-827,2656-2676`.

The codebase contains a reusable selected-bundle validator that checks code
identity and validation-data identity. `finalize_run` does not call it; it
loads `best_model.pt`, verifies only bundle shape and configuration
fingerprints, and installs the weights. A swapped or stale selected bundle
with a matching selection configuration can therefore be used for headline
test evaluation without the stronger identity contract. Recommended
remediation: make finalization use the same nonmutating validator as every
other best-model consumer before `load_state_dict`.

### M4. Resume validation accepts NaN and infinite model tensors

Location: `vfe3/run_artifacts.py:851-872,1777-1778,1853`.

`_validate_checkpoint_model_state` checks keys, tensor type, shape, dtype, and
layout but never finiteness. A current probe passed correctly shaped tensors
containing NaN, positive infinity, and negative infinity; all three were
accepted. An adversarial end-to-end probe then showed that poisoned active
weights yield nonfinite loss, cause `train_step` to skip the update, and remain
poisoned. This is an opt-in resume route rather than the active fresh-run
default, which calibrates severity to Medium. Recommended remediation: reject
nonfinite floating or complex model tensors before any live tensor is copied,
and cover raw resume and best-model bundles with the same predicate.

### M5. Fused chunked cross-entropy silently scores invalid class IDs

Location: `vfe3/model/model.py:1554-1571` and
`vfe3/model/prior_bank.py:896-931,1034-1063,1110-1136`.

The chunked decoders count every target other than `-100` as valid. A target
below zero or at least the vocabulary size belongs to no chunk, yet the fused
path returns a finite loss. A current probe returned approximately 2.07959 for
both target `-1` and target `V`; dense PyTorch cross-entropy raised
`IndexError`. Recommended remediation: reject every non-ignored target outside
`[0,V)` before chunk reduction and add parity tests for both invalid bounds.

### M6. Empty or all-ignored evaluation is reported as perfect perplexity

Location: `vfe3/train.py:795-867`.

Evaluation divides by `max(total_tok,1)`. A current all-ignored reproduction
returned `ce=0.0`, `ppl=1.0`, `bits_per_token=0.0`, and `bpc=None`. These
metrics are undefined, not perfect. Recommended remediation: fail closed on
zero valid targets, or return an explicitly invalid result that cannot enter
selection, summary, or comparison logic.

### M7. Fractional E-step truncation silently severs the complete E-step graph

Location: `vfe3/config.py:706,2493-2496` and
`vfe3/inference/e_step.py:1361-1372,1378-1409`.

The annotated integer `e_steps_backprop_last` is only checked for negativity.
The value `0.5` is accepted. With three iterations, `n_total - 0.5` places
every integer-indexed iteration under `no_grad`, while the equality-based
reattachment boundary can never be reached. A current probe showed that values
zero and one both retained a nonzero prior gradient, while `0.5` returned an
output with `requires_grad=False`, no `grad_fn`, and no prior gradient.
Recommended remediation: require a plain integer, excluding booleans, before
range validation.

### M8. The additive phi encoder can receive a group-chart optimizer update

Location: `vfe3/model/prior_bank.py:1405-1425`, `vfe3/train.py:201-218`, and
`vfe3/gauge_optim.py:810-845`.

Under `per_token_additive`, `phi_embed` is used as ordinary additive code. The
optimizer grouping nevertheless attaches the configured phi update policy
unconditionally, and the accepted additive-plus-`pullback_group` combination
therefore sends that code through a group-chart update. Recommended
remediation: either reject this semantic combination or give the additive
code an ordinary Euclidean optimizer group distinct from the geometric frame
coordinate.

### M9. Transport registration cannot add a new trainable state kind by registration alone

Location: `vfe3/geometry/transport.py:331-420`,
`vfe3/inference/e_step.py:136-165`, and `vfe3/model/model.py:375-416`.

The registry routes mean, covariance, and batch-independence behavior, but
trainable transport state is hardcoded as `connection_W`, `connection_M`, and
`connection_L` in model construction and call sites. A stateful transport
variant therefore requires editing central consumers, contrary to the
repository's registry-behind-every-seam constraint. Recommended remediation:
let a transport registration declare and construct its state bundle, expose a
single typed state handle to callers, and own its serialization contract.

### M10. Float32 full-Gaussian self-KL can produce a large false covariance gradient

Location: `vfe3/families/gaussian.py:491-525` and the accepted full-covariance
gradient route through `vfe3/geometry/retraction.py:516-543,708-749`.

For identical Gaussian operands, the analytic KL is exactly zero: the trace
term is `K`, the mean term is zero, and the two log determinants cancel. Its
total derivative with respect to the shared covariance is also zero. The
implementation computes in float32 unless an operand is float64. In a current
twenty-seed probe, seed 17 produced identical 4 by 4 SPD operands with
condition number about 950,197, self-KL 0.0126829, and maximum absolute shared
covariance gradient 8,645.65. The float64 evaluation produced self-KL
`1.44e-11` and maximum gradient `9.64e-06`.

The active click-run uses the diagonal family. Reaching this defect requires
the accepted combination of full covariance, scalar alpha mode, and gradient
E-step; the retraction also Fisher-preconditions and bounds the eventual
update. Those facts calibrate the raw numerical defect to Medium pending an
end-to-end corrupted-trajectory reproduction. Recommended remediation: use a
float64 island for the full-Gaussian KL factorization and cancellations, then
cast the scalar result back under an explicit dtype policy; add self-KL value
and shared-gradient tests near the admitted condition-number ceiling.

### M11. Growing-sequence extrapolation can be declared complete without its large-N tail

Location: `ablation.py:201-208,2331-2350,2735-2777`.

The configured extrapolation reuses batch size 32 through sequence length 512.
Failed points are dropped, while the completeness contract requires only two
surviving points. A base length plus one nearby point can therefore satisfy
completion even when the intended large-N tail failed for memory or runtime
reasons. Recommended remediation: declare mandatory sequence lengths or a
minimum maximum-N threshold, reduce batch size with N, and persist failure
reasons in the aggregate rather than silently shrinking the fitted domain.

### M12. Most boolean configuration fields do not require exact booleans

Location: `vfe3/config.py:2392-2420`.

Strict exact-type validation covers only a small subset of booleans. A current
constructor probe accepted `include_attention_entropy="false"` unchanged;
downstream truthiness enables the option. Recommended remediation: validate
every dataclass boolean with `type(value) is bool` before any compatibility
coercion or branch.

### M13. Core integer configuration fields accept booleans and fractional numbers

Location: `vfe3/config.py:914-920,2392-2409,2482-2496`.

Current constructor probes accepted `n_layers=1.5` and `n_layers=True`.
Similar count fields rely on comparisons rather than exact-integer contracts;
M7 is the reachable gradient consequence for one such field. Recommended
remediation: centralize a plain-integer validator, explicitly exclude booleans,
and apply it to all dimensions, counts, cadences, and iteration depths.

### M14. Ablation diagnostic flags coerce strings before validation

Location: `ablation.py:2712-2721,2758-2777,3718-3721`.

Sweep values pass through `bool()` before the strict configuration contract
can inspect them, so the literal string `"false"` becomes recorded `True`.
Recommended remediation: validate the raw sweep value as an exact boolean and
remove truthiness conversion from experiment construction.

### M15. Active routing carries dormant vocabulary tables in parameter counts and optimization

Location: `vfe3/model/prior_bank.py:347-348,679-705`,
`vfe3/train.py:216-230,339-352`, and `train_vfe3.py:120,290-292`.

The active model-channel plus linear-decode route reads neither the base mean
table nor the base variance table, yet both remain allocated, optimizer-
grouped, and included in realized parameter counts. This can distort capacity-
matched comparisons and optimizer diagnostics even when the dormant tables do
not affect outputs. Recommended remediation: do not construct dormant tables
for that route, or mark them as nonparameters excluded from optimizers and
scientific capacity accounting.

### M16. Finalization unconditionally performs a second full held-out test evaluation

Location: `vfe3/run_artifacts.py:2681-2689,2699-2717`.

When a test loader exists, finalization evaluates the selected model under the
configured E-step budget and then evaluates the complete test set again with
`n_e_steps=0`. There is no toggle or bounded-batch option. Recommended
remediation: make the diagnostic comparison explicit and opt in, or compute it
on a declared diagnostic subset that cannot be mistaken for the headline
held-out result.

### M17. String booleans can shuffle evaluation and omit its tail

Location: `vfe3/data/datasets.py:572-627`.

`make_dataloader` has no exact-type guard for `shuffle` or `drop_last`. A
current PyTorch reproduction showed that `shuffle="false"` selects a random
sampler. The same truthiness prevents padded-final-window construction, so an
evaluation caller can both randomize order and omit the final partial window.
Recommended remediation: require exact booleans at this public boundary before
constructing `TokenWindows` or `DataLoader`.

## Low findings

### L1. Figure-worker finalize fields use permissive coercion

Location: `vfe3/viz/figure_worker.py:413-433`.

Finalize-mode JSON fields are coerced instead of schema-validated;
`bool("false")` enables `allow_large`. Invalid devices usually fail later,
which limits impact. Require exact JSON types before dispatch.

### L2. A bare string process-tree command is decomposed into characters

Location: `vfe3/process_utils.py:176-188,205-216`.

A string satisfies the annotated sequence contract and is expanded one
character at a time into the child command. Reject string and bytes values at
the boundary and require a nonempty sequence of nonempty strings.

### L3. Boolean irrep multiplicity is accepted as one

Location: `vfe3/config.py:944-979` and
`vfe3/geometry/groups.py:475-480,522-527`.

`isinstance(True,int)` admits a boolean multiplicity, which later becomes
count one. Require `type(multiplicity) is int` and a positive value.

## Refuted candidates and negative theory checks

The independent verifier refuted nine candidates. These refutations matter
because they prevent disclosed approximations or intentional seams from being
reported as hidden mathematical defects.

The pure-path report was alleged to certify adaptive temperature or layer
normalization. It does not: `vfe3/run_artifacts.py:3202-3217` explicitly scopes
its booleans to named axes, records adaptive temperature in machine-readable
toggles, and persists the complete serialized configuration. The active route
has adaptive temperature off and both normalization routes set to `none`.

The diagonal Gaussian family is not a representation of general GL congruence,
but the code does not claim that it is. The family registry excludes it from
the invariant families, and `vfe3/run_artifacts.py:3241-3252,3330-3339`
records diagonal Regime II as `diagonal_projection_approximation`. Full
covariance remains the exact route.

An untied block-GL head mixer is likewise not an intertwiner after nontrivial
cross-head mixing. `vfe3/config.py:2611-2640` labels that combination
`independent_head_nonintertwiner`, and the artifact report makes gauge purity
false. Mixer-off and tied full-covariance paths remain available.

Finite BCH positional composition is recorded as a truncation, exposes a
residual gate and fidelity statistics, and has non-BCH alternatives. Means-
only RoPE emits an affine-incoherence warning and is off in the active config.
The `gauge_fixed` registry stub raises during configuration and cannot become
a silent executable route. Unknown serialized configuration fields warn in
compatibility mode and fail under the strict checkpoint paths. The gradient
finiteness scan and gradient clipping serve different requirements; the cited
implementation did not establish a duplicate forced host synchronization.

The variational investigator found no sustained sign, scaling, or envelope
defect. `attention_tau` implements `kappa * sqrt(d)`, attention logits use
negative energy divided by temperature, and the reduced canonical envelope is
`-tau * log Z`. Direct constrained minimization of
`sum_j beta_j E_j + tau sum_j beta_j log(beta_j/pi_j)` gives
`beta_j proportional to pi_j exp(-E_j/tau)` and the same reduced envelope.
The entropy-included and entropy-suppressed objectives are explicit separate
seams. The deployed target-blind structural EM route was not reclassified as
coordinate variational EM without an internal executable contradiction.

## Investigator lane summaries

The code-quality and security lane found the mutable cache, symlink metadata,
selected-model identity, and nonfinite-checkpoint seams. The runtime-debugging
lane found invalid fused targets, perfect empty evaluation, and fractional
truncation. The configuration and refactoring lane found the additive-code
optimizer mismatch, stateful transport registry gap, and dormant tables while
also identifying which pure-path suspicions required independent review.

The performance lane found the validation-logit allocation, incomplete
large-N contract, and duplicate held-out evaluation. The Python-contract lane
found exact-type defects at configuration, ablation, loader, worker, process,
and irrep boundaries. The numerical lane reproduced the full-Gaussian self-KL
cancellation failure. The gauge and differential-geometry lane produced
counterexamples for diagonal congruence and untied head mixing, after which
the verifier established that both are already disclosed and excluded from
the exact path. The variational and information-geometric lane verified the
temperature, Gibbs stationary point, reduced free-energy sign, and explicit
entropy seam and closed no new defect.

## Current mechanical verification

All evidence below was produced at
`7a9d7d86a02d80d78d7158d542bdf8fd84a28904` before this documentation-only
commit. An AST parse covered 266 Python files and reported no syntax failures.

The complete CPU command was `C:\Python314\python.exe -m pytest
--junitxml=.audit-evidence\full-cpu.xml`. The JUnit SHA-256 was
`8C9DC7D0D4254F2DCF18834A7A9C7F178698037853726976204F8813427A4D6B` and
recorded 3,904 cases, 10 failures, 0 errors, 37 skips, and 345.924 seconds.
The failing node IDs were:

1. `tests.test_2026_07_15_data_integrity_remediation::test_training_log_always_names_bits_per_token_when_bpc_is_available`
2. `tests.test_ablation_artifact_resume_20260712::test_missing_requested_diagnostics_output_forbids_contract_publication`
3. `tests.test_ablation_artifact_resume_20260712::test_run_single_terminal_merge_preserves_metadata_and_primary_val_ppl`
4. `tests.test_curated_geometry_math_20260709::test_prior_model_and_decode_variance_reads_share_guard`
5. `tests.test_diagnostics::test_attention_and_trace_reuse_snapshot_without_forward_replay`
6. `tests.test_estep_fixed_point_reporting_20260715::test_one_step_ahead_residual_is_distinct_from_configured_last_step`
7. `tests.test_report::test_generate_figures_reuses_one_same_token_snapshot`
8. `tests.test_report::test_generate_figures_memory_guard_uses_materialized_batch_peak`
9. `tests.test_run_diagnostics_2026_06_13::test_val_diagnostics_passes_explicit_diagonal_covariance_for_square_trace`
10. `tests.test_train::test_validation_finalizer_appends_to_existing_metrics_schema`

The CUDA command used `C:\anaconda\python.exe`, Torch
`2.10.0.dev20251210+cu128`, `VFE3_TEST_DEVICE=cuda`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` on the RTX 5090. The JUnit SHA-256 was
`84F41A33C3B8D5DBB5AD4796E16C135D4A5FAC483AFC41E8AF36591BAFD24E48` and
recorded 22 cases, 0 failures, 0 errors, 0 skips, and 8.845 seconds. The two
environment variables were restored after the run.

The audit's focused read-only probes reproduced M1, M4, M5, M6, M7, M10,
M12, M13, M17, and the exact H1 allocation formula. Remaining findings are
closed by exact revision-bound source-path inspection because their behavior
follows directly from unconditional branches, absent validation, or a
registry/call-site mismatch. The JUnit files and synthetic probe artifacts are
task-owned temporary evidence and are removed before integration; their hashes
above preserve the machine-readable result identities.

## Limitations and closure boundary

This was a codebase audit, not a repair pass. No production code was changed.
The audit did not run a complete 105,000-step experiment, mutate any real data
cache, substitute a real checkpoint, or claim a measured active-shape CUDA
peak. H1 is closed on exact active-config reachability, tensor shape, dtype,
copy semantics, and allocation arithmetic; an actual OOM remains unclaimed.
M10 is closed as a numerical kernel defect and remains Medium because no
materially corrupted post-retraction training trajectory was demonstrated.

The audit is not a green-build certificate. The ten CPU failures remain open,
and any source, dependency, active configuration, input, or artifact revision
change invalidates the affected evidence. Repairs should be performed in a
separate task with one regression per finding, followed by fresh CPU and CUDA
JUnit records and a new closure ledger.
