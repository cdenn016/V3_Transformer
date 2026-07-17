# Ten-agent deep audit after the 2026-07-16 remediation

Audit initiated: 2026-07-16

Audit completed: 2026-07-17

Audited base: `e007c4d96022dc2e2b2d5c74666fb2ca390eceab`

Scope: the executable VFE training, inference, geometry, analysis, artifact, and
figure paths. The EFE implementation was excluded at the user's direction. The
approved deletion of the slow UMAP tests was also excluded and was not treated
as a coverage defect.

## Executive conclusion

The post-remediation tree still contained reachable defects across experiment
completion, numerical representation, evaluation semantics, filesystem
ownership, artifact durability, and final-report memory accounting. The most
direct threat to future runs was in atomic publication: the new pre-replace
`fsync` reopened its temporary file as read-only, but Windows implements
`os.fsync` through a write-flush operation. That could fail during
`RunArtifacts` construction before training began. The handle is now reopened
as `r+b`, retaining the durability barrier without blocking Windows runs.

The reported end-of-training OpenMP abort remains addressed by the earlier
disposable figure-worker boundary in `e007c4d`. This audit extended that
containment to the three standalone analysis drivers that could otherwise load
NumPy or Torch inside Spyder before isolation. It also strengthened the report
memory guard so the child declines dense full-vocabulary work whose actual
decoder workspace can exceed the configured 8 GB threshold. The guard now
accounts for full covariance pair matrices and for the near-KL diagonal-family
branch that retains float32 inputs beside float64 coordinate intermediates.

All actionable in-scope findings below are remediated in source. No executable
verification was performed because the user explicitly prohibited testing.
The closure therefore means implemented, statically inspected, and independently
source-reviewed. It does not assert a runtime pass count.

## Method

Exactly ten distinct expert agents reviewed the code through these lenses:
runtime and configuration reachability, numerical analysis, differential
geometry, information geometry, variational semantics, transformer and
attention behavior, process and filesystem contracts, data and artifact
integrity, performance and concurrency, and cross-cutting coordination. They
followed executable call paths and active configuration seams. Comments and
docstrings were used only to locate intended contracts, never as proof of
runtime behavior.

The initial reports were reconciled against the shared remediation worktree.
Three existing experts then performed source-only adversarial rechecks of the
highest-risk patches. Those rechecks found four integration gaps: evaluation
APIs still inherited training-mode inner-loop behavior, stale ablation cells
could survive an ownership mismatch, the report guard omitted a reachable
float64 coordinate peak, and Windows atomic publication used a read-only
`fsync` handle. Each was patched and re-reviewed. The final rechecks returned
PASS without executing code.

## Runtime and experiment-completion findings

### R1 High: incomplete ablation invocations could report process success

The ablation driver isolated cell failures but its top-level process status did
not represent incomplete sweeps or failed requested figure scopes. Automation
could therefore accept a partial scientific result. `ablation.py:4677` now
publishes one explicit invocation status record, and `ablation.py:4827` returns
nonzero for incomplete requested work, global invalidation failure, or failed
requested render scopes. Configuration and setup errors are distinguished from
training failures at `ablation.py:3512`.

### R2 High: incomplete or empty multiseed analyses exited successfully

`multiseed_analysis.py:938` now treats a missing seed population and an
incomplete requested seed design as process failure. Figures are emitted only
for a complete publication cohort, and `main` returns one when the cohort is
incomplete. The analysis status file records the same decision.

### R3 Medium: three analysis drivers imported scientific runtimes in Spyder

`compare_vocab_figures.py`, `multiseed_analysis.py`, and
`scaling_analysis.py` previously reached Torch, NumPy, or plotting imports in
the long-lived parent interpreter. Each click-to-run parent is now standard
library only, launches a hidden disposable child, scopes
`KMP_DUPLICATE_LIB_OK=TRUE` to that child, forwards its output, and propagates
its exit status. This extends the end-of-run figure isolation policy without
setting the unsafe compatibility variable in training or Spyder.

## Numerical-analysis findings

### N1 High: full-SPD publication certificates used float32 checks

`vfe3/geometry/retraction.py:325` now performs the final eigenvalue and
Cholesky certificate in float64 for every caller dtype. The public tensor
remains float32 where configured, but near-boundary decisions no longer depend
on a float32 spectral check that can misclassify a represented matrix.

### N2 High: separately rounded forward and inverse transports did not share one represented group element

`vfe3/geometry/transport.py:1101` now exponentiates the forward factor once and
computes the inverse of that represented value in float64. Equal-block and
compact paths use the same rule. This removes finite-precision mismatch between
`exp(M)` and a separately rounded `exp(-M)` and restores telescoping for the
stored factors.

### N3 Medium: affine-invariant SPD distance materialized an inverse square root

`vfe3/metrics.py:762` now factors the reference covariance in float64 and uses
triangular solves to form the symmetric generalized-eigenvalue problem. It
avoids the less stable explicit inverse-square-root construction and returns in
the promoted caller dtype.

## Differential-geometry findings

### G1 High: the head mixer could leave the general linear group

The prior `I + delta` coordinate could become singular. The mixer at
`vfe3/model/head_mixer.py:155` now uses the matrix exponential of its learned
coordinate, so each represented component remains invertible for finite
inputs. Identity initialization and the existing opt-out gauge-pure route are
preserved.

### G2 Medium: E-step covariance residuals inferred rank from tensor shape

`vfe3/metrics.py:1436` now receives the covariance-rank contract explicitly,
and `vfe3/train.py` passes `model.cfg.diagonal_covariance`. Full-covariance
residuals therefore cannot silently take the diagonal metric branch when an
ambiguous shape reaches the diagnostic.

## Information-geometry findings

### I1 High: the model-channel prior discarded learned full covariance

The s-to-p route previously reused only the diagonal log-variance table even
when the configured model channel carried packed Cholesky coordinates.
`vfe3/model/prior_bank.py:1393` now reconstructs the complete covariance for a
full `model_channel` prior and shares that helper across the relevant encoders.

### I2 Medium: valid full-Gaussian Fisher precision had an unconditional ridge

`vfe3/families/gaussian.py:374` now attempts a ridge-free Cholesky first. Valid
SPD inputs receive their exact represented precision; escalating jitter and a
pseudoinverse remain recovery behavior for invalid or numerically failed rows.

## Variational-semantics finding

### V1 High: a complexity-only diagnostic was labeled as full variational free energy

The logged total excludes the observation likelihood, so the name
`free_energy_total` overstated its semantics. `vfe3/train.py:939` and
`vfe3/train.py:1634` now publish `inner_alignment_energy_total`; the old column
remains as an explicit legacy alias for existing analysis files.
`vfe3/viz/figures.py:823` labels the co-descent plot as inner alignment energy,
states that cross-entropy is separate, and describes Pearson correlation as
association rather than evidence of causation.

## Transformer and attention findings

### A1 Medium: positional and attention-prior caches grew with every shape or device key

The cache dictionaries in `vfe3/model/model.py` are now bounded to the active
entry by clearing on each miss at lines 644 and 663. Device or dtype movement
still clears both caches. Long exploratory Spyder sessions therefore cannot
accumulate one large tensor per historical shape and device combination.

### A2 High: detached training was classified as evaluation

Randomized E-step depth and evaluation halting were selected from ambient
autograd state. Detached training deliberately runs under `no_grad`, so it
could take evaluation policy. `vfe3/inference/e_step.py:1187` now accepts an
explicit `training` flag. Training forwards pass the module's mode, while
generation, diagnostic snapshots, and report extraction pass evaluation
semantics explicitly. The second-pass review verified that the previously
missed generation and snapshot call sites now pass `training=False`.

## Process and filesystem-contract findings

### C1 High: Windows child cleanup could block indefinitely or conceal containment failure

`vfe3/process_utils.py:16` defines bounded reap and `taskkill` intervals.
`vfe3/process_utils.py:152` centralizes termination and bounded collection for
job-assignment failure, gate failure, timeout, interruption, and ordinary
exception paths. A failed Job Object termination now falls back to the process
tree kill path rather than being silently accepted.

### C2 High: scaling cleanup lacked an exact ownership identity

`scaling.py:995` requires a versioned sentinel bound to route, label, and seed
before stale artifacts may be removed. One exact regular legacy marker can be
promoted once. Foreign, malformed, redirected, or mismatched directories fail
before destructive cleanup.

### C3 High: an ablation figure manifest could authorize arbitrary deletion names

`vfe3/viz/figure_worker.py:215` validates the manifest schema, path-component
portability, exact file-list types, uniqueness, and the deterministic filename
inventory owned by each scope. A renderer output outside that inventory is
rejected before publication or cleanup.

### C4 High: stale ablation success markers survived ownership mismatch

The first ownership patch protected destructive cleanup but did not protect
cache admission or result collection. A setup rejection could preserve an old
success marker, and the collector could use it to satisfy completion.
`ablation.py:3150` now provides one exact read-only owner check used by resume
admission and collection. A mismatched sentinel cannot be overwritten, cached,
or counted as current success.

### C5 Low: filesystem slugs could preserve a Windows reserved stem

`vfe3/path_utils.py:111` now validates the completed slug. A readable prefix
such as `CON.txt` is disambiguated with a leading underscore while retaining
the stable hash suffix.

## Data and artifact-integrity findings

### D1 Medium: package identity excluded executable root drivers

`vfe3/run_artifacts.py:114` now hashes the package plus the supported training,
ablation, scaling, analysis, and figure drivers. EFE entry points remain
excluded under this audit's scope. Artifact identity now changes when a root
driver that controls the run changes.

### D2 Medium: raw checkpoint resumes lacked a durable provenance decision

`vfe3/run_artifacts.py:1974` now publishes `resume_provenance.json` for raw
state resumes. It records saved and current semantic-config fingerprints, code
identity, Git identity, drift fields, checkpoint path, and the explicit policy
that the current configuration remains authoritative.

### D3 Medium: tokenizer identity was inferred from cache naming

`vfe3/data/datasets.py:137` validates an optional exact tokenizer-provenance
manifest bound to the payload digest, dataset, split, tokenizer tag, encoding,
and vocabulary size. Legacy caches remain readable but are labeled
`filename_inferred_unverified` instead of being presented as verified.

### D4 High: metrics publication could leave a partial CSV

`vfe3/run_artifacts.py:1323` now writes the rectangular candidate history to a
unique sibling, atomically replaces `metrics.csv`, and mutates in-memory
history only after publication. Field-set drift fails before either state is
changed.

### D5 High: atomic publication lacked a usable Windows durability barrier

`vfe3/run_artifacts.py:413` flushes each temporary before replacement and the
containing directory on supported non-Windows systems. The second-pass review
found that reopening the temporary as `rb` could make Windows `_commit` reject
the flush. Line 438 now uses `r+b`, preserving write access for `os.fsync`.
A pre-replace flush failure removes the temporary and leaves the prior
destination untouched. On supported non-Windows systems, a directory-flush
failure occurs after the atomic rename; it is propagated as a durability
failure even though the new destination may already be visible. That flush is
outside the destination-lock retry loop, so it cannot issue a second rename
after the temporary has been consumed.

## Performance and concurrency findings

### P1 High: the report guard omitted active decoder workspaces

The earlier guard budgeted retained batches and two `(B,N,V)` tensors but not
the full-family `(B,N,V,K,K)` or diagonal-family `(B,N,V,K)` intermediates.
`vfe3/viz/report.py:247` now accepts both workspace forms, and
`vfe3/viz/report.py:320` derives the requirement from the active decoder and
covariance rank.

The second-pass review then traced the near-KL diagonal Gaussian branch, where
float32 inputs coexist with several float64 coordinate tensors. Eight float32
worksets still underestimated that path. The constant at
`vfe3/viz/report.py:56` is now twelve float32-equivalent worksets, covering the
source-derived eleven-workset peak with one-workset margin. Full-covariance
pair matrices retain their separate conservative budget.

### P2 Medium: analysis isolation happened after heavy imports

This was the performance and process side of R3. Moving scientific imports to
the child prevents the parent from retaining large runtime state, avoids a
second OpenMP runtime initialization in Spyder, and ensures child completion is
the only success condition exposed by each driver.

## Cross-cutting coordinator findings

### X1 Medium: setup failures were classified as training crashes

The ablation loop now separates ownership, directory, and contract setup from
`run_single`. Setup exceptions carry `error_kind="setup"`; only exceptions
raised by the actual training call carry `error_kind="train"`. Completion and
status reporting preserve that distinction.

### X2 Medium: evaluation APIs inherited module training behavior

The explicit training flag initially fixed detached training but left several
evaluation call sites dependent on the model's default `training=True` state.
Generation, policy base decoding, rollouts, and diagnostic snapshot construction
now request evaluation semantics directly. Internal training forwards continue
to pass `self.training`.

### X3 Low: console and figure language retained the old full-F claim

The training console now prints `Inner alignment energy`. Figure labels,
publication labels, multiseed curve keys, and explanatory text use the same
name while accepting the legacy CSV key. The co-descent docstring no longer
turns correlation into a causal claim.

## Findings not sustained

The claim that the head-mixer correction introduced a prohibited neural
network was not sustained. The code changes an existing matrix coordinate from
`I + delta` to `exp(delta)`; it adds no linear layer, MLP, or activation, changes
no configuration default, and preserves the mixer-disabled pure path.

The proposed `mm_exact` defect was also not sustained as an executable
correctness failure. The reviewed concern was naming and configuration
interpretation, not a mismatched active kernel. No configuration value was
changed.

The EFE implementation received no correctness disposition because it was
outside scope. No EFE source was intentionally modified. The deleted native
UMAP tests remain an accepted test-budget choice and were not restored.

## Verification and residual risk

Verification was limited to `git diff --check`, conflict-marker scans, exact
call-site searches, staged-diff review, and independent source review. New and
updated regression tests document the intended contracts, but none was
executed. No pytest, Python interpreter, Ruff, compile, import, GPU, or figure
smoke command was run after these changes.

The remaining uncertainty is runtime integration, especially on the user's
Anaconda and RTX 5090 environment. Source review establishes that figure and
analysis work now occurs in disposable children and that Windows atomic
publication uses a write-capable handle. Only a future authorized run can
establish end-to-end behavior on that exact environment.
