# Ultra-Deep Codebase Audit — 2026-07-11

## Audit identity and outcome

This audit covers the repository at commit c7064e9f6a2c91a1dd7f109865d5e2dcff987ad4,
the fetched origin/main head when the isolated audit worktree was created. The review was performed
from branch codex/ultradeep-audit-20260711 in
C:\tmp\V3_Transformer-ultradeep-audit-20260711. The user's live checkout was not used for audit
edits, and no production or test source was changed.

The audit produced 36 deduplicated candidates. Independent source verification and an adversarial
severity challenge retained 33 findings: 1 High, 23 Medium, and 9 Low. Three candidates were
dropped because they described an explicit API contract, a prior intentional ruling, or an
unreachable interpretation rather than a defect. No Critical finding survived.

The one High finding is a deterministic device-memory failure in long-context generation:
the model retains a separate dense attention prior for every growing context length. The retained
prior storage grows cubically with the context bound for a progressively growing prompt, while the
existing generation warning accounts only for one forward pass. For lengths 1 through 4096, even a
single shared fp32 prior
retains 22,914,881,536 elements, approximately 85.4 GiB, before head replication and other model
state.

## Scope, evidence rules, and limitations

The inventory contained 206 Python files and 299 tracked or searchable files. The review covered
the 54-file production package, click-to-run entry points, 139 test files, reporting and artifact
code, registries, numerical kernels, geometry, inference, training state, generation, and the
relevant documentation. Findings were based on executable code and call paths rather than code
comments. The July 9 curated closure ledger, containing 132 closed findings, was treated as a
regression baseline so previously fixed, relabeled, intentional, or deferred items were not
recycled.

Five base investigators examined code quality and security, runtime failure paths, registry and
dead-code seams, performance, and Python contracts. Seven theory investigators separately reviewed
numerical analysis, gauge theory, differential geometry, information geometry, variational free
energy, transformer behavior, and implementation fidelity. Two independent verifier passes then
read every candidate source location and ran focused probes. Proposed High findings were challenged
by a skeptic and a defender before final severity was assigned.

The research wiki was consulted read-only through Research/index.md and the linked VFE Transformer,
GL(K), gauge-attention, variational-free-energy, information-geometry, and SPD-geometry notes. The
current working manuscript Research/manuscripts/GL(K)_attention.tex, lines 563–574, supplied the
local gauge law used to adjudicate frame diagnostics. The vault was not modified.

The environment used Python 3.14.4 and torch 2.11.0+cpu. CUDA was unavailable, so the one
CUDA-only test and RTX 5090 behavior could not be executed. GPU-specific correctness and peak-memory
measurements therefore remain unverified. The High generation-memory finding is an exact retained
element count from cache keys and tensor shapes, not a measured CUDA allocation.

## Verification baseline

The default suite was executed as:

    python -m pytest -x --junitxml=C:\tmp\vfe3-ultradeep-audit-baseline-20260711.xml

The JUnit child testsuite reported tests=2153, failures=0, errors=0, skipped=15, and time=326.525
seconds, yielding 2138 passes. Fourteen of the skips were the repository's opt-in slow
figure/artifact/UMAP integrations. They were then run by exact node ID: the longest report test
reported 1 passed in 96.08 seconds, and the remaining thirteen reported 13 passed in 327.21 seconds.
The remaining skip was CUDA-only.

python -m compileall -q vfe3 exited successfully. A fatal/static-name Ruff pass,
ruff check . --select E9,F63,F7,F82 --no-cache --output-format concise, found one error:
tests/test_reporting_additions.py:780:9 F821 Undefined name pytest. The unrestricted Ruff baseline
reported 455 findings, predominantly existing style and test warnings; it was used as inventory,
not converted wholesale into audit findings.

## Severity model

High means a valid supported operation can deterministically fail or exhaust resources at a scale
the configuration admits, with no effective guard. Medium means a valid nondefault path computes
the wrong quantity, can corrupt state or artifacts, crashes after construction, or has a material
operational cost. Low means the effect is confined to extensibility, diagnostics, test clarity, or
a narrower performance and cleanup edge. Malformed explicit numeric configuration was not rated
High unless it arose from ordinary runtime state.

## High finding

### H1 / C32 — Growing generation retains cubic dense-prior storage

**Evidence:** vfe3/model/model.py:404, 560–610, 612–627, and 1798–1882.

VFEModel owns non-evicting dictionaries for attention priors and RoPE rotations. The attention-prior
key includes the sequence length, and every miss stores the complete dense N by N tensor. Generation
from a one-token prompt repeatedly invokes a full forward at lengths 1, 2, and so on until
max_seq_len, so the cache retains the sum of all squared lengths. No eviction or generation-local
cleanup exists. Once the context
reaches the bound, later calls reuse the maximum-length entry, but all smaller entries remain
resident for the lifetime of the model.

The exact shared-prior element count through length M is M(M+1)(2M+1)/6. At M=4096 this is
22,914,881,536 fp32 elements, or approximately 85.4 GiB. A head-specific prior multiplies that
burden by the head count. The warning at model.py:1855–1876 estimates only the dominant transient
allocation of one maximum-length forward, so it does not report this retained cache cost. The
default length 128 limits the ordinary default impact, but max_seq_len is a supported configuration
bound and a valid one-token generation through length 4096 deterministically exhausts the stated
RTX 5090.

**Repair direction:** Store one maximum-length causal prior and slice it, use a bounded LRU keyed by
the few lengths needed outside generation, or clear generation-owned intermediate lengths after
each step. The warning should include retained cache storage. A regression should generate through
many lengths and assert a constant or explicitly bounded cache cardinality.

## Medium findings

### M1 / C01 — Gauge-pure reporting omits transport covariance

**Evidence:** vfe3/run_artifacts.py:1088–1100 and 1140.

The pure-path report records transport metadata but does not include the transport builder's
covariance declaration in the Boolean conjunction that determines on_gauge_pure_path. A focused
probe with a noncovariant Regime-II transport was labeled gauge-pure. This can cause experimental
provenance to certify a route whose transport violates the stated local gauge law.

**Repair direction:** Make transport covariance an explicit required flag in the pure-path
predicate and add negative tests for every registered noncovariant transport.

### M2 / C02 — A tied-block preconditioner configuration constructs and then crashes

**Evidence:** vfe3/config.py:1782 and vfe3/geometry/phi_preconditioner.py:385.

Configuration rejects tied_block_glk with killing_per_block but accepts pullback_per_block. The
accepted route later raises because a tied generator spans multiple blocks and the per-block solver
requires block-local generators. This violates the config boundary: an invalid combination survives
construction and fails during the first active preconditioner call.

**Repair direction:** Reject pullback_per_block for tied generators at configuration construction,
or implement a tied-aware block solve whose unknown is shared across the repeated blocks.

### M3 / C06 — Run directories collide at one-second timestamp resolution

**Evidence:** train_vfe3.py:497 and vfe3/run_artifacts.py:118.

Two same-label launches within one second resolve to the same path. RunArtifacts reopens that
directory with exist_ok=True, allowing checkpoints, histories, metrics, and figures from independent
runs to overwrite or interleave. This is reachable under parallel sweeps or rapid restarts.

**Repair direction:** Add a collision-resistant suffix such as a monotonic counter, process ID plus
random nonce, or creation with exclusive retry. Reopening an existing run directory should require
an explicit resume identity.

### M4 / C08 — Floating token caches are silently truncated to integers

**Evidence:** vfe3/data/datasets.py:88.

Loaded cache tensors are converted directly to torch.long before token validation. Fractional
values therefore truncate rather than fail, so a malformed cache can become a different valid token
stream without an error. Range checks performed after conversion cannot recover the lost evidence.

**Repair direction:** Validate integral dtype, or prove every floating value is finite and exactly
integral, before conversion. Reject Boolean and floating lookalikes at the data boundary.

### M5 / C10 — Checkpoint restoration is non-atomic

**Evidence:** vfe3/run_artifacts.py:413–479.

Loading mutates model and collaborator state before all config, RNG, cursor, and step fields have
been validated. A later exception can leave the caller with a partially restored model and optimizer
while reporting load failure. Retrying or continuing from that object no longer starts from the
pre-load state.

**Repair direction:** Validate and normalize the complete payload first, then apply mutations only
after validation succeeds. If staged validation is impossible, snapshot every mutable recipient and
roll back in a finally-safe transaction.

### M6 / C11 — Dirty-tree provenance can follow and read unbounded untracked files

**Evidence:** vfe3/run_artifacts.py:546.

The provenance collector hashes each untracked path with read_bytes. That operation follows file
symlinks and materializes the entire target in memory before the outer exception handler can help.
A large accidental artifact or symlink can stall finalization or exhaust host memory.

**Repair direction:** Use streaming hashes, lstat before opening, reject or record symlinks without
following them, and apply an explicit byte limit with a provenance marker for omitted content.

### M7 / C12 — EMA evaluation swaps are not exception-safe

**Evidence:** vfe3/train.py:1105 and 1163.

Training stores live parameters and installs EMA parameters before evaluation and best-checkpoint
work, but restoration is not protected by finally. An exception after copy_to leaves the averaged
weights installed as if they were the live optimizer iterate. Subsequent training, saving, or error
recovery can then proceed from the wrong parameter state.

**Repair direction:** Encapsulate the swap in a context manager whose exit always restores the live
parameters, including nested save and reporting failures.

### M8 / C13 — Ambient EMA can leave the direct-frame gauge group

**Evidence:** vfe3/ema.py:38–42, 66–70, and 82–86; vfe3/train.py:201–205, 1073, and 1323.

When both use_ema and gauge_parameterization=omega_direct are enabled, omega_embed is averaged in
ambient matrix coordinates with every other trainable parameter. GL and its SO/Sp subgroups are not
convex. Averaging I and -I produces the singular zero matrix, and less symmetric trajectories can
leave the registered subgroup. copy_to installs that invalid frame without retraction.

Both relevant features are default-off, so the adversarial panel downgraded the raw High label to
Medium. The accepted combination remains mathematically invalid and can break inverse transport.

**Repair direction:** Exclude group-valued parameters from ordinary EMA and maintain a
manifold-aware mean, or retract/project each shadow update with the same group-specific machinery
used by the optimizer. Reject the combination until that contract exists.

### M9 / C14 — Detached training is misclassified as evaluation inside the E-step

**Evidence:** vfe3/model/model.py:886–903 and vfe3/inference/e_step.py:1083–1106.

The detach estimator wraps a training E-step in torch.no_grad. The inner loop then uses only
torch.is_grad_enabled() to distinguish training from evaluation. As a result, randomized training
depth is disabled and the eval-only halting rule can become active during detached training. A
verifier probe configured randomized depth three and observed the detached training route stop after
one iteration.

The defect requires default-off estimator and loop-control combinations, so it is Medium rather
than High.

**Repair direction:** Pass an explicit training/evaluation control into e_step. Gradient recording
and operational mode are independent concepts and must not share torch grad-mode as a proxy.

### M10 / C15 — A skipped optimizer update can still mutate Metropolis reflection state

**Evidence:** vfe3/train.py:1071.

The Metropolis sweep runs after train_step without checking status_out["did_step"]. An overflow or
other guarded optimizer skip can therefore leave continuous parameters unchanged while mutating
reflection state and consuming RNG. That breaks the expected atomic meaning of a skipped training
step and can spoil exact resume comparisons.

**Repair direction:** Gate every post-step state mutation and EMA update on the same committed-step
status, and pin the no-mutation/no-RNG-advance contract in an overflow regression.

### M11 / C16 — Omega-direct training can penalize an inactive phi table

**Evidence:** vfe3/model/model.py:1409 and vfe3/model/prior_bank.py:1094.

Omega-direct encoding still carries phi_embed, although transport uses omega_embed. The mass_phi
term penalizes the inactive chart and produces nonzero phi_embed gradients in a focused backward
probe. The optimizer can spend work changing a parameter that does not define the active frame,
while the reported objective includes a coordinate penalty unrelated to the active state.

**Repair direction:** Bind mass_phi to the active parameterization, omit inactive phi state from
omega-direct beliefs, or reject mass_phi>0 on that route.

### M12 / C19 — Cached EFE decodes every position and keeps only the last

**Evidence:** vfe3/inference/belief_cache.py:259.

The cache-supported policy path decodes logits shaped B times Kp by L by V, then retains only the
last position. At realistic vocabulary sizes the unused L-1 positions dominate memory. This cost
recurs on every policy-scoring invocation and scales with candidate horizon L.

**Repair direction:** Pass only terminal mean and covariance slices to the decoder, matching the
decode_last optimization already used in ordinary generation.

### M13 / C20 — Figure generation replays the same inference state across extractors

**Evidence:** vfe3/viz/report.py:179.

The report driver invokes numerous inference extractors independently. Snapshot reuse exists in
other diagnostics, but the figure driver does not build and share one frozen forward result. A
single sequence can therefore incur many complete E-step replays during finalization, which helps
explain the measured 6–7 minute slow-integration group.

**Repair direction:** Build one immutable diagnostic snapshot per evaluated sequence and pass it to
all compatible extractors. Extractors requiring a distinct state should declare that requirement.

### M14 / C23 — The eigenvector damping floor is not scale-relative at small spectra

**Evidence:** vfe3/geometry/retraction.py:108–124.

The stabilized eigenspace adjoint forms a relative gap term and then clamps its square to an
absolute 1e-12. At covariance eigenvalues near 1e-6, the clamp sets an effective gap scale of 1e-6,
so it can materially damp a valid, resolved 1e-6 eigengap. A focused probe measured a 31.3 percent
gradient deviation for eigenvalues [2e-6, 3e-6, 8e-6].

PyTorch's
[official torch.linalg.eigh documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html)
states that eigenvector gradients depend on inverse eigenvalue gaps and become unstable as gaps
approach zero. That supports stabilization, but
does not justify an absolute floor that dominates valid low-scale spectra. This finding is the
audit's inference from the repository formula and probe. The affected full-covariance path is
nondefault, so the adversarial ruling is Medium.

**Repair direction:** Define the lower bound in dtype- and spectrum-relative units, distinguish a
small scale from a degenerate gap, and add scale-equivariance tests over uniformly rescaled SPD
matrices.

### M15 / C24 — Single-frame quantities are mislabeled as gauge invariants

**Evidence:** vfe3/metrics.py:574 and 606; current GL(K)_attention.tex:563–574.

The metrics name frame eigenphases, frame log-determinants, and singular-value anisotropy as gauge
invariants. Under the manuscript's local action U_i maps to h_i U_i, these quantities change for
general h_i. They would be conjugacy invariants under h U h^{-1}, but that is not the local frame
law used by the model.

**Repair direction:** Relabel them as gauge-dependent frame order parameters, or replace them with
holonomy, transport, or closed-loop quantities whose transformation law gives an actual invariant.

### M16 / C25 — Single-block out-of-group controls are mathematically vacuous

**Evidence:** vfe3/metrics.py:1222.

For a full Gaussian, joint congruence of both arguments preserves KL under any invertible matrix.
The current single-block SO/Sp diagnostic samples an out-of-group invertible matrix and applies the
same congruence test, so both in-group and out-of-group residuals are near zero. A verifier probe
measured approximately 1.48e-7 for both. The panel cannot distinguish the registered subgroup.

**Repair direction:** Test a property that depends on subgroup membership, such as preservation of
the defining metric or symplectic form, and mark the current energy-residual out-group control
unavailable for single-block full-Gaussian routes.

### M17 / C26 — The transpose shortcut breaks exact direct-frame cocycles under small drift

**Evidence:** vfe3/geometry/transport.py:1333.

For skew/orthogonal routes, group_element_inverse returns a transpose whenever the orthogonality
residual is at most 1e-4. A nonorthogonal stored frame below that threshold therefore receives an
approximate inverse. The direct-frame identity Omega_ij=U_i U_j^{-1} no longer telescopes exactly;
a 4.0e-5 drift produced a nonzero cocycle residual in the focused probe.

**Repair direction:** Use the true inverse for stored group elements, or limit the transpose
shortcut to a threshold derived from dtype roundoff and reproject before transport construction.

### M18 / C27 — NaN Rényi order is converted into a plausible saturated divergence

**Evidence:** vfe3/config.py:866 and vfe3/families/base.py:27.

The positivity comparison permits NaN because NaN <= 0 is false. The divergence becomes NaN and is
then mapped by nan_to_num to kl_max. Instead of failing at configuration, the run silently treats an
undefined divergence as a finite saturated energy.

**Repair direction:** Require math.isfinite(renyi_order) and the supported order interval before
constructing the model. Numerical guards should not convert invalid hyperparameters into data.

### M19 / C28 — Nonfinite objective weights pass negative-only validation

**Evidence:** vfe3/config.py:1319–1330 and 2371–2376; vfe3/free_energy.py:418–461.

Several mass and lambda checks reject negative values but accept NaN and positive infinity. Those
weights enter the objective directly and can yield nonfinite losses, gradients, and artifacts.
Because the values must be explicitly malformed, both adversarial reviewers downgraded the raw
High label to Medium.

**Repair direction:** Route every scalar objective weight through one finite-and-nonnegative
validator, including list-valued variants and values read from serialized config.

### M20 / C29 — Nonfinite b0 and c0 poison state-dependent alpha

**Evidence:** vfe3/config.py:1359–1381 and vfe3/alpha_i.py:87–127.

Scalar and per-coordinate positivity checks admit NaN. State-dependent alpha then evaluates
c0/(b0+D) and its regularizer with the invalid values, producing NaN coefficients and loss terms.
Positive infinity also satisfies the current positivity test while destroying the intended
parameterization. This requires an explicit nondefault adaptive-alpha configuration and is Medium.

**Repair direction:** Require every b0 and c0 element to be finite and strictly positive before
shape- or mode-specific validation.

### M21 / C30 — The phi substep omits the active two-hop objective

**Evidence:** vfe3/inference/e_step.py:496–581, 790–810, and 880–915.

When lambda_twohop>0, the mean and covariance natural gradients include the detached two-hop
coupling. The phi coordinate step calls phi_alignment_loss, whose interface and implementation
contain no lambda_twohop term. With e_phi_lr>0, the phi coordinate therefore descends a different
functional from the mean and covariance coordinates. A focused comparison found a material
difference from the full objective's phi gradient.

This is a valid but double-opt-in route because both lambda_twohop and e_phi_lr default to zero. The
panel therefore set Medium rather than High.

**Repair direction:** Thread lambda_twohop into phi_alignment_loss and construct the same fixed-hop
weight term used by the mean/covariance step. Add an autograd-of-full-F oracle for phi with both
toggles active.

### M22 / C31 — Per-layer totals mix pre- and post-transform belief states

**Evidence:** vfe3/model/model.py:2875 and 2894.

The per-layer pairwise terms are evaluated from the post-mixer or post-normalization belief, while
the self term uses cap["converged"], the pre-transform E-step state. Whenever a mixer, Clebsch-Gordan
coupling, or block normalization is active, the reported total is not the free energy of one state.
Downstream plots can therefore attribute changes to a layer using an internally mixed diagnostic.

**Repair direction:** Choose and document one layer state, then compute every reported free-energy
term from that same state. If both are useful, report separate pre-transform and post-transform
totals.

### M23 / C34 — Non-power-of-two ALiBi slopes differ from the labeled reference

**Evidence:** vfe3/attention_prior.py:29.

The repository uses one geometric sequence for every head count. The authors' reference
implementation uses a recursive workaround for non-power-of-two counts, so the shipped slopes
differ when H is not a power of two despite the implementation being labeled as faithful to the
Press formulation. This changes relative distance bias by head and can confound comparisons.

**Repair direction:** Implement the authors' non-power-of-two construction, rename the existing
variant as a distinct geometric schedule, or reject unsupported head counts under the faithful
label. See the
[authors' reference slope construction](https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742-L752).

## Low findings

### L1 / C03 — Metrics and numerical-monitor registries are not production dispatch seams

**Evidence:** vfe3/metrics.py:1397 and vfe3/numerics.py:317.

Repository-wide call search found compute_metrics and run_monitors only in definitions and tests.
Production diagnostics call concrete functions, so registering replacements does not affect live
behavior. This is an extensibility defect against the registry convention, not a current numerical
failure.

**Repair direction:** Route the production driver through the registries or remove the misleading
registration surface.

### L2 / C04 — The figure registry is bypassed by the report driver

**Evidence:** vfe3/viz/figures.py:496 and vfe3/viz/report.py:243.

Production reports call concrete plot functions, and get_figure has no non-test consumer. A new
registered figure is therefore not discoverable by report generation without editing the driver.

**Repair direction:** Make report assembly declarative over registered figure specifications,
including memory and input requirements.

### L3 / C05 — lambda_h modes remain closed over a fixed tuple

**Evidence:** vfe3/lambda_h_i.py:35.

A fixed _LAMBDA_H_MODES tuple rejects new alpha registrations before delegation. Adding a lambda_h
variant therefore requires editing central source despite the surrounding registry pattern.

**Repair direction:** Validate against the live registry and keep aliases in registration metadata.

### L4 / C07 — Empty evaluation fabricates CE zero and perplexity one

**Evidence:** vfe3/train.py:714.

An empty or all-ignore evaluation leaves total_tok at zero. Clamping the denominator produces CE=0
and PPL=1, which looks like a perfect result rather than absence of evidence.

**Repair direction:** Raise or return an explicit unavailable metric when no scored token exists.

### L5 / C09 — A save failure can leak pyplot figures

**Evidence:** vfe3/viz/figures.py:55 and vfe3/run_artifacts.py:1187.

The save helper can raise before returning its figure, while direct history-figure callers close
only after successful return. Repeated failures can accumulate registered figures and memory during
reporting.

**Repair direction:** Create, save, and close figures inside one try/finally ownership boundary.

### L6 / C17 — Sigma-gate approval is not bound to checkpoint content

**Evidence:** vfe3/inference/sigma_gate.py:184 and 213.

The artifact stores a caller-supplied checkpoint label, but verification checks PASS status and an
optional spec commit rather than a checkpoint content hash. A stale approval can be presented with
a different checkpoint carrying the same label.

**Repair direction:** Record and verify a cryptographic checkpoint identity and the active config
hash.

### L7 / C18 — Cached EFE re-encodes shared context for every candidate

**Evidence:** vfe3/inference/belief_cache.py:207 and 211.

The path replicates the common context across Kp candidates before encoding. Shared context
embeddings are therefore recomputed Kp times even though later transport inverses are reused.

**Repair direction:** Encode the common prefix once and broadcast only the candidate-dependent
continuation state.

### L8 / C21 — Generic family contractions silently truncate tuple arity

**Evidence:** vfe3/families/base.py:287.

Generic divergence branches contract natural parameters and sufficient statistics with zip but do
not require equal arity. A malformed registered family can silently omit trailing statistics.

**Repair direction:** Validate exact tuple lengths at the family boundary before contraction.

### L9 / C22 — A test fallback references undefined pytest

**Evidence:** tests/test_reporting_additions.py:780.

The nonfinite-JSON callback calls pytest without importing it. The test still fails, but through
NameError rather than the intended diagnostic. This is the sole finding from the strict
E9/F63/F7/F82 Ruff pass.

**Repair direction:** Import pytest in the test module and pin the intended failure message.

## Dropped candidates

### C33 — No EOS early-stop behavior

Dropped. generate explicitly promises exactly max_new_tokens and returns a tensor whose documented
shape adds that count. The model exposes no EOS contract. Early stopping would be a feature, not a
repair of violated behavior.

### C35 — Hard-support forward KL becomes infinite

Dropped. Curated closure-ledger item 35 intentionally preserves hard support as the true default,
and tests pin an explicit normalized finite support floor for tasks requiring finite risk. Infinite
forward KL when q assigns mass outside strict support is the expected mathematical result.

### C36 — Rectangular causal builders use zero-based query coordinates

Dropped. Production model prior construction is square, and the cached appended-query path slices
offset rows from a full square prior. The rectangular helper has coherent aligned-prefix semantics
and exposes no suffix-offset argument. The candidate assumed an API promise that does not exist.

## Independent verification and adversarial rulings

The verifier passes confirmed the source behavior of all 36 candidates before final classification.
The following candidates received a proposed High label or were otherwise disputed and therefore
received an explicit skeptic/defender challenge.

| Candidate | Skeptic | Defender | Final | Adjudication |
| --- | --- | --- | --- | --- |
| C13 | Medium | Medium | Medium | Real manifold failure, but requires EMA plus omega-direct, both default-off. |
| C14 | Medium | Medium | Medium | Real mode confusion, but requires detached training with opt-in loop controls. |
| C23 | Medium | High | Medium | Material gradient distortion on a supported full-SPD path, but scale-specific and nondefault. |
| C28 | Medium | Medium | Medium | Objective poisoning is real but begins with malformed explicit configuration. |
| C29 | Medium | Medium | Medium | Adaptive-alpha poisoning is real but begins with malformed nondefault configuration. |
| C30 | Medium | High | Medium | Wrong configured phi functional, but requires two independent default-off toggles. |
| C32 | High | Medium | High | Valid long generation deterministically retains cubic dense-prior storage and defeats the existing warning. |
| C33 | Drop | Drop | Drop | Exact-token-count generation is the documented contract. |
| C35 | Drop | Drop | Drop | Intentional hard-support semantics are already pinned by the closure ledger. |
| C36 | Drop | Drop | Drop | No production rectangular suffix call and no suffix-offset API contract. |

The complete disposition is:

| IDs | Result |
| --- | --- |
| C32 | High |
| C01, C02, C06, C08, C10–C16, C19, C20, C23–C31, C34 | Medium |
| C03–C05, C07, C09, C17, C18, C21, C22 | Low |
| C33, C35, C36 | Dropped |

## Areas inspected with no new surviving defect

The theory investigators found no new defect in the implemented forward Gaussian KL orientation,
the Rényi direction used by the registered family after excluding invalid NaN configuration, the
Gaussian natural-gradient factors on the established diagonal and full-covariance paths, the
attention-entropy stationary-point term, or the default flat pure path. The current default
configuration remains outside every Medium geometry/toggle interaction listed above. These are
negative audit results, not a proof of global correctness.

Security review found no credential material, shell injection route, path traversal regression, or
unsafe deserialization finding beyond the bounded provenance issue C11. Registry review found
several bypasses but no hidden neural-network module or violation of the no-MLP/no-Linear
constraint. Dead-code review did not identify a production branch safe to delete without a separate
reachability decision.

## Recommended repair order

Address H1 first because it can deterministically exhaust device memory while the current warning
understates the retained allocation. Next repair the objective and geometry correctness set:
M1, M8, M9, M14–M17, and M21–M22. Then close state and artifact integrity issues M3, M5–M7, and
M10–M11. Input/config validation M4 and M18–M20 can be handled together through shared finite-value
validators. Performance work M12–M13 and L7 should follow, then registry/reporting and test
diagnostic lows.

No repair was applied in this audit. Every surviving item needs a focused regression before its
implementation change, followed by the repository's machine-readable verification workflow.
