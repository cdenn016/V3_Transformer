# PriorBank Head-Evidence and Canonical-Content Design

## Objective

Add three default-off experimental components without changing the existing pure path:

1. an identity-initialized PriorBank-native head-evidence mixer;
2. an exact `canonical_content_gauge` encoder control; and
3. a practical `canonical_content_projected` encoder that retains the model's diagonal projection.

The exact encoder is a geometric falsification/control mode. It is not presented as a performance
improvement: the existing `gaussian_frame_diagonal` family already establishes that frame-intrinsic
content under a flat coboundary makes the relative frame cancel and leaves `phi_embed` without a
belief-path gradient. The projected encoder is the performance-oriented experiment because its
fixed-basis diagonal projection prevents that cancellation from being exact.

## Scope and names

- New config toggle: `use_priorbank_head_evidence_mixer: bool = False`.
- New encoder registry key: `canonical_content_gauge`.
- New encoder registry key: `canonical_content_projected`.
- No existing config value is changed automatically.
- Existing `use_head_mixer` remains independent and may be enabled simultaneously. The run metadata
  must continue to report its `block_glk` nonintertwiner status separately from the new decoder mixer.
- No claim is made that either encoder or mixer improves perplexity until matched multi-seed runs exist.

## 1. PriorBank-native head-evidence mixer

### Mathematical definition

Let the gauge group expose head or irrep blocks with dimensions `irrep_dims = [d_1, ..., d_H]`.
For a query Gaussian `q` and vocabulary prior `p_v`, let `q_h` and `p_vh` be their block marginals.
The bank owns one trainable logit per block,

```text
w = H * softmax(head_evidence_logits)
```

so every weight is positive, the weights sum to `H`, and zero initialization gives `w_h = 1`.
The weighted decoder divergence is

```text
D_head(q || p_v) = sum_h w_h KL(q_h || p_vh) + C(q)
```

For a diagonal query, `C(q) = 0`. For a full-covariance query scored against the existing diagonal
vocabulary bank,

```text
C(q) = 0.5 * (sum_h logdet(Sigma_q,hh) - logdet(Sigma_q)).
```

This query-only total-correlation term is deliberately not head-weighted. At initialization,
`D_head == KL_full` as an absolute logit identity, not only after softmax. Consequently the initial
path also preserves nonzero z-loss, unigram bias behavior, and dense/chunked parity. Once the weights
move, the model learns relative head evidence while retaining the global correlation penalty.

Each marginal KL is invariant under a common invertible coordinate change within its block. This
certifies the evidence operation only; it does not claim that the complete decoder has solved every
query/prior frame-alignment issue.

### Runtime and parameter ownership

- `PriorBank` creates `head_evidence_logits: Parameter[H]` only when the toggle is enabled.
- The parameter is initialized to exact zeros and grouped at `m_p_mu_lr`, with zero weight decay and
  optimizer role `mu`, matching the established decoder/head-mixer policy.
- A helper returns the normalized block weights and expanded coordinate weights. The fast Gaussian
  diagonal/full dense decoders and their chunked fused-CE twins consume those weights from one
  implementation seam.
- Default-off code takes the current functions without allocating a parameter or adding arithmetic.
- The first implementation supports canonical Gaussian KL decoders (`diagonal`,
  `diagonal_chunked`, `full`, and `full_chunked`). Generic family/non-KL decoders fail closed rather
  than pretending that an arbitrary divergence decomposes into Gaussian marginal KLs.
- Configuration requires `use_prior_bank=True`, at least two declared blocks, and a supported decoder.
  A simultaneous existing `use_head_mixer=True` is allowed and reported, not silently disabled.
- Diagnostics report normalized weights, weight entropy, and maximum absolute drift from one.

### Required deferred-work marker

At the model-to-decoder integration seam, add this intentional future-work comment:

```python
# TODO(frame-conjugated-head-mixer): thread the realized query frame through this boundary
# before implementing M_i = U_i (A kron I) U_i^{-1}; do not co-transform q and every
# vocabulary prior by the same M, because that is a KL-invariant no-op.
```

This marker identifies a separate vector-valued mixer. It is not part of this build and must not be
used to weaken the evidence mixer's completed behavior.

## 2. Exact `canonical_content_gauge`

### Representation

The existing mean and log-variance tables are interpreted as canonical/frame-intrinsic content
`(a_v, s_v)`, while `phi_embed[v]` supplies the token frame. The encoder returns those intrinsic
moments and the frame without materializing dense pushed-forward moments. This is the efficient
coordinate representation already required by `gaussian_frame_diagonal`:

```text
physical mean       = U_v a_v
physical covariance = U_v diag(s_v) U_v^T.
```

Under flat transport `Omega_ij = U_i U_j^-1`, the sender and receiver frames cancel from the exact
Gaussian comparison. The decoder therefore scores the converged intrinsic query directly against
the canonical vocabulary content tables. Learned positional frame composition also cancels from
this exact flat-cocycle comparison.

### Fail-closed configuration contract

`canonical_content_gauge` requires:

- `family='gaussian_frame_diagonal'`;
- `transport_mode='flat'`;
- `gauge_parameterization='phi'`;
- `prior_source='token'` and `s_e_step=False`;
- `use_prior_bank=True`;
- a diagonal Gaussian KL decoder (`diagonal` or `diagonal_chunked`); and
- reflection modes off in this first implementation.

The mode supports a tied or untied decode bank; an untied bank is cloned in canonical coordinates.
It emits a configuration notice that the flat exact construction makes the relative frame cancel
and that `phi_embed` receives no supervised belief-path gradient when no independent phi penalty or
nonbelief channel is active.

The mode remains separate from the family key even though their mathematics coincide: the encoder
key makes the scientific arm explicit in run provenance, prevents accidental use with a fixed-basis
family, and gives the projected sibling a matched naming and routing contract.

## 3. Practical `canonical_content_projected`

### Encode boundary

This mode keeps canonical tables `(a_v, s_v)` but materializes moments after token and positional
frame composition, before the model-channel/E-step stack. For the realized vertex frame `U_i`,

```text
mu_i       = U_i a_xi
Sigma_raw  = U_i diag(s_xi) U_i^T
sigma_i    = diag(Sigma_raw)
```

The belief family remains `gaussian_diagonal`, so every later covariance transport continues to use
the established fixed-basis diagonal projection. That projection is the intentional symmetry
breaking/regularization that keeps the frame load-bearing. The encode-time prior passed to the stack
is the same materialized object as `q(0)`, preserving `q(0) = p`.

The realized frame must include the configured positional right factor. The implementation reuses
the flat transport's vertex factors rather than independently reconstructing a differently clamped
matrix exponential. This avoids a content-frame/transport-frame mismatch.

### Canonical decode boundary

Before PriorBank scoring, the converged diagonal fixed-basis query is pulled back by the same realized
query frame:

```text
mu_c    = U_i^-1 mu_q
Sigma_c = U_i^-1 diag(sigma_q) U_i^-T.
```

`Sigma_c` is full even though the internal E-step family is diagonal. It is scored against the
canonical diagonal vocabulary bank through the existing full-Gaussian analytic decoder. Therefore
this encoder requires `decode_mode='full'` or `'full_chunked'`. The model threads the inverse vertex
factor into dense inference and fused training decode; a direct PriorBank projected-mode decode
without the required frame fails closed instead of silently comparing different frames.

### Fail-closed configuration contract

`canonical_content_projected` requires:

- `family='gaussian_diagonal'`;
- `transport_mode='flat'`;
- `gauge_parameterization='phi'`;
- `prior_source='token'` and `s_e_step=False`;
- `e_phi_lr=0.0` so the decode frame cannot diverge from the frame used to materialize `q(0)`;
- `use_prior_bank=True`;
- `decode_mode='full'` or `'full_chunked'`; and
- reflection modes off in this first implementation.

Untied decode tables remain canonical tables. Learned M-step phi and learned/frozen positional phi
are supported because every new forward rematerializes moments from the current composed frame.

## 4. Integration boundaries

### PriorBank

- Register both encoder names.
- Preserve existing table allocation and RNG ordering; neither mode creates duplicate canonical
  vocabulary tables.
- Add head-evidence parameter ownership, normalization, weighted Gaussian-KL helpers, and supported
  dense/fused decoder integration.
- Add an explicit projected-query canonicalization argument at the PriorBank decode boundary. It is
  mandatory only for `canonical_content_projected` and ignored nowhere silently.

### VFEModel

- Pass the new mixer toggle to `PriorBank`.
- After positional phi composition, materialize projected canonical content using the same flat
  vertex factors used by transport.
- Retain or expose the inverse query vertex through the belief/decode path and pass it to both dense
  and fused decode calls.
- Put the required frame-conjugated-mixer deferred-work comment at this boundary.
- Keep exact canonical content in intrinsic coordinates; do not materialize and then transport it a
  second time.

### Configuration, optimizer, and reporting

- Add registry-driven validation plus the explicit mode compatibility rules above.
- Group `head_evidence_logits` exactly once and retain the optimizer's fail-closed parameter coverage.
- Add the toggle and realized head weights to run configuration/diagnostics without changing the
  canonical pure-path predicate when the toggle is false.
- When enabled, mark the run as using a learned decoder evidence metric. Keep the existing vector
  HeadMixer compatibility metadata independent.

## 5. Error behavior and checkpoint compatibility

- All three components are default off and allocate no new state on the established path.
- Unsupported family/divergence/decode combinations fail during config construction.
- A projected PriorBank decode without a realized inverse frame raises a descriptive error.
- A malformed block partition or fewer than two blocks rejects the head-evidence mixer.
- Old checkpoints load unchanged under default settings. Toggle-on checkpoints contain the new
  head-evidence parameter; switching encoder modes does not rename the existing content/frame tables.
- Shape checks cover batch, sequence, head-block, covariance-rank, and vocabulary axes before a
  broadcast can silently succeed.

## 6. Test strategy

Implementation follows test-first red/green cycles.

### Head-evidence mixer

1. Default-off construction has no new parameter and preserves existing state-dict keys and logits.
2. Zero logits produce positive weights equal to one and summing to `H`.
3. Diagonal dense logits equal a direct per-head KL reference and gradients reach every weight.
4. Full dense logits equal a direct marginal-KL-plus-correlation reference; at initialization they
   equal the existing full decoder as absolute logits.
5. Full/diagonal chunked fused CE matches dense CE and gradients with nonzero z-loss, unigram bias,
   ignore indices, and a remainder vocabulary chunk.
6. Config rejects linear, single-block, and generic non-KL pairings; optimizer coverage includes the
   new parameter exactly once.
7. Simultaneous evidence mixer and existing HeadMixer execute and backpropagate without conflating
   their diagnostics.

### Exact canonical content

1. The registry and config constraints are reachable and fail closed on invalid pairings.
2. Encoded intrinsic moments and canonical decoder logits match a direct diagonal reference.
3. End-to-end flat-cocycle backward confirms the documented `phi_embed.grad is None` control while
   content tables train.
4. Tied and untied canonical decoder tables preserve step-zero parity.

### Projected canonical content

1. A hand-constructed frame, including a positional right factor, reproduces the specified pushed
   mean and projected diagonal covariance.
2. The canonical decode pullback reproduces a direct dense full-Gaussian KL reference.
3. Dense and full-chunked CE agree in value and gradients through content, variance, phi, and optional
   head-evidence weights.
4. End-to-end backward confirms the projected mode retains a nonzero phi gradient under a
   gradient-connected estimator, while the exact control does not.
5. Generation/last-position decode and diagnostic extraction use the same frame-aware boundary.
6. Missing-frame and unsupported-config tests fail with explicit messages.

## 7. Success criteria

- Every new production behavior is preceded by a test that fails for the missing feature.
- Default-off focused regression tests remain unchanged and pass.
- Head-evidence initialization is absolute-logit identical to the existing decoder for diagonal and
  full queries, including z-loss behavior.
- Exact and projected encoders have distinct, mechanically pinned phi-gradient semantics.
- Dense, fused, training, inference, and generation paths agree on the canonical decode frame.
- The CUDA-capable interpreter passes the focused CPU and GPU-relevant lanes, with machine-readable
  JUnit evidence and a validated verification ledger tied to the final artifact revision.

## 8. Non-goals

- Implementing the frame-conjugated vector HeadMixer.
- Claiming a perplexity improvement before matched experiments.
- Supporting nonflat/direct-link transport, omega-direct storage, active E-step phi retraction,
  reflections, model-channel priors, or generic non-Gaussian divergences in the first canonical-mode
  implementation.
- Replacing or deleting `per_token`, `per_token_additive`, `gaussian_frame_diagonal`, or the existing
  post-belief `HeadMixer`.
