# PriorBank Head-Evidence and Canonical-Content Encoders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in PriorBank-native head-evidence mixer and two explicit canonical-content encoder modes: an exact frame-intrinsic scientific control and a projected diagonal-family mode with a frame-aware full-covariance decoder.

**Architecture:** Keep the default execution graph unchanged. Put the head-evidence algebra in a small pure helper module and feed its coordinate weights into the existing analytic Gaussian dense and fused decoders. Put canonical pushforward/pullback algebra in a second pure helper module. The exact mode remains on the established `gaussian_frame_diagonal` intrinsic-coordinate route; the projected mode materializes diagonal moments after the realized token-plus-position frame is composed, then pulls the final diagonal query back to canonical coordinates immediately before full-Gaussian scoring.

**Tech Stack:** Python 3, PyTorch, pytest, VFE3 registries/configuration, CUDA-capable test interpreter `C:/anaconda/python.exe`, Git.

## Global Constraints

- Work only in `C:/tmp/vfe3-priorbank-head-evidence-canonical-content-20260808` on branch `codex/priorbank-head-evidence-canonical-content-20260808`.
- The live checkout contains protected user WIP in `ablation.py`, `train_vfe3.py`, `vfe3/config.py`, and `zzzzz.py`; do not touch, stash, reset, or copy those live files. All edits belong in this isolated worktree.
- Use `C:/anaconda/python.exe` for every test or command that imports torch. Use `VFE3_TEST_DEVICE=cuda` for the CUDA lane.
- Follow strict RED-GREEN-REFACTOR: add the named failing test, run it and record the expected failure, make the smallest production change, rerun it, then commit.
- Preserve the default-off state dict, random-number consumption, arithmetic order, and public behavior. Do not allocate the new parameter when its toggle is false, and do not route default decoders through new weighted helpers.
- The new mixer supports only the canonical built-in Gaussian KL routes: `gaussian_diagonal` with `diagonal`/`diagonal_chunked`, and `gaussian_full` with `full`/`full_chunked`. Fail closed for generic/custom families, non-KL functionals, or fewer than two gauge blocks.
- The exact encoder is intentionally a scientific control: its realized token frame cancels from the relative flat transport and supervised decode. Do not add a surrogate phi gradient.
- The projected encoder deliberately re-diagonalizes after every existing diagonal-family transport. Do not silently replace the diagonal family with a full-covariance internal state.
- The projected decoder must use the exact realized frame used by the forward path, including the right positional factor. Do not re-exponentiate phi along a separate numerical path.
- Preserve American English in code, comments, test names, error messages, and documentation.

---

## Task 1: Add pure head-evidence algebra and opt-in PriorBank state

**Files:**

- Create: `vfe3/model/head_evidence.py`
- Create: `tests/test_priorbank_head_evidence.py`
- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/config.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/train.py`

- [ ] **Step 1: Write RED tests for configuration, state, normalization, and optimizer ownership**

In `tests/test_priorbank_head_evidence.py`, build tiny banks with `irrep_dims=[2, 2]` and assert:

```python
def test_head_evidence_is_default_off_and_does_not_change_state_dict():
    torch.manual_seed(7)
    base = PriorBank(11, 4, 8, irrep_dims=[2, 2], use_prior_bank=True)
    torch.manual_seed(7)
    explicit = PriorBank(
        11, 4, 8, irrep_dims=[2, 2], use_prior_bank=True,
        use_priorbank_head_evidence_mixer=False,
    )
    assert base.state_dict().keys() == explicit.state_dict().keys()
    for key in base.state_dict():
        torch.testing.assert_close(base.state_dict()[key], explicit.state_dict()[key], rtol=0, atol=0)
    assert not hasattr(base, "head_evidence_logits")


def test_zero_logits_produce_identity_head_and_coordinate_weights():
    pb = PriorBank(
        11, 4, 8, irrep_dims=[1, 3], use_prior_bank=True,
        use_priorbank_head_evidence_mixer=True,
    )
    head, coord = pb.head_evidence_weights(dtype=torch.float64, device=torch.device("cpu"))
    torch.testing.assert_close(head, torch.ones(2, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(coord, torch.ones(4, dtype=torch.float64), rtol=0, atol=0)
```

Also assert the config default is false, invalid configurations raise targeted `ValueError`s, the parameter exists only when enabled and begins at exact zeros, and `build_optimizer` places it once at `m_p_mu_lr`, `weight_decay=0`, `role="mu"`. Cover simultaneous `use_head_mixer=True` and `use_priorbank_head_evidence_mixer=True` as legal and separately owned.

Run RED:

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_priorbank_head_evidence.py -k "default_off or zero_logits or optimizer or invalid_config"
```

Expected failure: the config keyword, PriorBank constructor keyword, helper, parameter, and optimizer group do not yet exist.

- [ ] **Step 2: Implement the pure normalized-weight helper**

Create `vfe3/model/head_evidence.py` with immutable output and no module parameters:

```python
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class HeadEvidenceWeights:
    head: Tensor
    coordinate: Tensor


def normalized_head_evidence_weights(
    logits: Tensor,
    irrep_dims: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> HeadEvidenceWeights:
    if logits.ndim != 1 or logits.numel() != len(irrep_dims):
        raise ValueError("head-evidence logits must contain one scalar per gauge block")
    if len(irrep_dims) < 2 or any(int(dim) <= 0 for dim in irrep_dims):
        raise ValueError("head evidence requires at least two positive gauge blocks")
    work = logits.to(device=device, dtype=dtype)
    raw = (work - work.max()).exp()
    head = raw / raw.mean()
    repeats = torch.as_tensor(tuple(int(dim) for dim in irrep_dims), device=device)
    return HeadEvidenceWeights(head=head, coordinate=torch.repeat_interleave(head, repeats))
```

Keep logits as the source tensor so autograd reaches the parameter through the dtype/device conversion. The `raw / raw.mean()` form is mathematically `H * softmax(logits)`, is overflow-safe, and produces exact ones for equal logits; that exact identity is needed by the baseline-plus-delta scorer below.

- [ ] **Step 3: Wire configuration, PriorBank construction, and optimizer grouping**

Add `use_priorbank_head_evidence_mixer: bool = False` beside the existing head-mixer toggle in `VFE3Config`. In validation, require:

- `use_prior_bank=True`;
- `len(group.irrep_dims) >= 2` through the same group metadata used to construct PriorBank;
- built-in `renyi` at order `1.0`;
- `gaussian_diagonal` plus `diagonal`/`diagonal_chunked`, or `gaussian_full` plus `full`/`full_chunked`;
- no registry alias or generic-family decode that cannot establish canonical KL semantics.

Pass the toggle from `VFEModel` to `PriorBank`. In `PriorBank.__init__`, store immutable block dimensions and, only when enabled, allocate:

```python
self.head_evidence_logits = nn.Parameter(torch.zeros(len(self.irrep_dims)))
```

Expose a small `head_evidence_weights(dtype, device)` method that delegates to the pure helper and raises if the mixer is disabled. In `build_optimizer`, add exactly one group for `prior_bank.head_evidence_logits` at `cfg.m_p_mu_lr`, zero weight decay, and role `mu`.

- [ ] **Step 4: Run GREEN and regression tests**

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_priorbank_head_evidence.py -k "default_off or zero_logits or optimizer or invalid_config"
C:/anaconda/python.exe -m pytest -q tests/test_prior_bank.py tests/test_head_mixer.py tests/test_train.py -k "prior or head_mixer or optimizer"
```

- [ ] **Step 5: Commit Task 1**

```powershell
git add vfe3/model/head_evidence.py vfe3/model/prior_bank.py vfe3/config.py vfe3/model/model.py vfe3/train.py tests/test_priorbank_head_evidence.py
git commit -m "feat: add opt-in PriorBank head-evidence state"
```

---

## Task 2: Apply baseline-plus-delta head evidence to dense Gaussian KL decoders

**Files:**

- Modify: `vfe3/model/head_evidence.py`
- Modify: `vfe3/model/prior_bank.py`
- Modify: `tests/test_priorbank_head_evidence.py`

- [ ] **Step 1: Write RED tests against direct blockwise KL references**

Add a test-only reference that partitions coordinates by `irrep_dims`:

```python
def diagonal_head_reference(mu_q, var_q, mu_v, var_v, weights, dims):
    pieces = []
    start = 0
    for weight, dim in zip(weights, dims):
        stop = start + dim
        term = (
            (var_q[..., start:stop, None] +
             (mu_q[..., start:stop, None] - mu_v[start:stop]).square()) /
            var_v[start:stop]
            + var_v[start:stop].log()
            - var_q[..., start:stop, None].log()
            - 1.0
        ).sum(dim=-2) * 0.5
        pieces.append(weight * term)
        start = stop
    return torch.stack(pieces).sum(dim=0)
```

Test `diagonal` with nonuniform logits and unequal block dimensions against this reference, including gradients for query mean/variance, decode mean/variance tables, and evidence logits. Test zero-logit output and gradients against an otherwise identical disabled bank; output must be bitwise equal where the legacy arithmetic route is retained.

For full queries, construct an SPD covariance with nonzero cross-block terms and use:

```python
correction = 0.5 * (
    sum(torch.linalg.slogdet(cov_q[..., sl, sl]).logabsdet for sl in block_slices)
    - torch.linalg.slogdet(cov_q).logabsdet
)
expected = sum(w_h * kl_block_h for w_h, kl_block_h in zip(weights, block_kls)) + correction
```

Assert the correction is unweighted, logits receive gradients only through the weighted block KLs, and zero logits reproduce the existing full KL including off-block mutual information.

Run RED:

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_priorbank_head_evidence.py -k "diagonal_reference or full_reference or initial_identity"
```

Expected failure: dense decoders ignore the new parameter.

- [ ] **Step 2: Extend analytic accumulation without perturbing the disabled route**

In `prior_bank.py`, add an optional coordinate delta to `_decode_av_lhs` and `_decode_av`, but branch before any extra multiplication:

```python
def _decode_av_lhs(sq: Tensor, mc_q: Tensor, coord_delta: Tensor | None = None) -> Tensor:
    lhs = sq + mc_q.square()
    return lhs if coord_delta is None else lhs * coord_delta
```

Broadcast `repeat_interleave(head - 1, irrep_dims)` to the query rank once. Use it only to evaluate the additive correction `sum_h (w_h - 1) KL_h`; do not replace the established full KL. When the mixer is disabled, call the old signatures and preserve their previous reduction order.

- [ ] **Step 3: Implement exact baseline-plus-delta scoring**

Use the identity

```text
D_head(q, p_v)
  = D_full(q, p_v) + sum_h (w_h - 1) KL(q_h || p_vh)
  = sum_h w_h KL(q_h || p_vh)
    + 0.5 * (sum_h logdet Sigma_q,hh - logdet Sigma_q).
```

For a full query, compute each marginal `KL(q_h || p_vh)` with the repository's safe-Cholesky policy for `logdet(Sigma_q,hh)`, not raw `slogdet`, and combine every block validity mask with the existing full-covariance mask. For a diagonal query, slice the existing diagonal KL terms. Vectorize the vocabulary-dependent additive terms with coordinate deltas; compute block entropy terms once per query outside vocabulary/chunk loops.

Always evaluate the established decoder first and add the delta. Because equal logits produce exact `w_h == 1`, the delta is exactly zero at initialization and enabled zero-logit output is bitwise identical to the legacy full KL, including cross-block mutual information, temperature, unigram bias, CE, and z-loss. The delta remains differentiable with respect to logits at zero.

- [ ] **Step 4: Apply the algebra to `diagonal`, `diagonal_chunked`, `full`, and `full_chunked` dense logits**

Obtain weights once per public decode call. Add the head delta to the pre-temperature divergence, then let the established code apply temperature, unigram bias, invalid-position exclusion, CE, and z-loss. Keep the learned evidence parameter out of noncanonical registered decode routes even if a custom route shares a string name. The zero-logit enabled path must be bitwise value-identical to the legacy scorer; tests must not weaken this to approximate equality.

- [ ] **Step 5: Run GREEN and decoder regressions**

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_priorbank_head_evidence.py -k "diagonal_reference or full_reference or initial_identity"
C:/anaconda/python.exe -m pytest -q tests/test_prior_bank.py tests/test_full_covariance.py tests/test_tier12_decode.py tests/test_audit_runtime_semantics_20260720.py
```

- [ ] **Step 6: Commit Task 2**

```powershell
git add vfe3/model/head_evidence.py vfe3/model/prior_bank.py tests/test_priorbank_head_evidence.py
git commit -m "feat: weight PriorBank Gaussian head evidence"
```

---

## Task 3: Cover fused CE, diagnostics, and the deferred frame mixer seam

**Files:**

- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/train.py`
- Modify: `tests/test_priorbank_head_evidence.py`
- Modify: `tests/test_run_diagnostics_2026_06_13.py`

- [ ] **Step 1: Write RED fused-CE tests with all reduction boundaries**

Parameterize diagonal/full chunked modes and require fused CE to match dense `F.cross_entropy` with:

- nonzero z-loss;
- nonuniform unigram bias;
- learned decode temperature;
- an ignored target;
- a vocabulary size not divisible by chunk size;
- finite gradients for query mean/covariance, decode tables, temperature, and evidence logits.

Also run one combined model with both the old Schur `HeadMixer` and the new evidence mixer enabled. Assert both parameter families receive separate gradients and neither changes the other's reported diagnostics.

Run RED:

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_priorbank_head_evidence.py -k "fused or simultaneous"
```

Expected failure: chunked CE does not use evidence weights and diagnostics are absent.

- [ ] **Step 2: Thread identical algebra through registered fused CE**

Refactor only enough that dense and fused chunk loops call the same weighted analytic scoring primitive. Compute full-query marginal block invariants once outside the vocabulary loop. Preserve the existing z-loss accumulation, unigram addition, ignore-index accounting, invalid-query behavior, and checkpoint boundaries.

- [ ] **Step 3: Add diagnostics and the required deferred-design comment**

Report, only when enabled:

```python
head_evidence_weights
head_evidence_entropy
head_evidence_max_abs_drift
```

Use normalized probabilities `weights / H` for entropy and `max(abs(weights - 1))` for drift. Keep old `HeadMixer` diagnostics under their existing names.

At the realized-frame boundary in `VFEModel`, add this exact intentional marker:

```python
# TODO(frame-conjugated-head-mixer): thread the realized query frame through this boundary
# before implementing M_i = U_i (A kron I) U_i^{-1}; do not co-transform q and every
# vocabulary prior by the same M, because that is a KL-invariant no-op.
```

This marker is deliberately deferred scope requested by the user; do not implement that alternative mixer in this change.

- [ ] **Step 4: Run GREEN and reporting regressions**

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_priorbank_head_evidence.py tests/test_run_diagnostics_2026_06_13.py tests/test_reporting_additions.py -k "head_evidence or head_mixer or diagnostics"
```

- [ ] **Step 5: Commit Task 3**

```powershell
git add vfe3/model/prior_bank.py vfe3/model/model.py vfe3/train.py tests/test_priorbank_head_evidence.py tests/test_run_diagnostics_2026_06_13.py
git commit -m "feat: complete fused head-evidence decoding"
```

---

## Task 4: Register and constrain exact `canonical_content_gauge`

**Files:**

- Create: `tests/test_canonical_content_gauge.py`
- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/config.py`
- Modify: `vfe3/model/model.py`

- [ ] **Step 1: Write registry, constraint, and exact-reference RED tests**

Add tests that prove:

- `canonical_content_gauge` is a live encoder registry key and produces the same canonical `(a_v, s_v, phi_v)` lookup as `per_token` for identical tables;
- it requires `family="gaussian_frame_diagonal"`, `transport_mode="flat"`, `gauge_parameterization="phi"`, `prior_source="token"`, `s_e_step=False`, `use_prior_bank=True`, `decode_mode in {"diagonal", "diagonal_chunked"}`, and reflection off;
- tied and untied banks preserve canonical coordinates and untied initialization remains an exact clone;
- direct frame-family transport/decode agrees with a hand-computed intrinsic-coordinate reference;
- with no explicit phi penalty, `phi_embed.grad is None` after supervised backward;
- construction emits one clear notice that the exact flat-cocycle mode cancels token phi from the supervised belief/decode path.

Run RED:

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_canonical_content_gauge.py
```

Expected failure: the encoder registry key and validation contract do not exist.

- [ ] **Step 2: Add an explicit encoder registration with existing table ownership**

Register a named encoder that intentionally delegates the canonical lookup semantics to the existing table path instead of creating duplicate parameters:

```python
@register_encode("canonical_content_gauge", can_omit_base_mean=True, can_omit_base_variance=True)
def _encode_canonical_content_gauge(pb: PriorBank, token_ids: Tensor) -> BeliefState:
    return _encode_per_token(pb, token_ids)
```

Import/use the existing `torch.Tensor` annotation style in this module. Do not consume extra random numbers, clone tables, or materialize dense pushforwards.

- [ ] **Step 3: Enforce exact-mode validation and provenance**

Put cross-field constraints in `VFE3Config`, while keeping direct `PriorBank` construction safe where local invariants are knowable. Add a one-time `UserWarning` explaining that this mode is frame-intrinsic and phi cancels under flat cocycle transport unless another objective explicitly acts on phi.

Keep the registry name distinct even though the lookup overlaps `per_token`; it carries scientific provenance and prevents accidental use with the ordinary diagonal family.

- [ ] **Step 4: Run GREEN and frame-family regressions**

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_canonical_content_gauge.py tests/test_frame_gaussian_family.py tests/test_exact_congruence_family.py tests/test_additive_table_control.py
```

- [ ] **Step 5: Commit Task 4**

```powershell
git add vfe3/model/prior_bank.py vfe3/config.py vfe3/model/model.py tests/test_canonical_content_gauge.py
git commit -m "feat: add exact canonical-content encoder"
```

---

## Task 5: Implement projected canonical materialization with a shared realized-frame context

**Files:**

- Create: `vfe3/model/canonical_content.py`
- Create: `tests/test_canonical_content_projected.py`
- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/config.py`
- Modify: `vfe3/contracts.py`

- [ ] **Step 1: Write pure pushforward/pullback RED tests**

Use a nonorthogonal hand matrix and a noncommuting right positional factor. Verify the exact formulas:

```python
frame = token_frame @ positional_frame
mu_materialized = torch.einsum("...ij,...j->...i", frame, mu_c)
var_materialized = torch.einsum("...ij,...j->...i", frame.square(), var_c)

mu_pulled = torch.einsum("...ij,...j->...i", frame_inv, mu_q)
cov_pulled = frame_inv @ torch.diag_embed(var_q) @ frame_inv.transpose(-1, -2)
```

Assert that reversing the positional multiplication order fails the reference. Include batched shapes, float64 gradcheck-sized inputs, and an inverse-consistency check.

Run RED:

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_canonical_content_projected.py -k "pushforward or pullback or positional_order"
```

Expected failure: the canonical-content helper module does not exist.

- [ ] **Step 2: Add pure canonical-content helpers and a typed frame context**

Create:

```python
@dataclass(frozen=True)
class CanonicalFrameContext:
    forward: Tensor
    inverse: Tensor


def project_canonical_diagonal(mu_c: Tensor, var_c: Tensor, frame: Tensor) -> tuple[Tensor, Tensor]:
    return (
        torch.einsum("...ij,...j->...i", frame, mu_c),
        torch.einsum("...ij,...j->...i", frame.square(), var_c),
    )


def pullback_diagonal_query(
    mu_q: Tensor, var_q: Tensor, frame_inv: Tensor,
) -> tuple[Tensor, Tensor]:
    mu_c = torch.einsum("...ij,...j->...i", frame_inv, mu_q)
    cov_c = frame_inv @ torch.diag_embed(var_q) @ frame_inv.transpose(-1, -2)
    return mu_c, cov_c
```

Add shape checks and preserve dtype/device/autograd. Put the context in the model contract where it can be captured without changing ordinary `BeliefState` semantics.

- [ ] **Step 3: Register and validate `canonical_content_projected`**

The encoder registry lookup must return canonical table coordinates without duplicate parameters. Require:

- `family="gaussian_diagonal"`;
- `transport_mode="flat"`;
- `gauge_parameterization="phi"`;
- `prior_source="token"`;
- `s_e_step=False`;
- `e_phi_lr=0`;
- `use_prior_bank=True`;
- `decode_mode in {"full", "full_chunked"}`;
- reflection off.

Support tied and untied prior banks. Learned or frozen positional frames and M-step phi updates remain legal.

- [ ] **Step 4: Materialize after positional composition and reuse the actual transport factors**

In `VFEModel.forward_beliefs`, immediately after the token frame and optional right positional frame have been composed, obtain forward/inverse vertex factors from the same `FactoredTransport` or `CompactFactoredTransport` object used by the flat diagonal-family transport. Do not call `matrix_exp` again.

For the projected encoder only:

1. treat the registry-returned mean/variance as canonical `(a_v, s_v)`;
2. compute `mu_i = U_i a_v` and `var_i = diag(U_i diag(s_v) U_i^T)` via the pure helper;
3. replace both q0 and the token prior with this same materialized diagonal-family object;
4. retain `CanonicalFrameContext(U_i, U_i^{-1})` through the forward seam for final decode.

The existing diagonal-family transport continues to project back to diagonal moments at each later transport. The exact encoder bypasses this materialization.

- [ ] **Step 5: Run projected materialization GREEN tests**

Add model-level tests that spy on the shared transport builder and prove the frame is built once, includes the right positional factor in the correct order, and is reused for both materialization and pullback. Assert q0 and p are exactly the same materialized object/value at the E-step entry. Verify nonzero gradients reach canonical mean/variance tables and `phi_embed` through a supervised scalar.

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_canonical_content_projected.py -k "materialize or shared_frame or phi_gradient"
```

- [ ] **Step 6: Commit Task 5**

```powershell
git add vfe3/model/canonical_content.py vfe3/model/prior_bank.py vfe3/model/model.py vfe3/config.py vfe3/contracts.py tests/test_canonical_content_projected.py
git commit -m "feat: materialize projected canonical content"
```

---

## Task 6: Add frame-aware projected full decoding across dense, fused, generation, and extraction paths

**Files:**

- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/viz/extract.py`
- Modify: `tests/test_canonical_content_projected.py`
- Modify: `tests/test_generate.py`
- Modify: `tests/test_extract_forward_fidelity.py`

- [ ] **Step 1: Write manual canonical pullback/decode RED tests**

For the final diagonal query `(mu_q, var_q)` and retained `U_i^{-1}`, calculate `(mu_c, Sigma_c)` with `pullback_diagonal_query`, then compare model logits to the existing analytic full-Gaussian decoder against canonical diagonal vocabulary priors. Cover `full` and `full_chunked`, tied and untied prior tables, nonuniform unigram bias, learned temperature, and a nontrivial right positional frame.

Assert direct `PriorBank.decode(mu_q, var_q)` raises a targeted error in projected mode when no frame context is supplied. Assert passing a shape-, device-, or dtype-incompatible context fails closed.

Run RED:

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_canonical_content_projected.py -k "manual_decode or missing_frame or incompatible_frame"
```

Expected failure: public decoding has no frame-context seam and scores the materialized query in the wrong coordinates.

- [ ] **Step 2: Add the explicit frame-aware decode seam**

Extend only the registered analytic full decode and fused-CE call boundary with an optional typed `canonical_frame` keyword. If `encode_mode != "canonical_content_projected"`, reject a supplied context with `ValueError`. If projected mode is active, require the context, validate its shape/device/dtype against the query, and pull the query into canonical coordinates before invoking the existing full/full-chunked scorer.

Do not materialize a full vocabulary covariance. The vocabulary bank stays canonical diagonal; the query alone becomes full after pullback. Reuse existing safe-Cholesky, invalid-position, temperature, and bias behavior.

- [ ] **Step 3: Thread the context through dense and fused training paths**

Return or capture the context from `forward_beliefs` without changing callers that do not request it. The main training forward must pass the exact context to:

- dense `PriorBank.decode`;
- registered `full_chunked.fused_ce`;
- `return_logits=True` and `decode_last=True` paths.

Add dense/fused parity tests with z-loss, ignore index, remainder chunk, and gradients for query mean/variance, canonical prior mean/variance, token phi, positional phi when learned, evidence logits when simultaneously enabled, and decode temperature. Require projected phi gradient to be finite and nonzero in a nondegenerate fixture.

- [ ] **Step 4: Keep generation and extraction on the model-owned decode seam**

Update generation and `vfe3/viz/extract.py` so projected mode never calls bare `prior_bank.decode` without the frame captured from the same forward pass. Prefer a model helper such as `_decode_belief_with_context(...)` over reconstructing frame state in each consumer.

Add tests proving ordinary encoders are unchanged and projected generation/extraction work for sequence length one, full sequence output, and `decode_last=True`.

- [ ] **Step 5: Run GREEN and consumer regressions**

```powershell
$env:VFE3_TEST_DEVICE='cuda'; C:/anaconda/python.exe -m pytest -q tests/test_canonical_content_projected.py tests/test_generate.py tests/test_extract_forward_fidelity.py
C:/anaconda/python.exe -m pytest -q tests/test_phase0_forward_beliefs.py tests/test_prior_bank.py tests/test_full_covariance.py tests/test_family_chunked_canonical_dispatch_20260808.py
```

- [ ] **Step 6: Commit Task 6**

```powershell
git add vfe3/model/prior_bank.py vfe3/model/model.py vfe3/viz/extract.py tests/test_canonical_content_projected.py tests/test_generate.py tests/test_extract_forward_fidelity.py
git commit -m "feat: decode projected canonical content"
```

---

## Task 7: Consolidated verification, claim ledger, and user-facing configuration documentation

**Files:**

- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-08-priorbank-head-evidence-canonical-content-design.md` only if implementation-established constraints differ from the approved design
- Create: `docs/verification/priorbank-head-evidence-canonical-content-ledger.json`
- Create: `docs/verification/priorbank-head-evidence-canonical-content-junit.xml` through pytest

- [ ] **Step 1: Document the three opt-in controls and their scientific boundary**

Add compact configuration examples for:

```python
use_priorbank_head_evidence_mixer = True
encode_mode = "canonical_content_gauge"      # exact intrinsic control; phi cancels
encode_mode = "canonical_content_projected"  # diagonal projection; frame-aware full decode
```

State compatible family/decode combinations, the exact mode's absent supervised phi gradient, and the projected mode's approximation boundary. Keep the deferred conjugated coordinate mixer out of current-feature claims.

- [ ] **Step 2: Run focused CPU/CUDA verification with machine-readable output**

Verify the CUDA interpreter before making any GPU claim:

```powershell
C:/anaconda/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

Run the complete focused lane and write JUnit evidence:

```powershell
$env:VFE3_TEST_DEVICE='cuda'; C:/anaconda/python.exe -m pytest -q --junitxml=docs/verification/priorbank-head-evidence-canonical-content-junit.xml tests/test_priorbank_head_evidence.py tests/test_canonical_content_gauge.py tests/test_canonical_content_projected.py tests/test_prior_bank.py tests/test_head_mixer.py tests/test_additive_table_control.py tests/test_exact_congruence_family.py tests/test_frame_gaussian_family.py tests/test_full_covariance.py tests/test_generate.py tests/test_extract_forward_fidelity.py
```

Then run the broader CPU-default regression lane:

```powershell
C:/anaconda/python.exe -m pytest -q tests/test_train.py tests/test_phase0_forward_beliefs.py tests/test_audit_runtime_semantics_20260720.py tests/test_reporting_additions.py tests/test_run_diagnostics_2026_06_13.py
```

- [ ] **Step 3: Create and validate the verification claim ledger**

Use the installed `verification` skill. Record one claim per check, at minimum:

1. default-off construction/state is unchanged;
2. diagonal and full head-evidence values match their direct references;
3. zero evidence logits recover canonical full KL including cross-block correction;
4. fused and dense objectives/gradients agree;
5. exact canonical mode has the documented phi-cancellation behavior;
6. projected pushforward/pullback uses the realized token-plus-position frame;
7. projected dense/fused training, generation, and extraction all use frame-aware decode;
8. invalid combinations fail closed.

Every closed code claim must cite the current commit, exact command/configuration, and JUnit testcase evidence. Use only `EVIDENCE_VERIFIED`, `REFUTED`, or `INCONCLUSIVE` as closure states; do not report `LLM_SUPPORTED` as complete. Run the skill's ledger validator and retain its output.

- [ ] **Step 4: Inspect defaults, diffs, and protected paths**

```powershell
git diff --check
git status --short
git diff f489fe7 -- train_vfe3.py ablation.py zzzzz.py
git grep -n "use_priorbank_head_evidence_mixer\|canonical_content_gauge\|canonical_content_projected\|frame-conjugated-head-mixer"
```

The protected-path diff must be empty. Confirm the required deferred marker occurs exactly once and every new config field defaults off.

- [ ] **Step 5: Commit verification and documentation**

```powershell
git add README.md
git add -f docs/verification/priorbank-head-evidence-canonical-content-ledger.json docs/verification/priorbank-head-evidence-canonical-content-junit.xml
git add -f docs/superpowers/specs/2026-08-08-priorbank-head-evidence-canonical-content-design.md
git commit -m "docs: verify canonical PriorBank features"
```

- [ ] **Step 6: Request final code review before integration**

Use `superpowers:requesting-code-review` against the full branch diff. Address only evidence-backed findings, rerun affected tests, and do not merge or modify the live checkout without separate user authorization.
