# Base investigator 4 — performance-engineer (hot paths)

Returned 2026-07-27 ~08:19 CDT. Verbatim findings; NOT yet verified.

## Negative results (useful — these were checked and found already optimized)

No **high**-severity findings. The other named blowup point (pairwise `(N,N,K,K)` transport) is
already factored to avoid dense materialization on the default `gauge_group='block_glk'` +
`transport_mode='flat'` path (`vfe3/geometry/transport.py:28-69, 2638-2732`,
`vfe3/inference/e_step.py:190-203 _can_fuse_flat`). The E-step already hoists the loop-invariant
flat transport once per call rather than rebuilding it every inner iteration
(`vfe3/inference/e_step.py:1318-1366`). The Killing-metric preconditioner is memoized across
iterations (`vfe3/geometry/phi_preconditioner.py:824-937`). Per-step logging / grad-norm / CE-sync
overhead in `vfe3/train.py` is already gated to logged steps only (`metrics_out is not None` /
`do_log` gates at train.py:564-620, 1545-1569).

---

### Full (B, N, V) logit tensor materialized every training step under the default `decode_mode`
**Location:** vfe3/model/model.py:1564-1588; vfe3/model/prior_bank.py:1628-1673 (dense) vs. 1675-1696 & 1000-1037 (chunked twin)
**Severity:** critical (investigator's rating — see orchestrator note)
**Evidence:** `config.py:498` sets `use_prior_bank: bool = False` but `config.py:502` independently
sets `decode_mode: str = "diagonal"` (not `"diagonal_chunked"`). In `model.py`:
`active_decode_mode = self.cfg.decode_mode if self.cfg.use_prior_bank else "linear"` (1564), then
`fused_chunked = (targets is not None and decode_registration.supports_chunked)` (1566-1569); the
`"diagonal"` registration is `@register_decode("diagonal", can_omit_base_mean=True,
can_omit_base_variance=True)` (prior_bank.py:1628) — no `supports_chunked=True`, unlike its sibling
`@register_decode("diagonal_chunked", supports_chunked=True,
fused_ce=PriorBank.decode_ce_diagonal_chunked, ...)` (prior_bank.py:1675-1681), which the same
module's docstring states returns logits "byte-identical to `decode_mode='diagonal'`" (1693).

So whenever `use_prior_bank=True` (the documented "pure" KL-to-prior decode) is combined with the
plain default `decode_mode='diagonal'`, `model.py:1585` runs
`logits = self.prior_bank.decode(mu_final.float(), sigma_final.float())` which inside
`_decode_diagonal` executes `a_v = lhs @ rhs.transpose(-1, -2)` producing a `(B, N, V)` tensor
(prior_bank.py:1666), followed by another same-shape `kl_v` (1671) and a `flat_logits` reshape
feeding `F.cross_entropy` (model.py:1588-1596), which itself builds a same-size `log_softmax`
buffer. At this repo's configured dims (`vocab_size=50257`, `batch_size=64`, `max_seq_len=128` in
`config.py:83,85,97`), one `(B,N,V)` fp32 tensor is `64*128*50257*4B ≈ 1.65 GB`; three to four such
tensors materialize per step. The chunked path (`decode_ce_diagonal_chunked`,
prior_bank.py:1000-1037) never materializes the full tensor, via gradient-checkpointed per-chunk
`logsumexp`/gather, and is already wired as `fused_ce` for the `diagonal_chunked` registration.
**Fix:** Set `decode_mode`'s default (or at minimum the config-validation guidance) to
`"diagonal_chunked"` whenever `use_prior_bank=True`, since it is documented byte-identical to
`"diagonal"` and already implemented.

> ORCHESTRATOR NOTE: the *memory* claim is a code fact and stands on its own. The *fix* is a
> change to a default config toggle, which the user has ruled out of scope ("I am constantly
> changing toggles"). Reframe for the punch list as: "these two independent defaults compose into
> a 1.65 GB/step path when a byte-identical chunked twin exists" — an actionable observation the
> user may act on, not a demand to change a default. Severity likely DOWNGRADED at challenge.

### `generate()` recomputes the full encode -> E-step -> decode pipeline for every generated token
**Location:** vfe3/model/model.py:2208-2253
**Severity:** medium
**Evidence:** `for _ in range(max_new_tokens): context = seq[:, -self.cfg.max_seq_len:] ...;
_belief, decoded = self.forward_beliefs(context, return_logits=True, decode_last=True,
training=False)` (2249-2253) — every one of `max_new_tokens` iterations reruns
`prior_bank.encode` plus the full `n_e_steps`-iteration pairwise E-step over the entire (up to
`max_seq_len`) context, rather than reusing the belief state already converged for previously-seen
tokens. The function self-instruments a memory-only warning for this cost (2227-2248), confirming
the O(max_new_tokens · N²·n_e_steps) cost is real and unaddressed, not a stale comment.
**Fix:** None available short of an incremental-belief cache (explicitly deferred); flagged
because it materially affects wall clock for periodic in-training sample generation.

### Per-parameter gradient-norm diagnostic forces one host sync per parameter tensor, and duplicates `clip_grad_norm_`'s own norm computation
**Location:** vfe3/train.py:678-694, 717-730
**Severity:** low
**Evidence:** Gated by `if metrics_out is not None:` (678, logged steps only), the loop
`for g in optimizer.param_groups: gsq = sum(float(p.grad.detach().pow(2).sum()) for p in
g["params"] if p.grad is not None)` (681-683) calls Python `float(...)` — a blocking
device-to-host sync — once per parameter tensor rather than batching into one
`torch.stack(...).tolist()` sync, as the file does elsewhere (line 661). Separately,
`torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)` (730) internally recomputes the
same per-parameter L2 reduction, so on a logged step with clipping the gradient norm is computed
twice.
**Fix:** Batch the per-role sums and take one `torch.stack(...).tolist()` at the end (matching
line 661), and derive the logged `grad_norm` from `clip_grad_norm_`'s return value.

### E-step trajectory diagnostic performs a per-iteration `.item()` sync
**Location:** vfe3/inference/e_step.py:1420-1425
**Severity:** low
**Evidence:** `def _record_diagnostic_state(b): if return_trajectory: traj.append(_f_diag(b).item())`
(1421-1422) calls `.item()` inside the `for t in range(n_total):` E-step loop (1448 onward) once
per inner iteration when `return_trajectory=True`. Gated off by default (`return_trajectory: bool
= False`, e_step.py:1272) and not invoked from the training path, so it costs nothing today —
flagged because it would silently serialize the loop the moment a caller enables trajectory
logging during training.
**Fix:** None needed for the default path; if ever wired into training, keep a tensor list and
`.tolist()` once at the end.
