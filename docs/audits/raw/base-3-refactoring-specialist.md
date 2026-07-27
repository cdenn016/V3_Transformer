# Base investigator 3 — refactoring-specialist (dead code, dead config fields)

Returned 2026-07-27 ~08:33 CDT. Verbatim findings; NOT yet verified.

## Negative results (the most load-bearing result of this agent)

No critical or high findings. **A targeted sweep of all ~171 `VFE3Config` fields** (grep word-count
of each field name against `vfe3/`, `train_vfe3.py`, `ablation.py`, excluding `config.py` itself)
found **every field has at least one real consumption site outside its own
declaration/validation — no field is fully dead.** This was the highest-value hypothesis going in
and it is refuted.

Leads traced end-to-end and found correctly wired or explicitly fail-closed: `emission_mode` /
`emission_weight` reaching the non-kernel autograd oracle, `precision_attention_per_head`,
`close_basis` auto-resolution, `connection_weight_decay` scope, `grad_clip_per_role`, figure-registry
reachability via `vfe3/viz/specs.py`. The `emission_mode != "off"` branch at `config.py:1648-1657`
raises specifically to prevent the "accepted-and-dropped by the autograd oracle" bug class.

Independently re-confirmed (not re-reported) while tracing `decode_bias`: `_decode_linear` skips
`decode_unigram_prior` at `prior_bank.py:1901-1920`, versus its correctly-wired sibling
`decode_ce_linear_chunked` at `prior_bank.py:1250-1310`.

---

### Copy-pasted `i`/`j`/mask boilerplate across seven attention-prior registry functions
**Location:** vfe3/attention_prior.py:89-267 (`prior_causal`, `prior_causal_noself`, `prior_alibi`, `prior_causal_alibi`, `prior_causal_alibi_noself`, `prior_windowed`, `prior_causal_windowed`)
**Severity:** medium
**Evidence:** Each of the seven functions independently repeats the identical
```python
i = torch.arange(n_query, device=device).unsqueeze(-1)
j = torch.arange(n_key, device=device).unsqueeze(0)
```
followed by `B = torch.zeros(...); return B.masked_fill(~allowed, float("-inf"))` (or, for the
ALiBi trio, `dist = (i - j).abs()`, `slopes = _press_slopes(...)`, `B = -slopes.view(...)*dist`,
then `B.masked_fill(...)`). Read at lines 100-104 (`causal`), 124-128 (`causal_noself`), 148-152
(`alibi`), 178-184 (`causal_alibi`), 207-213 (`causal_alibi_noself`), 237-241 (`windowed`),
262-267 (`causal_windowed`). The seven copies are currently mutually consistent (no drift today),
so this is a pure duplication finding, not yet a bug — but the next one-off edit to the causal-mask
convention is not guaranteed to propagate to its siblings.
**Fix:** Factor the shared `i`/`j`/`zeros`/`masked_fill` scaffold (and the ALiBi `dist`/`slopes`/`B`
triple) into one or two helpers each `register_prior` function calls.

### Orphaned import left behind by a real bug fix
**Location:** vfe3/viz/extract.py:1234
**Severity:** low
**Evidence:** `from vfe3.families.gaussian import DiagonalGaussian` is never referenced in the
function body. Confirmed by `python -m pyflakes vfe3` ->
`vfe3\viz\extract.py:1234:5: 'vfe3.families.gaussian.DiagonalGaussian' imported but unused`, and by
reading lines 1234-1272. The fix at audit 2026-07-25 F10 replaced the hardcoded family with
`get_family(cfg.family)` (line 1258) and `pb.r_parameters()` (line 1259) but left the import.
**Fix:** Delete the unused `DiagonalGaussian` import.

### Unused local variable in phi-group pullback direction
**Location:** vfe3/geometry/phi_preconditioner.py:527
**Severity:** low
**Evidence:** `gram_factor = preparation.gram_factor` is assigned in `_full_pullback_group_direction`
and never read (`gram_inverse_half`, the next line, feeds the subsequent `torch.matmul` whitening).
Confirmed by pyflakes -> `local variable 'gram_factor' is assigned to but never used`, and by
reading lines 508-556.
**Fix:** Remove the dead line.

### Unused local variable in exact-congruence vertex determinant helper
**Location:** vfe3/families/exact_congruence.py:144
**Severity:** low
**Evidence:** `n_transport_blocks = ell.shape[-1]` is computed in `_vertex_log_abs_det` but never
referenced; the function branches on `len(irrep_dims)` and
`list(irrep_dims) != list(transport_block_dims)` (lines 145-157). Confirmed by pyflakes and by
reading lines 100-157.
**Fix:** Remove the dead line.
