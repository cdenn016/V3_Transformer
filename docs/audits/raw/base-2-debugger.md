# Base investigator 2 — debugger (bugs + gradient flow)

Returned 2026-07-27 ~08:26 CDT. Verbatim findings; NOT yet verified.

Swept: `e_step.py`, `gradients/kernels.py`, `gradients/oracle.py`, `gradients/pairwise_stats.py`,
`free_energy.py`, `emission.py`, `model/model.py`, `model/block.py`, `model/stack.py`,
`model/prior_bank.py`, `config.py`, `train.py`, `attention_prior.py`, `alpha_i.py`,
`lambda_h_i.py`, `families/gaussian.py`, `geometry/retraction.py`,
`geometry/phi_preconditioner.py`, `gauge_optim.py`, `numerics.py`. Cross-checked against
`docs/2026-07-26-findings-triage.md` and this audit's already-known list to avoid re-reporting.

## Negative results (load-bearing — these are the classes that would have been CRITICAL)

**No critical findings.** No target-leakage, no train/eval mismatch, and no gradient-severance
defect that reaches the default/production path. The estimator toggles (`detach_e_step`,
`straight_through`, `unroll`), the emission factor's `e_step_update='mm_exact'` gate, and the
causal-mask / input-vs-target shift conventions (`vfe3/data/datasets.py`, `vfe3/attention_prior.py`)
all checked out correct and are consistently enforced by `config.py`'s `__post_init__` validation.

---

### Full-covariance mean trust region corrupts every coordinate through `solve_triangular`, defeating both `box` and `ball` modes
**Location:** vfe3/numerics.py:148-161
**Severity:** high
**Evidence:** Reproduced directly:
```
delta_mu_f = torch.tensor([1.0, float('inf'), 2.0])
sigma_full = torch.eye(3)
apply_mu_trust_region(delta_mu_f, sigma_full, trust=5.0, mode='box', is_diagonal=False)
-> tensor([nan, nan, nan])
```
Root cause isolated to `torch.linalg.solve_triangular`:
```python
whitened = torch.linalg.solve_triangular(safe_factor, delta_mu.unsqueeze(-1), upper=False).squeeze(-1)
...
bounded = whitened.clamp(-trust, trust)          # (box) -- but whitened is already all-NaN
full_out = (safe_factor @ bounded.unsqueeze(-1)).squeeze(-1)
```
Even solving against the identity, `solve_triangular` with one `inf` entry in the RHS returns
`[1., inf, nan]` — the unrelated third coordinate is corrupted by the LAPACK/BLAS routine before
the trust-region clamp ever runs, so `clamp(-trust, trust)` (line 160, the default and
"recommended" `box` mode) cannot rescue it. This is the E-step mean-trust-region guard
(`cfg.e_mu_q_trust` / `cfg.mu_trust_mode`, config-reachable via `family='gaussian_full'`), whose
entire purpose is to bound an exploding gradient step; instead, in exactly the gradient-blowup
regime it exists for, it turns one non-finite coordinate into a fully NaN belief mean that then
propagates through `mu = belief.mu - delta_mu` in `e_step.py`.
**Fix:** Sanitize/clamp `delta_mu` to finite values (or route through a NaN-safe whitening, e.g.
`torch.where` guarding non-finite input) before the triangular solve, and re-derive `bounded` from
a verified-finite `whitened` rather than trusting the solver's output on a non-finite RHS.

### Diagonal mean trust region's `ball` mode NaNs the exploded coordinate via `inf * 0`
**Location:** vfe3/numerics.py:143-145
**Severity:** medium
**Evidence:**
```python
if mode == "ball":
    norm2 = whitened.norm(dim=-1, keepdim=True)
    return delta_mu * (trust / norm2.clamp(min=eps)).clamp(max=1.0)
```
Reproduced: `apply_mu_trust_region(torch.tensor([1.0, float('inf'), 2.0]), torch.ones(3),
trust=5.0, mode='ball', is_diagonal=True)` returns `tensor([0., nan, 0.])` — the two originally
finite coordinates are **zeroed**, not merely left alone, because `norm2` is `inf` for the whole
row (a per-row reduction), making the ratio `0`; then `delta_mu(inf) * 0 = nan` at the exploded
entry while `delta_mu(finite) * 0 = 0` elsewhere, silently discarding the whole step. Same failure
class as the already-known C-09 (`retraction.py:458/706`) but in a distinct function (the E-step's
mean trust region, not the SPD covariance retraction). The default `box` mode is unaffected in the
diagonal case (`inf.clamp(-5,5) == 5`), so this is reachable only under `mu_trust_mode='ball'`.
**Fix:** As with C-09 — detect a non-finite `whitened`/`norm2` and clamp the tangent directly
(`nan_to_num` before scaling, or clamp `delta_mu` element-wise to the trust ball) rather than
relying on `ratio * exploded_value`.

### `reduced_free_energy`'s inline tau-broadcast does not share `_broadcast_tau`'s length-1 collapse fix
**Location:** vfe3/free_energy.py:376-379 (cf. the fixed sibling at vfe3/free_energy.py:38-51)
**Severity:** low
**Evidence:**
```python
if isinstance(tau, torch.Tensor) and tau.dim() == 1:
    _tau = tau.to(device=lz.device).reshape(tau.shape[0], 1)
elif isinstance(tau, torch.Tensor) and tau.dim() >= 2:
    _tau = tau.squeeze(-1)
else:
    _tau = tau
return -_tau * lz
```
`_broadcast_tau` (used by `attention_weights`/`log_partition`, same file) was fixed under audit
2026-07-26 E-06 to collapse a length-1 per-head tau (the shape `learnable_kappa_beta` produces on a
single-block group, e.g. `log_kappa_beta` of shape `(1,)`) to a bare scalar before broadcasting,
because reshaping it to `(1,1,1)` "prepended a phantom head that broke the attention-map einsum
outright." `reduced_free_energy`'s own tau handling never got the matching fix: for a length-1 tau
it reshapes to `(1,1)` and multiplies against a headless `lz` of shape `(N,)`/`(B,N)`, producing a
result with a spurious leading axis (verified: `torch.Size([1,3])` vs the expected `(3,)`).
Currently inert only because the sole call site (`e_step.py:803`, `phi_alignment_loss`'s canonical
branch) immediately `.sum()`s the result, which is shape-invariant. Any future or diagnostic caller
reading per-row values would silently receive a mis-shaped tensor.
**Fix:** Route `reduced_free_energy`'s tau handling through `_broadcast_tau` (with the appropriate
squeeze for the head-vs-query axis) so the length-1 collapse fix is not duplicated and cannot drift.
