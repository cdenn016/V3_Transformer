# Second, deeper multi-agent audit, 2026-07-25

Audit date: 2026-07-25.

Audited revision: `ad3a5ad` (`Merge pull request #177 from cdenn016/fix/2026-07-24-bf16-figure-numpy-boundary`),
the fetched `origin/main` at audit start. Work was performed on branch
`audit/2026-07-25-second-deeper`, created fresh from `origin/main`. No source file was
modified; the only file this audit adds is this report.

## Scope caveat: committed code only

This audit ran in a cloud checkout that sees **only committed code**. The owner routinely
carries uncommitted working-tree edits to `train_vfe3.py`, `vfe3/config.py`, `ablation.py`,
`scaling.py`, and `scaling_analysis.py` that were **not** visible here. Every reachability
claim below — every "the toggle that reaches this is X", every "default is Y" — is scoped to
committed `ad3a5ad` and may read differently against the live working tree. Where a finding
depends on a config default, the default is quoted from committed `vfe3/config.py`.

Per the owner's standing audit instruction, "the default toggle is impure" is never reported
as a finding; only a pure path that is broken, unreachable, or silently substituted is. Per
the code-focus instruction, no claim below rests on a comment or docstring: comments are
treated as evidence of intent only, and several findings are precisely that a comment
contradicts the executable code.

## Lenses that ran

Seven parallel investigator lenses, each read-only, each required to quote executable source
and to back mathematical claims with a derivation or a symbolic (sympy) check rather than
numerical agreement alone:

1. Gauge theory and transport (inverse-vs-transpose convention, cocycle consistency, holonomy,
   equivariance of the assembled free energy).
2. Lie algebra and manifold numerics (BCH coefficients, retraction axioms, the matrix-exponential
   Frobenius clamp, SPD guards).
3. Information geometry and divergences (the Renyi alpha to 1 limit, KL argument order at every
   call site, the Fisher factor of two, the f-divergence registry).
4. Gradient correctness (whether the closed forms in `vfe3/gradients/kernels.py` are the true
   gradient of the same functional `vfe3/gradients/oracle.py` differentiates).
5. Free-energy bookkeeping (whether the decomposition reported by `vfe3/metrics.py` sums to the
   scalar the optimizer steps on, verified numerically at tiny dimensions).
6. Attention masking and reduced-precision numerics (the all-masked causal row, and every eps
   floor that can meet a bf16 tensor).
7. Deprecation and dead code (legacy/compat adjudication, active-inference/EFE residue,
   completed migration shims, unread config fields, unselectable registry keys).

Every finding below was re-verified by the lead auditor by re-reading the cited source. Claims
whose only support was a comment, a docstring, or a commit message were rejected. The verifier
verdict is recorded per finding.

## Executive conclusion

No Critical finding was established. The audit closes **one High**, **eight Medium**, and
**twenty Low** findings, and refutes — with evidence — the four mathematical hypotheses the
audit brief prioritized most highly. That refutation is itself the most important result: the
gauge and divergence core of this codebase is in substantially better shape than the brief
assumed.

The single High finding is an asymmetry in mixed-precision handling that two independent lenses
found separately and by different routes. The autograd oracle in `vfe3/gradients/oracle.py`
protects itself with an explicit float32 island under AMP; the closed-form analytic kernel in
`vfe3/gradients/kernels.py`, which is the **default** belief-descent route, has no counterpart.
Under `amp_dtype='bf16'` the two sides of a seam that golden tests pin to agree at float32
tolerance differ by roughly 0.3 percent relative, the gradient-active pair mask flips on 22
percent of pairs, and the `reuse_pairwise_kl_stats` fast path silently self-disables.

The most consequential Medium findings are two diagnostic-versus-objective mismatches in the
model channel — the reported gamma coupling is `n_heads` times the scale that enters the loss,
and the attention-entropy gate that the loss honors is not applied to the reported gamma
meta-entropy — and the matrix-exponential Frobenius clamp, which returns a surrogate operator
that is not `exp(M)` with no warning and no error whenever both of its guard toggles are off.

Refuted, each with quoted evidence and a derivation or symbolic check: the Renyi kernel has an
explicit alpha to 1 KL branch in every divergence path and never divides by a near-zero;
`Omega^{-1}` is computed as a true float64 inverse everywhere it matters and is never
substituted by a transpose except for RoPE rotations, where transpose genuinely is the inverse;
the flat cocycle composes exactly and has identity holonomy for every registered group,
including the non-orthogonal ones; every BCH coefficient is exact and the truncation order is as
documented; every registered retraction satisfies `R_x(0) = x` and `dR_x(0) = id`; the KL
argument order is correct at every one of roughly thirty call sites; the Fisher metric carries
the correct factor in both the variance and the log parameterizations; the `causal_alibi_noself`
prior does **not** produce a fully masked row zero; and the premise that a `1e-6` eps floor is a
no-op in bfloat16 is arithmetically false, because bfloat16 carries float32's exponent range.

## HIGH findings

### H1. The closed-form belief kernel has no float32 island under AMP; its own reference oracle does

Path: `vfe3/gradients/kernels.py:396` (and `:528` in `mm_exact_update`) versus
`vfe3/gradients/oracle.py:140-146`. Severity: **high**.

The oracle guards itself explicitly:

```python
    if torch.is_autocast_enabled(mu.device.type):
        # The outer training autocast scales only the final loss. Inner derivative construction has
        # already happened by then, so a bf16/fp16 transport or objective here cannot be recovered by
        # GradScaler. Re-enter only this oracle under an explicit fp32 island; ...
        with torch.autocast(device_type=mu.device.type, enabled=False):
            return belief_gradients_autograd(
                mu.float(), sigma.float(), mu_p.float(), sigma_p.float(),
                _transport_to_float(omega),
```

The kernel branch has no counterpart. It begins directly at

```python
    mu_k, sigma_k = mu.detach(), sigma.detach()
    mu_t = transport_mean(omega, mu_k)                 # rank-agnostic: (N,N,K) or (B,N,N,K)
```

and runs `transport_mean`, `transport_covariance`, `pairwise_energy` and the einsum-based
`_pair_contract` in the ambient dtype.

Measured, same tiny model, `amp_dtype=None` versus `'bf16'`:

```
amp=None : kernels._diag_kl_filtering_kernel {mu_q: float32, sigma_q: float32, sigma_p: float32, sigma_t: float32,  beta: float32}
amp=bf16 : kernels._diag_kl_filtering_kernel {mu_q: float32, sigma_q: float32, sigma_p: float32, sigma_t: bfloat16, beta: float32}
amp=bf16 : transport_covariance {sigma_in: float32, sigma_out: bfloat16}
amp=bf16 : transport_mean       {mu_in:    float32, mu_out:    bfloat16}
```

and under `torch.autocast("cpu", bfloat16)` with float32 inputs, on a gradient of scale 3.90:

```
|kernel_amp - kernel_fp32|_inf = 1.27e-2
|oracle_amp - oracle_fp32|_inf = 0.0
|kernel_amp - oracle_amp|_inf  = 1.27e-2      (fp32 agreement between the two is ~1e-7)
```

`mm_exact_update`'s target mean shifts by 6.85e-3 for the same reason.

Three consequences follow from the same root cause and are folded into this finding.

First, the saturation mask that exists specifically to keep the hand kernel exactly equal to the
oracle changes its classification. At `kernels.py:437`,
`pair_mask = ((energy > 0.0) & (energy < kl_max)).to(beta.dtype)`; computed from an energy built
on bf16 operands, 2 of 9 entries flip at a convergence-scale gap of 1e-4, and the isolated energy
error is `E_fp32 = 0.0` exactly against `E_bf16 = 5.332e-06` — a spurious nonzero energy where
float32 says the pair has exactly converged.

Second, the sigma pair term is destroyed outright near convergence. `0.5*(1/sigma_t - 1/sigma_q)`
at `sigma_t/sigma_q - 1 = 1e-3` gives `g32 = +1.043081e-05` but `g_bf16 = +0.000000e+00`, because
bf16 rounds `sigma_t` onto `sigma_q`.

Third, the `reuse_pairwise_kl_stats` performance path silently self-disables. At
`kernels.py:412-414` the gate requires
`all(tensor.dtype == torch.float32 for tensor in (mu, sigma, mu_t, sigma_t))`, and `mu_t`/`sigma_t`
are bfloat16 under AMP (measured above). `vfe3/model/model.py:912` and `vfe3/model/block.py:69`
both hardcode `reuse_pairwise_kl_stats=True`, so the AMP toggle cancels the performance toggle
with no diagnostic. (This is adjacent to, but distinct from, the first pass's finding that the
reuse path is inert under bf16: that pass reported the inertness, this one identifies the shared
root cause and the correctness consequences alongside it.)

A separate reduced-precision hazard lives on the same lines and is reachable only under
`amp_dtype='fp16'`. At `kernels.py:153`, `st = sigma_t.clamp(min=eps)` with `cfg.eps = 1e-6`
(`config.py:75`), and at `:190`, `diff_sig = 0.5 * (1.0 / st - ...)`. Since float16's maximum is
65504 and `1/1e-6 = 1e6`, the reciprocal overflows:

```
sigma_t dtype=torch.float32   clamp->1.000000e-06  grad_sigma finite=True   grad_sigma[0,0]=4.999987e+05
sigma_t dtype=torch.bfloat16  clamp->9.983778e-07  grad_sigma finite=True   grad_sigma[0,0]=5.017587e+05
sigma_t dtype=torch.float16   clamp->1.013279e-06  grad_sigma finite=False  grad_sigma[0,0]=inf
```

Scope, mechanically checked on the installed torch: `aten::reciprocal` has an `AutocastCUDA`
kernel but no `AutocastCPU` kernel, so on the RTX 5090 under CUDA autocast the reciprocal is
promoted to float32 and this particular overflow does not fire; it fires under CPU autocast and
anywhere a float16 `sigma_t` reaches the kernel outside autocast. `amp_dtype='fp16'` is a
validated, reachable config (`config.py:2474`; `train.py:1348` wires a `GradScaler` for it).

Reaching toggle: `cfg.amp_dtype='bf16'` or `'fp16'` (committed default `None`); the E-step is
wrapped in autocast at `model.py:1039`. `tests/test_amp.py` contains no kernel-under-autocast
test, which is why this survived.

Fix: wrap the kernel branch of `belief_gradients` and the body of `mm_exact_update` in the same
`torch.autocast(device_type=..., enabled=False)` plus `.float()` island the oracle already uses;
that also restores the `reuse_pairwise_kl_stats` gate and removes the fp16 reciprocal overflow.

Verifier verdict: **CONFIRMED**. Independently reported by the gradient-correctness lens and the
numerics lens, by different routes; the lead auditor re-read `kernels.py:386-421` and
`oracle.py:138-170` and confirms the island exists on one side only and that the reuse gate
requires float32.

## MEDIUM findings

### M1. The reported gamma model-coupling block is `n_heads` times the scale that enters the loss

Path: `vfe3/model/model.py:2870` (diagnostics) versus `vfe3/model/model.py:2050` (objective).
Severity: **medium** (reporting/metrics, not the trained objective).

The objective reduces the gamma rows over heads by **mean**:

```python
            c_rows, me_rows = self._gamma_coupling_rows(
                token_ids,
                model_phi,
                head_reduction="mean",
```

The diagnostic path reduces the same rows by **sum**:

```python
                c_rows, me_rows = self._gamma_coupling_rows(         # (1, N) rows: sum over heads
                    token_ids[:1], gamma_model_phi, head_reduction="sum",
```

This is not a shared convention that merely differs from the loss: the belief channel is
head-sum in both the objective and the diagnostic (`_belief_free_energy_rows._collapse` is
`x.sum(dim=-2)`), so only the gamma block diverges.

Measured, `lambda_gamma=0.5`, ratio of the reported weighted gamma contribution to the gamma
contribution actually present in the loss:

```
n_heads=2 (block_glk):     0.0012723712 vs 0.0006361008  -> ratio 2.000267
n_heads=4:                 0.0017028458 vs 0.0004258156  -> ratio 3.999022
single-block glk:          0.0012692357 vs 0.0012693405  -> ratio 0.999917
```

The discrepancy propagates into `d["total"]` (`model.py:2878-2885` folds `cfg.lambda_gamma * c_rows[0]`
at sum scale) and therefore into the CSV columns `inner_alignment_energy_total`,
`free_energy_total`, `gamma_coupling` and `gamma_meta_entropy` (`train.py:1835-1837`).

Reaching toggle: `lambda_gamma > 0` (or `s_e_step=True`) with any multi-block group — `block_glk`,
the committed default. Fix: pass `head_reduction="mean"` at `model.py:2870` and drop the
compensating rescale, or divide the folded gamma rows by `len(group.irrep_dims)` before they
reach `hierarchical_free_energy_terms`.

Verifier verdict: **CONFIRMED** by direct re-read of both call sites.

### M2. `include_attention_entropy=False` gates the belief entropy out of the report but leaves the gamma meta-entropy in

Path: `vfe3/model/model.py:2879` versus `vfe3/model/model.py:2058-2061`. Severity: **medium**.

The objective gates the meta-entropy row:

```python
            meta_entropy_rows = (
                cfg.lambda_gamma * me_rows
                if cfg.include_attention_entropy else torch.zeros_like(c_rows)
            )
```

The diagnostic does not:

```python
                model_coupling_rows = cfg.lambda_gamma * c_rows[0]   # (N,)
                meta_entropy_rows = cfg.lambda_gamma * me_rows[0]
```

Measured with `lambda_gamma=0.5` and `include_attention_entropy=False`: the reported total
`0.0130803268` reconciles exactly (to 1e-10) with
`self 0.0002453040 + belief 0.0077455379 + 0.5*(gamma_coupling 0.0101729427 + gamma_meta_entropy 0.0000060270)`.
The belief `attention_entropy = 0.0000044844` is correctly excluded; the gamma meta-entropy
contribution `3.0135e-06` is incorrectly included.

Reaching toggle: `include_attention_entropy=False` together with `lambda_gamma > 0`. Fix: apply
the same gate at `model.py:2879` that the objective applies at `:2058-2061`.

Verifier verdict: **CONFIRMED** by direct re-read of both call sites.

### M3. The matrix-exponential Frobenius clamp silently returns a surrogate operator when both guards are off

Path: `vfe3/geometry/transport.py:1330-1353`. Severity: **medium**.

```python
    with torch.no_grad():
        raw_mat_norm = matrix.norm(dim=(-2, -1), keepdim=True)
        ...
        if validity_max_norm is not None and bool((raw_mat_norm > validity_max_norm).any()):
            ... raise ValueError(...)
        mat_norm = raw_mat_norm.clamp(min=1e-8)
        scale = (max_norm / mat_norm).clamp(max=1.0)
        if clamp_monitor:
            ... warnings.warn(... 'returned factor is a surrogate, not exp(M).')
    matrix = matrix * scale
```

With `transport_clamp_monitor=False` and `transport_chart_max_norm=None` — both the committed
defaults at `vfe3/config.py:151` and `:155` — neither guarded branch is taken, `matrix` is
rescaled unconditionally, and the function returns `exp(20*M/||M||_F)`. Measured at K=4,
`||M||_F = 30`:

```
warnings raised: []
rel||out - exp(M)||    = 0.976
rel||out - surrogate|| = 1.58e-07
det(out) = 0.1635      det(exp M) = 0.0662
```

A 98 percent relative error and a wrong determinant, silently. The fail-closed pure path exists
and is one config field away (`transport_chart_max_norm` raises; `transport_clamp_monitor` warns),
which is why this is Medium and not High.

Diagnostic coverage is partial. The only always-on surfacing is `train._warn_phi_transport_clamp`
(`vfe3/train.py:99-163`), which runs on the log/eval cadence only, warns once per channel per
**process** (module globals `_PHI_CLAMP_WARNED`/`_S_PHI_CLAMP_WARNED` are never reset, so a second
run in the same process is silent), and inspects only the four raw tables `phi_embed`,
`s_phi_embed`, `pos_phi_free`, `s_pos_phi_free` individually — whereas the matrix actually
exponentiated is the composed frame (`model._apply_pos_phi`, `model.py:723-736`), whose norm is
not the maximum of the parts.

Reaching toggles: the committed defaults, for any `transport_mode` other than a skew group —
`compute_transport_operators` (`transport.py:1547`) passes `max_norm=inf` only when
`group.skew_symmetric`. Fix: default `transport_chart_max_norm` to a finite value below
`TRANSPORT_CLAMP_MAX_NORM`, or make the clamp branch warn or raise unconditionally when
`scale < 1` (the reduction is already computed under `no_grad` on that line).

Verifier verdict: **CONFIRMED** by direct re-read of `transport.py:1322-1353` and
`config.py:146-155`.

### M4. The coordinate cap that is supposed to keep the exponent inside the clamp does not bound it for `tied_block_glk`

Path: `vfe3/geometry/transport.py:1533-1536` (the stated conditioning argument) against
`vfe3/geometry/lie_ops.py:727-729` (the executable cap). Severity: **medium** (latent on
committed defaults).

The comment states the retraction "bounds `||phi||` (coords) by `max_norm=5.0`, so each block's
Frobenius norm is far inside fp32 matrix_exp's exact regime." `retract_phi` clamps the
**coordinate** norm, so that inference holds only for a unit-norm generator basis. Measured
generator norms and the worst-case embedded norm at `||phi|| = 5`:

```
glk            K=4 H=1  ||G_a||_F=1.000     -> 5.00
block_glk      K=4 H=2  ||G_a||_F=1.000     -> 5.00
tied_block_glk K=4 H=2  ||G_a||_F=1.414     -> 7.07
tied_block_glk K=4 H=4  ||G_a||_F=2.000     -> 10.00
sp             K=4      ||G_a||_F=1.0-1.414 -> 7.07
```

`tied_block_glk` uses `kron(I_H, E_ij)`, so `||G_a||_F = sqrt(H)` and the embedded norm is
`sqrt(H)*||phi||`. At `n_heads >= 16` the retraction's own ceiling permits
`||M||_F >= 20 = TRANSPORT_CLAMP_MAX_NORM`, making M3's silent surrogate reachable with no
drifting parameter at all. The code at `transport.py:1543-1546` already recognizes this
under-bounding, but only for skew groups, where it responds by disabling the clamp; the identical
problem on the non-skew `tied_block_glk` is unaddressed.

Reaching toggle: `gauge_group="tied_block_glk"` with `n_heads >= 16`. Committed configs use
`block_glk` at `n_heads=2`, so this is latent. Fix: clamp `retract_phi`'s `max_norm` against the
embedded norm (`gauge_optim.embedded_phi_frobenius_norm` already computes it exactly) rather than
the raw coordinate norm.

Verifier verdict: **CONFIRMED** as to the mechanism (generator norms and the coordinate-only cap
are both directly readable); the `n_heads >= 16` threshold is arithmetic from those two facts.

### M5. `s_channel_refinement` hardcodes `DiagonalGaussian` at a `KL(s||r)` call site

Path: `vfe3/viz/extract.py:1250-1254`. Severity: **medium** (diagnostics and figures, not training).

```python
    r_sigma = bounded_variance_from_log(pb.r_sigma_log, eps=cfg.eps).expand_as(s1_sigma)
    kl = get_functional("renyi")                                  # KL = renyi at alpha=1
    r  = DiagonalGaussian(r_mu, r_sigma)
    kl_s0_r = kl(DiagonalGaussian(s0_mu, s0_sigma), r, alpha=1.0, kl_max=cfg.kl_max, eps=cfg.eps)  # (N,)
```

Under `family='gaussian_full'`, `PriorBank.encode_s` (`prior_bank.py:715-718`) returns
`(B,N,K,K)` and `r_parameters` (`prior_bank.py:737-740`) returns `(K,K)`, but this site ignores
`cfg.family`, reads `pb.r_sigma_log` directly (bypassing `r_parameters`), and constructs a
`DiagonalGaussian`. Every sibling extraction (`extract.py:830, 976, 1076, 1292`) is family-keyed
via `get_family(cfg.family)`, as are `e_step.py:1414` and `model.py:1762`. This one is not.

Reproduced end-to-end with `family='gaussian_full'`, `s_e_step=True`, `prior_source='model_channel'`,
`lambda_h=1.0`, `vocab_size=5`, `n_layers=1`, `n_heads=2`:

```
K=4 N=4: kl_s0_r shape = (4, 4)   (expected (N,) = (4,))    <- silent wrong shape when N == K
K=4 N=3: RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1
```

Reaching config: `family='gaussian_full'` with `s_e_step=True` (which forces
`prior_source='model_channel'`); the config constructs without complaint. Fix: build both operands
through `get_family(cfg.family)` and `pb.r_parameters()`.

Verifier verdict: **CONFIRMED** by direct re-read of `extract.py:1240-1256`.

### M6. `covariance_from_packed`'s `log_diag` is a log Cholesky pivot, not a log-variance, and the decode reference reads it as a variance

Path: `vfe3/families/covariance_tables.py:23-44`, `vfe3/model/prior_bank.py:1526-1534` and
`:1687-1690`/`:1772-1780`. Severity: **medium**.

```python
    diagonal_variance = bounded_variance_from_log(log_diag, eps=eps)
    chol.diagonal(dim1=-2, dim2=-1).copy_(torch.sqrt(diagonal_variance))
    return chol @ chol.transpose(-1, -2)
```

so `(L L^T)_ii = exp-bounded(log_diag_i) + sum_{j<i} L_ij^2`. That is, `log_diag` parameterizes
`log(L_ii^2)`, the squared Cholesky pivot — **not** the marginal variance. The parameter is
annotated `# (..., K) log-variances` and `_prior_sigma_log_table` is documented as "the (V, K)
log-variance prior table feeding p_i". That is a comment-versus-code mismatch, and a substantive
one: under `family='gaussian_full'` with `prior_source='model_channel'`, the encode and
self-coupling prior is `p_v = N(mu_v, L L^T)` (`prior_bank.py:1528-1533`) while the decode KL's
reference argument is `diag(exp(log_diag)) = diag(L_ii^2)` (`_decode_full`). The same vocabulary
entry therefore carries two different priors, and the decode diagonal is not the marginal diagonal
of the encode prior.

Reproduced with `PriorBank(V=4, K=3, n_gen=1, family='gaussian_full', prior_source='model_channel',
diagonal_covariance=False)` and `s_sigma_lower_embed ~ N(0, 0.5)` simulating training drift:

```
encode prior cov diag (token 0): [1.0, 1.0001089572906494, 1.196299433708191]
decode prior variance (token 0): [1.0, 1.0, 1.0]
```

At zero-init `s_sigma_lower_embed` the two agree exactly, so the divergence appears only as the
packed table trains — which is why no golden test catches it.

Reaching toggles: `family='gaussian_full'` with `prior_source='model_channel'` and
`untie_decode_bank=False`. Fix: have the decode table read
`torch.diagonal(covariance_from_packed(...), dim1=-2, dim2=-1)` on that route, or rename the field
to reflect Cholesky-pivot semantics and state the choice explicitly.

Verifier verdict: **CONFIRMED** as to the parameterization by direct re-read of
`covariance_tables.py:23-44`; the algebra `(L L^T)_ii = L_ii^2 + sum_{j<i} L_ij^2` is immediate
from the assembled Cholesky factor.

### M7. `check_audit_fixes.py` is permanently broken: an orphaned test node from the EFE removal aborts the whole driver

Path: `check_audit_fixes.py:32`. Severity: **medium**.

```python
    ("M3", "tests/test_belief_cache.py::test_cache_supported_gates_result_changing_toggles"),
```

`tests/test_belief_cache.py` was deleted by the 2026-07-18 active-inference/EFE removal and does
not exist (`ls tests/test_belief_cache.py` -> "No such file or directory"). An unresolvable node
id is a pytest **usage** error (exit 4), not a test failure, so collection aborts for the entire
batch and all thirty other pinned audit-fix regressions silently go unrun:

```
$ VFE3_TEST_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 python3 check_audit_fixes.py
collected 0 items
ERROR: file or directory not found: tests/test_belief_cache.py::test_cache_supported_gates_result_changing_toggles
AUDIT-FIX VERIFICATION: FAILURES  (0 passed, 0 failed, 0 errors, 0 skipped, pytest exit code 4, device=cpu)
EXITCODE=4
```

An AST scan of every `"tests/test_*.py::..."` string in the root drivers and `tests/` found
exactly one real orphan; the two other hits (`test_pytest_policy.py:199`,
`test_verification_gate.py:80`) are synthetic fixture strings, not node ids. Fix: delete line 32,
whose subject was removed along with the EFE subsystem.

Verifier verdict: **CONFIRMED** by direct re-read of `check_audit_fixes.py:28-36` and by the
absence of the referenced file.

### M8. The `vfe3/numerics.py` monitor registry is a registry seam with zero production consumers

Path: `vfe3/numerics.py:336-393`. Severity: **medium** (a modularity-constraint gap, not a
correctness defect).

`_MONITORS`, `register_monitor`, `get_monitor`, `run_monitors` and the three registered probes
`_mon_nan`/`_mon_absmax`/`_mon_cond` form a registry whose sole dispatcher is `run_monitors`, and
`run_monitors` has no caller outside its own definition and one test file:

```
$ grep -rn "run_monitors" . | grep -v "^./.git/\|^./.venv/"
./tests/test_numerics.py:10:    run_monitors,
./tests/test_numerics.py:136:def test_run_monitors_record():
./tests/test_numerics.py:137:    rec = run_monitors(torch.tensor([1.0, 2.0, float("nan")]))
./tests/test_numerics.py:143:    rec2 = run_monitors(M, ["condition_number"])
./vfe3/numerics.py:9:    in without editing call sites. ``run_monitors`` emits a CSV/JSON-friendly record.
./vfe3/numerics.py:383:def run_monitors(
```

The actual numerical-health reporter bypasses the registry entirely: `vfe3/viz/extract.py:940`
`numerical_health` imports `nan_inf_fraction` directly and takes conditioning from
`metrics.belief_spectrum(...)["condition"]`. No config field selects monitor names, so the
registered probes are unreachable from training, artifacts, viz, and every driver. This matters
under the project's "config-selected registry behind every seam" constraint: the seam exists but
is not the one production uses. Fix: route `numerical_health` through `run_monitors`, or delete
the registry and keep the direct calls.

A sub-item at `vfe3/numerics.py:265-270` (severity low): `condition_number`'s
`kind: Literal["auto","full","diagonal"] = "auto"` is never supplied by any caller anywhere, so
`kind='full'` and `kind='diagonal'` are dead options and every call takes the shape inference the
docstring itself calls legacy.

Verifier verdict: **CONFIRMED** by the quoted repository-wide grep.

## LOW findings

The following are recorded with path, severity, and fix, in descending order of practical
consequence. Each was verified against the executable source.

**L1 — Two-hop descent direction is not the gradient of the reported F.**
`vfe3/free_energy.py:469`, mirrored at `kernels.py:171` and `:585`. `w2 = beta.detach() @ beta.detach()`
drops the `dW2/dmu . E` term. Kernel and oracle agree with each other to 8.9e-16 (same convention),
but both differ from central finite differences of the scalar `free_energy(...)` that file
assembles: `|analytic - FD|_inf = 1.71e-1` on a gradient of scale 3.20 at `lambda_twohop = 0.3`,
versus 1.9e-9 at `lambda_twohop = 0`. The pure path (`lambda_twohop = 0.0`, the committed default
at `config.py:731`) is exact. Fix: drop the detach on both sides, or state in the `free_energy`
docstring that under `lambda_twohop != 0` the reported F is not the potential of the descent
direction.

**L2 — The reported per-term metrics fields are raw and unweighted, so the logged columns do not sum to the logged total.**
`vfe3/metrics.py:384-409`. `belief_coupling`, `entropy` and `twohop_coupling` are computed without
`lambda_beta`/`lambda_twohop`, while `total` routes through `_belief_free_energy_rows`, which
applies them. The source comment at `metrics.py:380-383` documents this deliberately, but
`train.py:1686-1691` writes the raw fields and `free_energy_total` as sibling per-token columns
with no weight column. Measured naive column-sum against `total`: `lambda_beta=0.5` gives
`0.00828822` versus `0.00418519` (98 percent relative); `lambda_twohop=0.3` gives `0.01764930`
versus `0.01079066` (64 percent); every other config agrees to 7.2e-08. Fix: emit the weights as
CSV columns, or add `*_weighted` siblings as was already done for `hyper_prior_weighted`.

**L3 — Stale comment: the clamp constant is 20.0, the comment says 15.**
`vfe3/config.py:148` reads "the hard Frobenius clamp (max_norm=15) fires" while
`vfe3/geometry/transport.py:1257` sets `TRANSPORT_CLAMP_MAX_NORM: float = 20.0`, and the emitted
warning reads `max_norm=20.0`. Fix: interpolate the constant or drop the number.

**L4 — Stale certificate comment: `skew_symmetric` is documented as a transpose-for-inverse fast path that does not exist.**
`vfe3/geometry/groups.py:40` reads `skew_symmetric: bool  # exp(-M) = exp(M)^T fast path`. No such
path exists: `stable_matrix_exp_pair` produces the negative factor only via `_checked_group_inverse`
(`transport.py:1400`, `:1407`), and `group_element_inverse` explicitly discards the group before
inverting (`del group; return _checked_group_inverse(omega)`, `transport.py:1601-1602`). Every
`skew_symmetric` consumer traced is unrelated to inversion: `max_norm=inf`, the fp64-island key,
the reorth cadence, and retraction defaults. Fix: name what the flag actually gates.

**L5 — Stale rationale for `omega_reorth_every`.** `vfe3/gauge_optim.py:574-576` claims `U^T`
ceasing to be the exact inverse matters for "the transpose fast path build_transport_from_element
relies on for skew groups". It relies on no such path. Verified: `group_element_inverse(U, so_k)`
returns bit-identical tensors at `residual_tol=1e-12` and `1e2` (`torch.equal -> True`) while
`|U^T - U^{-1}| = 2.38e-07`, so the transpose is never consulted. The mechanism itself remains
legitimate (it keeps stored `U` inside `O(K)`); only the stated justification is wrong, and it
would mislead someone deciding whether the toggle is safe to leave off.

**L6 — `_apply_reflection` has no `CompactFactoredTransport` branch.**
`vfe3/inference/e_step.py:240-279`. The dispatcher handles `DirectLinkTransport` and
`FactoredTransport`, then falls through to `omega = built.clone()` at `:276`.
`CompactFactoredTransport` is a dataclass with no `clone` (confirmed at runtime:
`hasattr(CompactFactoredTransport, 'clone') -> False`), so the fallthrough raises `AttributeError`.
Unreachable today — every call site gates compactness on `reflection is None` — but latent if a
future caller drops the gate. Fix: add a compact branch or raise an explicit `TypeError` naming
the incompatibility.

**L7 — `_omega_retract_cayley` NaNs in double-backward at an exactly-zero algebra step.**
`vfe3/geometry/lie_ops.py:884-885`. The Frobenius norm is not wrapped in `torch.no_grad()`, unlike
the structurally identical clamp in `stable_matrix_exp_pair`, whose comment
(`transport.py:1313-1321`) states the reason verbatim. Measured: cayley at `xi=0` gives a
non-finite double-backward; `lie_exp` does not. Not currently reachable — the only call site,
`GaugeOptimizer.step`, is decorated `@torch.no_grad()` — but it is a latent hazard in a public
geometry primitive. Fix: wrap the two norm/scale lines in `torch.no_grad()`.

**L8 — Order-4 BCH truncation is inaccurate over the frame domain the retraction itself permits, and the accuracy gate is default-OFF.**
`vfe3/geometry/lie_ops.py:549`, `:574-579`, `:653-655`. In production `X` is the stored frame
(bounded only by the retraction cap) and `Y` the trust-region step, so the governing term is
`~||X||^{order+1}||Y||`, not the docstring's symmetric `O(||X||^{order+2} + ||Y||^{order+2})`.
Measured relative operator residual on `block_glk` K=4, H=2 at `||Y|| = 0.1`: at `||X|| = 1` the
order sequence converges (o2 2.4e-5, o3 5.9e-6, o4 1.7e-7), but at `||X|| = 5` it has stopped
converging in order (o2 5.3e-3, o3 6.0e-3, o4 2.7e-3) and the order-4 result is off by ~0.3
percent. `bch_residual_max` is the fail-closed gate and is `None` by default; the exact
alternative `gauge_parameterization="omega_direct"` is registered and reachable, so the pure path
exists. Fix: restate the docstring bound, or enable `bch_residual_max` by default when
`phi_retract_mode="bch"`.

**L9 — `mode="bch"` phi retraction does not satisfy `dR_phi(0) = id` in the coordinate chart.**
`vfe3/geometry/lie_ops.py:524-656` via `retraction.py:752-804`. Measured relative
`|dR(0)v - v|`: `euclidean` 1.9e-09, `bch` 7.1e-02 (glk K=3), 6.3e-02 (`block_glk`), 5.1e-02
(`so_k`). Structural, not numerical: `BCH(X,Y) = X + dexp^{-1}_X(Y) + O(Y^2)`. It **is** the
textbook exponential retraction `R_U(v) = U exp(v)` on the group manifold, so no axiom is violated
there; but the incoming `delta_phi` is a chart gradient, so under `phi_retract_mode="bch"` the
applied direction is `dexp^{-1}_phi(-grad F)`, a phi-dependent linear transform of the descent
direction. `mode="euclidean"` is the `retract_phi` default and is exact. Informational.

**L10 — Below the eps floor the kernel keeps the sigma derivative while the oracle zeroes it.**
`vfe3/gradients/kernels.py:152`. The oracle differentiates through `sigma_q.clamp(min=eps)` inside
`renyi_closed_form`, so `dF/dsigma_q` is identically zero once `sigma_q < eps`; the kernel applies
the clamp only inside the arithmetic and never gates. Measured at `sigma[0,0] = eps/2`: kernel
`-9.9998e5`, oracle `0.0`. At `sigma = eps` exactly they agree (clamp backward is inclusive at both
endpoints). Not reachable on the shipped pipeline — `PriorBank`, the SPD retraction, and `mm_exact`
all floor at `cfg.eps`; the one producer that can emit an unfloored sigma into a following block is
the opt-in head mixer (`head_mixer.py:228`/`:236`, `mix_dispersion = A^2 sigma`, no floor). Fix
(defense in depth): gate the kernel's sigma terms on `(sigma_q > eps)`, or floor at the head-mixer
output.

**L11 — A latent silent-uniform-row hazard in the gamma prior fold.** `vfe3/model/model.py:2367-2370`.
`pi = pi / pi.sum(dim=-1, keepdim=True).clamp(min=log_eps)` followed by `log(pi.clamp(min=log_eps))`
turns a row with zero mixture mass into an exactly uniform row over the allowed keys, with no NaN
and no warning:

```
row sums after masking  : [0.0, 0.0, 0.0, 0.0]
resulting log-prior row : [-27.63102149963379, -inf, -inf, -inf]
softmax of that row     : [1.0, 0.0, 0.0, 0.0]
```

Degeneration requires `gamma_prior_weight == 1.0` **and** disjoint beta/gamma supports on some row.
No pair drawn from the current registry achieves that; verified end-to-end that
`gamma_prior_weight=1.0` with `causal_noself`/`causal_alibi_noself` on both channels gives a finite
loss. This is a registry-extension risk: `register_prior` is the documented add-a-variant seam, and
a new prior with a disjoint support would silently produce physically wrong uniform attention
rather than failing. Fix: assert `pi.sum(-1) > 0` on the support before the normalize.

**L12 — Under `causal_noself`, query 0 receives zero belief coupling.**
`vfe3/attention_prior.py:124`/`:210` with `vfe3/gradients/kernels.py:437`. Row 0 places all of beta
on `(0,0)`, which is exactly the `E_ii ~ 0` structural sink the prior exists to remove, so the
saturation mask `(energy > 0) & (energy < kl_max)` kills it:

```
beta[.., row0]                         = [1.0, 0.0, 0.0, 0.0]
pair_mask row0                         = [0.0, 1.0, 1.0, 1.0]
effective pair weight (beta*pair_mask) = [0.0, 0.0, 0.0, 0.0]
```

Token 0's belief is then driven by the alpha self-coupling alone. Arguably the intended semantics
(row 0 has no legal neighbor), but the docstring's stated rationale mentions only the NaN hazard,
not this mask interaction. Comment-versus-code gap.

**L13 — Two metrics are registered under keys the only production selector never contains.**
`vfe3/metrics.py:1763` and `:1774`. At runtime, registered
`['attention_entropy','effective_rank','free_energy_terms','gauge_trace_spread','holonomy_deviation']`
versus selected `['attention_entropy','free_energy_terms','effective_rank']`. `compute_metrics` is
called from exactly one production site (`model.py:2806`) with the hardcoded
`DIAGNOSTIC_METRIC_NAMES` tuple; no config field extends it. Meanwhile `model.py:2904` and `:3400`
call `holonomy_deviation_sampled` and `gauge_trace_spread` directly, bypassing the wrappers. Fix:
drop the two wrappers, or add their keys to `DIAGNOSTIC_METRIC_NAMES` and delete the direct calls.

**L14 — The belief-gradient kernel registry is gated by a hardcoded family literal, so no other kernel is selectable.**
`vfe3/gradients/kernels.py:298-308`. The predicate conjoins `family == "gaussian_diagonal"` with
`has_kernel(family)`, making the latter a tautology (`has_kernel` has exactly one consumer, this
conjunct). `_KERNELS` contains one key. `register_kernel` is advertised as a modularity seam, but a
kernel registered for `gaussian_full` or `laplace_diagonal` can never be dispatched. Distinct from
the first pass's transport-class whitelist (different registry, different file). Fix: replace the
literal with `has_kernel(family)` alone, or document the kernel route as gaussian-diagonal-only.

**L15 — `encode_mode="gauge_fixed"` is a registry key no config can select.**
`vfe3/model/prior_bank.py:1579` with `vfe3/config.py:2098`/`:2109`. The key passes registry
validation and is then unconditionally rejected with `NotImplementedError` two lines later. Of
every key across eighteen registries run through `VFE3Config(field=key)`, this was the only one
rejected unconditionally. Honest and self-documented, but it inflates the registry with an
unreachable key. Fix: remove the registration and keep the stub as a docstring note.

**L16 — `e_step_update` validation is a hardcoded literal desynced from the alias registry.**
`vfe3/config.py:2477-2484`. The canonical source is `_E_STEP_UPDATE_ALIASES` plus the public
`canonical_e_step_update()` helper (`vfe3/inference/e_step.py:48`, `:55`), which `e_step.py:879`
and `extract.py:870` both use. `config.py` neither imports the alias dict nor calls the helper — it
re-spells the map inline as a same-named local variable, so a new alias would be rejected by
`_require` and mis-canonicalized here. The correct pattern is used one file over at
`config.py:1854` and `:1991`. Fix: validate against `tuple(sorted(_E_STEP_UPDATE_ALIASES))` and
call the helper.

**L17 — `merge_legacy_transport_state`'s duplicate-provision guard is unreachable, and the legacy kwargs are production-dead.**
`vfe3/geometry/transport.py:485-511`. An AST scan of every call to `vfe_stack`/`vfe_block`/`e_step`/
`e_step_iteration`/`merge_legacy_transport_state` across `vfe3/`, the root drivers and `tests/`
found zero sites supplying both `transport_state=` and any `connection_W/M/L=`, so the `ValueError`
branch cannot fire. The legacy kwargs are exercised only by `tests/test_e_step.py:165,183` and
`tests/test_extract_forward_fidelity.py:49`. Fix: drop the unreachable guard, or retire one of the
two APIs.

**L18 — `tests/test_extract_forward_fidelity.py` replays a different API than the extractor it claims to mirror.**
`tests/test_extract_forward_fidelity.py:46-58` passes `connection_W/M/L=getattr(model, ...)` while
all three production sites (`extract.py:526`, `:664`, `:1052`) pass
`transport_state=model.transport_state`. The test drifted at `1de67f8` ("refactor: register
trainable transport state"). Latent, because every registered transport's state keys are among the
three legacy names — but a future transport registering a differently-named state tensor would be
silently dropped by the replay while production keeps it, so the fidelity golden would pass against
the wrong forward. Fix: switch `_stack` to `transport_state=model.transport_state`.

**L19 — `CONTROLLED_MAX_TOKENS` is a declared protocol cap that is neither enforced nor persisted.**
`vfe3/viz/embedding_comparison.py:20`. Exactly one occurrence repository-wide (its own definition);
every other `CONTROLLED_*` sibling is read by `figures.py` and/or `report.py`. `controlled_contract()`
persists the other six knobs into the comparison schema but never `max_tokens`, and no path
truncates to 16,384 tokens. Fix: enforce and persist it, or delete it.

**L20 — Two deprecated entry points whose only consumers are their own tests.**
`vfe3/train.py:2028 run_training` (docstring: "DEPRECATED / minimal: superseded by
`train_vfe3.main()`") — no driver calls it; consumers are `tests/test_train.py:230` and
`tests/test_gradient_clipping_config_wiring.py:108-126`. `vfe3/model/head_mixer.py:164-169
HeadMixer.mixer_delta` — no production reader; every production site uses `mixer_deltas`; sole
consumer is `tests/test_head_mixer_isotypic.py:25`. Both classify as TEST-ONLY LIVE, not dead. The
neighboring `_load_from_state_dict` hook at `head_mixer.py:193-198` is a genuine checkpoint shim
and is legitimately reachable.

## Refuted hypotheses and negative results

These matter as much as the findings: four of them are the audit brief's own highest-priority
suspicions, and all four are false against committed `ad3a5ad`.

**The Renyi kernel does not divide by near-zero at `renyi_order = 1.0`.** An explicit KL branch
keyed on `abs(alpha - 1.0) < 1e-6` exists in every divergence path: `gaussian.py:217`
(`DiagonalGaussian.renyi_closed_form`), `gaussian.py:279` (`renyi_per_coord`), `gaussian.py:577`
(`FullGaussian.renyi_closed_form`), `laplace.py:242`, and `base.py:427` (the generic Bregman form).
A second float64 island (`_RENYI_KL_BAND = 1e-2`) covers the cancellation band outside the switch.
Measured against an exact float64 reference: relative error 6.5e-08 at `alpha = 1.0`, 2.0e-08 at
`1+1e-8`, 4.4e-06 at `1+1e-6` (the last KL-branch point), 8.7e-09 at `1+2e-6` (the first divide
point). No NaN, no inf, no catastrophic cancellation. Continuity across the switch is ~5e-06
relative, which is exactly `|dD/dalpha| * 1e-6`, i.e. the true derivative rather than a jump.
`config.py:928-930` correctly requires `renyi_order` finite and positive without forbidding 1.0.

**The inverse-versus-transpose convention is correct.** `Omega` construction from frames uses a
true float64 inverse on all three routes and never a transpose (`stable_matrix_exp_pair` ->
`_checked_group_inverse`; `_blockwise_group_inverse`; `build_transport_from_element` ->
`group_element_inverse`). For `glk`/`block_glk`/`tied_block_glk`/`sp`/`sp_n` the frames are
strongly non-orthogonal (`|U U^T - I|` up to 5.46) yet `|U U^{-1} - I| <= 1.8e-07`. The covariance
congruence `Omega Sigma Omega^T` correctly uses the transpose — the push-forward of a covariance
under a linear map is `A Sigma A^T` for any invertible `A`, and an inverse there would be the bug.
The only transpose standing for an inverse is `transport.py:1962` for RoPE, where `R` is a block
rotation and `|R R^T - I| = 5.96e-08`. The `omega_direct` M-step's `extract_phi(U^T E)`
(`gauge_optim.py:775`, `:790`) is also correct: under left trivialization `U -> U exp(xi)`,
`<E, U xi>_F = <U^T E, xi>_F`, so `U^T E` is the Frobenius-metric trivialized gradient, not a
stand-in for `U^{-1} E`.

**Cocycle consistency and holonomy hold for every registered transport.** The registry was
enumerated programmatically (`flat`, `regime_ii`, `regime_ii_covariant`, `regime_ii_link`,
`regime_ii_link_charted`) across five groups. `flat`, and every non-flat regime at its documented
flat limit, satisfy `Omega_ij Omega_jk = Omega_ik` to <= 4.8e-07, `Omega_ii = I` to <= 1.2e-07, and
`Omega_ij Omega_ji = I` to <= 3.1e-07. Regime-I triangle holonomy `Omega_01 Omega_12 Omega_20 - I`
is <= 2.4e-07 for every group including the non-compact ones. With a nonzero connection the
regime_ii family is path-dependent by construction; that is the curvature it exists to provide.

**Gauge equivariance holds on the pure path, for general-linear as well as orthogonal gauges.**
Three independent probes: at operator level (production `transport_mean`,
`FullGaussian.transport_dispersion`, `pairwise_energy`, `attention_weights`) with a global in-group
gauge applied to beliefs and frames, maximum relative energy deviation <= 9.2e-07 and maximum
`delta beta` <= 8.9e-08 across `block_glk`, `glk`, `so_k`, `tied_block_glk`, `sp` — including a
gauge with condition number 314. End-to-end on the scalar `free_energy_value`, relative `delta F`
<= 1.0e-07, and <= 1.9e-07 with gauge-RoPE at `rope_full_gauge=True`. At builder level every
registry `covariance_class` label is honest: `flat` 2.5e-07, `regime_ii_covariant` 6.5e-07 with a
trained M, `regime_ii_link_charted` 2.9e-07 with a trained L (all declared "covariant"), against
`regime_ii` 2.9e+00 and `regime_ii_link` 1.7e-01 with trained state (both declared "gauge-fixed").

**Every BCH coefficient is exact and the truncation order is as documented.** Verified symbolically
rather than by eye: the free associative algebra over {X,Y} with exact rational coefficients
truncated at total degree 6, `Z = log(exp X exp Y)` computed from the formal series, minus the
code's series transcribed literally from `_bch_dynkin_correction` (`lie_ops.py:263-286`). The
lowest nonzero residual degree is 3, 4, 5, 6 for `order` 1, 2, 3, 4 respectively — so an `order=k`
truncation reproduces the true series exactly through total degree `k+1`, with the first omitted
term at degree `k+2`, matching the docstring. All six degree-5 terms, all signs, and the constants
`1/2`, `1/12`, `-1/24`, `-1/720`, `+1/360`, `+1/120` check out. The packed-coordinate path is
bit-identical to the dense cascade (`max|compact - dense| = 0.0`).

**Every registered retraction satisfies both axioms.** Central-difference test at K=3, float64:
`spd_affine` diagonal `|R(0)-x| = 0.0`, relative `|dR(0)v - v| = 3.9e-11`; `spd_affine` full
6.2e-15 / 3.5e-09; `log_euclidean` diagonal 0.0 / 8.4e-11, full 1.2e-14 / 1.5e-09; omega `lie_exp`
0.0 / 2.1e-10; omega `cayley` 0.0 / 5.1e-10. Source-level argument matches in each case.
`torch.autograd.gradcheck` passes for all four `_SymmetricSpectralMap` kinds, so the
Daleckii-Krein adjoint at `retraction.py:228-261` is correct; `_EighDamped` backward matches stock
`eigh` to 2.2e-16; `_frechet_log_spd` matches finite-differenced `logm` to 1.9e-09.

**KL argument order is correct at every call site.** Left-operand semantics were established
empirically rather than by name: `renyi(q, p, alpha=1) = 5.487412929534912` against analytic
`KL(q||p) = 5.48741340637207` and `KL(p||q) = 5.509253978729248`. `pairwise_energy` orientation
verified against a manual double loop with a random non-symmetric `Omega_ij`:
`max|E - KL(q_i || Omega_ij q_j)| = 1.19e-07` versus `max|E - KL(Omega_ij q_j || q_i)| = 1.39e+00`.
Roughly thirty call sites were enumerated across `e_step.py`, `kernels.py`, `oracle.py`, `model.py`,
`extract.py`, `prior_bank.py`, `pairwise_stats.py`, `transport.py` and `metrics.py`, including the
hand-rolled decode duplicates; all agree with the seam orientation. The barycenter `r`-update
(`prior_bank.py:790-809`) is the moment-match m-projection, the correct minimizer of
`sum_v KL(s_v||r)`.

**The Fisher factor of two is correct in both parameterizations.** Derived symbolically from
score-outer-product integrals: in the variance coordinate `I = diag(1/v, 1/(2 v^2))` with inverse
`diag(v, 2 v^2)`; in log-variance `diag(e^{-l}, 1/2)`; in standard deviation `diag(1/s^2, 2/s^2)`.
The code's `sigma` slot is the variance, and `retraction.natural_gradient` (`retraction.py:736-747`)
applies `nat_mu = sigma*grad_mu`, `nat_sigma = 2*sigma^2*grad_sigma`, and full-covariance
`2 Sigma G Sigma` — exactly `I^{-1}` at that parameterization. `DiagonalLaplace.natural_gradient`
applies `b^2` on both blocks, matching the derived `I_mu = I_b = 1/b^2`. No Fisher preconditioner
is formed over a log-sigma coordinate anywhere, so the log-coordinate factor never enters.
`metrics.py:775` honestly states that `spd_geodesic_distance` omits the Fisher one-half.

**Every registered f-divergence is the divergence it claims to be.** Verified against deterministic
1-D quadrature over 4M points in float64, for both Gaussian and Laplace families: Renyi at
alpha 0.25/0.5/0.75 matches to ~1e-8, as do squared Hellinger, Bhattacharyya and Jeffreys.
Self-divergence `|D(q||q)|` is at most 6.7e-08 over both families, all four functionals and
alpha in {0.5, 1.0, 1.5}. The per-coordinate decomposition matches the summed form to <= 1.9e-06,
and `squared_hellinger` is correctly excluded from the per-coordinate registry.

**The closed-form gradients are the true gradient of the same functional the oracle differentiates,
and that functional is the canonical F.** Three identities were proved symbolically with residual
exactly zero: the Gibbs envelope
`d/dE_j [ sum_k beta_k E_k + tau sum_k beta_k log(beta_k/pi_k) ] = beta_j` for any log-prior and any
tau (this is why the kernel may treat beta as a constant); the state-dependent alpha envelope
`d/dD [ alpha*(D) D + R(alpha*(D)) ] = alpha*(D)` for `alpha* = c0/(b0+D)`; and the hardcoded
diagonal-KL derivatives. Numerically, across dense/factored/compact/direct-link/RoPE transports,
three alpha forms, four `irrep_dims` layouts, scalar/per-head/per-query tau, dense and `-inf`-masked
log-priors, and the `lambda_beta`/`lambda_twohop`/`value`/`need_sigma_grad`/`kl_max`-saturation
toggles: `max |kernel - oracle| <= 9e-16` in float64, and `<= 4e-15` at second order (backprop of
`||grad||^2`), so the kernel is the correct function and not merely correct at a point. The
canonical-versus-surrogate gap is real and correctly characterized — measured
`||canonical - surrogate||_inf = 0.432` and
`||surrogate - (canonical - Cov_beta(E, grad E)/tau)||_inf = 2.2e-16` — but unreachable for the
kernel: `include_attention_entropy=False` is excluded by `uses_kernel_route` (`kernels.py:304`) and
routed to the oracle. The envelope argument here is over beta only, and beta is the exact
closed-form argmin of a strictly convex row problem recomputed at every inner iteration; it is not
a Danskin argument over the E-step fixed point, so it requires neither stationarity nor convergence
and holds at any truncation count.

**`causal_alibi_noself` does not produce a fully masked row zero.** The executable line is
`allowed = (j < i) | ((i == 0) & (j == 0))` at `attention_prior.py:210` (and `:124` for
`causal_noself`), so `(0,0)` is retained. Measured: `head0 row0 = [-0.0000, -inf, -inf, -inf]`,
`beta row0 = [1.0, 0.0, 0.0, 0.0]`, no NaN. Every registered prior was scanned for N = 1..8 with
`window=1` (the config minimum, pinned at `config.py:1471-1473`): zero all-masked rows. Every
`_attention_log_prior` call site passes `(n, n)`, so no `n_query != n_key` route exists. The
mechanism the brief asked about **is** real if it were ever reached — `torch.softmax` on an
all `-inf` row yields NaN, `logsumexp` yields `-inf`, and the NaN reaches the scalar F unfiltered,
with no masked or renormalized fallback anywhere — but no registry entry reaches it. The `-inf`
entries that do exist are handled correctly everywhere downstream: `free_energy.py:450`, `:660`,
`metrics.py:395` and `model.py:1998` all apply `torch.where(torch.isfinite(log_pi), log_pi, zeros)`
after `log_softmax`, and the entropy uses `torch.special.xlogy`, so there is no `0 * -inf`.

**The bf16 eps premise is arithmetically false.** bfloat16 carries float32's exponent range (8
exponent bits, tiny = 1.175e-38), so an absolute clamp floor is representable and binds correctly:
`tensor(0.0, bf16).clamp(min=1e-6) -> 9.98377799987793e-07`, and its reciprocal `1003520.0` is
finite. The premise conflated absolute representability with relative mantissa resolution; clamping
is an absolute-value operation. The related additive fact **is** true — `x + 1e-6 == x` in bf16 for
`x >= ~5e-4` — but an instrumented live forward under `amp_dtype='bf16'` found exactly two additive
eps sites, `retraction.py:458` and `norms.py:126`, both float32, with zero no-op entries. Only two
small-eps clamp sites see a bf16 tensor at all (`kernels.py:153`, `transport.py:1676`) and both
behave correctly; `gaussian.py:209-216` upcasts via `compute_dtype` before clamping and
`pairwise_stats.py:88-90` uses explicit `.float()`, so those are genuine float32 islands. The real
reduced-precision defects are H1's dtype asymmetry and the fp16 reciprocal overflow, not any eps
floor.

**Active-inference/EFE residue is clean apart from M7.** A word-boundary grep for
`efe|expected_free_energy|active_inference|epistemic|pragmatic|ambiguity|risk|salience|novelty`
across `*.py|*.json|*.toml|*.yaml` found only `tests/test_removed_policy_surface.py` (the
intentional removal-contract test), the `policy_sigma_gate_artifact` tombstone at `config.py:2711`,
and three unrelated English uses. All ten former policy fields live only in
`_RETIRED_POLICY_CONFIG_FIELDS` (`config.py:2703-2714`) as load-compat tombstones that raise a
retirement warning and construct no runtime state. No orphaned registry entries, metric keys, or
serialized-artifact fields; no retired field name appears in `README.md`, `AGENTS.md` or `CLAUDE.md`.

**No config field is read by nothing, and no import is orphaned.** All 162 `VFE3Config` fields were
greped for attribute access outside `config.py`; the three apparent zeros resolved on inspection
(`consumed_retired_keys` belongs to `ConfigMigration`, `e_step_gradient` is read via the
`effective_e_step_gradient` property at `model.py:1028`, `force_large_figures` via `getattr` at
`run_artifacts.py:2961`). An AST validator resolved every `vfe3` import across the package, tests,
drivers, benchmarks and tooling against the on-disk module and its module-level names: zero broken
references. Note that `git log --diff-filter=D` was unusable here — the clone is shallow (69
commits, `.git/shallow` present) and contains no deletions or renames — so the migration-shim check
was performed statically instead.

Two dead defensive branches are recorded without being filed as findings, since neither changes
behavior: `pairwise_energy`'s `irrep_dims is None` disjunct (`free_energy.py:192`) is unreachable
because `GaugeGroup.irrep_dims` is a non-Optional `List[int]` and every production caller passes it,
but `len(irrep_dims) == 1` covers the same code; and `_effective_rank_denominator_floor(family=None)`
(`metrics.py:491-495`) is unreachable because `model.py:2792` always supplies `cfg.family`.

Finally, the free-energy reconciliation lens established a framing worth recording explicitly. The
decomposition `metrics.free_energy_terms` reports is a faithful, exactly-reconciling decomposition
of the **inner E-step free energy**, while the scalar the optimizer steps on is cross-entropy plus
outer regularizers; the `-E_q[log p(o|x)]` data term of the canonical F is a gated stub with no
production caller (`free_energy.py:471`; `model.py:1590` is `loss = ce`). They are two different
scalars by design, and the code says so at `train.py:1732-1734`. Within their own scopes both
reconcile to float32 rounding: `metrics.free_energy_terms["total"]` against `free_energy.free_energy`
agrees to at most 4.8e-07 across eight branches; `model.diagnostics()["total"]` against the F the
E-step actually descends agrees to 0.0 on the default path; and the optimized loss decomposes
exactly into `ce + mass_phi + model_channel` (worst gap 1.25e-07). The E-step estimator makes no
difference — `detach_e_step` and `straight_through` produce byte-identical decompositions to
`unroll`. The only genuine bookkeeping defects found are M1, M2 and L2. Sign conventions are
correct: `-E_q[log p]` is subtracted, and the `tau beta log(beta/pi)` entropy term is present in the
reported decomposition.

## Ranked punch list (surviving critical/high/medium)

No fixes were applied. The owner authorizes remediation separately.

1. **H1** — Give the closed-form belief kernel the same float32 island the oracle has
   (`kernels.py:396`, `mm_exact_update` at `:528`). Single highest-value fix: it restores
   kernel/oracle agreement under AMP, un-flips the saturation mask, removes the fp16 reciprocal
   overflow at `:153`/`:190`, and re-enables `reuse_pairwise_kl_stats`. Add a
   kernel-under-autocast test to `tests/test_amp.py`, which currently has none.
2. **M3** — Make the matrix-exponential clamp non-silent: default `transport_chart_max_norm` to a
   finite bound below `TRANSPORT_CLAMP_MAX_NORM`, or warn/raise unconditionally when `scale < 1`
   (`transport.py:1342-1353`).
3. **M1** — Reconcile the gamma head reduction between diagnostics (`model.py:2870`, `"sum"`) and
   the objective (`model.py:2050`, `"mean"`); the reported gamma block is currently `n_heads` times
   the loss scale and that error is in the CSV.
4. **M6** — Resolve the `log_diag` semantics in `covariance_from_packed`: either have the decode
   table read the marginal diagonal of `L L^T`, or rename the field away from "log-variance". The
   encode and decode priors currently diverge as the packed table trains.
5. **M5** — Family-key the `KL(s||r)` operands at `extract.py:1250-1254`; `family='gaussian_full'`
   with `s_e_step=True` either raises or silently returns the wrong shape.
6. **M2** — Apply the `include_attention_entropy` gate to the reported gamma meta-entropy
   (`model.py:2879`) as the objective does at `:2058-2061`.
7. **M7** — Delete `check_audit_fixes.py:32`; the whole audit-fix verification driver currently
   exits 4 and runs zero tests.
8. **M4** — Bound `retract_phi`'s cap against the embedded norm rather than the coordinate norm, so
   `tied_block_glk` at `n_heads >= 16` cannot silently reach M3's surrogate.
9. **M8** — Decide whether the `numerics.py` monitor registry is the seam or the direct calls are,
   and make one of them real.

The twenty Low findings are listed above with fixes; L1 (two-hop gradient/potential mismatch), L3,
L4 and L5 (three stale comments that actively misdescribe the code) and L14/L16 (two registry seams
blocked by hardcoded literals) are the ones most worth folding into whichever fix pass follows.

## Mechanical verification

Environment note: this container had **no Python packages installed at session start** — `torch`
itself was absent. The audit installed `torch 2.13.0+cu130`, `numpy`, `matplotlib`, `pytest`,
`pytest-xdist`, `sympy`, and subsequently the optional `viz`/`data` extras
(`scikit-learn 1.9.0`, `scipy 1.17.1`, `umap-learn`, `networkx`, `arabic-reshaper`, `python-bidi`,
`tiktoken`). This is **not** the owner's pinned environment, and two test outcomes below are
sensitive to that difference; they are reported as environment-attributable rather than as findings.

Baseline lane, before the optional extras were installed:

```
$ VFE3_TEST_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 python3 -m pytest -n 12 --dist loadscope \
      -m "not slow and not cuda and not external" --junitxml=out.xml
EXIT=1
```

JUnit XML, read mechanically (no `-q` added; counts taken from the `testsuite` element, never from
stdout or memory):

```
tests="4217"  failures="28"  errors="0"  skipped="19"  time="756.624"
```

Of those 28 failures, 24 were attributable to absent optional dependencies rather than to the code
under audit: 18 `ModuleNotFoundError: No module named 'sklearn'`, 2
`RuntimeError: Arabic figure text requires the 'viz' dependencies arabic-reshaper and python-bidi`,
1 `UserWarning: Glyph 26085 ... missing from font(s) DejaVu Sans`, 1
`FileNotFoundError: ... 'powershell'` (`test_verification_skill`, a Windows-only path), and 1
`RuntimeError: Found no NVIDIA driver on your system` (`test_cpu_train_on_cuda_capable_host_never_calls_cuda_helpers`,
which requires a CUDA-capable host to exercise its contract).

Two failures are **torch-version-attributable** and are explicitly *not* claimed as findings:

- `tests/test_gauge_optim.py::test_default_adamw_one_step_is_byte_identical_to_golden` — a
  byte-identity hash pin (`866b5860f0a3...` observed against `75c40d1b09a0...` expected). A
  byte-identity golden is by construction sensitive to the torch build.
- `tests/test_audit_full_gaussian_numerics_20260720.py::test_audited_seed_17_reproduces_legacy_failure_and_float64_oracle`
  — `992363.5 == 950196.8125 ± 0.5`. This test deliberately reproduces a float32
  catastrophic-cancellation magnitude, which is precisely the quantity most sensitive to BLAS and
  kernel differences across torch builds.

The audit brief states committed main was green at `ad3a5ad`. This audit cannot confirm or refute
that on the owner's pinned environment; it reports only what this environment produced. The open
obligation is to re-run both tests on the owner's pinned torch build before treating either as a
regression.

Final lane, after installing the optional extras (same command and markers, written to a private
output path so no concurrent investigator could overwrite it):

```
$ VFE3_TEST_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 python3 -m pytest -n 12 --dist loadscope \
      -m "not slow and not cuda and not external" --junitxml=<private>/final.xml
EXIT=1

tests="4217"  failures="8"  errors="0"  skipped="9"  time="562.291"
```

Installing the optional extras removed 20 of the 28 baseline failures, leaving eight, **none of
which is attributable to the code under audit**. Six are environmental: three
`requests.exceptions.ProxyError` reaching `openaipublic.blob.core.windows.net` for the GPT-2 BPE
vocabulary (`test_get_tiktoken_decoder_roundtrips_when_tiktoken_present`,
`test_default_manifest_forms_are_exact_single_gpt2_tokens`,
`test_auto_default_sample_decoder_emits_at_gpt2_vocab` — this container's proxy blocks that host),
one missing CJK glyph in DejaVu Sans, one missing `powershell` binary, and one absent NVIDIA
driver. The remaining two are the torch-version-attributable pair described above
(`test_default_adamw_one_step_is_byte_identical_to_golden` and
`test_audited_seed_17_reproduces_legacy_failure_and_float64_oracle`), unchanged between the two
lanes.

So on this environment the suite is green apart from six environmental gaps and two
build-sensitive goldens; the audit found no test failure caused by a finding in this report, and
correspondingly none of the findings above is evidenced by a failing test — each is evidenced by
quoted source plus a direct probe.

Per-lens targeted lanes, each read from its own JUnit XML: transport/gauge suite
(`test_transport.py`, `test_gauge_groups.py`, `test_regime_ii*.py`, `test_rope.py`,
`test_tier12_transport.py`, `test_audit_transport_registry_20260720.py`, `test_fix_gauge_audit.py`,
`test_full_gaussian_transport_precision_20260721.py`,
`test_p1_compact_phi_block_transport_20260711.py`) — `tests=251 failures=0 errors=0 skipped=0`.
Divergence/families suite (`test_divergence.py`, `test_families.py`, `test_alpha_i.py`,
`test_laplace_family.py`, `test_2026_07_15_family_remediation.py`) — `tests=129 failures=0 errors=0
skipped=1` (the skip is CUDA-gated). Gradients suite (`test_gradients_kernels.py`,
`test_gradients_oracle.py`, `test_free_energy.py`, `test_mm_exact_prior_anchor.py`) — `tests=47
failures=0 errors=0 skipped=0`. Attention/reuse suite (`test_tier12_attention.py`,
`test_p3_pairwise_stats_reuse_20260711.py`) — `tests=95 failures=0 errors=0 skipped=1`.

None of the existing pins covers H1 (no kernel-under-autocast test exists), M1, or M2 — the gamma
head-reduction commensurability and the gamma meta-entropy gate are unpinned, which is why both
survived the first pass.

### Verification-policy status

Under this repository's verification control plane, the findings above are `EVIDENCE_VERIFIED` by
the mechanical output quoted inline, with the following explicit exceptions carried as
`INCONCLUSIVE` with named open obligations: (i) the CUDA-specific halves of H1's fp16 analysis rest
on torch dispatcher-key queries rather than an executed CUDA run — obligation: re-run the kernel
dtype trace and the fp16 reciprocal probe on the RTX 5090 under `torch.autocast('cuda', bf16)` and
`('cuda', fp16)`; (ii) the two torch-version-attributable test failures above — obligation: re-run
both on the owner's pinned torch build; (iii) every reachability claim is bound to committed
`ad3a5ad` and not to the owner's uncommitted working tree. The deterministic ledger validator was
not run and no ledger is named for this report, so this audit does not claim closure-mode status;
it is a findings report, not a ledger closure.
