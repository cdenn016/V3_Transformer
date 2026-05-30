# Diagnostics tier — 2026-05-30

Post-core-build modules (the Phase 0–7 transformer is complete and on `main`): numerical
monitoring + conditioning fallbacks, a metrics registry, and publication-quality
visualizations. Each is modular (registry-backed) and V3-internal tested.

## Numerical monitoring + conditioning fallbacks

### Files created

- `vfe3/numerics.py` — `safe_spd_inverse`, `floor_eigenvalues`, `condition_number`,
  `nan_inf_fraction`, `check_finite`, and a `register_monitor`/`get_monitor`/`run_monitors`
  registry (`nan_fraction`, `abs_max`, `condition_number` probes).
- `tests/test_numerics.py` — 7 tests.

### Changes

Two concerns, both modular. **Conditioning fallbacks** keep the SPD math finite under
ill-conditioning: `safe_spd_inverse` tries `cholesky_inverse` on `M + (eps·10^t)I` for
escalating `t`, falling back to `pinv` (the pure path is `t=0` with the documented ridge;
larger jitter is the guard); `floor_eigenvalues` projects a symmetric matrix to SPD by
clamping its eigenvalues up to a floor; `condition_number` is the spectral `λ_max/λ_min`.
**Runtime monitors** report numerical health as plain scalars through a registry so a new
probe slots in by name; `run_monitors` emits a CSV/JSON-friendly record, and `check_finite`
warns (or raises) on non-finite entries. A theoretically pure path is always available; the
fallbacks activate only when the pure path fails.

### Analytic anchors

- `safe_spd_inverse` matches `torch.linalg.inv` on a well-conditioned SPD (atol 1e-3) and
  stays finite on a singular input (jitter/pinv fallback).
- `floor_eigenvalues` clamps a `{5, 1e-9, −0.3}` spectrum to `≥ 1e-3`.
- `condition_number(I)=1`, `condition_number(diag(1,100))=100`.
- `nan_inf_fraction` counts non-finite entries exactly; `run_monitors` returns the record.

### Test results

```
167 passed
```

7 new tests; no regressions in the 160 prior.

### Commits

- `feat(numerics): SPD conditioning fallbacks + runtime monitor registry`

## Metrics registry

### Files created

- `vfe3/metrics.py` — `effective_rank`, `attention_entropy`, `holonomy_deviation`,
  `gauge_trace_spread`, `free_energy_terms`, and a `register_metric`/`get_metric`/
  `compute_metrics` registry.
- `tests/test_metrics.py` — 7 tests.

### Changes

A registry of pure diagnostic measurements over beliefs / transport / attention, each
reading what it needs from a keyword context so a new probe slots in by name;
`compute_metrics` emits a CSV/JSON-friendly record. `effective_rank` is the spectral
participation ratio `(Σλ)²/Σλ²` (K for a flat spectrum, →1 for one dominant mode).
`attention_entropy` is the mean row entropy of β (log N for uniform). `holonomy_deviation`
is the mean Frobenius departure of the triangle holonomy `Ω_ij Ω_jk Ω_ki` from identity — a
curvature proxy that is ~0 for the flat (Regime I) cocycle and >0 otherwise.
`gauge_trace_spread` is the std of `log|det Ω| = tr(embed(φ))` across tokens (0 at φ=0).
`free_energy_terms` decomposes F into self-coupling / belief-coupling / attention-entropy.

### Analytic anchors

- `effective_rank([1,1,1,1])=4`, `effective_rank([1,1e-9,1e-9,1e-9])≈1`.
- `attention_entropy(uniform)=log N`, `attention_entropy(one-hot)≈0`.
- `holonomy_deviation(flat φ-cocycle)≈0` (every triangle closes), `>1e-2` for random transport.
- `gauge_trace_spread(φ=0)=0`; `free_energy_terms` sums its parts to `total`.

### Test results

```
174 passed (full suite, consolidated)
```

7 new tests; metrics is additive (new files only), no regressions in the 167 prior.

### Commits

- (this entry) `feat(metrics): diagnostic registry (effective rank, attention entropy, holonomy, F-decomposition)`
