# Duel 5 — `per_head_gauge_invariants` publishes an SVD ratio as a gauge invariant

## SKEPTIC (attack) — returned 10:31 CDT

**Verdict argued: DOWNGRADE to LOW.** Concedes the math and reproduces it; destroys the blast-radius
claim.

### Its probe (block_glk K=4, irrep_dims=[2,2], N=6, float64, same in-group conjugation)
Run on **the columns the report actually publishes**:
```
column                             base   conjugated   rel move
gauge_invariant_mean          -0.363889    -0.363889  4.576e-16
gauge_head_logdet_spread       0.744460     0.744460  0.000e+00
gauge_head_aniso_mean          3.822328     8.481656  1.219e+00
vertex_cond_median             4.943132     9.320270  8.855e-01
vertex_cond_p99                9.387227    27.450897  1.924e+00
vertex_cond_max                9.427705    28.225274  1.994e+00
```

### Attacks that LANDED

1. **The report does NOT publish it.** The run report's gauge/geometry dashboard is `_hist_subset`
   at `run_artifacts.py:4489-4500`, which lists `gauge_invariant_spread`,
   `gauge_head_logdet_spread`, and `vertex_cond_median/p95/p99` — **`gauge_head_aniso_mean` is
   absent, and `aniso` appears NOWHERE in `run_artifacts.py`** (grep: no matches). The held-out
   dashboard list at `:4521-4530` likewise omits it. In figure F7 the anisotropy is loaded at
   `viz/figures.py:2854` but panel B draws it only in the `head_entropy is None` branch
   (`figures.py:2866-2868`); `viz/report.py:846` passes `head_entropy` from `amaps`
   (`report.py:681`), so **on any run with attention maps the anisotropy is computed and
   discarded**. Where it IS drawn, the axis label is `$s_{\max}/s_{\min}$` with title "Per-head
   shear" — the definition, not an invariance claim.

2. **No consumer reads it as an invariant.** Its only live sink is a raw CSV column
   (`train.py:1810`), emitted beside the actual invariant channel
   `gauge_invariant_mean`/`gauge_invariant_spread` (`model.py:3023-3024`), which is fed by the
   group-dispatched `group_gauge_invariant` — the correct function, correct number, correct key. The
   docstring describes it as "its shear/anisotropy `anisotropy` = s_max / s_min"
   (`metrics.py:945-946`); the emitted key says `aniso`; the pinning test pins it AS the SVD ratio
   (`tests/test_omega_direct.py:1218,1225`); `tests/test_metrics.py:252-254` pins only its identity
   value. No gate, no loss, no stop criterion, no manuscript reference (repo-wide grep for
   `gauge_head_aniso` returns `model.py:3052`, `train.py:1810`, three tests, and audit docs).

3. **DECISIVE on severity.** The same in-group conjugation moves `vertex_cond_median/p99/max` by
   **89%, 192%, and 199%** — as much as or more than the anisotropy's 122%. Those four columns **ARE**
   in the report dashboard (`run_artifacts.py:4494`), **ARE** CSV columns (`train.py:1808,1813`), and
   were **the load-bearing evidence closing a prior audit obligation**
   (`docs/audits/audit-obligations-closed-2026-07-25.md:49-51`). The project deliberately publishes
   frame-dependent conditioning/health probes and names them honestly — the same convention that
   labels `holonomy_deviation` "(frame-dependent)" at `viz/figures.py:1592`. **If publishing a
   frame-dependent SVD ratio were a medium defect, `vertex_cond_*` would be four larger instances of
   it.** The anisotropy differs in exactly one respect: the name of the enclosing function.

4. **The finding's "200% under the left vertex action" is NOT probative.** The codebase's stated and
   certified action is conjugation `Omega -> g Omega g^{-1}` (`metrics.py:1554`), and under the left
   action the finding's own preferred `logdet` moves by 1.05 too — so that half of its evidence would
   condemn the entry it endorses.

### What the skeptic CONCEDED
`viz/report.py:843` calls the misnamed function directly, so a future caller could take both
returned entries as invariants. The rename is worth doing for consistency with the `sp` precedent at
`metrics.py:924-931`. **That is a latent API-hygiene hazard — low.**

### Skeptic's decisive evidence
`run_artifacts.py:4489-4500` (the report's gauge panel publishes `gauge_head_logdet_spread` and
`vertex_cond_*`, never the anisotropy) together with its probe showing `vertex_cond_median/p99/max`
move 0.89-1.99 under the identical conjugation.

## DEFENDER — returned 10:55 CDT

**Verdict argued: UPHELD at MEDIUM.** Directly refutes the skeptic's central claim.

### It found the publication surface the skeptic missed
The skeptic checked the run-report *dashboard* (`_hist_subset`, `run_artifacts.py:4489-4500`) and
correctly found the anisotropy absent. But the load-bearing surface is **metrics.csv**:
`train.py:1563` calls `model.diagnostics(tokens)` whenever `do_log or do_csv`; **`train.py:1810`
lists `"gauge_head_aniso_mean"` in the key loop** that writes `row[_dk] = d[_dk]`; `train.py:1902`
calls `artifacts.log_metrics(row)`; `run_artifacts.py:1342` binds
`self.csv_path = self.run_dir / "metrics.csv"` and `:1404-1407` writes through `csv.DictWriter`. At
the live `log_interval = 500`, every 500th step publishes this column. **And
`tests/test_run_diagnostics_2026_06_13.py:202-216` pins the key as a required "Tier-2a" diagnostic —
it is contract, not incidental.**

Also: all THREE branches of the `out.omega` dispatch produce the same quantity —
`model.py:2996`, `:3022`, and `:2985-2988` which inlines `block_svd[..., 0] / block_svd[..., -1]` —
so the finding is branch-independent.

### Measured, float64, block_glk K=4, irrep_dims=[2,2], 8 draws
Using the exact in-group draw the codebase's own equivariance certificate uses
(`metrics.py:1649-1650`):
```
CONJUGATION  A -> g A g^-1   (the action metrics.py:1638 applies to Omega)
  logdet          rel resid : max 3.724e-15  median 2.259e-15
  anisotropy      rel resid : max 3.011e+00  median 6.192e-01
  eig-mod squeeze rel resid : max 8.304e-15  median 2.249e-15

PUBLISHED gauge_head_aniso_mean base = 1.925207
  drift under conjugation (%): [24.1, 6.3, 8.7, 9.2, 0.9, 4.0, 47.7, 67.5]
gauge_head_logdet_spread base = 0.325651  (conj resid max 3.724e-15)
```
**The eigenvalue-modulus squeeze IS the correct invariant at 8.3e-15 — six orders below anything the
SVD ratio does.**

It also settled the action question numerically (`transport.py:69` convention
`Omega_ij = exp(phi_i) exp(-phi_j)`, matched bit-exactly against
`compute_transport_operators["Omega"]`, residual `0.0`): both `A -> g A g^{-1}` and `A -> g A`
induce the SAME transport transformation, and `A -> A h` leaves `Omega` literally unchanged. The
vertex factor carries gauge redundancy the transport observables cannot see.

### THE DECISIVE EVIDENCE — the project already adjudicated this exact construct as HIGH
`metrics.py:924-931`, the `sp` sibling, carries this comment in the source:
```python
    if name in ("sp", "sp_n"):
        # The gauge action is GL(K) CONGRUENCE, so exp(phi) transforms by conjugation g exp(phi) g^{-1};
        # the singular-value squeeze log(s_max/s_min) is invariant only under ORTHOGONAL conjugation, and
        # Sp(2m,R) is not orthogonal. Use the eigenvalue-MODULUS squeeze instead ...
        # (was frame-dependent under svdvals; audit 2026-06-13 review).
```
with the regression test at `tests/test_run_diagnostics_2026_06_13.py:306-321` whose own comment
reads **`Adversarial-review HIGH fix: the sp/sp_n invariant must be invariant under GL congruence
(conjugation of exp(phi)), which the singular-value squeeze was NOT`**. Defender ran both:
`tests=2 failures=0 errors=0 skipped=0`.

**So the project has already classified this exact defect class as a HIGH fix and enforced it — on
`sp`, a group that is DEFAULT OFF — while leaving the identical construct in place on `block_glk`,
the group the live config actually runs.** The per-head test (`tests/test_metrics.py:248-254`)
asserts only identity-init values (`logdet ~ 0`, `anisotropy ~ 1`) with no invariance assertion —
precisely the hole through which this survived.

### Intent is documented as invariance, not as a probe
`metrics.py:942`: `r"""Per-head, per-token GL(d_head) invariants from the converged vertex
factor."""` — plural, covering BOTH returned keys. Contrast the honest twin four lines away:
`model.py:2994-2995` computes the same SVD ratio on the full vertex and publishes it at
`model.py:3031` as `vertex_cond_max` — **named as conditioning, a numerics probe.** And the repo
demonstrably knows how to disclaim frame-dependence when it means to: `viz/figures.py:2760` prints
`"Belief charts are gauge-fixed coordinate diagnostics, not gauge-invariant observables."` No such
disclaimer here.

**No gauge-fixing runs before it.** The only gauge-fixing encoder is a rejected stub
(`config.py:2389-2391` raises on `encode_mode == "gauge_fixed"`; `prior_bank.py:1612-1623` is the
matching `NAMED STUB`). `project_phi_to_slk` is a traceless projection of POSITIONAL coordinates —
it removes the determinant, not the conjugation freedom.

### Defender's four unprompted concessions
1. **The proposed eigenvalue-modulus squeeze fixes conjugation but is NOT invariant under the left
   lift** (measured median 2.5, max 7.8). No function of a single vertex `A_i` can be. It is the
   correct fix for the action the codebase implements and certifies, matching the `sp` precedent —
   not a complete invariance guarantee, and the finding should not be read as claiming one.
2. **`gauge_head_logdet_spread` is clean** (conjugation residual 3.7e-15; the left lift shifts every
   token's logdet by the same `log|det g|`, cancelling in the standard deviation). Severity attaches
   to ONE CSV column, not to `per_head_gauge_invariants` wholesale.
3. **The F7 figure surface is weak evidence** — `figures.py:2849-2850` calls only the log-volume
   "the group-correct gauge invariant", and in the normal report path the anisotropy violin is the
   fallback branch. `run_artifacts.py:4489-4500` also excludes it. **The load-bearing surface is
   metrics.csv.** (This concedes the skeptic's point 1 while relocating the harm.)
4. **The optimizer does not drift freely along the orbit** — `config.py:637`
   `phi_weight_decay = 0.065` (applied `train.py:217`) is itself not gauge-invariant and picks out a
   representative.

## ADJUDICATION — **UPHELD at MEDIUM** (rejecting the skeptic's DOWNGRADE-to-low)

The skeptic's central factual claim — "the report does not publish it" — is **true of the report
dashboard and false of the actual publication surface.** The defender traced
`gauge_head_aniso_mean` to `metrics.csv` through `train.py:1810 -> :1902 ->
run_artifacts.py:1404-1407`, and it is **test-pinned as a required Tier-2a diagnostic key**. A
column written every 500 steps and enforced by contract is published. The skeptic conceded the
figure surface was the weaker evidence; so did the defender, from the other side. Both converge:
the CSV is what matters.

The skeptic's strongest argument — that `vertex_cond_*` moves 89-199% under the same conjugation,
IS in the dashboard, and is honestly named — is real and I accept the factual claim. But it defeats
itself on inspection: `vertex_cond_max` is the **honestly-named conditioning twin of this very
quantity**, computed four lines away at `model.py:2994-2995`. The project already demonstrates it
knows how to publish a frame-dependent conditioning probe under an honest name. The defect is that
this one is published under a `gauge_*` key, from a function named `per_head_gauge_invariants`,
whose docstring says "invariants" — plural, covering both keys. The skeptic said "the only
difference is the name of the enclosing function." That difference **is** the defect.

Decisive on severity: **the project itself already classified this exact construct as an
"Adversarial-review HIGH fix" in its own source comment and test** (`metrics.py:924-931`,
`tests/test_run_diagnostics_2026_06_13.py:306-321`) — and applied the fix only to `sp`, which is
default OFF, leaving it on `block_glk`, which is live. That is not a naming quibble by the project's
own recorded judgment.

Held at MEDIUM rather than high because it is a diagnostic column, not a gate, a loss, or a stop
criterion; `logdet` beside it is genuinely clean; and the defender honestly conceded the proposed
fix does not achieve full lift-invariance. Held above LOW because it is published, contract-pinned,
live on the running group, and its comment (`model.py:3047-3051`) explicitly directs the reader to
interpret it as head specialization on `block_glk` — inviting exactly the wrong inference when
"anisotropy rose over training" can be pure orbit position.

**Punch-list framing:** one CSV column. Either rename it to a conditioning probe (it duplicates
`vertex_cond`) or swap in the eigenvalue-modulus squeeze already used for `sp`, and add the
invariance assertion `tests/test_metrics.py:248-254` currently lacks.
