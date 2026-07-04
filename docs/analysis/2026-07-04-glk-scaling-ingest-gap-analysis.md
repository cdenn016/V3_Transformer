# GL(K) scaling ingest — gap analysis (grow_K_GL10)

Date: 2026-07-04. Scope: the `vfe3_scaling_results/grow_K_GL10` sweep and its ingest into the
GL(K) manuscripts. Method: 5 expert lenses (ML-scaling, numerical/stats, variational/E-step,
reproducibility, referee) produced 33 candidate gaps; each was adversarially verified against
primary sources. This document consolidates the survivors and records the dismissed candidates
with their disproof.

## Bottom line

The scaling results are **already ingested** and the reported numbers are **correct**:

- `GL(K)_supplementary.tex` §`app:vfe3_scaling` has Table `tab:vfe3_scaling` (all 12 widths
  K10–K120, params, Test PPL mean±std) — every mean/std reproduces from `scaling_points.csv`.
- The pooled fit L(N)=21.12·N^(−0.0873), CI [0.0627, 0.0957], R²=0.968 matches
  `scaling_summary.json` exactly.
- Both figures (`vfe3_gl10_scaling_ce_vs_params.png`, `vfe3_gl10_ppl_vs_embed_dim_offset.png`)
  are current and byte-identical across the generated `figures/` dir and both manuscript copies
  (Research-vault WIP + repo mirror).
- `GL(K)_attention.tex` Simulations table references the sweep best point (K120 = 74.1 PPL).

Nothing needs to be *re-*ingested. What follows is what is missing or at-risk for a
referee-proof presentation, ranked.

## Sweep parameters (verified)

12 widths K∈{10,20,…,120} (knob `embed_dim`) × seeds {6,23,64} = 36 runs, GL⁺(10) block gauge
(`block_glk`, d_head=10, n_heads=K/10, n_gen=10K), 1 layer, 1 E-step, diagonal Gaussian
covariance, `use_prior_bank=false`, WikiText-103, vocab 50257, 60k steps, 491.52M tokens/run.

## Should-fix gaps (ranked)

### 1. The reproducible sweep runs a learned head mixer for K20–K120, undisclosed — contradicts the "no learned NN" purity claim

`scaling.py:131` sets `use_head_mixer=True` as the sweep default; `scaling.py:423–424` disables
it only for single-head cells (`h<2`), i.e. K10 alone. Per-run configs confirm: K10
`use_head_mixer=false`; **K20, K40, K80, K120 all `use_head_mixer=true`** (33 of 36 runs). The
head mixer (`head_mixer.py`) is a learned Schur-commutant `nn.Parameter` coupling — a documented,
user-accepted CLAUDE.md exception that is exact at identity init and breaks strict gauge
equivariance as it drifts.

The manuscript presents this sweep as the pure path: `GL(K)_attention.tex:2063` — "no MLPs,
pointwise activation functions, or learned attention projections (W_Q, W_K, W_V); only a linear
output projection to vocabulary logits is retained"; and `tab:glk_spec` caption (2081) — "a linear
vocabulary decode (no learned W_Q/W_K/W_V, no MLP, no pointwise activations)". Neither the caption
nor the sweep description (2065) mentions the head mixer.

This is a manuscript-accuracy gap, not a claim that the toggle is wrong (the toggle is
intentional). Two ways to close it:

- **Disclose** — add a `use_head_mixer` column to `tab:glk_spec` (and to `scaling_points.csv`),
  and state that the sweep activates the learned head mixer for the multi-head widths.
- **Re-run pure** — if a strictly no-learned-NN scaling claim is wanted, re-run the sweep with
  `use_head_mixer=false` so the config matches the advertised path.

The pure-path requirement of CLAUDE.md is about the pure path *existing under a toggle* (it does:
`use_head_mixer=false`), so re-running is optional; the disclosure is the actual obligation.

### 2. Irreducible-loss convention is inconsistent: E=0 forced on the CE headline, a floor fit on PPL

`scaling_analysis.py:36` sets `with_offset=False`, so the committed CE law L(N)=A·N^(−α) forces
E=0 (extrapolates to CE→0, PPL→1 as N→∞). The same appendix (`GL(K)_supplementary.tex:1464`)
separately fits an offset PPL law PPL(K)=a·K^b+c with c=64.0, R²=0.999, concluding "the curve
decelerates rather than remaining scale-free". A referee comparing to Chinchilla (which reports
the irreducible term E in nats) will flag a floor on one metric and none on the other.

Verified by direct recompute on the 12 per-width means: the offset **CE** fit (with_offset=True,
supported at `figures.py` `_fit_power_law`) gives **E≈3.95 nats, α≈0.558, R²≈0.9996** — the
offset form is strongly preferred and its exponent is ~6.5× the headline α=0.087. So the headline
exponent is convention-dependent. (Note: the CE-space floor is E≈3.95 nats, not ln(64); the PPL
offset law and the CE offset law are separate models, not one transformed into the other.)

Fix: report the CE law both with and without offset, give E in nats, and state which convention
the headline α uses. If E=0 is retained, justify it against the PPL deceleration the paper already
documents.

### 3. α is a fixed-data-budget slice; the top-end models are data-limited — consequence unstated

All 36 runs share `tokens_seen=491,520,000` (single value). The budget is disclosed three times
(1460, 1486) but the interpretive consequence is not: at K120, ~90.7M params against 491.5M
tokens is ~5.4 tokens/param, far below Chinchilla-optimal ~20, so the large-N models are
under-trained and α is a fixed-D slice, not the infinite-data N-exponent. Because this α is
cross-referenced to the RG universality prediction (`app:rg_universality`), the caveat is
load-bearing. Fix: one sentence stating D is fixed, α is an N-exponent at fixed data (data-limited
at the top end), and no D-exponent or compute-optimal frontier is claimed. (This is also why the
generated `scaling_ce_vs_flops.png`, a monotone rescale of N at fixed D, is correctly omitted.)

### 4. Ten distinct source commits with no code-invariance attestation

Provenance: exactly 10 distinct `git_sha`, 1 `data_sha256` across the 36 runs. The caption fences
this as "development-provenance evidence... rather than a frozen-commit benchmark", which is
honest, but there is no attestation that the load-bearing code (model forward, E-step, eval,
metric computation) was invariant across those 10 commits. Fix: either attest that the relevant
modules were unchanged across the SHAs (a diff over the 10 commits, restricted to the model/eval
path), or re-run the endpoint widths (K10, K120) on a single frozen commit to bound the drift.

### 5. E-step capacity signal: only K10 is a clean datum, and it is absent from both the CSV and the appendix

`estep_capacity_gain = test_ce_no_estep − test_ce` is recorded per run, but `test_ce_no_estep` is
6.51 nats at K10 (mixer off) and **48.0 / 64.5 / 52.5 / 64.3** nats at K20/40/80/120 — 4–6× worse
than the uniform-vocabulary bound (ln 50257 = 10.82 nats). The no-E-step decode at K≥20 is
confidently wrong because the model co-adapted with the head mixer and the E-step; the ~43–60 nat
"gain" is a degenerate-baseline artifact, not graceful inference-time capacity. Only K10
(6.60→5.39, a 1.21-nat / 18% gain over a sub-uniform prior) is a defensible inference-time-capacity
point. Additionally, `scaling_points.csv` omits `test_ce_no_estep` and `estep_capacity_gain`
entirely (it keeps only `estep_final_f_per_token`), so the signal is not in the committed tabular
artifact.

Recommendation: do **not** headline "a single E-step yields a large CE reduction" from the K≥20
numbers. If the E-step capacity result is reported at all, use K10 as the clean point and fence
the K≥20 values as a degenerate (mixer-co-adapted) baseline. Add the two columns to the CSV.

## Nice-to-have polish (bundle)

- **Persist the descriptive fits.** The offset-law coefficients (b=−1.05, c=64.0, R²=0.999) and
  the pure-power R²=0.958 live only in prose plus the matplotlib label; they are not in
  `scaling_summary.json` or `SCALING_ANALYSIS.md` (which report only the E=0 power law). They are
  deterministically regenerable from the CSV, so this is polish, but having `scaling_analysis.py`
  emit both the E=0 and offset fits (CE and PPL) into the summary would make every appendix number
  traceable to a committed file. Also persist a weighted reduced-χ² and per-point residuals (the
  pure power law is misspecified relative to seed precision: reduced χ² ≈ 13).
- **Figure uncertainty.** Neither panel draws a fit-uncertainty band from the bootstrap; the
  PPL-offset panel also lacks per-seed scatter and a residual subpanel that the CE panel has. Add
  a shaded α-CI band and the seed cloud.
- **Self-contained caption.** State in the table/figure caption what is held fixed (1 layer, 1
  E-step, d_head=10) and that n_gen=10K couples params to K, so "Params" is not an independent axis.
- **RG cross-reference precision.** State that α here is a parameter-count (N) exponent, distinct
  from the data exponent β the RG predictions concern; the sweep tests neither the R(K) ratio (no
  baseline) nor the same-β prediction (D fixed).
- **Reproduction pointer.** The appendix has no pointer to the entry-point script
  (`scaling.py` → `scaling_analysis.py`); the released path is cited only in the attention.tex body.
- Optional: footnote that the PPL column is a seed-mean of exp(CE), not exp of the fitted CE
  (Jensen gap ≈ 0.001%, immaterial).

## Dismissed candidates (checked, not gaps)

- **Non-embedding-N (Kaplan) refit.** Disproved by recompute: `active_params_per_token / n_params`
  is constant (0.3306→0.3326) across all 12 K, so the two conventions differ by a constant factor
  and α is invariant (0.08571 vs 0.08554, shift 0.00017, inside the CI); only the prefactor
  rescales. The mechanism (differential embedding-vs-core scaling) does not apply at n_layers=1
  where every component scales linearly in K.
- **Bootstrap CI asymmetry / 12-cluster coarseness.** The asymmetric percentile interval is the
  expected output for a skewed small-slope estimator, not an undercoverage artifact; BCa would not
  symmetrize it. Standard, defensible practice.
- **Silent scipy fallback mislabeling the offset figure.** Real code footgun (`figures.py` labels
  "offset fit" without asserting `form=='offset_power_law'`), but the published figure shows
  c=64.0/R²=0.999, which the E=0 fallback cannot produce — so scipy was present at generation and
  the artifact is a true offset fit. Code-hardening nit, not a results gap.
- **Jensen gap** (PPL-mean vs exp(mean CE)): 0.001–0.004% across all widths, three orders below
  the stated 2% non-interpretability floor.
- **Single-layer / width-only scope, fixed-budget disclosure, zero-inference-points.** Already
  stated in the manuscript (subsection title "Embedding-Dimension Scaling"; "M-step only" column;
  triple budget disclosure).
- **Generated CE-vs-FLOPs figure unused.** Correct to omit — at fixed D it carries no independent
  compute information.

## Suggested order of operations

1. Decide on gap 1 (disclose head mixer vs re-run pure) — this is the only item needing a call.
2. Text-only edits for gaps 2, 3, 4, 5-caveat and the RG cross-reference (Research-vault
   manuscripts first, then mirror to `Manuscripts-Theory`).
3. Pipeline polish (persist offset/χ²/residual fits; add CSV columns; figure bands) — coordinated
   with the live WIP in `scaling.py` / `scaling_analysis.py`.

Edit target for any manuscript change: the Research-vault WIP
(`Research/manuscripts/GL(K)_*.tex`), then mirror to the repo `Manuscripts-Theory/`.
