# Verification of the Metrics/Figures Documentation-Gap Investigation

Date: 2026-06-28.

Scope: read-only re-verification of every specific `file:line` claim made in
`docs/analysis/2026-06-28-metrics-figures-documentation-gap-investigation.md`
against source at `origin/main` HEAD `5e88afc941e4b08f96b2b057ad6c1cb41a0fdfa1`.
Nine parallel verifiers (one per cited file plus a documentation-claims agent)
located each symbol, quoted the actual code, and graded the claim. Among the
cited files only `ablation.py` is dirty in the working tree; its two claims were
checked against the committed version (`git show HEAD:ablation.py`), and the
working-tree diff touches only the `SWEEP_ORDER` list (~lines 1062-1093), so the
cited line numbers match HEAD. No source was edited and no experiment was run.

## Verdict

56 specific claims checked: **55 CONFIRMED, 1 PARTIAL, 0 WRONG.**

The report is factually sound. Every metric-key spelling, figure-output
filename, and registry decorator it cites exists in the code exactly as
described, almost always within ~12 lines of the cited location. There are no
fabricated citations and no claim that misdescribes what the code does. This is
materially more reliable than the machine-generated audits the project's
`CLAUDE.md` warns about.

## Per-file results

| File | Claims | Confirmed | Partial | Wrong |
|------|-------:|----------:|--------:|------:|
| `vfe3/run_artifacts.py` | 16 | 15 | 1 | 0 |
| `vfe3/viz/report.py` | 5 | 5 | 0 | 0 |
| `vfe3/viz/figures.py` | 4 | 4 | 0 | 0 |
| `vfe3/viz/extract.py` | 1 | 1 | 0 | 0 |
| `vfe3/model/model.py` | 6 | 6 | 0 | 0 |
| `vfe3/train.py` | 13 | 13 | 0 | 0 |
| `ablation.py` | 2 | 2 | 0 | 0 |
| `scaling_analysis.py` | 7 | 7 | 0 | 0 |
| docs (digest B1, missing spec) | 2 | 2 | 0 | 0 |
| **Total** | **56** | **55** | **1** | **0** |

## The one PARTIAL

`run_artifacts.py:490` is cited as emitting `sigma_trace_cv`. Line 490 actually
emits `fd_gradient_worst_rel_error`; the `sigma_trace_cv` key is real and
spelled exactly, but lives at line 504 (drift of 14, just past the ~12-line
tolerance). The adjacent claims (`sigma_ce_spearman`, `sigma_trace_cv_gate_pass`)
are correct. This is a citation-precision nit, not a substantive error — the
metric exists and is written as described.

## Nuances worth carrying forward (all claims still confirmed)

- **The covariance-gap metric is even more sweep-local than stated.** The report
  self-corrects that `attention_entropy_cov_gap` is not missing. Confirmed: it is
  defined at `extract.py:530` and returns `cov_gap` / `cov_gap_per_token`. The
  wiring at `ablation.py:1402` is inside the generic per-cell diagnostics probe
  and is gated by `if cfg.n_layers == 1:`, so it only fires on single-block cells
  and is consumed only by the `attention_entropy` sweep plotter
  (`_plot_attention_entropy`, ~1892-1910). It is not emitted on a normal
  multi-layer training run, which reinforces the report's point that this is a
  sweep-local diagnostic, not a routine reported one.

- **Citation style: def-line / trigger-line, not emission-line.** Many claims
  cite the function `def` or the triggering call rather than the exact line that
  assigns the metric key (e.g. `run_artifacts.py:367/471/514` are defs;
  `train.py:382/392/538/555/569/883` are the call/guard with the keys one to a
  few lines below). All land within tolerance, but a reader chasing an exact key
  should expect it a few lines past the cited number.

- **The "missing spec" is a real dangling pointer.** `vfe3/viz/figures.py:444`
  comments a path to `docs/superpowers/specs/2026-06-04-publication-figures-design.md`,
  which is genuinely absent (`docs/superpowers/specs/` holds only the 2026-06-21
  and 2026-06-22 specs). This is the report's most concrete actionable item:
  either restore the spec or update the pointer.

- **B1 "produced no artifact" means "not run."** The digest sentence the report
  leans on is in `docs/2026-06-27-ablation-manuscript-digest.md:165` (not
  `2026-06-27-edits.md`): "B1 (Sigma_q calibration) produced no artifact — not
  run." So the gap is that B1 was never executed, while the calibration code path
  (`run_artifacts.py:367` + `report.py:303`) can produce the B1 figure family.
  The report frames this correctly.

- **The scaling negative claim holds.** `scaling_analysis.py` writes only
  `scaling_points.csv` (open at line 114); there is no `json.dump` and no
  `SCALING_ANALYSIS.md` write anywhere, so pooled fits / bootstrap CIs / the
  route F-test currently go only to `print()`. The report's recommendation to
  persist a `scaling_summary.json` + `SCALING_ANALYSIS.md` is well-founded.

## Bottom line

The investigation's evidence base is trustworthy: its `file:line` citations,
metric-key names, and figure inventories check out against HEAD. Its central
thesis — that the codebase already computes far more falsification-relevant
diagnostics (calibration, gauge/SPD/Fisher health, E-step capacity, free-energy
co-descent, validation sanity, optimizer geometry) than the report/digest/
manuscript surface — rests on accurate readings of the code. The recommendations
can be acted on without re-checking the underlying facts; the only correction is
the 14-line drift on the `sigma_trace_cv` citation.
