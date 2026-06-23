# Data-Pending Tracker (relocated from manuscript TeX source)

Date: 2026-06-23

Pass-17 punch-list #1/#2 asked to keep unresolved internal audit notes out of the
distributable TeX source (`Manuscripts-Theory/GL(K)_attention.tex`,
`GL(K)_supplementary.tex`). These notes flag **real open data-reconciliation tasks** for
experiments still being rerun, so they are **preserved verbatim here** (not deleted) and
removed from the `.tex` sources. Re-resolve each against the finalized data before any
public source release, then close it here. The `% \editor{TBD}` placeholders (ATT:39,
SUPP:39) were inert and were removed outright.

## Open items (verbatim, with original source location)

### ATT (was `GL(K)_attention.tex:2086`)
```
% TODO(F3): placeholder hyperparameter table -- cells marked TBD to be populated
```

### ATT (was `GL(K)_attention.tex:2284`)
```
% TODO(review 2026-06-01, DATA-PENDING): these in-text K-sweep PPLs do not match the cited artifact publication_outputs/scaling_analysis/aggregated_K_sweep.csv, which gives mean_test_ppl 222.70(K=10) ... 72.71(K=120); the value 64.9 (and 194.5) appears nowhere in the CSV. The single overlapping point (K=90 GL(10)=76.4) DOES match the CSV. Likely a best-validation-checkpoint vs final-checkpoint policy mismatch (methods.md records final-checkpoint; train_vfe.py reloads best-val before test). Reconcile the curve and the tab:glk_spec "64.9" footnote with the finalized data, and state the checkpoint-selection policy explicitly in the caption and methods.md.
```

### ATT (was `GL(K)_attention.tex:2291`)
```
% TODO(review 2026-06-01, DATA-PENDING): the causal attribution "the VFE's advantage stems from its geometric structure" is not yet tested by the experiment designed to test it. A controlled gauge-ON/OFF ablation harness exists (transformer/vfe/vfe_ablation_suite.py:415-428: arms gauge_on / gauge_off [phi_scale=0,m_phi_lr=0,e_phi_lr=0 => Omega=I] / gauge_frozen_random) but the 'gauge_transport' sweep is commented out of SWEEP_ORDER (:484) and unrun. Run it (>=3 seeds) and report gauge_on vs Omega=I vs frozen-random as a row/panel in tab:glk_results; keep this causal sentence only if the ablation supports it.
```

### ATT (was `GL(K)_attention.tex:2415`)
```
% TODO(review 2026-06-01): the 1.91x/1.87x ratios are computed against the GL(10) VFE (test PPL 76.4); refresh them when the finalized PPLs are in. The honest parameter-matched comparison remains the 84.2M standard transformer (PPL 48.5), which outperforms the VFE.
```

### SUPP (was `GL(K)_supplementary.tex:1023`)
```
% TODO(review 2026-06-01, DATA-PENDING): the g1_tot column violates its own defining identity g1_tot = g1_orig + g1_emer (stated at SUP eq near line 935 and ATT:2295). Row-by-row, g1_orig+g1_emer = 0.300, 2.471, 1.214, 0.798, 1.442, 1.194, 0.674 vs the printed g1_tot 0.300, 42.47, 35.29, 33.15, 47.29, 46.64, 34.74 (a non-constant 17-52x discrepancy, so not a single units typo). Re-extract or recompute the g1_tot column from its definition against the finalized data and verify the additive identity at every level before resubmission; state the norm convention in the caption if g1_tot is measured by a different norm than its components. The qualitative claim (g1_tot does not decay) is independent of the column's absolute scale. ALSO (pass 8, DATA-PENDING): the printed graph-based exponents use inconsistent fit windows: y2=-0.66 matches the full levels 0-6 window, while y3=+0.17 reproduces only from levels 1-5, yet the caption states the y3 fit uses levels 0-5 (which gives ~+0.6 to +0.8 from the printed g3 column). State the exact level window per exponent and make the y2/y3 windows consistent when re-extracting from the finalized CSV.
```
