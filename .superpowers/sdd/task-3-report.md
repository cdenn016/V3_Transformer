# Task 3: Family-Aware Dispersion, Fisher Geometry, and Head Mixing

Date: 2026-07-16

## Outcome

Task 3 closes S2-I1 through S2-I6, the shared S2-I5/B5 configuration defect, S2-T2, and S2-C1. Family-dependent covariance, Fisher, trust-region, mixing, and diagnostic semantics now dispatch through the existing belief-family registry. Gaussian behavior remains the compatibility baseline. Laplace paths now interpret the stored `sigma` slot as scale `b`, with covariance diagonal `2*b**2`, mean Fisher precision `1/b**2`, trust-region scale `b`, and independent-component mixed scale `sqrt(sum_j A_ij**2*b_j**2)`.

The implementation retains the existing affine decoder, default route choices, and mixer-off default. No live checkout, daily ledger, Research vault, remote branch, or full/slow test suite was touched.

## Root cause and finding disposition

The registry already distinguished Gaussian and Laplace belief objects, but several downstream consumers bypassed it and treated every dispersion tensor as Gaussian variance or covariance. S2-I1 is closed by the family-owned `mix_dispersion` hook consumed by `HeadMixer`. S2-I2 is closed by the family-owned mean Fisher precision consumed by `MahalanobisNorm`. S2-I3 is closed by family-dispatched trust-region whitening. S2-I4 is closed by converting dispersion to the family covariance diagonal before forming attention reliability.

S2-I5/B5 is closed by requiring both `kl_max` and `renyi_order` to be finite and strictly positive. S2-I6 is closed by routing model diagnostics, registered effective rank, numerical health, extraction, covariance-spectrum figures, and covariance ellipses through family-aware statistics and labels. Spectrum and trace requests do not compute a Fisher inverse, so an indefinite full-covariance diagnostic input is reported with its nonpositive eigenvalue and infinite condition number instead of raising during an unrelated inversion.

S2-T2 is closed by explicit configuration and artifact metadata. An active independent-head `block_glk` mixer is labeled `independent_head_nonintertwiner`. Active diagonal-dispersion tied/isotypic routes are labeled as diagonal-projection nonintertwiners under the existing full-congruence contract. Full-covariance tied/isotypic routes are labeled intertwiners. Disabling the mixer remains the gauge-compatible independent-head route. Current defaults are unchanged.

S2-C1 is closed without a forward-path tensor-to-Python conversion. The no-grad identity shortcut uses a certificate containing each parameter's Python identity and tensor version. Any write, replacement, checkpoint load, deep copy, or whole-module deserialization fails the certificate closed and executes the tensor mixer. The public diagnostic `is_identity()` remains explicit and is not called by `forward`.

## RED evidence

The focused regression module was written before production edits. The first run was:

```text
python -m pytest tests/test_2026_07_15_family_remediation.py --tb=short
.FFFFFFFFFFFFFFFFFFFF                                                   [100%]
20 failed, 1 passed, 8 warnings in 0.41s
```

Representative initial failures were `HeadMixer.__init__() got an unexpected keyword argument 'family'`, analogous missing family arguments in normalization, trust-region, and reliability seams, acceptance of nonfinite config values, raw Laplace scale in diagnostics/extraction, absent compatibility metadata, and the forward-path `.item()` probe at `head_mixer.py:148`.

Review corrections also followed literal RED cycles. The tied diagonal compatibility regression failed with `assert True is False`. A copied trained mixer failed because `copied_mean is mean`. The full-Gaussian diagnostic robustness probe raised `torch._C._LinAlgError` while inverting a singular ridged matrix. Laplace numerical health reported `1.000252366065979` where covariance semantics required `1.000504732131958`, and registered effective rank reported `1.923076868057251` where `2*b**2` required `1.7422680412371134`.

## GREEN evidence

The final focused command was:

```text
python -m pytest -p no:cacheprovider tests/test_2026_07_15_family_remediation.py --junitxml=.superpowers\sdd\task-3-focused-green.xml
.....................                                                    [100%]
21 passed in 0.10s
```

The parsed JUnit attributes were `tests=21`, `failures=0`, `errors=0`, `skipped=0`, and `time=0.102`.

The final neighboring command covered family, Laplace, mixer, isotypic mixer, norms, trust region, numerics, config, metrics, model, extraction, artifacts, precision toggles, diagnostics, report, and figure modules:

```text
python -m pytest -p no:cacheprovider tests/test_families.py tests/test_laplace_family.py tests/test_head_mixer.py tests/test_head_mixer_per_block.py tests/test_head_mixer_isotypic.py tests/test_audit_fixes_2026_06_10.py tests/test_norms.py tests/test_mu_trust_region.py tests/test_numerics.py tests/test_config.py tests/test_metrics.py tests/test_experiment_metrics.py tests/test_model.py tests/test_extract.py tests/test_extract_forward_fidelity.py tests/test_model_channel_diagnostics_2026_06_13.py tests/test_run_artifacts.py tests/test_precision_toggles.py tests/test_run_diagnostics_2026_06_13.py tests/test_report.py tests/test_figures_tail.py tests/test_diagnostics.py --junitxml=.superpowers\sdd\task-3-neighbor-green.xml
494 passed, 14 skipped, 58 warnings in 42.00s
```

The parsed JUnit attributes were `tests=508`, `failures=0`, `errors=0`, `skipped=14`, and `time=41.995`. The warnings are expected warnings already exercised by the neighboring configuration tests. A fresh-process `from vfe3.numerics import apply_mu_trust_region` also succeeded after moving registry resolution inside the function, closing the review-discovered circular import. The two JUnit files were read as machine evidence and then removed as task-owned temporary artifacts.

`git diff --check` passed. Ruff was not available in the worktree environment (`No module named ruff`), so no Ruff result is claimed.

## Review and scope

A separate read-only review challenged the family mathematics, Gaussian compatibility, full-covariance behavior, identity certificate, gauge metadata, diagnostics, and import order. Its four concrete findings were corrected: the fresh-process circular import, stale identity state after deep copy, remaining family-blind published diagnostics, and eager full-covariance Fisher inversion during spectrum-only diagnostics. No additional family abstraction was added beyond the hooks required by the task brief.

The intended commit contains the family implementations, their existing consumers, artifact/report propagation required for truthful diagnostics, the focused regression module, and this report. Shared untracked briefs, progress files, the daily ledger, and other agents' reports remain unmodified and unstaged.
