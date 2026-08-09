# Task 7 Report: Documentation and Revision-Bound Verification

## Status before ledger activation

Task 7 documentation and machine-readable test artifacts are complete against final reviewed source
HEAD `dae4bbe3339a24c54b52676d7fa9d9c853b955f8`. This report, the progress ledger, README, design
specification, and both JUnit files will be committed together before the closure ledger is started.
The exact pre-ledger commit hash is reported in the final handoff because a commit cannot contain its
own hash.

The custom closure ledger will be activated after that commit at
`docs/verification/priorbank-head-evidence-canonical-content-ledger.json`. It will intentionally
remain uncommitted: committing after activation would change `HEAD` and invalidate the activated
revision even though the active custom ledger file itself is excluded from the worktree digest. No
tracked artifact may change after `verification_gate.py start`.

## User-facing documentation

`README.md` now documents all three default-off controls:

- `use_priorbank_head_evidence_mixer=True` as normalized per-irrep Gaussian KL evidence weighting;
- `encode_mode="canonical_content_gauge"` as the exact frame-intrinsic control; and
- `encode_mode="canonical_content_projected"` as realized-frame diagonal materialization followed by
  same-forward canonical pullback and analytic full decode.

The README and design specification distinguish the exact mode's phi cancellation from the
projected mode's repeated diagonal projection. They identify the optional unigram term as the
separate post-divergence base rate
`-intrinsic_gaussian_divergence / tau_eff + kappa * log(pi_v)`, not as part of the Gaussian
divergence, frame action, or covariance pullback.

The final reviewed fail-closed boundaries are also explicit. Both canonical modes pin their
import-time encoder registration and callable. The exact mode pins `gaussian_frame_diagonal`. The
projected mode pins the `gaussian_diagonal` canonical-table family and the `gaussian_full` family
plus analytic decoder used after pullback. The head-evidence mixer pins its Gaussian family, Renyi
functional, and decoder identities. These checks apply at both `VFE3Config` and direct `PriorBank`
construction while leaving ordinary registry extension paths open.

Run-artifact provenance is documented as implemented: `pure_path_report.json` records the mixer
toggle, `encode_mode`, and `priorbank_head_evidence_role`; enabling the mixer marks the
free-energy/decode pure-path axis false and records `decoder_kl_irrep_weights`, independently of the
existing post-belief `HeadMixer`. Periodic diagnostics retain realized weights, entropy, and maximum
drift.

Read-only wiki context came from `[[VFE Transformer Program]]`,
`[[Diagonal truncation as gauge regularization]]`, and `[[Parallel transport]]`. Those pages support
the same scientific boundary used here: a flat vertex cocycle cancels shared realized frames in an
intrinsic representation, while general GL congruence does not close the diagonal family and its
re-diagonalization is an approximation.

## Mathematical closure derivations

### Zero-logit full-KL recovery

Let the canonical vocabulary prior be block diagonal,
`p = product_h p_h`, and let a full query have covariance `Sigma`. Its marginal block covariance is
`Sigma_hh`. Direct expansion gives

```text
KL(q || p)
  = 1/2 [sum_h trace(D_h^-1 Sigma_hh)
         + sum_h delta_h^T D_h^-1 delta_h
         - K + sum_h logdet(D_h) - logdet(Sigma)].

sum_h KL(q_h || p_h)
  = 1/2 [sum_h trace(D_h^-1 Sigma_hh)
         + sum_h delta_h^T D_h^-1 delta_h
         - K + sum_h logdet(D_h) - sum_h logdet(Sigma_hh)].
```

Therefore

```text
KL(q || p) = sum_h KL(q_h || p_h)
             + 1/2 [sum_h logdet(Sigma_hh) - logdet(Sigma)].
```

The implemented head-evidence divergence is

```text
D_head(q, p) = sum_h w_h KL(q_h || p_h)
               + 1/2 [sum_h logdet(Sigma_hh) - logdet(Sigma)].
```

Equal logits produce exactly `w_h = 1`, so `D_head = KL(q || p)`, including the unweighted
cross-block correction. For a diagonal query the correction is zero. The derivation also shows why
evidence-logit gradients act only through the weighted block KLs.

### Exact canonical phi cancellation

Write the physical query and source as pushforwards of intrinsic Gaussians,
`q_i = (U_i)_# qbar_i` and `p_j = (U_j)_# pbar_j`. Under the flat vertex cocycle,
`Omega_ij = U_i U_j^-1`, so

```text
(Omega_ij)_# p_j
  = (U_i U_j^-1)_# (U_j)_# pbar_j
  = (U_i)_# pbar_j.
```

Gaussian KL is invariant under a common invertible pushforward. Hence

```text
KL(q_i || (Omega_ij)_# p_j)
  = KL((U_i)_# qbar_i || (U_i)_# pbar_j)
  = KL(qbar_i || pbar_j).
```

The same identity holds when each realized `U_i` includes the configured right positional factor,
provided that exact same factor is used by the flat transport. The exact encoder and decoder stay in
intrinsic coordinates, so the supervised Gaussian divergence is independent of token/positional phi.
Without a separate phi objective, its derivative with respect to `phi_embed` is therefore zero. The
optional `kappa * log(pi_v)` term is vocabulary-only and does not reintroduce frame dependence.

## Final JUnit evidence

CUDA interpreter/device check:

```powershell
C:/anaconda/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

```text
2.10.0.dev20251210+cu128 True NVIDIA GeForce RTX 5090
```

Focused CUDA lane:

```powershell
$env:VFE3_TEST_DEVICE='cuda'
C:/anaconda/python.exe -m pytest -q -p no:cacheprovider `
  --junitxml=docs/verification/priorbank-head-evidence-canonical-content-junit.xml `
  tests/test_priorbank_head_evidence.py tests/test_canonical_content_gauge.py `
  tests/test_canonical_content_projected.py tests/test_prior_bank.py `
  tests/test_head_mixer.py tests/test_additive_table_control.py `
  tests/test_exact_congruence_family.py tests/test_frame_gaussian_family.py `
  tests/test_full_covariance.py tests/test_generate.py tests/test_extract_forward_fidelity.py
```

JUnit aggregate: `tests=230 failures=0 errors=0 skipped=0 time=5.15`.

Broader regression lane:

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
C:/anaconda/python.exe -m pytest -q -p no:cacheprovider `
  --junitxml=docs/verification/priorbank-head-evidence-canonical-content-regression-junit.xml `
  tests/test_train.py tests/test_phase0_forward_beliefs.py `
  tests/test_audit_runtime_semantics_20260720.py tests/test_reporting_additions.py `
  tests/test_run_diagnostics_2026_06_13.py
```

JUnit aggregate: `tests=138 failures=0 errors=0 skipped=1 time=118.118`.

The first pre-fix broader attempt exposed a reproducible interpreter/environment conflict: importing
Torch and then invoking SciPy's MKL-backed bounded `curve_fit` loaded duplicate Intel OpenMP
runtimes and aborted with OMP Error 15. A minimal Torch-to-SciPy reproducer confirmed the boundary;
`MKL_THREADING_LAYER=SEQUENTIAL` selects MKL's non-OpenMP threading layer, made the isolated
`test_ppl_offset_renders` pass, and produced the successful final broader JUnit above. No source
workaround was added.

The planned focused and broader lanes do not include the repository-wide default test. The known
unrelated `tests/test_config.py::test_config_model_defaults` assertion still expects `diagonal`,
while the baseline dataclass default is already `diagonal_chunked`; it was not changed and no
repository-wide-suite claim is made.

## Static and ownership checks

- Ruff passed for all feature implementation files and focused tests, including the three final
  source-review fix commits.
- A wider Ruff invocation that included the longstanding `tests/test_reporting_additions.py`
  fixture reported 18 pre-existing E702 semicolon findings and one pre-existing F821 in a
  failure-only callback. The Task 7/docs diff does not touch those lines; no unrelated lint cleanup
  was attempted. The executable broader lane passed.
- `git diff --check` exited 0.
- `git diff f489fe7 -- train_vfe3.py ablation.py zzzzz.py` was empty.
- The deferred `TODO(frame-conjugated-head-mixer)` occurs exactly once in Python source, at
  `vfe3/model/model.py`; the plan/spec quote it only as documentation.
- `use_priorbank_head_evidence_mixer` remains default false in `vfe3/config.py`.

## Deferred minor and concerns

The Task 4 low-priority explicit `diagonal_chunked` test for exact-mode unigram separation remains
deferred. The independent exact-mode oracle currently pins `diagonal`; general chunked bias/identity
coverage exists elsewhere, but it is not the named explicit separation test. Adding it during this
docs-only closure would not supply a genuine RED phase against the already implemented behavior, so
Task 7 did not manufacture a TDD cycle.

No functional Task 7 blocker remains before ledger activation. The ledger validator result and exact
activated revision will be reported after the pre-ledger commit.

## Task 7 files

- `README.md`
- `docs/superpowers/specs/2026-08-08-priorbank-head-evidence-canonical-content-design.md`
- `.superpowers/sdd/2026-08-08-priorbank-head-evidence-canonical-content/task-7-report.md`
- `.superpowers/sdd/2026-08-08-priorbank-head-evidence-canonical-content/progress.md`
- `docs/verification/priorbank-head-evidence-canonical-content-junit.xml`
- `docs/verification/priorbank-head-evidence-canonical-content-regression-junit.xml`

The post-activation custom ledger is the only Task 7 artifact intentionally left uncommitted.
