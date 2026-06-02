# 2026-06-01 — Per-coordinate self-coupling alpha^(k)

Branch: vfe3-artifacts-priorbank-2026-05-31 (committed directly).
Design spec: docs/superpowers/specs/2026-06-01-per-coordinate-alpha-design.md.

## Motivation

Investigating warnings from `train_vfe3.py` surfaced that `alpha_mode='state_dependent_per_coord'`
was a stub: the registry advertised a per-coordinate self-coupling but the pipeline fed it the
summed per-position self-divergence, so it broadcast to one alpha per token (identical to
`state_dependent`) and emitted a `RuntimeWarning` on every call (`vfe3/alpha_i.py:129`,
`vfe3/model/model.py:245`). This change realises the per-coordinate path so the mode delivers a
self-term `sum_k alpha^(k) D^(k)` with `alpha^(k)* = c0/(b0 + D^(k))`. The Triton
`cuobjdump.exe` / `nvdisasm.exe` warnings in the same console output are unrelated environment
notices (missing CUDA-toolkit disassembly binaries on PATH) and were not code-addressed.

## What changed

`vfe3/alpha_i.py`
  `register_alpha(name, *, per_coord=False)` now stores a per-form flag; `alpha_is_per_coord(mode)`
  queries it; `state_dependent_per_coord` is registered with `per_coord=True`. The degradation
  `warnings.warn` (and the now-unused `import warnings`) were removed — the form receives the
  per-coordinate divergence and no longer degrades.

`vfe3/divergence.py`
  New `gaussian_diagonal_renyi_per_coord(...) -> (..., K)`: the diagonal Renyi/KL terms left
  unsummed, each clamped independently by `safe_kl_clamp`. The summed kernel is untouched (its
  clamp-the-sum semantics and golden tests are preserved).

`vfe3/free_energy.py`
  New `self_divergence_per_coord(...)` (dispatches on family and functional; raises for anything
  but diagonal + renyi) and `self_divergence_for_alpha(..., *, alpha_mode, ...)` — the single
  routing seam that returns `(..., N, K)` per-coordinate when the selected alpha form declares
  `per_coord=True`, else the summed `(..., N)`.

`vfe3/gradients/oracle.py`, `vfe3/gradients/kernels.py`, `vfe3/inference/e_step.py`,
`vfe3/model/model.py`
  All four alpha consumers now obtain the self-divergence through `self_divergence_for_alpha`.
  In the analytic kernel the unconditional `.unsqueeze(-1)` on the coefficient is gated to the
  per-position case (per-coordinate `sd` is already `(N,K)`), and `_diag_kl_filtering_kernel`
  selects a per-coordinate saturation mask (new `_raw_diag_kl_per_coord`) when `alpha_coef` is
  `(N,K)`, so a saturated coordinate is gated without killing its unsaturated neighbours.

`vfe3/config.py`
  `__post_init__` rejects a per-coordinate alpha form together with a non-diagonal family (the
  per-coordinate divergence does not exist for full covariance), matching the existing
  `tied_block_glk` / `killing_per_block` cross-validation pattern.

## Modularity

No consumer hardcodes the mode name. Each alpha form declares its divergence-reduction need at
registration (`per_coord=`), and `self_divergence_for_alpha` reads that declaration. A future
per-coordinate alpha variant slots in by registering with `per_coord=True`; no call site is edited.
The default pure path (`constant`, `state_dependent`) is unchanged.

## Tests

Six new tests (test-driven, all watched RED then GREEN):
  - `test_alpha_is_per_coord_declares_reduction_need`, `test_register_alpha_per_coord_flag_is_modular`
    (registry declaration, test_alpha_i.py)
  - `test_self_divergence_for_alpha_routes_by_declared_reduction`,
    `test_self_divergence_per_coord_requires_diagonal_renyi` (router + guards, test_free_energy.py)
  - `test_per_coord_alpha_requires_diagonal_family` (config guard, test_config.py)
  - `test_per_coord_alpha_saturation_mask_is_per_coordinate` (the critical gate, test_gradients_kernels.py):
    a mixed-saturation belief (coordinate 0 saturated, coordinate 1 not) pins the analytic kernel
    equal to the filtering autograd oracle and proves coordinate 1's restoring force survives — the
    only regime in which the new per-coordinate mask is observable.

Full suite after the change: `tests=254 failures=0 errors=0` (read from junitxml; 248 baseline + 6
new). End-to-end smoke (forward + diagnostics under `state_dependent_per_coord`, `RuntimeWarning`
promoted to error) passes, confirming the degradation warning is gone and all four consumers route
correctly.

## Mathematical verification

- Per-coordinate decomposition: the diagonal Gaussian KL/Renyi sums over coordinates, so the
  unsummed per-coordinate term is well-defined and `sum_k D^(k)` recovers the pre-clamp summed
  divergence (the `-K` of the summed form becomes `-1` per coordinate). Verified by
  `test_self_divergence_for_alpha_routes_by_declared_reduction` (`per.sum(-1) == summed`).
- Full covariance: KL couples coordinates through `tr(Sigma_t^{-1} Sigma_q)` and the
  log-determinants and does not decompose; the per-coordinate path raises rather than summing
  the wrong thing. Verified by guard test and config cross-validation.
- Envelope cancellation holds coordinate-wise: at `alpha^(k)* = c0/(b0 + D^(k))` with `R^(k)`
  present in F, `d/d(belief)[alpha^(k)* D^(k) + R^(k)] = alpha^(k)* dD^(k)/d(belief)` independently
  per k, because each `D^(k)`, `alpha^(k)`, `R^(k)` depends on coordinate k alone under the
  diagonal family. The analytic kernel therefore matches the autograd oracle with no product-rule
  correction. Rigorously verified by `test_per_coord_alpha_saturation_mask_is_per_coordinate`
  (kernel == oracle for both a saturated and an unsaturated coordinate).
- Per-coordinate clamp: each coordinate's `D^(k)` is clamped at `kl_max`, so a token's total can
  reach `K * kl_max` — a deliberate per-coordinate regularisation scale (design decision), not a
  bug.

## M1+M4 — family `cov_kind` seam (branch vfe3-buildout-roadmap-2026-06-01)

First item implemented from the multi-expert buildout roadmap (`docs/2026-06-01-buildout-roadmap.md`,
findings M1 + M4). Separate session/branch from the per-coordinate-alpha work above; logged here per
the one-post-edit-doc-per-day policy.

### Motivation

The buildout investigation found the codebase inferred a belief's covariance structure (diagonal vs
full) from a name SUBSTRING — `is_diagonal = "diagonal" in family` (`free_energy.py`) — and coupled
config validation to the literal `family == "gaussian_diagonal"` (`config.py`). A future covariance
family whose registered name lacks the "diagonal" token would silently take the wrong energy branch
(wrong broadcast axis, no exception), and a new diagonal family could not be configured without
editing `config.py`'s hardcoded family list. The fix makes covariance structure DATA the family
declares at registration, not a property guessed from its name.

### What changed

`vfe3/divergence.py`
  `register_divergence(name, *, cov_kind)` now REQUIRES a covariance-structure tag ("diagonal" |
  "full"), stored in a new `_COV_KIND` registry. `family_cov_kind(name)` returns it;
  `divergence_families()` returns the registered family names. `gaussian_diagonal` registers
  `cov_kind="diagonal"`, `gaussian_full` `cov_kind="full"`.

`vfe3/free_energy.py`
  `pairwise_energy`'s diagonal-vs-full branch reads `family_cov_kind(family) == "diagonal"` instead
  of `"diagonal" in family`; `self_divergence_per_coord`'s guard dispatches on `cov_kind` likewise.

`vfe3/config.py`
  The hardcoded `_VALID_DIVERGENCE_FAMILIES` tuple is removed; `__post_init__` validates `self.family`
  against `divergence_families()` (the registry) and drives the `diagonal_covariance` consistency
  check and the per-coordinate-alpha guard off `family_cov_kind`, so a newly registered family is a
  valid config family with the correct diagonal/full semantics WITHOUT a config edit.

### Modularity

`cov_kind` is the on-ramp to the spec's `families/` `ExponentialFamily` seam (roadmap item 1): adding
a covariance family is now write-and-register (declare `cov_kind`), never edit-the-call-site. The
gradient-kernel fast-path guard (`kernels.py`: `family == "gaussian_diagonal"`) is deliberately left
as-is — it is a has-this-hand-kernel check, not a cov-kind sniff, and any new family correctly falls
through to the autograd oracle.

### Tests (TDD, watched RED then GREEN)

  - `test_register_divergence_records_cov_kind`, `test_family_cov_kind_unregistered_raises`
    (`test_divergence.py`)
  - `test_pairwise_energy_dispatches_on_declared_cov_kind_not_name` (`test_free_energy.py`) — pins M1:
    a diagonal family whose name lacks "diagonal" still takes the diagonal energy path and equals
    `gaussian_diagonal`.
  - `test_config_diagonal_covariance_cross_check_uses_cov_kind`,
    `test_config_accepts_newly_registered_family_without_editing_config` (`test_config.py`) — the
    latter pins M4: a newly registered diagonal family validates through config with no config edit.
  - Existing `test_registry_register_and_get` updated for the required `cov_kind` kwarg.
  Full suite: `tests=259 failures=0 errors=0` (read from junitxml; 254 baseline + 5 new).

### Behavior preservation

`cov_kind` matches each shipped family's structure, so every existing path is bit-identical; this is a
refactor of the dispatch mechanism plus a new modular seam, not a formula change. The pure default
path (`gaussian_diagonal`) is unchanged.

## families/ seam — Phase 2 byte-identity gate (branch vfe3-buildout-roadmap-2026-06-01)

Plan: `docs/superpowers/plans/2026-06-01-families-exponential-family-seam.md`, Task 5 (Phase 2).
Phase 1 (the additive `vfe3/families/{base,gaussian}.py` layer — `BeliefParams` ABC, `DiagonalGaussian`,
`FullGaussian`, the family/functional registries, and the generic Renyi/KL-from-`A(theta)` path) was
already committed; this entry covers routing the legacy tensor API through that layer.

### What changed

`vfe3/divergence.py`
  Rewritten into a thin tensor-API facade over `vfe3.families`. The `renyi`/`kl`/
  `gaussian_diagonal_renyi_per_coord` entry points keep their historical `(mu_q, sigma_q, mu_t, sigma_t)`
  signature but now wrap the moments in the registered `BeliefParams` subclass and delegate to
  `vfe3.families.base.{renyi,kl}` / `DiagonalGaussian.renyi_per_coord`. The old `_DIVERGENCES`/`_COV_KIND`
  registries, `register_divergence`/`get_divergence`, and the inline `_gaussian_diagonal_renyi`/
  `_gaussian_full_renyi` kernel bodies are removed (they now live in `families/gaussian.py`).
  `safe_kl_clamp`, `family_cov_kind`, `divergence_families`, `register_functional`, `get_functional`, and
  `_warn_alpha_gt_one` are re-exported from `families.base` so callers and back-compat imports are
  unaffected. Importing `divergence` populates the family registry via `from vfe3.families import gaussian`.

  The functional registry now lives in `families.base` (which registers the PARAM-typed `renyi` under
  `"renyi"`). Because `free_energy.pairwise_energy`/`self_divergence` still invoke the functional with the
  TENSOR signature in Phase 2, the facade re-registers the tensor `renyi` under `"renyi"` (mutating the
  shared `base._FUNCTIONALS`) so those call sites keep working; Phase 3 flips them to parameter objects.

`vfe3/free_energy.py`
  Unchanged — its `from vfe3.divergence import family_cov_kind, gaussian_diagonal_renyi_per_coord,
  get_functional` resolves against the re-exports.

### Test migration

The three tests that referenced the removed registry symbols were migrated to the families registry
(each registers a `BeliefParams` subclass via `register_family` and cleans up `_FAMILIES` in a `finally`):
  - `test_divergence.py`: the superseded `test_registry_register_and_get` / `test_registry_unknown_raises`
    (which tested `_DIVERGENCES`/`register_divergence`/`get_divergence`, now covered by
    `test_families.py`) were DELETED; a new `test_divergence_delegates_to_families` pins that the tensor
    `renyi` routes through `DiagonalGaussian` (`atol=0.0`). `test_register_divergence_records_cov_kind`
    and `test_family_cov_kind_unregistered_raises` stay (they use the re-exported `family_cov_kind`).
  - `test_free_energy.py::test_pairwise_energy_dispatches_on_declared_cov_kind_not_name`: registers a
    `DiagonalGaussian` subclass under a no-"diagonal"-substring name (`elliptical_scale_test`) and asserts
    `pairwise_energy(..., family=name) == pairwise_energy(..., family="gaussian_diagonal")`.
  - `test_config.py::test_config_accepts_newly_registered_family_without_editing_config`: registers a
    `DiagonalGaussian` subclass (`laplace_diagonal_test`) and asserts `VFE3Config(family=name,
    diagonal_covariance=True)` passes while `diagonal_covariance=False` raises.

### Byte-identity gate

The closed forms are the same code ported verbatim into `families/gaussian.py`, so the live numerics are
bit-identical; the full suite is the equivalence gate. Full suite after Phase 2:
`tests=269 failures=0 errors=0` (read from junitxml; 270 prior − 2 deleted superseded tests + 1 new
delegation test). No production module references a removed symbol (grep across `vfe3/` for
`register_divergence`/`_DIVERGENCES`/`get_divergence`/`_COV_KIND` returns none; the only `_gaussian_*_renyi`
hits are docstring/comment references, not imports).

## Families seam — Phase 3 (M2): parameter-object divergence interface

The divergence interface was flipped from the four-tensor signature
`(mu_q, sigma_q, mu_t, sigma_t, *, family=..., ...)` to two `BeliefParams` objects
`(q, p, *, ...)`, and every consumer converted in one atomic change (the suite is RED
between the flip and the last consumer, so it was done all at once and gated on the full
suite). The covariance-family selection that used to be a `family=` string is now expressed
by WHICH `BeliefParams` subclass the caller constructs (`get_family(family)(mu, sigma)`);
`divergence_family` (the functional, `"renyi"`) is still a kwarg.

- `vfe3/free_energy.py`: `pairwise_energy(q, key, *, ...)` now takes the query belief and the
  transported key belief, broadcasts the query via `q.broadcast_over_keys()`, and slices irrep
  blocks via `q.block(start, end)` / `key.block(...)`; the `family`/`is_diagonal` plumbing and
  the manual `unsqueeze(-2)`/`unsqueeze(-3)` key-axis logic are gone (the params own their
  layout). `self_divergence`/`self_divergence_per_coord`/`self_divergence_for_alpha` take
  `(q, p)`; the per-coord guard now checks `q.cov_kind == "diagonal"` (message updated to
  "diagonal-covariance family") and dispatches `q.renyi_per_coord(p, ...)`. The unused imports
  `family_cov_kind` and `gaussian_diagonal_renyi_per_coord` were dropped. `free_energy()` (the
  scalar assembler over already-reduced tensors) is unchanged.
- `vfe3/divergence.py`: reduced to a re-export module. `renyi`/`kl` now re-export the
  parameter-typed `vfe3.families.base.renyi`/`kl`; the tensor-signature `renyi`/`kl`/
  `gaussian_diagonal_renyi_per_coord` facades and the Phase-2 `register_functional("renyi")`
  scaffold are DELETED, so `get_functional("renyi")` returns base's param `renyi`. The
  families-registry helpers (`safe_kl_clamp`, `family_cov_kind`, `divergence_families`,
  `register_functional`, `get_functional`, `get_family`, `register_family`) are still re-exported
  (plus a new `__all__`).
- Consumers wrap moment tensors into params at each call site via `fam = get_family(family)`:
  `vfe3/inference/e_step.py` (`free_energy_value`, `phi_alignment_loss`),
  `vfe3/gradients/oracle.py`, `vfe3/gradients/kernels.py` (the two `self_divergence_for_alpha`/
  `pairwise_energy` calls in `belief_gradients`; the analytic kernel math stays tensor-based and
  the `family == "gaussian_diagonal"` kernel-availability guard is untouched),
  `vfe3/model/model.py` (`diagnostics`), and `vfe3/model/prior_bank.py` (`reference_decode` with
  `gaussian_diagonal`, `_decode_full` with `gaussian_full`; the fused `_decode_diagonal` matmul is
  unchanged, not a `kl` call).
- Tests converted to the param API by wrapping moment tensors in `DiagonalGaussian`/`FullGaussian`
  (or `get_family(name)(...)`): `test_divergence.py`, `test_families.py` (the two `legacy_renyi`
  bridge tests now compare `fam_renyi` against the family's own `renyi_closed_form` and assert
  `divergence.renyi is families.base.renyi`), `test_free_energy.py`, `test_gradients_oracle.py`,
  `test_gauge_groups.py`, `test_transport.py`, `test_prior_bank.py`. `test_gradients_kernels.py`,
  `test_e_step.py`, `test_model.py`, `test_use_prior_bank.py` needed no edits (they call only
  `belief_gradients*`/the model, not the divergence functions directly).

### Byte-identity gate

Pure signature/plumbing change with identical numerics. Full suite (excluding `test_viz.py`, which
cannot be collected on this CPU box due to a pre-existing matplotlib/umap `MemoryError` at import,
unrelated to this change): `tests=261 failures=0 errors=0 skipped=0` (read from junitxml), unchanged
from the pre-change baseline of 261. A repo grep confirms no surviving four-tensor call or stray
`family=` passed to `renyi`/`kl`/`pairwise_energy`/`self_divergence*` in any `.py` under `vfe3/` or
`tests/` (the only matches are in `docs/` plan/spec files and the `gaussian.py` docstring source
reference). (The `261` excludes `test_viz.py`'s 8 tests, which fail to COLLECT only under transient
memory pressure; the full suite including viz is `tests=269 failures=0 errors=0`.)

### Final-review fix: generic Bregman-KL reducer for matrix sufficient statistics

The whole-implementation review surfaced a latent bug that undercut the M2 expandability promise.
`_renyi_from_log_partition`'s alpha=1 Bregman inner product summed only the last axis
(`(g*(b-a)).sum(dim=-1)`), correct for a vector sufficient statistic (diagonal Gaussian, the toy
exponential) but wrong for a MATRIX statistic. It was latent (`FullGaussian` always uses its
`renyi_closed_form`, never the generic path), but it is exactly the trap a future matrix-parameter
family using the generic path would hit. Fixed: the reducer now contracts each natural-parameter
component over ITS parameter axes (the trailing dims beyond the batch, inferred from `A(theta)`'s
batch rank), so a `(..., K, K)` statistic is Frobenius-contracted. Pinned by
`test_generic_kl_from_A_works_for_matrix_sufficient_statistic` (`FullGaussian` generic KL equals its
closed form). Suite `tests=270 failures=0 errors=0` (+1 new test). Two cosmetic minors also
addressed: the stale `test_register_divergence_records_cov_kind` was renamed to
`test_divergence_reexports_family_cov_kind`, and the unused `_warn_alpha_gt_one` back-compat
re-export was dropped from `divergence.py`.
