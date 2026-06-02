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

### Opt-in M-step self-coupling regularizer alpha_hat * sum_i KL(q_i*||p_i)

Branch: vfe3-roadmap-overnight-2026-06-02. Manuscript Algorithm 1 (GL(K)_attention.tex:2083) writes
the M-step loss as `L = L_CE + alpha_hat * sum_i KL(q_i*||p_i) + (alpha_phi/2)||phi||^2`. The training
loss in `vfe3/model/model.py` forward previously carried only CE plus the optional `mass_phi` gauge
penalty (the alpha_phi term); the self-coupling KL term was absent from every path. It is now wired as
an OPT-IN, DEFAULT-OFF fixed scalar coefficient (like `mass_phi`, not a learned parameter), so the
pure/current path is byte-identical at the default.

Files: `vfe3/config.py` adds `mstep_self_coupling_weight: float = 0.0` (alpha_hat) with a
`>= 0.0` `__post_init__` check mirroring `mass_phi`. `vfe3/model/model.py` forward adds, guarded by
`cfg.mstep_self_coupling_weight > 0.0`, the term `weight * sc` where `sc` is the mean self-divergence
of the CONVERGED belief (`out.mu`/`out.sigma`, BEFORE head_mixer/norm) vs the per-block prior. The
last-block prior is reconstructed exactly as `diagnostics()` does it: start from the encode belief and
fold `vfe_stack`'s `prior_handoff` blend over `n_layers-1` (`rho=prior_handoff_rho`,
`rho_s=prior_handoff_sigma`), then `self_divergence_for_alpha(fam(out.mu,out.sigma),
fam(mu_p,sigma_p), ...).mean()`. The term is grad-connected (no detach), so it backprops to the
learned prior tables like `mass_phi`. Exact at `n_layers=1` (the fold loop is empty, so p = encode
belief); an approximation otherwise (one converged belief stands in for the per-block intermediates).

Tests (`tests/test_mstep_self_coupling.py`, TDD oracle-first, watched RED then GREEN): `test_noop_at_weight_zero`
(the key oracle — at weight 0, returned `loss == ce` with `mass_phi=0`, so the new code changes
nothing), `test_linear_in_weight` (pins the term — `loss` allclose `ce + w * sc` with `sc`
independently recomputed by the forward recipe; `assert sc > 1e-6` keeps it non-vacuous),
`test_config_validation` (`-1.0` raises, `0.0`/`0.5` accepted), and
`test_backward_finite_grads_on_prior_tables` (grad-connected: `loss.backward()` yields finite,
nonzero `mu_embed.grad`). Suite `tests=274 failures=0 errors=0` (+4 new; viz collected normally).

### register_retraction seam over the SPD retraction (roadmap item 4)

Branch: vfe3-roadmap-overnight-2026-06-02. Design spec:
`docs/superpowers/specs/2026-06-01-spd-retraction-variants-design.md` (Phase 0). The SPD covariance
retraction was the one geometry seam still dispatched by a hardcoded tensor-rank branch
(`belief.sigma.dim() == belief.mu.dim() + 1` selecting `retract_spd_full` vs `retract_spd_diagonal`
in `e_step_iteration`) rather than a config-selected registry, violating the clean-room spec's
"add a variant by writing-and-registering, never by editing call sites" (sec 4.2). This is a
BYTE-IDENTITY refactor that adds the registry and registers the current affine-invariant retraction
as the default; it adds NO new retraction variants (log-Euclidean / Bures-Wasserstein remain
deferred to the design spec for the user to decide).

Boundary choice: the NARROW seam. The registered retraction owns only the diagonal-vs-full rank
decision plus the SPD retraction call; the Fisher metric conversion (`natural_gradient`) stays in
the E-step, so the tangent arrives already preconditioned. (The spec's broader boundary — folding
`natural_gradient` into each mode and returning `(nat_mu, sigma_new)` — is open user-decision #2,
needed only once variants land; building it now would implement an unapproved decision and enlarge
the byte-identity surface.) The task's own oracle is the discriminator: it compares the seam to the
bare `retract_spd_{diagonal,full}` functions alone, not to a `natural_gradient`+retraction
composition.

Files:
- `vfe3/geometry/retraction.py`: new `_RETRACTIONS` registry, `register_retraction(name)` decorator,
  `get_retraction(name)` (KeyError-with-available-list on miss), mirroring `_PRECOND`/`register_precond`
  in `phi_preconditioner.py`. New `retract_spd_affine(sigma, delta_sigma, mean_ndim, *, step_size,
  trust_region, eps, sigma_max)` registered under `"spd_affine"` — a thin dispatcher that replicates
  the E-step's old branch (`sigma.dim() == mean_ndim + 1` -> `retract_spd_full`, else
  `retract_spd_diagonal`) and forwards verbatim. The `mean_ndim` int is passed because the seam sees
  only `sigma` (rank-3 is ambiguous: batched-diagonal `(B,N,K)` vs unbatched-full `(N,K,K)`); the
  reference is the belief mean's rank, exactly the quantity the old branch used. The bare
  `retract_spd_diagonal`/`retract_spd_full`/`natural_gradient` functions are untouched.
- `vfe3/config.py`: new field `spd_retract_mode: str = "spd_affine"` beside `phi_retract_mode`,
  validated in `__post_init__` against the retraction REGISTRY (`tuple(sorted(_RETRACTIONS))`, the
  `divergence_families()` pattern) so a future registered retraction is selectable without a config
  edit. Default keeps the manuscript-canonical pure path.
- `vfe3/inference/e_step.py`: the rank branch in `e_step_iteration` collapses to
  `get_retraction(spd_retract_mode)(belief.sigma, -e_sigma_lr * nat_sigma, belief.mu.dim(), ...)`;
  `spd_retract_mode` threaded as an `e_step_iteration` parameter and as an explicit accept-and-ignore
  knob on `free_energy_value` (the `e_step` call site forwards one kwarg bag to both, so a missing
  declaration would TypeError on the trajectory/diagnostics path). Import trimmed to `get_retraction,
  natural_gradient, retract_phi` (the two bare SPD functions are no longer referenced here).
- `vfe3/model/block.py`: threads `spd_retract_mode=cfg.spd_retract_mode` into the `e_step` call,
  beside `phi_retract_mode=cfg.phi_retract_mode`.

Byte-identity gate: `tests/test_retraction.py` adds `test_retraction_registry_round_trip` (register/
get round-trip + unknown-name KeyError, registration cleaned up in `finally`),
`test_spd_affine_is_registered`, and `test_spd_affine_bit_identical_{diagonal,full}` — the oracles,
asserting `torch.equal` (atol=0) between `get_retraction("spd_affine")(...)` and the bare
`retract_spd_{diagonal,full}` calls on fixed-seed `(B,N,K)` and `(B,N,K,K)` inputs.
`tests/test_config.py` adds `test_config_spd_retract_mode_validated`. `tests/test_e_step.py` adds
`test_e_step_iteration_spd_affine_default_is_byte_identical` (the default-routed `e_step_iteration`
sigma update equals a hand-composed `natural_gradient`+`retract_spd_diagonal`, atol=0); the
pre-existing `test_fixed_seed_regression` checksum stayed green as an additional end-to-end guard.
Full suite `tests=280 failures=0 errors=0 skipped=0` (read from junitxml; 274 baseline + 6 new; viz
collected normally).

### cross_couplings reachable from config (roadmap item)

Branch: vfe3-roadmap-overnight-2026-06-02. The cross-coupled GL(K) gauge basis (off-block head
generators) was already implemented and verified green in the geometry layer — `groups.py`'s
`_build_block_glk` accepts a `cross_couplings` kwarg that calls `generate_glk_cross_head`
(`generators.py`) and optionally closes the basis under the Lie bracket (`closure.py`) — but it was
unreachable from `VFE3Config`: there was no `cross_couplings` field, and `build_group` dispatched
purely on the builder's positional arity, so it could never forward a kwarg. This change is the
config WIRING only; no geometry was touched.

Files:
- `vfe3/config.py`: new field `cross_couplings: Optional[List[Tuple[int, int]]] = None` beside the
  gauge-seam fields (imports `List`, `Tuple` added). `__post_init__` validates (after the
  `gauge_group` `_require`, so the group and `n_heads` are resolved): when not None it must be a
  list of distinct in-range directed `(int, int)` head pairs with `a != b` and each index in
  `[0, n_heads)`, and the selected `gauge_group`'s builder must accept the kwarg. Support is checked
  by `inspect.signature` of the registered builder — NOT a hardcoded group-name list — so only
  `block_glk` qualifies (its is the only builder with the param); `glk`, `so_k`, and
  `tied_block_glk` are rejected when cross_couplings is set. (NOTE: the roadmap brief's parenthetical
  "block_glk / tied_block_glk" is stale — the actual `_build_tied_block_glk` builder does NOT accept
  `cross_couplings`; per CLAUDE.md CODE-FOCUS the signature is the source of truth, so tied is
  rejected.) Default None reproduces current behavior exactly.
- `vfe3/model/model.py::build_group`: the arity dispatch is widened to build a `kwargs` dict that
  carries `cross_couplings` only when `cfg.cross_couplings is not None` AND the builder's signature
  has the parameter, then splats it into the existing arity-1/arity-2 calls. None -> empty kwargs ->
  the SAME group object as before. The glk/so_k (arity 1) and no-cross-coupling block_glk paths are
  unchanged.

Scope: only `cross_couplings` is exposed; the builder's `close_basis` (bracket closure) stays at its
default `False`, so the config-reachable cross-coupled basis is the un-closed cross-head basis
(block-diagonal + off-block generators), which is already strictly larger than the direct sum
(base `n_heads * d_head^2`, plus `d_head^2` per coupling pair). The bracket-closed subalgebra
(`close_basis=True`) remains builder-only / not config-reachable by design (one new field, no scope
creep). For a cross-coupled group `irrep_dims` is `[K]` (single super-block; the per-block
decomposition is a deferred transport concern), matching the existing geometry code and test.

Default-None byte-identity: `test_build_group_default_none_is_byte_identical` asserts
`torch.equal(build_group(VFE3Config(...block_glk...)).generators, get_group("block_glk")(8,2).generators)`
(and equal `irrep_dims`). The wiring adds nothing when unset.

Tests (TDD, watched RED then GREEN):
  - `tests/test_config.py::test_config_cross_couplings_default_none_and_validated` — default None;
    valid `[(0,1)]` accepted under block_glk; self-coupling `(0,0)` and out-of-range `(0,2)` raise;
    `so_k` and `tied_block_glk` with cross_couplings set raise.
  - `tests/test_gauge_groups.py::test_build_group_default_none_is_byte_identical` — the byte-identity
    invariance guard (GREEN from the start once the field exists).
  - `tests/test_gauge_groups.py::test_build_group_forwards_cross_couplings` — the forwarded kwarg
    grows the basis (32 -> 48 for embed_dim 8 / n_heads 2 / one pair) and reports `irrep_dims=[K]`.
  - `tests/test_model.py::test_model_runs_under_cross_coupled_block_glk` — the end-to-end oracle: a
    tiny `VFEModel` under `cross_couplings=[(0,1)]` runs a forward + `loss.backward()` with finite
    loss and finite, grad-connected gradients on the prior tables (mirrors the verify-first check).

Full suite after the change: `tests=284 failures=0 errors=0 skipped=0` (read from junitxml; 280
baseline + 4 new; viz collected normally; 1 xpassed pre-existing).

### Autoregressive `generate()` (roadmap item)

Branch: vfe3-roadmap-overnight-2026-06-02. The model could only do teacher-forced CE training; it
had no way to produce text. Added an ADDITIVE, training-isolated `VFEModel.generate(token_ids,
max_new_tokens, *, temperature=1.0, top_k=None, top_p=None, greedy=False) -> (B, N0 +
max_new_tokens)` on `vfe3/model/model.py` (the only production file touched). It REUSES the existing
`forward` (encode -> E-step -> decode) rather than reimplementing the belief pipeline: each step
feeds the running sequence -- truncated to the last `cfg.max_seq_len` tokens, since the model and
its attention prior are built for `N <= max_seq_len` -- through `forward(seq)` (`targets=None` ->
logits `(B, N, V)`), reads `logits[:, -1, :]`, turns it into a next token, and appends. The returned
sequence keeps the FULL prompt (including any portion beyond `max_seq_len`) followed by the generated
ids. Decorated `@torch.no_grad()`; because it never calls the training/loss branch it cannot corrupt
training (that isolation is the safety oracle).

Samplers (minimal, no registry): `greedy=True` takes the argmax and returns BEFORE any
temperature/top_k/top_p logic (so those are ignored under greedy). Otherwise logits are divided by
`temperature`, then `top_k` (keep the k largest, `-inf` the rest via the k-th-largest threshold),
then `top_p` (nucleus: `-inf` every token for which the strictly-preceding sorted cumulative softmax
mass already reaches `p`, which always keeps the top token, then scatter the sorted mask back to
vocab order), then softmax + `torch.multinomial`. Cost note: this is the correct-but-slow first
version -- it re-runs the FULL forward (encode -> E-step -> decode) for every generated token;
incremental belief reuse across steps is a future optimization, documented in the docstring.

Tests (`tests/test_generate.py`, TDD oracle-first, watched RED -- 9 `AttributeError: no attribute
'generate'` -- then GREEN): `test_shape_in_vocab_and_prompt_preserved` (shape `(B, N0+5)`, all ids in
`[0, V)`, prompt columns preserved), `test_greedy_is_deterministic` (two greedy calls equal),
`test_greedy_equals_forward_argmax_first_token` (the pin: first greedy token == `argmax` of
`forward(prompt)[:, -1, :]`; first token only, since step 2+ conditions on a longer sequence),
`test_greedy_ignores_temperature_topk_topp` (wild temperature + aggressive top_k/top_p alongside
`greedy=True` change nothing -- pins the branch ordering), `test_top_k_one_is_argmax_first_token`
(`top_k=1` not-greedy is deterministic and equals argmax), `test_top_k_membership_first_token` (the
first sampled token lies among the k largest of the last-position logits), `test_top_p_and_
temperature_paths_run_in_vocab` (both paths run, stay in-vocab), `test_prompt_longer_than_max_seq_len_
does_not_error` (a prompt longer than `max_seq_len` does not error; full prompt preserved in the
return), and `test_generate_is_training_isolated` (the safety oracle: `mu_embed` unchanged before/after
a generate call, and the training forward still returns a finite loss afterward).

Full suite after the change: `tests=293 failures=0 errors=0 skipped=0` (read from junitxml; 284
baseline + 9 new; viz collected normally; 1 xpassed pre-existing). `generate` is purely additive: it
changed no existing test.

### register_transport seam over the gauge transport (roadmap item)

Branch: vfe3-roadmap-overnight-2026-06-02. Design spec for the deferred non-flat builder:
`docs/superpowers/specs/2026-06-01-regime-ii-connection-design.md`. The clean-room spec (sec 4.2)
names the connection REGIME as a registry-backed modular axis "on equal footing with the structure
group ... config-selected, added by writing-and-registering, never editing call sites". The
structure-group axis already IS a registry (`register_group`/`get_group`); the transport/connection
axis was NOT — `vfe3/inference/e_step.py` imported and called `compute_transport_operators` directly.
This is a BYTE-IDENTITY refactor adding the missing seam with the current flat phi-cocycle as the
default registered entry. It builds ONLY the seam + flat default; the non-flat Regime II builder is
deferred to the design spec for the user to decide (NOT built here).

Orthogonality: `transport_mode` is the connection-REGIME axis (is the connection flat at all),
ORTHOGONAL to the pre-existing `gauge_parameterization` (phi | omega_direct), which only chooses how
a single flat transport is parameterized. The two are distinct seams; the field comment in
`config.py` and the registry docstring in `transport.py` state this.

Files:
- `vfe3/geometry/transport.py`: new `_TRANSPORTS` registry, `register_transport(name)` decorator,
  `get_transport(name)` (KeyError-with-available-list on miss), mirroring `register_group`/`get_group`
  and `register_retraction`/`get_retraction`. The flat phi-cocycle is registered under `"flat"` as a
  thin adapter `_build_flat(phi, group, *, gauge_mode="learned", **kwargs)` that forwards verbatim to
  `compute_transport_operators(phi, group, gauge_mode=gauge_mode)` and TOLERATES extra keyword args (so
  a future stateful non-flat builder shares the call shape). `compute_transport_operators` and
  `compute_transport_operators_direct` are untouched. Regime II is NOT registered.
- `vfe3/config.py`: new field `transport_mode: str = "flat"` beside the gauge-seam fields, validated
  in `__post_init__` against the transport REGISTRY (`tuple(sorted(_TRANSPORTS))`, the
  `divergence_families()` / `_RETRACTIONS` pattern) so a future registered regime is selectable
  without a config edit. Local import avoids a config <- transport <- groups cycle. Default keeps the
  flat pure path.
- `vfe3/inference/e_step.py`: the PRIMARY E-step belief-transport build in `e_step_iteration` routes
  through the registry — `_transport` gains a `*, transport_mode="flat"` kwarg and swaps its internal
  `compute_transport_operators` for `get_transport(transport_mode)(...)` (the 2-D/3-D rank logic stays
  in one place, so the 2-D diagnostics/`free_energy_value` callers keep the `"flat"` default
  byte-identical with no edit). `e_step_iteration` accepts `transport_mode` and passes it at the
  build; `free_energy_value` declares `transport_mode` as an explicit accept-and-ignore knob (the
  `e_step` call site forwards one kwarg bag to both `e_step_iteration` AND `free_energy_value` via the
  `_f_diag` trajectory path, so a missing declaration would TypeError on `return_trajectory=True`).
  Import widened to add `get_transport`.
- `vfe3/model/block.py`: threads `transport_mode=cfg.transport_mode` into the `e_step` call, beside
  `spd_retract_mode=cfg.spd_retract_mode`.

Intrinsically-flat helpers left on the direct call (byte-identity preserved, conscious skips): the
mixed-frame `_transport_qk` (FILTERED objective; cannot share the single-phi flat builder),
`phi_alignment_loss` (phi-objective, `e_phi_lr=0.0` by default; threading it is not required for this
task), and `model.py::diagnostics` (`_transport(out.phi, ...)` at the `"flat"` default). These keep
calling `compute_transport_operators` / the defaulted `_transport`, unchanged.

Byte-identity gate: `tests/test_transport.py` adds `test_transport_registry_round_trip` (register/get
round-trip + unknown-name KeyError, registration cleaned up in `finally`), `test_flat_is_registered`,
`test_flat_builder_bit_identical_to_direct_call` (the ORACLE — `torch.equal` on Omega / exp_phi /
exp_neg_phi between `get_transport("flat")(phi, group)` and the bare
`compute_transport_operators(phi, group)` on a fixed-seed phi), and
`test_flat_builder_tolerates_extra_kwargs`. `tests/test_config.py` adds
`test_config_transport_mode_validated`. `tests/test_e_step.py` adds
`test_transport_flat_kwarg_is_byte_identical_to_default` (the routed `_transport` default equals the
explicit `"flat"` on both the 2-D and 3-D paths, atol=0) and
`test_e_step_iteration_transport_flat_default_is_byte_identical` (the default-routed iteration's
mu/sigma/phi equal the run with no `transport_mode`, atol=0). The wired-forward gate is
`tests/test_perf_equivalence.py` passing in the full green suite. Full suite
`tests=300 failures=0 errors=0 skipped=0` (read from junitxml; 293 baseline + 7 new; viz collected
normally; 1 xpassed pre-existing).

### squared-Hellinger f-divergence functional (second registry member; roadmap item 11)

Branch: vfe3-roadmap-overnight-2026-06-02. Design spec:
`docs/superpowers/specs/2026-06-01-f-divergence-functional-design.md`. The functional axis of the
divergence seam (`register_functional`/`get_functional`/`_FUNCTIONALS` in `families/base.py`) was a
genuine registry carrying exactly one member (`renyi`), so the de-facto interface was the single
alpha-parameterized `renyi(...)` signature. This adds squared Hellinger — the first non-Renyi
f-divergence — demonstrating the seam, and generalizes the functional contract so a member can
ignore params it does not use, with ZERO call-site edits.

Identity used (spec sympy-VERIFIED, exact diff=0, re-verified this session): for Gaussians the
Bhattacharyya coefficient is `BC = exp(-D_{1/2}(q||p)/2)` where `D_{1/2}` is the Renyi-1/2 divergence
the pinned `renyi` kernel already computes, so `H^2(q||p) = 1 - exp(-D_{1/2}(q||p)/2)`. Hellinger is
thus a thin wrapper over machinery already golden-pinned — no new family-specific Cholesky/blend math.

Files:
- `vfe3/families/base.py`: `renyi` gains a trailing `**kwargs` (additive, harmless — the permissive
  functional contract; `renyi` still consumes `alpha`). New `squared_hellinger(q, p, *, kl_max=100.0,
  eps=1e-6, **kwargs)`: absorbs any `alpha` the call sites forward (Hellinger has no order — never
  reaches `renyi`, so the alpha>1 blend warning cannot fire), forwards `kl_max` so the inner `D_{1/2}`
  stays bounded in `[0, kl_max]`, and returns `1.0 - torch.exp(-0.5 * renyi(q, p, alpha=0.5, ...))`.
  NO output `.clamp` — `1 - exp(-D/2)` with `D in [0, kl_max]` is provably in `[0, 1)` (a clamped
  `D=kl_max` maps to the maximal-Hellinger limit `1 - exp(-kl_max/2)`, which composes correctly), so a
  second clamp would be dead. Registered `register_functional("squared_hellinger")`. New
  `divergence_functionals() -> tuple(sorted(_FUNCTIONALS))` helper (mirrors `divergence_families()`).
- `vfe3/divergence.py`: re-export `squared_hellinger` and `divergence_functionals` (import + `__all__`).
- `vfe3/config.py`: `divergence_family` validation is now registry-derived — a local
  `from vfe3.divergence import divergence_functionals` in the divergence-seam block validates against
  `divergence_functionals()` (the `divergence_families()` pattern), so a new functional is
  config-selectable WITHOUT editing config. The hardcoded `_VALID_DIVERGENCE_FUNCTIONALS = ("renyi",)`
  tuple was removed (grep-confirmed dead — its only use was the one `_require` call). Field comment
  notes `alpha_div` is ignored by non-alpha functionals.

`reverse_kl` (a spec-proposed regression rung) was deliberately NOT registered — the task asks only
for `squared_hellinger`, and the registry-derived validation makes the valid set exactly
`{renyi, squared_hellinger}`, which is correct (no scope creep).

Independent oracles (test_divergence.py, TDD watched RED — 13 failing for unregistered functional /
missing helper / config rejection — then GREEN; the oracles do NOT re-assert the definition):
- `test_squared_hellinger_diagonal_matches_analytic`: vs the analytic diagonal Gaussian H^2 computed
  in float64 as `1 - prod_k BC_k`, `BC_k = sqrt(2 sqrt(s_q s_p)/(s_q+s_p)) exp(-(mu_q-mu_p)^2/
  (4(s_q+s_p)))` — Bhattacharyya factorizes over coordinates, so BC is a PRODUCT then 1-BC (NOT a
  per-coord H^2 summed). atol 1e-5.
- `test_squared_hellinger_full_matches_analytic`: vs the analytic full-covariance Bhattacharyya
  distance `D_B = 1/8 dmu^T Sbar^{-1} dmu + 1/2(ln|Sbar| - 1/2 ln|S_q| - 1/2 ln|S_p|)`, `Sbar =
  (S_q+S_p)/2`, via slogdet/solve — a DIFFERENT numerical path than the kernel's Cholesky-of-blend
  (independently sympy/numeric-checked to ~1e-7 this session). atol 1e-5.
- `test_squared_hellinger_equals_definitional_identity`: pins `H^2 = 1 - exp(-D_{1/2}/2)`.
- `test_squared_hellinger_is_symmetric` (diagonal + full): the STRONG independent check — Hellinger is
  symmetric, unlike KL / Renyi at alpha != 1/2; `H^2(q,p) allclose H^2(p,q)` atol 1e-5.
- `test_squared_hellinger_self_is_zero`, `test_squared_hellinger_bounded` (both families):
  `H^2(q,q) = 0`, `0 <= H^2 <= 1`.
- `test_squared_hellinger_ignores_alpha_and_does_not_warn`: `fn(q,p)` allclose `fn(q,p,alpha=2.0)`
  (atol 0) AND no RuntimeWarning fires — proves `alpha=2.0` never reaches `renyi`'s alpha>1 branch.
- `test_divergence_functionals_registry_derived`, `test_config_accepts_squared_hellinger_and_rejects_
  unknown`: registry exposes both members and config accepts `squared_hellinger`/`renyi`, rejects an
  unknown name.
- `test_model_forward_under_squared_hellinger`: end-to-end VFEModel forward + finite loss with the new
  functional flowing through `pairwise_energy`/`self_divergence`.

Renyi/default path byte-identical (the suite is the gate). Full suite after the change:
`tests=313 failures=0 errors=0 skipped=0` (read from junitxml; 300 baseline + 13 new; viz collected
normally; 1 xpassed pre-existing). Note: squared_hellinger trains through the autograd oracle
(`kernels.py` guards renyi-only for the hand kernel) and refuses the per-coord alpha path
(`self_divergence_per_coord` guards renyi-only) — both automatic and correct; the per-coord refusal
is the documented known incompatibility, and the forward+finite-loss smoke test covers the oracle
training path.

## 2026-06-02 — Log-Euclidean SPD retraction variant (spec 2a; 2b deferred)

Branch: vfe3-roadmap-overnight-2026-06-02 (committed directly).
Design spec: docs/superpowers/specs/2026-06-01-spd-retraction-variants-design.md, PHASE 1 = reading 2a
(the pure log-Euclidean retraction) ONLY. 2b (the Frechet / Daleckii-Krein natural-gradient kernel)
remains a deferred sub-flag per the spec — NOT built.

### Files
- `vfe3/geometry/retraction.py`: new bare `retract_logeuclidean_full` (two-eigh logm/expm) + registered
  `@register_retraction("log_euclidean")` `retract_log_euclidean(sigma, delta_sigma, mean_ndim, *,
  step_size, trust_region, eps, sigma_max)` — the SAME signature as `retract_spd_affine`, so the E-step
  dispatch stays uniform.
- `vfe3/config.py`: `spd_retract_mode="log_euclidean"` already validates against the retraction registry
  (registration alone suffices). Added a config-time `UserWarning` (not error) when
  `spd_retract_mode=="log_euclidean"` is paired with a diagonal family.
- `tests/test_retraction.py`: 8 new tests (below).

### The LE formula (spec reading 2a)
    Sigma_new = expm( logm(Sigma) + step_size * sym(delta_sigma) ).
logm/expm via `torch.linalg.eigh` (logm(Sigma) = V diag(log lambda_j) V^T; expm(M) = U diag(exp mu_j)
U^T), the same two-eigh structure and fp32-island as `retract_spd_full`. Input eigenvalues floored at
`eps` before log; output spectrum projected to [eps, sigma_max^2]. SPD-preserving for ANY step (expm of
a symmetric matrix is SPD); the trust region is a stability knob, not a positivity guard. The trust
region clamps the TANGENT term only (`logm(Sigma) + frobenius_clamp(step*delta)`), NOT the base point,
so the retraction axiom R(Sigma, 0) = Sigma holds (matching the affine path, which clamps the whitened
tangent). Diagonal reduction: `sigma_new = sigma * exp(step_size * delta_sigma)`.

### VERIFIED diagonal relationship (spec contradiction — DONE_WITH_CONCERNS)
The spec (sec 2, DECISION 5) claims LE EQUALS affine on the diagonal. Verified FALSE under THIS seam's
tangent convention. Phase 0 kept the Fisher metric conversion in the E-step (`natural_gradient`,
e_step.py:212), so the seam receives an ALREADY-preconditioned `delta_sigma`. The affine diagonal
retraction (`retract_spd_diagonal`) then whitens that tangent by 1/sigma (`whitened = delta/sigma`,
retraction.py:57), giving `sigma exp(step*delta/sigma)`; LE does NOT whiten, giving `sigma
exp(step*delta)`. Equal only at sigma = I. The spec's equality holds for the 2b log-chart NATURAL
gradient, not for 2a under a pre-whitened tangent. Consequently the config WARN is worded TRUTHFULLY:
on a diagonal family LE is a non-canonical log-chart step (lacks the affine Fisher whitening) and does
NOT reduce to spd_affine — prefer spd_affine on the diagonal family, or use gaussian_full. LE is a
genuinely new variant only for full covariance (logm != elementwise log). The
`test_log_euclidean_diagonal_differs_from_affine` test pins this finding; the spec's proposed
`equals_affine` test was NOT written (it would assert a falsehood).

### Oracles
- SPD-preservation (core): `test_log_euclidean_stays_spd_unconditionally` — random full-cov SPD Sigma +
  symmetric tangent, no trust region, symmetric PD output across step sizes; guards that neither the eps
  floor nor the sigma_max cap engaged (the genuine expm map, not a clamp), plus an ill-conditioned-base
  contrast where a naive Euclidean step leaves the cone but LE stays PD.
- Independent matrix reference: `test_log_euclidean_full_matches_independent_expm_logm` — equals an
  INDEPENDENTLY computed `expm(logm(Sigma) + step*sym(delta))` via `torch.linalg.matrix_exp` + an
  eigh-based logm written in the test (distinct code path), well-conditioned Sigma + modest step so no
  clamp binds, atol 1e-5.
- `test_log_euclidean_identity_tangent_is_identity` — R(Sigma, 0) = Sigma at the operational
  trust_region=5.0 (pins the tangent-only clamp fix).
- `test_log_euclidean_diagonal_is_log_chart_step` — diagonal `sigma*exp(step*delta)`.
- `test_log_euclidean_diagonal_differs_from_affine` — the verified scope finding above.
- `test_log_euclidean_registered_and_config_accepts`, `test_log_euclidean_diagonal_pairing_warns` —
  registry + config validation + the diagonal-pairing WARN (pytest.warns).
- `test_log_euclidean_e_step_full_cov_runs` — full-covariance E-step forward+backward under
  spd_retract_mode='log_euclidean': finite SPD covariance, finite grads.

### Default path + suite
Default `spd_affine` path byte-identical (the two `test_spd_affine_bit_identical_*` use torch.equal /
atol=0 and pass; additive variant, no edits to affine code). Full suite after the change:
`tests=321 failures=0 errors=0 skipped=0` (read from junitxml; 313 baseline + 8 new; 320 passed +
1 xpassed pre-existing). 2b deferred.

## 2026-06-02 — Extensible BeliefState (roadmap M3)

Same running log (per the project one-doc-per-day convention; appended here on the M3 task's
explicit instruction). Branch: vfe3-roadmap-overnight-2026-06-02.

### Motivation
M3 (modularity architecture): make `BeliefState` carry optional extra per-token channels (the
future hyper-prior `s_i`/`r_i`, natural params, etc.) WITHOUT a signature sweep, a precondition
for the hyper-prior/model-coupling work — while keeping the 3-field default byte-identical.

### Form chosen and why
Kept `BeliefState` a `typing.NamedTuple` and ADDED two trailing optional fields with `None`
defaults: `s: Optional[torch.Tensor] = None`, `r: Optional[torch.Tensor] = None`. A
codebase-wide audit (`vfe3/` + `tests/`) found every construction uses keyword arguments
(`BeliefState(mu=, sigma=, phi=)`) and every read uses attribute access (`.mu/.sigma/.phi`); NO
site relies on a NamedTuple-only behavior that trailing defaulted fields would break — no 3-way
positional unpack of a belief (a naming-agnostic `^\s*\w+,\s*\w+,\s*\w+\s*=` grep confirmed no
BeliefState ever sits on an unpack RHS), no indexing, no iteration, no `_replace`/`_asdict`. This
is the lowest-surface extensible form (the dataclass conversion was unnecessary).

### Behaviors preserved
Keyword and positional construction, `.mu/.sigma/.phi` attribute access, `_replace`, indexing,
and iteration all unchanged. The two new fields default to `None`, so nothing reads them and there
is no numeric path through them — byte-identity at the model level is the full green suite.

### New capability
`BeliefState(mu, sigma, phi, s=t)` round-trips (`.s is t`); a default-constructed belief has
`.s is None` and `.r is None`. A second belief channel can now be threaded without editing every
signature that passes a belief.

### Tests
New `tests/test_belief.py` (6 tests): `test_three_field_construction_and_attribute_access`,
`test_optional_channels_default_to_none`, `test_positional_construction_still_works`,
`test_replace_preserves_namedtuple_semantics`, `test_extra_channel_round_trips`,
`test_both_extra_channels_round_trip`. RED first (4 of 6 failed before the field add), then GREEN.
Full suite after the change: `tests=327 failures=0 errors=0 skipped=0` (read from junitxml; 321
baseline + 6 new; 326 passed + 1 xpassed pre-existing).

## 2026-06-02 — Batched per-head `pairwise_energy` over equal irrep blocks (bit-identical)

Branch: vfe3-roadmap-overnight-2026-06-02. A perf refactor of `pairwise_energy`'s per-irrep-block
loop; no formula change.

### What changed
`vfe3/free_energy.py`: when the irrep blocks are EQUAL size and more than one (the default
`block_glk` case), the H per-block divergences -- the same functional over H disjoint coordinate
slices -- are now computed in ONE functional call instead of a Python `for`-loop of per-block calls.
The H equal blocks of the broadcast query and the transported key are sliced and stacked along a NEW
LEADING axis (`type(q_b).stack(...)`, `type(key).stack(...)`), the functional is invoked once over
that stacked head axis producing `(H, ..., N, N)`, and `torch.movedim(e, 0, -3)` moves the head axis
to position `-3` to match the loop's `torch.stack(energies, dim=-3)` layout `(..., H, N, N)` exactly.
The unequal-block path and the single-block/`None` path keep the existing loop / direct call.

`vfe3/families/base.py`, `vfe3/families/gaussian.py`: a family-agnostic batching primitive
`BeliefParams.stack(parts, *, dim=0)` -- a concrete classmethod that raises `NotImplementedError`
(mirroring `expected_statistic`, so the toy subclasses in `test_families.py` still instantiate), with
`DiagonalGaussian`/`FullGaussian` overrides that `torch.stack` the underlying `mu`/`sigma` tensors.

### The bit-identical oracle and the unequal-block fallback
Stacking plus one functional call is the SAME arithmetic in a different layout (every op in the
closed forms is elementwise / last-axis-sum / last-two-axis Cholesky/solve/diagonal, no cross-head
interaction; `movedim` is a pure permutation), so the batched output is `torch.equal` (atol=0) to the
explicit per-block loop for both `DiagonalGaussian` and `FullGaussian`. The new tests recompute the
loop reference in-test and assert `torch.equal`. The default `block_glk` (equal heads) now takes the
batched path and the frozen-oracle / model / e_step / per-head-attention tests stay bit-for-bit green
-- the full suite is the real bit-identity gate.

The equal-block branch is guarded (`_stackable_for_batching`) so it fires ONLY when stacking does not
perturb the mu/sigma broadcast: the misclassified case where `sigma` carries a leading batch dim `mu`
lacks (Gaussian canonical rank is sigma == mu for diagonal, mu+1 for full) would right-align the new
head axis against sigma's first batch dim and broadcast spuriously, so it falls back to the loop. A
family that does not expose mu/sigma tensors also falls back. Unequal block sizes fall back to the
loop. Every fallback is bit-identical to the loop because it IS the loop.

### Tests (8 new)
`tests/test_families.py`: `test_diagonal_stack_round_trips`, `test_full_stack_round_trips` (the genuine
RED-first pair -- `stack` missing -> `AttributeError` before the override, GREEN after).
`tests/test_free_energy.py`: `test_pairwise_energy_equal_blocks_batched_is_bit_identical_to_loop_diagonal`,
`..._full`, `..._with_leading_batch_dim` (the `(B,N,K)` training layout -> `(B,H,N,N)`),
`test_pairwise_energy_unequal_blocks_fall_back_to_loop`,
`test_pairwise_energy_single_block_and_none_unchanged`,
`test_pairwise_energy_equal_blocks_mismatched_sigma_rank_falls_back_to_loop` (pins the guard). These
six `pairwise_energy` tests are characterization/guard tests -- green before AND after, since the
pre-refactor `pairwise_energy` IS the loop -- recomputing the loop in-test and asserting `torch.equal`
against the (now batched) implementation. The pre-existing `test_pairwise_energy_per_head_splits_by_
irrep_block` (`irrep_dims=[2,2]`, equal) now routes through the batched path as a free extra guard.

Full suite after the change: `tests=335 failures=0 errors=0 skipped=0` (read from junitxml; 327
baseline + 8 new; 334 passed + 1 xpassed pre-existing; viz collected normally).

## 2026-06-02 — Learnable self-coupling alpha (opt-in nn.Parameter; sanctioned NN exception)

Branch: vfe3-roadmap-overnight-2026-06-02. Design spec:
`docs/superpowers/specs/2026-06-01-learnable-alpha-design.md` (Path a, the LEARNABLE form; the
fully-Bayesian b2 variant is deferred). The user SANCTIONED a learnable self-coupling alpha as a
third documented neural-network exception (alongside `use_prior_bank=False`'s linear decode and
`use_head_mixer`), ON THE CONDITION that an NN comment sits AT THE FUNCTION and AT THE CONFIG
TOGGLE. Both are present.

### The learnable form
A single learnable SCALAR alpha (not per-coord/block in this first version). The consumed
coupling is `alpha = exp(log_alpha)`, where `log_alpha` is a model-owned `nn.Parameter`; init
`log_alpha = 0` gives `alpha = exp(0) = 1.0`, exactly the `constant alpha=1.0` default at step 0,
and `exp` keeps alpha strictly positive for any real `log_alpha`. Because `alpha` is now a FREE
parameter (not a Gamma precision posterior summary), there is NO regularizer: the form returns
`(exp(log_alpha) * ones_like(kl), zeros_like(kl))`, so F carries the plain self-term `alpha*D` and
the belief gradient is the plain `alpha*dD` (the `constant` form's contract). The
alpha-envelope cancellation that `state_dependent` relies on (an explicit R(alpha) whose
product-rule path cancels at the stationary alpha*) does NOT apply and is NOT added.

### The NN-exception comments (where)
- `vfe3/alpha_i.py` — `alpha_learnable` docstring opens "NEURAL-NETWORK EXCEPTION (sanctioned,
  default-off): a LEARNED scalar self-coupling alpha = exp(log_alpha) ... model-owned
  nn.Parameter trained by backprop (cf. use_head_mixer / use_prior_bank)", and states the default
  no-NN forms are unchanged.
- `vfe3/config.py` — comment AT the `alpha_mode` field: the default-and-pure no-NN forms are
  `constant`/`state_dependent`/`state_dependent_per_coord` and are unchanged; "NEURAL-NETWORK
  EXCEPTION: 'learnable' introduces a model-owned scalar nn.Parameter log_alpha (alpha =
  exp(log_alpha)) trained by backprop -- a sanctioned, default-OFF learned-parameter exception".
- `vfe3/model/model.py` — comment AT the `self.log_alpha` parameter definition, same wording;
  notes that for every other (pure) alpha_mode the parameter is NOT created at all.

### Files / threading path
- `vfe3/alpha_i.py`: new `@register_alpha("learnable")` form `alpha_learnable(kl, *,
  log_alpha=None, **kwargs)`. `alpha_gradient_coefficient` gains a `log_alpha` kwarg and forwards
  it into `self_coupling_alpha` (the kernel's envelope coefficient is `exp(log_alpha)` since R=0).
- `vfe3/config.py`: `"learnable"` added to `_VALID_ALPHA_MODES` (the hardcoded validated set; the
  `alpha_is_per_coord("learnable")` guard is False, so the per-coord/diagonal-family check is
  inert). No new config field — one new value of the existing `alpha_mode` knob.
- `vfe3/model/model.py`: `VFEModel.__init__` creates `self.log_alpha = nn.Parameter(torch.zeros(()))`
  ONLY when `cfg.alpha_mode == "learnable"` (no attribute otherwise — param-free pure path).
  `forward` reads `getattr(self, "log_alpha", None)` and threads it through `vfe_stack`; the
  M-step self-coupling block uses only `self_divergence_for_alpha` (a fixed-weight term, no alpha
  consumption) so it is untouched; `diagnostics` passes `log_alpha` into its `vfe_stack` and
  `self_coupling_alpha` calls.
- Threading (one extra defaulted-None keyword, uniform): `VFEModel.forward` -> `vfe_stack(...,
  log_alpha=)` -> `vfe_block(..., log_alpha=)` -> `e_step(..., log_alpha=)` (flows through `e_step`'s
  `**kwargs` to BOTH `e_step_iteration` and `free_energy_value` via `_f_diag`) ->
  `e_step_iteration(..., log_alpha=)` -> `belief_gradients(..., log_alpha=)` -> the hand kernel via
  `alpha_gradient_coefficient(..., log_alpha=)` OR `belief_gradients_autograd(..., log_alpha=)` ->
  `self_coupling_alpha(..., log_alpha=)`. `free_energy_value` consumes `log_alpha` in its
  `self_coupling_alpha` call (it is part of F, not an iteration-only accept-and-ignore knob).
  When `None`/not-learnable, every signature behaves exactly as before; the default
  `self_coupling_alpha(..., log_alpha=None)` lands in each pure form's `**kwargs` and is ignored.

### Grad-flow confirmation + a known scope limit
On the DEFAULT kernel path (renyi + gaussian_diagonal + KL alpha_div=1 + filtering + canonical) the
self-coupling coefficient `exp(log_alpha)` is grad-connected through the analytic kernel into the
updated belief and thence the unrolled-E-step loss, so `loss.backward()` populates
`model.log_alpha.grad` (verified: finite, non-None, nonzero). NOTE: `belief_gradients_autograd`
detaches `grad_mu/grad_sigma`, so on the OVERSAMPLE oracle fallback (smoothing, non-KL functional,
Renyi alpha != 1, non-diagonal family) the learned alpha would not receive an E-step gradient
(only an F/loss-path gradient if present) — out of scope for this scalar first version, which
targets the default kernel path.

### Default-off + init==constant-1.0 oracles
- `test_default_off_no_log_alpha_attribute`: a `constant` (default) and a `state_dependent` model
  have NO `log_alpha` attribute (`not hasattr`) — the pure path is param-free.
- `test_learnable_init_equals_constant_one` (the independent oracle): a `learnable` model at init
  (`log_alpha=0`) produces `logits` and `loss` `torch.equal` (byte-identical) to the SAME
  seed/config with `alpha_mode="constant", alpha=1.0`. This pins learnable-at-init == the
  constant-1.0 pure path.
- `test_learnable_log_alpha_grad_populated`: `log_alpha.grad` finite, non-None, nonzero after
  `forward + backward`.
- `test_learnable_alpha_changes_forward_when_log_alpha_moves`: moving `log_alpha` 0 -> log(5)
  changes the loss (the alpha is genuinely consumed, not dead).
- Form-level (`tests/test_alpha_i.py`): `test_learnable_alpha_is_exp_log_alpha_zero_reg`
  (alpha == 1.0 at log_alpha=0, == 2.0 at log(2), zero reg) and
  `test_learnable_alpha_gradient_flows_to_log_alpha` (grad reaches `log_alpha`).
- Also `test_config_accepts_learnable_alpha_mode`, `test_learnable_creates_scalar_log_alpha_param`,
  `test_learnable_diagnostics_runs`.

### Default byte-identity + suite
The default (non-learnable) paths are byte-identical: every new keyword defaults to None and is
ignored by the pure forms; the frozen-seed regression / byte-identity tests
(`test_e_step.py`) stay green. New tests: 9 (7 in `tests/test_learnable_alpha.py`, 2 in
`tests/test_alpha_i.py`), TDD watched RED (8 failing for unregistered form / missing param) then
GREEN. Full suite after the change: `tests=344 failures=0 errors=0 skipped=0` (read from junitxml;
335 baseline + 9 new; 343 passed + 1 xpassed pre-existing; viz collected normally).
