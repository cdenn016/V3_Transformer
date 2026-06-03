# 2026-06-03 — Deep-audit fixes

Follow-up pass applying the fixes from the deep multi-agent audit
(`docs/audits/audit-2026-06-03.md`, which also lists the full surviving punch list and the deferred
items). All fixes done test-first (failing test → fix → green), implemented directly on shared files.
Suite: **497 tests, 0 failures, 0 errors, 1 skipped** (was 490 / 10-failed). `verified.md` updated with
the three corrected math/theory invariants.

## Source changes

- `vfe3/geometry/lie_ops.py` — `project_phi_to_slk` / `clamp_phi_trace`: per-block `1/||V_h||^2`
  projection → joint Gram solve `pinv(V V^T)`. Fixes the n_heads-fold over-subtraction of the per-block
  trace under `tied_block_glk` (det Omega → 1); byte-identical on untied `block_glk`. Orphaned `eps`
  kwarg removed.
- `vfe3/geometry/retraction.py` — `retract_spd_full` / `retract_logeuclidean_full`: full-cov eigenvalue
  ceiling `sigma_max*sigma_max` → `sigma_max` (one variance convention across the diagonal/full seam).
- `vfe3/inference/e_step.py` — `phi_alignment_loss` now takes `transport_mode`/`connection_W`/
  `cocycle_relaxation` and builds Omega under the active regime; the phi E-step call site threads them
  (connection_W detached). Fixes the regime_ii phi-step descending the flat objective.
- `vfe3/config.py` — corrected the `pos_phi` "default-off" comment (it is default `"learned"`); added a
  `decode_mode` ⟺ covariance-family rank cross-check (gated on `use_prior_bank`); added a
  `pos_rotation='rope'` + `gauge_group='sp'` structure-group warning; `_require(value)` → `Optional[str]`;
  dropped a stale comment reference to a deleted function.
- `vfe3/families/gaussian.py` — `FullGaussian.entropy` raw `cholesky` → `safe_cholesky` (non-PD-safe).
- `vfe3/geometry/transport.py` — clarified the Frobenius-clamp docstring (operator substitution at
  `||M||>max_norm`); **deleted** dead `compute_transport_operators_direct` and `omega_to_block_exp_pairs`.
- `vfe3/free_energy.py` — **deleted** dead `effective_temperature`.
- `vfe3/alpha_i.py`, `vfe3/geometry/rope.py`, `vfe3/gradients/oracle.py` — `Optional`/union type-annotation
  fixes (annotation declared non-Optional but default was `None`; `oracle.omega` typed to its real
  `Tensor|FactoredTransport|RopeTransport` contract).

## Test changes

- New (failing-first) tests: tied-gauge det-control (`test_phi_retraction.py` ×2), `sigma_max` cross-arm
  ceiling (`test_retraction.py`), regime_ii phi-step (`test_regime_ii.py`), `decode_mode`/family guards +
  rope/sp warning (`test_config.py` ×4), non-PD entropy (`test_full_covariance.py`).
- Re-baselined for the `pos_phi="learned"` default: `test_config` (default assertion), `test_train`
  (optimizer groups), and `pos_phi="none"` pins in the s-channel/alpha/frozen-oracle property tests
  whose oracles require no positional composition. `test_viz` umap test skips gracefully on the native
  `OSError`. Removed the two golden-pin tests for the deleted dead functions.

## NOT touched

`ablation.py` and `train_vfe3.py` had concurrent uncommitted edits to their click-to-run config dicts
(not part of this audit) — left untouched and excluded from the audit-fixes commit.
