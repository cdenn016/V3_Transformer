# 2026-06-03 — Deep-audit fixes

Follow-up pass applying the fixes from the deep multi-agent audit
(`docs/audits/audit-2026-06-03.md`, which also lists the full surviving punch list and the deferred
items). All fixes done test-first (failing test → fix → green), implemented directly on shared files.
Suite: **497 tests, 0 failures, 0 errors, 1 skipped** (was 490 / 10-failed). `verified.md` updated with
the three corrected math/theory invariants.

## Source changes

- `vfe3/geometry/lie_ops.py` — `project_phi_to_slk` / `clamp_phi_trace`: per-block `1/||V_h||^2`
  projection → joint Gram solve `pinv(V V^T)`. Fixes the n_heads-fold over-subtraction of the per-block
  trace under `tied_block_glk` (det Omega → 1); untied `block_glk` unchanged within golden tolerance
  (atol 1e-5 — `pinv` is SVD-based, not bit-exact). Orphaned `eps` kwarg removed.
- `vfe3/geometry/retraction.py` — `retract_spd_full` / `retract_logeuclidean_full`: full-cov eigenvalue
  ceiling `sigma_max*sigma_max` → `sigma_max` (one variance convention across the diagonal/full seam).
  **BEHAVIOR CHANGE on the full-cov pure path:** at default `sigma_max=5.0`, full-cov eigenvalues now
  cap at 5.0 instead of 25.0 (~5× tighter); the suite stayed green only because no tested config had
  eigenvalues in (5,25). Raise `sigma_max` if a full-cov run wants the looser ceiling.
- `vfe3/inference/e_step.py` — `phi_alignment_loss` now takes `transport_mode`/`connection_W`/
  `cocycle_relaxation` and builds Omega under the active regime; the phi E-step call site threads them
  (connection_W detached). Fixes the regime_ii phi-step descending the flat objective.
- `vfe3/config.py` — corrected the `pos_phi` "default-off" comment (it is default `"learned"`); added a
  `decode_mode` ⟺ covariance-family rank cross-check (gated on `use_prior_bank`); added a
  `pos_rotation='rope'` + `gauge_group='sp'` structure-group warning; `_require(value)` → `Optional[str]`;
  dropped a stale comment reference to a deleted function. **BEHAVIOR CHANGE:** a full-covariance family
  without `decode_mode="full"` (and `use_prior_bank=True`) now raises at config CONSTRUCTION instead of
  crashing at the first forward — a config dict that relied on constructing such a pairing will now error.
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

## Deferred-fix pass (user opted into all four tiers)

Done in dependency order, each tier committed separately, TDD + golden-verified.

### Tier 1 — cosmetic type-precision + bit-identical micro-perf (no behavior change)
- `vfe3/gradients/kernels.py` — `_beta_to_coordinate`: equal-block path uses `expand/reshape` instead
  of `repeat_interleave` (skips the gather; BIT-identical, pinned by a new perf-equivalence test);
  unequal blocks fall back to the gather.
- `vfe3/model/model.py` — `_attention_log_prior` generalized to take a `prior` name (cache key now
  `(name, N, device, dtype)`); the gamma block reuses it instead of rebuilding `gamma_log_prior` every
  forward. Byte-identical; redundant local import dropped.
- `vfe3/families/base.py` — parameterized the bare `Callable` registry returns (`register_family`,
  `register_functional`, `get_functional`, `_FUNCTIONALS`); annotated the Bregman accumulator
  `inner: 'torch.Tensor | float'` (runtime unchanged).
- `vfe3/model/block.py`, `vfe3/model/stack.py` — `block_norm: Optional[Any]` → `Optional[Callable[..., torch.Tensor]]`.
- `vfe3/model/prior_bank.py` — `encode_s` return type `'tuple[...]'` (string) → `Tuple[torch.Tensor, torch.Tensor]`.

### Tier 2 — behavior/numerics changes
- `vfe3/free_energy.py` — new `attention_tau(kappa, irrep_dims) = kappa*sqrt(irrep_dims[0])`: the
  softmax temperature now keys off the dimension the energy accumulates over (the irrep BLOCK):
  `sqrt(K)` for single-block groups (glk/so_k/sp), `sqrt(d_head)` for per-head multi-block (block_glk).
  Wired into `block.py` (E-step) and `model.py` (diagnostics/decode); the train banner prints it.
  **BEHAVIOR CHANGE on single-block groups only:** glk/so_k/sp (n_heads>1) tau goes from
  `kappa*sqrt(K/n_heads)` to `kappa*sqrt(K)` (a `sqrt(n_heads)` softer softmax, the manuscript
  `tau=kappa*sqrt(K)` convention). The DEFAULT block_glk is UNCHANGED (`irrep_dims[0]=d_head`), so the
  frozen oracle and all multi-block golden pins are bit-identical. `cfg.tau` kept as a logging
  convenience with an updated docstring.
- `vfe3/model/model.py` — `pos_phi_free` is now initialized from a DEDICATED `cfg.seed` generator
  instead of the global RNG stream, so its init is independent of whether the conditional model-channel
  s/r tables were drawn. **BEHAVIOR CHANGE on the pos_phi="learned" path:** pos_phi_free init values
  change (mu/sigma/phi inits are unchanged; the global stream is untouched, so pos_phi="none" and the
  frozen oracle are bit-identical). This makes the s-channel byte-identity oracles hold under the
  learned default, so the copy-equivalence (test_prior_source) and hyperprior-linearity tests were
  UN-PINNED from `pos_phi="none"`. The gamma-linearity and learnable-alpha tests stay on `pos_phi="none"`
  for unrelated reasons (the gamma oracle assumes no positional composition; the learnable-alpha signal
  is damped below the 1e-6 floor by pos_phi magnitude) — neither is an RNG issue.

## NOT touched

`ablation.py` and `train_vfe3.py` had concurrent uncommitted edits to their click-to-run config dicts
(not part of this audit) — left untouched and excluded from every audit/deferred commit.
