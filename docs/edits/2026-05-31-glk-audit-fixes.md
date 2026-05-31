# 2026-05-31 — GL(K) audit fixes + deep audit

Branch `glk-audit-fixes-2026-05-31` (fresh from main). Verified and fixed the six findings of
`docs/audits/audit-2026-05-30.md`, ran a fresh multi-agent deep audit, and fixed the confirmed
new issues. Full audit report: `docs/audits/audit-2026-05-31.md`. All work TDD (test written and
seen to fail first). Full suite: 182 (baseline) -> 202 passed, 0 failed.

## Commits

1. `fix(audit): reject dead/trapping toggles, apply seed, wire alpha/phi seams, fix killing_per_block crash`
   - config: reject `gauge_parameterization='omega_direct'`, `use_prior_bank=False`,
     `encode_mode='gauge_fixed'` (live + enforced, not silent/trapping)
   - `torch.manual_seed(cfg.seed)` in `VFEModel.__init__` + `run_training`
   - new `b0`/`c0` config fields threaded into state-dependent alpha
   - new `phi_retract_mode` (euclidean|bch) threaded to `retract_phi`
   - `phi_precond_mode='killing_per_block'`: thread `group.irrep_dims` (was a hard crash)
   - register `free_energy_terms` in the metrics registry

2. `feat(audit): full-covariance (gaussian_full) pure path end-to-end`
   - prior_bank: `diagonal_covariance`-gated full SPD encode + full Cholesky decode
   - e_step: `retract_spd_full` when the covariance is full-rank
   - model: thread `cfg.diagonal_covariance`; config: cross-validate it against `family`

3. `feat(audit): per-head (per-irrep-block) GL(K) attention + divergence-functional seam`
   - `pairwise_energy` per-head via `irrep_dims` (manuscript Algorithm 1); single-block reduces
     bit-identically to the legacy path
   - `divergence_family` is a live divergence-FUNCTIONAL registry (renyi); distinct from `family`
   - hand kernel consumes a per-coordinate beta; oracle per-head via autograd
   - gate: hand kernel == oracle for block_glk multi-head canonical

4. `fix(audit): mass_phi in the phi E-step (#5) + filtered F tracks the current query frame (#6)`
   - `mass_phi` enters `phi_alignment_loss` (E-step penalized objective); M-step term kept
   - `free_energy_value(keys=...)` uses current query phi_i + frozen key phi_j (`_transport_qk`)

## Decisions recorded

- Divergence config is three distinct live seams: `divergence_family` (functional, renyi+alpha),
  `family` (covariance kernel), `diagonal_covariance` (bool, cross-validated). Not collapsed.
- Per-head beta is the default behavior (keyed off `group.irrep_dims`), not an opt-in toggle.
- Temperature kept global `tau = kappa*sqrt(K)` per CLAUDE.md; per-head `sqrt(d_head)` flagged
  in the audit report, not applied.
- Oracle `.detach()` left intact (intentional straight-through estimator; backprop allowed).

## Deferred (in the audit report, not implemented)

Always-on perf rewrites (per-batch E-step loop, dense Omega materialization, full-K matrix_exp)
parked for a GPU-verification branch; gated perf caches (Killing inverse, causal mask, norm
objects); pre-existing dead `effective_temperature()` left in place per the surgical policy.
