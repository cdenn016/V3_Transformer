# Deep Audit — Round 3 — 2026-07-01

Third-round ultradeep sweep, continuing beyond the 30 findings (F1–F12, C1–C18) fixed in
commit `1c8f046` and merged to main as `7de76c6`. Investigators were briefed to skip the
already-fixed findings and to treat the fix commit itself as prime hunting ground for
regressions. Branch: `audit/ultradeep-round3-20260701` (equal to main; the uncommitted
`scaling.py` ROUTES WIP edit was excluded from scope as intentional working state).

## Scope

Whole repo: the `vfe3/` package (config, free_energy, belief, divergence, alpha_i, lambda_h_i,
attention_prior, numerics, metrics, ema, train, run_artifacts, gauge_optim; families/, geometry/,
gradients/, inference/, model/, data/, viz/) plus the ten click-to-run entry scripts. `tests/`
read as evidence of intent only.

## Investigators Dispatched

- Base (wave 1): code-reviewer, debugger, refactoring-specialist, performance-engineer, python-pro.
- Experts (wave 2, theory gate met — CLAUDE.md declares gauge/VFE/SPD/divergence invariants; whole-repo
  audit, so the full `audit-*` pool ran): audit-gauge-theorist, audit-geometer, audit-info-geometer,
  audit-numerical-analyst, audit-variational, audit-transformer-ml, audit-implementation-engineer.
- Verifier: one independent `general-purpose` agent per merged finding (source-only rule).
- Challenge: audit-skeptic vs audit-defender on every CONFIRMED critical/high finding; orchestrator
  adjudicated against cited source.

Operational note: the first workflow run hit a session token limit after the base wave plus two
experts; it was resumed from the workflow journal (cached base results replayed; the five remaining
experts, dedup, verification, and challenge ran live). The two experts whose first-run outputs
carried a safety-classifier-unavailable caveat (geometer, info-geometer) contributed only findings
that were independently re-verified and orchestrator spot-checked by direct grep.

## Findings Summary

26 raw findings from 12 investigators, deduplicated to 17 unique. Verifier verdicts:
16 CONFIRMED, 1 REFUTED, 0 INCONCLUSIVE. One cross-investigator contradiction (geometer's
"only register_compose guarded" vs three others' "four registries guarded") was resolved by
the verifier and an orchestrator grep: exactly four registries carry the F12 fail-closed guard
(`alpha_i.py:34`, `attention_prior.py:42`, `geometry/lie_ops.py:95`, `inference/policy.py:54`).

Per-investigator raw counts: code-reviewer 3, debugger 3, refactoring-specialist 4,
performance-engineer 1, python-pro 2, gauge 2, geometer 1, info-geometer 2, numerics 1,
variational 3, transformer 1, implementation-engineer 3.

## Verifier Verdicts

| # | Finding | Sev (filed) | Verdict | Decisive source |
|---|---------|-------------|---------|-----------------|
| 1 | F1 fix adds a second unconditional per-step CPU-GPU sync on the default (disabled-scaler) training path | high | CONFIRMED | `vfe3/train.py:387-393`, gate outside the metrics block; default `amp_dtype=None` (`config.py:606`) disables the scaler |
| 2 | F12 fail-closed duplicate-registration guard covers only 4 of ~22 `register_*` decorators | medium | CONFIRMED | guard message greps to exactly 4 files; ~18 seams incl. transport, family/functional, kernel, retraction, encode/decode, preference/ambiguity are plain `dict[name] = fn` |
| 3 | `generate_efe.py` loads checkpoints with `weights_only=False` — the only ungated unsafe-deserialization path in the repo | medium | CONFIRMED | `generate_efe.py:63,71`; contrast gated fallback `run_artifacts.py:324-335`, safe load `sigma_gate_measure.py:78` |
| 4 | `vfe3/inference/masked_retrieval.py` is an entire orphaned module | medium | CONFIRMED | zero importers repo-wide; single WIP commit `762863e` ("experiment not run"); design doc still exists (finding's deletion claim was wrong on that point only) |
| 5 | `ablation.py` BASELINE_CONFIG silently diverged from `train_vfe3.py` despite its "currently matches" claim | medium | CONFIRMED | `ablation.py:143,145,279` (`adam`/`killing_per_block`/`0.0035`) vs `train_vfe3.py:135,137,272` (`heavy_ball`/`pullback_per_block`/`0.0045`) |
| 6 | `register_policy` `override` param not keyword-only, diverging from its sibling F12 fixes | low | CONFIRMED | `inference/policy.py:48` vs `alpha_i.py:19`, `attention_prior.py:34`, `lie_ops.py:87`; introduced in `1c8f046` |
| 7 | Sigma-gate artifact slugify (C12 fix) lacks the collision hash its sibling C15 fix got | low | CONFIRMED | `inference/sigma_gate.py:185-188` (lossy slug, mode "w") vs `ablation.py:1626-1627` (sha1 suffix) |
| 8 | `report.py _emit` leaks the matplotlib figure when the thunk raises after figure creation | low | CONFIRMED | `viz/report.py:182-189` closes only on success; thunks can raise in `tight_layout`/`savefig` after `plt.subplots` |
| 9 | `_atomic_replace` orphans the `.tmp` file when `os.replace` exhausts retries or hits a non-PermissionError | low | CONFIRMED | `run_artifacts.py:58-65`; no unlink anywhere in vfe3/; reachable from `train.py:995` and `:857` |
| 10 | Dead figures.py helpers `_scatter_by_category` / `_annotate_frequent_tokens` superseded by clustering path | low | CONFIRMED | `viz/figures.py:1433,1449` — definitions are the only repo-wide hits; note `_token_category_labels` plumbing IS still live, delete only the two functions |
| 11 | `connection_m/l_norm`, `connection_l_offdiag_norm`, `head_mixer_drift` logged to metrics.csv but never plotted | low | CONFIRMED | `model/model.py:1721-1732` + `train.py:938-941` allow-list; zero consumers in viz/ or ablation plotting |
| 12 | `holonomy_wilson_sampled` per-head Wilson decomposition mis-slices unequal irrep towers | low | CONFIRMED | `metrics.py:764-783` divisibility-only guard + equal-width reshape; unequal so_n/sp_n towers exist; latent (production callers use n_heads=1) |
| 13 | `viz/extract.py` hardcodes transport-mode string tuples instead of `_TRANSPORT_NEEDS_MU/SIGMA` registry | low | CONFIRMED | `viz/extract.py:421-422,510-511,588-589` vs registry pattern at `model.py:1536`, `e_step.py:257-258`; latent until a new belief-dependent regime registers |
| 14 | `build_belief_transport` hard-codes per-mode state routing; a new stateful transport silently builds stateless | low | CONFIRMED | `inference/e_step.py:151-163` if/elif over four literals with empty else; registry metadata unused here; latent |
| 15 | `fisher_trace` returns half the mean-block Fisher trace it claims | low | REFUTED | `metrics.py:255,258` explicitly declare tr(Σ⁻¹)/2; consumers label ⟨tr Σ⁻¹⟩/2 (`figures.py:3160,3188`); pinned by `test_metrics.py:150` — selective docstring quote, no code/claim mismatch |
| 16 | `FullGaussian.entropy()` discards the safe_cholesky ok-mask, finite-but-wrong entropy on non-PD Σ | low | CONFIRMED, no action | code-fact true (`families/gaussian.py:306`), but behavior is test-pinned intentional (`test_full_covariance.py:128-137`) and was already adjudicated as L9 "tried and reverted"; zero production callers |
| 17 | `clamp_monitor` Frobenius-surrogate diagnostic (C14 fix) has no config field or production call site | low | CONFIRMED | `geometry/transport.py:713,757-765`; all five production `stable_matrix_exp_pair` call sites omit it; no `VFE3Config` field |

## Adversarial Challenge (CONFIRMED critical/high only)

One finding qualified (#1). Skeptic and defender both ran; orchestrator adjudicated.

| # | Finding | Skeptic (attack) | Defender | Verdict | Reason |
|---|---------|------------------|----------|---------|--------|
| 1 | F1 fix second unconditional per-step sync | Real but severity-inflated: `train.py:331` `float(loss.detach())` already drains the pipeline every step, so the marginal cost is kernel-launch overhead, not a fresh stall → low | Reachability ironclad; concedes high, argues medium: doubles the loop's tracked one-unconditional-sync property, O(P) launches every silent step, cheap fix | **DOWNGRADED → medium** | `train.py:331` bounds the marginal stall (kills high); but an unconditional O(P) per-step tax with a device-side fix available is above low — defender's conceded position adopted |

## Surviving Punch List (post-challenge)

No critical or high findings survived. Actionable items, ranked:

1. **[medium]** F12 registry-guard coverage — `vfe3/geometry/transport.py:130` et al. — apply the same
   fail-closed `override`-gated guard to the remaining ~18 `register_*` decorators (transport, family,
   functional, functional_per_coord, kernel, retraction, norm, group, irrep, precond, pos_rotation,
   pos_phi, encode, decode, preference, ambiguity, metric, monitor, figure).
2. **[medium, downgraded from high]** F1 finiteness-gate hot-path regression — `vfe3/train.py:387-393` —
   replace the per-parameter Python/host check with a fused device-side reduction (or identity-scale
   GradScaler `found_inf` bookkeeping) so the default path returns to one unconditional sync per step.
3. **[medium]** `generate_efe.py:63,71` — load with `weights_only=True` (the same bundle already loads
   safely that way in `sigma_gate_measure.py:78`), or gate the unsafe fallback like `run_artifacts.load_checkpoint`.
4. **[medium]** `ablation.py:143,145,279` — re-sync BASELINE_CONFIG to `train_vfe3.py`'s operating point
   or drop/correct the "currently matches" claim (note: `e_phi_lr` sweep cells exercise a different
   Riemannian metric than production).
5. **[medium]** `vfe3/inference/masked_retrieval.py` — wire into an experiment script or delete
   (user decision; single WIP commit, never imported).
6. **[low]** `inference/policy.py:48` — make `override` keyword-only.
7. **[low]** `inference/sigma_gate.py:185-188` — add the sha1 collision suffix to the artifact slug.
8. **[low]** `viz/report.py:182-189` — close the figure in a finally/except path.
9. **[low]** `run_artifacts.py:58-65` — best-effort unlink of the orphaned `.tmp` before re-raising.
10. **[low]** `viz/figures.py:1433,1449` — delete the two dead helpers (keep `_token_category_labels`).
11. **[low]** Surface or document the four unplotted trainability diagnostics (`model.py:1721-1732`).
12. **[low]** Latent extensibility trio: irrep-aware per-head Wilson slicing (`metrics.py:764-783`);
    registry-driven mu/sigma gating in `viz/extract.py` (3 sites) and `build_belief_transport`
    (`e_step.py:151-163`); wire `clamp_monitor` to a default-OFF config field (`transport.py:713`).

Non-actions: #15 REFUTED (fisher_trace correctly documents and labels the /2 convention);
#16 confirmed as code-fact but intentionally pinned by test and prior adjudication (L9) — do not "fix".

## Test Suite

- Command: `python -m pytest --junitxml=<scratchpad>/pytest-round3.xml` (global Python 3.14; repo
  `.venv` is a minimal torch-only env without pytest)
- JUnit XML: `tests=1450 errors=0 failures=0 skipped=1`, 233.2s
- Console: `1448 passed, 1 skipped, 1 xpassed, 204 warnings in 233.17s`
- Failures: none

## Token/Agent Accounting

32 agent dispatches across two workflow runs (resume after session-limit interruption),
~5.0M subagent tokens total, ~840 tool uses.

## Implementation Status (2026-07-02)

All actionable punch-list items implemented on `audit/ultradeep-round3-20260701` via a 6-worker
file-disjoint workflow, each worker followed by an independent diff reviewer (the geometry
reviewer hit a session limit; the orchestrator reviewed that diff directly — pass):

1. Registry guards: fail-closed duplicate-registration guard (keyword-only `override=True`
   escape) added to all 19 remaining `register_*` seams (16 in the registry sweep + transport,
   metric, figure), bringing coverage to 23/23; `register_policy`'s `override` made keyword-only.
2. F1 sync regression: default path back to exactly ONE unconditional D2H sync per step — the
   loss value and grad-finite flag ride one fused `torch.stack(...).tolist()` transfer.
3. `generate_efe.py`: both loads now `weights_only=True`.
4. `ablation.py` docstring: false "currently matches" claim replaced by the three real deltas
   (config values untouched).
5. `masked_retrieval.py`: DEFERRED — user decision (wire or delete).
6. Sigma-gate artifact slug: sha1[:8] suffix (mirrors C15).
7. `report.py _emit`: closes thunk-created figures on failure via fignums diff.
8. `_atomic_replace`: best-effort tmp unlink on raising paths only.
9. figures.py: two dead helpers deleted; `plot_geometry_health` gains a self-gating
   "Learned-connection trainability" panel (connection_w/m/l norms, l_offdiag, head_mixer_drift).
10. `holonomy_wilson_sampled`: keyword-only `irrep_dims` for unequal towers (None path
    byte-identical).
11. Registry-driven wiring: `build_belief_transport` + three `viz/extract.py` sites gate
    mu/sigma on `_TRANSPORT_NEEDS_MU/_SIGMA`; new `VFE3Config.transport_clamp_monitor`
    (default False) threads to all five production `stable_matrix_exp_pair` sites.

Non-actions honored: #15 (fisher_trace, refuted) and #16 (FullGaussian.entropy, test-pinned
L9 decision) untouched. Six new test files (`tests/test_round3_*.py`, 27 tests).
