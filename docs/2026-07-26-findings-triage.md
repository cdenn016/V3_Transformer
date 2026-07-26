# Findings triage, 2026-07-25 to 2026-07-26

Six agents traced the last two days of work — the 2026-07-24 and 2026-07-26 deep audits, the RoPE
R-1..R-6 investigation, the B-01 data-integrity fallout, the performance/hypothesis threads, and the
test lanes — against the actual source on `main` @ `1070b12`. This is the ranked residue.

Everything below was read from code or from artifacts on disk. Where an agent's claim did not survive
direct checking, the correction is recorded inline rather than dropped.

## Verified clean (no action)

The three test lanes reproduce their recorded numbers exactly, machine-read from JUnit XML: CPU fast
`tests=4413 failures=0 errors=0 skipped=28`, CUDA `tests=24 failures=0 errors=0 skipped=0` on
`2.10.0.dev20251210+cu128` with `cuda.is_available()=True`, CPU slow `tests=3 failures=1` — the same
pre-existing `test_finalize_skips_figures_when_disabled` RuntimeError at `run_artifacts.py:2838`, not
a new regression. Local `main` is identical to `origin/main`; the stash is empty; `ablation.py` and
`train_vfe3.py` are the only uncommitted changes.

Nine of the twelve punch-list items are genuinely fixed in the executable path, not merely claimed:
B-01, A-01, B-02, E-01, C-04, C-05, C-06, D-02, and the E-02/E-03/E-04/E-05 registry cluster. Six
deferred findings were silently fixed by later work (B-05, B-06, B-07, B-09, D-05, E-10).

## Ranked list

### 1. Seven training runs from 2026-07-26 exist on disk and are recorded in no document

Two agents found this independently; I verified the configs directly. Among them is the R-4 decisive
experiment, whose result answers a pre-registered question and is written down nowhere.

Single-field diffs, confirmed from each run's `config.json`:

| arm | differing field | test PPL |
|---|---|---|
| `164.12_no rope-no-pos-phi` | `pos_rotation='none'` (control) | 164.12 |
| `163.99_left-rope-no-pos-phi` | legacy left composition | 163.99 |
| `161.64_right-rope-no-pos-phi` | `rope_insertion='right'` | **161.64** |
| `172.83_right-rope-on-value` | `rope_on_value=True` | 172.83 |
| `178.17_right-rope-no-alibi` | `causal_noself` | 178.17 |
| `178.78_RIGHT-ROPE-BASE=1000` | `rope_base=1000` | 178.78 |
| `181.19_...s6` | no rope, `causal_noself` (control) | 181.19 |

Read against the project's working noise floor of **±0.75 PPL** for multi-seed spread (a band ~1.5
wide), three consequences, none of them recorded:

**The R-4 fix is real and it was masking the whole rope effect.** Right buys 2.35 PPL over left and
2.48 over no rope — both ~3.2x the half-spread, comfortably outside noise. The legacy left arm bought
0.13 PPL over no-rope, which is deep inside the band, i.e. exactly zero. So the earlier reading "rope
contributes approximately nothing" was correct *about the left composition* and does not survive the
fix: with the rotation folded into the frame, rope contributes ~2.5 PPL.

This splits the pre-registered decision rule at `docs/2026-07-26-rope-investigation.md:344` ("if the
corrected rope still lands near 164, the direction closes") into two questions it conflated. Does rope
explain the ~22.7 PPL `pos_phi='learned'` gap? **No** — 161.64 is far nearer 164 than 141, and that
half of the rule fires. Does rope contribute at all? **Yes, ~2.5 PPL, outside noise.** R-4 was the
blocker for rope's contribution; rope's ceiling is simply ~2.5 PPL rather than ~22.7. The direction
closes as a *primary* performance lever, not as a real effect.

`rope_on_value=True` is **11.2 PPL worse** than `False` at otherwise identical config — an order of
magnitude past the noise band. This directly contradicts `docs/2026-07-26-edits.md:560`, which states
"`rope_on_value=True` is the better pairing." That recommendation was argued from the gradient path
(it keeps `mm_exact` legal so the R-1 freeze cannot occur) and is correct on those grounds, but it is
now measured as costly, and the doc's unqualified wording should not stand.

`rope_base=1000` versus `100` is **178.78 vs 178.17 — a 0.61 PPL difference, inside the noise band.**
This is a null result and carries no information about R-5 in either direction. It should not be cited
as support for rejecting "raise the base"; that rejection continues to rest on the `d_head=10`
band-count argument alone.

**Action:** record the seven runs and these three conclusions in `docs/`. Cost is prose only. Also
verify `140.48_alibi-slope=2` and `141.95_1estep-2s-steps` before citing them — one agent reported a
val figure inconsistent with the directory name, and the alibi arm may differ in `oracle_unroll_grad`
as well as slope, which would make it a two-field comparison.

### 2. `docs/audit-results.md` is now the stalest artifact in the set

Lines 1121-1124 state that the K=20 pair-precision share "cannot be traced to any artifact on disk",
that "attention share rises with width" **cannot be repaired**, and that re-establishing it "needs a
fresh K=20 `mm_exact` + `s_e_step=True` run."

That run exists. `vfe3_runs/138.40_wikitext-103_K20_block_glk_linear_mix_s6/config.json` gives
`embed_dim=20`, `s_e_step=True`, `e_step_update='mm_exact'`, `n_layers=1`; its `estep_character.json`
gives `measured_channel='belief'`, `n_sequences=64`, `precision_split_available=true`,
`recompute_max_abs_err=0.0`, and `points[*].pair_precision_share = 0.15327950264845877`. (The
top-level `pair_precision_share` key is `None` — the value is per-point. A top-level read reports the
obligation as unmet, which is how this stayed unnoticed.)

The audit of record is the document most likely to be consulted first, and it currently sends a reader
to re-run an experiment that is done. **Action:** append a resolution note pointing at that artifact
and at the attribution correction in `docs/2026-07-26-edits.md:344-352`.

### 3. The published K=300 pair share 0.196 has no on-disk provenance

`0.196` is the primary corrected value. It is quoted in `docs/2026-07-25-state-of-knowledge.md:164`
and `:201` and on four wiki pages. I searched the repo and the external checkpoint directory
(`Desktop/data/55.41_wikitext-103_K300_block_glk_linear_mix_s6/`) — there is no `estep_character.json`
at either location. The only K=300 artifact that exists is the 8-sequence
`docs/2026-07-26-b01-remeasurement.json`, which reports **0.213**, not 0.196.

So the corrected width pair is half-evidenced: the K=20 endpoint has an artifact, the K=300 endpoint
does not. This is the weakest link in the chain that replaced the retracted numbers, and it is the
same class of defect B-01 was — a published figure with no traceable source.

**Action:** re-emit and persist the K=300 64-sequence `estep_character.json`, or restate the claim at
the 8-sequence value that does have an artifact. Also either add a `superseded_by` field to
`b01-remeasurement.json` or stop citing it at `state-of-knowledge.md:157` as the "Raw record" for a
table whose values it does not contain.

### 4. The B-03 fix is correct but unpinned — a revert would pass the whole suite

`run_artifacts.py:3136` correctly reads `pinned_s_depth = trained_s_depth if trained_s_depth is not
None else trained_depth`. But `tests/test_estep_depth_sensitivity_channels.py:63` builds its model
without `s_e_step_n_iter`, so the fallback makes both values equal and the assertion passes under the
original buggy code. No test in the suite asserts `s_depth == s_e_step_n_iter` at a config where the
two differ.

B-03 was rated high and triple-confirmed with a repro. It is now protected by nothing.
**Action:** one test case with `s_e_step_n_iter != n_e_steps`.

### 5. C-07 — no config-reachable fail-closed path for a non-group-inverse transport

Promoted from low on new evidence rather than re-judgment. `TRANSPORT_CLAMP_MAX_NORM = 20.0`
(`transport.py:1393`) sits past where the fp32 exponential pair stops being a group inverse, and all
three guards that would catch it are unreachable from a config:

- `_checked_group_inverse` is called at `transport.py:1612`, `:1674`, `:1683` with no `residual_tol`,
  and the parameter defaults to `None`, which by its own docstring skips the identity residual check.
- `validity_max_norm` — the fail-closed chart bound threaded end-to-end through `transport.py`
  (`:798`, `:832`, `:1023`) — has **no `VFE3Config` field at all**.
- `transport_clamp_monitor` defaults `False` (`config.py:601`).

The project's standing rule is that a theoretically pure path must exist under appropriate toggles.
Here there is no toggle that makes transport fail closed, the failure produces wrong numbers rather
than a crash, and it is on the training hot path. Whether production `||phi||` actually reaches the
clamp is unmeasured — and the monitor that would answer it is the one that is off by default.

### 6. `so_n` / `sp_n` cannot use RoPE on the default insertion

`config.py:2121` probes the group with `_builder(K=self.embed_dim).irrep_dims` inside `except
TypeError`. For `so_n`/`sp_n` the builder raises **`ValueError`** when `group_n`/`irrep_spec` are
absent (`groups.py:495`), and the probe never forwards them. Reproduced by the agent: a config with
`gauge_group='so_n', group_n=3, irrep_spec=[('l1',1),('l2',1)], rope_insertion='right'` raises
`ValueError` blaming the user for fields they did set; the same config with `rope_insertion='left'`
constructs fine.

Secondly, the acceptance rule `len(_dims) < 2` is weaker than the fusion rule `_can_fuse_flat`, which
also requires `len(set(irrep_dims)) == 1` (`e_step.py:200-203`) — so an so_n spec like `[3, 5]` passes
config validation and then hits the runtime raise in `RopeTransport.score_operator`
(`transport.py:266-270`). Fails closed, but at the wrong layer.

Mechanical fix; worth doing independently of rope's measured value, since it is a config-validator bug.
The whole `rope_insertion` rejection rule is also currently unpinned — no test matches the rejection
strings.

### 7. C-09 — the trust-region guard NaNs the row it is meant to bound

`retraction.py:458` (twin at `:706`):
`exp_arg = exp_arg * torch.clamp(trust_region / (tangent_norm + eps), max=1.0)`. `tangent_norm` is a
per-row reduction, so one `inf` coordinate makes the ratio `0`, `inf * 0 = NaN`, and the NaN survives
both the `.clamp(-50, 50)` at `:459` and the final `clamp(min=lower_bound)` at `:462` — the entire
`(..., K)` row goes NaN. The guard destroys the step exactly in the regime it exists to rescue, and the
NaN propagates into the belief rather than aborting.

### 8. C-01/C-02/C-03 — the congruence-guard obligation is still undischarged

The guard is byte-for-byte the reported defect (`transport.py:1428-1430`), three call sites still pay
the per-call host sync (`:2527`, `:2619`, `:2714`), and no conditioning proxy exists. The audit's
stated precondition — instrument `cond(U)` on a K=300 run — has not been met: the only recorded
`vertex_cond` figures are K=20-era (`docs/audits/audit-obligations-closed-2026-07-25.md:49-51`).

The instrumentation already exists (`vertex_cond_max`, `model.py:2945-2994`), so this is a run, not a
code change. Severity stays medium *because* the operating point is unknown; if K=300 `cond(U)`
exceeds ~1e3 the silent-error band is live and this becomes high.

### 9. Retracted B-01 numbers still circulating

Repo-side, the correction largely landed (`state-of-knowledge.md` §6 is properly struck through). What
remains:

- `docs/2026-07-25-performance-brainstorm.md:130` and `:258` — the retracted 0.298/0.702 split, with a
  downstream λ-tuning calculation built on it (`λ*0.298/(λ*0.298+0.702)` → "0.46 at λ=2"; at the
  corrected 0.196/0.804 it is ≈0.33). **The audit never named this file**, so it was missed entirely.
- `wiki/methods/Iterative amortized inference.md:62-63` — 20%/70-73%/0.96-0.98 presented as current in
  the same paragraph whose next sentence was corrected.
- `wiki/projects/VFE Transformer Program.md:137` — "its rise with width (0.190 → 0.298)" and
  "0.521 vs 0.147", unmarked, positioned *after* the correction callout.
- `wiki/concepts/Precision weighting.md:75` — `0.147` unmarked.
- `index.md:1105` — the token-prior note's entry lacks the supersession marker that `:1104` has.

The vault is not circulating retractions unmarked — it has a proper correction note, an index entry, a
log line, and three fully-corrected pages. This is residue, not systemic. Wiki edits require the
user's confirmation and have not been made.

### 10. The repo contradicts itself on the layer-stack prior handoff

The "degenerate one-token-per-agent shadow" characterization is still verbatim at
`docs/2026-07-25-state-of-knowledge.md:135` and `docs/2026-07-25-shadow-prior-investigation.md:110`.
The audit established it is wrong — `mu_p_i` and `mu_q_i` live in the same fiber over token `i`, so no
transport belongs there — and `docs/2026-07-25-performance-brainstorm.md:118-123` independently
reaches the correct conclusion. Two documents in `docs/` now state opposite things.

Only the closing illustration of shadow-prior §2 depends on it; the "do not build it" verdict rests
independently on §3 and §4 and survives.

### 11. `docs/2026-07-25-vfe4-performance-hypotheses.md` supersession banner is wrong

Lines 13-23 still list proposals 1.5/2.1 (the frame-intrinsic family, called "the strongest single
proposal") as "survives… not yet tested". They are measured-refuted: 139.3 → 308.1 PPL with
`phi_embed.grad is None`. Flagged on 2026-07-25 and still uncorrected.

### 12. D-08 — `mass_phi` blocks the ablation axis it exists to measure

`e_step.py:776` uses `0.5 * mass_phi * (phi ** 2).sum()`; `model.py:1603` uses
`0.5 * cfg.mass_phi * (belief.phi ** 2).mean()`. The two differ by `phi.numel()` (10³–10⁵), and
`free_energy_value` still accepts-and-ignores the knob (`e_step.py:469`), so the logged F carries
neither. Latent at the `0.0` default, but this is a swept axis, and on a sweep the F log measures a
functional the phi substep does not descend.

### 13. Lower-priority residue

R-3 is only partly resolved: the second figure set exists and is subtitled, but the original
`attention/` figures still ship with the unqualified title `"Attention (step N) - layer L head H"`
(`figure_worker.py:366`), and the numeric beta diagnostics still read the post-block re-score
(`model.py:2777`, `:3413`, `extract.py:1102`). Those are internally consistent, so this is labeling
exposure, not wrong numbers — but it is the same "right quantity, wrong point" class that started the
investigation.

Also open: D-04's false docstring premise (`extract.py:1145-1147`, research-facing only); D-07's fp64
reduce gating a boolean `pair_mask` (`pairwise_stats.py:96-98`, `:120`) — a comparability risk, no
mask flip observed; E-14's docstring recommending the non-gauge-equivariant `box` mode
(`numerics.py:127`) in a gauge-equivariance project; A-03's ~73x-low overflow probe under the frame
family (`model.py:3003`); C-12's BCH residual guard inert at its `None` default; D-06's unchecked
block-diagonal assumption (`exact_congruence.py:190-200`); E-13's `Popen` bypassing process
containment (`figures.py:237`). C-10 I would **demote**: the audit's stated mechanism does not hold, as
a negative and a `0.0` both land on `eps` after the downstream `clamp(min=eps)` — the real defect is
fail-silent admission of a boundary point, not wrong numbers.

Repo hygiene: 21 fully-merged local branches, three stale worktrees (one locked, inside the repo under
`.claude/`, pinning an otherwise-merged branch; one on a detached HEAD unreachable from any branch),
and loose `.verification/*.xml` at that directory's root. All gitignored, none tracked.

## Reading the corpus against the ±0.75 PPL noise floor

Every comparison in this two-day corpus is single-seed, but the project's working multi-seed spread is
**±0.75 PPL** (a band ~1.5 wide), which is enough to sort them. Applied to the K=20 arms:

| comparison | delta | verdict |
|---|---|---|
| `pos_phi='learned'` gap | ~22.7 | real, dominant |
| no-head-mixer 148.25 vs 138.40 | 9.85 | real |
| `rope_on_value` True vs False | 11.19 | real |
| `s_e_step_n_iter=2` 141.95 vs 138.40 | 3.55 | real |
| rope without ALiBi 181.19 vs 178.17 | 3.02 | real |
| route `gradient` 141.38 vs `mm_exact` 138.40 | 2.98 | real |
| rope right vs no-rope | 2.48 | real (~3.3x half-spread) |
| rope right vs left | 2.35 | real (~3.1x half-spread) |
| `alibi_slope=2` 140.48 vs 138.40 | 2.08 | probably real, but possibly two-field |
| 2-layer 139.97 vs 138.40 | 1.57 | marginal, ~2x half-spread |
| `rope_base` 1000 vs 100 | 0.61 | **inside noise — null** |
| rope left vs no-rope | 0.13 | **inside noise — zero** |
| K=20 bound decomposition, total spread | 0.235 | **inside noise — not established** |

Two results that were being carried as findings dissolve at this floor: the `rope_base` comparison is
a null, and the entire K=20 phi-bound decomposition in
`docs/2026-07-25-phi-bound-calibration-and-stage0-report.md` — the basis for the
`phi_mstep_max_matrix_norm = 13` recommendation — has a total spread of 0.235 PPL across all its arms,
so it does not separate them. That materially weakens the case for porting bound-13 to K=300 as the
primary arm, since the K=20 evidence it rests on is indistinguishable from seed noise.

Everything else on the list clears the band, so the single-seed caveat does not undercut items 1 or 2.
The seed-0 replicate in flight (`vfe3_runs/20260726-181852_..._s0`) is still worth having to confirm
the floor holds at this specific config, but it is no longer a precondition for acting.

## Highest-value untested experiments

Ranked by how much the outcome would change what happens next:

1. **The three-way split of row-centered logit variance** (content / token-frame / position-frame).
   Cheap — checkpoint-only, no retraining. Everything structural in the performance brainstorm is
   priced against a number nobody has measured.
2. **Re-establish the phi-bound decomposition above the noise floor at K=20.** Its 0.235 PPL total
   spread is inside the ±0.75 band, so the bound-13 recommendation currently rests on nothing that
   separates. Either replicate to shrink the error on the differences, or widen the swept range until
   the arms separate. Cheap (~30 min/arm) and it gates item 3.
3. **`phi_mstep_max_matrix_norm = 13` with `group_product` at K=300.** Medium (one run; the last K=300
   run was 18.7 h). Named the primary arm and unrun for three weeks — but see item 2: its supporting
   K=20 evidence does not currently clear the noise floor, so this should not go first.
4. **The Stage 1 config bundle at K=300** (`exp_fp64_mode='norm'` + threshold 21, weight-decay
   exemptions, `min_lr_frac=0`, warmup ≈1500). Cheap to configure; the fp64 change alone is claimed at
   up to 279 ms of a 366 ms step, which buys every later experiment.
5. **The registered `s_e_step=False` flat-depth-curve prediction.** Cheap, checkpoint-only, and it is a
   falsifiable prediction already on the record with no cost to settle.

## Numbers that do not reconcile across documents

The K=300 baseline is quoted three ways for one checkpoint — val 54.18, test 55.41, test CE 4.0148 —
used interchangeably, with `state-of-knowledge.md:11-12` pairing a val PPL with a test CE. The
recorded 55.414/CE 4.014838 versus re-scored 55.275/CE 4.012327 split was reconciled once (stage0 §5)
and then quoted without the caveat. `pos_loss_ratio` is 0.9796 logged against 0.9282 computed from the
M4 curve, a discrepancy stage0 could not resolve and that hypotheses §2.3 still builds on. Parameter
count appears as both 528,901,458 and 498,747,258 (both correct for different revisions; the brainstorm
header says "529M" unqualified). A K=300 depth-1 CE of 3.774 — a single-batch figure — is tabulated
beside full-split numbers.
