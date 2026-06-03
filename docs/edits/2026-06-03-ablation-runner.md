# 2026-06-03 — Ablation/sweep runner (`ablation.py`)

## What was added

A new top-level `ablation.py` provides click-to-run hyperparameter sweeps over the
operating point defined in `train_vfe3.py`. It is a fresh, V3-native module (not a port):
the VFE_2.0 `transformer/vfe/vfe_ablation_suite.py` served only as a reference for what an
ablation suite should provide. Nothing is imported from VFE_2.0; the runner depends only on
the V3 codebase, so V3 remains standalone.

The baseline configuration is imported from `train_vfe3.py` (`from train_vfe3 import config`)
rather than copied, so a sweep always ablates around exactly what a normal training run would
use and there is no second operating point to drift out of sync. Each cell builds a fresh
`VFE3Config` and `VFEModel`, trains through the existing `vfe3.train.train`, and reuses the
existing `RunArtifacts` persistence layer so every run produces the usual self-contained
directory (`config.json`, `metrics.csv`, `best_model.pt`, figures) nested under its sweep,
plus an `ablation_result.json` headline used for resume and the sweep leaderboard. Two sweep
shapes are supported through a single `SWEEPS` registry: single-field sweeps that vary one
`VFE3Config` field across an explicit `values` list or an arithmetic `range`, and multi-arm
sweeps that declare a list of named `configs` whose arms may differ in several fields at once
(for example the full-covariance arm that flips `family` and `diagonal_covariance` together
and moves off the per-coordinate alpha form). An optional `requires` block per sweep merges
prerequisite overrides into every cell before the swept field, so a cross-field constraint is
pre-satisfied rather than left to fail. The `CONFIG` dict at the bottom selects the action
(`train`, `analyze`, `plot`, `list`), the dataset, the output directory, an optional
`max_steps`/`max_tokens` cap, the seed, and whether to resume.

## Design choices specific to V3

Model selection is validation-only. The held-out test split is deliberately not scored per
cell, since that would leak the test set into selection and cost a full extra evaluation on
every run. The headline metric is `best_val_ppl` (or the final validation perplexity when no
periodic eval fired). To obtain the test number for a winning configuration, its fields are
copied into `train_vfe3.py`, which already calls `finalize_run` for the test evaluation. Per-
cell checkpointing is forced off (`checkpoint_interval = 0`) so a many-cell sweep does not
fill the disk with resumable checkpoints.

Resume is made safe against baseline drift. Because the imported baseline can change between
sessions (the `train_vfe3.py` config is edited in place), a cached cell is reused only when its
persisted `config.json` still equals the configuration this run would build; otherwise the cell
is re-run. A cell keyed only by its `param=value` label would otherwise be served stale after an
unrelated baseline edit such as `embed_dim`, which for a comparison tool is a trust hazard rather
than a convenience.

Three guards adapt the runner to V3's strict configuration surface. First, every swept field
name is checked against the real `VFE3Config` dataclass fields at startup, so a typo aborts
loudly with the offending names rather than being silently ignored (which would make every
cell identical and read as "this field has no effect"). Second, the data loader is memoised on
`(dataset, max_seq_len, batch_size, split)`, so a sweep over `batch_size` or `max_seq_len`
builds a distinct matching loader instead of reusing the wrong one, while ordinary sweeps still
pay the corpus-cache load only once. Third, a configuration that `VFE3Config.__post_init__`
rejects (a cross-field violation) is caught and tagged `error_kind = "config"`, kept distinct
from a training crash (`"train"`), so a mis-specified cell is not silently bucketed with a
genuine failure. Per-cell failures are isolated: one crash is recorded and the sweep continues.

To make comparisons trustworthy, each cell is reseeded after model construction and before
training, reseeding both the global RNG and any loader's own generator. Model construction
consumes a config-dependent amount of RNG, and a cached loader's shuffle otherwise advances
across runs, so without this step the same configuration would see a different data stream
depending on its position in the sweep. With the reseed, a cell's result is identical
regardless of order; this was verified directly (a forward and reversed `kappa` sweep produce
a bit-identical perplexity for the shared point).

## Verification

A throwaway smoke test (synthetic period-3 stream, tiny dimensions, run on CPU and then
removed) exercised the runner end to end and confirmed: the field-name guard rejects a bogus
field; a cross-field violation is tagged `config` and the sweep survives; a per-cell training
crash is isolated with both rows still written to `sweep_results.csv`; and a forward/reversed
sweep is order-independent to the bit. A separate dry-construct pass built every arm of all
twelve shipped sweeps at the real baseline dimensions and found no dead-on-arrival configuration,
and the resume staleness check was verified to cache on an unchanged baseline and re-run after an
`embed_dim` edit. The `analyze`, `plot`, and `list` paths were also exercised. All of this ran on
the synthetic stream on CPU; the wikitext-103 / `make_dataloader` path the user actually trains
on was not executed here (its cache is absent on this box and torch is CPU-only), so the order-
independence argument for the global-RNG shuffle that `make_dataloader` uses holds by construction
(the reseed precedes every loader's first iteration) rather than by a bit-for-bit corpus run.

## Model-side bug fixed: the E-step oracle under no_grad

Building the runner surfaced a real, general bug in the E-step belief oracle, and it has now
been fixed. The initial "degenerate batch" hypothesis was wrong; instrumenting the crash site
showed `grad_enabled = False` at the failing `torch.autograd.grad` call with an ordinary nonzero
free energy, so `F` had lost its graph only because the forward was running under `no_grad`.
Tracing the call stack pinned the cause: `vfe3/train.py::evaluate` is decorated `@torch.no_grad()`,
and the VFE forward runs the E-step belief minimization as part of inference. For the closed-form
kernel path (filtering, diagonal Gaussian, KL, canonical) that is fine, but every non-kernel path
(the surrogate `include_attention_entropy = False`, smoothing, full covariance, Renyi order other
than one) computes the belief gradient through `vfe3/gradients/oracle.py::belief_gradients_autograd`,
whose `torch.autograd.grad(F, [mu_q, sigma_q])` cannot run under `no_grad`. The phi sub-step already
guards itself with an `enable_grad` island for exactly this reason; the belief oracle was the one
path missing it. The consequence was that evaluating, or running `diagnostics`, `attention_maps`,
or `generate` (all `@torch.no_grad`), any non-kernel configuration raised "element 0 of tensors
does not require grad and does not have a grad_fn." Training was unaffected because it runs grad-
enabled. The earlier "only reproduced after a canonical run" appearance was an artifact of the
probes calling the real `evaluate` only on the second cell; any non-kernel cell crashes at its own
`evaluate`, independent of order.

The fix decorates `belief_gradients_autograd` with `@torch.enable_grad()`, giving it the same local
grad island the phi step uses. `create_graph` stays `False` and the returned gradients are already
detached, so under a `no_grad` caller the result is a constant tangent with no graph leaking to the
outer scope, and on the grad-enabled unrolled training path the decorator is a no-op and behaviour
is byte-identical. A regression test was added
(`tests/test_gradients_oracle.py::test_oracle_runs_under_no_grad`): it calls the oracle under
`torch.no_grad()` and asserts the result is value-identical to the grad-enabled call and detached.
It was confirmed to fail before the fix (the original `RuntimeError`) and to pass after. The full
suite was run with and without the one-line oracle change and the failure set was identical, which
establishes that this change introduces no regressions. Those pre-existing failures (one native
UMAP `OSError` crash plus several assertion failures) are present on the branch independent of this
change; their root cause was not investigated, as it is outside this task. A spot check with the
unrelated `train_vfe3.py` working-tree knob edits reverted left four of the nine assertion failures
still failing and five passing, which indicates the latter are tied to those edits or are seed-
sensitive rather than caused here, but this was not pursued further. The runner's `entropy_term`
sweep, whose surrogate arm previously crashed, now completes with both arms succeeding.

## Full toggle coverage (train_vfe3 config + ablation registry)

A follow-up request asked for the full set of toggles in both config surfaces. An audit showed
`train_vfe3.py`'s `config` dict carried only 51 of the 74 `VFE3Config` fields (the docstring
already claimed every toggle), so the 23 missing fields were added in their proper groups with
inline comments: the connection regime (`transport_mode`, `cocycle_relaxation`, `cross_couplings`),
the positional block (`pos_phi_compose`, `bch_pe_order`, `pos_phi_scale`, `pos_phi_project_slk`,
`pos_rotation`, `rope_base`, `rope_full_gauge`), the model channel (`mstep_self_coupling_weight`,
`lambda_h`, `gamma_coupling`, `kappa_gamma`, `gamma_attention_prior`, `prior_source`),
`spd_retract_mode`, `decode_chunk_size`, and the training knobs (`e_step_gradient`,
`grad_accum_steps`, `eval_max_batches`, `amp_dtype`). The dict now holds all 74 fields and
constructs.

`ablation.py` was changed (at the user's request) to stop importing the baseline and instead carry
its own self-contained `BASELINE_CONFIG` listing all 74 toggles, so the ablation operating point is
fully visible and editable in one place. The runner's safe-resume guard already protects against the
resulting drift between the two independent copies. The `SWEEPS` registry was expanded from twelve
to sixty-two entries so there is a ready single-field or multi-arm sweep for every meaningful
toggle: sixty-four fields are covered by a sweep and the remaining ten are listed in
`NON_SWEPT_FIELDS` with a reason (dataset-fixed, single live value, or bookkeeping). `SWEEP_ORDER`
stays a curated eight-sweep default; `mode="list"` now prints the whole registry, marking the
active ones.

Validation: every arm of all sixty-two sweeps was dry-constructed at the real baseline (zero
dead-on-arrival), the field-name guard passed over the whole registry, and a representative
non-default cell of every path-changing sweep was run end-to-end (build, train, the `@no_grad`
evaluate) on the synthetic stream. The runner correctly surfaced four model-side limitations the
smoke turned up; three are now fixed (next section) and one is a deeper, codebase-deferred issue.
The `cross_couplings` sweep merges the heads into one super-block, which the two-or-more-block head
mixer cannot apply to, so it sets `use_head_mixer=False` (verified to run).

The full test suite was re-run after the twenty-three-field `train_vfe3.py` config addition and
returned the same pre-existing failure set as before, confirming that the additions (each at its
`VFE3Config` default) introduce no regressions.

## Second-round model fixes (regime_ii grouping + full-cov KL Cholesky)

Two of the flagged limitations were root-caused and fixed test-first, plus a related one-line gap.
`build_optimizer` documented but never closed a known gap: the Regime-II learned connection
`connection_W` (transport_mode='regime_ii') and the learnable self-coupling scalar `log_alpha`
(alpha_mode='learnable') are trainable model-level parameters that were created but added to no
optimizer group, so the exact-coverage guard rejected the whole model. Both are now grouped
(`connection_W` at `m_phi_lr`, `log_alpha` at `m_mu_lr`), so those toggles train; `transport_mode`,
`cocycle_relaxation`, and `alpha_mode='learnable'` are back in the sweeps and verified to run end to
end. Separately, the α=1 branch of the full-covariance Renyi/KL (`FullGaussian.renyi_closed_form`)
used a raw `torch.linalg.cholesky` that raises on a numerically non-PD covariance, while the α≠1
branch already used the jittered `safe_cholesky`; the α=1 branch now uses `safe_cholesky` too, so a
failed factorization clamps to `kl_max` (round 0 adds no jitter, so valid-SPD inputs stay
byte-identical). Two regression tests were added — `test_optimizer_groups_regime_ii_connection_and_learnable_alpha`
and `test_full_kl_survives_non_pd_covariance` — both confirmed RED before and GREEN after, and the
full suite still returns the same pre-existing failure set (no regressions).

`decode_mode='full'` is the one that stays excluded. Hardening its KL Cholesky above moved the
failure one layer down: on the prior-bank path it drives the full-covariance SPD retraction
(`retraction.py::retract_spd_full`) into a `torch.linalg.eigh` that fails to converge on an
ill-conditioned spectrum — and that file explicitly defers a gap-regularized robust eigh. Chasing
that is a deeper, wider-blast-radius numerical change (it touches every full-covariance retraction,
not just the decode), so following debugging discipline it was stopped and surfaced rather than
pursued unilaterally. The `decode_mode` sweep therefore still ships only its two diagonal variants;
full-covariance training remains exercised by the `covariance` sweep.

## Files

- `ablation.py` — new; the sweep runner. Self-contained 74-toggle `BASELINE_CONFIG`; 60-sweep
  registry covering 62 fields with 12 documented non-swept.
- `vfe3/gradients/oracle.py` — `belief_gradients_autograd` decorated `@torch.enable_grad()` so the
  oracle computes the belief gradient even when the caller (eval / diagnostics / generate /
  detached E-step) runs under `no_grad`.
- `tests/test_gradients_oracle.py` — added `test_oracle_runs_under_no_grad`.
- `vfe3/train.py` — `build_optimizer` now groups `connection_W` (regime_ii, at `m_phi_lr`) and
  `log_alpha` (learnable alpha, at `m_mu_lr`); docstring updated.
- `vfe3/families/gaussian.py` — `FullGaussian.renyi_closed_form` α=1 branch hardened with
  `safe_cholesky` (matches the α≠1 branch).
- `tests/test_train.py` — added `test_optimizer_groups_regime_ii_connection_and_learnable_alpha`.
- `tests/test_full_covariance.py` — added `test_full_kl_survives_non_pd_covariance`.
- `train_vfe3.py` — `config` dict completed from 51 to all 74 `VFE3Config` toggles (the 23 missing
  fields added in their groups). Pre-existing click-to-run knob edits left in place.
