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

## Files

- `ablation.py` — new; the sweep runner.
- `vfe3/gradients/oracle.py` — `belief_gradients_autograd` decorated `@torch.enable_grad()` so the
  oracle computes the belief gradient even when the caller (eval / diagnostics / generate /
  detached E-step) runs under `no_grad`.
- `tests/test_gradients_oracle.py` — added `test_oracle_runs_under_no_grad`.
- `train_vfe3.py` — unchanged by this work (the working-tree edits are unrelated click-to-run
  knob changes left in place).
