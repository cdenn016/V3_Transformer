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

## 2026-05-31 - Training console logging + click-to-run entry (train_vfe3.py)

This change adds VFE_2.0-style per-step console output to the training loop and a
click-to-run training entry point, without altering the numerical path.

Files changed:

- vfe3/config.py: two console-only fields, log_interval (default 50) and eval_interval
  (default 0 = off), validated >= 0 in __post_init__. They never enter the E-step or
  M-step computation.
- vfe3/model/model.py: a new @torch.no_grad() VFEModel.diagnostics(token_ids) that, for
  one sequence, re-runs encode plus the block stack, reconstructs the prior entering the
  final block by mirroring the stack.py handoff, and recomputes the converged-belief
  energy E_ij = D(q_i || Omega_ij q_j), attention beta = softmax(B - E/tau),
  self-divergence D(q_i || p_i) and self-coupling alpha_i with the same primitives
  free_energy_value uses, reducing them through the metrics registry to
  {attn_entropy, effective_rank, self_coupling, belief_coupling, total}. It is never
  called by forward or train_step.
- vfe3/train.py: evaluate(model, loader, *, max_batches, device) returning token-weighted
  mean {ce, ppl, bpc} (CE in nats, ppl = exp(min(ce, 20)), bpc = ce / ln 2);
  print_banner(...) (params, structure, M/E learning rates, VFE weights); a private
  _log_step(...) emitting "Step i/N | Loss | CE | H(beta) | it/s | PPL"; and optional
  logging keyword arguments on train(...) (log_interval, eval_interval, val_loader,
  logger). run_training now prints the banner and forwards the config intervals.
- train_vfe3.py (new, repo root): the click-to-run entry, a config dict plus an
  if __name__ == '__main__' block, mirroring VFE_2.0's train_vfe.py. The default
  DATASET = 'synthetic-period3' trains end-to-end with no external data; real dataset
  names use the token cache and fall back to the synthetic stream with a warning when the
  cache is absent, so the run never crashes. The device is auto-detected.
- tests/test_train.py: four tests - evaluate returns finite CE/PPL/BPC with the expected
  identities; logging is byte-identical to the silent path under a fixed seed; diagnostics
  returns finite keyed values; the shipped train_vfe3.py config builds a valid VFE3Config
  and its synthetic loader trains.

Hot-path guarantee. With log_interval and eval_interval unset (the train() default), the
loop takes no extra forward and uses no RNG, so it is byte-identical to the prior silent
path. model.diagnostics and the logging CE read run under torch.no_grad() off the training
graph; the regression test test_logging_does_not_change_losses asserts the loss history is
identical with logging on and off.

Verification. Full suite: tests=212 failures=0 errors=0 skipped=0 (208 prior plus the four
new). A live python train_vfe3.py run prints the banner, per-step lines, validation blocks
at steps 100 and 200, and a final evaluation, with the loss decreasing from about 1.42 to
0.85, below the period-3 unigram floor ln 3 ~ 1.099 (the cutover learnability signal).

Deferred (named). A real-data tokenizer and offline cache builder (VFE_3.0 ships neither,
so only the synthetic stream runs without a pre-populated ~/.cache/tokenized_cache); a
FLOPs-per-step banner field; learning-rate retuning for GPU runs at scale; checkpointing
and metrics-CSV / figure emission.

## 2026-05-31 - Full toggle surface + wikitext-103 default (train_vfe3.py)

Only `train_vfe3.py` changed. The click-run `config` dict was expanded from a ~16-field
subset to the complete 49-field `VFE3Config` surface, and the default corpus was switched
to the cached wikitext-103 (gpt2/tiktoken, vocab 50257). No `vfe3/` source or test was
touched.

What changed in `train_vfe3.py`:

- `config` dict now carries every `VFE3Config` field, grouped exactly as in
  `vfe3/config.py` (numerics, divergence seam, model structure, gauge seam, belief family,
  free-energy coupling, attention, E-step, decode/encode, handoff, normalization,
  M-step/training). Each registry field lists its valid keys inline (`gauge_group`,
  `alpha_mode`, `attention_prior`, `gradient_mode`, `phi_precond_mode`, `phi_retract_mode`,
  `decode_mode`, `encode_mode`, `norm_type_*`, `family`, `divergence_family`). The three
  `NotImplementedError`-guarded toggles (`gauge_parameterization='omega_direct'`,
  `encode_mode='gauge_fixed'`, `use_prior_bank=False`) ship at their working/pure value with
  the rejected alternative named only in a comment, so the dict always constructs. The
  `diagonal_covariance == (family == 'gaussian_diagonal')` cross-constraint is noted in place.
- `DATASET` default is now `"wikitext-103"` and `vocab_size` is `50257`, kept consistent so
  click-to-run loads the real corpus out of the box (a vocab-6 default would crash on the
  gpt2 token ids). The cache is read from `~/.cache/tokenized_cache`.
- New module-level `MAX_TOKENS` knob caps only the *train* stream for fast smoke runs
  (`None` = the full 116.8M-token corpus); the small validation split is always read in full.
- `_select_loader` gained a `split` keyword and `main` now builds a held-out validation
  loader from the `"validation"` split; the final evaluation reports val (not train) CE/PPL.
  A `FileNotFoundError` still falls back to the synthetic period-3 stream with a warning.
- `synthetic_period3_loader` now grows its stream to `seq_len * batch_size * 4` tokens when
  needed. At the new click-run dims (`max_seq_len=128, batch_size=64`) the old fixed 600-token
  stream yielded 4 windows -> zero batches under `drop_last`, which would `StopIteration` both
  the fallback and the `tests/test_train.py` click-run test; the scaled stream yields several
  batches. Small-dim callers (`seq_len=8, batch_size=8`) are unchanged (n stays 600).

Verification. Two read-only expert agents: (1) config completeness - 49/49 fields present,
all values pass `__post_init__`, cross-constraint holds, every inline option list matches the
`_VALID_*` tuples; (2) wikitext-103 data path - the computed `.pt` paths exist, meta tokenizer
is tiktoken/gpt2, loaded validation max token id 50254 < 50257 (no index overflow), 29
validation batches at N=128/B=64, and no silent synthetic fallback with the cache present.
`tests/test_train.py`: tests=8 failures=0 errors=0 skipped=0 (7 passed, 1 xpassed). The
click-run test (`test_train_vfe3_clickrun_importable_and_runs_one_step`) now runs ~256s on
CPU because it exercises a real vocab-50257 forward (a ~1.6 GB logit tensor for one B=64,N=128
step); fast on the CUDA interpreter. A full `max_steps` wikitext-103 run is a genuine GPU job,
not a CPU smoke.
