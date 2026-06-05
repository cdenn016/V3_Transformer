# Triage of the Codex deep audit (audit-2026-06-05-new.md / branch codex/deep-audit-20260605)

Four expert investigators (two implementation-engineers, one variational, one gauge-theorist)
verified each of Codex's eight findings against the current source. Every verdict below is
primary-source-confirmed with path:line evidence; severities are re-judged under the project's
"a theoretically-pure path must EXIST under appropriate toggles" policy (default-config purity
is not required). Two findings were overstated and two carry false-positive sub-claims; four are
real and actionable, none is a correctness bug in the training loop.

## Verdict table

| # | Finding | Verdict | Codex sev | Real sev | Impact |
|---|---------|---------|-----------|----------|--------|
| 1 | val/test loaders shuffle + drop the tail | PARTIAL | high | medium | reported metrics |
| 2 | free-energy descent artifact mixes domains/reductions | CONFIRMED | medium | medium | figure only |
| 3 | learned lambda_beta ignored by descent figure scaling | CONFIRMED | medium | medium | figure only |
| 4 | registry seams blocked by hardcoded config lists | PARTIAL | medium | low | modularity contract |
| 5 | model-channel coupling inert/detached | PARTIAL | medium | low | docs/scoping |
| 6 | Regime-II connection_W not gauge-equivariant | CONFIRMED | medium | medium | undisclosed opt-in impurity |
| 7 | holonomy diagnostic uses low-index triangle prefix | CONFIRMED | low-med | low | diagnostics only |
| 8 | banner prints config-level tau (wrong for single-block) | CONFIRMED | low | low | reporting only |

## Real and actionable

Finding 1 (medium, was high). `make_dataloader` hardcodes `drop_last=True` with `shuffle`
defaulting True (datasets.py:134,147), and `_select_loader` (train_vfe3.py:237-239) passes no
split override, so validation and test run `shuffle=True, drop_last=True`. The `evaluate`
docstring (train.py:239) promises "a partial last batch" that `drop_last=True` discards. At
seq_len=128, batch_size=64 the held-out test number is computed on ~97% of wikitext-103 (62 of
1918 windows, ~3%, dropped), and the dropped tail is a different random subset each eval, so the
reported single-seed metric wobbles run-to-run. Codex overstated the severity: the "zero batches
-> total_tok=0 -> ce=0 -> ppl=1 artificial-perfect metric" case is unreachable on any shipped
non-synthetic path, because `MAX_TOKENS` caps only the train split (train_vfe3.py:237) and the
synthetic loader grows n to avoid zero batches. Fix: make `_select_loader` split-aware
(train: shuffle=True, drop_last=True; val/test: shuffle=False, drop_last=False), thread a
`drop_last` param through `make_dataloader`, and add a regression test that a val/test loader
consumes the full window count in a deterministic order.

Finding 6 (medium, undisclosed opt-in impurity). Under the opt-in `transport_mode="regime_ii"`,
`connection_W` is an unconstrained `(n_gen, K, K)` nn.Parameter (model.py:110-111), and the edge
factor is `E_ij = exp(delta_ij . G)` with `delta_ij^a = mu_i^T W^a mu_j` a raw bilinear
contraction (transport.py:178-181). The gauge-theorist corrected Codex's framing: the required
property is gauge-INVARIANCE of the edge factor (the vertex factors exp(phi_i), exp(-phi_j)
already carry the full g_i(.)g_j^{-1} conjugation), not covariance. Invariance requires
g_i^T W^a g_j = W^a for all g in the group, whose only constant solution is W = 0. So a trained
nonzero `connection_W` breaks strict gauge equivariance, exactly at the same "equivariant at zero
init, drifts under training" footprint as the head mixer, verified empirically (break grows
monotonically with ||W||: 0.0 at W=0, 62.3 at ||W||~1.0 against ||E||~9.6). This is NOT covered
by existing caveats: CLAUDE.md names only `use_head_mixer` and lists Regime-II merely as an
NN-exception; no manuscript or test asserts regime_ii equivariance, and test_regime_ii.py has no
equivariance test. Fix: add the head-mixer-style caveat to CLAUDE.md (trained regime_ii / nonzero
connection_W breaks strict gauge equivariance; only W=0 is invariant) and a property test in
test_regime_ii.py asserting the edge factor is invariant at W=0 and deviates as ||W|| grows.
Codex's suggested remedies ("constrain W to an invariant tensor family", "W -> Ad(g) W") are both
off — the only constant invariant is W=0, so constraining to an invariant family is vacuous.

Findings 2 and 3 (medium, figure-only, coupled — same descent figure). The free-energy descent
artifact mixes incompatible quantities: `val_ce` is a token-weighted mean over the validation
corpus, while `self_coupling`/`belief_coupling`/`attention_entropy`/`free_energy_total` come from
`model.diagnostics(tokens)` on the live training batch (train.py:396-419), and `free_energy_terms`
returns sums over one sequence (metrics.py:124-132). Panel A (figures.py:406-416) stacks the four
terms including CE; Panel B plots `free_energy_total` = `self + lambda_beta*(belief + entropy)`,
which excludes the CE data term — so one panel includes CE and the other does not. Separately
(Finding 3), on a `learnable_lambda_beta=True` run the per-row learned `exp(log_lambda_beta)` IS
recorded (train.py:416-418) but `plot_free_energy_descent` is handed the static `cfg.lambda_beta`
(run_artifacts.py:278-280), so Panel A scales by the frozen config value while Panel B's total
embeds the learned one. Both are confined to one descriptive PNG: `diagnostics` is
`@torch.no_grad` (model.py:583) and never reaches the loss. Fix: compute the figure's F terms on
the same split with the same reduction as val_ce (or relabel the figure as descriptive and drop
the "closes to runtime F" claim), and pass row-wise `hist["lambda_beta"]` when present.

## Real but low

Finding 4 (low, was medium; with a false-positive sub-claim). Four seams validate against static
tuples that currently equal their registries (gauge_group vs _GROUPS, alpha_mode vs _ALPHAS,
attention_prior vs _PRIORS, norms vs _NORMS; config.py:317,420,447,566), a contract-consistency
deviation from the six sibling seams that already validate against the registry. Nothing is
unreachable and every pure path exists, so this is low, not "blocked/medium." Codex's headline
"most-visible example" — `decode_mode="linear"` cannot pass validation — is a FALSE POSITIVE:
`linear` is reached through the `use_prior_bank=False` gate (prior_bank.py:238), and admitting it
via `decode_mode` would collide with the use_prior_bank rank cross-check; `encode_mode="gauge_fixed"`
is likewise an intentional NotImplementedError stub. PR #27 already reached the same decode
conclusion. Fix: replace the four exactly-matching static tuples with `tuple(sorted(_registry))`;
keep decode/encode as explicit post-registry second-gates.

Finding 5 (low, mostly already-disclosed). All of Codex's factual sub-claims hold: under
`gamma_coupling>0` the model-channel term is loss-only (model.py:520), detaches phi
(`out.phi.detach()`, model.py:502), and uses flat `compute_transport_operators` that ignores the
active transport_mode/connection_W/rope. But the normative thrust ("mischaracterized as the full
channel / under-disclosed") is largely a FALSE POSITIVE: the code already names itself exactly
what Codex asks (config.py:147-159, model.py:471-490 call it a detached, predictively-inert,
reduced-envelope s-table regularizer with s->q deferred). And the "no predictive channel" claim
is scoped to the default `prior_source="token"`; under the opt-in `prior_source="model_channel"`
the same s tables back the belief prior and DO feed predictions (verified: logits move, CE delta
6.2). The pure path (gamma=0) exists and is disclosed. Minimal/no fix; optionally scope the
inertness wording to "under prior_source=token" and share one transport between the beta and gamma
blocks under regime_ii.

Finding 7 (low, diagnostics only). `metrics.holonomy_deviation` (logged at model.py:688) builds
triples by row-major nested loops and at the default N=128, max_triangles=512 covers only anchor
i=0, j in {1..5} — one token's local neighborhood. A better estimator
`holonomy_deviation_sampled` (metrics.py:547-585, random triples + bootstrap CI) exists but is
wired only into figures. No training impact, and on the default flat path every triangle closes
(H~I) so the biased sample still reads ~0; it misleads only under regime_ii. Fix: point the
periodic diagnostic at the sampled estimator (or log both).

Finding 8 (low, reporting only; distinct from PR #27). The entry banner prints `cfg.tau`
(train_vfe3.py:216), the per-head convenience `kappa*sqrt(K/n_heads)`, while the active attention
temperature is `attention_tau(kappa, irrep_dims) = kappa*sqrt(K)` for single-block groups
(glk/so_k/sp), understating by sqrt(n_heads); on the default block_glk they coincide. The in-package
banner (train.py:446) already prints the group-aware value. Reporting-only (the sole runtime reader
of cfg.tau is this banner). Distinct from the PR #27 gamma-tau fix, which corrected the
loss-affecting tau_gamma in the model-coupling block — a different field, different file. Fix: print
`attention_tau(cfg.kappa, model.group.irrep_dims)` in the banner.

## Fixes applied (2026-06-05, branch fix-codex-audit-2026-06-05)

The user selected F1, F4, F6, F7, F8 for repair (F2/F3 deferred). All applied with TDD where a
behavior changed; full suite 583 passed / 0 failures (junit XML).

- F1: `make_dataloader` gained a `drop_last` parameter (default True, train-regime); `_select_loader`
  now requests `shuffle=False, drop_last=False` for validation/test and `True/True` only for train
  (datasets.py, train_vfe3.py). Tests: `test_data.py::test_make_dataloader_eval_keeps_tail_and_is_sequential`,
  `test_train.py::test_select_loader_is_split_aware`.
- F4: the four exactly-matching static tuples (gauge_group, alpha_mode, attention priors, norms) now
  validate against `tuple(sorted(_REGISTRY))`, matching the transport/retraction siblings; the orphaned
  `_VALID_*` constants were removed; decode/encode stay explicit second-gates. Tests:
  `test_config.py::test_gauge_group_validation_reads_registry_not_static_list`,
  `test_config.py::test_decode_mode_linear_stays_a_rejected_second_gate`.
- F6: CLAUDE.md now records the trained-`connection_W` gauge-equivariance break (only W=0 invariant),
  parallel to the head-mixer caveat; characterization test
  `test_regime_ii.py::test_regime_ii_edge_factor_breaks_gauge_invariance_for_nonzero_W`.
- F7: the periodic holonomy diagnostic (model.py) switched to `holonomy_deviation_sampled(...)["mean"]`
  (seeded random distinct triples); the dict key is unchanged so existing diagnostics tests hold.
- F8: the train_vfe3.py banner prints `attention_tau(cfg.kappa, model.group.irrep_dims)` (group-aware),
  matching the in-package banner.

Deferred (F2/F3): the free-energy descent figure's domain/reduction/panel mismatch and the learned
lambda_beta scaling — figure-only, no training impact; left for the user.

## Non-findings (Codex's own, confirmed)

No hidden nn.Linear/MLP/activations/CLI parsing; the canonical beta-entropy path is coherent
(lambda_beta kept outside the softmax); the full-cov decode/retraction are opt-in pure paths, not
bugs. Verification run cited by Codex: 573 tests, 0 failures (matches the current suite, 575 with
the two new Renyi tests on the fix-renyi branch).
