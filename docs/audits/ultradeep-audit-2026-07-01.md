# Ultradeep Codebase Audit, 2026-07-01

This audit was run from a fresh isolated worktree at `C:\tmp\vfe3-ultradeep-audit-20260701`, branch `audit/ultradeep-20260701`, based on `origin/main` commit `7b0f208ca422e1a530db2c20b5fd8af326e79bf9` (`Merge pull request #155 from cdenn016/feat/regime-ii-link-impl-20260629`). The desktop checkout was left untouched. The sweep enumerated 296 repository files and focused the manual source verification on the executable seams that the wiki, specs, tests, and parallel agent waves identified as highest risk.

The Research vault was consulted for the VFE program context, active-inference/EFE expectations, sigma-gate status, and masked-retrieval plan. The most relevant vault and repo-local context was: `Research/index.md`, `wiki/projects/VFE Transformer Program.md`, `wiki/projects/VFE Transformer Research Directions (2026-06-21).md`, `wiki/concepts/Active Inference.md`, `wiki/concepts/Expected Free Energy.md`, `sources/runs/2026-06-29-sigma-gate-fail-and-collapse.md`, `sources/runs/2026-06-28-active-inference-efe-policy-scorer-spec.md`, and the repo docs under `docs/research/active-inference/`.

## Waves Run

Wave 0 established a clean `origin/main` worktree, read repo instructions, checked the research wiki, and mapped code/tests/docs. Wave 1 ran parallel code-quality, runtime, implementation-wiring, performance, typing/config, and feature-drift auditors. Wave 2 ran targeted variational/EFE, gauge-theory, and numerical-analysis auditors. Wave 3 challenged the highest-severity findings adversarially. The EFE horizon finding was upheld but downgraded to medium because it is opt-in/direct-scorer semantics rather than a default-path training failure. The nonfinite-gradient training finding was upheld as high. The `sigma_max` SPD finding was upheld but downgraded to medium because it requires an explicit invalid config override.

## Verification

Pytest was run with a JUnit XML output and the XML was parsed before cleanup. Machine-readable result: `tests=1388`, `failures=0`, `errors=0`, `skipped=1`, `time=228.151`. The pytest console summary also reported `1386 passed, 1 skipped, 1 xpassed, 191 warnings in 228.16s`.

Two small probes were run for high-risk numerical claims. A finite scalar loss with a manually nonfinite gradient produced `skip_step=False`, `grad_finite=False`, and `param_finite_after_step=False` after `AdamW.step()`. A `sigma_max` probe showed that `VFE3Config(sigma_max=1e-9)`, `VFE3Config(sigma_max=-1.0)`, and `VFE3Config(sigma_max=nan)` are accepted; the diagonal retraction then returns values below `eps`, negative values, or NaNs respectively.

## Confirmed Findings

### F1. Nonfinite gradients can still poison AdamW when the scalar loss is finite

Severity: High.

`vfe3/train.py:400` records only `math.isfinite(step_loss)`, `vfe3/train.py:410-411` sets `skip_step` only from scalar loss finiteness on the disabled-scaler path, `vfe3/train.py:412-413` calls `clip_grad_norm_` without `error_if_nonfinite=True`, and `vfe3/train.py:414-418` still steps the optimizer when `skip_step` is false. The current regression around nonfinite loss does not cover the reachable case where the loss value is finite but at least one parameter gradient is NaN or Inf.

Impact: a single optimizer step can silently corrupt parameters and AdamW moments. This is not theoretical: the local probe produced a NaN parameter from a finite scalar loss plus NaN gradient.

Fix: after unscale and before clipping/stepping, check all gradients for finiteness. If any gradient is nonfinite, zero gradients and skip optimizer, scaler update where appropriate, scheduler, EMA, and barycenter updates that assume a real optimizer step. Add a regression where scalar loss is finite but a parameter gradient is NaN.

### F2. `sigma_max` can violate the SPD invariant under accepted configs

Severity: Medium.

`sigma_max` is declared at `vfe3/config.py:378`, but `VFE3Config.__post_init__` validates nearby numerical fields without validating `sigma_max` against finiteness or `eps` (`vfe3/config.py:596-601`). The diagonal and full SPD retractions clamp with `max=sigma_max` at `vfe3/geometry/retraction.py:133`, `:191`, `:288`, and `:343`. Invalid caps therefore reach the covariance retractions.

Impact: accepted configs can make retractions return covariance values below `eps`, negative values, or NaNs. The default click-to-run configs use positive caps, so this is not an active default-run corruption, but it is a real accepted-config invariant break.

Fix: validate `sigma_max is None or math.isfinite(sigma_max) and sigma_max >= eps` in config, and defensively reject invalid caps in retraction helpers. Add tests for `sigma_max < eps`, negative, and NaN.

### F3. `efe_rollout` H>1 scores only the terminal predicted outcome, not the horizon sum

Severity: Medium.

The local EFE spec defines `G(pi)` as a horizon sum at `docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md:31-33`. The executable H>1 scorer accepts candidate sequences at `vfe3/inference/policy.py:381-390`, but `_efe_score` calls `_rollout_predictive` once and sums score terms, not time steps, at `vfe3/inference/policy.py:312-318`. `_rollout_predictive` reads `logits[:, -1, :]` at `vfe3/inference/policy.py:255-257`, and the cache path reads `logits[:, -1, :]` at `vfe3/inference/belief_cache.py:189-190`.

Impact: direct H-step harnesses receive a terminal-outcome scorer while the local spec and active-inference terminology imply a per-step expected-free-energy sum. This does not affect default `policy_mode='none'` or the current one-step ring experiment, so the adversarial challenge downgraded it from high to medium.

Fix: either return per-horizon predictive distributions and sum risk/ambiguity for `tau=1..H`, or rename/gate the current scorer as terminal-outcome rollout scoring and update docs/tests accordingly.

### F4. Generic `generate()` accepts `efe_rollout` configs but builds one-token candidates

Severity: Medium.

`VFE3Config` validates `policy_mode='efe_rollout'` when `policy_horizon > 1` at `vfe3/config.py:1408-1411`. The generic model generation path, however, always builds `candidates = topk.unsqueeze(-1)` at `vfe3/model/model.py:1366-1367`, a length-one candidate menu. `_policy_efe_rollout` requires `candidate length == horizon` at `vfe3/inference/policy.py:408-412`, so a config can validate and then fail at generation dispatch. `generate_efe.py:19-20` documents that its helper script does not support `efe_rollout`, but the core config/model path still permits the invalid pairing.

Impact: this is an opt-in API reachability bug, not a default-path issue. It matters because the config says the mode is valid, while the model's generic generation branch cannot supply a valid H-token policy menu.

Fix: reject `policy_mode='efe_rollout'` for generic `generate()` unless an H-step candidate generator is implemented, or implement a real `(B, Kp, H)` candidate generator with matched-compute accounting.

### F5. `sigma_mc` and its validation flag have no executable consumer

Severity: Medium.

`policy_sigma_ambiguity_validated` checks for a PASS artifact at `vfe3/config.py:1438-1445`, but `sigma_mc` is registered as an ambiguity mode that unconditionally raises at `vfe3/inference/policy.py:210-221`. The model policy dispatch at `vfe3/model/model.py:1372-1375` does not pass an ambiguity mode or a validation flag, and tests pin the current raising behavior at `tests/test_efe_scorer.py:164-165`.

Impact: a PASS artifact can validate a config flag, but no generation or task scorer can actually use a sigma-derived ambiguity term. This is consistent with the current sigma-gate failure in the wiki, but the executable surface overstates what can be unlocked.

Fix: either add a real `policy_ambiguity_mode` seam whose `sigma_mc` implementation checks the validated artifact, or remove/rename the fields as precondition records until a consumer exists.

### F6. Full-covariance chunked decode ignores Cholesky failure status

Severity: Medium.

The full-covariance chunked decode path calls `safe_cholesky` at `vfe3/model/prior_bank.py:482` and discards the `ok` mask. The resulting log determinant is used in `kl_v` at `vfe3/model/prior_bank.py:807-808`. The dense path maps degenerate non-PD cases to invalid logits, while the chunked invariant path can return finite logits for an invalid covariance.

Impact: under invalid full-covariance inputs, dense and chunked decode semantics diverge. This should be rare if the SPD retractions hold, but F2 shows that invalid covariance values can be admitted under bad accepted configs.

Fix: carry the `ok` mask through chunked full-cov decode and map failed Cholesky positions to `+inf` KL or `-inf` logits, matching dense semantics. Add a non-PD full-cov regression.

### F7. Model-channel gamma and `s_e_step` use flat transport under non-flat belief transports

Severity: Medium.

The belief E-step forwards the active transport and RoPE arguments into `vfe_stack` at `vfe3/model/model.py:768-776`. The model-channel refinement path explicitly passes `transport_mode="flat"` at `vfe3/model/model.py:639`, and `_gamma_energy` builds model-channel transport with `transport_mode="flat"` at `vfe3/model/model.py:1150`. Config validation allows non-flat belief transport and nonzero gamma/model-channel settings independently (`vfe3/config.py:790-791`, `vfe3/config.py:935-940`).

Impact: a run can combine non-flat belief transport with flat model-channel comparison. That may be intentional for a limited extension, but it should not be described as sharing the active connection or as a fully covariant s-fiber path.

Fix: add a model-channel transport seam, or reject/warn on `lambda_gamma > 0` / `s_e_step=True` with non-flat belief transport until the s-channel transport law is explicit.

### F8. The pure-path report omits gauge-affecting toggles

Severity: Medium.

`_pure_path_report` defines `on_pure_path` from seven flags at `vfe3/run_artifacts.py:754-764`, but it omits several behavior-affecting toggles from `config_toggles` at `vfe3/run_artifacts.py:766-775`, including `gauge_transport`, `pos_rotation`, `rope_full_gauge`, `rope_on_value`, `lambda_gamma`, and `s_e_step`. RoPE is forwarded into the belief stack at `vfe3/model/model.py:775-776`, and gamma uses the model-channel path in F7.

Impact: the report can say `on_pure_path=True` while gauge or model-channel settings materially change the executed path. This is a reporting/certification bug, not a training bug.

Fix: split the certificate into explicit axes, for example canonical-free-energy/decode purity and gauge-equivariant path purity, and include every toggle that changes either axis.

### F9. Reporting extractors and default figures materialize full logits/probabilities

Severity: Medium.

`vfe3/viz/extract.py:139` and `:199-203` call `model(tokens)` and materialize `(B,N,V)` logits/probabilities for CE, confidence, and calibration. `vfe3/viz/extract.py:873-876` computes a full softmax over `(B,N-1,V)` for vocab statistics. `finalize_run` calls `generate_figures` by default when `generate_figures` is truthy at `vfe3/run_artifacts.py:726-730`.

Impact: training has fused/chunked CE paths, but reporting can still allocate full-vocab tensors and replay multiple model passes. This can surprise large runs, especially on the RTX 5090 where the training path may be tuned but finalization/reporting can still become the memory peak.

Fix: add streaming/chunked extractor APIs for CE, confidence, calibration, and vocab statistics. Make heavy figures tiered or require an explicit large-run opt-in.

### F10. Generation and Regime II still have unguarded expensive paths

Severity: Medium.

`generate()` performs a full `self.forward(context)` per emitted token at `vfe3/model/model.py:1311-1339`, then grows the sequence with `torch.cat`. `_build_regime_ii` builds dense all-pairs tensors such as `delta_mat` and `exp_delta` over `(B,N,N,K,K)` at `vfe3/geometry/transport.py:315-348`. The covariant Regime II variant already has query chunking at `vfe3/geometry/transport.py:371-383` and `:462-466`, which shows the unchunked plain Regime II path is avoidable.

Impact: this is primarily performance/memory risk. It does not change outputs on small runs, but it can dominate long-context generation and non-flat transport experiments.

Fix: add an incremental generation cache or last-position decode path, accumulate generated tokens without repeated `torch.cat`, and port query-axis chunking or memory budget guards to `_build_regime_ii`.

### F11. Best-effort validation/reporting can still stale or abort outside the intended catch points

Severity: Medium.

If validation diagnostics fail, `last_val_diag.update(...)` is skipped and the old values are retained at `vfe3/train.py:805-808`. Attention-map helpers are described as best-effort, but the expressions `model.attention_maps(tokens)` and `model.gamma_attention_maps(tokens)` are evaluated before entering the save helper at `vfe3/train.py:813-816`.

Impact: diagnostics can silently carry forward stale values after a replay failure, and attention/gamma map generation can still abort training if the model replay throws before the helper's internal catch.

Fix: reset diagnostics to explicit NaNs on failure, and wrap the model replay plus save call in the same try/except block.

### F12. Config and registry validation has several lower-severity footguns

Severity: Low to Medium.

Examples confirmed in source: several numeric checks use bare comparisons rather than finite validators (`vfe3/config.py:596-601`), `generate_figures` is used through truthiness at `vfe3/run_artifacts.py:726`, registry decorators such as `vfe3/alpha_i.py:19-29`, `vfe3/attention_prior.py:34-37`, and `vfe3/inference/policy.py:48-51` overwrite duplicate keys silently, and checkpoint config loaders drop unknown fields in `generate_efe.py:78-80`.

Impact: these are not current default-run failures, but they reduce fail-closed behavior for a codebase that relies heavily on config-selected seams.

Fix: centralize finite scalar/list validators, validate bool fields with `type(value) is bool`, reject duplicate registry keys unless an explicit override is requested, and fail on unknown checkpoint config fields except for named legacy migrations.

## Missing Feature and Buildout Inventory

The wiki/spec comparison points to these missing or intentionally deferred buildouts. These should not all be implemented blindly; several are gated by theory and validation.

1. A real H-step EFE objective if `efe_rollout` is meant to claim canonical expected free energy. The current code scores the terminal predicted outcome, not the horizon sum.

2. A closed-loop masked-retrieval runner with matched-compute arms. `vfe3/inference/masked_retrieval.py` provides task/data/checkpoint scaffolding (`sample_batch`, `predictive_adequacy`, `train_checkpoint`), while `docs/research/active-inference/2026-06-29-masked-retrieval-task-design.md:69-84` describes a receding-horizon runner and arms. The closeout doc explicitly says the runner and arms were deliberately not built at `docs/research/active-inference/2026-06-29-v3-active-inference-closeout.md:103-106`, so this is deferred work, not an accidental runtime bug.

3. A sigma-derived ambiguity implementation. The sigma gate currently failed in the vault context, and `sigma_mc` is non-executable. Do not promote it until sigma is actually calibrated or the covariance update gains a justified observation-precision channel.

4. Observation/data precision in the live E-step. `free_energy(log_likelihood=...)` is documented and implemented as an optional stub at `vfe3/free_energy.py:341` and `:401-402`, while production callers in `vfe3/inference/e_step.py:299-302` and `vfe3/gradients/oracle.py:141-146` do not supply it. The closeout doc says naive wiring is not recommended. If built, it needs an explicit no-leakage design, not a one-line call.

5. Model-channel transport semantics. If `s_e_step` or `lambda_gamma` are used under non-flat belief transport, the s-fiber needs its own explicit transport registry or a fail-closed warning.

6. Large-run reporting and Regime II memory guardrails. Reporting should use streaming/chunked extraction, and plain Regime II should inherit the query chunking already present in the covariant variant.

7. A pure-path certificate that separates mathematical axes. The repo should distinguish free-energy/decode purity, gauge-covariance purity, reporting purity, and experimental ablation status rather than compressing them into one `on_pure_path` boolean.

## Checked and Not Filed

The ring task evaluation wrapper is already under `@torch.no_grad()` at `vfe3/inference/ring_task.py:257`, so the earlier suspicion that it builds training graphs was refuted.

The float64 islands inspected in Gaussian Renyi and transport code are narrow precision safeguards rather than a blanket violation of the float32 policy. They should remain documented as precision islands, not treated as accidental dtype drift.

AMP/bf16 is opt-in (`amp_dtype=None` by default), and sensitive decode/E-step sections opt out where needed. This does not appear to be a current precision regression.

Bare direct-link transport is a reachable non-covariant mode, but a charted covariant replacement exists at `vfe3/geometry/transport.py:632-674`. Keep the bare mode out of covariant/pure-path claims rather than treating its existence as an automatic bug.

`generate_efe.py` uses `torch.load(..., weights_only=False)` at `generate_efe.py:63` and `:71`, but the script states it is loading the user's own trusted checkpoints. This is a security hardening item rather than a core model bug; `vfe3/run_artifacts.py` uses `weights_only=True` by default and requires an explicit trusted-resume fallback.

## Recommended Fix Order

1. Fix F1 immediately: finite-gradient gate plus scheduler/EMA skip semantics, with a regression for finite loss plus NaN gradient.

2. Fix F2 and F6 together: validate `sigma_max` and carry Cholesky failure masks through full-cov chunked decode.

3. Fail closed around EFE: either make `generate()` reject `efe_rollout`, or build real H-token candidates; then decide whether `efe_rollout` is terminal-outcome scoring or true horizon-summed EFE.

4. Clean up sigma/ambiguity semantics: do not expose a config state that appears to unlock `sigma_mc` unless an executable consumer exists and the gate has passed.

5. Repair reporting certificates and heavy extractors before the next large run: pure-path toggles, stale diagnostics, attention-map exception scope, streaming extractors, and Regime II memory guards.


