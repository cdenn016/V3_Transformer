# Ultradeep Codebase Audit Continuation, 2026-07-01

This is the continuation pass for `docs/audits/ultradeep-audit-2026-07-01.md`. It was run in the same isolated audit worktree, `C:\tmp\vfe3-ultradeep-audit-20260701`, branch `audit/ultradeep-20260701`, at commit `7b0f208ca422e1a530db2c20b5fd8af326e79bf9`. The desktop checkout and user WIP were left untouched. This report is documentation-only; no source code, configs, generated run artifacts, or Research vault files were modified.

The continuation added one broad specialist wave, one local source-and-probe wave, and one adversarial verification wave. Six specialists covered information geometry, SPD/gauge geometry, transformer/training behavior, implementation wiring, code quality/security, and Python contracts. Two skeptic agents challenged the highest-risk claims, and a verifier classified each candidate as keep, drop, duplicate, or downgrade against live source. The Research vault was consulted only for context. The vault confirms that the active-inference EFE scorer is specified but still sigma-limited, that the sigma-validation gate failed because the belief covariance is target-blind, and that Regime II covariant transport is intended to be the exact non-flat gauge-covariant route while Yang-Mills and Route A remain open build-out directions.

## Verification

The full test suite was rerun with JUnit XML after the continuation wave. Machine-readable XML result: `tests=1388`, `failures=0`, `errors=0`, `skipped=1`, `time=209.091`. The console summary was `1386 passed, 1 skipped, 1 xpassed, 191 warnings in 209.10s`. The temporary XML at `C:\tmp\vfe3-ultradeep-continuation-20260701-pytest.xml` was removed after parsing.

Targeted probes were also run. A shuffled DataLoader resume probe showed that restoring global RNG does not reproduce the uninterrupted iterator's next shuffled batches: `continuous_remaining=[[28, 16, 48, 0], [40, 8, 20, 24]]`, `resumed_next=[[44, 72, 64, 0], [20, 24, 36, 68]]`, `match=False`. A Regime II covariant detached-oracle probe comparing the current oracle to a direct sigma-threaded autograd reference showed a nonzero sigma-gradient difference: local probe `max_abs_grad_sigma_diff=0.0016948208212852478`; the adversarial check narrowed this to detached/no-grad/direct oracle calls, with `create_graph=True sigma_diff=0` and `create_graph=False sigma_diff=0.370851725`. Config probes showed that `VFE3Config(trust_resume_checkpoint="False")`, `VFE3Config(max_steps=0)`, `VFE3Config(warmup_steps=-1)`, and `VFE3Config(max_steps=4.0)` are accepted.

## Confirmed Continuation Findings

### C1. Resume does not restore the shuffled training data stream

Severity: Medium.

`vfe3/run_artifacts.py:220-232` saves model, optimizer, RNG, config, scaler, and EMA state, but not a DataLoader sampler state, generator cursor, epoch, or batch offset. `vfe3/train.py:702` starts the step range at `start_step`, while `vfe3/train.py:715` creates a fresh `iter(loader)` and `vfe3/train.py:723-726` consumes from that fresh iterator. Real loaders can shuffle because `vfe3/data/datasets.py:168-202` forwards `shuffle` and `generator` into `DataLoader`, and `train_vfe3.py:381-383` builds the train loader with `shuffle=is_train` and a fixed `torch.Generator` when `DATA_SEED` is set. The existing resume tests mask this because `tests/test_checkpoint_resume.py:27-32` uses a constant, `shuffle=False` stream.

Impact: an interrupted run resumed from a mid-training checkpoint is not numerically equivalent to an uninterrupted shuffled run. The skeptic upheld the bug but downgraded severity because resume is opt-in and the active click-run checkpoint interval is terminal in the default script. The bug matters for real recovery checkpoints, long runs, and any attempt to compare resumed training with a sealed uninterrupted trajectory.

Fix: persist sampler/generator state plus batch cursor, or replace DataLoader shuffle with a deterministic global-step sampler. Add a nonconstant shuffled resume-equivalence regression.

### C2. Resume loses prior best-validation model-selection state

Severity: Medium.

`RunArtifacts` initializes `best_val_ppl=float("inf")` and `best_step=None` at `vfe3/run_artifacts.py:58-62`. `save_checkpoint` omits those fields at `vfe3/run_artifacts.py:220-232`. `maybe_save_best` updates only the in-memory object and `best_model.pt` at `vfe3/run_artifacts.py:100-105`. `finalize_run` reloads `best_model.pt` if it exists, but reports the current artifact object's in-memory `best_val_ppl` and `best_step` at `vfe3/run_artifacts.py:613-629`.

Impact: a resumed continuation can forget that a pre-resume validation checkpoint was already best. If no later validation improves, finalization can report the wrong best metadata and can evaluate the current continuation's best or final state rather than the true run-wide best.

Fix: serialize and restore `best_val_ppl`, `best_step`, and the best model state or path in the resume bundle, or rehydrate `RunArtifacts` from the existing run directory before training continues.

### C3. Enabling EMA on resume from a non-EMA checkpoint seeds the average from fresh weights

Severity: Medium.

`train()` constructs `ema = EMA(model, ...)` before `load_checkpoint` at `vfe3/train.py:676-679`. `EMA.__init__` clones current model parameters into `self.shadow` at `vfe3/ema.py:38`. Checkpoint loading then replaces the model weights at `vfe3/run_artifacts.py:288`, but EMA state is loaded only when `ckpt.get("ema_state") is not None` at `vfe3/run_artifacts.py:296-297`. If the checkpoint lacks EMA state and the current config enables EMA, `ema.update(model)` at `vfe3/train.py:741` blends loaded checkpoint weights into a shadow initialized from fresh random weights, and `ema.copy_to(model)` can make that shadow the final model at `vfe3/train.py:943`.

Impact: a shape-preserving config drift from `use_ema=False` to `use_ema=True`, or a legacy checkpoint without `ema_state`, can silently corrupt the averaged evaluation/final weights. This is opt-in and drift-dependent, so medium severity is appropriate.

Fix: reject `use_ema` drift unless an EMA state exists, or initialize/reinitialize EMA after model weights are loaded when no checkpoint EMA state is present.

### C4. Regime II covariant detached oracle drops `dOmega/dsigma`

Severity: Medium.

`regime_ii_covariant` declares that it needs covariance and reads covariance-derived features in `vfe3/geometry/transport.py:386-399` and `vfe3/geometry/transport.py:456,488-497`. The oracle creates differentiable `sigma_q` and `sigma_k` leaves at `vfe3/gradients/oracle.py:100-108`, but the omega-builder contract accepts only `(mu_q, mu_k)` and is called that way at `vfe3/gradients/oracle.py:112-122`. The E-step closure in `vfe3/inference/e_step.py:454-464` then closes over `belief.sigma` and `belief.sigma.detach()` instead of receiving the oracle's sigma leaves.

Impact: detached/no-grad/direct oracle calls in the Regime II covariant path omit the transport's covariance dependence from the sigma gradient. The adversarial pass narrowed the scope: live unrolled training with `oracle_unroll_grad=True` aliases `sigma_q = sigma` at `vfe3/gradients/oracle.py:98-100`, so the live path can still see the captured tensor and did not reproduce the difference. The bug remains real for detached oracle values, diagnostics, direct unit uses, and any future path that expects the detached oracle's gradient value to include `dOmega/dsigma`.

Fix: extend the omega-builder contract to accept `(mu_q, sigma_q, mu_k, sigma_k)` or a structured object containing both live and key-role leaves. Mirror the filtering/smoothing detach semantics for sigma just as the code already does for mu.

### C5. The exact Regime II covariant claim is inconsistent with the accepted diagonal covariance family

Severity: Medium.

The configuration accepts `transport_mode="regime_ii_covariant"` with `family="gaussian_diagonal"`; the default family is declared at `vfe3/config.py:188`, transport registry keys are validated independently at `vfe3/config.py:785-791`, and the local probe accepted that combination. The covariant builder treats diagonal sigma as a supported branch at `vfe3/geometry/transport.py:456,488-490`. The exact end-to-end covariance-law test uses full SPD covariances at `tests/test_regime_ii_covariant.py:366-390`. The admissibility verifier already states the mathematical obstruction: GLK with diagonal Gaussian is not invariant under general congruence at `tests/test_admissibility_verifier.py:51-56`. Meanwhile `vfe3/config.py:1612-1620` warns that full-family Regime II can be numerically risky and recommends diagonal or compact groups.

Impact: the code simultaneously points users toward diagonal covariance for numerical stability and toward full SPD covariance for exact GL congruence. A diagonal covariance cone is not closed under general `S Sigma S.T`; diagonal readout is therefore an approximation under non-orthogonal GL gauges. This is not a crash, but it is a theory-fidelity gap in the advertised exact covariant path.

Fix: make the exactness contract explicit in config and reports. Either require `family="gaussian_full"` for exact GL-covariant Route B, restrict the exact diagonal path to groups/actions that preserve diagonal structure, or present diagonal Regime II covariant as a controlled approximation. Add tests that exercise the config boundary rather than only the full-SPD covariance law.

### C6. Frobenius holonomy diagnostics are frame-dependent under non-orthogonal GL gauges

Severity: Medium.

`vfe3/metrics.py:50-88` and `vfe3/metrics.py:696-724` report Frobenius `norm(H-I)` for triangle holonomy. The gauge-invariant Wilson trace observable exists separately at `vfe3/metrics.py:737-785`, and model diagnostics log both around `vfe3/model/model.py:1579-1585` and `vfe3/model/model.py:1900-1901`. Default reporting still plots Frobenius `holonomy_deviation` in places such as `vfe3/run_artifacts.py:835` and `vfe3/viz/figures.py:998-1000`.

Impact: under non-orthogonal GL frame changes, holonomy transforms by conjugation and Frobenius distance to identity is not conjugation-invariant. The metric is still useful as a frame-fixed diagnostic, but reports should not treat it as a gauge-invariant curvature scalar. The Wilson/eigenvalue route is the right invariant quantity.

Fix: rename or label Frobenius holonomy as frame-dependent, promote Wilson trace/eigenvalue observables for gauge-invariant curvature claims, and make figure captions and pure-path reports distinguish the two.

### C7. Plain Regime II bilinear transport is a gauge-fixed mode, not a covariant mode

Severity: Medium.

Plain `regime_ii` builds edge coefficients from raw coordinates, `mu_i^T W mu_j`, in `vfe3/geometry/transport.py:226-236` and `vfe3/geometry/transport.py:315`. The model owns a fixed-coordinate `connection_W` at `vfe3/model/model.py:223-224`. The tests explicitly pin that nonzero `W` breaks gauge invariance at `tests/test_regime_ii.py:409-435`.

Impact: this is not a newly broken code path; it is a naming/reporting risk. If downstream reports or docs group plain `regime_ii` with the covariant Route B path, users can read a gauge-fixed bilinear edge as an exact gauge-covariant connection.

Fix: label `transport_mode="regime_ii"` as gauge-fixed/non-covariant in reports and use `regime_ii_covariant` or link/charted routes for exact covariance claims.

### C8. Gradient accumulation token-weights non-CE regularizers

Severity: Medium.

The accumulation path computes valid-token counts per microbatch, then applies `w = n_mb / n_tot` to the whole `loss_mb` at `vfe3/train.py:350-356`. The model loss starts from CE but can add non-CE terms: mass phi at `vfe3/model/model.py:910-916`, M-step self-coupling at `vfe3/model/model.py:917-967`, the hyper-prior term at `vfe3/model/model.py:1012`, and gamma coupling at `vfe3/model/model.py:1031`.

Impact: with uneven padding or ignored targets across microbatches, regularizers are scaled by valid target-token fractions rather than by their own batch/state denominator. An all-ignore microbatch contributes no regularizer gradient even though its belief/state regularizers are still defined.

Fix: accumulate CE as a valid-token numerator/denominator, but accumulate non-CE regularizers by their own intended reduction. A clean fix likely requires returning CE and regularizer components separately from `model.forward` or adding a structured loss object for accumulation.

### C9. Visualization belief-bank extractors can drift from forward semantics

Severity: Medium.

The positive control is `belief_ce_bank`, which replays the live `s_e_step` and precision fold at `vfe3/viz/extract.py:160-212`. In contrast, `_encode_one` replays `s_e_step` at `vfe3/viz/extract.py:42-55` but returns raw `log_prior` without folding precision. `belief_bank` directly encodes priors and calls `vfe_stack` without either the `s_e_step` anchor or precision fold at `vfe3/viz/extract.py:245-277`. The forward path folds precision before `vfe_stack` at `vfe3/model/model.py:762-767`.

Impact: UMAP/belief-bank style visualizations can describe a different belief trajectory than the one that produced logits under `precision_weighted_attention=True`, and `belief_bank` also diverges under `s_e_step=True`. This is report/figure semantics, not the training forward path.

Fix: centralize the pre-stack helper used by forward, diagnostics, and extractors. Reuse it in `_encode_one`, `belief_bank`, and trace helpers. Add semantic tests with `precision_weighted_attention=True` and `s_e_step=True`, comparing extractor beliefs to the forward/diagnostics belief path.
### C10. Ablation and scaling resume freshness ignores `max_tokens`

Severity: Medium.

The loader cache includes `max_tokens` correctly in both `ablation.py:1254-1272` and `scaling.py:567-583`. The resume freshness check is the gap: `_cell_cfg_dict` includes `max_steps` but not `max_tokens` at `ablation.py:1296-1313` and `scaling.py:601-610`; `_cell_is_current` compares only dataset and serialized `VFE3Config` at `ablation.py:1559-1591` and `scaling.py:614-626`.

Impact: under resume, a smoke-run cell trained with a capped train split can be reused as current for a later full-data run if the model config and dataset name match. The loader itself would build the right stream for a fresh run, so this is specifically a stale-cache bug.

Fix: persist and compare `max_tokens`, and preferably the effective train-token count or train split hash, in the per-cell metadata used by `_cell_is_current`.

### C11. Run artifact writes are not atomic

Severity: Low-Medium.

`RunArtifacts.save_json` writes directly with `path.write_text` at `vfe3/run_artifacts.py:74-77`. `maybe_save_best` overwrites `best_model.pt` directly at `vfe3/run_artifacts.py:100-105`. `save_checkpoint` writes `step_<N>.pt` directly at `vfe3/run_artifacts.py:198-233`.

Impact: a crash, power loss, or Windows file-lock collision can leave partial JSON, a corrupt checkpoint, or a missing best-model state. This host has already had Windows checkpoint overwrite/open-handle failures in prior runs, so atomic persistence is not just theoretical.

Fix: write to a same-directory temporary file, flush where practical, then use `os.replace`. For Windows, add a narrow retry/backoff for `PermissionError` on best-model replacement.

### C12. Sigma-gate artifact writer trusts `checkpoint_id` as a path component

Severity: Low-Medium.

`write_sigma_gate_artifact` accepts `checkpoint_id` at `vfe3/inference/sigma_gate.py:167-171` and builds `path = os.path.join(out_dir, f"{checkpoint_id}.json")` at `vfe3/inference/sigma_gate.py:182`.

Impact: a checkpoint id containing path separators or parent components can escape the intended directory or create nested paths. This is a local research artifact writer, not a network-exposed API, so severity is low-medium.

Fix: restrict checkpoint ids to a safe slug, or resolve the final path and enforce containment under `out_dir`.

### C13. `generate()` lacks sampler argument validation on the normal path

Severity: Low.

`generate()` validates call-time sampler knobs only when `policy_mode != "none"` at `vfe3/model/model.py:1306-1310`. On the normal path, `range(max_new_tokens)` silently accepts negative values as an empty loop at `vfe3/model/model.py:1312`; logits divide by `temperature` at `vfe3/model/model.py:1320`; `topk(top_k)` is called at `vfe3/model/model.py:1321-1322`; and `top_p` filtering runs at `vfe3/model/model.py:1324-1330`.

Impact: invalid values can fail late, return the prompt unchanged, or produce invalid probabilities. This is a public helper contract issue.

Fix: validate `max_new_tokens >= 0`, `temperature > 0`, `1 <= top_k <= vocab_size`, and `0 < top_p <= 1` at entry. Add invalid-argument tests.

### C14. The Lie exponential clamp can silently turn learned gauge frames into a surrogate

Severity: Low-Medium.

`stable_matrix_exp_pair` rescales large algebra matrices by `scale = (max_norm / mat_norm).clamp(max=1.0)` and multiplies the matrix before exponentiation at `vfe3/geometry/transport.py:689-740`. The prior-bank gauge parameter is unconstrained at `vfe3/model/prior_bank.py:156`, and optimizer groups step it through `vfe3/train.py:90` and `vfe3/train.py:417`. Regime II edge builders have upstream soft caps at `vfe3/geometry/transport.py:337-345`, so the risk is lower there; the broader learned gauge frame path still lacks a runtime clamp diagnostic.

Impact: once the clamp activates, the reported group element is no longer the exact exponential of the learned algebra element. That can blur whether a run used the intended Lie-group map or a stability surrogate.

Fix: expose clamp activation in diagnostics and optionally add a post-step gauge retraction or norm monitor for learned gauge frames.

### C15. Ablation label sanitization can collide distinct cells

Severity: Low.

`_sanitize` replaces `"="`, space, `/`, `\`, `".."`, and `":"` with `_` characters and strips leading punctuation at `ablation.py:1594-1599`. `run_dir = sweep_dir / _sanitize(label)` at `ablation.py:1664`.

Impact: distinct raw labels such as `a=b`, `a b`, and `a/b` collapse to the same run directory. A resumed or additive sweep can overwrite or merge distinct cells.

Fix: append a stable short hash of the raw label, or reject collisions before running. Persist the raw label in the marker and validate it on resume.

### C16. `sigma_gate_measure.py` contains a machine-specific absolute checkpoint path

Severity: Low.

`sigma_gate_measure.py:31` hardcodes a checkpoint under `C:\Users\chris and christine\Desktop\V3_Transformer\...`.

Impact: the click-to-run script is not portable to a fresh worktree, another clone, or another machine. It also risks accidentally measuring a stale checkpoint outside the current checkout.

Fix: make the checkpoint field empty/required by default, or resolve a repo-relative path from `Path(__file__).resolve().parent`.

### C17. Ablation range expansion silently returns empty values for sign-mismatched ranges

Severity: Low.

`_expand_range` rejects only zero step at `ablation.py:1195-1201`. It computes `n = int(round((stop - start) / step))` and loops over `range(n + 2)` at `ablation.py:1205-1206`. If `start < stop` with a negative step, or `start > stop` with a positive step, the loop can be empty without an error.

Impact: a malformed sweep can run no cells without a clear config error.

Fix: reject `(stop - start) * step < 0` unless `start == stop`, and assert the expanded values list is nonempty.

### C18. Public helper signatures violate the repo argument-order convention

Severity: Low.

The repo mandates ordering tensor args, numeric args, defined bools, Optional args, and `**kwargs`. Concrete violations include `fisher_trace(..., diagonal: Optional[bool] = None, eps: float = ...)` at `vfe3/metrics.py:244-249`, which places Optional before a defined float, and `generate(... top_k: Optional[int], top_p: Optional[float], greedy: bool = False)` at `vfe3/model/model.py:1269-1279`, which places Optional before a defined bool.

Impact: this is consistency debt rather than a behavioral bug. It matters because the project uses signature order as an explicit review constraint and because inconsistent surfaces make registry extension harder to audit.

Fix: reorder keyword-only parameters while preserving names at call sites, and add a lightweight signature-order lint if the convention is meant to remain mandatory.

## Addenda To Prior Finding F12

These are not counted as new continuation findings because the first audit already filed F12 as the config/registry validation footgun cluster, but they are source-confirmed subcases that should be included in that repair pass.

`trust_resume_checkpoint` is annotated as bool at `vfe3/config.py:567`, but the config validation block around `vfe3/config.py:1768-1778` does not type-check it. `load_checkpoint` uses `bool(getattr(cfg, "trust_resume_checkpoint", False))` at `vfe3/run_artifacts.py:276`, so the string `"False"` enables the unsafe `weights_only=False` fallback at `vfe3/run_artifacts.py:287-288`.

`max_steps` and `warmup_steps` are fields at `vfe3/config.py:537-538`, but the validation block checks batch, grad accumulation, log, eval, and checkpoint intervals without validating those two fields. The scheduler consumes both at `vfe3/train.py:222-224`, the training loop consumes `n_steps` through `range(start_step, n_steps)` at `vfe3/train.py:702`, and run artifacts cast `cfg.max_steps` into `tokens_seen` at `vfe3/run_artifacts.py:674`. Probes accepted `max_steps=0`, `warmup_steps=-1`, and `max_steps=4.0`.

`bch_pe_order` is declared at `vfe3/config.py:164` and `pos_phi_compose` is registry-validated at `vfe3/config.py:1256`, but no numeric guard rejects `bch_pe_order=0`. `compose_bch` starts with `Z = X + Y` and adds corrections only under `order >= 1` at `vfe3/geometry/lie_ops.py:244-312`, so `pos_phi_compose="bch"` with order zero becomes additive while still labeled BCH.

## Dropped Or Refuted Candidates

The Fisher-trace candidate was dropped. `fisher_trace` documents and returns `tr(Sigma^-1)/2` at `vfe3/metrics.py:251-265`, and the relevant figure labels include `/2`, so the half scale is not a hidden mislabeled mean-block Fisher trace.

The unsafe default checkpoint-load concern was not refiled. The default load path uses `weights_only=True` at `vfe3/run_artifacts.py:278`, and the unsafe fallback is gated; the remaining problem is the bool-coercion validator gap recorded under F12.

The direct-link transport propagation suspicion was refuted by the implementation/verifier waves. Link modes forward `connection_L` into the E-step and tests assert nonzero gradients in the direct-link path. The Laplace family protocol suspicion was also refuted: the optional family has concrete implementations and focused tests. Token/target alignment in `TokenWindows` was checked and not refiled.

## Missing Feature Inventory From Code Plus Wiki

The active-inference sigma route is not complete. The Research vault records the 2026-06-29 sigma-gate failure: `tr(Sigma_q)` is anti-correlated with realized cross-entropy and near-static because the belief E-step is target-blind. The missing feature is an observation-precision term in the E-step covariance gradient, not another reporting toggle. Without that, sigma-derived epistemic or ambiguity arms should remain gated off or reported-only.

The exact non-flat gauge route needs a clean contract. The vault records `regime_ii_covariant` as the default-off exact gauge-covariant Route B and notes open Yang-Mills/Route A work. The code now needs that contract reflected in config: exact GL covariance should either require full SPD covariance, restrict to diagonal-preserving groups/actions, or explicitly present diagonal covariance as an approximation. Wilson/eigenvalue diagnostics should be promoted for invariant curvature claims.

Resume/reproducibility should become a first-class feature rather than a partial checkpoint loader. A run-wide checkpoint should include sampler/cursor state, best-validation state, EMA state semantics, and atomic artifact writes. The existing loader and artifact code is close, but the current resume surface is not a sealed experiment-resume primitive.

Reporting extractors need a shared forward-preparation helper. The code already has one correct positive control in `belief_ce_bank`; the rest of the visualization stack should reuse the same pre-stack semantics so figures, calibration gates, diagnostics, and logits describe the same belief trajectory under active toggles.

Sweep provenance should carry all data-stream knobs, including `max_tokens`, raw label identity, and preferably effective token count or data hash. This is needed before long ablation/scaling resumes can be treated as sealed empirical evidence.

