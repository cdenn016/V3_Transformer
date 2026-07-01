# Ultradeep Audit 2026-07-01 — Verified Fix Plan

Resume doc for implementing the fixes for `docs/audits/ultradeep-audit-2026-07-01.md`
(F1–F12) and `docs/audits/ultradeep-audit-continuation-2026-07-01.md` (C1–C18 + F12
addenda). Produced by an 8-agent verification workflow that confirmed every finding
against live source at HEAD and specced the minimal surgical fix + regression tests.

Branch for the work: `audit-fixes-2026-07-01` (off main tip ea1a947). The uncommitted
`scaling.py:~521` ROUTES `grow_K_GL10` grid is intentional WIP — DO NOT touch that line.

## Status: verification COMPLETE, implementation NOT STARTED (no source edits made yet).

## Scope decision
All 35 sub-findings CONFIRMED against live source. Nine have an "ideal" fix that is a
large theory-gated buildout the audit says NOT to build blindly — implement the
fail-closed / guard / label SAFE VARIANT for these instead (marked `deferred=True`):
C1 (resume shuffle-stream: warn only), C8 (grad-accum regularizer weighting: warn only),
F3 (efe_rollout horizon sum: relabel terminal-outcome + doc), F5 (sigma_mc ambiguity:
keep fail-closed + doc), F7 (s-channel non-flat transport: config warn), C5 (diagonal
regime_ii_covariant exactness: config warn + report label), C14 (Lie-exp clamp: opt-in
default-OFF diagnostic), F9 (streaming extractors: memory-budget guard + force_large_figures
opt-in), F10 (generate KV cache: warn; PLUS port existing covariant chunking into
_build_regime_ii which IS surgical). Everything else is a full surgical fix.

## Conflict-free implementation partition (each file owned by exactly ONE worker)
- A train/artifacts: vfe3/train.py, vfe3/ema.py, vfe3/run_artifacts.py — F1, F11, C1, C2,
  C3, C8, C11, F8, F9(finalize half), C5(report flag), C6(title), C7(report field).
- B config: vfe3/config.py, vfe3/alpha_i.py, vfe3/attention_prior.py, vfe3/geometry/lie_ops.py
  — F2-config, F12-bool, F12-steps, F12-bch, F12-registry(non-policy), F7(warn), C5(warn),
  F5(comments), F3(policy_horizon doc), F9(add `force_large_figures: bool=False`).
- C numerical/decode: vfe3/geometry/retraction.py, vfe3/model/prior_bank.py — F2(retraction), F6.
- D policy/model: vfe3/inference/policy.py, vfe3/model/model.py — F3(docstrings), F4, F5(raise
  msg + register_policy guard), C13, C18(generate reorder), F10(generate guard).
- E transport/metrics: vfe3/geometry/transport.py, vfe3/metrics.py, vfe3/gradients/oracle.py,
  vfe3/inference/e_step.py, vfe3/gradients/kernels.py, vfe3/viz/figures.py — C4, C6(metrics+figs),
  C7(transport docstring), C14, F10(chunk _build_regime_ii), C18(fisher_trace reorder).
- F viz: vfe3/viz/extract.py — C9.
- G ablation/robustness: ablation.py, scaling.py, vfe3/inference/sigma_gate.py,
  sigma_gate_measure.py — C10, C15, C17, C12, C16.
Cross-worker contract: field name is `force_large_figures` (B defines, A reads).

## Per-finding verified specs (exact edits + tests) follow.

====================================================================================================
F1 [high] verdict=confirmed deferred=False
TITLE: Finite scalar loss with a nonfinite parameter gradient still steps AdamW and poisons its moment buffers
ROOT_CAUSE: skip_step is derived solely from math.isfinite(step_loss). A finite scalar loss can still produce a NaN/Inf parameter gradient through the unrolled E-step on a degenerate batch; that gradient reaches _scaler.step(optimizer) and permanently corrupts AdamW exp_avg/exp_avg_sq. The audit's own local probe reproduced param_finite_after_step=False from a finite loss plus NaN grad.
LOCS:
   - vfe3/train.py:400 (metrics_out['loss_finite'] = math.isfinite(step_loss) only)
   - vfe3/train.py:410 (_scaler_enabled = scaler is not None and scaler.is_enabled())
   - vfe3/train.py:411 (skip_step = (not _scaler_enabled) and (not math.isfinite(step_loss)) -- gates ONLY on scalar-loss finiteness)
   - vfe3/train.py:412-413 (clip_grad_norm_ without error_if_nonfinite)
   - vfe3/train.py:414-418 (steps optimizer + scaler.update when skip_step False)
   - vfe3/train.py:424-426 (barycenter + scheduler.step run unconditionally)
MINIMAL_FIX:
In vfe3/train.py train_step, right after the unscale block (after line 366 `_scaler.unscale_(optimizer)`), compute the scaler-enabled flag and a finite-gradient gate with a single device->host sync: 
    _scaler_enabled = scaler is not None and scaler.is_enabled()
    grad_finite = True
    if not _scaler_enabled:
        _flags = [torch.isfinite(p.grad).all()
                  for g in optimizer.param_groups
                  for p in g['params'] if p.grad is not None]
        grad_finite = bool(torch.stack(_flags).all()) if _flags else True
Delete the now-duplicate `_scaler_enabled = ...` at line 410 and change line 411 to:
    skip_step = (not _scaler_enabled) and ((not math.isfinite(step_loss)) or (not grad_finite))
Gate the barycenter M-step (lines 424-425) under `not skip_step`:
    if (not skip_step) and _cfg.learnable_r and _cfg.r_update_mode == 'barycenter':
        model.prior_bank.barycenter_r_()
In the metrics block (near line 400) expose the gate for tests/CSV:
    metrics_out['step_skipped'] = float(skip_step)
    metrics_out['grad_finite'] = float(grad_finite)
Do NOT skip scheduler.step() (line 426): the resume path rebuilds LambdaLR with last_epoch=start_step-1 assuming exactly one scheduler.step per loop iteration, so skipping it would desync the LR schedule on resume. Leave the caller's ema.update() unchanged: on a skipped step params are unchanged and finite, and EMA.update already guards with torch.isfinite(p).all() (vfe3/ema.py:56), so no shadow corruption occurs. The fp16 GradScaler path (scaler enabled) is untouched -- it still skips internally via found_inf, so the gate is byte-identical there.
NEW_TESTS:
   - tests/test_train.py: add test_train_step_skips_on_nonfinite_grad_with_finite_loss -- build a tiny VFEModel+optimizer, register a backward hook on one parameter that returns torch.full_like(g, float('nan')), run one train_step(..., grad_accum_steps=1, scaler=disabled, metrics_out={}); assert metrics_out['loss_finite']==1.0, metrics_out['grad_finite']==0.0, metrics_out['step_skipped']==1.0, and that all params + all optimizer exp_avg/exp_avg_sq buffers remain finite after the call (mirror the finite-check style in test_fp16_gradscaler.py).
RISK_NOTES: The added gate performs one extra D2H sync per step on the disabled-scaler default path (host-side branch on grad finiteness); this is a small per-step cost but required for correctness. It runs on EVERY step now, not just logged steps -- confirm no throughput-regression test asserts a fixed step time. Keeping scheduler.step() unconditional is a deliberate deviation from the audit's wording to preserve the resume last_epoch invariant (test_checkpoint_resume.py). Barycenter gating only affects learnable_r + r_update_mode='barycenter' runs (default learnable_r=False, so pure path unchanged). No change to the fp16 path keeps test_fp16_gradscaler.py green.

====================================================================================================
F11 [medium] verdict=confirmed deferred=False
TITLE: Stale val diagnostics retained on replay failure; attention/gamma map replay evaluated outside any try/except can abort training
ROOT_CAUSE: Two best-effort seams leak. (1) On a _val_diagnostics failure the except at 807-808 logs but does not reset last_val_diag, so the previous eval's probe values are carried forward into later CSV rows as if fresh. (2) The expensive model replays model.attention_maps(tokens) and model.gamma_attention_maps(tokens) are argument expressions evaluated in train() before control enters the save helper, so an exception there escapes the helper's internal catch and kills the run.
LOCS:
   - vfe3/train.py:720 (last_val_diag initialized to NaN per _VAL_DIAG_KEYS)
   - vfe3/train.py:805-808 (try: last_val_diag.update(_val_diagnostics(...)); except: warn only -- stale prior values retained)
   - vfe3/train.py:813 (artifacts.save_attention_maps(step+1, model.attention_maps(tokens), ...) -- model.attention_maps(tokens) evaluated in the CALLER, before the helper's internal try/except)
   - vfe3/train.py:816 (same for model.gamma_attention_maps(tokens))
   - vfe3/run_artifacts.py:123-152 and :180-195 (save_* helpers only catch errors that occur INSIDE the helper, not in the argument expression)
MINIMAL_FIX:
In vfe3/train.py, in the except block at 807-808 reset the diagnostics to explicit NaN so a failed replay never carries stale values forward:
    except Exception as exc:
        logger.warning('       (validation diagnostics failed: %s); continuing', exc)
        last_val_diag.update({k: float('nan') for k in _VAL_DIAG_KEYS})
Then wrap the two attention-map replays + saves (lines 813 and 816) in a shared try/except so the model replay is inside the guard:
    try:
        artifacts.save_attention_maps(step + 1, model.attention_maps(tokens), logger=logger)
        artifacts.save_gamma_attention_maps(step + 1, model.gamma_attention_maps(tokens), logger=logger)
    except Exception as exc:
        logger.warning('       (attention-map replay failed: %s); continuing', exc)
Leave artifacts.maybe_save_best (line 809) as-is (cheap, no model replay).
NEW_TESTS:
   - tests/test_train.py: add test_attention_map_replay_failure_does_not_kill_training -- monkeypatch model.attention_maps to raise RuntimeError, run train() with artifacts + eval_interval set so an eval fires; assert train() returns normally (no exception) and a warning was logged.
   - tests/test_train.py: add test_val_diagnostics_failure_resets_to_nan -- monkeypatch vfe3.train._val_diagnostics to raise, run train() through one eval with artifacts; assert the last CSV row's _VAL_DIAG_KEYS columns are NaN/blank rather than a prior eval's values.
RISK_NOTES: Wrapping both save calls in one try/except means a gamma-map failure now also skips the (already-succeeded) attention-map save on the SAME eval only if attention itself threw first; ordering is preserved so attention save still runs before gamma. Confirm no test asserts that save_gamma is reached when save_attention raised. Resetting last_val_diag to NaN changes CSV cells on a diagnostics-failure eval from stale-value to blank -- align any test in test_run_artifacts.py that reads those columns.

====================================================================================================
C1 [medium] verdict=confirmed deferred=True
TITLE: Resume does not restore the shuffled DataLoader stream
ROOT_CAUSE: On resume the loop restarts at start_step but creates a fresh iter(loader). For a shuffled RandomSampler the epoch permutation and intra-epoch cursor are not persisted, so the resumed run draws a different sequence of batches than an uninterrupted run would have from that step. Restoring global RNG (which IS bundled) does not reproduce the in-flight iterator's remaining permutation -- the audit probe confirmed match=False. A truly sealed fix (persisting sampler/generator state + batch cursor + epoch, or replacing shuffle with a deterministic global-step sampler) is a resume-primitive buildout that also risks changing default data-ordering semantics, so it should not be built blindly.
LOCS:
   - vfe3/run_artifacts.py:226-233 (save_checkpoint bundle: step, model_state, optimizer_state, rng_state, config, scaler_state, ema_state -- NO sampler/generator cursor or epoch/batch offset)
   - vfe3/train.py:715 (it = iter(loader) -- fresh iterator on resume)
   - vfe3/train.py:721-726 (loop consumes next(it) from the fresh iterator; StopIteration rebuilds a fresh epoch)
   - vfe3/data/datasets.py:199-202 (make_dataloader forwards shuffle + generator into DataLoader; shuffle default True)
   - train_vfe3.py:381-383 (_select_loader builds train loader with shuffle=is_train and a fixed Generator only when DATA_SEED set)
   - tests/test_checkpoint_resume.py:27-32 (_const_loader uses shuffle=False constant stream, masking the gap)
MINIMAL_FIX:
Defer the full sampler-state resume. Apply the fail-closed/label variant: (1) In vfe3/train.py, at the top of the resume branch (right after load_checkpoint at line 679-680), detect a shuffling train loader and warn once that mid-stream shuffle order is not restored:
    import warnings
    _samp = getattr(loader, 'sampler', None)
    if _samp is not None and type(_samp).__name__ == 'RandomSampler':
        warnings.warn('resume: the training DataLoader shuffles; the pre-interruption shuffle permutation and cursor are NOT restored, so a resumed shuffled run is not batch-identical to an uninterrupted run (RNG/weights/optimizer ARE restored). Use shuffle=False or a deterministic step sampler for a sealed resume.', UserWarning, stacklevel=2)
(2) Add a regression that PINS the known limitation so it is documented and cannot silently change.
SAFE_VARIANT:
Emit the UserWarning above at resume time when the loader uses a RandomSampler, and add a regression test tests/test_checkpoint_resume.py::test_shuffled_resume_is_not_batch_equivalent_and_warns that (a) asserts the warning fires and (b) documents non-equivalence: build a shuffled loader, run N steps, checkpoint at N/2, resume, and assert the resumed batch stream diverges from the uninterrupted stream (pinning current behavior). This turns a silent gap into a tested, warned limitation without the buildout.
NEW_TESTS:
   - tests/test_checkpoint_resume.py: test_shuffled_resume_warns_and_is_not_batch_equivalent -- with a shuffle=True loader over a NONCONSTANT token stream, assert a UserWarning is raised on resume and that resumed batches differ from the continuous run (documents the limitation).
RISK_NOTES: Do NOT switch the default loader to a deterministic sampler as part of this fix -- that would change data ordering for every run and break byte-identical/golden expectations. The warning must be scoped to the resume branch so from-scratch runs are unaffected. Ensure the RandomSampler type check does not fire for eval/test loaders (shuffle=False -> SequentialSampler).

====================================================================================================
C2 [medium] verdict=confirmed deferred=False
TITLE: Resume loses best_val_ppl / best_step model-selection state
ROOT_CAUSE: best_val_ppl and best_step live only on the in-memory RunArtifacts object and are never serialized. A resumed continuation starts them at inf/None, so if no post-resume eval improves, finalize_run reports wrong best metadata and best_model.pt may reflect only the continuation's best rather than the run-wide best.
LOCS:
   - vfe3/run_artifacts.py:61-62 (best_val_ppl=inf, best_step=None initialized fresh)
   - vfe3/run_artifacts.py:100-107 (maybe_save_best updates in-memory best_val_ppl/best_step + best_model.pt only)
   - vfe3/run_artifacts.py:226-233 (save_checkpoint bundle omits best_val_ppl/best_step)
   - vfe3/run_artifacts.py:287-315 (load_checkpoint restores model/optimizer/scaler/ema/rng, never best state)
   - vfe3/run_artifacts.py:611-629 (finalize_run reports the current artifact object's in-memory best_val_ppl/best_step)
MINIMAL_FIX:
Surgical serialize+restore. (1) In vfe3/run_artifacts.py RunArtifacts.save_checkpoint torch.save dict (lines 226-233) add two entries:
    'best_val_ppl': float(self.best_val_ppl),
    'best_step':    self.best_step,
(2) Add an optional artifacts parameter to load_checkpoint (keyword-only, after ema): `artifacts: Optional['RunArtifacts'] = None`; after the model/optimizer restore block, add:
    if artifacts is not None and ckpt.get('best_val_ppl') is not None:
        artifacts.best_val_ppl = float(ckpt['best_val_ppl'])
        artifacts.best_step    = ckpt.get('best_step')
(3) In vfe3/train.py at the load_checkpoint call (line 679-680) pass `artifacts=artifacts` (already in train() scope, may be None -> handled). Bundles written before this field simply skip restore (backward compatible).
NEW_TESTS:
   - tests/test_checkpoint_resume.py: test_resume_restores_best_val_state -- create RunArtifacts, set best_val_ppl/best_step via maybe_save_best, save_checkpoint, then a fresh RunArtifacts + load_checkpoint(artifacts=new_art); assert new_art.best_val_ppl and best_step equal the saved values.
   - tests/test_run_artifacts.py: extend test_save_checkpoint_is_loadable to assert the bundle dict now contains 'best_val_ppl' and 'best_step' keys.
RISK_NOTES: best_model.pt itself is NOT carried in the bundle; this fix restores only the scalar metadata, which is correct when resuming inside the SAME run_dir (best_model.pt already present for finalize_run's reload at line 613-617). Cross-run-dir resume would still lack the best weights file -- note this as a documented limitation, not covered by the surgical fix. Adding a param to load_checkpoint is inside run_artifacts.py (no import cycle); verify no other caller of load_checkpoint passes positionally past the keyword-only barrier (it is keyword-only, safe).

====================================================================================================
C3 [medium] verdict=confirmed deferred=False
TITLE: EMA enabled on resume from a non-EMA checkpoint seeds the shadow from fresh pre-load weights
ROOT_CAUSE: EMA is built before the checkpoint overwrites the model, so its shadow is a clone of the fresh random init. When the bundle carries no ema_state (use_ema=False->True drift, or a legacy checkpoint), load_checkpoint leaves the shadow at that fresh-init value, and the running average is then corrupted by blending real weights into random-init noise; the final copy_to can write it into the evaluated/checkpointed model.
LOCS:
   - vfe3/train.py:676 (ema = EMA(model, ...) constructed BEFORE load_checkpoint)
   - vfe3/ema.py:38-42 (EMA.__init__ clones current model params into self.shadow)
   - vfe3/train.py:679-680 (load_checkpoint replaces model weights AFTER EMA construction)
   - vfe3/run_artifacts.py:296-297 (ema.load_state_dict only when ckpt.get('ema_state') is not None)
   - vfe3/train.py:741 (ema.update blends loaded weights into a fresh-seeded shadow)
   - vfe3/train.py:943 (ema.copy_to can make that shadow the final model)
MINIMAL_FIX:
Reinitialize the shadow from the loaded weights when no bundled ema_state exists. (1) Add a small method to vfe3/ema.py EMA:
    @torch.no_grad()
    def reset(self, model: torch.nn.Module) -> None:
        r'''Reseed the shadow from the model's current params (e.g. after loading resumed weights).'''
        self.shadow = {name: param.detach().clone()
                       for name, param in model.named_parameters()
                       if param.requires_grad}
(2) In vfe3/run_artifacts.py load_checkpoint, replace the current guard (lines 296-297) with:
    if ema is not None:
        if ckpt.get('ema_state') is not None:
            ema.load_state_dict(ckpt['ema_state'])
        else:
            ema.reset(model)   # no bundled shadow: reseed from the just-loaded weights, not the pre-load fresh init
This runs after model.load_state_dict (line 288), so ema.reset captures the resumed weights. Pure path (use_ema=False -> ema is None) is unchanged.
NEW_TESTS:
   - tests/test_ema.py: test_ema_resets_shadow_after_load_when_no_ema_state -- build model A, mutate its weights, save_checkpoint with ema=None (no ema_state); build model B + EMA(B) (fresh shadow), load_checkpoint(B, ema=ema_B); assert every ema_B.shadow[name] equals the loaded B weights (i.e. reset happened), not B's pre-load init.
   - tests/test_checkpoint_resume.py: test_resume_with_ema_from_non_ema_ckpt_shadow_tracks_loaded_weights -- resume a use_ema=True run from a use_ema=False checkpoint; assert the EMA shadow after resume matches loaded weights rather than fresh init.
RISK_NOTES: ema.reset reaches the same shadow-construction logic as __init__; keep the requires_grad filter identical so frozen params (learnable_r=False) stay excluded consistently. This does not affect the normal same-config resume (ema_state present -> load_state_dict branch unchanged). Verify test_ema.py existing state_dict/load_state_dict round-trip test is untouched.

====================================================================================================
C8 [medium] verdict=confirmed deferred=True
TITLE: Gradient accumulation token-weights the whole fused loss, including non-CE regularizers
ROOT_CAUSE: model.forward returns one fused loss = CE + regularizers. The accumulation path weights that entire fused loss by the microbatch's valid-target-token fraction n_mb/n_tot. CE is a token-mean and needs that weighting, but the non-CE regularizers are means over (B,N)/state and do NOT scale with target tokens; an all-ignore or heavily-padded microbatch then contributes little or no regularizer gradient even though its belief/state regularizers are well-defined. Correctly separating the two denominators requires the forward to return CE and regularizer components separately (a structured-loss buildout).
LOCS:
   - vfe3/train.py:349-359 (per-microbatch counted-token counts _mb_tok; w = n_mb/n_tot applied to the whole loss_mb and its .backward())
   - vfe3/model/model.py:909 (loss = ce)
   - vfe3/model/model.py:910-916 (+ mass_phi term)
   - vfe3/model/model.py:917-967 (+ mstep_self_coupling term)
   - vfe3/model/model.py:1012 (+ hyper-prior lambda_h term)
   - vfe3/model/model.py:1031 (+ gamma coupling term) -- forward returns a single fused (_, loss, ce)
MINIMAL_FIX:
Do NOT build the structured-loss separation blindly. Apply the fail-closed guard: in vfe3/train.py, in the grad_accum branch, after computing _mb_tok and n_tot (line 350), warn once when microbatch counted-token counts are unequal (the only regime where the mis-weighting bites; the default unpadded loader has equal counts so w == 1/grad_accum_steps exactly and behavior is byte-identical):
    if grad_accum_steps > 1 and _mb_tok and (max(_mb_tok) != min(_mb_tok)):
        import warnings
        warnings.warn('grad_accum_steps>1 with uneven counted-token microbatches: non-CE regularizers (mass_phi, mstep_self_coupling, lambda_h, gamma) are token-weighted by n_mb/n_tot rather than by their own reduction, so their accumulated gradient is an approximation. Use an unpadded/equal-token loader or grad_accum_steps=1 for the exact objective.', RuntimeWarning, stacklevel=2)
The existing grad_accum_tok_spread metric (line 402-403) already surfaces this spread; the warning makes it fail-loud. The full fix (return CE + regularizer components from forward and accumulate each by its own denominator) is deferred.
SAFE_VARIANT:
Emit the RuntimeWarning above only when microbatch token counts are unequal (spread > 0). This leaves the exact equal-token default path byte-identical and warns exactly in the regime where the audit shows the regularizer weighting diverges, without touching model.forward's return contract.
NEW_TESTS:
   - tests/test_grad_accum.py: test_uneven_microbatch_token_counts_warns -- build a batch where one microbatch is all -100 targets (or padded) so _mb_tok spread > 0, run train_step with grad_accum_steps=2; assert a RuntimeWarning about non-CE regularizer weighting is raised. Also assert the existing equal-token case (unpadded) raises NO warning and remains byte-identical to grad_accum_steps=1 accumulation (reuse the equivalence assertion already in test_grad_accum.py).
RISK_NOTES: The default click-run loader is unpadded/equal-token, so the warning never fires there and the pure path is unchanged -- do not let the warning trip existing equal-token grad-accum equivalence tests. Building the real structured-loss fix would touch model.forward's (logits, loss, ce) return signature and every caller (train_step, evaluate, diagnostics), a broad change deferred per audit. Keep the guard confined to train_step's accum branch.

====================================================================================================
F2-config [medium] verdict=confirmed deferred=False
TITLE: sigma_max is never validated in VFE3Config.__post_init__
ROOT_CAUSE: sigma_max is declared as a plain float at vfe3/config.py:378 (default 10.0). __post_init__ (starts vfe3/config.py:596) validates eps and kl_max in its numerics block (:598-601) but grep of the whole file finds sigma_max only at :378 -- it is validated nowhere. The SPD retractions clamp covariance with max=sigma_max, so a nonfinite or sub-eps cap reaches the covariance update and can produce values below eps, negative, or NaN. (The audit's cited line :596-601 is the numerics block; sigma_max is simply absent from it.)
LOCS:
   - vfe3/config.py:378 (sigma_max declaration)
   - vfe3/config.py:596-601 (__post_init__ numerics block, no sigma_max)
   - vfe3/config.py:9 (import math)
MINIMAL_FIX:
In vfe3/config.py __post_init__ numerics block, right after the kl_max check at line 601, add (math is already imported at config.py:9):

    if self.sigma_max is not None and not (math.isfinite(self.sigma_max) and self.sigma_max >= self.eps):
        raise ValueError(
            f"sigma_max must be None or finite and >= eps ({self.eps}), got {self.sigma_max}"
        )

Keep the `is not None` guard so the field stays permissive even though its current annotation is a bare float; retraction.py defensive clamp is owned by the numerical agent, not here.
NEW_TESTS:
   - tests/test_config.py: add test asserting VFE3Config(sigma_max=-1.0), VFE3Config(sigma_max=float('nan')), and VFE3Config(sigma_max=1e-9) (below default eps=1e-6) each raise ValueError; assert VFE3Config(sigma_max=10.0) (default) still constructs.
RISK_NOTES: Existing tests construct sigma_max=5.0 only inside e_step/metrics helper calls (tests/test_e_step.py, tests/test_metrics.py), not via VFE3Config, so no config test passes a sub-eps cap -- the new guard will not break the 1388-suite. Default configs use sigma_max=10.0 >= eps. Do not tighten below eps or you would reject the legitimate default.

====================================================================================================
F12-bool [low] verdict=confirmed deferred=False
TITLE: bool config fields validated by truthiness/coercion, not type (trust_resume_checkpoint, generate_figures)
ROOT_CAUSE: trust_resume_checkpoint is annotated bool at vfe3/config.py:567 but the __post_init__ validation blocks never type-check it; load_checkpoint reads bool(getattr(cfg,'trust_resume_checkpoint',False)) (run_artifacts.py:276) so the string "False" coerces to True and enables the unsafe weights_only=False fallback. generate_figures (config.py:579) is likewise consumed by truthiness at run_artifacts.py:726. Continuation probe confirmed VFE3Config(trust_resume_checkpoint="False") is accepted.
LOCS:
   - vfe3/config.py:567 (trust_resume_checkpoint: bool)
   - vfe3/config.py:579 (generate_figures: bool)
   - vfe3/run_artifacts.py:276 (bool(getattr(...)) coercion, per audit)
MINIMAL_FIX:
In vfe3/config.py __post_init__ (place near the other bool/interval checks around :1768-1793), add explicit identity type checks that reject non-bool (including the int/str footguns):

    for _bname in ("trust_resume_checkpoint", "generate_figures", "use_ema"):
        _bval = getattr(self, _bname)
        if type(_bval) is not bool:
            raise ValueError(f"{_bname} must be a bool, got {type(_bval).__name__}: {_bval!r}")

Use `type(x) is not bool` (not isinstance) so ints/strings are rejected. Keep the list to the security/behavior-relevant bools called out by the audit; do not sweep every bool field (over-validation risk).
NEW_TESTS:
   - tests/test_config.py: assert VFE3Config(trust_resume_checkpoint="False") raises ValueError; assert VFE3Config(generate_figures=1) raises; assert defaults (both real bools) still construct.
RISK_NOTES: Confirm no existing test or config passes 0/1 for these bools -- grep of tests/ shows no such construction. use_ema is included because it also gates a code path; if any test passes a truthy non-bool for use_ema this would newly fail (none found). Keep the field list minimal to avoid breaking configs that legitimately pass Python bools.

====================================================================================================
F12-registry [low] verdict=confirmed deferred=False
TITLE: Registry decorators overwrite duplicate keys silently
ROOT_CAUSE: register_alpha (vfe3/alpha_i.py:28-31 _ALPHAS[name]=fn), register_prior (vfe3/attention_prior.py:36-38 _PRIORS[name]=fn), and register_policy (vfe3/inference/policy.py:50-52 _POLICIES[name]=fn) assign into the dict with no duplicate-key check. A second @register with an existing name silently shadows the first -- a config-selected seam can dispatch to an unintended implementation. register_compose in lie_ops.py shares the same pattern.
LOCS:
   - vfe3/alpha_i.py:28-31
   - vfe3/attention_prior.py:34-38
   - vfe3/inference/policy.py:48-52
   - vfe3/geometry/lie_ops.py:243 (register_compose('bch'))
MINIMAL_FIX:
In each decorator's inner _wrap, guard before assignment. For vfe3/alpha_i.py register_alpha add an `override: bool = False` kwarg and:

    def _wrap(fn: Callable) -> Callable:
        if name in _ALPHAS and not override:
            raise KeyError(f"alpha form {name!r} already registered; pass override=True to replace")
        _ALPHAS[name] = fn
        _ALPHA_PER_COORD[name] = per_coord
        return fn

Apply the identical guard to register_prior (_PRIORS), register_policy (_POLICIES), and register_compose (_COMPOSE), each with its own `override=False` keyword. This preserves write-and-register modularity while failing closed on accidental key collision.
NEW_TESTS:
   - tests/test_alpha_i.py (or tests/test_config.py registry section): assert re-registering an existing alpha name via register_alpha raises KeyError, and that register_alpha(name, override=True) succeeds. Mirror one assertion for register_prior and register_policy.
RISK_NOTES: Verified there are currently NO duplicate @register_alpha/@register_prior/@register_policy/@register_compose keys in vfe3/ (uniq -d empty), so a hard raise will not break module import or the 1388-suite. Risk only if a test module intentionally re-registers a name to monkeypatch -- none found; if one is added later it must pass override=True.

====================================================================================================
F12-ckpt [low] verdict=confirmed deferred=False
TITLE: generate_efe.py checkpoint config loader silently drops unknown fields
ROOT_CAUSE: In vfe3/generate_efe.py _build_model, cfg_dict = {k: v for k, v in config_dict.items() if k in valid} drops any checkpoint config key not in VFE3Config's current fields without notice (comment 'drop any stale/unknown keys'). A renamed/removed field in an older checkpoint silently reverts to the current default, which can change the reconstructed architecture/behavior with no warning.
LOCS:
   - vfe3/generate_efe.py:_build_model (valid = {f.name for f in fields(VFE3Config)}; cfg_dict = {k:v ... if k in valid}, ~lines 78-82)
MINIMAL_FIX:
In vfe3/generate_efe.py _build_model, immediately after computing `valid` and before the comprehension, warn on dropped keys (non-fatal, since dropping genuine legacy fields is intended):

    import warnings
    dropped = sorted(set(config_dict) - valid)
    if dropped:
        warnings.warn(
            f"generate_efe: checkpoint config has {len(dropped)} field(s) unknown to the current "
            f"VFE3Config, dropping them (behavior falls back to defaults): {dropped}",
            UserWarning, stacklevel=2,
        )
    cfg_dict = {k: v for k, v in config_dict.items() if k in valid}

Keep it a warning, not a raise: this is a click-to-run helper for the user's own trusted checkpoints and legacy-field drop is a legitimate migration.
NEW_TESTS:
   - No new test required for a research helper script; optionally tests/test_generate_efe.py (if it exists) can assert a UserWarning is emitted when _build_model is fed a config dict containing a bogus extra key.
RISK_NOTES: generate_efe.py is not imported by the training/test path (it is a standalone script), so this warn cannot affect the 1388-suite. Do not convert to a raise -- that would break loading of legitimately older checkpoints, which is the documented purpose of the drop.

====================================================================================================
F12-steps [low] verdict=confirmed deferred=False
TITLE: max_steps and warmup_steps accept 0, negative, and float values
ROOT_CAUSE: max_steps (config.py:537) and warmup_steps (config.py:538) are consumed by the scheduler (train.py:222-224) and the training loop range(start_step, n_steps) (train.py:702) and by run_artifacts token accounting, but __post_init__ validates batch/accum/log/eval/checkpoint intervals without validating these two. Continuation probes accepted max_steps=0 (empty training loop), warmup_steps=-1, and max_steps=4.0 (a float that later TypeErrors in range()). The scheduler already defends division with max(1, ...) so warmup_steps=0 is legitimate (no warmup); only <0 and non-int must be rejected, and max_steps must be a positive int.
LOCS:
   - vfe3/config.py:537-538 (max_steps, warmup_steps)
   - vfe3/train.py:222-224 (scheduler: step/max(1,cfg.warmup_steps), max(1,cfg.max_steps-cfg.warmup_steps))
   - vfe3/config.py:1768-1777 (interval validation block that omits them)
MINIMAL_FIX:
In vfe3/config.py __post_init__, alongside the interval checks (after checkpoint_interval at :1777), add:

    if type(self.max_steps) is not int or self.max_steps < 1:
        raise ValueError(f"max_steps must be an int >= 1, got {self.max_steps!r}")
    if type(self.warmup_steps) is not int or self.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be an int >= 0, got {self.warmup_steps!r}")

`type(...) is not int` rejects both float (4.0) and bool; warmup_steps>=0 keeps the legitimate no-warmup config.
NEW_TESTS:
   - tests/test_config.py: assert VFE3Config(max_steps=0), VFE3Config(max_steps=4.0), and VFE3Config(warmup_steps=-1) each raise ValueError; assert VFE3Config(warmup_steps=0) constructs (no-warmup is valid) and default max_steps=15000 constructs.
RISK_NOTES: Check no existing test constructs max_steps as a float or 0 -- grep of tests/ found none. Small-max_steps smoke tests (e.g. max_steps=2) remain valid. Do NOT reject warmup_steps=0; several fast tests may run with zero warmup and the scheduler's max(1,...) already handles it.

====================================================================================================
F12-bch [low] verdict=confirmed deferred=False
TITLE: bch_pe_order=0 silently makes pos_phi_compose='bch' additive (no BCH correction)
ROOT_CAUSE: pos_phi_compose is registry-validated at config.py:1256 (_require against _COMPOSE) but no numeric guard constrains bch_pe_order (config.py:164, default 4). model.py:569 passes order=self.cfg.bch_pe_order into apply_positional_phi -> compose_bch, and compose_bch starts Z = X + Y (lie_ops.py:299) and adds Dynkin corrections only under `order >= 1` (lie_ops.py:300-312). So pos_phi_compose='bch' with bch_pe_order=0 degrades to plain additive composition while still labeled BCH. (phi_retract_mode='bch' at model.py:625 does not thread bch_pe_order, so it keeps compose_bch's default order=4 and is unaffected.)
LOCS:
   - vfe3/config.py:164 (bch_pe_order default 4)
   - vfe3/config.py:1256 (_require pos_phi_compose)
   - vfe3/model/model.py:569 (order=self.cfg.bch_pe_order)
   - vfe3/geometry/lie_ops.py:299-312 (Z=X+Y; corrections only order>=1)
MINIMAL_FIX:
In vfe3/config.py __post_init__, immediately after the pos_phi_compose _require at line 1256, add:

    if self.pos_phi != "none" and self.pos_phi_compose == "bch" and self.bch_pe_order < 1:
        raise ValueError(
            f"pos_phi_compose='bch' needs bch_pe_order >= 1 (order 0 drops all Dynkin "
            f"corrections and is plain additive, not BCH), got {self.bch_pe_order}"
        )

Guard on pos_phi != 'none' because _apply_pos_phi (model.py:564) returns early for pos_phi='none' and never reaches compose_bch, so bch_pe_order is inert there.
NEW_TESTS:
   - tests/test_config.py: assert VFE3Config(pos_phi='learned', pos_phi_compose='bch', bch_pe_order=0) raises ValueError; assert the default (bch_pe_order=4) and pos_phi='none' with bch_pe_order=0 both construct. tests/test_config.py:431 already pins default bch_pe_order==4.
RISK_NOTES: Only fires when pos_phi is active AND compose is bch AND order<1 -- a narrow combination no existing passing config uses (defaults are order=4). Confirm no test constructs pos_phi_compose='bch' with order 0. Does not touch phi_retract_mode path.

====================================================================================================
C13 [low] verdict=confirmed deferred=False
TITLE: generate() lacks sampler-argument validation on the normal (policy_mode='none') path
ROOT_CAUSE: generate() validates temperature/top_k/top_p only when policy_mode!='none' (model.py:1306-1310). On the default path range(max_new_tokens) (model.py:1312) treats negative as an empty loop (returns the prompt unchanged), logits/temperature (model.py:1320) accepts 0/negative, logits.topk(top_k) (model.py:1322) accepts out-of-range k, and top_p filtering (model.py:1324-1332) accepts p outside (0,1]. Invalid values fail late or produce invalid probabilities. vocab_size is available as self.cfg.vocab_size (config.py:49).
LOCS:
   - vfe3/model/model.py:1306-1310 (policy-only validation)
   - vfe3/model/model.py:1312-1332 (unvalidated normal sampling path)
   - vfe3/config.py:49 (vocab_size)
MINIMAL_FIX:
In vfe3/model/model.py generate(), insert validation right after the existing policy_mode check (after line 1310) and before `seq = token_ids`:

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if not greedy and self.cfg.policy_mode == "none":
        if not (temperature > 0.0):
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if top_k is not None and not (1 <= top_k <= self.cfg.vocab_size):
            raise ValueError(f"top_k must be in [1, vocab_size={self.cfg.vocab_size}], got {top_k}")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

Gate the temperature/top_k/top_p checks on `not greedy and policy_mode=='none'` because greedy ignores them and the policy path already rejects non-default knobs; max_new_tokens>=0 is checked unconditionally.
NEW_TESTS:
   - tests/test_model.py (or wherever generate is exercised): assert model.generate(prompt, -1) raises ValueError; model.generate(prompt, 4, temperature=0.0) raises; top_k=0 and top_k=vocab_size+1 raise; top_p=0.0 and top_p=1.5 raise; a valid greedy call with temperature=0.0 still runs (knobs ignored under greedy).
RISK_NOTES: Confirm greedy-path tests that pass temperature defaults still work (they do: greedy skips the checks). Any existing test that calls generate with max_new_tokens=0 remains valid (>=0). Do not validate temperature under greedy or you would break greedy calls that leave a nonstandard temperature set.

====================================================================================================
C17 [low] verdict=confirmed deferred=False
TITLE: _expand_range silently returns empty for sign-mismatched ranges
ROOT_CAUSE: ablation.py _expand_range rejects only step==0 (:1200-1201). It computes n = int(round((stop-start)/step)) and loops range(n+2) with an early break when v passes stop (:1205-1209). For start<stop with negative step, or start>stop with positive step, n is negative so range(n+2) is empty (or the first value already breaks), yielding [] with no error -- a malformed sweep runs zero cells silently.
LOCS:
   - ablation.py:1195-1211 (_expand_range)
   - ablation.py:1200-1201 (only step==0 rejected)
   - ablation.py:1205-1210 (n and loop)
MINIMAL_FIX:
In ablation.py _expand_range, after the step==0 check (line 1201) add a sign-consistency guard, and after building `values` assert nonempty:

    if start != stop and (stop - start) * step < 0:
        raise ValueError(
            f"'range' step sign disagrees with [start, stop] direction: {spec!r}"
        )
    ... (existing loop) ...
    if not values:
        raise ValueError(f"'range' expanded to no values: {spec!r}")

The `start != stop` exception preserves the degenerate single-point range [x, x, step] (which legitimately yields [x]).
NEW_TESTS:
   - tests/test_ablation.py (range expansion tests): assert _expand_range([0, 5, -1]) and _expand_range([5, 0, 1]) raise ValueError; assert _expand_range([2, 2, 1]) == [2] still works; assert an ascending range like [0, 4, 2] == [0, 2, 4] is unchanged.
RISK_NOTES: Confirm existing ablation sweep configs/tests use only sign-consistent ranges (ascending start<stop with positive step, or descending with negative step) -- a descending sweep [5,0,-1] must still pass, which it does since (0-5)*(-1)=5>0. The single-point [x,x,step] exemption avoids breaking any degenerate sweep.

====================================================================================================
F3 [medium] verdict=confirmed deferred=True
TITLE: efe_rollout (H>1) scores only the terminal predicted outcome, not a per-step horizon sum
ROOT_CAUSE: The shared EFE body _efe_score (vfe3/inference/policy.py:293-319) calls _rollout_predictive ONCE and then does `score = sum(terms[t] for t in score_terms)` -- it sums over the risk/ambiguity TERM set, not over timesteps tau=1..H. _rollout_predictive reads only the last appended position's predictive (`logits[:, -1, :]` at policy.py:255-257, and the cache path `logits[:, -1, :]` at belief_cache.py:190). So for an H-action policy candidate the score is KL[q(o|pi_H)||p(o|C)] + H[q(o|pi_H)] on the TERMINAL outcome only. The local EFE spec (docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md:31-33) defines G(pi) as a sum over the horizon. The _policy_efe_rollout docstring itself admits the terminal read ('q(o|pi)=p(o|q*_pi) is read from the LAST position', policy.py:400-401), so the code and its own docstring already describe terminal-outcome scoring; only the spec/name imply a horizon sum. Does not touch default policy_mode='none' or the one-step ring experiment (efe_one_step, horizon=1).
LOCS:
   - vfe3/inference/policy.py:255-257 (_rollout_predictive reads logits[:, -1, :])
   - vfe3/inference/policy.py:312-319 (_efe_score single rollout, sum over score_terms not timesteps)
   - vfe3/inference/policy.py:397-404 (_policy_efe_rollout docstring already says terminal LAST position)
   - vfe3/inference/belief_cache.py:190 (cache path reads logits[:, -1, :])
MINIMAL_FIX:
DEFERRED buildout (a real H-step EFE sum is theory-gated -- do not build the per-timestep accumulation blindly). Apply the label/document safe variant only: (1) In vfe3/inference/policy.py rewrite the _policy_efe_rollout docstring (lines 397-404) to state plainly that this is a TERMINAL-OUTCOME rollout scorer: G(pi) is evaluated on the single terminal predictive q(o|pi_H) read from the last appended position, NOT the active-inference per-step sum sum_{tau=1..H} G_tau; the H-step rollout only advances the belief so the terminal predictive is conditioned on all H actions. (2) Add one clarifying clause to the _efe_score docstring (policy.py:307-311) noting `sum` there is over score_terms, not over timesteps. (3) Add a one-line note to the policy_horizon field docstring in vfe3/config.py so config authors know H>1 changes only the rollout depth, not the scoring reduction. No functional change; the golden values in tests/test_belief_cache.py::test_efe_rollout_unlocked_on_supported_config stay byte-identical.
SAFE_VARIANT:
Relabel and document the scorer as terminal-outcome rollout scoring (docstring edits in policy.py _policy_efe_rollout + _efe_score and the config.py policy_horizon field), plus a regression test that pins the terminal-only semantics. Do NOT implement the per-timestep horizon sum -- that is the theory-gated H-step EFE build.
NEW_TESTS:
   - tests/test_belief_cache.py: extend the existing test_efe_rollout_unlocked_on_supported_config (line 100) -- after `out = get_policy('efe_rollout')(ctx, cand, pref, m, gamma=1.0, horizon=H)`, independently roll the H-action candidate through model.rollout_beliefs, take log_softmax(logits[:,-1,:]) as q_log_terminal, and assert torch.allclose(out.risk, _efe_terms(q_log_terminal, pref)[0]); this documents that risk is computed on the terminal predictive only, not a horizon sum.
RISK_NOTES: Label/doc + one assertion only. A genuine horizon-sum implementation would change the golden scores in tests/test_belief_cache.py:100-114 and tests/test_efe_scorer.py:159-163 and is theory-gated -- do NOT change _efe_score's reduction. The new assertion reuses _efe_terms (already imported/available in policy.py) so it cannot drift from the implementation.

====================================================================================================
F4 [medium] verdict=confirmed deferred=False
TITLE: Generic generate() lets policy_mode='efe_rollout' validate at config time then die mid-dispatch on a length-1 candidate menu
ROOT_CAUSE: VFE3Config.__post_init__ accepts policy_mode='efe_rollout' as long as policy_horizon>1 (the only pairing check is config.py:1408-1411; there is NO reject of efe_rollout for the generate path). tests/test_policy_registry.py:74 explicitly asserts `VFE3Config(policy_mode='efe_rollout', policy_preference='flat', policy_horizon=2)` validates. But the generic generate path _policy_select (vfe3/model/model.py:1342-1381) always builds a one-step menu `candidates = topk.unsqueeze(-1)` -> (B, Kp, 1) at model.py:1367, then dispatches get_policy('efe_rollout') at model.py:1372-1376 with horizon=cfg.policy_horizon(>1). _policy_efe_rollout requires candidate length L==horizon (policy.py:408-412), so it raises the cryptic 'candidate length L=1 must equal horizon=2' ValueError at generation time. No H-token candidate generator exists anywhere: ring_task.py:326 also builds `topk.unsqueeze(-1)` (length-1) and only ever uses efe_one_step. So a config can pass validation and then fail the first time generate() is called.
LOCS:
   - vfe3/config.py:1408-1411 (only the horizon>1 pairing check; no reject of efe_rollout for generate)
   - vfe3/model/model.py:1364-1376 (_policy_select builds candidates=topk.unsqueeze(-1) length-1, then dispatches cfg.policy_mode)
   - vfe3/inference/policy.py:408-412 (_policy_efe_rollout requires L==horizon, raises otherwise)
   - tests/test_policy_registry.py:74 (config accepts efe_rollout+horizon=2)
   - vfe3/inference/ring_task.py:326,341-345 (harness also builds length-1 candidates; no H-token generator anywhere)
MINIMAL_FIX:
Fail-closed in the generate path (NOT at config construction -- test_policy_registry.py:74 requires the config to validate, and ring_task.py passes its own policy_mode arg rather than cfg.policy_mode, so a global config reject would over-reach). In vfe3/model/model.py::_policy_select, immediately after `from vfe3.inference.policy import get_policy, get_preference` (line 1363) and before building candidates (line 1364), add:

    if self.cfg.policy_mode == "efe_rollout":
        raise NotImplementedError(
            "policy_mode='efe_rollout' (horizon>1) is not reachable through generate(): the generic "
            "policy path builds a one-step (B, Kp, 1) candidate menu, but efe_rollout requires an "
            "H-token (B, Kp, H) policy menu and no H-step candidate generator exists. Drive efe_rollout "
            "through a harness that builds H-action candidates and calls get_policy('efe_rollout') "
            "directly, or set policy_mode='efe_one_step' (horizon=1).")

This converts the mid-dispatch cryptic ValueError into an early, clear NotImplementedError at the exact point the missing generator is needed. cfg.policy_mode is consumed ONLY here (grep confirms model.py:1372 is the sole cfg.policy_mode dispatch; ring_task takes policy_mode as a function arg), so the guard is complete for the generic path.
NEW_TESTS:
   - tests/test_efe_scorer.py or tests/test_generate.py: build a small model with policy_mode='efe_rollout', policy_preference='flat', policy_horizon=2 (reuse the _model/_tiny_model helper), then assert `with pytest.raises(NotImplementedError): m.generate(prompt, 1, greedy=True)` and check the message mentions 'candidate menu' / 'H-token'. This pins that generate() fails closed with a clear error rather than the mid-dispatch ValueError.
RISK_NOTES: MUST place the guard in _policy_select, not in config.__post_init__: tests/test_policy_registry.py:74 requires VFE3Config(policy_mode='efe_rollout', ...) to construct without raising, and ring_task drives its own policy_mode arg. Only generate() is affected. No default-path impact (policy_mode='none' short-circuits before _policy_select). Overlaps conceptually with the existing F4 preference fail-closed at config.py:1424 but is a distinct axis (candidate length, not preference context).

====================================================================================================
F5 [medium] verdict=confirmed deferred=True
TITLE: sigma_mc ambiguity and its policy_sigma_ambiguity_validated flag have no executable consumer -- the config advertises an unlock that cannot be reached
ROOT_CAUSE: _amb_sigma_mc (vfe3/inference/policy.py:210-222) unconditionally raises RuntimeError. There is NO policy_ambiguity_mode config field, and neither generate's _policy_select (model.py:1372-1376) nor ring_task (ring_task.py:341-345) passes an ambiguity_mode argument to the scorer, so _efe_score always uses the default ambiguity_mode='likelihood_entropy' (policy.py:314,336,392). Meanwhile VFE3Config validates policy_sigma_ambiguity_validated at construction (config.py:1438-1445, calling verify_gate_artifact) and stores policy_sigma_gate_artifact (config.py:428-429). So a user can set the flag True with a PASS artifact and it validates, yet NO code path ever reads the flag to route sigma_mc, and sigma_mc still raises. The executable surface therefore overstates what a passing gate unlocks. This is consistent with the sigma gate having failed per the wiki -- fail-closed is the correct current behavior; the defect is only the misleading config state.
LOCS:
   - vfe3/inference/policy.py:210-222 (_amb_sigma_mc unconditionally raises)
   - vfe3/model/model.py:1372-1376 (_policy_select dispatch passes gamma/horizon/score_terms/log_prior/base_logits -- no ambiguity_mode)
   - vfe3/inference/ring_task.py:341-345 (harness dispatch also passes no ambiguity_mode)
   - vfe3/config.py:428-429 (field defs) and 1438-1445 (flag validated via verify_gate_artifact, but no consumer)
   - tests/test_efe_scorer.py:164-165 (pins sigma_mc raises); tests/test_policy_registry.py:137-151 (pins flag validation)
MINIMAL_FIX:
DEFERRED buildout (do NOT wire a sigma-derived ambiguity term; the sigma gate failed). Apply the document/clarify safe variant, keeping the code fail-closed: (1) In vfe3/config.py at the policy_sigma_ambiguity_validated / policy_sigma_gate_artifact field definitions (around lines 422-429) rewrite the comment to state this is a PRECONDITION RECORD ONLY: setting it True (with a PASS artifact) records that the pre-registered sigma gate passed but does NOT enable any sigma-derived ambiguity -- there is currently no consumer, ambiguity is always likelihood_entropy, and get_ambiguity('sigma_mc') still raises. (2) Strengthen the _amb_sigma_mc raise message (policy.py:219-222) to add that the policy_sigma_ambiguity_validated flag ALONE does not unlock this estimator; a Phase-3 consumer that reads the validated artifact must be added first. No behavior change; sigma_mc stays fail-closed. Do NOT add a policy_ambiguity_mode seam or route the flag -- that is the theory-gated build the audit warns against.
SAFE_VARIANT:
Keep sigma_mc fail-closed (unconditional raise stays). Document the config fields as precondition records with no live consumer (config.py field comments + strengthened raise message in policy.py). Do NOT build the sigma ambiguity term or a policy_ambiguity_mode seam.
NEW_TESTS:
   - tests/test_policy_registry.py: add a test that builds a model/config with policy_sigma_ambiguity_validated=True + a PASS policy_sigma_gate_artifact (reuse the ok-artifact fixture at lines 150-151) and asserts the scorer still uses likelihood_entropy -- e.g. run get_policy('efe_one_step')(...) or m.generate(...) and assert it returns finite scores with no sigma_mc dispatch (does NOT raise), documenting that the validated flag has no executable consumer and does not unlock sigma_mc.
   - tests/test_efe_scorer.py: keep the existing line 164-165 raise assertion; optionally assert the new raise message text mentions the flag alone does not unlock the estimator.
RISK_NOTES: Pure documentation + message text; no functional change. Existing tests that pin the raise (test_efe_scorer.py:164) and the flag validation (test_policy_registry.py:137-151) must keep passing -- if the raise message is edited, do not tighten those tests to match exact text unless they already assert substrings. Do NOT add policy_ambiguity_mode or route the flag (deferred buildout). Overlaps with sigma_gate.py which owns verify_gate_artifact -- leave that untouched.

====================================================================================================
F2 [medium] verdict=confirmed deferred=False
TITLE: Invalid sigma_max cap silently yields sub-eps / negative / NaN covariance in the SPD retraction helpers
ROOT_CAUSE: The four SPD retraction helpers clamp the output spectrum with max=sigma_max but never validate sigma_max. torch.clamp(min=eps, max=sigma_max) has no protection when sigma_max is invalid: when sigma_max < eps the max bound wins (returns sigma_max, below the eps variance floor), when sigma_max < 0 every variance becomes negative (breaks the SPD invariant), and when sigma_max is NaN clamp propagates NaN. An accepted bad config (config validator does not gate sigma_max, per the config-agent slice) therefore reaches the retraction and corrupts Sigma. Confirmed by direct probe: retract_spd_diagonal(sigma=2.0, delta=0) returns 9.9999997e-10 for sigma_max=1e-9 (below eps=1e-6), -1.0 for sigma_max=-1.0 (negative variance), and nan for sigma_max=nan.
LOCS:
   - vfe3/geometry/retraction.py:133 (retract_spd_diagonal clamp max=sigma_max)
   - vfe3/geometry/retraction.py:191 (retract_spd_full eig_new clamp)
   - vfe3/geometry/retraction.py:288 (retract_logeuclidean_full eig_new clamp)
   - vfe3/geometry/retraction.py:343 (retract_log_euclidean diagonal clamp)
   - probe output: sigma_max=1e-9 -> 9.9999997e-10; -1.0 -> -1.0; nan -> nan
MINIMAL_FIX:
In vfe3/geometry/retraction.py add a module-level guard right after the imports (near line 22):

    def _check_sigma_max(sigma_max: Optional[float], eps: float) -> None:
        r"""Reject an eigenvalue ceiling that would violate the SPD/eps invariant."""
        if sigma_max is None:
            return
        if not math.isfinite(sigma_max) or sigma_max < eps:
            raise ValueError(
                f"sigma_max must be None or finite and >= eps ({eps}); got {sigma_max!r}"
            )

Call it as the first statement (before the autocast block) in each of the four helpers that clamp with sigma_max: retract_spd_diagonal (before line 124), retract_spd_full (before line 157), retract_logeuclidean_full (before line 257), and in retract_log_euclidean before the diagonal-branch clamp (put it at the top of the function, before line 330, so both branches are covered; the full branch also re-checks via retract_logeuclidean_full which is a harmless idempotent double-check). retract_spd_affine and retract_log_euclidean forward to the bare helpers, so guarding the four helpers covers every registered retraction. None stays allowed (pure path: eps floor only, unchanged). math is already imported.
NEW_TESTS:
   - tests/test_retraction.py: add test_retract_rejects_invalid_sigma_max parametrized over sigma_max in {1e-9, -1.0, float('nan')} and over retract_spd_diagonal (sigma (K,), delta zeros) and retract_spd_full (sigma = I (K,K), delta zeros); assert pytest.raises(ValueError) for each.
   - tests/test_retraction.py: assert sigma_max=None and sigma_max=10.0 still return finite PD output (no raise) so the guard does not regress the pure/default path.
RISK_NOTES: In normal runs the E-step passes VFE3Config.sigma_max (default 10.0, valid), so the guard never fires and existing golden/regression tests are unaffected. The pure path (sigma_max=None) is explicitly preserved. Overlaps with the config-agent's F2 config-validator slice (that fix stops the bad value at construction; this fix is the defense-in-depth at the retraction seam) - both are wanted, neither conflicts. Only risk: a test that deliberately passes an out-of-range sigma_max would now raise; grep of tests/test_retraction.py should confirm none do before landing.

====================================================================================================
F6 [medium] verdict=confirmed deferred=False
TITLE: Full-cov chunked decode discards safe_cholesky ok mask, diverging from dense -inf-logit semantics on non-PD Sigma_q
ROOT_CAUSE: _full_cov_query_invariants calls `L, _ = safe_cholesky(sq_reg, eps=self.eps, rounds=5)` (prior_bank.py:482) and drops the ok mask, then computes logdet_q = _logdet_chol(L) (:483). safe_cholesky's own docstring (numerics.py:74-78) states callers MUST mask off ok, not finiteness: on all-rounds failure cholesky_ex returns a finite PARTIAL factor, so _logdet_chol(L) is a finite-but-wrong log-det, not NaN. That finite logdet flows into per_pos = K + logdet_q at both chunked call sites (_decode_full_chunked prior_bank.py:807 and decode_ce_full_chunked :523), yielding finite logits for a non-PD covariance. The dense path (_decode_full, :774) routes the same non-PD Sigma_q through gaussian_full's safe_cholesky+ok gating (gaussian.py:375-411) which injects NaN, and with kl_max=inf safe_kl_clamp (base.py:25-27: NaN/+inf -> kl_max) maps it to +inf KL -> -inf logit. Dense and chunked therefore disagree on a non-PD query.
LOCS:
   - vfe3/model/prior_bank.py:482 (`L, _ = safe_cholesky(...)` discards ok)
   - vfe3/model/prior_bank.py:483 (logdet_q = _logdet_chol(L))
   - vfe3/model/prior_bank.py:807-808 (per_pos = K + logdet_q; kl_v) and :523 (fused CE per_pos)
   - vfe3/model/prior_bank.py:774 (dense _decode_full kl with kl_max=inf)
   - vfe3/numerics.py:74-78 (safe_cholesky docstring: callers MUST drive off ok)
   - vfe3/families/base.py:25-27 (safe_kl_clamp maps NaN/+inf -> kl_max)
   - vfe3/families/gaussian.py:375-411 (dense ok-gating -> NaN)
MINIMAL_FIX:
In vfe3/model/prior_bank.py `_full_cov_query_invariants`, capture the ok mask and force the log-det to -inf at failed positions so per_pos = K + logdet_q becomes -inf, driving kl_v = 0.5*(a_v - per_pos) to +inf and the logit to -inf at every vocab entry (matching dense). Replace lines 482-483:

    L, ok = safe_cholesky(sq_reg, eps=self.eps, rounds=5)
    logdet_q = _logdet_chol(L)                                         # (B, N)
    logdet_q = torch.where(ok, logdet_q, logdet_q.new_full((), float("-inf")))
    return diag_sq_reg, logdet_q

No signature change; the -inf propagates automatically through BOTH callers (per_pos at :807 and :523 broadcasts -inf across V). This is the smallest change that closes the dense/chunked divergence. Update the misleading docstring sentence at :475-476 that claims the mask is not replicated.
NEW_TESTS:
   - tests/test_fullcov_alpha_roadmap_2026_06_13.py: add test_full_cov_chunked_matches_dense_on_non_pd. Build a Sigma_q with one (B,N) position strongly non-PD so all 5 safe_cholesky rounds fail (e.g. a symmetric matrix with an eigenvalue << -eps*1e5, others positive), keep the rest PD. Call the dense `_decode_full` and the chunked `_decode_full_chunked` (both registered decodes) and assert both return -inf logits (torch.isneginf) at the bad position and are allclose (atol 1e-3) at the PD positions.
   - tests/test_fullcov_alpha_roadmap_2026_06_13.py: assert the all-PD value-equality pin still holds byte-for-byte (torch.where selects logdet_q when ok, so nothing changes on the pure path).
RISK_NOTES: On the normal training path the SPD retraction keeps Sigma_q PD, so ok is all-True and torch.where selects logdet_q unchanged -> byte-identical to current behavior, the 1388-test value-equality pins are unaffected. Only prior_bank.py is touched. One honest caveat: at a genuinely non-PD position the fused decode_ce_full_chunked will now produce a -inf target logit -> logsumexp(-inf)-(-inf) = NaN CE for that token, which is exactly what dense F.cross_entropy(_decode_full(...)) already does; this finding is about matching dense, not about eliminating the NaN (that would need the deferred all-rounds-fail masking the gaussian_full family carries). The pure dense path is unchanged.

====================================================================================================
C14 [low] verdict=confirmed deferred=True
TITLE: Lie exponential Frobenius clamp silently returns a stability surrogate exp(max_norm*M/||M||) with no runtime activation diagnostic
ROOT_CAUSE: stable_matrix_exp_pair (transport.py:686) rescales the algebra matrix in a no_grad block: `mat_norm = matrix.norm(...).clamp(min=1e-8); scale = (max_norm/mat_norm).clamp(max=1.0); matrix = matrix * scale` (:726-733). When ||M||_F > max_norm=15 the returned factor is exp(max_norm*M/||M||_F), NOT exp(M) - singular values/determinant of the reported group element differ from the true exponential, as the docstring at :698-706 states. There is no runtime signal that the clamp fired; the code intentionally omits the check because detecting activation needs a tensor reduction (host sync) on this hot path. The learned prior-bank gauge parameter (prior_bank.py:156) is unconstrained and stepped by the optimizer, so a drifted frame can silently cross into the surrogate regime. Regime II edge factors are soft-capped upstream (transport.py:337-345), lowering that risk, but the general learned-gauge path lacks any diagnostic.
LOCS:
   - vfe3/geometry/transport.py:686 (def stable_matrix_exp_pair)
   - vfe3/geometry/transport.py:698-706 (docstring: SAFEGUARD NOT THE EXACT OPERATOR; runtime monitor intentionally omitted)
   - vfe3/geometry/transport.py:726-733 (no_grad clamp: mat_norm, scale, matrix = matrix * scale)
   - vfe3/geometry/transport.py:337-345 (regime_ii upstream soft cap)
   - vfe3/model/prior_bank.py:156 (unconstrained learned gauge parameter)
MINIMAL_FIX:
Do NOT add an always-on host-sync reduction to the hot path or a post-step gauge retraction (that is the deferred buildout). Apply the safe opt-in diagnostic below.
SAFE_VARIANT:
Add an opt-in, default-OFF diagnostic keyword to stable_matrix_exp_pair that surfaces clamp activation only when the caller asks (accepting the host sync only then). In vfe3/geometry/transport.py extend the signature with `clamp_monitor: bool = False` (placed in the defined-bools group per the signature convention). Immediately after `scale = (max_norm / mat_norm).clamp(max=1.0)` in the no_grad block add:

    if clamp_monitor:
        frac = (scale < 1.0).float().mean()
        if bool(frac > 0):
            import warnings
            warnings.warn(
                f'stable_matrix_exp_pair: Frobenius clamp active on {float(frac):.1%} of matrices '
                f'(max_norm={max_norm}); returned factor is a surrogate, not exp(M).',
                RuntimeWarning, stacklevel=2,
            )

Default False keeps the hot path byte-identical with no host sync (the `(scale < 1.0)` reduction and `.item()` fire only under the flag). This exposes clamp activation on demand without the deferred post-step retraction. Optionally thread the same flag from build_link_transport_operators / the gauge-frame call sites so a diagnostics run can enable it.
NEW_TESTS:
   - tests/test_transport.py (or the existing stable_matrix_exp_pair test module): add test_clamp_monitor_warns_when_active - pass a matrix with ||M||_F >> max_norm and clamp_monitor=True, assert a RuntimeWarning is raised (pytest.warns); pass a small-norm matrix with clamp_monitor=True and assert no warning.
   - tests/test_transport.py: assert clamp_monitor=False (default) returns output bit-identical to the current call (no behavior change, no warning) so the pure/hot path is untouched.
RISK_NOTES: Default-OFF flag means zero change to any existing test or the hot path; the reduction/host-sync happens only when explicitly enabled. Signature grows by one defined-bool kwarg - keep the vertical alignment of names/types/defaults per the project convention. No cross-file coupling unless the caller opts to thread the flag. The genuine post-step gauge retraction / norm monitor the audit mentions remains deferred; this variant only adds observability, it does not alter the modeled operator or the equivariance-breaking footprint of the learned gauge exception.

====================================================================================================
F7 [medium] verdict=confirmed deferred=True
TITLE: Model-channel gamma and s_e_step transport the s tables under flat cocycle while the belief channel runs a non-flat connection
ROOT_CAUSE: The s-fiber (model channel) has no non-flat transport law. Both the s E-step refine (_refine_s) and the gamma pairwise energy (_gamma_energy) hard-code transport_mode="flat" (model.py:639, model.py:1150), whereas the belief E-step forwards the active connection_W/connection_M/rope into vfe_stack (model.py:767-777). Config validates lambda_gamma>=0 and s_e_step independently of transport_mode (config.py:935; transport registry check at config.py:785-791), so a run can enable a non-flat belief transport (regime_ii / regime_ii_covariant) together with lambda_gamma>0 or s_e_step=True. The two channels then run different connections and the s-channel comparison is silently non-covariant, though nothing in code labels it as such.
LOCS:
   - vfe3/model/model.py:639
   - vfe3/model/model.py:1150
   - vfe3/model/model.py:767-777
   - vfe3/config.py:935
   - vfe3/config.py:785-791
   - vfe3/config.py:1612-1622
MINIMAL_FIX:
Do NOT build a model-channel transport seam blindly (that is the theory-gated s-channel transport law). Apply the fail-closed config WARNING in VFE3Config.__post_init__ (config.py), placed just after the existing regime_ii x gaussian_full warning block that ends near config.py:1622, mirroring that block's non-breaking UserWarning style:

    # F7 (audit 2026-07-01): the s-channel (model coupling + s E-step) transports the s tables under
    # the FLAT phi-cocycle only (_gamma_energy / _refine_s pass transport_mode="flat"), regardless of
    # cfg.transport_mode. Under a NON-FLAT belief transport the belief and model channels then run
    # different connections -- the s-fiber has no non-flat transport law yet, so the model-channel
    # comparison is NOT gauge-covariant. Warn (non-breaking) so a run does not describe it as sharing
    # the active connection.
    if self.transport_mode != "flat" and (self.lambda_gamma > 0.0 or self.s_e_step):
        import warnings
        warnings.warn(
            f"transport_mode={self.transport_mode!r} is non-flat, but the model channel "
            f"(lambda_gamma={self.lambda_gamma}, s_e_step={self.s_e_step}) transports the s tables "
            "under the FLAT phi-cocycle only; the s-fiber has no non-flat transport law. The "
            "model-channel coupling is NOT gauge-covariant under this connection.",
            UserWarning, stacklevel=2,
        )

Do not change model.py:639 or model.py:1150 (the flat s-transport is the intentional current behavior and the pure path). transport_mode=="flat" is the only flat registry key, so the != test is correct.
SAFE_VARIANT:
The config UserWarning above IS the safe variant: warn (non-breaking) when transport_mode is non-flat and (lambda_gamma>0 or s_e_step=True), leaving all executable paths untouched. The full fix (a registered s-channel transport seam that threads cfg.transport_mode + connection_W/M into _gamma_energy/_refine_s) is the deferred buildout and must not be added blindly.
NEW_TESTS:
   - tests/test_config.py (or tests/test_regime_ii.py): assert pytest.warns(UserWarning, match="non-flat") for VFE3Config(transport_mode="regime_ii", lambda_gamma=1.0) and for VFE3Config(transport_mode="regime_ii_covariant", s_e_step=True).
   - Negative: assert no such warning for VFE3Config(transport_mode="flat", lambda_gamma=1.0, s_e_step=True) and for VFE3Config(transport_mode="regime_ii", lambda_gamma=0.0) with s_e_step=False (use warnings.catch_warnings / filterwarnings and inspect record).
RISK_NOTES: Confirm the exact field names in VFE3Config are lambda_gamma and s_e_step (both referenced live: config.py:935 and model.py:659 self.cfg.s_e_step). A new UserWarning can trip tests that assert zero warnings under strict filters; scope any new test with warnings.catch_warnings. Overlaps F8 (pure-path report omits lambda_gamma/s_e_step) -- coordinate so both name the same toggles. No math or pure-path change; the warning fires only under the double opt-in.

====================================================================================================
C4 [medium] verdict=confirmed deferred=False
TITLE: regime_ii_covariant oracle drops dOmega/dsigma on the detached/no-grad/direct path
ROOT_CAUSE: The autograd oracle constructs differentiable query/key covariance leaves sigma_q and sigma_k (oracle.py:98-108: sigma_q live-aliased under use_live else a detached requires_grad clone; sigma_k = sigma_q.detach() filtering / sigma_q smoothing), but the omega-builder contract is (mu_q, mu_k) only and is invoked that way at oracle.py:116 (omega = omega_builder(mu_q, mu_k)). The e_step closure _omega_builder(mu_q, mu_k) (e_step.py:454-463) then supplies the transport's covariance from the CLOSED-OVER belief tensors: sigma=belief.sigma, sigma_key=belief.sigma.detach(). regime_ii_covariant's gauge-invariant edge features read that covariance (transport.py:396-399, 456, 486-497), so torch.autograd.grad(F, [mu_q, sigma_q]) at oracle.py:147 sees Omega's sigma-dependence only when sigma_q IS belief.sigma. That holds solely on the live unrolled path (use_live=True aliases sigma_q=sigma at oracle.py:100); on detached / no_grad / direct-oracle / diagnostic calls sigma_q is a fresh clone, so dOmega/dsigma is silently dropped (probe: create_graph=False sigma_diff=0.37, create_graph=True sigma_diff=0).
LOCS:
   - vfe3/gradients/oracle.py:98-108
   - vfe3/gradients/oracle.py:112-116
   - vfe3/gradients/oracle.py:147-150
   - vfe3/inference/e_step.py:451-464
   - vfe3/geometry/transport.py:386-399
   - vfe3/geometry/transport.py:456
   - vfe3/geometry/transport.py:486-497
   - vfe3/gradients/kernels.py:233
MINIMAL_FIX:
Extend the omega-builder contract from 2 to 4 tensor args and thread the oracle's sigma leaves, mirroring the mu detach split. Three surgical edits, all confined to the mu-dependent oracle path (flat transport uses omega_builder=None and is untouched):
(1) oracle.py:116 -- change `omega = omega_builder(mu_q, mu_k)` to `omega = omega_builder(mu_q, sigma_q, mu_k, sigma_k)` (sigma_q, sigma_k already exist at oracle.py:100-108).
(2) oracle.py:78 -- widen the type hint `Callable[[torch.Tensor, torch.Tensor], ...]` to `Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...]`.
(3) e_step.py:454-463 -- change the closure signature to `def _omega_builder(mu_q, sigma_q, mu_k, sigma_k):` and inside build_belief_transport pass `sigma=sigma_q, sigma_key=sigma_k` in place of the current `sigma=belief.sigma, sigma_key=belief.sigma.detach()`. Keep connection_W / connection_M / rope kwargs as-is (build_belief_transport already accepts sigma/sigma_key; the plain regime_ii builder ignores sigma via **kwargs, so threading it there is a harmless no-op). Optionally update the (mu_q, mu_k) comment at kernels.py:233.
NEW_TESTS:
   - tests/test_regime_ii_covariant.py: add test_covariant_detached_oracle_includes_sigma_grad -- build a regime_ii_covariant model with nonzero connection_M, call the oracle/e_step with create_graph=False (detached path), and a direct sigma-threaded autograd reference; assert max_abs(grad_sigma_oracle - grad_sigma_ref) < 1e-5 (currently ~0.37). Reuse the fixtures from tests/test_regime_ii_covariant.py:366-390.
   - tests/test_regime_ii_covariant.py: regression that omega_builder is called with 4 positional args (monkeypatch or a spy) to pin the extended contract.
RISK_NOTES: omega_builder has exactly one closure (e_step.py:454) and one call site (oracle.py:116); kernels.py:233 only forwards it -- no other caller relies on the 2-arg contract, so the signature change is contained. Filtering must detach sigma on the key slot exactly as it does mu (sigma_k already = sigma_q.detach() at oracle.py:106), preserving mean-field coordinate ascent -- do not pass sigma_q for both slots under filtering. The live-unroll numeric path is byte-identical because sigma_q aliases belief.sigma there, so golden regime_ii_covariant tests should not move; verify the zero-M and none-M flat-reduction tests (transport.py fast path at :439) still pass since they never reach the builder body.

====================================================================================================
C5 [medium] verdict=confirmed deferred=True
TITLE: Exact regime_ii_covariant claim is inconsistent with the accepted gaussian_diagonal family (diagonal cone not closed under GL congruence)
ROOT_CAUSE: transport_mode='regime_ii_covariant' with family='gaussian_diagonal' is accepted (family default config.py:188; transport/family registries validated independently at config.py:785-791), and the covariant builder treats diagonal sigma as a first-class branch (transport.py:456, 488-490: torch.diag_embed then Omega Sigma Omega^T). But GL(K)-covariance is exact only when the transported covariance stays in the family. A diagonal covariance cone is NOT closed under a general congruence Omega Sigma Omega^T for non-orthogonal Omega (the off-diagonal terms are discarded on diagonal readout), so the advertised exact Route B is a controlled approximation under gaussian_diagonal. The only existing guard warns about gaussian_full numerics (config.py:1612), pushing users the opposite way; the exact end-to-end covariance-law test uses full SPD (tests/test_regime_ii_covariant.py:366-390) and the admissibility verifier already states the obstruction (tests/test_admissibility_verifier.py:51-56).
LOCS:
   - vfe3/config.py:188
   - vfe3/config.py:785-791
   - vfe3/config.py:1612-1622
   - vfe3/geometry/transport.py:456
   - vfe3/geometry/transport.py:488-490
   - tests/test_regime_ii_covariant.py:366-390
   - tests/test_admissibility_verifier.py:51-56
MINIMAL_FIX:
Make the exactness contract explicit with a non-breaking config WARNING (do NOT hard-require gaussian_full -- that would remove the numerically-stable diagonal path). In VFE3Config.__post_init__, adjacent to the existing regime_ii x gaussian_full warning near config.py:1612:

    if self.transport_mode == "regime_ii_covariant" and self.family == "gaussian_diagonal":
        import warnings
        warnings.warn(
            "transport_mode='regime_ii_covariant' with family='gaussian_diagonal': the diagonal "
            "covariance cone is NOT closed under general GL(K) congruence Omega Sigma Omega^T, so the "
            "diagonal readout is a CONTROLLED APPROXIMATION of the exact GL-covariant transport. Use "
            "family='gaussian_full' for exact Route B covariance, or an orthogonal/diagonal-preserving "
            "gauge.",
            UserWarning, stacklevel=2,
        )

Also surface the approximation in the pure-path certificate (run_artifacts._pure_path_report): add a boolean like `regime_ii_covariant_exact = (transport_mode != 'regime_ii_covariant') or (family == 'gaussian_full')` to config_toggles so reports do not present diagonal covariant as exact.
SAFE_VARIANT:
Config warn + report label (above): present diagonal regime_ii_covariant as a controlled approximation rather than removing it. The deferred/theory-gated route is either requiring family='gaussian_full' for exactness or restricting the connection/action to a diagonal-preserving (orthogonal) subgroup where the cone IS closed -- do not implement that restriction blindly.
NEW_TESTS:
   - tests/test_regime_ii_covariant.py (config-boundary test): assert pytest.warns(UserWarning, match="CONTROLLED APPROXIMATION") for VFE3Config(transport_mode="regime_ii_covariant", family="gaussian_diagonal"); assert NO such warning for family="gaussian_full".
   - tests/test_run_artifacts.py (or wherever _pure_path_report is tested): assert the new regime_ii_covariant_exact flag is False for diagonal covariant and True for full covariant / flat.
RISK_NOTES: gaussian_full already emits its own numeric warning at config.py:1612 for regime_ii_covariant; the two warnings are complementary (one per family) and both should fire only under the respective opt-in. Do not change transport.py:488-490 (the diagonal branch is a legitimate approximation and part of a working path). New warning may need warnings scoping in tests that assert clean config construction. Overlaps C6 (both feed the reports/pure-path labeling pass).

====================================================================================================
C6 [medium] verdict=confirmed deferred=False
TITLE: Frobenius holonomy deviation is reported as a curvature scalar but is frame-dependent under non-orthogonal GL gauges
ROOT_CAUSE: holonomy_deviation and holonomy_deviation_sampled return the Frobenius departure ||H_ijk - I||_F (metrics.py:50-91, 696-724), labeled 'Curvature proxy'. Under a GL(K) frame change the triangle holonomy transforms by conjugation H -> g_i H g_i^{-1}; ||H-I||_F = ||g(H-I)g^{-1}||_F is invariant only for orthogonal/unitary g, so for the non-compact block_glk frames the Frobenius scalar is NOT a gauge-invariant curvature (Nakahara 2003 section 10.5; Baez and Muniain 1994). The conjugation-invariant observable is the Wilson trace Re Tr(H)/K, which already exists at metrics.py:737-788 (holonomy_wilson_sampled) and is logged in diagnostics (model.py:1579-1585, 1900-1901). Default reports still headline the Frobenius value (run_artifacts.py:835-843 title 'Holonomy deviation (curvature proxy)'; figures.py:998-1000, 1839-1840, 3157).
LOCS:
   - vfe3/metrics.py:50-91
   - vfe3/metrics.py:696-724
   - vfe3/metrics.py:737-788
   - vfe3/model/model.py:1579-1585
   - vfe3/model/model.py:1900-1901
   - vfe3/run_artifacts.py:835-843
   - vfe3/viz/figures.py:998-1000
   - vfe3/viz/figures.py:1839-1840
   - vfe3/viz/figures.py:3157
MINIMAL_FIX:
Pure labeling; change NO math and remove NO metric. (a) metrics.py:56 docstring: state 'frame-dependent (Frobenius; conjugation-invariant only for orthogonal gauges) -- use holonomy_wilson_sampled for gauge-invariant curvature.' (b) run_artifacts.py:841 title string 'Holonomy deviation (curvature proxy)' -> 'Holonomy deviation (frame-dependent Frobenius)'; and keep/promote the Wilson trajectory (holonomy_wilson is already logged) as the gauge-invariant panel. (c) figures.py:1000 and 1839 legend labels: append '(frame-dependent)' to the ||H-I||_F entries and '(gauge-invariant)' to the Wilson entries at figures.py:998, 1091, 1840; update the metric-label dict entries at figures.py:3157 similarly. No change to metrics.py:91 / :724 / :773 computations or to the pinned holonomy_deviation key semantics.
NEW_TESTS:
   - tests/test_metrics.py: test_wilson_trace_is_conjugation_invariant -- build a random Omega, apply a non-orthogonal per-vertex GL conjugation g_i Omega_ij g_j^{-1}, assert holonomy_wilson_sampled deviation_mean is unchanged (atol ~1e-5) while holonomy_deviation_sampled['mean'] CHANGES under the same conjugation. This pins the invariant/frame-dependent distinction the labels now assert.
   - tests/test_metrics.py: assert the label/docstring split is intentional by checking both functions still return ~0 on a flat cocycle (existing flatness certificate).
RISK_NOTES: Label-only edits must not touch the metric keys 'holonomy_deviation' / 'holonomy_wilson' that train.py:496-542, run_artifacts.py:779/968/991, and test_regime_ii.py:238 read -- keep dict keys and numeric values byte-identical. figures.py label strings are LaTeX; preserve raw-string escaping. The new invariance test needs a per-vertex conjugation helper; ensure g_i are invertible (e.g. I + small random) so H stays finite.

====================================================================================================
C7 [medium] verdict=confirmed deferred=False
TITLE: Plain regime_ii bilinear transport is gauge-fixed, not covariant, and must be labeled as such in reports/pure-path
ROOT_CAUSE: Plain regime_ii builds edge coefficients delta_ij^a = mu_i^T W^a mu_j from RAW coordinates (transport.py:315; builder transport.py:226-236) using the model's fixed-coordinate connection_W (model.py:223-224). Under a gauge change mu_i -> g mu_i the coefficient maps to mu_i^T g^T W^a g mu_j, invariant iff g^T W^a g = W^a for all group elements g; for full GL(K) the only such constant W is 0, so any trained nonzero W breaks gauge invariance. This is pinned by tests/test_regime_ii.py:409-435 (nonzero W -> holonomy/invariance deviation > 1e-4). The math is correct and user-accepted; the risk is purely that reports/docs could group plain regime_ii with the covariant Route B and read a gauge-FIXED bilinear edge as an exact covariant connection.
LOCS:
   - vfe3/geometry/transport.py:226-236
   - vfe3/geometry/transport.py:242-269
   - vfe3/geometry/transport.py:315
   - vfe3/model/model.py:223-224
   - tests/test_regime_ii.py:409-435
MINIMAL_FIX:
Labeling only; do not alter the regime_ii math or fast path. (a) Add an explicit one-line header to the _build_regime_ii docstring at transport.py:242: 'GAUGE-FIXED / NON-COVARIANT: delta_ij = mu_i^T W mu_j is gauge-invariant only at W=0; a trained nonzero W breaks gauge equivariance. For exact GL-covariant transport use transport_mode="regime_ii_covariant" (Route B).' (b) In run_artifacts._pure_path_report, add a report field classifying the active transport, e.g. transport_covariance_class = {'flat':'covariant (flat)', 'regime_ii':'gauge-fixed (non-covariant)', 'regime_ii_covariant':'covariant', 'regime_ii_link':'gauge-fixed', 'regime_ii_link_charted':'covariant'}.get(cfg.transport_mode, cfg.transport_mode), and include it in config_toggles so no report presents regime_ii as covariant.
NEW_TESTS:
   - tests/test_run_artifacts.py (or test_pure_path): assert the new transport_covariance_class field equals 'gauge-fixed (non-covariant)' for a regime_ii config and 'covariant' for regime_ii_covariant / 'covariant (flat)' for flat. No numeric assertion needed; this pins the report label.
   - No new transport test needed -- test_regime_ii.py:409-435 already pins the underlying non-covariance the label describes.
RISK_NOTES: Docstring and report-field edits only; no executable transport change, so all regime_ii golden tests are unaffected. Ensure the class-map dict covers every registered transport key (flat, regime_ii, regime_ii_covariant, regime_ii_link, regime_ii_link_charted) with a sensible default so a newly registered mode does not KeyError. Overlaps C5/C6/F8 (all touch _pure_path_report / config_toggles) -- batch these report-label additions in one pass to avoid conflicting edits to the same dict.

====================================================================================================
F8 [medium] verdict=confirmed deferred=False
TITLE: Pure-path report omits gauge/model-channel toggles and collapses distinct purity axes into one boolean
ROOT_CAUSE: _pure_path_report (vfe3/run_artifacts.py:736-780) derives on_pure_path from a 7-entry pure_flags dict (lines 754-762) covering only the canonical-FE/decode axes (attention entropy, flat transport, constant lambda_alpha, static lambda_beta, prior-bank decode, no head mixer, unweighted attention). It never reads gauge_transport (config.py:79, default 'on'), pos_rotation (config.py:172, 'none'), rope_full_gauge (174), rope_on_value (182), lambda_gamma (295), or s_e_step (355). These change the executed belief/gauge path (RoPE is threaded into the stack at model.py:775-776; s_e_step folds an anchor before vfe_stack at model.py:750-755; lambda_gamma drives the model-channel path). So on_pure_path can read True while gauge or model-channel settings materially alter the run, and the single boolean conflates the free-energy/decode axis with the gauge-equivariance axis.
LOCS:
   - vfe3/run_artifacts.py:754-762 (pure_flags, 7 entries)
   - vfe3/run_artifacts.py:764 (on_pure_path=all(pure_flags.values()))
   - vfe3/run_artifacts.py:766-775 (config_toggles, 8 entries, omits the six)
   - vfe3/config.py:79 gauge_transport; :172 pos_rotation; :174 rope_full_gauge; :182 rope_on_value; :295 lambda_gamma; :355 s_e_step
   - tests/test_reporting_additions.py:130-136 (asserts exact top-level key set + on_pure_path/pure_flags)
MINIMAL_FIX:
In _pure_path_report (vfe3/run_artifacts.py), keep the existing pure_flags/on_pure_path unchanged (free-energy/decode axis, preserves the pinned flat_transport assertion). Add a SECOND axis just before the return: gauge_flags = {"learned_gauge_transport": cfg.gauge_transport == "on", "no_positional_rotation": cfg.pos_rotation == "none", "no_model_channel_coupling": cfg.lambda_gamma == 0.0 and not cfg.s_e_step}. Extend the returned dict with "gauge_flags": gauge_flags and "on_gauge_pure_path": all(gauge_flags.values()). Extend config_toggles (lines 766-775) with the six reporting entries: "gauge_transport": cfg.gauge_transport, "pos_rotation": cfg.pos_rotation, "rope_full_gauge": bool(cfg.rope_full_gauge), "rope_on_value": bool(cfg.rope_on_value), "lambda_gamma": float(cfg.lambda_gamma), "s_e_step": bool(cfg.s_e_step). Direct attribute access is fine (VFE3Config always defines these); the whole call is already wrapped in try/except at run_artifacts.py:712. Base the gauge-purity flag on pos_rotation=='none' rather than the rope sub-toggles, because rope_full_gauge/rope_on_value are inert when RoPE is off (they are reported in config_toggles for transparency only).
NEW_TESTS:
   - tests/test_reporting_additions.py::test_pure_path_report_structure_and_flags: update the expected set to {"on_pure_path","pure_flags","config_toggles","converged_stress","gauge_flags","on_gauge_pure_path"}; add gauge_transport="on", pos_rotation="none", rope_full_gauge=False, rope_on_value=True, lambda_gamma=0.0, s_e_step=False to BOTH the pure and impure SimpleNamespace fixtures (they are now read); assert rep["on_gauge_pure_path"] is True on pure.
   - tests/test_reporting_additions.py: add a case flipping cfg.gauge_transport="off" (or s_e_step=True) and assert on_gauge_pure_path is False while on_pure_path stays True — proving the axes are independent.
RISK_NOTES: The existing test asserts the EXACT top-level key set (tests/test_reporting_additions.py:131) and builds cfg from a partial SimpleNamespace; adding new keys AND new cfg attribute reads both break that test unless the test is updated in the same change — must be done together. Purely a reporting-dict change: no training/forward semantics touched, no other module consumes _pure_path_report output besides this test and the JSON artifact. Keeping on_pure_path/pure_flags byte-identical preserves back-compat for any downstream JSON reader.

====================================================================================================
F9 [medium] verdict=confirmed deferred=True
TITLE: Reporting extractors materialize full (B,N,V) logits/probabilities; default figure pass runs them unguarded
ROOT_CAUSE: The inference-path extractors allocate dense full-vocab tensors: per_unit_eval_nats/belief_ce_bank call model(tokens) producing (B,N,V) logits and softmax them (vfe3/viz/extract.py:139, :199-203), and vocab_prediction_stats softmaxes (B,N-1,V) per batch (extract.py:873-876). finalize_run runs generate_figures by default when cfg.generate_figures is truthy (run_artifacts.py:726-730), which fans out to these extractors. On a large run (e.g. vocab 50257, N=1024, batch 16) the (B,N,V) logits+probs pair is ~6-7 GB, so finalization/reporting can become the memory peak even when the training path is chunked. A full streaming/chunked extractor rewrite is the ideal fix but is a large buildout the audit lists as deferred (item 6).
LOCS:
   - vfe3/viz/extract.py:139 (per_unit_eval_nats full (B,N,V) logits)
   - vfe3/viz/extract.py:199-203 (belief_ce_bank full logits + softmax)
   - vfe3/viz/extract.py:873-876 (vocab_prediction_stats full (B,N-1,V) softmax)
   - vfe3/run_artifacts.py:726-732 (finalize_run generate_figures default-on, try/except-wrapped)
   - vfe3/viz/report.py:138-152 (each extractor individually _safe-wrapped)
MINIMAL_FIX:
DEFERRED buildout — do NOT write the streaming extractor API blindly. Apply the safe_variant guard instead (see safe_variant).
SAFE_VARIANT:
Add one opt-in config field in vfe3/config.py near generate_figures: `force_large_figures: bool = False` (defined bool). In finalize_run (vfe3/run_artifacts.py) replace the `if getattr(cfg, 'generate_figures', True):` gate at line 726 with a memory-budget guard: compute approx_gb = 8.0 * int(cfg.vocab_size) * int(cfg.max_seq_len) * int(cfg.batch_size) / 1e9  # fp32 logits+probs (B,N,V) peak; if generate_figures is on AND approx_gb > 8.0 AND not cfg.force_large_figures, logger.warning('skipping publication figures: est full-vocab peak ~%.1f GB exceeds 8 GB guard; set force_large_figures=True to override', approx_gb) and skip; otherwise generate as today inside the existing try/except. This bounds the surprise memory peak on large runs while leaving an explicit large-run opt-in, and does not touch the extractor internals or the pure path.
NEW_TESTS:
   - tests/test_reporting_additions.py: add test_finalize_run_skips_figures_over_memory_guard — construct a cfg with a large vocab_size/max_seq_len/batch_size so approx_gb>8 and force_large_figures=False, call finalize_run with a tiny model + monkeypatched vfe3.viz.report.generate_figures spy, assert generate_figures was NOT invoked and a warning was logged.
   - tests/test_reporting_additions.py: add the force_large_figures=True counterpart asserting generate_figures IS invoked (or attempted) despite the large estimate.
RISK_NOTES: Existing try/except at run_artifacts.py:726-732 plus per-extractor _safe wrappers (report.py:138-152) already prevent an OOM from destroying numeric results, so this is memory-peak/UX, not data-loss — severity stays medium. Adding force_large_figures is a new config field; keep it default False so no existing run behavior changes at small sizes (approx_gb below 8 keeps default-on behavior byte-identical). Do not lower the threshold below realistic small-run footprints or CI figure tests that build small models will start being skipped. The 8 GB constant should sit comfortably under the 32 GB RTX 5090 budget.

====================================================================================================
C9 [medium] verdict=confirmed deferred=False
TITLE: Viz belief extractors (_encode_one, belief_bank) drift from forward: missing precision-bias fold and s_e_step anchor
ROOT_CAUSE: forward folds the precision-weighted-attention bias into log_prior right before vfe_stack, after the s_e_step anchor (vfe3/model/model.py:750-755 then :762 model._fold_precision_bias(log_prior, beliefs.sigma)). The shared extractor helper _encode_one (vfe3/viz/extract.py:42-57) replays the s_e_step anchor (51-53) but returns log_prior WITHOUT the fold (55), so every extractor built on it (e_step_belief_trace, across_layer_belief_trace, numerical_health, converged_state) scores a different attention prior than forward under precision_weighted_attention=True. belief_bank (extract.py:245-305; body 272-286) is worse: it applies neither the s_e_step anchor nor the precision fold before vfe_stack, so it also diverges under s_e_step=True. belief_ce_bank (extract.py:206-210) is the correct positive control that does both. The centralized helper already exists (model._fold_precision_bias); the extractors just don't call it.
LOCS:
   - vfe3/model/model.py:750-755 (forward s_e_step anchor), :762 (forward precision fold)
   - vfe3/viz/extract.py:51-53 (_encode_one s_e_step refine), :55 (raw log_prior, no fold)
   - vfe3/viz/extract.py:272-286 (belief_bank: no anchor, no fold)
   - vfe3/viz/extract.py:206-210 (belief_ce_bank correct positive control)
   - vfe3/viz/extract.py:576 (attention_entropy_cov_gap folds precision itself after _encode_one — double-fold risk if fold is centralized into _encode_one)
MINIMAL_FIX:
1) In _encode_one (vfe3/viz/extract.py, after line 55 `log_prior = model._attention_log_prior(n, token_ids.device)`) add: `log_prior = model._fold_precision_bias(log_prior, belief.sigma)`  # no-op unless precision_weighted_attention; belief.sigma is post-s_e_step so it matches forward's beliefs.sigma at model.py:762. 2) In attention_entropy_cov_gap remove the now-redundant fold at extract.py:576 (`log_prior = model._fold_precision_bias(log_prior, belief.sigma)`) to avoid double-folding, since _encode_one now returns the folded prior (the fold is already documented in that function's docstring at lines 564-566). 3) In belief_bank mirror belief_ce_bank: after `beliefs = beliefs._replace(phi=model._apply_pos_phi(beliefs.phi))` (line 273) add `if cfg.s_e_step:` then `s_mu1, s_sigma1 = model._refine_s(tokens, beliefs.phi); beliefs = beliefs._replace(mu=s_mu1, sigma=s_sigma1)`, and after `log_prior = model._attention_log_prior(n, device)` (line 275) add `log_prior = model._fold_precision_bias(log_prior, beliefs.sigma)`. cfg is already bound at line 264. Do NOT change any training/forward code — these are report-fidelity edits only.
NEW_TESTS:
   - tests/test_reporting_additions.py (or the existing viz-extract test module): add test_extractor_belief_matches_forward_under_precision_weighted_attention — build a tiny VFEModel with precision_weighted_attention=True, run one sequence through converged_state / e_step_belief_trace and through the forward belief path, assert the extractor beta/belief equals the forward belief (they diverge before the fix). Reuse the small-model construction pattern already in the reporting tests.
   - tests/test_reporting_additions.py: add test_belief_bank_matches_forward_under_s_e_step — with s_e_step=True (and optionally precision_weighted_attention=True), assert belief_bank's converged mu/sigma for a sequence equals belief_ce_bank's / forward's for the same tokens (currently mismatched because belief_bank skips the anchor+fold).
   - Regression guard: assert both new tests are byte-identical to the pre-fix values on the DEFAULT config (precision_weighted_attention=False, s_e_step=False) so the fold/anchor are proven no-ops on the pure path.
RISK_NOTES: model._fold_precision_bias returns log_prior unchanged when precision_weighted_attention is False (model.py:1397-1398) and the s_e_step branch only fires when s_e_step is True, so on the default/pure path all edits are exact no-ops — the 1388-test suite on default configs is unaffected. The one cross-consumer hazard is the double-fold in attention_entropy_cov_gap: step 2 MUST be applied together with step 1 or that extractor will fold twice under precision_weighted_attention=True. _encode_one is shared by 5 extractors (e_step_belief_trace, across_layer_belief_trace, numerical_health, converged_state, attention_entropy_cov_gap); all except attention_entropy_cov_gap consume the raw returned prior and should receive the folded one, so centralizing in _encode_one is the intended fix. No overlap with F8 (different function) beyond both living in the reporting/viz layer.

====================================================================================================
F10 [medium] verdict=confirmed deferred=True
TITLE: Generation re-runs full forward per token and plain Regime II builds unchunked dense (B,N,N,K,K) transports
ROOT_CAUSE: Two independent peak-memory/compute hazards on opt-in paths. (1) VFEModel.generate() (vfe3/model/model.py:1311-1339) has no incremental belief/KV cache: for each of max_new_tokens it slices context = seq[:, -max_seq_len:] and calls self.forward(context) over the WHOLE window, keeps only logits[:, -1, :], then grows the sequence with seq = torch.cat([seq, next_token], dim=-1). Cost is O(max_new_tokens * full_forward(N)) and the docstring itself (lines 1298-1300) calls it 'the correct-but-slow first version'. (2) The plain regime_ii transport builder _build_regime_ii (vfe3/geometry/transport.py:300-354) materializes the full dense delta (B,N,N,n_gen) line 315, delta_mat (B,N,N,K,K) line 324, exp_delta (B,N,N,K,K) line 347-350, and omega (B,N,N,K,K) line 353 all at once -- no query-axis chunking -- even though its sibling _build_regime_ii_covariant already chunks the identical all-pairs structure with _regime_ii_query_chunk() (transport.py:371-383) and a query-chunk loop (transport.py:462-521), proving the chunking is available and value/grad-equivalent (there is no cross-query reduction). Both are opt-in (regime_ii transport_mode and generate()), so no default-run corruption; this is peak-memory risk on long-context generation and non-flat transport runs.
LOCS:
   - vfe3/model/model.py:1311-1339 (generate loop: forward(context) per token, torch.cat growth)
   - vfe3/model/model.py:1298-1300 (docstring confirms full re-run, incremental reuse deferred)
   - vfe3/geometry/transport.py:300-354 (_build_regime_ii unchunked dense delta/delta_mat/exp_delta/omega)
   - vfe3/geometry/transport.py:315,324,347-350,353 (dense (B,N,N,*) allocations)
   - vfe3/geometry/transport.py:371-383 (_regime_ii_query_chunk helper available to reuse)
   - vfe3/geometry/transport.py:462-521 (covariant query-chunk loop proving equivalence)
   - tests/test_regime_ii_covariant.py:220-254 (existing chunk-size + chunked-equivalence tests to mirror)
MINIMAL_FIX:
Port the EXISTING covariant chunking into _build_regime_ii (vfe3/geometry/transport.py). Leave the flat fast-path (lines 300-301) and the fac/exp_phi/exp_neg_phi setup (lines 306-307) unchanged. Replace the current unchunked block (lines 314-354, from `mu_k = mu_key ...` through the final `return {...}`) with a query-chunk loop that reuses the module-level _regime_ii_query_chunk() helper (already defined at transport.py:371-383) and the covariant loop shape (transport.py:462-521):

    mu_k = mu_key if mu_key is not None else mu

    generators = group.generators                                # (n_gen, K, K)
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_dim    = max(block_dims) if block_dims is not None else None

    B, N, K = mu.shape[0], mu.shape[1], mu.shape[-1]
    cols  = torch.arange(N, device=mu.device)                    # key indices (self-edge mask)
    chunk = _regime_ii_query_chunk(B, N, K)

    omega_chunks: List[torch.Tensor] = []
    for i0 in range(0, N, chunk):
        i1        = min(i0 + chunk, N)
        exp_phi_c = exp_phi[:, i0:i1]                            # (B, C, K, K)
        delta_c   = cocycle_relaxation * torch.einsum(
            "bik,akl,bjl->bija", mu[:, i0:i1], connection_W, mu_k)   # (B, C, N, n_gen)
        rows      = torch.arange(i0, i1, device=delta_c.device)
        self_edge = rows.unsqueeze(-1) == cols.unsqueeze(0)     # (C, N)
        delta_c   = delta_c.masked_fill(self_edge.view(1, i1 - i0, N, 1), 0.0)
        delta_mat_c = torch.einsum("bija,akl->bijkl", delta_c, generators)
        fro_sq      = delta_mat_c.pow(2).sum(dim=(-2, -1), keepdim=True)
        delta_mat_c = delta_mat_c * torch.rsqrt(1.0 + fro_sq / (delta_soft_cap * delta_soft_cap))
        exp_delta_c, _ = stable_matrix_exp_pair(
            delta_mat_c, skew_symmetric=group.skew_symmetric, only_forward=True,
            block_dims=block_dims, exp_dim=exp_dim,
        )
        omega_chunks.append(
            torch.einsum("bikl,bijlm,bjmn->bijkn", exp_phi_c, exp_delta_c, exp_neg_phi))

    omega = torch.cat(omega_chunks, dim=1) if len(omega_chunks) > 1 else omega_chunks[0]
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}

This exactly mirrors the covariant self-edge masking (transport.py:499-504) and the tail cat (transport.py:520). List is already imported (used by the covariant builder). The self-edge mask (delta_ii:=0) is preserved by comparing the GLOBAL query index rows=arange(i0,i1) against cols, so the eye-based masking at old lines 320-322 is subsumed. When chunk>=N the loop is a single iteration bit-for-bit identical to the current path (B=1 diagnostic builds collapse to one chunk).
SAFE_VARIANT:
For the generate() half (the true deferred buildout -- a full incremental KV/belief cache is a large change and MUST NOT be built blindly), do NOT implement a cache. Apply only a fail-closed memory-budget guard: at the top of VFEModel.generate() (vfe3/model/model.py, just after the policy-knob validation at line 1310, before `seq = token_ids`), estimate the per-forward peak from cfg.max_seq_len, K, n_heads, V and, if it exceeds a documented threshold, emit a single warnings.warn(...) naming the quadratic full-forward-per-token cost and pointing at the deferred incremental-cache TODO -- mirroring the existing D3 memory-estimator / AMP-precision warning pattern added in commit f3387b9. This changes no outputs and preserves the pure path.
NEW_TESTS:
   - tests/test_regime_ii.py: add test_regime_ii_chunked_matches_unchunked mirroring tests/test_regime_ii_covariant.py:228-254 -- build _build_regime_ii Omega (nonzero connection_W, requires_grad, block_glk, e.g. B=2,N=5,K=4) once with T._REGIME_II_CHUNK_ELEMS = 10**12 (single chunk) and once with T._REGIME_II_CHUNK_ELEMS = 1 (forced size-1 chunks), restoring the saved constant in a finally; assert torch.allclose(omega_one, omega_chunk, atol=1e-5, rtol=1e-5) AND allclose on the connection_W .grad from a .sum().backward() on each. This pins value+gradient equivalence of the new chunking.
   - tests/test_regime_ii.py: add test_regime_ii_query_chunk_used asserting _regime_ii_query_chunk(64, 128, 20) < 128 (OOM-scale config chunks) and _regime_ii_query_chunk(1, 3, 4) == 3 (tiny build = one chunk), matching tests/test_regime_ii_covariant.py:220-225.
RISK_NOTES: Chunking touches only the non-flat regime_ii transport (transport_mode='regime_ii'); the default flat builder _build_flat (transport.py:223) and the pure Regime-I path never reach _build_regime_ii, so the 1388-test suite's default paths are untouched. The change is value/grad-equivalent only up to fp32 op-reorder across chunk boundaries (covariant sibling pins atol/rtol 1e-5, not bit-exact) -- any existing golden test on _build_regime_ii that asserts bit-identity across a >1-chunk shape could drift; existing regime_ii tests use small B/N (e.g. B=2,N=3 in tests/test_regime_ii.py:_phi_mu) which collapse to a single chunk and stay bit-for-bit, so they are safe. The gauge-invariance-breaking test test_regime_ii_edge_factor_breaks_gauge_invariance_for_nonzero_W (tests/test_regime_ii.py:409) is unaffected (single-chunk shapes). Do not alter _regime_ii_query_chunk's _REGIME_II_LIVE_TRANSIENTS (=5) constant tuned for the covariant builder: the plain builder holds fewer simultaneous transients, so the existing budget is conservative (over-chunks slightly) but never wrong. The generate() guard must reuse the existing warnings pattern, not raise, to avoid breaking generation smoke tests.

====================================================================================================
C10 [medium] verdict=confirmed deferred=False
TITLE: Ablation/scaling resume freshness ignores max_tokens (stale capped-run cache)
ROOT_CAUSE: max_tokens is a loader seam (train-split token cap), not a VFE3Config field, so it never lands in config.json (RunArtifacts writes asdict(cfg)+dataset). _cell_is_current compares only dataset + VFE3Config, so a smoke cell trained with max_tokens=10000 and a later full run (max_tokens=None) that share model config + dataset produce byte-identical config.json and the stale capped cell is served as [CACHED].
LOCS:
   - ablation.py:1296-1314 (_cell_cfg_dict includes max_steps not max_tokens)
   - ablation.py:1559-1591 (_cell_is_current compares dataset + serialized VFE3Config only)
   - ablation.py:1669 (call site)
   - ablation.py:1514-1526 (run_single result dict omits max_tokens)
   - ablation.py:1271,1472-1473 (max_tokens threaded to get_loader train cap)
   - scaling.py:601-611 (_cell_cfg_dict)
   - scaling.py:614-624 (_cell_is_current)
   - scaling.py:651 (call site)
   - scaling.py:684-688 (scaling_cell.json write)
   - scaling.py:667,740-741 (max_tokens flow)
MINIMAL_FIX:
ablation.py: (1) In run_single, add max_tokens to the returned result dict so it lands in the marker; after line 1525 add `"max_tokens": (int(max_tokens) if max_tokens is not None else None),`. (2) In _cell_is_current add keyword-only `max_tokens: Optional[int] = None` after `max_steps`; after the existing config==built comparison, replace the final `return saved_obj.get("config") == built` with: `if saved_obj.get("config") != built: return False`, then read the marker `try: marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8")) except Exception: return False`, then `cur = int(max_tokens) if max_tokens is not None else None`, then `return marker.get("max_tokens", None) == cur`. (3) At the call site line 1669 pass `max_tokens=max_tokens`. scaling.py: (1) In the scaling_cell.json dict (684-688) add `"max_tokens": (int(max_tokens) if max_tokens is not None else None),`. (2) Change _cell_is_current signature to `def _cell_is_current(run_dir: Path, cfg: VFE3Config, dataset: str, max_tokens: Optional[int] = None) -> bool:`; after confirming dataset+config match, read scaling_cell.json (guarded try/except -> return False) and `return cellmeta.get("max_tokens", None) == (int(max_tokens) if max_tokens is not None else None)`. (3) At line 651 pass `max_tokens=max_tokens`. Do NOT touch scaling.py line ~521 (user WIP ROUTES grid).
NEW_TESTS:
   - tests/test_ablation_tackon.py: new test builds a run_dir, writes config.json = {"dataset": ds, "config": asdict(VFE3Config(**ablation._cell_cfg_dict(overrides, seed=6, max_steps=1)))} and a marker ablation_result.json carrying max_tokens=1000; assert ablation._cell_is_current(run_dir, overrides, seed=6, dataset=ds, max_steps=1, max_tokens=1000) is True and the same call with max_tokens=None is False
   - tests/test_scaling_mup.py: analogous test writing summary.json + config.json + scaling_cell.json(max_tokens=1000); assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) True and max_tokens=None False
RISK_NOTES: New kwarg defaults None so all existing callers and the test_ablation_tackon markers (which never set max_tokens) stay backward-compatible: a full-data resume (cur None == get None) still hits cache, an old marker missing the key against a capped run re-runs (fail-closed, safe). scaling _cell_is_current now also requires scaling_cell.json to exist -> very old cells without it re-run (over-invalidation, safe). No pure-path impact (ablation/scaling are experiment drivers, not the model path).

====================================================================================================
C11 [medium] verdict=confirmed deferred=False
TITLE: Run-artifact writes are not atomic (partial JSON / corrupt checkpoint on crash or Windows lock)
ROOT_CAUSE: All three writers write in place. A crash, power loss, or a Windows open-handle/PermissionError mid-write leaves a truncated config.json/summary.json, a corrupt step_N.pt, or an unreadable best_model.pt. The audit notes this host has already hit Windows checkpoint overwrite/open-handle failures.
LOCS:
   - vfe3/run_artifacts.py:74-78 (save_json path.write_text direct)
   - vfe3/run_artifacts.py:100-107 (maybe_save_best torch.save(model.state_dict(), self.best_path) direct)
   - vfe3/run_artifacts.py:219-233 (save_checkpoint torch.save to self.ckpt_dir/step_N.pt direct)
   - vfe3/run_artifacts.py:26-32 (imports csv/json/logging/math/dataclasses/pathlib/typing/torch; no os)
MINIMAL_FIX:
vfe3/run_artifacts.py: add `import os` to the import block. Add a module-level helper `def _atomic_replace(final: Path, tmp: Path, retries: int = 5, delay: float = 0.2) -> None:` that loops `for i in range(retries): try: os.replace(tmp, final); return; except PermissionError: if i == retries - 1: raise; time.sleep(delay)` (import time locally or at top) so Windows best_model.pt lock collisions get retry/backoff. save_json: write to `tmp = self.run_dir / (name + ".tmp")` via `tmp.write_text(json.dumps(obj, indent=2, default=str))` then `_atomic_replace(path, tmp)`; return path. maybe_save_best: `tmp = self.best_path.with_suffix(".pt.tmp"); torch.save(model.state_dict(), tmp); _atomic_replace(self.best_path, tmp)`. save_checkpoint: `tmp = path.with_suffix(".pt.tmp"); torch.save({...}, tmp); _atomic_replace(path, tmp); return path`. Same-directory temp makes os.replace an atomic rename on one volume.
NEW_TESTS:
   - tests/test_run_artifacts.py: test_writes_are_atomic_no_temp_left - after save_json, maybe_save_best, save_checkpoint, assert no '*.tmp' files remain in run_dir/ckpt_dir and each target loads (json.loads / torch.load weights_only=True) cleanly
   - tests/test_run_artifacts.py: test_best_model_overwrite_replaces - call maybe_save_best twice with improving ppl; assert best_model.pt reloads to the second state_dict (os.replace over an existing file succeeds)
RISK_NOTES: torch.save to .pt.tmp then os.replace is transparent to load_checkpoint / finalize_run (they read the final name). test_save_checkpoint_is_loadable (line 80-85) still passes since save_checkpoint returns the final path. The retry helper must re-raise non-PermissionError. Adding `import os` cannot affect the pure model path.

====================================================================================================
C12 [medium] verdict=confirmed deferred=False
TITLE: Sigma-gate artifact writer trusts checkpoint_id as a filename path component
ROOT_CAUSE: checkpoint_id is interpolated directly into the output filename, so a value containing os.sep, a drive letter, or '..' (e.g. '../../evil') makes os.path.join escape out_dir and write outside the intended artifact directory.
LOCS:
   - vfe3/inference/sigma_gate.py:167-174 (write_sigma_gate_artifact signature)
   - vfe3/inference/sigma_gate.py:181 (payload stores raw checkpoint_id)
   - vfe3/inference/sigma_gate.py:182 (path = os.path.join(out_dir, f"{checkpoint_id}.json"))
   - vfe3/inference/sigma_gate.py:25-27 (imports json, os, typing; no re)
MINIMAL_FIX:
vfe3/inference/sigma_gate.py: add `import re` to the import block. In write_sigma_gate_artifact, after `os.makedirs(out_dir, exist_ok=True)` slugify: `slug = re.sub(r"[^A-Za-z0-9._-]", "_", checkpoint_id).strip("._") or "artifact"`, then build `path = os.path.join(out_dir, f"{slug}.json")`. Keep the RAW checkpoint_id inside payload (line 181 unchanged) so provenance is preserved; only the filename is neutralized. Optionally assert containment: `assert os.path.commonpath([os.path.abspath(out_dir), os.path.abspath(path)]) == os.path.abspath(out_dir)`.
NEW_TESTS:
   - tests/test_sigma_gate.py: test_write_artifact_slugs_checkpoint_id - write_sigma_gate_artifact({'status':'PASS'}, checkpoint_id='../../evil', spec_commit='x', seeds=(6,), out_dir=tmp_path); assert the returned path's parent resolves to tmp_path (no escape) and the loaded JSON's checkpoint_id == '../../evil' (raw id preserved)
   - tests/test_sigma_gate.py: test_write_artifact_separator_id - checkpoint_id='a/b:c' writes a single file directly under out_dir (no nested 'a' directory)
RISK_NOTES: verify_gate_artifact reads by explicit path and config Guard 4 references the artifact by its stored path string, so slugifying only the filename does not break the PASS-record lookup. Preserving raw checkpoint_id in payload keeps measure_sigma_gate / existing test_sigma_gate assertions on the record intact. `import re` is inert elsewhere.

====================================================================================================
C15 [low] verdict=confirmed deferred=False
TITLE: Ablation label sanitization collides distinct cells into one run dir
ROOT_CAUSE: _sanitize is a lossy char-replace: distinct raw labels 'a=b', 'a b', 'a/b' all collapse to 'a_b', so their run dirs collide and a resumed/additive sweep overwrites or merges distinct cells.
LOCS:
   - ablation.py:1594-1599 (_sanitize maps '=', ' ', '/', '\\', '..', ':' all to '_')
   - ablation.py:1664 (run_dir = sweep_dir / _sanitize(label))
   - ablation.py:1515,1690-1692 (raw label already persisted in marker result['label'])
   - ablation.py:42-56 (imports; hashlib NOT imported)
   - tests/test_ablation_tackon.py:18 (test derives cell dir via ablation._sanitize itself)
MINIMAL_FIX:
ablation.py: add `import hashlib` to the import block. Change _sanitize to append a stable short hash of the RAW label so distinct labels get distinct dirs while staying deterministic: after `out = out.lstrip("._") or "_"`, add `h = hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]` and `return f"{out}__{h}"`. The raw label is already persisted in the marker (result['label'] at 1515), so no marker change is needed. _sanitize stays deterministic in label, so the resume [CACHED] path (run_dir = sweep_dir / _sanitize(label)) still finds the same dir on re-run.
NEW_TESTS:
   - tests/test_ablation_tackon.py: test_sanitize_distinct_labels_do_not_collide - assert ablation._sanitize('a=b') != ablation._sanitize('a b') != ablation._sanitize('a/b'), and ablation._sanitize('kappa=2') == ablation._sanitize('kappa=2') (determinism)
RISK_NOTES: Existing test_ablation_tackon.py tests compute cell dirs THROUGH ablation._sanitize (line 18) and read labels from marker JSON, not dir names, so they stay green (including test_int_float_spellings_stay_distinct). Behavior change: dir names for already-completed sweeps change, so a resume of a PRE-fix sweep re-runs its cells once (fail-safe; _collect_sweep_results globs '*/ablation_result.json' so old-named dirs still appear in the leaderboard). No model-path impact.

====================================================================================================
C16 [low] verdict=confirmed deferred=False
TITLE: sigma_gate_measure.py hardcodes a machine-specific absolute checkpoint path
ROOT_CAUSE: The click-to-run CONFIG dict ships with an absolute, user-and-run-specific checkpoint path, so a fresh clone/worktree or another machine runs against a nonexistent or stale checkpoint. The fail-closed guard for an empty value already exists at lines 93-94.
LOCS:
   - sigma_gate_measure.py:31 (checkpoint= r"C:\Users\chris and christine\Desktop\V3_Transformer\vfe3_runs\...\step_15000.pt")
   - sigma_gate_measure.py:93-94 (main() already raises ValueError when CONFIG['checkpoint'] is empty)
MINIMAL_FIX:
sigma_gate_measure.py:31: set `checkpoint="",` (empty string). The existing guard at 93-94 (`if not cfg["checkpoint"]: raise ValueError("set CONFIG['checkpoint'] to an operating-point checkpoint path before running")`) already fails closed with a clear message, so no other change is needed. Keep the inline comment 'REQUIRED: path to the operating-point checkpoint (.pt)'.
NEW_TESTS:
   - tests/test_sigma_gate.py: test_measure_script_checkpoint_not_hardcoded - `import sigma_gate_measure; assert not sigma_gate_measure.CONFIG['checkpoint']` (guards against re-introducing a machine-absolute default)
RISK_NOTES: This is committed source (git status shows only scaling.py dirty), not an uncommitted WIP toggle, so editing it does not touch the user's live config. The user re-points CONFIG['checkpoint'] per run; emptying it just makes the script fail loudly instead of silently measuring a stale checkpoint. No test imports sigma_gate_measure with a live path at module load, so blanking it cannot break the suite.

====================================================================================================
C18 [low] verdict=confirmed deferred=False
TITLE: fisher_trace and generate() violate the tensor/number/bool/Optional argument-order convention
ROOT_CAUSE: The mandated keyword-only order is defined floats, defined ints, defined bools, then Optional, then **kwargs. fisher_trace places Optional[bool] diagonal before the defined float eps; generate places Optional top_k/top_p before the defined bool greedy.
LOCS:
   - vfe3/metrics.py:244-249 (fisher_trace: diagonal: Optional[bool] before eps: float)
   - vfe3/model/model.py:1269-1279 (generate: top_k/top_p Optional before greedy: bool)
   - vfe3/model/model.py:1642 (fisher_trace(out.sigma, diagonal=_diag, eps=cfg.eps) - kwargs)
   - tests/test_metrics.py:148,151 (fisher_trace positional sigma only)
   - generate_efe.py:88, vfe3/train.py:794, tests/test_generate.py:38-158 (all generate() calls use kwargs for temperature/top_k/top_p/greedy)
MINIMAL_FIX:
vfe3/metrics.py fisher_trace (247-249): swap the two keyword-only lines so eps (defined float) precedes diagonal (Optional), preserving names/defaults and vertical alignment: `eps: float = 1e-12` then `diagonal: Optional[bool] = None`. vfe3/model/model.py generate (1276-1279): reorder the keyword-only block to `temperature: float = 1.0`, then `greedy: bool = False`, then `top_k: Optional[int] = None`, then `top_p: Optional[float] = None` (defined float, defined bool, then Optionals); keep the inline shape/purpose comments aligned. No body changes. All call sites pass these by keyword (or omit them), so the reorder is behavior-neutral.
NEW_TESTS:
   - tests/test_metrics.py: extend the fisher_trace test to also call by keyword in the new order - `fisher_trace(sigma, eps=1e-9, diagonal=False)` and assert it matches the positional-sigma result
   - tests/test_generate.py: add test_generate_kwargs_order - `model.generate(prompt, max_new_tokens=2, greedy=False, top_k=2, top_p=0.9, temperature=0.7)` runs without TypeError
RISK_NOTES: Both params are keyword-only (after `*`), so no positional caller breaks; verified every call site (model.py:1642, train.py:794, generate_efe.py:88, test_generate.py, test_metrics.py) uses kwargs or positional-tensor-only. Sibling metrics fns (lines 208-209, 373-374, 458-459, 908-909) share the diagonal-before-eps pattern but are OUT of C18's cited scope; leave them (surgical) rather than mass-reorder. The optional signature-order lint the audit suggests is deferred, not required to close this finding.
