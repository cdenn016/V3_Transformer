# Active-Inference Purity Audit

Date: 2026-06-28

Scope: this audit inspected the active-inference and expected-free-energy paths in the live tree, with emphasis on theoretical purity, mathematical semantics, and reachable code behavior. The inspected paths were `vfe3/inference/policy.py`, `vfe3/inference/ring_task.py`, `vfe3/inference/sigma_gate.py`, `vfe3/model/model.py`, `vfe3/config.py`, `efe_ring_experiment.py`, `sigma_gate_measure.py`, the active-inference pre-registration notes, and the current sigma-gate artifact.

The short version is that the v1 ring-task path is an honest pragmatic one-step EFE controller under a task-defined preference, but the generic generation path is not yet a meaningful active-inference language-generation policy. The epistemic path is still gated out by both implementation and measurement: `sigma_mc` has no live consumer, and the available pure operating-point sigma artifact is a FAIL. These are not cosmetic issues. They define the boundary between a mathematically pure, narrow pragmatic experiment and broader active-inference claims that the code cannot yet support.

## What Is Sound

The code preserves a pure default path. `policy_mode="none"` leaves normal generation outside the policy scorer, and the active-inference machinery is opt-in. The policy scorer also exposes the right algebraic decomposition for the v1 point-belief regime: risk, ambiguity, and epistemic diagnostics are returned separately, and at one step with the default `likelihood_entropy` ambiguity, `epistemic = predictive_entropy - ambiguity` is identically zero. In that regime, the score reduces to pragmatic preference matching rather than pretending to contain a live information-gain term.

The ring environment is also now closer to the intended closed-loop experiment than the earlier broken harness. The action menu is the three control actions, the ring goal sampler no longer doubles mass on the clockwise neighbor, and the finite preference floor prevents infinite forward KL from turning the policy posterior into `nan`. Focused active-inference tests and the full suite currently pass under the machine-readable counts listed below.

## Findings

### F1. Generic `generate(policy_mode="efe_one_step")` collapses to the base model under the only safe preference

Severity: high for claim scope, medium for code correctness.

`VFEConfig` correctly rejects `task` and `held_out_predictive` preferences for generic `generate`, because that API has no per-episode goal or held-out distribution to pass into the preference builder. The only allowed preference is therefore `flat`. With a flat preference, `_efe_terms` computes a constant risk equal to `log(V)` for every candidate, and the default ambiguity and epistemic terms cancel at one step. The policy posterior is then just the base candidate prior:

```
q(pi | o, C) proportional to p(pi) exp(-gamma * constant) = p(pi).
```

A live probe confirmed this collapse: greedy flat-policy generation returned the same token as base greedy generation, the flat score had only floating-point roundoff variation, and the policy posterior equaled the base softmax over the candidate menu. This is mathematically pure as an inert fallback, but it means generic language generation has no preference-directed active-inference behavior today.

Fix direction: keep this guard, but name and document the generic flat path as inert log-probability control. If generic generation is meant to become active inference, add a real preference-provider interface keyed by context, task, or evaluation state before making any language-generation claim.

### F2. The active-inference spec provenance is unsafe while the spec file is untracked

Severity: high for reproducibility.

`sigma_gate_measure.spec_commit()` tries `git log -1 --format=%H -- <spec>` and falls back to `git rev-parse HEAD` if the spec has no tracked history. The active-inference implementation spec is currently untracked, so the recorded `spec_commit` in `vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json` is the repository HEAD (`9959790da271f6021de8d3940be14dee2b360695`), not a commit that proves the spec content used for the run. The current artifact is therefore bound to the code revision, but not to the actual untracked pre-registration text.

This matters because the sigma gate is supposed to be a pre-registered epistemic unlock. A gate artifact should not be able to appear content-bound when the governing spec is outside git.

Fix direction: for official artifacts, fail if the spec is untracked or dirty, or record a content SHA256 plus dirty-tree status. Prefer tracking or force-adding the spec before any official artifact is treated as provenance.

### F3. The current ring-experiment script is configured as a smoke-scale run while the docs still describe the sealed 15k budget

Severity: high for experimental interpretation, low for ordinary editability.

`efe_ring_experiment.CONFIG` currently uses `steps=3500`, while the pre-registration amendment note says the official constants still include a 15k training budget for the three sealed seeds. This is not automatically a code bug, because the project intentionally keeps click-to-run configs editable for smoke runs. The shortcoming is that the script can still produce a normal-looking sealed verdict payload from a non-official budget unless the operator reads the config carefully.

Fix direction: add an explicit `run_kind` or `official` flag. Official mode should assert the sealed constants before writing a GO/NO-GO verdict; smoke mode should write a smoke-labeled result and suppress official claims.

### F4. The aggregate GO gate can pass on a nonempty admitted subset

Severity: high for experimental correctness.

The ring experiment aggregates over `admitted` seed results. If at least one seed passes predictive adequacy, the code computes aggregate metrics over that subset and can emit a GO verdict from that subset alone. It does not require `len(admitted) == len(cfg["seeds"])`.

For a sealed three-seed experiment, this is too permissive. A model family that only satisfies the adequacy precondition on one seed has not passed the intended three-seed experiment, even if that one admitted seed beats the baselines.

Fix direction: require every sealed seed to be admitted for official GO/NO-GO. If fewer are admitted, emit a separate `ADEQUACY_SHORTFALL` or `PARTIAL` status and report the admitted-subset statistics as diagnostic only.

### F5. Several invalid policy configurations pass validation and fail later at runtime

Severity: medium for code correctness.

The config validator checks the registered policy mode, horizon positivity, precision positivity, safe generic preference, and sigma-gate artifact status. It does not check all policy contracts. Live probes showed:

```
policy_score_terms=("nonsense",)        -> config ok, KeyError during generate
policy_mode="efe_one_step", horizon=2   -> config ok, ValueError during generate
policy_mode="efe_rollout"               -> config ok, NotImplementedError during generate
policy_top_k > vocab_size               -> config ok, torch topk runtime error
```

These failures are not mathematically dangerous, because they fail before silently producing a wrong policy. They are still correctness gaps: invalid active-inference configurations should be rejected at construction with clear messages.

Fix direction: validate that `policy_score_terms` is nonempty and contained in `{"risk", "ambiguity", "epistemic"}`; validate the mode/horizon pairing; reject `efe_rollout` until the belief/key-value cache exists; and ensure `policy_top_k <= vocab_size`.

### F6. The sigma gate is implemented as a measurement and config guard, but there is no live `sigma_mc` active-inference consumer

Severity: medium-high for feature completeness.

`vfe3/inference/sigma_gate.py` can measure and verify a sigma-gate artifact, and `VFEConfig` correctly refuses `policy_sigma_ambiguity_validated=True` unless the artifact is a PASS. However, `_amb_sigma_mc` in `vfe3/inference/policy.py` always raises `RuntimeError`, and the generic policy path has no config field for selecting `ambiguity_mode="sigma_mc"`. The flag therefore cannot unlock a sigma-based ambiguity estimator in generation today.

This is acceptable only if the flag is understood as a prerequisite for future Phase 3 code, not as a live epistemic-policy switch.

Fix direction: either implement the gated `sigma_mc` ambiguity consumer and expose it through config, or rename the flag/docs so they state that the artifact is a prerequisite record only.

### F7. The current pure operating-point sigma artifact is a FAIL, so epistemic active inference is not available at this checkpoint

Severity: high for theory claims, not a source-code defect.

The official pure operating-point artifact at `vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json` reports:

```
status: FAIL
sigma_ce_spearman: 0.10511072131114393
spearman_ci: [0.09563755024571517, 0.11476804144481124]
permutation_floor: 0.008340782302998864
monotone: false
sigma_binned_ece: 0.030328254087537527
spearman_min threshold: 0.2
ece_max threshold: 0.05
```

The signal is positive, but it is well below the 0.2 Spearman threshold and fails strict monotonicity. Since the v1 one-step point-belief path has zero information gain without a validated sigma ambiguity, this artifact keeps the epistemic arms gated out. The correct claim is that Phases 1-2 exercise pragmatic preference control, not validated epistemic active inference.

Fix direction: keep the FAIL artifact visible and treat sigma-derived arms as reported-only. Do not promote epistemic claims until a PASS artifact exists and a live `sigma_mc` consumer is implemented.

### F8. The ring task validates pragmatic action control, not the pure prior-bank route or broad language-generation active inference

Severity: medium for scope clarity.

The ring checkpoint helper configures `use_prior_bank=False` and `use_head_mixer=False`. That is a coherent operating point for the current controlled task, but it means this experiment does not certify the pure KL prior-bank path or the richer hierarchical active-inference story. The task is fully observed, deterministic, one-step action steering with a distance-graded utility. That is a valid pragmatic active-inference slice; it is not evidence that the generic transformer generation path is now active inference.

Fix direction: keep the current result scoped to "one-step pragmatic closed-loop control." Add a separate Phase 4 experiment for the prior-bank route if the goal is to certify the mathematically pure hierarchical path.

### F9. The policy branch ignores call-time sampler knobs and replaces them with config policy knobs

Severity: medium for API correctness.

`generate()` respects `temperature`, `top_k`, and `top_p` only on the normal path. Once `policy_mode != "none"`, `_policy_select` uses `policy_top_k` and `policy_precision` from config. This design is defensible, because the policy posterior has its own precision parameter, but the API surface can mislead callers who pass generation-time sampler knobs and expect them to matter.

Fix direction: document this explicitly, or reject mixed call-time sampler knobs when `policy_mode != "none"` unless the policy path is extended to consume them deliberately.

## Verification

Focused active-inference slice:

```
python -m pytest tests/test_efe_scorer.py tests/test_policy_registry.py tests/test_ring_task.py tests/test_efe_ring_experiment.py tests/test_sigma_gate.py tests/test_phase0_forward_beliefs.py tests/test_generate.py --junitxml=C:\tmp\vfe3-active-inference-purity-audit-20260628.xml
```

Machine-readable result: `tests=62 failures=0 errors=0 skipped=0`. Console result: `62 passed in 5.08s`.

Full suite:

```
python -m pytest -x --junitxml=C:\tmp\vfe3-active-inference-full-audit-20260628.xml
```

Machine-readable result: `tests=1321 failures=0 errors=0 skipped=1`. Console result: `1319 passed, 1 skipped, 1 xpassed, 181 warnings in 253.79s`.

## Bottom Line

The active-inference code is currently pure in the narrow sense that the default path is untouched, the v1 EFE algebra honestly degenerates to pragmatic control under the point-belief one-step regime, and the epistemic path is gated rather than faked. The main shortcomings are scope and experimental-contract issues: generic `generate(policy_mode=...)` is inert under the only safe preference; the sealed-run provenance and admission rules are not yet strong enough; invalid policy configs reach runtime; and the sigma/epistemic path remains both unimplemented as a live consumer and empirically failed at the current pure operating point.

## Resolution (2026-06-28)

Worked on branch `fix/active-inference-purity-audit-20260628`. Three findings were genuine code defects fixable without touching the author's live config, and were fixed; the rest are either intentional design respected by the project conventions or scope statements that the code already documents.

Three were fixed. F5 (invalid policy configs reaching runtime) is now caught at construction in `VFE3Config.__post_init__`: `policy_score_terms` must be a nonempty subset of `{risk, ambiguity, epistemic}`, `policy_mode="efe_one_step"` requires `policy_horizon == 1`, and `policy_top_k <= vocab_size` is enforced once a scorer is enabled. `efe_rollout` deliberately still validates at construction, because it is a registered placeholder that raises a clear `NotImplementedError` at dispatch and an existing test pins that registered keys validate; its dispatch error is already a clean fail-fast, not a silent wrong result. F9 (call-time sampler knobs silently ignored under a policy scorer) now raises in `generate()` when `temperature`/`top_k`/`top_p` are supplied with `policy_mode != "none"`, while `greedy` stays honored. F2 (spec provenance) was addressed by tracking the pre-registration spec in git and by hardening `spec_commit()` so it returns a content-bound stamp (`untracked:<sha12>`, `<commit>+dirty:<sha12>`, or the bare commit when tracked and clean) rather than silently falling back to the repository HEAD; a gate artifact can no longer appear commit-bound while the governing spec is untracked or dirty. The three fixes added four tests and the focused active-inference slice is `tests=66 failures=0 errors=0`.

Two were waived as intentional. F3 (smoke-scale vs sealed budget) is the author's editable click-to-run `CONFIG`, which the project intentionally keeps editable for smoke runs; the analogous finding was already waived in the prior `efe-ring-audit-2026-06-28.md`, and project convention is not to re-deliberate config values. F4 (aggregate GO over an admitted seed subset) was explicitly waived by the author in that same prior audit, so the subset-admission behavior is left unchanged by request.

Four are scope or empirical statements with no code defect. F1 (generic flat `generate` collapses to the base candidate prior) is the honest inert log-probability-control fallback the docstrings already describe, and no language-generation efficacy is claimed; the new F9 guard makes the inert policy path fail loudly on misuse rather than silently. F6 (no live `sigma_mc` consumer) is by design: the flag is a prerequisite record only, `sigma_mc` raises, and both the config comment and the policy docstring already state this. F7 (the pure operating-point sigma artifact is FAIL) is empirical, not a source defect; the epistemic arms stay reported-only and gated, which is correct. F8 (the ring task certifies one-step pragmatic control, not the prior-bank route or broad language-generation active inference) is the correct scope, with the prior-bank route left to a separate future experiment.
