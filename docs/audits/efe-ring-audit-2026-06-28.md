# EFE Ring Experiment Audit - 2026-06-28

## Scope

This audit reads the live feature-branch tree as of 2026-06-28, with the worktree already dirty before
the audit began. It covers the EFE ring experiment path and adjacent policy scorer path:
`efe_ring_experiment.py`, `vfe3/inference/ring_task.py`, `vfe3/inference/policy.py`,
`vfe3/model/model.py`, `vfe3/config.py`, `tests/test_ring_task.py`, `tests/test_efe_scorer.py`,
`tests/test_policy_registry.py`, and `tests/test_generate.py`.

The theory baseline is the pre-registration spec
`docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md`, the amendment record
`docs/research/active-inference/2026-06-28-prereg-amendments.md`, and the Research wiki pages
`[[Expected Free Energy]]`, `[[Active Inference]]`, and `[[VFE Transformer Program]]`. The code review
below treats executable source and probe output as decisive; comments and docs are used only to define
the intended contract.

## Executive Verdict

The current ring harness is not ready for the official sealed run. The core EFE scorer and ring tests
pass, and the post-amendment action-constrained path is coherent as a v1 pragmatic reranker, but three
experiment-contract defects can invalidate an official verdict: the training budget is 7500 steps while
the sealed spec says 15000, episode sampling is biased rather than uniform over `g != s0`, and the
aggregate gate can print `GO` after silently shrinking the three-seed set to any nonempty admitted
subset. Two additional defects affect the advertised generic `generate(policy_mode=...)` policy seam,
especially on CUDA, but do not hit the current `ring_task.run_episodes()` task-preference path.

## Confirmed Findings

### F1. The sealed training budget is set to 7500, not 15000

**Severity:** high.

**Evidence:** `efe_ring_experiment.py:25-29` labels `CONFIG` as the pre-registration surface and sets
`steps=7500`. The spec seals "15k optimizer steps" at
`docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md:405`, repeats the
15k recipe at lines 427 and 473, and the amendment document says the 15k-step budget is unchanged at
`docs/research/active-inference/2026-06-28-prereg-amendments.md:53`. A probe printed
`config_steps 7500`.

**Impact:** an official run from the current script would not be the sealed run. It would train each
checkpoint for half the pre-registered budget while still presenting the result as the pre-registered
surface.

**Fix:** restore `CONFIG["steps"] = 15000` before any official verdict. Add a small regression test that
pins the sealed constants against the amendment record or against a single `SEALED_CONFIG` object.

### F2. `sample_episodes()` biases the goal distribution

**Severity:** high.

**Evidence:** `efe_ring_experiment.py:46-52` samples `s0` and `goals` independently, then replaces
collisions by `(goal + 1) % M`. The spec states that each episode samples an initial state and a goal
`g != s0` uniformly on the ring at
`docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md:357`. A probe over
160000 episodes produced `delta_counts [0, 20125, 9943, 10043, 9829, 10024, 10189, 9908, 9871, 9931,
9843, 10239, 10097, 10192, 9802, 9964]`, so clockwise-neighbor goals (`delta=1`) occur roughly twice
as often as the other nonzero deltas.

**Impact:** the official paired test set is not the pre-registered uniform ring task. It overrepresents
one-step clockwise goals, which can inflate or distort success rates and directional action priors even
though arms remain paired on the same biased episodes.

**Fix:** sample a nonzero offset uniformly, for example `offset = randint(1, M)` and
`goals = (s0 + offset) % M`, or rejection-resample collisions until no `goal == s0` remains. Pin exact
uniformity with a deterministic test over a large fixed sample or a small exhaustive sampler.

### F3. The aggregate verdict can declare `GO` with fewer than three admitted seeds

**Severity:** medium-high.

**Evidence:** `efe_ring_experiment.py:176-188` excludes seeds that miss predictive adequacy and appends
only admitted seeds. The aggregate block at `efe_ring_experiment.py:196-208` computes
`all_seeds_go` over `admitted` and prints the GO verdict if all admitted seeds pass. There is no
requirement that `len(admitted) == len(cfg["seeds"])`. The spec says the synthetic-task checkpoint
family is a three-seed set and that "the three admitted seeds supply" the comparisons and floor band at
`docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md:405`.

**Impact:** if one or two seeds miss adequacy, the script can still issue the full GO verdict on the
remaining subset. That is not silent in the per-seed JSON, but the top-level `verdict` string can
overstate the result.

**Fix:** add an aggregate admission gate. Either require all three sealed seeds to clear adequacy before
`all_seeds_go` can be true, or change the verdict to a clearly non-final "PARTIAL / ADEQUACY SHORTFALL"
when any sealed seed is excluded.

### F4. The advertised opt-in `generate()` policy path fails under the default policy preference

**Severity:** medium.

**Evidence:** `vfe3/config.py:410` defaults `policy_preference` to `"task"`. `VFEModel._policy_select`
calls `get_preference(self.cfg.policy_preference)(self.prior_bank)` with no task goal at
`vfe3/model/model.py:1335`. The registered task preference requires keyword-only `goal` at
`vfe3/inference/policy.py:135-140`. A probe constructing
`VFE3Config(policy_mode="efe_one_step")` and calling `generate()` failed with
`TypeError _pref_task() missing 1 required keyword-only argument: 'goal'`.

**Impact:** turning on the advertised policy mode in the generic generation path is not a valid
configuration unless the user also knows to switch `policy_preference="flat"`. This does not affect
the ring harness task arm, because `run_episodes()` builds the ring task preference directly at
`vfe3/inference/ring_task.py:247-250`.

**Fix:** make the generic generate path reject `"task"` and `"held_out_predictive"` with an explicit
config error unless a context provider is supplied, or set the only generic generate preference to
`"flat"` and reserve task/data preferences for harnesses that pass goals or `p_data`.

### F5. Generic policy preference builders allocate CPU tensors and will break CUDA generation

**Severity:** medium.

**Evidence:** `_pref_flat` returns `torch.full((V,), -math.log(V))` with no device at
`vfe3/inference/policy.py:120-131`. `_pref_task` creates `torch.zeros(V)` or `torch.zeros(B, V)` on
CPU at `vfe3/inference/policy.py:150-162`. `_policy_select` passes that preference directly into the
policy scorer at `vfe3/model/model.py:1335-1339`, where scorer tensors come from the model/device. The
ring harness works around the flat preference with `.to(device)` at `vfe3/inference/ring_task.py:249-250`,
but the generic `generate()` path does not. CUDA was not available in this audit environment, so this
was source-confirmed rather than GPU-executed.

**Impact:** on the user's RTX 5090 path, `generate()` with `policy_mode != "none"` and a registry
preference built by `policy.py` can hit a CPU/CUDA device mismatch. Even on CPU, this hides the fact
that preference builders do not follow the model device contract.

**Fix:** make preference builders device-aware, either by deriving device from `prior_bank` or by
passing `device` explicitly from `_policy_select`. Add a CUDA-conditional test; if CUDA is unavailable,
the same test should at least assert the returned preference device matches the model device on CPU.

### F6. Tests do not pin the pre-registration invariants that failed here

**Severity:** low-medium.

**Evidence:** focused tests pass, but searches found no tests for `sample_episodes`, no test that
`efe_ring_experiment.CONFIG["steps"]` remains 15000, and no test for the aggregate admission gate.
`tests/test_efe_scorer.py:168` covers `generate()` only with `policy_preference="flat"`, so it does not
catch the default opt-in `policy_preference="task"` failure.

**Impact:** the current suite is strong on algebraic scorer correctness and smoke behavior, but weak on
the experiment contract. This is why the official-run blockers above survived a green suite.

**Fix:** add small tests for sealed constants, uniform non-start goal sampling, aggregate verdict under
excluded seeds, default opt-in generate failure messaging, and preference device placement.

## Theory And Purity Notes

The v1 scorer is a pragmatic preference-matching reranker, not an epistemic active-inference result.
That is correctly stated in the spec at
`docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md:457` and in the wiki's
`[[Expected Free Energy]]` page. The ring task can validate that the action layer uses a preference to
steer a closed loop, but it cannot validate information-seeking active inference because
`PolicyScore.epistemic` is identically zero at this operating point.

The Phase 1 path is also not the mathematically pure KL-to-prior decode path. `train_ring_checkpoint`
sets `use_prior_bank=False` at `vfe3/inference/ring_task.py:166-170`, matching the operating-point
ablation in the spec, and Phase 4 is the deferred pure-prior-bank confirmation. Claims should continue
to say "linear-decode operating point" until Phase 4 exists.

The action-constrained candidate menu, the finite preference floor, and the distance-graded
preference are justified by the amendment record and the ring task implementation. They are not hidden
impurities now, but they must remain visible in any result write-up because they narrow the conclusion:
the experiment tests the amended ring-control policy space, not a generic top-K token continuation
policy.

## Verification

Targeted probe command results:

```text
config_steps 7500
config_seeds (6, 23, 64)
config_candidate_mode actions
delta_counts [0, 20125, 9943, 10043, 9829, 10024, 10189, 9908, 9871, 9931, 9843, 10239, 10097, 10192, 9802, 9964]
delta_1_vs_delta_2 20125 9943
zero_delta_count 0
default_policy_generate_error TypeError _pref_task() missing 1 required keyword-only argument: 'goal'
cuda_available False
```

Focused path tests:

```text
python -m pytest tests/test_ring_task.py tests/test_efe_scorer.py tests/test_policy_registry.py tests/test_generate.py --junitxml=C:\tmp\vfe3-efe-ring-audit-20260628.xml
36 passed in 1.58s
JUnit: tests=36 failures=0 errors=0 skipped=0
```

Full suite:

```text
python -m pytest --junitxml=C:\tmp\vfe3-full-audit-20260628.xml
pytest: 1304 passed, 1 skipped, 1 xpassed, 181 warnings in 214.92s
JUnit: tests=1306 failures=0 errors=0 skipped=1 time=214.912
```

The warnings are existing warning-heavy coverage around guarded configuration combinations and are not
new evidence for the EFE ring findings.

## Consolidated Punch List

1. Restore `CONFIG["steps"] = 15000` before any official sealed run.
2. Fix `sample_episodes()` to draw uniformly over `g != s0`.
3. Gate the top-level verdict on all three sealed seeds being admitted, or print a non-final partial
   verdict when any seed misses adequacy.
4. Harden the generic `generate(policy_mode=...)` preference contract so default opt-in config cannot
   fail with a missing task goal.
5. Make preference builders device-aware and add CUDA-conditional coverage.
6. Add pre-registration-invariant tests for sealed constants, sampler distribution, admission gating,
   and generic policy-seam device/config behavior.
