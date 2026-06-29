# Deep Multi-Agent Audit - 2026-06-27 Continuation

## Scope

This audit covered the whole `origin/main` Python codebase in the isolated worktree
`C:\tmp\V3_Transformer-deep-audit-20260627`, branch `audit/deep-audit-20260627`, at commit
`5e88afc` (`Merge pull request #140 from cdenn016/fix/codex-audit-20260627`). The active user
checkout at `C:\Users\chris and christine\Desktop\V3_Transformer` was not modified; it contained
uncommitted experiment WIP in `ablation.py`, `scaling.py`, `train_vfe3.py`, and
`vfe3_scaling_results/`.

Before creating the worktree, `git fetch` refreshed `origin/main`; the remote log showed:

```text
5e88afc (origin/main, origin/HEAD) Merge pull request #140 from cdenn016/fix/codex-audit-20260627
c7b0dc0 docs: digest vfe3 ablation/scaling results + GL(K) manuscript-fit recommendation
94f599a (main) Merge pull request #139 from cdenn016/fix/codex-audit-20260627
a4b5d2a fix(audit): apply verified Codex audit-2026-06-27 fixes
```

The audit used five parallel expert investigators followed by one independent verifier. The verifier
re-read source and judged reachability against the current code. Historical files
`docs/audits/audit-2026-06-27.md` and `docs/audits/audit-2026-06-27-investigation.md` were treated as
background only; current source at `5e88afc` was authoritative. The research wiki was consulted for
context on the VFE Transformer Program, the 2026-06-27 gauge-transport ablation suite, and GL(K)
gauge-equivariant attention.

## Summary

- Investigator findings: 25.
- Verifier-confirmed findings: 25.
- Refuted findings: 0.
- Inconclusive findings: 0.
- Confirmed high findings: 1.
- Confirmed medium findings: 8.
- Confirmed low findings: 16.
- Test suite: `tests=1268`, `failures=0`, `errors=0`, `skipped=1`; pytest stdout reported
  `1266 passed, 1 skipped, 1 xpassed, 181 warnings`.

The highest-risk issue is in the `regime_ii_covariant` oracle path: the autograd oracle creates live
sigma leaves, but the omega-builder closure only accepts the live mean leaves and closes over
`belief.sigma`, even though the registered transport declares `needs_sigma=True`. That can truncate or
misrepresent covariance-dependent gradients for Route B. Medium findings cluster around missed warnings
for the string detach route, skipped-step scheduler/EMA state, s-channel transport semantics, hard-coded
Gaussian/KL prior-bank decode behavior, and avoidable host/device synchronization.

## Investigator Findings

### 1. code-reviewer

#### Default Training Path Uses a Learned Linear Readout

**Location:** `vfe3/config.py:378`  
**Severity:** high as reported; verifier adjusted to low  
**Evidence:** `use_prior_bank: bool = False`; `train_vfe3.py` sets `use_prior_bank = False`;
`PriorBank` allocates `output_proj_weight`; linear decode returns `x @ pb.output_proj_weight.T`.  
**Fix:** Make the prior-bank/KL decoder the default path, or keep the learned linear projection behind
an explicit experimental decode toggle.

#### String Detach Misses the Linear-Decode Freeze Warning

**Location:** `vfe3/model/model.py:323`  
**Severity:** medium  
**Evidence:** the linear-decode freeze warning checks `cfg.detach_e_step`, while the forward path uses
`cfg.effective_e_step_gradient` and detaches when it equals `"detach"`.  
**Fix:** Gate the warning on `cfg.effective_e_step_gradient == "detach"` and add a regression test for
the string route.

#### Trusted Resume Can Fall Back to Unsafe Pickle Loading

**Location:** `vfe3/run_artifacts.py:287`  
**Severity:** medium as reported; verifier adjusted to low  
**Evidence:** safe `torch.load(..., weights_only=True)` is attempted first, but
`trust_resume_checkpoint=True` falls back to `weights_only=False`.  
**Fix:** Keep the trusted legacy path separate from normal resume, or make the trust boundary more
explicit in tooling.

#### Dataset Cache Paths Accept Unvalidated Path Components

**Location:** `vfe3/data/datasets.py:43`  
**Severity:** low  
**Evidence:** `dataset` and `split` are interpolated into cache filenames before loading cached tensors.  
**Fix:** Whitelist accepted dataset and split values or reject path separators, drive prefixes, and `..`.

### 2. debugger / theory

#### Covariant Oracle Cannot Pass Sigma Leaves Into the Omega Rebuild

**Location:** `vfe3/inference/e_step.py:426`  
**Severity:** high  
**Evidence:** `_omega_builder(mu_q, mu_k)` forwards live means but closes over `belief.sigma`; the oracle
creates `sigma_q` and `sigma_k`; `regime_ii_covariant` is registered with `needs_sigma=True` and consumes
covariance features.  
**Fix:** Extend the oracle builder interface to pass `sigma_q` and `sigma_k` into covariance-dependent
transports, detaching the key covariance only for filtering.

#### Scheduler and EMA Advance After a Skipped Optimizer Step

**Location:** `vfe3/train.py:409`  
**Severity:** medium  
**Evidence:** a non-finite unscaled loss sets `skip_step`, clears gradients, and skips optimizer stepping,
but `scheduler.step()` and later `ema.update(model)` still run.  
**Fix:** Have `train_step` report whether an optimizer step occurred, and gate scheduler and EMA updates on
that flag.

#### Route-B Connection Norm Is Computed But Dropped From CSV Metrics

**Location:** `vfe3/train.py:861`  
**Severity:** low  
**Evidence:** diagnostics compute `connection_m_norm`, but the CSV conditional whitelist includes
`connection_w_norm` and omits `connection_m_norm`.  
**Fix:** Add `connection_m_norm` to the CSV whitelist.

#### Ablation Baseline No Longer Matches the Click-To-Run Operating Point

**Location:** `ablation.py:89`  
**Severity:** low  
**Evidence:** `train_vfe3.py` and `ablation.py` encode different `embed_dim` baselines, and ablation cells
copy `BASELINE_CONFIG`.  
**Fix:** Either derive ablation baselines from the click-to-run config or name them as independent
historical operating points.

### 3. refactoring-specialist

#### Model-Channel Transport Ignores Configured Transport

**Location:** `vfe3/model/model.py:1047`  
**Severity:** medium  
**Evidence:** `_refine_s` passes `transport_mode="flat"` and `_gamma_energy` builds flat transport, while
the q-channel path threads `connection_W`, `connection_M`, and RoPE arguments.  
**Fix:** Decide whether the s/gamma channel should share configured transport; then either thread the same
transport state through or reject configs that imply unsupported sharing.

#### Prior-Bank Decode Bypasses Family And Divergence Registries

**Location:** `vfe3/model/prior_bank.py:357`  
**Severity:** medium  
**Evidence:** diagonal decode calls `get_family("gaussian_diagonal")`; full decode calls
`get_family("gaussian_full")`; config allows only Gaussian families for `use_prior_bank=True` but still
uses a warning for divergence mismatch.  
**Fix:** Either dispatch prior-bank decode through selected family/divergence registries or hard-error on
unsupported mismatches.

#### Click-To-Run Configs Are Duplicated And Divergent

**Location:** `train_vfe3.py:76`  
**Severity:** medium as reported; verifier adjusted to low  
**Evidence:** `train_vfe3.py`, `ablation.py`, and `scaling.py` encode different baselines and optimizer
choices.  
**Fix:** Move shared baselines into a builder or explicitly label each script as an independent operating
point.

### 4. performance-engineer

#### Dense Full-Logit Decode Is Still the Config Default

**Location:** `vfe3/config.py:378`  
**Severity:** high as reported; verifier adjusted to low  
**Evidence:** raw `VFE3Config()` defaults to `use_prior_bank=False` and `decode_mode="diagonal"`, so the
dense path can materialize logits; current `train_vfe3.py` overrides to `decode_mode="diagonal_chunked"`.  
**Fix:** Make dataclass defaults chunked for large-vocab training, or require explicit dense-decode opt-in.

#### Full-Covariance Cholesky Retries Synchronize CUDA Before Retrying

**Location:** `vfe3/numerics.py:83`  
**Severity:** medium as reported; verifier adjusted to low  
**Evidence:** `safe_cholesky` checks `bool(ok.all())` inside retry logic; full-covariance Gaussian paths call
it.  
**Fix:** Split a fast no-retry path from exceptional repair, or use fixed device-side retry logic.

#### Training Synchronizes the Loss Every Optimizer Step

**Location:** `vfe3/train.py:326`  
**Severity:** medium  
**Evidence:** the default single-step path converts `loss.detach()` to a Python float every step for the
non-finite guard.  
**Fix:** Move the finite-loss guard to a lower-sync path or make it an opt-in debug hardening mode.

#### Evaluation Synchronizes Once Per Validation Batch

**Location:** `vfe3/train.py:461`  
**Severity:** medium  
**Evidence:** evaluation converts both valid-token count and CE to host scalars inside the batch loop.  
**Fix:** Accumulate CE times count and token counts as device tensors, then transfer once after the loop.

#### Validation Diagnostics Bypass the Chunked CE Path

**Location:** `vfe3/train.py:561`  
**Severity:** medium  
**Evidence:** per-position validation diagnostics call `model(val_tok)` and dense
`F.cross_entropy(vlog.reshape(...))`, materializing `(B,N,V)` logits.  
**Fix:** Add a chunked per-position CE helper for diagnostics.

#### Sampling Re-Runs the Full Model And Reallocates the Sequence For Every Token

**Location:** `vfe3/model/model.py:1199`  
**Severity:** low  
**Evidence:** generation loops over `max_new_tokens`, calls `self.forward(context)` each time, and appends
with `torch.cat`.  
**Fix:** Preallocate or collect generated tokens and add an incremental belief/transport cache before
long-generation use.

### 5. python-pro

#### Registered Free-Energy Metric Lets Missing Tensors Through as None

**Location:** `vfe3/metrics.py:1175`  
**Severity:** medium as reported; verifier adjusted to low  
**Evidence:** required tensors default to `None` and are then passed into `free_energy_terms(...)`.  
**Fix:** Make `self_div`, `energy`, `beta`, and `alpha` required typed keyword-only parameters.

#### Non-Flat Gradient Builder Contract Is Erased

**Location:** `vfe3/gradients/kernels.py:233`  
**Severity:** medium as reported; verifier adjusted to low  
**Evidence:** wrapper uses `Optional[Callable]` while the oracle has a more precise builder contract.  
**Fix:** Reuse a precise callable alias for the omega-builder contract.

#### Decode Registry Uses Bare Callable Types

**Location:** `vfe3/model/prior_bank.py:49`  
**Severity:** medium as reported; verifier adjusted to low  
**Evidence:** `_ENCODERS`, `_DECODERS`, and getters use generic `Callable`.  
**Fix:** Define encode/decode callable aliases or Protocols and apply them to registries and getters.

#### Chunked Decode Methods Violate Optional-Last Signature Order

**Location:** `vfe3/model/prior_bank.py:361`  
**Severity:** low  
**Evidence:** optional `tau` and `chunk_size` precede defined `ignore_index`.  
**Fix:** Move defined parameters before optional parameters in chunked decode signatures.

#### Training API Signature Order Drifts From the Project Contract

**Location:** `vfe3/train.py:588`  
**Severity:** low  
**Evidence:** optional parameters precede a defined float parameter in the public training API.  
**Fix:** Reorder keyword-only parameters to match the project signature convention.

#### Generate Puts a Defined Boolean After Optional Controls

**Location:** `vfe3/model/model.py:1165`  
**Severity:** low  
**Evidence:** `greedy: bool` follows `top_k` and `top_p` optional controls.  
**Fix:** Move `greedy` before optional controls.

#### Visualization Extraction API Leaves the Model Contract Untyped

**Location:** `vfe3/viz/extract.py:105`  
**Severity:** low  
**Evidence:** public extraction helpers accept unannotated `model` parameters.  
**Fix:** Annotate with `VFEModel` or a narrow Protocol and use typed result dictionaries where public.

#### Test Suite Broadly Ignores Signature Annotation Policy

**Location:** `tests/test_alpha_i.py:6`  
**Severity:** low  
**Evidence:** test functions and helpers lack return and parameter annotations.  
**Fix:** Add return annotations and typed helper/fixture signatures in reusable tests.

## Verifier Verdicts

| ID | Investigator | Finding | Verdict | Severity | Source | Rationale |
|---|---|---|---|---|---|---|
| CR-1 | code-reviewer | Default training path uses learned linear readout | CONFIRMED | low | `train_vfe3.py:114`; `vfe3/model/prior_bank.py:170`; `vfe3/model/prior_bank.py:834` | Click-to-run selects the learned projection, but the pure prior-bank path exists under `use_prior_bank=True`. |
| CR-2 | code-reviewer | String detach misses linear-decode freeze warning | CONFIRMED | medium | `vfe3/model/model.py:323`; `vfe3/model/model.py:696` | The forward detaches for effective string detach, but the specific warning checks only legacy `detach_e_step`. |
| CR-3 | code-reviewer | Trusted resume can fall back to unsafe pickle loading | CONFIRMED | low | `vfe3/run_artifacts.py:276`; `vfe3/run_artifacts.py:287` | Unsafe load is reachable only behind explicit `trust_resume_checkpoint=True`. |
| CR-4 | code-reviewer | Dataset cache paths accept unvalidated path components | CONFIRMED | low | `vfe3/data/datasets.py:33`; `vfe3/data/datasets.py:43` | Cache filenames interpolate dataset and split without validation, but live callers use constants. |
| DBG-1 | debugger/theory | Covariant oracle cannot pass sigma leaves into Omega rebuild | CONFIRMED | high | `vfe3/inference/e_step.py:426`; `vfe3/gradients/oracle.py:98`; `vfe3/geometry/transport.py:372`; `vfe3/geometry/transport.py:475` | The oracle creates live sigma leaves, but the omega builder accepts only means while covariant transport consumes covariance. |
| DBG-2 | debugger/theory | Scheduler and EMA advance after skipped optimizer step | CONFIRMED | medium | `vfe3/train.py:406`; `vfe3/train.py:421`; `vfe3/train.py:713`; `vfe3/train.py:717` | A nonfinite unscaled step skips optimizer stepping but still advances scheduler and EMA. |
| DBG-3 | debugger/theory | Route-B connection norm dropped from CSV metrics | CONFIRMED | low | `vfe3/model/model.py:1514`; `vfe3/train.py:861` | Diagnostics compute `connection_m_norm`, but CSV whitelisting omits it. |
| DBG-4 | debugger/theory | Ablation baseline no longer matches click-to-run operating point | CONFIRMED | low | `train_vfe3.py:76`; `ablation.py:89`; `ablation.py:1305` | The baselines diverge, but this is an experiment-consistency issue rather than broken execution. |
| REF-1 | refactoring-specialist | Model-channel transport ignores configured transport | CONFIRMED | medium | `vfe3/model/model.py:618`; `vfe3/model/model.py:1047`; `vfe3/model/model.py:730` | Belief transport receives configured connection/RoPE state, while s/gamma hardcode flat transport. |
| REF-2 | refactoring-specialist | Prior-bank decode bypasses family and divergence registries | CONFIRMED | medium | `vfe3/model/prior_bank.py:357`; `vfe3/model/prior_bank.py:773`; `vfe3/config.py:1414` | Prior-bank decode hardcodes Gaussian KL kernels; config handles some mismatch cases but not by registry dispatch. |
| REF-3 | refactoring-specialist | Click-to-run configs are duplicated and divergent | CONFIRMED | low | `train_vfe3.py:76`; `ablation.py:89`; `scaling.py:153` | Separate scripts encode separate operating points; this is maintenance risk. |
| PERF-1 | performance-engineer | Dense full-logit decode is still config default | CONFIRMED | low | `vfe3/config.py:378`; `vfe3/config.py:389`; `vfe3/model/model.py:792` | Dataclass defaults can route to dense logits; current `train_vfe3.py` uses chunked CE. |
| PERF-2 | performance-engineer | Full-covariance Cholesky retries synchronize CUDA | CONFIRMED | low | `vfe3/numerics.py:83`; `vfe3/numerics.py:87`; `vfe3/families/gaussian.py:351` | Python boolean checks on CUDA masks are real in opt-in full-covariance paths. |
| PERF-3 | performance-engineer | Training synchronizes loss every optimizer step | CONFIRMED | medium | `vfe3/train.py:326`; `vfe3/train.py:406` | The default unscaled path converts loss to a Python float every step for the finite-loss guard. |
| PERF-4 | performance-engineer | Evaluation synchronizes once per validation batch | CONFIRMED | medium | `vfe3/train.py:461`; `vfe3/train.py:462` | Validation transfers token count and CE to host inside the loop. |
| PERF-5 | performance-engineer | Validation diagnostics bypass chunked CE path | CONFIRMED | medium | `vfe3/train.py:561`; `vfe3/train.py:563` | Per-position diagnostics materialize dense logits and run dense CE. |
| PERF-6 | performance-engineer | Sampling reruns full model and reallocates sequence each token | CONFIRMED | low | `vfe3/model/model.py:1199`; `vfe3/model/model.py:1201`; `vfe3/model/model.py:1221` | Generation is correct but nonincremental and reallocates via `torch.cat`. |
| PY-1 | python-pro | Registered free-energy metric lets missing tensors through as None | CONFIRMED | low | `vfe3/metrics.py:1175`; `vfe3/metrics.py:1183` | Required tensors default to `None`, causing a deeper failure instead of an immediate argument error. |
| PY-2 | python-pro | Non-flat gradient builder contract is erased | CONFIRMED | low | `vfe3/gradients/kernels.py:233`; `vfe3/gradients/oracle.py:78` | The wrapper uses bare `Callable` while the oracle has a precise callable type. |
| PY-3 | python-pro | Decode registry uses bare Callable types | CONFIRMED | low | `vfe3/model/prior_bank.py:49`; `vfe3/model/prior_bank.py:91` | Registries and getters are typed as generic `Callable`. |
| PY-4 | python-pro | Chunked decode methods violate Optional-last signature order | CONFIRMED | low | `vfe3/model/prior_bank.py:368`; `vfe3/model/prior_bank.py:370` | Optional parameters precede defined `ignore_index`. |
| PY-5 | python-pro | Training API signature order drifts from project contract | CONFIRMED | low | `vfe3/train.py:594`; `vfe3/train.py:597`; `vfe3/train.py:600` | Optional parameters precede a defined float. |
| PY-6 | python-pro | Generate puts defined boolean after Optional controls | CONFIRMED | low | `vfe3/model/model.py:1173`; `vfe3/model/model.py:1175` | `greedy: bool` follows optional controls. |
| PY-7 | python-pro | Visualization extraction API leaves model contract untyped | CONFIRMED | low | `vfe3/viz/extract.py:105`; `vfe3/viz/extract.py:232` | Public extraction helpers leave `model` unannotated. |
| PY-8 | python-pro | Test suite broadly ignores signature annotation policy | CONFIRMED | low | `tests/test_alpha_i.py:6`; `tests/test_alpha_i.py:37` | Test functions and helpers lack annotations. |

## Confirmed Punch List

1. **[high]** Fix the `regime_ii_covariant` oracle rebuild so sigma query/key leaves reach the transport
   builder, `vfe3/inference/e_step.py:426` and `vfe3/gradients/oracle.py:116`.
2. **[medium]** Warn on `use_prior_bank=False` with effective string detach, not only legacy
   `detach_e_step=True`, `vfe3/model/model.py:323` and `vfe3/model/model.py:696`.
3. **[medium]** Gate `scheduler.step()` and `ema.update()` on an actual optimizer step,
   `vfe3/train.py:406`, `vfe3/train.py:421`, and `vfe3/train.py:717`.
4. **[medium]** Decide and enforce whether the model channel should share configured transport/RoPE;
   current s/gamma code hardcodes flat transport, `vfe3/model/model.py:618` and
   `vfe3/model/model.py:1047`.
5. **[medium]** Make prior-bank decode either registry-dispatched or a hard error on unsupported
   family/divergence mismatch, `vfe3/model/prior_bank.py:357` and `vfe3/config.py:1414`.
6. **[medium]** Remove avoidable GPU synchronizations in training/eval and validation diagnostics,
   especially `vfe3/train.py:326`, `vfe3/train.py:461`, and `vfe3/train.py:561`.

## Test Suite

- Command: `python -m pytest -x --junitxml=C:\tmp\vfe3-deep-audit-20260627.xml`
- XML result: `tests=1268`, `failures=0`, `errors=0`, `skipped=1`, `time=201.276`.
- Pytest stdout summary: `1266 passed, 1 skipped, 1 xpassed, 181 warnings in 201.29s`.
- Failures: none.

The temporary JUnit XML was read for the machine-readable count and then removed.
