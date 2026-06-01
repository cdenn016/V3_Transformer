# Deep Audit — 2026-05-31 (Task-1 buildout: artifacts + use_prior_bank + head mixer)

Five-lens parallel investigation + per-finding adversarial source verification (a fresh verifier
re-read the cited source for every claim; comment-only evidence => REFUTED). 33 agents, 28 raw
findings, **28 CONFIRMED, 0 REFUTED, 0 INCONCLUSIVE**. The high confirmation rate reflects that
most findings are *factually-true observations* — but triage matters: only ~8 warrant a code
change; the rest are intentional/documented behavior, pre-existing code outside this task, or the
explicitly-deferred perf work. Comments are intent, not behavior; the verifier cited `path:line`.

## Scope
`vfe3/` + `train_vfe3.py` + `tests/`, priority on the Task-1 additions (run_artifacts.py,
head_mixer.py, prior_bank.py linear decode, model.py mixer wiring, train.py optimizer/artifacts,
config.py toggles).

## Lens summaries
- **code-reviewer (quality+security):** Task-1 code largely clean; one real defect — `torch.load`
  without `weights_only=True` (contradicts datasets.py precedent). build_optimizer coverage assert
  verified correct for every toggle combination.
- **debugger (bugs+gradients+theory):** No hard-constraint violations — the no-NN pure path exists
  under defaults, sandwich transport `Ω Σ Ωᵀ` is correct, the head-mixer algebra (`kron(A,I_d)` on
  μ; `A²` diagonal-sandwich and the full two-einsum sandwich on Σ) is algebraically correct. Two
  *intentional* silent gradient freezes the coverage assert does not (and is not meant to) catch;
  a duplicate-CE log line.
- **refactoring-specialist (dead-code):** `decode_log_scale`/`decode_tau` inert on the linear path;
  `reference_decode` orphan; duplicate `_banner`; `run_training` a superseded dead entry point.
- **performance-engineer (perf):** dominant cost is the dense `(B,N,N,K,K)` Ω at transport.py:152
  (pre-existing, = deferred perf P0 #2); block-diagonal structure unexploited in Ω formation; a
  per-forward GPU→CPU sync; 3 forwards per log step; a per-iteration `torch.tensor` alloc.
- **python-pro (types):** missing `Optional[nn.Parameter]` on `output_proj_weight`; union `forward`
  return; `finalize_run` `Dict[str,float]` broken by `None` entries; bare `figs` param.

## Verifier verdicts — all 28 CONFIRMED (triaged)

### A. Fixed in the audit-fix commit (genuine, surgical, mostly Task-1 code)
| Finding | Location | Sev | Action |
|---|---|---|---|
| `torch.load` без `weights_only=True` | run_artifacts.py:142 | med | add `weights_only=True` (best_path is a pure state_dict) |
| Signature `=`-alignment violation (MANDATORY) | run_artifacts.py:46 | low | re-pad annotation column |
| `output_proj_weight` missing class annotation | prior_bank.py:132 | med | add `output_proj_weight: Optional[nn.Parameter]` |
| `finalize_run` return `Dict[str,float]` vs `None`/`int`/`bool` entries | run_artifacts.py:126 | med | widen to `Dict[str, object]` |
| `_save_free_energy_bar(figs)` untyped | run_artifacts.py:200 | low | annotate `figs: ModuleType` |
| Validation log prints CE in both Loss+CE cols | train.py:236, train_vfe3.py:251 | low | drop the redundant "Loss:" column on val lines |
| Assert guards grouping, not gradient-flow | train.py:55 | (doc) | comment enumerating the intentional null-grad toggle cases |
| `decode_log_scale`/`decode_tau` inert on linear path | prior_bank.py:167,325 | (doc) | clarifying comment (do NOT gate creation — would break the assert + state_dict) |
| use_prior_bank=False + detach_e_step=True freezes encode tables | model.py:131 | (doc) | a `warnings.warn` guardrail for the joint footgun |

### B. Confirmed but intentional / by-design (no behavior change)
- **phi_embed null-gradient under `detach_e_step=True`** (train.py:58): test-pinned as as-built
  semantics (test_model.py:102). The coverage assert is about *grouping*, not gradient flow — now
  documented (A).
- **Hand kernel key-side `.detach()`** (kernels.py:152): the defining filtering-vs-smoothing
  distinction (oracle.py:58-61), correct and intentional.

### C. Confirmed, pre-existing, left as-is (CLAUDE.md: mention pre-existing dead code, don't delete)
- `PriorBank.reference_decode` orphan (prior_bank.py:170) — a public verification seam; tests use a
  local clone. Left in place.
- Duplicate `_banner` in train_vfe3.py:159 and vfe3/train.py:269 — pre-existing copy-paste. Left
  (surgical: not this task's code). A `run_training` deprecation note added since Task-1b made it
  divergent (it lacks the artifacts hook → train=val leakage; no callers).
- `run_training` superseded / `checkpoint_interval` unreachable via it (train.py:287,263) — dead
  entry point (no callers); noted in its docstring.
- Pre-existing type-precision (forward union return model.py:120; diagnostics bare `dict`
  model.py:171; `block_norm: Optional[Any]` stack.py:27) — static-only, no runtime effect, outside
  this task. Left.

### D. Confirmed perf — DEFERRED (pre-existing hot path; needs GPU re-profile + golden tests)
- **[high] Dense `(B,N,N,K,K)` Ω materialized + autograd-saved each step** (transport.py:152) — this
  is exactly the deferred perf **P0 #2** ("skip the dense Ω via factored transport"). Substantial,
  correctness-sensitive (needs the factored==dense golden test first), and explicitly deferred by
  the perf session. Not touched in an audit-fix commit.
- **[high] Block-diagonal Ω structure unexploited in formation/transport einsums** (transport.py:152)
  — same area; factor transport into H blocks of `(B,N,d,d)`. Deferred with P0 #2.
- **[med] GPU→CPU sync from `flat_targets.any()`** (model.py:150) — pre-existing all-ignore guard;
  GPU-only cost, test-pinned, subtle to make sync-free. Deferred to the GPU perf pass.
- **[med] 3 E-step forwards per log step** (train.py:217) — off the hot path (log cadence);
  pre-existing logging design.
- **[low] `torch.tensor(irrep_dims)` per E-step iter** (kernels.py:178); **[low] matrix_exp scatter
  for-loop** (transport.py:102) — pre-existing micro-costs; fold into the GPU perf pass.

## Test suite
- Pre-audit baseline: 240 tests, 0 failures, 0 errors (`--junitxml`).
- Post audit-fix: see the same-day run logged below / in the edit doc.

## Recommendation
Fix set A (surgical, Task-1 code + documentation) applied as a separate commit. Set D (the dense-Ω
factoring) is the highest-value follow-up but belongs in a dedicated, golden-test-pinned perf pass
on the 5090, not an audit-fix commit — it remains the top item on the speedup roadmap.
