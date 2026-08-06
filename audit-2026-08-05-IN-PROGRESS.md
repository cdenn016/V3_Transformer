# Deep Audit — 2026-08-05 — IN PROGRESS (resume state)

**Status:** wave 1 dispatched. Resume at step 4 below.
**Skill:** `deep-audit` (follow it exactly; this file only carries state, not the method).

---

## READ THIS FIRST IF YOU ARE A SCHEDULED CLOUD RUN

You have zero prior context. This file is your entire brief. Three environment facts change
how you must execute:

1. **The `deep-audit` skill and the `audit-*` domain-expert subagents DO NOT EXIST in your
   environment.** They live in the user's home directory (`~/.claude/`), not in this repo, so
   they will not resolve. Dispatch **`general-purpose` subagents** instead and bake each expert
   role into its own prompt. The method you need is reproduced in "Workflow state" below — you
   do not need the skill file.
2. **This clone does NOT contain the user's uncommitted edits** to `ablation.py`,
   `train_vfe3.py`, and `vfe3/config.py`. The "Active config" section below records the exact
   configuration those edits produce. **Reason about reachability from that recorded config, not
   from the committed `ablation.py`.**
3. **If `audit-2026-08-05.md` already exists at the repo root**, the originating local session
   finished first. Read it and fill only genuine gaps — do not redo it.

Deliverables, in the user's priority order:
- **(c) Explain the performance symptom** — `3.12 it/s` while using only 6GB of 32GB on an
  RTX 5090, at K=20, H=2, N=64, B=48, fp32, `gaussian_full`. Under-utilized memory plus low
  throughput points at host-device syncs, kernel-launch-bound tiny-tensor work, per-head/per-chunk
  Python loops, and `torch.linalg` latency on (20,20)/(10,10) blocks. Name concrete speed-ups
  with expected magnitude.
- **(a) Shortcomings and BLOCKED FEATURES** in the gaussian_full / pure path that should be built
  out: `NotImplementedError`, hard-gated branches, config knobs accepted but never read under
  `gaussian_full`, and diagonal-only implementations with no full-covariance counterpart.
- **(b) Optimizations** — including whether the idle 26GB can be spent (larger batch/seq) and
  what currently blocks that.

Highest-value single question, from the user's own observation: **full-covariance runs ASCEND
the free energy F while diagonal runs descend it.** Investigate whether any code path makes the
`gaussian_full` E-step update non-descent — wrong sign, wrong metric, missing Jacobian factor,
preconditioner mismatched to the full-covariance geometry, or a trust-region clip applied in the
wrong basis.

**Output:** write the finished report to `audit-2026-08-05.md` **at the repo root** (`docs/` is
gitignored — that is why this file is at the root), commit it on a new branch, and open a PR so
the user can read it. **Do not fix any of the findings** — present the punch list only.

---

## User's request (verbatim intent)

> perform an ultradeep audit of the 'gaussian_full'/'pure path' codebase by deploying several
> expert agents. identify any shortcomings or blocked features that should be built out.
> identify any optimizations/speed-ups that can be leveraged. note: K=20, 2 heads,
> gaussian-full, seq-length = 64, batch = 48 hits only 6 GB out of 32GB available but is slow
> to compute at 3.12 it/sec (see current ablation.py configuration).

Three deliverables, all required:
1. Shortcomings / **blocked features** that should be built out.
2. **Optimizations / speed-ups.**
3. Explanation of the perf symptom: 3.12 it/s while using 6GB of 32GB (RTX 5090) — i.e.
   under-utilized memory, compute/launch-bound.

## Scope

The `gaussian_full` / pure-path subsystem:

```
vfe3/families/{gaussian,base,exact_congruence,laplace}.py
vfe3/geometry/{transport,retraction,lie_ops,groups,phi_preconditioner}.py
vfe3/inference/e_step.py
vfe3/free_energy.py
vfe3/model/{model,prior_bank}.py
vfe3/gradients/kernels.py
vfe3/gauge_optim.py
vfe3/numerics.py
vfe3/attention_prior.py
vfe3/metrics.py
vfe3/config.py
vfe3/train.py
```

Repo source is `vfe3/` (~46k lines) + top-level drivers + `tests/`. The 6,941 `.py` count from
`find` is almost entirely `.venv/` — ignore it, the repo is NOT >500 real source files.

## Established constraints (forward these to every agent)

Derived from `README.md` + `vfe3/config.py` (there is **no** `CLAUDE.md`/`AGENTS.md` at repo root —
`README.md` was the instruction-file route used).

- Gauge-theoretic variational-free-energy transformer. The **pure path** profile has no learned
  Q/K/V projections, no MLP, no pointwise activation. Gauge purity is a correctness property.
- The repo is **registry-heavy**: families, gauge groups, transports, decode modes and attention
  priors are decorator-registered behind Callable seams. Many toggles are deliberately default-OFF
  measurement instruments and are NOT dead code. A false "dead code" report is actively harmful.
- Comments and docstrings are evidence of **intent**, not behavior. Only executable code counts.

## Active config for reachability reasoning (user's live `ablation.py` BASELINE_CONFIG)

```
family="gaussian_full"  (diagonal_covariance=False -> per-token (K,K) covariance)
use_prior_bank=True, decode_mode="full_chunked", untie_decode_bank=True,
decode_unigram_prior=True, transport_mode="flat", gauge_group=block_glk,
embed_dim K=20, n_heads H=2 (d_h=10), max_seq_len N=64, batch_size B=48,
n_layers L=1, n_e_steps T=1, e_step_update="gradient", e_step_mu_precond="raw",
oracle_unroll_grad=True, lambda_gamma=0, lambda_h=0, lambda_alpha_mode="constant",
amp_dtype=None (FP32, no autocast), grad_clip_per_role=False,
skip_belief_sigma_update=False, use_head_mixer=False, use_ema=False,
e_mu_q_trust=1, e_sigma_q_trust=10.0, sigma_max=10,
e_q_mu_lr=0.2, e_q_sigma_lr=0.001, e_phi_lr=0.0, m_phi_lr=0.01,
emit_expensive_diagnostics=True, generate_figures=True,
log_interval=100, eval_interval=1500, max_steps=7500
```

Hardware: RTX 5090, 32GB.

## Theory-invariant gate: **MET** (all three signal groups)

1. Source signals: `Manuscripts-Theory/{GL(K)_attention,GL(K)_supplementary,PIFB2}.tex`;
   modules named for geometry/transport/free-energy; `torch.linalg` on covariance-like tensors.
2. Instruction file: `README.md` declares equivariance / SPD / divergence invariants.
3. Verification ledger: `.verification/ledger.json` exists with math/code domain claims.

=> The domain-expert tier is **mandatory**, and because this is a whole-subsystem audit,
run the **entire relevant `audit-*` pool**, not a 3–6 subset.

## Workflow state

- [x] **Step 1–2** — scope check + constraints established (above).
- [x] **Step 3 — wave 1, base five dispatched in parallel** (2026-08-05, ~21:0x CDT):
      `code-reviewer` (quality/security), `debugger` (bugs/gradient-flow — briefed on the
      user's report that full-cov runs ASCEND F while diagonal runs descend),
      `refactoring-specialist` (dead code / blocked features), `performance-engineer`
      (the 3.12 it/s symptom — top priority), `python-pro` (shape-polymorphic contracts).
      **If their results were lost to a rate limit, re-dispatch wave 1 — the briefs are
      reconstructible from the Scope + Constraints + Active-config sections above.**
- [ ] **Step 4 — wave 2, domain-expert tier (MANDATORY).** Dispatch in parallel, whole pool:
      `audit-variational` (free-energy assembly, E/M separation, softmax stationarity — the
      F-ascent question), `audit-info-geometer` (KL/divergence formula, Fisher metric, natural
      gradient, alpha routing), `audit-geometer` (SPD manifold, exp/log, retraction, transport),
      `audit-gauge-theorist` (equivariance, cocycle, holonomy under block_glk),
      `audit-numerical-analyst` (conditioning of (20,20)/(10,10) covariances, clamp saturation,
      matrix-function accuracy), `audit-ml-engineer` (optimizer, trust-region reachability, LR,
      measurement validity), `audit-transformer-ml` (softmax axis, masking, per-head, RoPE),
      `audit-implementation-engineer` (config→runtime wiring, accepted-and-ignored knobs,
      dead-under-active-config branches). Same output contract as wave 1.
      (~8 agents = within the ~10 parallel cap; if trimming is ever needed, split into two waves
      and SAY SO in the report — never silently drop a lens.)
- [ ] **Step 5 — verifier.** One `general-purpose` agent, all wave-1 + wave-2 summaries inlined.
      Must re-read cited source, cite `file.py:line`, reject comment-only evidence, and mark
      CONFIRMED / REFUTED / INCONCLUSIVE.
- [ ] **Step 6 — adversarial challenge tier** on every CONFIRMED critical/high (≈6 duels max;
      excess goes to a second wave, stated in the report). `audit-skeptic` + `audit-defender`
      in parallel per finding, then adjudicate UPHELD / DOWNGRADED / DROPPED with a cited reason.
      Theory findings may be escalated below high at the orchestrator's discretion.
- [ ] **Step 7 — write `docs/audits/audit-2026-08-05.md`** using the skill's template, and
      delete/supersede this IN-PROGRESS file.
- [ ] **Step 8 — run the test suite**, append pass count.
- [ ] **Step 9 — report to user.** Present the punch list; do NOT start fixing without approval.

## Known pre-existing test noise (do not attribute to audit findings)

45 tests currently fail across `test_regime_ii*.py`, `test_straight_through.py`, `test_viz.py`,
`test_train.py::test_optimizer_groups_regime_ii_connection`,
`test_run_diagnostics_2026_06_13.py`, `test_source_identity_snapshot_performance_20260716.py`,
`test_2026_07_15_driver_reliability_remediation.py`, and
`test_checkpoint_resume.py::test_legacy_false_omega_direct_resume_...`.

Single cause: the **uncommitted** `vfe3/config.py` edit flips the `pos_phi_compose` **default**
from `"bch"` to `"group_product"`, and the validator at `vfe3/config.py:994` requires
`group_product` to pair with `transport_mode='flat'` + `gauge_parameterization='phi'`. Every test
that builds a config without naming that field now dies at construction. Verified: the failure set
is byte-identical with `vfe3/train.py` at `HEAD~1` (45 = 45).

## Uncommitted working-tree state at audit start

`M ablation.py`, `M train_vfe3.py`, `M vfe3/config.py`, `?? zzzzz.py` — these are the user's
in-flight experiment edits and are IN SCOPE for reachability reasoning (the active config above
comes from them) but must NOT be committed as part of the audit.

Last commit on `main`: `6483f19` (merge of PR #217, the selection-data-identity finalization fix).
