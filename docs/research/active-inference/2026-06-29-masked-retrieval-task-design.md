# Masked Key-Value Retrieval Task — design note (2026-06-29)

The Phase-3 H>=2 epistemic task deferred by the EFE policy-scorer spec
(`docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md`, Section 5 Phase 3) and
the Phase-2 scope note (`2026-06-28-phase2-scope-note.md`). It is the non-degenerate home for the
matched-compute beam / best-of-N baselines, which reduce to existing arms at H=1 and need an H>=2
task over action sequences. This note pins the concrete instantiation; the spec only sketches it.

## Why this task (and the prediction)

The ring is a fully observed, purely pragmatic H=1 task: greedy preference-matching solves it and the
epistemic term is inert by construction. Masked retrieval is the opposite: it is partially observed,
and the only reliable route to success is an information-seeking action (a probe) followed by an
answer, so the expected-information-gain term is load-bearing.

The prediction, before any run, is that the EFE rollout FAILS this task, and that the failure is the
result. The information-gain term `I` is identically zero at the v1 point belief even at H>=2 (the MI
bridge `I = H[q(o|pi)] - E_q H[p(o|s)]` collapses to 0 without marginalized belief spread, spec
Section 2.8), the `sigma_mc` estimator that would carry that spread is gated off by the FAILED
sigma-validation gate (`2026-06-28-sigma-gate-prereg.md`), and the why-sigma-collapses investigation
(`2026-06-29-why-sigma-collapses.md`) showed `sigma` is anchored to a per-vocabulary prior table with
no data-precision channel, so it cannot carry the contextual (probed-vs-unprobed) uncertainty this
task turns on. A `sigma_mc` arm is therefore run REPORTED-ONLY (never as a validated claim, spec
Section 4.5) to show whether any uncertainty signal would rescue the task; the structural finding
predicts it will not. This is a falsification test of the epistemic side, not a capability demo.

## Environment

A sealed small vocabulary. With `n_keys` keys and `n_vals` values:

- value symbols `v_0..v_{n_vals-1}` (ids `0..n_vals-1`) — appear in the table, as the probe-revealed
  observation, and as the answer content;
- key symbols `k_0..k_{n_keys-1}` — index the table and the query;
- control `MASK`, `SEP`, `QUERY`, `EOS`;
- compound action tokens `PROBE_k_i` (one per key) and `ANSWER_v_j` (one per value), so each decision
  is a SINGLE token (the closed-loop runner commits one token per step, as on the ring).

Default `n_keys=4`, `n_vals=8` -> `V = 2*n_vals + 2*n_keys + 4 = 28`.

Each episode samples a value for every key (`vals[k]`, values may repeat) and a target key `t`. The
context renders the table with the TARGET's value masked, then the query:

```
k_0 v_{vals[0]} SEP k_1 v_{vals[1]} SEP ... k_t MASK SEP ... QUERY k_t
```

The target value `vals[t]` is unknowable from the context (masked); every other key's value is shown
but irrelevant. The environment responds to actions:

- `PROBE_k_i` -> the env appends the revealed value token `v_{vals[i]}` to the sequence (a new
  observation). Probing the target `k_t` reveals `vals[t]`; probing any other key reveals a value the
  context already showed (a distractor).
- `ANSWER_v_j` -> commits and terminates; the episode is correct iff `j == vals[t]`.

The optimal policy is the 2-step `PROBE_k_t ; ANSWER_v_{vals[t]}`, realizable only at H>=2.

## Training

Teacher-forced next-token prediction on the optimal trajectory
`[context] PROBE_k_t v_{vals[t]} ANSWER_v_{vals[t]} EOS`. The model learns: after `QUERY k_t` predict
`PROBE_k_t` (probe the queried key); after a probe, the env-revealed value is teacher-forced (the model
cannot predict the masked target value from context — that is the genuine uncertainty); after the
revealed value predict `ANSWER_v_{that value}`. Predictive adequacy is measured at the answer position
(given the revealed value, does the model answer it) so a model that has learned the answer mechanics
is admitted; the open question the experiment tests is whether the SCORER chooses to probe.

## Arms

A receding-horizon closed loop (commit the first action of the chosen H-step policy, let the env
respond, repeat up to a small budget). Forward-pass / belief-rollout count is logged per arm so the
matched-compute comparison is honest (spec Section 4.4 wall-clock honesty).

- `efe_rollout` (H>=2): scores candidate H-action policies by EFE (risk + ambiguity) via the
  belief-prefix cache, commits the first action of the lowest-G policy. Pragmatic at the point belief
  (`I==0`); the genuine v1-available scorer.
- `efe_rollout` + `sigma_mc` ambiguity: REPORTED-ONLY (gate FAILED), to probe whether a marginalized
  uncertainty signal would value probing.
- `beam` (matched-compute): goal-blind beam search over H-action sequences by model sequence log-prob,
  beam width matched to the EFE candidate budget; commits the first action of the best beam.
- `best_of_n` (matched-compute): goal-blind; sample N H-action sequences by model log-prob, score by
  sequence log-prob, commit the first action of the best; N matched to the EFE candidate budget.
- `random` (placebo), `greedy_ref` (unmodified argmax decode).

The matched-compute claim is that beam / best-of-N are charged the same belief-rollout budget as the
EFE rollout, so a win cannot be a compute artifact.
