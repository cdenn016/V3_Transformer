# Active-Inference EFE Policy Scorer — Phase 2 Scope Note (2026-06-28)

Clarifies the Phase 2 scope of `docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md`
(Sections 4.3, 4.6, and the Phase 2 entry in Section 5). It records which arms of the Section 4.3
ablation matrix are run at the v1 one-step horizon and which are deferred, and why. No empirical result
is drawn here; this is a pre-run scope decision, made before the sealed Phase 2 run, analogous to the
pre-registration amendments in `2026-06-28-prereg-amendments.md`.

## The v1 degeneracy of the matched-compute and near-competitor arms

Section 4.3 marks the matched-compute beam and best-of-N baselines, and the length-normalized and
confidence near-competitors, as "live at v1." On the controlled ring task at the v1 one-step horizon
(`H = 1`, Section 4.2) they are not, in fact, distinct arms:

The per-decision choice is over an exhaustively evaluable candidate menu (the three control actions
under the amended candidate generator, or the top-Kp tokens otherwise). At a single decision there is
no sequence to search, so beam search reduces to ranking the menu by a per-token score, committed
greedily one token at a time. With a goal-free score (model log-probability) that is the `logprob`
baseline; with the goal preference as the score it is the same one-step argmin the EFE arm already
computes, so it would tie full EFE rather than test it. Best-of-N over a single decision likewise
collapses: sampling N draws from a three-action menu and selecting the best is either the goal-free
sampling baseline (no steering) or, if selection uses the goal, the same exhaustive evaluation EFE
performs. Genuine beam and best-of-N, where the search strategy and the model's multi-step lookahead
quality matter, require `H >= 2` over action sequences, which Section 4.2 gates behind the belief or
key-value cache (Phase 3). The length-normalized logprob is a no-op at a single emitted token (length
one for every candidate), and the argmax-confidence reranker is the greedy `logprob` arm; both are
therefore already represented by `logprob_baseline` at v1.

## What Phase 2 runs at v1

The genuinely v1-distinct standard decoders are the goal-free sampling strategies over the candidate
menu, which are run as reported (not gated) arms to show that no standard decoder steers without the
goal: temperature sampling, nucleus (top-p), and locally-typical sampling. The temperature-tuned
logprob baseline (temperature sampling with the temperature tuned on the development split by the same
single-degree-of-freedom rule as the policy precision) is the primary matched-tuning baseline that full
EFE must beat in the conjunctive gate. The unmodified `generate` greedy path (`policy_mode='none'`) is
run as the byte-identical reference. The decomposition arms (risk-only, ambiguity-only, flat-preference)
and the held-out-predictive `p_data` control are carried over from Phase 1.

The gate machinery added in Phase 2 is the conjunctive primary gate under Holm-Bonferroni (full EFE
must beat the `p_data` control and the temperature-tuned logprob baseline by more than `delta_min` with
corrected significance), Benjamini-Hochberg false-discovery-rate control at `q = 0.05` over the broader
arm grid (full EFE versus every other arm on the primary success metric), the random-lesion gate, and
the closed-loop causality lesion check (the committed action must measurably change the next
observation, which the deterministic ring transition guarantees by construction). Risk and ambiguity
are logged separately from raw log-probability for the scorer arms so a win can be attributed to a
component (Section 4.4).

## Caveat on the temperature-tuned logprob primary at v1

On this task the temperature-tuned logprob baseline has little steering leverage, and this is expected,
not a defect. The ring training renders the action token uniformly and independently of the goal and
state, so the model's distribution over the three actions at a decision is approximately uniform.
Temperature scaling of a near-uniform menu is approximately a no-op, so the tuned temperature carries
almost no degree of freedom and the arm collapses toward the uniform random arm. The conjunctive
primary gate therefore pairs the goal-free `p_data` control with a baseline that is, on this task, near
redundant with the random lesion: full EFE beating it confirms the falsifier "a tuned standard decoder
matches EFE" is false, but it is weak independent evidence here. The substantive test of whether EFE's
structured one-step scoring beats a tuned standard decoder lives in the language-modeling setting, where
the decoding temperature genuinely changes behavior, and in the matched-compute beam and best-of-N
comparison at `H >= 2`. This is recorded so the v1 result is not over-read.

## Deferred to Phase 3

The matched-compute beam and best-of-N baselines, and the matched-compute element of the conjunctive
gate that names them, are deferred to the horizon phase (Phase 3), where `H >= 2` and the belief or
key-value cache make them non-degenerate and the wall-clock honesty check meaningful. The
length-normalized and argmax-confidence near-competitors are deferred likewise, or read off the
`logprob` arm, since they are not distinct at a single-token decision. Until then the v1 conjunctive
gate is the `p_data` control and the temperature-tuned logprob baseline; the Phase 2 to Phase 3 go
decision is read with that scope in mind.

## Status

This note fixes the Phase 2 arm scope before the sealed run. The pre-registered constants of Section 4.7
are unchanged. The v1 result remains a pragmatic-steering validation only; the epistemic term is inert
at `H = 1` (Section 2.8), and the matched-compute comparison that would test whether EFE's structured
one-step scoring beats brute-force search at equal compute is genuinely available only at `H >= 2`.
