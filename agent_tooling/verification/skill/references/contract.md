# Claim Ledger Contract

The gate validates `schema_version` `1.0`, a ledger-level `artifact_revision`, and every claim's matching revision. Preserve the fields required by `schemas/claim-ledger.schema.json`; do not invent fields. Every criterion receives a numeric score from 0 to 20 and a specific name.

Use `CANDIDATE` while a claim is queued, `LLM_SUPPORTED` for analysis that lacks closure evidence, `EVIDENCE_VERIFIED` for a closed supporting record, `REFUTED` for a closed contradicting record, and `INCONCLUSIVE` when an open obligation remains. `INCONCLUSIVE` requires at least one nonempty obligation. Closed claims cannot retain open obligations. Any evidence whose artifact revision differs from the ledger revision is stale and cannot close a claim.

`llm_judgment` may inform triage but cannot by itself assign `EVIDENCE_VERIFIED`. Code and experiment claims require current `mechanical` or `reproduced_output` evidence for verification and refutation. Mathematics claims require current `derivation` or `formal_proof`; numerical evidence does not prove mathematical correctness. Evidence, research, source, and general claims require current `primary_source` or `reproduced_source` evidence to verify. A refutation records counterevidence with `supports: false`.

A high or critical `EVIDENCE_VERIFIED` or `REFUTED` claim records both `verifier-skeptic` and `verifier-adjudicator`. The skeptic seeks a decisive counterexample, stale input, missing assumption, or alternative explanation. The adjudicator checks that the ledger state follows from the recorded evidence and resolves neither disagreement nor missing evidence by majority vote.

Run the exact commands from `SKILL.md`: `start` creates the candidate ledger and activation marker; `validate` checks the completed ledger. The final response must name the validated ledger so the Stop hook can permit completion.
