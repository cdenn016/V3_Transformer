---
name: verification
description: Run evidence-gated, multi-view verification for code, mathematics, sources, and experiments.
---

# Verification

Use this skill for a verifier request, audit, proof, correctness claim, experimental result, or source-backed factual claim. The ledger separates an LLM's assessment from evidence that can close a claim.

## Workflow

1. Start the control plane from the repository root:

   ```powershell
   {{VERIFICATION_GATE_COMMAND}} start --cwd . --ledger .verification/ledger.json --mode closure
   ```

2. Add one ledger claim per check, with its artifact revision, domain, severity, aggregate 0 to 20 criterion scores, two or more uniquely identified view scores, calibration kind, comparison record, evidence, counterevidence, verifiers, and open obligations. Read [the contract](references/contract.md) before assigning a state.

3. Read the criterion file for each domain in scope: [code](references/criteria-code.md), [mathematics](references/criteria-math.md), [evidence](references/criteria-evidence.md), [experiments](references/criteria-experiment.md), and [general](references/criteria-general.md). Dispatch independent domain roles and record every result at the criterion level.

4. Begin with two views. Adaptively escalate to four or eight for small margins, high dispersion, criterion disagreement, or high severity. Record each applicable reason in `escalation_triggers` using `small_margin`, `high_dispersion`, `criterion_disagreement`, or `high_severity`; do not silently omit the reason and do not invent numeric thresholds. Reverse A/B order for each pairwise comparison. For more than four candidates, use a balanced pivot tournament. Critical and high closure requires both `verifier-skeptic` and `verifier-adjudicator`.

5. Do not close a claim by vote. Missing evidence or unresolved disagreement yields `INCONCLUSIVE`, never majority-vote acceptance. LLM judgment alone may support only `LLM_SUPPORTED`, never `EVIDENCE_VERIFIED`.

6. Run the deterministic check after the ledger is complete:

   ```powershell
   {{VERIFICATION_GATE_COMMAND}} validate .verification/ledger.json
   ```

   Repair every reported error. In the final response, name `.verification/ledger.json` as the validated ledger.

## State and evidence rules

Use `CANDIDATE` before assessment and `LLM_SUPPORTED` for a reasoned but unverified assessment only in `triage` mode. `closure` mode rejects both intermediate states: missing eligible evidence or unresolved disagreement requires `INCONCLUSIVE` with a specific open obligation. `EVIDENCE_VERIFIED` requires current eligible evidence and `REFUTED` requires current counterevidence. Code and experiment correctness require current mechanical or reproduced-output evidence. Mathematical correctness requires a derivation or formal proof; numerical evidence alone is insufficient. See [the contract](references/contract.md) for the exact closure requirements.
