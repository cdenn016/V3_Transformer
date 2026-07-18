# Verification Control Plane Design

**Date:** 2026-07-18

**Status:** Approved for implementation

## Objective

Build a shared verification protocol for Claude Code and Codex that uses LLM judgment to rank, challenge, and route claims while reserving closure for current mechanical evidence, primary sources, reproduced execution, complete derivations, or formal proof. The protocol must work for code, mathematics, research claims, experiments, and general factual work without changing ordinary non-verifier tasks.

## Evidence model

Every material claim receives a stable identifier and one of five states: `CANDIDATE`, `LLM_SUPPORTED`, `EVIDENCE_VERIFIED`, `REFUTED`, or `INCONCLUSIVE`. LLM agreement alone may reach `LLM_SUPPORTED` but may not reach `EVIDENCE_VERIFIED`. The ledger records the inspected artifact revision, criterion-level scores, replicate scores when available, evidence locations, counterevidence, verifier roles, open obligations, and whether a later edit invalidated the evidence.

For code and experiment claims, `EVIDENCE_VERIFIED` requires current mechanical or reproduced-output evidence. For mathematical claims, it requires a derivation or formal proof; numerical probes may falsify or support a derivation but cannot prove a universal statement. Research, source, and general factual claims require current primary-source or reproduced-source evidence. Critical and high claims require a skeptic and an adjudicator before closure.

## LLM scoring

Criteria are scored independently on a 0-to-20 scale. When the model API exposes scoring-token log probabilities, the verifier uses the expected continuous score described in *LLM-as-a-Verifier* with granularity 20. Repetition is adaptive: begin with two views and escalate to four or eight when margins are small, dispersion is large, criteria disagree, or severity is high. Pairwise comparisons reverse candidate order; candidate sets larger than four use a balanced pivot tournament. Scores route work and determine escalation. They never substitute for evidence.

When log probabilities are unavailable, verifiers emit the same criterion records with categorical support, refutation, or abstention and label the result uncalibrated. A two-stage adapter may later use a closed model for reasoning and an open model for log-probability scoring. This adapter is optional and is not part of the initial implementation.

## Components

The version-controlled source lives under `agent_tooling/verification/`. Its installable skill contains the shared contract, domain criteria, JSON schemas, ledger validator, Stop-hook gate, and skill evaluations. Neutral agent specifications generate Claude Markdown agents and Codex TOML agents so their developer instructions remain semantically identical.

The installed agent set comprises `verifier-orchestrator`, `verifier-code`, `verifier-math`, `verifier-evidence`, `verifier-skeptic`, and `verifier-adjudicator`. Investigators return structured support, refutation, or abstention results. Only the adjudication stage assigns a ledger state, and the validator independently checks whether the evidence permits that state.

An installer copies the skill into both user skill catalogs, renders both agent formats, inserts an identical marked policy block into user and project instruction files, adds a marked integration block to both deep-audit skills, and merges a Stop-hook command into existing hook configuration without replacing unrelated settings. Re-running the installer is idempotent.

## Hook behavior

The hook is dormant unless the current working directory contains `.verification/active.json`. The verification skill creates that marker and an initial ledger. On `Stop`, the hook validates the referenced ledger. It blocks completion with machine-readable reasons when the ledger is missing, malformed, stale, or overclaims its evidence. A valid ledger is permitted only when the final response names it; the hook then removes the task-owned activation marker. User interruption remains unaffected, and ordinary tasks without a marker incur only a small file-existence check.

The hook validates artifact structure and evidence eligibility, not substantive mathematical truth. Verifier agents and mechanical tools supply the evidence; the deterministic gate prevents unsupported state transitions.

## Deep-audit integration

Deep audit retains its parallel investigator and domain-expert waves but replaces the single free-form verifier with the verification skill. Each candidate becomes a claim-ledger entry. Claims are checked criterion by criterion, critical and high findings receive skeptic and adjudicator results, and the report is generated from the validated ledger rather than raw concatenated agent prose. Test counts must come from terminal summaries or JUnit XML, and evidence becomes stale after later edits.

## Evaluation

Unit tests cover valid and invalid state transitions, stale revisions, mathematical numerical-only overclaiming, severity challenge requirements, hook activation, hook blocking, successful marker cleanup, installer idempotence, preservation of unrelated settings, and Claude/Codex agent parity. The installable skill includes behavioral evaluations for code, mathematics, sources, experiments, stale evidence, and abstention.

The first deployment is evaluated against historical failure modes: source-only overconfidence, stale line references, severity inflation, comments that contradict execution, tests run before the last edit, numerical evidence presented as proof, and large verifier output without a parsed final synthesis. The implementation does not claim calibrated real-world accuracy until repeated with-skill and without-skill runs have been graded.

## Failure handling

Missing tools, unavailable log probabilities, verifier disagreement, missing primary evidence, or unresolved proof obligations produce `INCONCLUSIVE`, not a majority-vote acceptance. Invalid ledgers fail closed while the task marker is active. Installation aborts before modifying destination files if a source artifact is missing or an existing JSON file cannot be parsed.

## Non-goals

This phase does not install TurboAgent, add an inference proxy, send source code to a new external service, or claim that LLM verification proves correctness. It does not rewrite existing debate skills wholesale. It supplies a reusable control plane and integrates the audit workflow most directly affected by the current single-verifier design.
