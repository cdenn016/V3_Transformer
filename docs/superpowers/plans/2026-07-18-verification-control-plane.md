# Verification Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and install a cross-platform, evidence-gated verification system for Claude Code and Codex.

**Architecture:** A version-controlled installable skill owns the claim contract, domain criteria, schemas, validator, and opt-in Stop hook. Neutral JSON agent specifications render to Claude Markdown and Codex TOML, while an idempotent installer merges marked policy and deep-audit integration blocks into existing user configuration without replacing unrelated content.

**Tech Stack:** Python 3 standard library, JSON, TOML text generation, Markdown/YAML skill metadata, pytest, Claude Code hooks, Codex hooks.

## Global Constraints

- LLM judgment alone may not assign `EVIDENCE_VERIFIED`.
- Code and experiment closure requires current mechanical or reproduced-output evidence.
- Mathematical closure requires a derivation or formal proof; numerical evidence alone is insufficient.
- Critical and high closure requires skeptic and adjudicator participation.
- The hook is dormant unless `.verification/active.json` exists in the current working directory.
- Installation must preserve unrelated user settings, agents, skills, hooks, and repository WIP.
- Claude and Codex verifier agents must be generated from the same neutral specifications.
- All skill text and generated artifacts use American English.

---

### Task 1: Claim-ledger validator and Stop gate

**Files:**
- Create: `agent_tooling/__init__.py`
- Create: `agent_tooling/verification/__init__.py`
- Create: `agent_tooling/verification/skill/scripts/verification_gate.py`
- Create: `agent_tooling/verification/skill/schemas/claim-ledger.schema.json`
- Test: `tests/test_verification_gate.py`

**Interfaces:**
- Produces: `validate_ledger(data: dict[str, object]) -> list[str]`, `run_hook(payload: dict[str, object]) -> tuple[int, dict[str, object] | None]`, and CLI commands `start`, `validate`, and `hook`.
- The `start` command creates `.verification/active.json` and a candidate ledger. The `hook` command reads hook JSON from stdin.

- [ ] **Step 1: Write failing validator tests**

  Add fixtures for a mechanically verified code claim, an LLM-only code claim, a numerical-only mathematical claim, stale evidence, a high claim without challenge roles, and an inconclusive claim with open obligations. Assert that only evidence-eligible state transitions validate.

- [ ] **Step 2: Run the validator tests and confirm RED**

  Run `python -m pytest tests/test_verification_gate.py --junitxml=C:\tmp\vfe3-verification-gate-red-20260718.xml` and confirm collection fails because `verification_gate.py` does not exist.

- [ ] **Step 3: Implement the minimal validator and schema**

  Implement strict required-field, enum, score-range, evidence-kind, artifact-revision, open-obligation, domain-specific closure, and high-severity challenge checks using only the standard library. Return every discovered error in deterministic claim-ID order.

- [ ] **Step 4: Add failing hook lifecycle tests**

  Assert that an inactive directory passes, an active invalid ledger returns `{"decision": "block"}`, a valid ledger without a final ledger reference blocks, and a valid referenced ledger passes and removes only `.verification/active.json`.

- [ ] **Step 5: Implement the hook lifecycle and CLI**

  Read `cwd`, `stop_hook_active`, and `last_assistant_message` from stdin. Resolve ledger paths beneath the active working directory, reject traversal, and emit Claude/Codex-compatible Stop-hook JSON. Do not mutate the ledger.

- [ ] **Step 6: Run GREEN verification**

  Run `python -m pytest tests/test_verification_gate.py --junitxml=C:\tmp\vfe3-verification-gate-green-20260718.xml` and require zero failures and errors in the XML.

### Task 2: Installable verification skill and behavioral evaluations

**Files:**
- Create: `agent_tooling/verification/skill/SKILL.md`
- Create: `agent_tooling/verification/skill/references/contract.md`
- Create: `agent_tooling/verification/skill/references/criteria-code.md`
- Create: `agent_tooling/verification/skill/references/criteria-math.md`
- Create: `agent_tooling/verification/skill/references/criteria-evidence.md`
- Create: `agent_tooling/verification/skill/references/criteria-experiment.md`
- Create: `agent_tooling/verification/skill/references/criteria-general.md`
- Create: `agent_tooling/verification/skill/evals/evals.json`
- Test: `tests/test_verification_skill.py`

**Interfaces:**
- Consumes: the Task 1 CLI and ledger schema.
- Produces: a cross-platform `verification` skill whose workflow activates the gate, creates claims, dispatches domain roles, records criterion-level results, validates the ledger, and names it in the final response.

- [ ] **Step 1: Write failing skill-structure tests**

  Assert valid YAML frontmatter, required trigger phrases, references to every domain criterion, explicit state rules, adaptive repetition, order reversal, abstention, and the exact gate commands.

- [ ] **Step 2: Run skill tests and confirm RED**

  Run `python -m pytest tests/test_verification_skill.py --junitxml=C:\tmp\vfe3-verification-skill-red-20260718.xml` and confirm missing-skill failures.

- [ ] **Step 3: Write the minimal skill, contract, criteria, and eval corpus**

  Keep the main skill procedural and move domain detail into directly linked references. Add six evaluations covering code, mathematics, source claims, experiments, stale evidence, and required abstention.

- [ ] **Step 4: Validate the skill and run GREEN tests**

  Run the skill-creator `quick_validate.py` against `agent_tooling/verification/skill`, then run the focused pytest module and require zero failures and errors.

### Task 3: Neutral verifier agents and cross-platform renderer

**Files:**
- Create: `agent_tooling/verification/agents/verifier-orchestrator.json`
- Create: `agent_tooling/verification/agents/verifier-code.json`
- Create: `agent_tooling/verification/agents/verifier-math.json`
- Create: `agent_tooling/verification/agents/verifier-evidence.json`
- Create: `agent_tooling/verification/agents/verifier-skeptic.json`
- Create: `agent_tooling/verification/agents/verifier-adjudicator.json`
- Create: `agent_tooling/verification/install.py`
- Test: `tests/test_verification_install.py`

**Interfaces:**
- Produces: `render_claude_agent(spec: dict[str, object]) -> str`, `render_codex_agent(spec: dict[str, object]) -> str`, `upsert_marked_block(text: str, marker: str, block: str) -> str`, and `install(args: Namespace) -> None`.

- [ ] **Step 1: Write failing renderer and installer tests**

  Require both formats to preserve the exact neutral instruction body, render valid frontmatter/TOML quoting, install six agents, copy the skill, preserve unrelated files, merge hooks, and remain byte-identical on a second installation.

- [ ] **Step 2: Run installer tests and confirm RED**

  Run `python -m pytest tests/test_verification_install.py --junitxml=C:\tmp\vfe3-verification-install-red-20260718.xml` and confirm import or missing-artifact failures.

- [ ] **Step 3: Implement renderers, block merging, and installation**

  Render Claude `.md` and Codex `.toml` files from the same six JSON specs. Copy the skill recursively, upsert marked policy and deep-audit blocks, and append one Stop hook command while preserving all existing JSON keys and hook entries.

- [ ] **Step 4: Run GREEN and parity verification**

  Run the focused installer tests and parse every generated Claude and Codex instruction body to confirm semantic equality.

### Task 4: Repository and user-surface integration

**Files:**
- Create: `agent_tooling/verification/blocks/global-policy.md`
- Create: `agent_tooling/verification/blocks/project-policy.md`
- Create: `agent_tooling/verification/blocks/deep-audit-integration.md`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `docs/2026-07-18-edits.md`
- Install outside repository: user Claude/Codex instruction files, skill catalogs, agent catalogs, hook files, and deep-audit skills.

**Interfaces:**
- Consumes: the Task 3 installer.
- Produces: identical marked verifier policy across Claude and Codex, plus active deep-audit integration.

- [ ] **Step 1: Add marked policy blocks to repository instruction files**

  State the five claim statuses, evidence hierarchy, freshness rule, mandatory verification-skill trigger, and prohibition on LLM-only closure. Keep existing project instructions unchanged outside the marked block.

- [ ] **Step 2: Run installation into temporary homes**

  Use temporary Claude and Codex homes populated with representative existing settings. Inspect the second-run diff and require no change.

- [ ] **Step 3: Verify source hashes and install into real user homes**

  Capture pre-install hashes, run the installer with explicit user-home paths, parse both JSON hook files, run skill validation on both installed copies, and compare installed agent instructions with their neutral specs.

- [ ] **Step 4: Record the dated post-edit description**

  Append a verifier-control-plane section to the existing `docs/2026-07-18-edits.md`, including baseline failures, focused test evidence, installed surfaces, and the distinction between LLM support and evidence closure.

### Task 5: Final verification and repository lifecycle

**Files:**
- Verify all changed files.
- Track no temporary XML, marker, cache, or generated home fixture.

**Interfaces:**
- Produces: a verified commit, pushed task branch, merged `origin/main`, safely fast-forwarded live checkout when WIP permits, and removed task worktree/branch.

- [ ] **Step 1: Run focused and static verification**

  Run all verifier-control-plane tests with JUnit, the skill validator on source and installed copies, JSON/TOML parsers, `git diff --check`, and `git status --short`.

- [ ] **Step 2: Run the full suite and compare failure node IDs with baseline**

  Run `python -m pytest --junitxml=C:\tmp\vfe3-verifier-control-plane-final-20260718.xml`. Parse both XML files and require zero new failure node IDs; report the actual totals.

- [ ] **Step 3: Review requirements and staged diff**

  Re-read the approved design, verify every requirement against a file or test, inspect the staged diff, and ensure only intended files are tracked.

- [ ] **Step 4: Commit, push, merge, and clean up**

  Commit the implementation, push the task branch, merge it into `main`, push `main`, fetch and inspect `origin/main`, safely fast-forward the live checkout without touching WIP, remove the temporary worktree, delete the local task branch, and show final status for the live checkout.
