# Two-Panel Audit Remediation Design

## Scope and authority

This design governs remediation of the two July 15 audit reports:
`docs/audits/ultradeep-audit-codebase-2026-07-15.md` and
`docs/audits/ultradeep-audit-codebase-second-panel-2026-07-15.md`.
Their union contains 59 unique findings after deduplicating first-panel P8 with
S2-D6 and first-panel B5 with S2-I5.

The user supplied two binding architectural adjudications after the audits.
First, the affine output projection selected by `use_prior_bank=False` is
allowed; `use_prior_bank=True` is the opt-in pure decode path. S2-T1 therefore
requires no projection removal or default change. It is closed by preserving
and regression-testing the reachable prior-bank path and by describing the
affine path accurately. Second, a mathematically pure path need not be the
default or current value. Existing default and click-run configuration choices
must not be changed merely to select a pure route.

The implementation target is therefore 58 repairs plus one adjudicated
architecture-contract regression. A route-specific approximation is repaired
when its behavior is accurately named, its exactness metadata is truthful, its
unsafe domain is rejected or explicitly gated, and a reachable pure
alternative is covered by tests. Data-corruption, objective-fidelity,
serialization, and state-transition findings require behavioral fixes rather
than documentation alone.

## Design rules

The live checkout and its existing configuration WIP are read-only. All work
occurs on `codex/second-audit-remediation-20260715` in the isolated worktree.
No task may edit `train_vfe3.py`, `ablation.py`, or `scaling.py` merely to alter
the user's current regime. Those files may change only where a confirmed
driver, validation, artifact, or experiment-construction defect requires it.

Every behavioral repair follows red-green TDD. Tests must reproduce the exact
audit failure before production code changes. Exact geometry is tested under
nonorthogonal congruence or a trusted high-precision oracle as appropriate.
Approximate routes remain available under explicit toggles. Pure routes remain
reachable, and tests pin that reachability.

Family-dependent quantities are supplied through the family registry or one
authoritative family-dispatched helper. Gaussian formulas must not silently
consume Laplace scale parameters. The Laplace covariance is
`2 * diag(b**2)`, its mean Fisher weight is `1 / b**2`, and its trust-region
whitening scale is `b`.

Integrity contracts fail closed. Checkpoint resume binds cursor and RNG state
to dataset, tokenizer, token cap, cache identity, and content identity. Binary
cache metadata must agree exactly with the mapped byte length. Boolean-like
strings must be normalized or rejected before construction. Artifact and
figure publication uses unique temporary names and atomic final replacement.

Training state advances only after an accepted optimizer update. Scheduler,
EMA, projection, and Metropolis cadence all consume the same `did_step`
decision. Inner E-step differentiation runs in an explicit float32 island when
outer autocast is enabled. Performance repairs may eliminate transfers,
duplicate inference, repeated hashing, or repeated factorizations, but may not
change model semantics.

## Remediation units

| Unit | Findings |
|---|---|
| Objective and training transitions | B1, B2, B3, B6, S2-V1, S2-M1, S2-M2, S2-M3 |
| Gauge, SPD, and numerical geometry | B4, D5, S2-G1, S2-G2, S2-G3, S2-G4, S2-N1, S2-N2 |
| Statistical families and mixer | S2-I1 through S2-I6, S2-T2, S2-C1 |
| Data, resume, evaluation, and memory | S2-D1 through S2-D7, P3, P8 |
| Belief cache and serialization | T1, T2, T3, T4, Q4 |
| Drivers, reproducibility, and publication | Q1 through Q9, D1 through D4 |
| Runtime performance and diagnostics | P1, P2, P4 through P7, S2-C2 through S2-C4 |
| Architecture adjudication | S2-T1 |

Here unprefixed B, D, P, Q, and T identifiers refer to the first-panel report;
all `S2-` identifiers refer to the second-panel report. First-panel P8 and
second-panel S2-D6 share one implementation. First-panel B5 and second-panel
S2-I5 share one finite-configuration contract.

## Verification

Each remediation unit receives focused regression tests and a task-scoped
review. The final branch must pass the default and `--runslow` suites with
JUnit XML, parse every tracked Python file through `ast.parse`, and pass
`python -m pip check`. Test counts are reported only from JUnit. CUDA scheduling
repairs are source- and behavior-tested in the CPU-only environment; no GPU
timing claim is permitted unless CUDA is available in the final interpreter.

The required post-edit record is `docs/2026-07-15-edits.md`. It records each
finding's disposition as fixed, deduplicated, or user-adjudicated, with focused
and full verification evidence. The Research vault remains read-only during
this repository task.
