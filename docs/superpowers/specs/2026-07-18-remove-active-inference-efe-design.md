# Active-Inference and EFE Removal Design

## Status and scope

The user approved complete removal of the live active-inference and Expected-Free-Energy policy subsystem from V3 Transformer. The removal covers executable behavior, public configuration, dedicated experiments, policy-specific measurement artifacts, current user guidance, and tests whose only contract is the deleted subsystem. It does not remove the model's variational-free-energy objective, Gaussian belief refinement, attention entropy, ordinary autoregressive generation, or general theory recorded in historical audit documents.

Historical dated edit records and audits remain unchanged because they describe the repository at their recorded commits. The separate Research vault is outside this change. A new dated edit entry records the deletion without rewriting provenance.

## Production removal boundary

The three root drivers `efe_ring_experiment.py`, `generate_efe.py`, and `sigma_gate_measure.py` are deleted. The dedicated inference modules `vfe3/inference/policy.py`, `vfe3/inference/ring_task.py`, `vfe3/inference/belief_cache.py`, `vfe3/inference/candidate_menu.py`, and `vfe3/inference/sigma_gate.py` are deleted with the sigma preregistry and tracked sigma-gate result.

`VFE3Config` no longer exposes any `policy_*` field or validates policy, preference, ambiguity, horizon, candidate-menu, or sigma-gate state. `VFEModel.generate()` retains one ordinary autoregressive path with its existing greedy, temperature, top-k, top-p, finite-logit, context-truncation, and memory-warning behavior. The policy branch, `_policy_select()`, sigma consumer gate, full-context policy behavior, and policy-specific sampler validation are removed.

The policy-only `AmbiguityEstimate` and `PolicyRollout` contracts are removed. Policy-only code-identity inputs, sigma behavior projections, and model behavior fingerprints are removed from `vfe3/run_artifacts.py`. Current README guidance and visualization text that present the policy scorer or sigma gate as available behavior are removed. Unrelated configuration registries, inference modules, geometry code, generation behavior, reporting, and checkpoint integrity logic remain unchanged.

## Serialized-checkpoint compatibility

Ordinary checkpoints written while the default-off policy fields existed commonly contain those inert fields in their serialized configuration. Removing the dataclass fields without migration would make strict checkpoint consumers reject otherwise valid non-policy checkpoints.

The serialized-config migration boundary therefore recognizes the ten former policy field names as retired keys. It removes them before constructing `VFE3Config`, records them in `consumed_retired_keys`, and emits a retirement warning. Strict unknown-field rejection remains active for every unrecognized key. Raw historical configuration fingerprints are still verified before migration, and selection compatibility still compares the effective current configuration after migration. This tombstone is load compatibility only: it creates no runtime setting, registry, scorer, dispatch path, or EFE behavior. EFE-specific experiment continuation is unsupported after the removal.

## Test deletion and retained coverage

The dedicated belief-cache, candidate-menu, EFE scorer, EFE ring, policy-registry, ring-task, and sigma-gate test modules are deleted with their production subjects. Mixed test modules lose only imports, fixtures, and test cases that exercise those subjects. Tests for ordinary generation, configuration construction, serialized migration, artifact fingerprints, visualization, geometry, training, and non-policy runtime behavior remain.

Implementation begins with a removal-contract test that fails against the pre-removal tree. The retained regression contract proves that the public config has no policy controls, ordinary generation remains available, and a historical serialized config containing the retired inert fields migrates to the current config without restoring policy behavior. Repository searches then verify that live Python and current README surfaces contain no policy scorer, EFE dispatch, sigma gate, or active-inference experiment. Remaining matches are limited to historical provenance, the explicit compatibility tombstone, and the dated removal record.

## Verification and acceptance

The clean `origin/main` baseline JUnit contains 3,919 tests, 14 failures, zero errors, and 17 skips. Three failures belong to the deleted EFE subsystem; the other eleven are accepted pre-existing failures. The user authorized a differential verification gate: the final suite may retain those eleven unrelated node IDs but may introduce no new failure node ID.

Focused verification covers the removal contract, configuration migration, ordinary generation, checkpoint loading, run-artifact behavior, and every surgically edited mixed test module. Full verification writes JUnit XML and compares final failures programmatically against the accepted baseline set. Counts are read only from JUnit. Static checks include `git diff --check` and repository searches for forbidden live surfaces. Task-owned XML and other temporary evidence stay outside the repository and are deleted during closeout.

The change is complete only when the intended diff is committed, the branch is pushed, the verified branch is merged into `main`, `origin/main` is pushed, the user's live checkout is fast-forwarded only if its existing WIP can be preserved, and the temporary worktree and local task branch are removed.
