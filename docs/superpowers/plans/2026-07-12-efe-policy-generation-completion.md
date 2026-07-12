# EFE Policy Generation Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the registered H-step efe_rollout policy reachable through VFEModel.generate, and turn sigma_mc from a permanent stub into a strictly PASS-gated ambiguity consumer without changing any default inference path.

**Architecture:** Add a bounded top-k beam menu that returns H-token candidate policies and their accumulated base-model log prior. Keep the existing terminal-outcome EFE scorer and verified prefix-cache requirement. Extend rollout results with terminal Gaussian state so the sigma estimator can sample the same predicted belief without a second rollout. Bind every sigma-gate artifact to a deterministic behavior fingerprint of the measured semantic config plus model state, the current validation-specification identity, and a content-bound code identity; derive every consumer identity from the live model and repository rather than editable labels.

**Tech Stack:** Python 3, PyTorch, dataclasses and NamedTuple contracts, the existing policy/preference/ambiguity registries, pytest, and JUnit XML.

## Global Constraints

- Preserve policy_mode="none" byte-for-byte: its generate branch, sampling order, RNG consumption, logits validation, and output must not change.
- Preserve the existing efe_one_step menu and score behavior under all current defaults. New behavior is selected only by policy_mode="efe_rollout" or policy_ambiguity_mode="sigma_mc".
- Keep the default preference flat in generic generation. Task and held-out-predictive preferences remain harness-driven; a typed generic preference-context API is a future design outside this completion plan.
- Keep efe_rollout fail-closed outside belief_cache.cache_supported. The H-step menu does not relax the scorer cache gate.
- Keep ordinary autoregressive generation and menu expansion uncached. A generic incremental generation cache is a future performance design outside this completion plan, not a correctness prerequisite.
- Keep sigma_mc fail-closed unless the artifact is valid JSON, has status="PASS", and matches the current specification identity, content-bound code identity, and a deterministic fingerprint computed from the live semantic config plus model state. A user-entered checkpoint label is provenance only and cannot unlock the gate. The repository's current empirical sigma result is FAIL, so no shipped config may enable sigma_mc.
- Synthetic PASS artifacts may test gate plumbing, but documentation and test names must state that they do not establish empirical validity.
- Add no neural network, learned parameter, state-dict entry, CLI parser, or silent fallback.
- All new numerical tests run on CPU with embed_dim below 6. Keep float32 and use a local generator for Monte Carlo samples so the global RNG stream is unchanged.
- Follow the repository tensor-first signature convention and annotate every signature.
- Run pytest without an additional -q. Read pass/failure/error counts from JUnit XML before reporting them.
- Append implementation notes to docs/2026-07-12-edits.md; do not create a second post-edit document for the same date.

---

### Task 1: Build a bounded H-step candidate-policy menu

**Files:**

- Create: vfe3/inference/candidate_menu.py
- Create: tests/test_candidate_menu.py
- Reference: vfe3/model/model.py

**Interfaces:**

    @torch.no_grad()
    def build_topk_policy_menu(
        context:     torch.Tensor,
        base_logits: torch.Tensor,
        model:       "VFEModel",

        *,
        horizon: int,
        width:   int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return candidates (B, width, horizon) and normalized log_prior (B, width)."""

The menu is a bounded beam, not the Cartesian top-k product. Its live beam count is at most width at every depth. The first action uses the already-computed base_logits. Each later depth batches the current prefixes through model.rollout_beliefs with return_logits=True and decode_last=True, adds the next-token log probability to the accumulated policy score, and retains the best width joint sequences. The returned log_prior is log_softmax of the retained accumulated scores.

- [ ] Write tests that fail because candidate_menu.py does not exist.

    def test_topk_policy_menu_has_horizon_shape_and_normalized_prior():
        model = tiny_model(embed_dim=4, n_heads=1, vocab_size=11)
        context = torch.tensor([[1, 2]], dtype=torch.long)
        base_logits = model.rollout_beliefs(
            context, return_logits=True, decode_last=True
        )[1][:, 0, :]
        candidates, log_prior = build_topk_policy_menu(
            context, base_logits, model, horizon=3, width=4
        )
        assert candidates.shape == (1, 4, 3)
        assert log_prior.shape == (1, 4)
        torch.testing.assert_close(log_prior.exp().sum(-1), torch.ones(1))

    def test_topk_policy_menu_matches_exhaustive_search_on_tiny_vocabulary():
        model = tiny_model(embed_dim=4, n_heads=1, vocab_size=4)
        context = torch.tensor([[1, 2]], dtype=torch.long)
        _, decoded = model.rollout_beliefs(
            context, return_logits=True, decode_last=True
        )
        base_logits = decoded[:, 0, :]
        candidates, log_prior = build_topk_policy_menu(
            context, base_logits, model, horizon=2, width=4
        )

        first = torch.arange(4).reshape(1, 4, 1)
        expanded_context = context.unsqueeze(1).expand(1, 4, 2)
        extended = torch.cat([expanded_context, first], dim=-1).reshape(4, 3)
        _, next_decoded = model.rollout_beliefs(
            extended, return_logits=True, decode_last=True
        )
        first_logp = torch.log_softmax(base_logits[0], dim=-1)
        next_logp = torch.log_softmax(next_decoded[:, 0, :], dim=-1)
        joint = first_logp[:, None] + next_logp
        expected_score, flat = joint.reshape(-1).topk(4)
        expected = torch.stack((flat // 4, flat % 4), dim=-1)

        torch.testing.assert_close(candidates[0], expected)
        torch.testing.assert_close(
            log_prior[0], torch.log_softmax(expected_score, dim=-1)
        )

    def test_topk_policy_menu_batches_beams_and_never_exceeds_width(monkeypatch):
        model = tiny_model(embed_dim=4, n_heads=1, vocab_size=7)
        context = torch.tensor([[1, 2], [2, 3]], dtype=torch.long)
        _, decoded = model.rollout_beliefs(
            context, return_logits=True, decode_last=True
        )
        base_logits = decoded[:, 0, :]
        observed_batch_sizes = []
        original = model.rollout_beliefs

        def traced_rollout(token_ids, *args, **kwargs):
            observed_batch_sizes.append(token_ids.shape[0])
            return original(token_ids, *args, **kwargs)

        monkeypatch.setattr(model, "rollout_beliefs", traced_rollout)
        candidates, _ = build_topk_policy_menu(
            context, base_logits, model, horizon=4, width=3
        )

        assert candidates.shape == (2, 3, 4)
        assert observed_batch_sizes == [6, 6, 6]

- [ ] Run the focused tests and confirm the expected import failure.

    python -m pytest tests/test_candidate_menu.py --junitxml=C:\tmp\vfe3-candidate-menu-red.xml

Expected result: JUnit reports at least one error caused by the missing module or function, and no unrelated collection error.

- [ ] Implement the bounded beam exactly once in candidate_menu.py.

    if context.dim() != 2:
        raise ValueError(f"context must have shape (B, N), got {tuple(context.shape)}")
    B, N = context.shape
    if base_logits.dim() != 2 or base_logits.shape[0] != B:
        raise ValueError(
            f"base_logits must have shape (B, V) with B={B}, got {tuple(base_logits.shape)}"
        )
    vocab_size = base_logits.shape[-1]
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not 1 <= width <= vocab_size:
        raise ValueError(f"width must be in [1, {vocab_size}], got {width}")
    if not torch.isfinite(base_logits).all():
        raise ValueError("base_logits must be finite")
    _validate_policy_context(context, horizon, model.cfg.max_seq_len)

    first_logp = torch.log_softmax(base_logits, dim=-1)
    beam_score, first_token = first_logp.topk(width, dim=-1)
    candidates = first_token.unsqueeze(-1)

    for depth in range(1, horizon):
        beam_count = candidates.shape[1]
        ctx = context.unsqueeze(1).expand(B, beam_count, N).reshape(B * beam_count, N)
        actions = candidates.reshape(B * beam_count, depth)
        _, decoded = model.rollout_beliefs(
            torch.cat([ctx, actions], dim=-1),
            return_logits=True,
            decode_last=True,
        )
        next_logp = torch.log_softmax(decoded[:, 0, :], dim=-1)
        next_logp = next_logp.reshape(B, beam_count, vocab_size)
        if not torch.isfinite(next_logp).all():
            raise ValueError(f"rollout logits are non-finite at depth {depth + 1}")
        joint = beam_score.unsqueeze(-1) + next_logp
        beam_score, flat_index = joint.reshape(B, -1).topk(width, dim=-1)
        parent = torch.div(flat_index, vocab_size, rounding_mode="floor")
        token = flat_index.remainder(vocab_size)
        kept = torch.gather(
            candidates, 1, parent.unsqueeze(-1).expand(-1, -1, depth)
        )
        candidates = torch.cat([kept, token.unsqueeze(-1)], dim=-1)

    return candidates, torch.log_softmax(beam_score, dim=-1)

The validation above is part of the implementation, including the shared `_validate_policy_context` sequence-bound rule.

- [ ] Re-run the focused tests.

    python -m pytest tests/test_candidate_menu.py --junitxml=C:\tmp\vfe3-candidate-menu-green.xml

Expected result: JUnit failures=0 and errors=0.

- [ ] Commit the task.

    git add vfe3/inference/candidate_menu.py tests/test_candidate_menu.py
    git commit -m "feat: add bounded H-step policy menu"

### Task 2: Reach efe_rollout through generate without touching the no-policy branch

**Files:**

- Modify: vfe3/model/model.py
- Modify: generate_efe.py
- Modify: tests/test_generate.py
- Modify: tests/test_belief_cache.py

**Interfaces:**

VFEModel.generate keeps its public signature. VFEModel._policy_select keeps returning the committed first token with shape (B, 1).

- [ ] Replace test_generate_rejects_efe_rollout_policy_mode with a failing reachability test. Use a cache-supported VFE3Config, embed_dim=4, policy_horizon=2, and policy_top_k no greater than vocab_size.

    def test_generate_reaches_efe_rollout_and_commits_first_action():
        model = tiny_cache_supported_policy_model(
            policy_mode="efe_rollout",
            policy_horizon=2,
            policy_top_k=3,
        )
        prompt = torch.tensor([[1, 2]], dtype=torch.long)
        out = model.generate(prompt, max_new_tokens=1, greedy=True)
        assert out.shape == (1, 3)

- [ ] Add a scorer-spy test proving that candidates have length policy_horizon, log_prior is normalized, and generate appends candidates[selected_menu_index, 0], not the terminal action.

- [ ] Add regression tests proving that policy_mode="none" never imports or calls build_topk_policy_menu and that the existing seeded no-policy generation result is unchanged.

- [ ] Add a failure test proving that efe_rollout still rejects a configuration for which cache_supported is false.

- [ ] Run the focused tests and confirm that only the old NotImplementedError behavior fails.

    python -m pytest tests/test_generate.py tests/test_belief_cache.py --junitxml=C:\tmp\vfe3-efe-generate-red.xml

Expected result: the new reachability test fails at VFEModel._policy_select with the current H-step-generator error.

- [ ] In _policy_select, leave the one-step block unchanged and branch only at candidate construction:

    horizon = self.cfg.policy_horizon if self.cfg.policy_mode == "efe_rollout" else 1
    _validate_policy_context(context, horizon, self.cfg.max_seq_len)
    _belief, decoded = self.forward_beliefs(
        context, return_logits=True, decode_last=True
    )
    base_logits = decoded[:, 0, :]

    if self.cfg.policy_mode == "efe_rollout":
        candidates, log_prior = build_topk_policy_menu(
            context,
            base_logits,
            self,
            horizon=horizon,
            width=self.cfg.policy_top_k,
        )
    else:
        topk = base_logits.topk(self.cfg.policy_top_k, dim=-1).indices
        candidates = topk.unsqueeze(-1)
        menu_logits = torch.gather(base_logits, 1, topk)
        log_prior = torch.log_softmax(menu_logits, dim=-1)

After policy scoring, gather the selected row from candidates and return its first action:

    selected = torch.gather(
        candidates,
        1,
        idx.unsqueeze(-1).expand(-1, -1, candidates.shape[-1]),
    )
    return selected[:, 0, :1]

Keep every current base-logit and retained-logit finite check. Do not move policy code above the existing policy_mode="none" branch in generate.

- [ ] Update generate_efe.py configuration comments so policy_mode documents efe_rollout and its cache-supported, horizon-greater-than-one requirement. Keep the default policy_mode and horizon unchanged.

- [ ] Re-run the focused tests.

    python -m pytest tests/test_candidate_menu.py tests/test_generate.py tests/test_belief_cache.py --junitxml=C:\tmp\vfe3-efe-generate-green.xml

Expected result: JUnit failures=0 and errors=0.

- [ ] Commit the task.

    git add vfe3/model/model.py generate_efe.py tests/test_generate.py tests/test_belief_cache.py
    git commit -m "feat: reach H-step EFE policy generation"

### Task 3: Carry terminal belief state and validate a matching sigma gate

**Files:**

- Modify: vfe3/contracts.py
- Modify: vfe3/inference/policy.py
- Modify: vfe3/inference/belief_cache.py
- Modify: vfe3/inference/sigma_gate.py
- Modify: vfe3/run_artifacts.py
- Modify: vfe3/model/model.py
- Modify: vfe3/config.py
- Modify: sigma_gate_measure.py
- Restore: docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md from tracked blob `c05b927`
- Restore: docs/research/active-inference/2026-06-28-sigma-gate-prereg.md from its exact pre-deletion blob at `06d9238^`
- Restore: vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json from historical blob `f8467c51016014a6f0a5a76b569045581c921871`
- Create: vfe3/inference/sigma_gate_preregistry.json as the tracked, non-code authorization manifest
- Modify: tests/test_efe_scorer.py
- Modify: tests/test_belief_cache.py
- Modify: tests/test_policy_registry.py
- Modify: tests/test_run_artifacts.py
- Modify: tests/test_sigma_gate.py
- Modify: tests/test_generate.py

**Interfaces:**

    class PolicyRollout(NamedTuple):
        q_log:   torch.Tensor
        log_prob: torch.Tensor
        mu:       torch.Tensor
        sigma:    torch.Tensor

    def rollout_predictive_state_cached(
        context:     torch.Tensor,
        candidates:  torch.Tensor,
        model:       object,

        *,
        base_logits: Optional[torch.Tensor] = None,
    ) -> PolicyRollout:

    def verify_sigma_consumer_gate(
        path: str,

        *,
        actual_model_behavior_sha256: str,
        actual_spec_identity:          str,
        actual_code_identity_sha256:  str,
        actual_measurement_context_sha256: str,
    ) -> Dict[str, object]:

    def model_behavior_fingerprint(
        semantic_config: Mapping[str, object],
        state_dict: Mapping[str, torch.Tensor],
    ) -> str:
        """Hash canonical semantic config plus sorted tensor metadata and bytes."""

    def sigma_behavior_config(
        cfg: "VFE3Config | Mapping[str, object]",
    ) -> Dict[str, object]:
        """Return the non-policy config projection that controls belief transition and decode."""

    def sigma_measurement_context(
        cfg: "VFE3Config | Mapping[str, object]",

        *,
        cache_dir: Optional[Path] = None,
    ) -> Dict[str, object]:
        """Return the sealed loader/statistic/sampler context plus current corpus identity."""

    def _policy_select(
        self,
        context: torch.Tensor,

        *,
        greedy:               bool          = True,
        model_behavior_sha256:      Optional[str] = None,
        sigma_spec_identity:        Optional[str] = None,
        sigma_code_identity_sha256: Optional[str] = None,
        sigma_measurement_context_sha256: Optional[str] = None,
    ) -> torch.Tensor:

    def _efe_score(
        context:    torch.Tensor,
        candidates: torch.Tensor,
        preference: torch.Tensor,
        model:      object,

        *,
        gamma:               float,
        score_terms:         Tuple[str, ...],
        ambiguity_mode:      str,
        log_prior:           Optional[torch.Tensor],
        base_logits:         Optional[torch.Tensor],
        model_behavior_sha256:      Optional[str] = None,
        sigma_spec_identity:        Optional[str] = None,
        sigma_code_identity_sha256: Optional[str] = None,
        sigma_measurement_context_sha256: Optional[str] = None,
    ) -> PolicyScore:

    def _policy_efe_one_step(
        context:    torch.Tensor,
        candidates: torch.Tensor,
        preference: torch.Tensor,
        model:      object,

        *,
        gamma:               float                  = 1.0,
        horizon:             int                    = 1,
        score_terms:         Tuple[str, ...]        = ("risk", "ambiguity"),
        ambiguity_mode:      str                    = "likelihood_entropy",
        log_prior:           Optional[torch.Tensor] = None,
        base_logits:         Optional[torch.Tensor] = None,
        model_behavior_sha256:      Optional[str]          = None,
        sigma_spec_identity:        Optional[str]          = None,
        sigma_code_identity_sha256: Optional[str]          = None,
        sigma_measurement_context_sha256: Optional[str]    = None,
        **kwargs,
    ) -> PolicyScore:

    def _policy_efe_rollout(
        context:    torch.Tensor,
        candidates: torch.Tensor,
        preference: torch.Tensor,
        model:      object,

        *,
        gamma:               float                  = 1.0,
        horizon:             int                    = 2,
        score_terms:         Tuple[str, ...]        = ("risk", "ambiguity"),
        ambiguity_mode:      str                    = "likelihood_entropy",
        log_prior:           Optional[torch.Tensor] = None,
        base_logits:         Optional[torch.Tensor] = None,
        model_behavior_sha256:      Optional[str]          = None,
        sigma_spec_identity:        Optional[str]          = None,
        sigma_code_identity_sha256: Optional[str]          = None,
        sigma_measurement_context_sha256: Optional[str]    = None,
        **kwargs,
    ) -> PolicyScore:

Add these VFE3Config fields next to the existing sigma gate fields:

    policy_ambiguity_mode:   str = "likelihood_entropy"
    policy_sigma_mc_samples: int = 16

- [ ] Write failing tests for the terminal-state contract on both full and cached rollout paths. Assert q_log and log_prob remain equal to the existing two-tensor wrappers, and terminal mu/sigma have shapes (B, Kp, K) or (B, Kp, K, K).

- [ ] Restore both deleted governing preregistrations and the resolved FAIL artifact before gate work. Materialize the exact tracked blobs `c05b927:docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md`, `06d9238^:docs/research/active-inference/2026-06-28-sigma-gate-prereg.md`, and `06d9238^:vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json` at their original paths. Verify the first two against historical `git rev-parse <rev>:<path>` blob hashes (the prereg note begins `3010a5dd`) and the JSON against blob `f8467c51016014a6f0a5a76b569045581c921871`. Add `test_sigma_gate_spec_identity_is_known_on_tracked_tree` and `test_repository_sigma_gate_artifact_is_immutable_fail`. Do not reconstruct or silently revise these records from memory.

- [ ] Write failing config tests for unknown ambiguity keys, every sigma sample count other than the preregistered `16`, and sigma_mc without `policy_sigma_ambiguity_validated=True` or `policy_sigma_gate_artifact`. Configuration validates PASS status and the now-restored current specification identity; live model identity is unavailable until a model is constructed and is checked at the consumer boundary below.

- [ ] Write synthetic-artifact tests for missing, unreadable, FAIL, stale-specification, stale-code, missing-behavior-fingerprint, and wrong-live-model records. Construct two tiny models with different parameters and prove that an artifact measured from model A cannot open sigma_mc for model B even when both use the same human `checkpoint_id`. Also hold the state dict fixed while changing a behavior field such as `decode_tau` and require rejection. Use an artificial PASS record only to prove that the control-flow gate opens:

    synthetic_spec = sigma_gate_spec_identity(root=synthetic_governing_root)
    synthetic_code = sigma_consumer_code_identity(root=synthetic_code_root)
    measurement_context = sigma_measurement_context(model_a.cfg, cache_dir=tmp_path)
    record = {
        "status": "PASS",
        "checkpoint_id": "synthetic-test-checkpoint",
        "model_behavior_sha256": model_behavior_fingerprint(
            sigma_behavior_config(model_a.cfg), model_a.state_dict()),
        "spec_commit": synthetic_spec,
        "code_identity_sha256": synthetic_code,
        "measurement_context": measurement_context,
        "measurement_context_sha256": semantic_config_fingerprint(measurement_context),
        "seeds": [6, 23, 64],
        "sigma_ce_spearman": 0.5,
        "spearman_ci": [0.3, 0.7],
        "permutation_floor": 0.1,
        "stratified_ce": {"monotone": True},
        "sigma_binned_ece": 0.01,
        "thresholds": measurement_context["thresholds"],
    }
    synthetic_path.write_text(json.dumps(record), encoding="utf-8")
    synthetic_artifact_sha256 = canonical_json_sha256(synthetic_path)
    synthetic_manifest.write_text(json.dumps({synthetic_spec: {
        "status": "PASS", "artifact_sha256": synthetic_artifact_sha256, "test_only": True,
    }}), encoding="utf-8")

The test fixture injects these providers consistently through VFE3Config validation, `VFEModel.generate`, and `_amb_sigma_mc` defense-in-depth: the temporary spec identity, temporary code identity, precomputed temporary-cache measurement context, and temporary manifest loader. Production code should resolve them through `vfe3.inference.sigma_gate` at call time rather than capture unpatchable aliases. The test name and docstring must say synthetic PASS plumbing test. Its synthetic statistics must themselves satisfy the sealed rule; it must not call the empirical arm validated. Parameterize strict rejection over changed dataset/split bytes, effective sequence length, batch/max-batch selection, seed list, statistic threshold, MC seed/rule, and stored-context fingerprint tampering. Copy the restored historical FAIL JSON, change only `status` to PASS, and require rejection because its production manifest entry remains FAIL and recomputed status remains FAIL.

- [ ] Run the focused tests and confirm the expected missing-contract and missing-config failures.

    python -m pytest tests/test_efe_scorer.py tests/test_belief_cache.py tests/test_policy_registry.py tests/test_run_artifacts.py tests/test_sigma_gate.py tests/test_generate.py --junitxml=C:\tmp\vfe3-sigma-contract-red.xml

- [ ] Add PolicyRollout to contracts.py. Introduce _rollout_predictive_state in policy.py and rollout_predictive_state_cached in belief_cache.py. The full path takes the last appended position from the returned BeliefState; the cached path takes the last appended position after the same block_norm and final_norm processing already used for q_log.

- [ ] Keep _rollout_predictive and rollout_predictive_cached as compatibility wrappers returning exactly (state.q_log, state.log_prob). Do not change existing external unpacking.

- [ ] Add `model_behavior_fingerprint` to `vfe3.run_artifacts`. Prefix the digest with `semantic_config_fingerprint(semantic_config)`, then hash sorted state-dict keys with length delimiters, each tensor's dtype and shape, and `tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)` bytes. Reject non-tensor values. `sigma_behavior_config(cfg)` accepts either a `VFE3Config` or mapping, starts from `asdict(cfg)` or `dict(cfg)`, and removes every key beginning with `policy_`: these fields choose menus, horizons, preferences, score weights, ambiguity dispatch, and gate authorization around an already defined candidate, but do not alter the underlying `rollout_predictive_state` transition or `PriorBank.decode` distribution whose sigma utility was measured. Every non-policy field remains bound, including `decode_tau`, family/divergence, E-step, transport, and prior-bank settings. Tests prove key-order invariance, sensitivity to one changed value, non-policy config/dtype/shape sensitivity, equality before and after save/load, and equality between an otherwise identical checkpoint config with `policy_mode="none"` and a consumer config with `policy_mode="efe_rollout"` plus different preference/score/top-k/horizon/gate fields.

- [ ] Move the specification-identity calculation from `sigma_gate_measure.py` into `sigma_gate_spec_identity(root=None)` in `vfe3.inference.sigma_gate`. Sort the governing relative paths lexicographically, then hash each UTF-8 path and its UTF-8 content after canonical CRLF/CR-to-LF normalization, with an 8-byte big-endian length delimiter immediately before each path and content payload; it never includes a git commit SHA, so the restored-doc commit is not circular. The exact restored pair yields `c136c3242abb6a091d091c67020f0a73746401f478e19fb74b2d6d1e53096691`; add a golden test that reads the historical blobs and pins this literal. Missing or undecodable files return `"unknown"`. Producer and consumer call this helper. Tests prove LF/CRLF parity, input-order invariance, and that changing either file changes identity. Add `sigma_consumer_code_identity(root=None) -> str`, which hashes normalized relative path plus bytes for every sorted `vfe3/**/*.py` file and `sigma_gate_measure.py`, excluding `__pycache__`, generated artifacts, tests, docs, and the JSON preregistry. It raises when any declared source cannot be read. Tests prove that writing/replacing the gate JSON or updating the preregistry leaves code identity unchanged and that changing a copied policy/model source changes it.

- [ ] Centralize the sealed measurement context in `vfe3.inference.sigma_gate`: dataset `wikitext-103`, split `test`, requested sequence length 128, batch size 16, max batches 20, `shuffle=False`, `drop_last=True`, seeds `(6,23,64)`, sigma samples 16, `mc_seed=0`, `sampling_rule="antithetic_shared_v1"`, and the exact gate statistic settings `spearman_min=0.2`, `ece_max=0.05`, `n_strata=10`, `n_bins=10`, `n_boot=2000`, `n_perm=1000`, `alpha=0.05`, statistic seed 0. `sigma_measurement_context` adds the effective `min(128, cfg.max_seq_len)`, tokenizer tag, and `cache_source_identity(dataset, split, cache_dir)` supplied by the earlier artifact-integrity plan. `sigma_gate_measure.py` must reject any edited CONFIG value that differs from this sealed mapping rather than silently measuring another context. The consumer derives the same mapping from the live config and current cache; missing or changed corpus bytes fail closed.

- [ ] Encode preregistration outcomes in `vfe3/inference/sigma_gate_preregistry.json`, loaded fail-closed as a mapping keyed by the exact content-based governing identity. The shipped manifest contains only `c136c3242abb6a091d091c67020f0a73746401f478e19fb74b2d6d1e53096691` with `status="FAIL"` and historical artifact canonical-JSON SHA-256 `f2b55e2f45e9d7146c9f96b371c2a971df43c7a2c6affb6bf1b2941a28205d9f`. Canonical artifact hashing parses JSON and hashes `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`, so LF/CRLF/indentation cannot change identity. `verify_sigma_consumer_gate` rejects unregistered identities before reading PASS, rejects every status other than PASS, and for a future PASS requires the artifact's canonical hash to match the manifest. `sigma_gate_measure.py` and the writer refuse to publish or overwrite a PASS under resolved FAIL. A reproduction is diagnostic-only. A future real PASS uses a new prereg identity added as UNMEASURED, measurement under unchanged Python code, then a separate reviewed manifest-only update to PASS plus canonical artifact hash; because the JSON manifest is excluded from behavior-code hashing, this two-phase authorization is non-circular. Synthetic PASS tests use a distinct temporary governing root and temporary manifest; they never use live `sigma_gate_spec_identity()` or the production manifest.

- [ ] Make `model_behavior_sha256`, `code_identity_sha256`, `measurement_context`, and `measurement_context_sha256` required artifact fields. `sigma_gate_measure.py` derives them from the semantic config/state actually loaded for measurement, declared source identity, and sealed loader/statistic context, then passes them through `measure_sigma_gate()` and `write_sigma_gate_artifact()`. The writer recomputes the context fingerprint from the stored mapping. Keep `checkpoint_id` as a human-readable label, but never use it for authorization.

- [ ] Implement verify_sigma_consumer_gate as a strict wrapper over derived identities:

    prereg = load_sigma_gate_preregistry().get(actual_spec_identity)
    if prereg is None or prereg.get("status") != "PASS":
        raise ValueError("sigma-gate governing identity is not registered as PASS")
    if canonical_json_sha256(Path(path)) != prereg.get("artifact_sha256"):
        raise ValueError("sigma-gate artifact bytes do not match the preregistration registry")
    record = verify_gate_artifact(
        path,
        expected_spec_commit=actual_spec_identity,
        require_pass=True,
    )
    if record.get("model_behavior_sha256") != actual_model_behavior_sha256:
        raise ValueError(
            "sigma-gate model-behavior fingerprint does not match the live model"
        )
    if record.get("code_identity_sha256") != actual_code_identity_sha256:
        raise ValueError("sigma-gate code identity does not match the live source")
    context = record.get("measurement_context")
    if (not isinstance(context, Mapping)
            or semantic_config_fingerprint(context) != record.get("measurement_context_sha256")
            or record.get("measurement_context_sha256") != actual_measurement_context_sha256):
        raise ValueError("sigma-gate measurement context does not match sealed live data/statistics")
    if record.get("seeds") != context.get("seeds"):
        raise ValueError("sigma-gate duplicated seed provenance contradicts its sealed context")
    if record.get("thresholds") != context.get("thresholds"):
        raise ValueError("sigma-gate measured thresholds contradict its sealed context")
    thresholds = context["thresholds"]
    computed_pass = bool(
        float(record["sigma_ce_spearman"]) >= float(thresholds["spearman_min"])
        and float(record["spearman_ci"][0]) > 0.0
        and float(record["spearman_ci"][0]) > float(record["permutation_floor"])
        and record["stratified_ce"]["monotone"] is True
        and float(record["sigma_binned_ece"]) < float(thresholds["ece_max"])
    )
    if record.get("status") != ("PASS" if computed_pass else "FAIL"):
        raise ValueError("sigma-gate status contradicts its stored statistics")
    return record

- [ ] Validate policy_ambiguity_mode through the ambiguity registry. When it is sigma_mc, require policy mode efe_one_step or efe_rollout, family gaussian_diagonal or gaussian_full, policy_sigma_ambiguity_validated=True, a gate artifact, and exactly the preregistered `policy_sigma_mc_samples=16`; direct `_amb_sigma_mc` dispatch also rejects any other S. During VFE3Config construction, derive `current_spec = sigma_gate_spec_identity()`, reject `current_spec == "unknown"`, and call a prereg-aware artifact verifier that rejects unregistered/resolved-FAIL identities, byte-hash mismatch, status/statistic contradiction, and stale spec. It does not pretend configuration can verify the not-yet-constructed live model or current corpus; those checks remain at the consumer boundary.

- [ ] At the start of `VFEModel.generate`, when and only when `policy_ambiguity_mode="sigma_mc"`, compute `model_behavior_fingerprint(sigma_behavior_config(self.cfg), self.state_dict())`, `sigma_gate_spec_identity()`, `sigma_consumer_code_identity()`, and the sealed measurement-context fingerprint once, reject an `"unknown"` specification identity, and call `verify_sigma_consumer_gate` with all four derived values. Pass them through the exact Optional keyword chain above. `_efe_score` requires all four values when `ambiguity_mode="sigma_mc"` and ignores them for every other registered ambiguity. A direct `_policy_select` or scorer call therefore remains source-compatible under existing modes and fails closed under sigma_mc when any identity is absent. The policy_mode="none" branch must not hash the model, code, or corpus, inspect the specification, or read the artifact.

- [ ] Retain the existing validation behavior when policy_sigma_ambiguity_validated=True under another ambiguity mode. This flag must never turn sigma_mc on by itself.

- [ ] Re-run the focused tests.

    python -m pytest tests/test_efe_scorer.py tests/test_belief_cache.py tests/test_policy_registry.py tests/test_run_artifacts.py tests/test_sigma_gate.py tests/test_generate.py --junitxml=C:\tmp\vfe3-sigma-contract-green.xml

Expected result: JUnit failures=0 and errors=0.

- [ ] Commit the task.

    git add vfe3/contracts.py vfe3/inference/policy.py vfe3/inference/belief_cache.py vfe3/inference/sigma_gate.py vfe3/inference/sigma_gate_preregistry.json vfe3/run_artifacts.py vfe3/model/model.py vfe3/config.py sigma_gate_measure.py docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md docs/research/active-inference/2026-06-28-sigma-gate-prereg.md vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json tests/test_efe_scorer.py tests/test_belief_cache.py tests/test_policy_registry.py tests/test_run_artifacts.py tests/test_sigma_gate.py tests/test_generate.py
    git commit -m "feat: validate sigma ambiguity consumer gate"

### Task 4: Implement the gated sigma_mc ambiguity estimator

**Files:**

- Modify: vfe3/contracts.py
- Modify: vfe3/model/prior_bank.py
- Modify: vfe3/inference/policy.py
- Modify: tests/test_efe_scorer.py
- Modify: tests/test_generate.py
- Modify: tests/test_policy_registry.py
- Modify: tests/test_prior_bank.py

**Interfaces:**

    class AmbiguityEstimate(NamedTuple):
        predictive_log_prob:          torch.Tensor  # (B, Kp, V), normalized MC marginal
        expected_conditional_entropy: torch.Tensor  # (B, Kp), E_s H[p(o|s)]

    def decode_point(
        self,
        mu_s: torch.Tensor,
    ) -> torch.Tensor:
        """Decode p(o|s) using an eps-floor covariance of the bank's configured rank."""

    def _amb_sigma_mc(
        q_log: torch.Tensor,

        *,
        mu:          torch.Tensor,
        sigma:       torch.Tensor,
        model:       object,
        num_samples: int,
        model_behavior_sha256:      Optional[str] = None,
        spec_identity:              Optional[str] = None,
        code_identity_sha256:       Optional[str] = None,
        measurement_context_sha256: Optional[str] = None,
        **kwargs,
    ) -> AmbiguityEstimate:

- [ ] Replace the permanent-raise test with fail-closed runtime tests. Direct dispatch must still raise when the validated config and matching artifact are absent, even if a caller bypasses VFE3Config construction.

- [ ] Add an artificial-PASS estimator test with embed_dim=4 and a fixed terminal Gaussian. Compute the same antithetic draws explicitly, average sample probabilities to the predictive marginal, and compare both returned marginal log-probabilities and expected conditional entropy.

- [ ] Add tests proving that zero covariance approaches likelihood_entropy, nonzero covariance changes the estimator on a nondegenerate decoder, candidate permutation permutes both result tensors, repeated calls are deterministic, and the global torch RNG state is unchanged. Cover both diagonal and full families; the full zero/near-singular covariance case must remain finite and converge to likelihood_entropy through the repository's safe Cholesky policy. From the same direct samples, require `predictive_entropy - expected_conditional_entropy >= -1e-6` and compare risk against KL of the sampled marginal, preventing a point-decode/MC-ambiguity mismatch. Require `predictive_log_prob.exp().sum(-1) == 1` within tolerance under extreme logits, include exact `-inf` sample-log-probability tails without NaN entropy, and reject direct S values other than 16.

- [ ] Add `test_efe_rollout_sigma_mc_cuda_synthetic_pass` to `tests/test_generate.py`, guarded by the repository CUDA-device convention. It uses the same temporary governing root, code root, cache, manifest, canonical artifact hash, and call-time provider injection as the CPU synthetic fixture. Build the tiny live model only after injection, keep the providers active through `generate` and `_amb_sigma_mc`, and include the RNG/device assertions specified in Task 6. The name and docstring must state that this is synthetic PASS plumbing only.

- [ ] Add prior-bank tests proving decode_point delegates to the active registered decoder, includes the unigram bias when enabled, and adds no parameter or state-dict key.

- [ ] Run the focused tests and confirm the existing RuntimeError is the expected red failure.

    python -m pytest tests/test_efe_scorer.py tests/test_prior_bank.py --junitxml=C:\tmp\vfe3-sigma-estimator-red.xml

- [ ] Implement `decode_point` with `self.eps` as the state covariance floor. Infer the rank from `self.diagonal_covariance`: use `torch.full_like(mu_s, self.eps)` for the diagonal/linear path and `self.eps * eye(K)` expanded over `mu_s.shape[:-1]` for full covariance. Route through `PriorBank.decode` so decode-mode registration and unigram bias remain single-sourced. Add a full-family test that monkeypatches the registered decoder and asserts it receives shape `(B, Kp, S, K, K)`, preventing silent use of the diagonal floor.

- [ ] Implement the sealed `antithetic_shared_v1` reparameterized sampler. Require `num_samples == 16`. Initialize a local `torch.Generator` on the belief device with `mc_seed=0`, draw `num_samples // 2` standard-normal samples shared across candidates in each batch, and concatenate their negatives in positive-then-negative order. For diagonal covariance use `sigma.clamp_min(model.cfg.eps).sqrt()`. For full covariance call `safe_cholesky(sigma, eps=model.cfg.eps, rounds=5)`, require its per-item `ok` mask, and use the returned jittered factor; do not call raw `torch.linalg.cholesky`. Preserve the leading axes: sampled means have shape `(B, Kp, S, K)`, `decode_point` receives that tensor and constructs covariance `(B, Kp, S, K)` or `(B, Kp, S, K, K)`, and registered decode returns logits `(B, Kp, S, V)`. Compute `sample_log_prob=log_softmax(logits, -1)`, then form the marginal stably as `predictive_log_prob = logsumexp(sample_log_prob, dim=2) - log(S)` followed by `log_softmax(..., dim=-1)` to remove numerical normalization drift. For entropy use `sample_prob=sample_log_prob.exp()` and `where(sample_prob > 0, sample_prob * sample_log_prob, 0)` before summing and averaging, so exact `-inf` tails cannot produce `0 * -inf` NaNs. Return both in `AmbiguityEstimate`; do not flatten B/Kp/S.

- [ ] At the start of _amb_sigma_mc, require all four identities derived by `VFEModel.generate` and call `verify_sigma_consumer_gate(model.cfg.policy_sigma_gate_artifact, actual_model_behavior_sha256=model_behavior_sha256, actual_spec_identity=spec_identity, actual_code_identity_sha256=code_identity_sha256, actual_measurement_context_sha256=measurement_context_sha256)` again as defense in depth. This rereads the artifact on every sigma dispatch so replacement after model construction fails closed, but it does not rehash the model/source/corpus on every generated token. Direct registry dispatch without any derived identity must raise.

- [ ] Change both ambiguity registrants to return `AmbiguityEstimate`. `likelihood_entropy` returns the input point `q_log` unchanged plus its entropy, preserving the default scorer exactly. Migrate every direct registry consumer and `tests/test_policy_registry.py` assertion to read the two named fields; no direct tensor-return contract remains undocumented. Change `_efe_score` to dispatch before computing risk, then use the returned predictive marginal consistently:

    rollout = _rollout_predictive_state(
        context, candidates, model, base_logits=base_logits
    )
    estimate = get_ambiguity(ambiguity_mode)(
        rollout.q_log,
        mu=rollout.mu,
        sigma=rollout.sigma,
        model=model,
        num_samples=model.cfg.policy_sigma_mc_samples,
        model_behavior_sha256=model_behavior_sha256,
        spec_identity=sigma_spec_identity,
        code_identity_sha256=sigma_code_identity_sha256,
        measurement_context_sha256=sigma_measurement_context_sha256,
    )
    risk, pred_ent = _efe_terms(estimate.predictive_log_prob, preference)
    ambiguity = estimate.expected_conditional_entropy
    epistemic = pred_ent - ambiguity

Pass `ambiguity_mode=self.cfg.policy_ambiguity_mode`, `model_behavior_sha256=model_behavior_sha256`, `sigma_spec_identity=sigma_spec_identity`, `sigma_code_identity_sha256=sigma_code_identity_sha256`, and `sigma_measurement_context_sha256=sigma_measurement_context_sha256` from `VFEModel._policy_select`. Both EFE policy functions forward those keywords to `_efe_score`. The default likelihood_entropy path ignores the four identity values and preserves its output.

- [ ] Re-run the focused tests.

    python -m pytest tests/test_efe_scorer.py tests/test_generate.py tests/test_prior_bank.py tests/test_policy_registry.py --junitxml=C:\tmp\vfe3-sigma-estimator-green.xml

Expected result: JUnit failures=0 and errors=0. All executable sigma tests use explicitly labeled synthetic PASS records; the current empirical FAIL remains authoritative.

- [ ] Commit the task.

    git add vfe3/contracts.py vfe3/model/prior_bank.py vfe3/inference/policy.py tests/test_efe_scorer.py tests/test_generate.py tests/test_policy_registry.py tests/test_prior_bank.py
    git commit -m "feat: add fail-closed sigma ambiguity estimator"

### Task 5: Expose only the validated policy fields in the click-to-run driver

**Files:**

- Modify: generate_efe.py
- Modify: tests/test_policy_registry.py
- Modify: README.md
- Modify: docs/2026-07-12-edits.md

- [ ] Extend generate_efe.py _POLICY_FIELDS with policy_ambiguity_mode, policy_sigma_mc_samples, policy_sigma_ambiguity_validated, and policy_sigma_gate_artifact. Do not expose checkpoint, behavior-fingerprint, specification, code, or measurement-context override fields: the consumer derives all four authorization identities from the live model, declared source surface, and sealed corpus/statistic context.

- [ ] Preserve every existing user-edited `generate_efe.py` CONFIG value, including its live `policy_mode="efe_one_step"`, horizon, preference, score terms, and top-k. Add only `policy_ambiguity_mode="likelihood_entropy"`, `policy_sigma_mc_samples=16`, `policy_sigma_ambiguity_validated=False`, and `policy_sigma_gate_artifact=None` as inert defaults. Do not include a PASS path or change any protected policy choice.

- [ ] Add a driver test proving that an attempted sigma_mc override with the repository's FAIL record is rejected before generation.

- [ ] Update the README policy status table: efe_rollout is available only with an H-step bounded menu plus cache-supported scorer; sigma_mc has an executable estimator but remains gate-closed until a matching empirical PASS exists.

- [ ] Record the exact interface, fail-closed behavior, and test commands in the dated edit document.

- [ ] Run the focused driver and policy tests.

    python -m pytest tests/test_generate.py tests/test_policy_registry.py --junitxml=C:\tmp\vfe3-efe-driver-green.xml

Expected result: JUnit failures=0 and errors=0.

- [ ] Commit the task.

    git add generate_efe.py tests/test_policy_registry.py README.md docs/2026-07-12-edits.md
    git commit -m "docs: expose completed EFE policy controls"

### Task 6: Verify numerical and pure-path regressions

- [ ] Compile the touched Python modules.

    python -m compileall vfe3 generate_efe.py sigma_gate_measure.py

Expected result: exit code 0.

- [ ] Run the complete focused policy suite with machine-readable results.

    python -m pytest tests/test_candidate_menu.py tests/test_generate.py tests/test_efe_scorer.py tests/test_belief_cache.py tests/test_policy_registry.py tests/test_prior_bank.py tests/test_sigma_gate.py tests/test_run_artifacts.py --junitxml=C:\tmp\vfe3-efe-policy-final.xml

Expected result: JUnit failures=0 and errors=0.

- [ ] Run the full suite once.

    python -m pytest --junitxml=C:\tmp\vfe3-efe-policy-full.xml

Expected result: exit code 0 and JUnit failures=0 and errors=0. Report the tests count only from the XML tests attribute.

- [ ] Run one RTX 5090 smoke with `K=4`, a tiny vocabulary, horizon two, `policy_mode="efe_rollout"`, a temporary cache implementing the sealed measurement-context fixture, and an explicitly labeled synthetic PASS artifact whose behavior/spec/code/context identities match the live test model. Reuse the CPU synthetic fixture's call-time provider injection—temporary `sigma_gate_spec_identity`, code identity, measurement context, and manifest loader—before constructing VFE3Config, and keep it active through generate plus defense-in-depth verification. Capture global CPU and CUDA RNG states immediately before two identical sigma_mc generations, require finite scores on CUDA, CUDA-resident terminal beliefs, equal outputs from the estimator's private local generator, and unchanged captured global RNG states after each call. The test name and assertion messages must state that synthetic PASS exercises plumbing only and does not supersede the immutable empirical FAIL.

    $env:VFE3_TEST_DEVICE = "cuda"
    python -m pytest tests/test_efe_scorer.py tests/test_generate.py -k "efe_rollout_sigma_mc_cuda_synthetic_pass" --junitxml=C:\tmp\vfe3-efe-policy-cuda.xml

Expected result: the CUDA test runs rather than skips on the RTX 5090, and JUnit records zero failures/errors.

- [ ] Inspect the final diff for accidental changes to the policy_mode="none" body and for state-dict additions.

    git diff origin/main...HEAD -- vfe3/model/model.py vfe3/model/prior_bank.py vfe3/inference
    git status --short

- [ ] Commit any verification-document correction, then follow the repository lifecycle: push the task branch, merge into main after verification, push main, safely fast-forward the live checkout only if user WIP is untouched, remove the temporary worktree, and show final git status.
