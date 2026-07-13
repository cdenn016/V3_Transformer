"""Tests for VFEModel.generate -- the additive autoregressive sampler.

generate reuses VFEModel.forward (targets=None -> logits (B, N, V)), takes the
last-position logits, turns them into a next token (greedy or sampled), appends,
and repeats. It never touches the training/loss branch, so it cannot corrupt
training; that training-isolation is the safety oracle below.

Forward-comparison oracles (greedy == forward+argmax, top_k membership) hold only
for the FIRST generated token: generate's step-1 internal forward sees exactly the
prompt, so logits[:, -1, :] there equals forward(prompt)[:, -1, :]. Step 2+
conditions on a longer sequence and has no relation to forward(prompt). Those
oracles therefore use max_new_tokens=1.
"""

import os

import pytest
import torch

import generate_efe
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

_DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _tiny_model(seed: int = 0, **overrides) -> VFEModel:
    """A tiny fixed-seed VFEModel. vocab_size is kept comfortably larger than any
    top_k used below so torch.topk never sees k > vocab_size."""
    base = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0, seed=seed)
    base.update(overrides)
    cfg = VFE3Config(**base)
    torch.manual_seed(seed)
    return VFEModel(cfg)


def _tiny_cache_supported_policy_model(seed: int = 0, **overrides) -> VFEModel:
    """A tiny cache-supported policy model (audit PB-05). The tiny defaults (n_layers=1, n_e_steps=1,
    e_phi_lr=0, flat transport, causal filtering) are the cache-supported regime, so efe_rollout is
    reachable through generate(); the assert pins that the fixture stays supported if the defaults
    drift (an unsupported fixture would make efe_rollout fail closed instead of exercising the menu)."""
    from vfe3.inference.belief_cache import cache_supported
    model = _tiny_model(seed=seed, **overrides)
    assert cache_supported(model.cfg), "fixture must be cache-supported for efe_rollout"
    return model


def test_shape_in_vocab_and_prompt_preserved():
    model = _tiny_model()
    V = model.cfg.vocab_size
    B, N0 = 2, 3
    prompt = torch.randint(0, V, (B, N0))
    out = model.generate(prompt, max_new_tokens=5)
    assert out.shape == (B, N0 + 5)
    assert (out >= 0).all() and (out < V).all()            # every id in [0, V)
    assert torch.equal(out[:, :N0], prompt)                # prompt preserved


def test_greedy_is_deterministic():
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    a = model.generate(prompt, max_new_tokens=4, greedy=True)
    b = model.generate(prompt, max_new_tokens=4, greedy=True)
    assert torch.equal(a, b)


def test_greedy_equals_forward_argmax_first_token():
    # The pin: generate's first greedy token == argmax of the last-position logits
    # from a direct forward on the SAME prompt (no hidden divergence). First token
    # only -- step 2+ conditions on a longer sequence. Prompt <= max_seq_len so the
    # internal forward and this comparison forward see identical input.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    logits = model(prompt)                                 # (B, N0, V), targets=None
    expected = logits[:, -1, :].argmax(dim=-1)             # (B,)
    out = model.generate(prompt, max_new_tokens=1, greedy=True)
    assert torch.equal(out[:, -1], expected)


def test_generate_decodes_only_last_position(monkeypatch):
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    empty = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(ValueError, match=r"decode_last=True requires a nonempty token context"):
        model.forward_beliefs(empty, return_logits=True, decode_last=True)
    with pytest.raises(ValueError, match=r"generate requires a nonempty token context"):
        model.generate(empty, max_new_tokens=1, greedy=True)
    original_forward_beliefs = model.forward_beliefs
    calls = []

    def tracked_forward_beliefs(
        token_ids,
        *,
        return_logits=False,
        decode_last=False,
        **kwargs,
    ):
        belief, logits = original_forward_beliefs(
            token_ids,
            return_logits=return_logits,
            decode_last=decode_last,
            **kwargs,
        )
        calls.append((return_logits, decode_last, tuple(logits.shape)))
        return belief, logits

    monkeypatch.setattr(model, "forward_beliefs", tracked_forward_beliefs)
    out = model.generate(prompt, max_new_tokens=2, greedy=True)

    assert out.shape == (2, 5)
    assert calls == [
        (True, True, (2, 1, V)),
        (True, True, (2, 1, V)),
    ]


def test_generate_rejects_nonfinite_logit_rows(monkeypatch):
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    injected = {"logits": torch.zeros(2, 1, V)}

    def fake_forward_beliefs(
        token_ids,
        *,
        return_logits=False,
        decode_last=False,
        **kwargs,
    ):
        assert return_logits and decode_last
        return None, injected["logits"]

    monkeypatch.setattr(model, "forward_beliefs", fake_forward_beliefs)

    injected["logits"] = torch.zeros(2, 1, V)
    injected["logits"][0, 0, 0] = float("nan")
    injected["logits"][1, 0, 1] = float("inf")
    with pytest.raises(
        ValueError,
        match=r"generation logits contain NaN or \+inf values in rows \[0, 1\]",
    ):
        model.generate(prompt, max_new_tokens=1, greedy=True)

    injected["logits"] = torch.zeros(2, 1, V)
    injected["logits"][0] = float("-inf")
    with pytest.raises(ValueError, match=r"generation logits have no finite value in rows \[0\]"):
        model.generate(prompt, max_new_tokens=1, greedy=False, top_k=2)

    injected["logits"] = torch.zeros(2, 1, V)
    injected["logits"][0, 0, 4:] = float("-inf")
    out = model.generate(prompt, max_new_tokens=1, greedy=False, top_k=2)
    assert out.shape == (2, 4)                              # filtered-out -inf remains valid
    with pytest.raises(
        ValueError,
        match=r"generation retained logits contain non-finite values in rows \[0\]",
    ):
        model.generate(prompt, max_new_tokens=1, greedy=False)


def test_greedy_ignores_temperature_topk_topp():
    # Greedy returns before any temperature/top_k/top_p logic: a wild temperature
    # and aggressive top_k/top_p alongside greedy=True must not change the result.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    plain = model.generate(prompt, max_new_tokens=4, greedy=True)
    wild = model.generate(prompt, max_new_tokens=4, greedy=True,
                          temperature=137.0, top_k=1, top_p=0.01)
    assert torch.equal(plain, wild)


def test_top_k_one_is_argmax_first_token():
    # top_k=1 (not greedy) leaves a single survivor -> softmax mass 1.0 -> multinomial
    # returns it deterministically; it equals the argmax token. No seed needed.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    logits = model(prompt)
    expected = logits[:, -1, :].argmax(dim=-1)
    out = model.generate(prompt, max_new_tokens=1, greedy=False, top_k=1)
    assert torch.equal(out[:, -1], expected)


def test_top_k_membership_first_token():
    # The first sampled token must lie among the k largest of the last-position logits.
    model = _tiny_model()
    V = model.cfg.vocab_size
    k = 3
    prompt = torch.randint(0, V, (2, 3))
    logits = model(prompt)
    topk_ids = logits[:, -1, :].topk(k, dim=-1).indices     # (B, k)
    torch.manual_seed(0)
    out = model.generate(prompt, max_new_tokens=1, greedy=False, top_k=k)
    chosen = out[:, -1]                                     # (B,)
    assert (chosen.unsqueeze(-1) == topk_ids).any(dim=-1).all()


def test_top_p_and_temperature_paths_run_in_vocab():
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    torch.manual_seed(0)
    out_p = model.generate(prompt, max_new_tokens=3, greedy=False, top_p=0.9)
    out_t = model.generate(prompt, max_new_tokens=3, greedy=False, temperature=0.7)
    for out in (out_p, out_t):
        assert out.shape == (2, 3 + 3)
        assert (out >= 0).all() and (out < V).all()


def test_prompt_longer_than_max_seq_len_does_not_error():
    # A prompt longer than max_seq_len must not error: the loop truncates to the last
    # max_seq_len tokens before each forward. The returned sequence keeps the FULL prompt.
    model = _tiny_model()
    V = model.cfg.vocab_size
    L = model.cfg.max_seq_len
    long_prompt = torch.randint(0, V, (2, L + 4))           # longer than max_seq_len
    out = model.generate(long_prompt, max_new_tokens=2, greedy=True)
    assert out.shape == (2, (L + 4) + 2)
    assert torch.equal(out[:, : L + 4], long_prompt)        # full prompt preserved
    assert (out >= 0).all() and (out < V).all()


def test_generate_is_training_isolated():
    # The safety oracle: generate changes no parameter and does not break training.
    model = _tiny_model()
    V = model.cfg.vocab_size
    before = model.prior_bank.mu_embed.detach().clone()
    prompt = torch.randint(0, V, (2, 3))
    torch.manual_seed(0)
    model.generate(prompt, max_new_tokens=4, greedy=False, top_k=2, top_p=0.9)
    after = model.prior_bank.mu_embed.detach().clone()
    assert torch.equal(before, after)                       # no parameter changed
    # training forward still produces a finite loss after a generate call
    tokens = torch.randint(0, V, (2, 4)); targets = torch.randint(0, V, (2, 4))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)


def test_generate_rejects_invalid_sampler_args():
    # audit C13 (2026-07-01): the normal (policy_mode='none') path validates its sampler knobs up
    # front. A negative max_new_tokens would silently no-op (empty loop, prompt returned unchanged);
    # temperature<=0, out-of-range top_k, and top_p outside (0, 1] previously failed late or produced
    # invalid probabilities.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    with pytest.raises(ValueError):
        model.generate(prompt, -1)
    with pytest.raises(ValueError):
        model.generate(prompt, 4, temperature=0.0)
    for bad_k in (0, V + 1):
        with pytest.raises(ValueError):
            model.generate(prompt, 4, top_k=bad_k)
    for bad_p in (0.0, 1.5):
        with pytest.raises(ValueError):
            model.generate(prompt, 4, top_p=bad_p)
    # max_new_tokens=0 stays valid (>= 0): the prompt comes back unchanged
    assert torch.equal(model.generate(prompt, 0), prompt)
    # greedy ignores the sampler knobs, so a nonstandard temperature must still run under greedy
    out = model.generate(prompt, 2, greedy=True, temperature=0.0)
    assert out.shape == (2, 3 + 2)


def test_generate_kwargs_order():
    # audit C18 (2026-07-01): the keyword-only block was reordered to the mandated convention
    # (defined float, defined bool, then Optionals); every knob stays keyword-passable in any order.
    model = _tiny_model()
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (1, 3))
    torch.manual_seed(0)
    out = model.generate(prompt, max_new_tokens=2, greedy=False, top_k=2, top_p=0.9, temperature=0.7)
    assert out.shape == (1, 3 + 2)
    assert (out >= 0).all() and (out < V).all()


def test_generate_reaches_efe_rollout_and_commits_first_action():
    # audit PB-05 (2026-07-12): efe_rollout (horizon>1) is now REACHABLE through generate() on a
    # cache-supported config -- it builds a bounded H-step beam menu and commits the FIRST action of
    # the selected policy. Replaces the old fail-closed NotImplementedError test.
    model = _tiny_cache_supported_policy_model(
        policy_mode="efe_rollout", policy_preference="flat", policy_horizon=2, policy_top_k=3)
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=1, greedy=True)
    assert out.shape == (1, 3)
    assert (out >= 0).all() and (out < model.cfg.vocab_size).all()
    assert torch.equal(out[:, :2], prompt)                            # prompt preserved


def test_efe_rollout_commits_first_action_of_selected_policy(monkeypatch):
    # audit PB-05: prove the H-step menu carries length-horizon candidates with a normalized log_prior,
    # and that generate commits candidates[selected_menu_index, 0] -- the FIRST action of the selected
    # policy, NOT its terminal action. A scorer spy captures the menu and forces the selected index.
    from vfe3.inference import policy as policy_module
    model = _tiny_cache_supported_policy_model(
        policy_mode="efe_rollout", policy_preference="flat", policy_horizon=2, policy_top_k=3)
    captured = {}

    def fake_scorer(context, candidates, preference, mdl, *, log_prior, **kwargs):
        captured["candidates"] = candidates.clone()
        captured["log_prior"] = log_prior.clone()
        B, Kp, _H = candidates.shape
        differ = candidates[..., 0] != candidates[..., -1]            # (B, Kp) first != terminal action
        assert bool(differ.any()), "fixture needs a beam whose first and terminal actions differ"
        chosen = torch.where(
            differ.any(-1), differ.float().argmax(-1), torch.zeros(B, dtype=torch.long))
        captured["chosen"] = chosen
        post = torch.zeros(B, Kp)
        post[torch.arange(B), chosen] = 1.0                          # force the selected menu index
        z = torch.zeros(B, Kp)
        return policy_module.PolicyScore(z, z, z, z, z, post)

    monkeypatch.setattr(policy_module, "get_policy", lambda name: fake_scorer)
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=1, greedy=True)

    cand = captured["candidates"]
    chosen = captured["chosen"]
    assert cand.shape == (1, 3, model.cfg.policy_horizon)            # (B, Kp, H): length-horizon menu
    assert torch.allclose(                                            # log_prior E(pi) is normalized
        captured["log_prior"].exp().sum(-1), torch.ones(1), atol=1e-6)
    first_action = cand[0, chosen[0], 0]
    terminal_action = cand[0, chosen[0], -1]
    assert out[0, -1].item() == first_action.item()                  # committed the FIRST action
    assert first_action.item() != terminal_action.item()            # and it is genuinely not the terminal one


def test_none_mode_generate_never_calls_build_topk_policy_menu(monkeypatch):
    # audit PB-05: the H-step menu builder is on the efe_rollout branch only. Under policy_mode='none'
    # generate() never reaches _policy_select, so build_topk_policy_menu is never imported/called; a
    # monkeypatched raise proves it, and the seeded output stays byte-identical to the pre-change pin.
    import vfe3.inference.candidate_menu as candidate_menu

    def boom(*args, **kwargs):
        raise AssertionError("build_topk_policy_menu must not be called under policy_mode='none'")

    monkeypatch.setattr(candidate_menu, "build_topk_policy_menu", boom)
    model = _tiny_model(seed=0)                                       # policy_mode defaults to 'none'
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=4, greedy=True)
    assert out.tolist() == [[1, 2, 3, 10, 3, 10, 3]]                 # base-commit golden, unchanged


def test_generate_efe_rollout_rejects_cache_unsupported_config():
    # audit PB-05: the H-step menu does NOT relax the scorer's cache gate. On a cache-unsupported
    # config efe_rollout still fails closed (get_policy('efe_rollout') raises), never silently paying
    # the dishonest full recompute (spec Section 3.5).
    from vfe3.inference.belief_cache import cache_supported
    model = _tiny_model(
        policy_mode="efe_rollout", policy_preference="flat", policy_horizon=2, policy_top_k=3,
        n_e_steps=2)                                                 # n_e_steps=2 breaks cache support
    assert not cache_supported(model.cfg)
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(NotImplementedError, match="belief-prefix cache"):
        model.generate(prompt, 1, greedy=True)


def test_policy_mode_rejects_call_time_sampler_knobs():
    # audit F9 (2026-06-28): under a policy scorer, generate() routes through _policy_select, which uses
    # policy_top_k / policy_precision and does NOT consume call-time temperature/top_k/top_p. Those must
    # be rejected at the call rather than silently dropped; 'greedy' stays honored. ('flat' is the only
    # generic-safe preference -- see test_generic_policy_rejects_context_requiring_preference.)
    model = _tiny_model(policy_mode="efe_one_step", policy_preference="flat")
    V = model.cfg.vocab_size
    prompt = torch.randint(0, V, (2, 3))
    for kwargs in ({"temperature": 0.7}, {"top_k": 3}, {"top_p": 0.9}):
        with pytest.raises(ValueError):
            model.generate(prompt, max_new_tokens=1, **kwargs)
    # the call-time defaults (temperature=1.0, no top_k/top_p) are accepted; greedy is honored
    out = model.generate(prompt, max_new_tokens=2, greedy=True)
    assert out.shape == (2, 3 + 2) and (out >= 0).all() and (out < V).all()


def test_generate_efe_builds_both_arms_before_pairing_cpu_and_cuda_rng(monkeypatch):
    events = []
    cpu_states = []
    cuda_states = []
    cuda_current = [torch.tensor([11], dtype=torch.uint8)]

    def fake_build(config_dict, state_dict, *, policy_overrides, device):
        events.append(("build", policy_overrides["policy_mode"]))
        torch.rand(1)                                             # construction may consume global RNG
        return policy_overrides["policy_mode"]

    def fake_generate(prompt_ids, model, cfg):
        events.append(("generate", model))
        cpu_states.append(torch.random.get_rng_state().clone())
        cuda_states.append([state.clone() for state in cuda_current])
        torch.rand(3)                                             # stand in for stochastic token sampling
        cuda_current[0] = torch.tensor([99], dtype=torch.uint8)
        return torch.tensor([[0]], dtype=torch.long)

    def fake_cuda_get_rng_state_all():
        return [state.clone() for state in cuda_current]

    def fake_cuda_set_rng_state_all(states):
        cuda_current[:] = [state.clone() for state in states]

    monkeypatch.setattr(generate_efe, "_build_model", fake_build)
    monkeypatch.setattr(generate_efe, "_generate", fake_generate)
    monkeypatch.setattr(generate_efe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generate_efe.torch.cuda, "get_rng_state_all", fake_cuda_get_rng_state_all)
    monkeypatch.setattr(generate_efe.torch.cuda, "set_rng_state_all", fake_cuda_set_rng_state_all)
    cfg = {
        "policy_mode":        "efe_one_step",
        "policy_preference":  "flat",
        "policy_score_terms": ("ambiguity",),
        "policy_top_k":       8,
        "policy_precision":   1.0,
        "policy_horizon":     1,
        "max_new_tokens":     1,
        "greedy":             False,
    }

    torch.manual_seed(7)
    base_out, policy_out = generate_efe._run_generation_arms(
        torch.tensor([[1]], dtype=torch.long), {}, {}, cfg, device="cpu",
    )

    assert events == [
        ("build", "none"),
        ("build", "efe_one_step"),
        ("generate", "none"),
        ("generate", "efe_one_step"),
    ]
    assert torch.equal(cpu_states[0], cpu_states[1])
    assert torch.equal(cuda_states[0][0], cuda_states[1][0])
    assert base_out.tolist() == [[0]]
    assert policy_out.tolist() == [[0]]


# ======================================================================================
# PB-06: sigma_mc consumer gate wiring in generate() -- fail-closed, and never touched off the arm.
# ======================================================================================

def test_generate_none_policy_never_touches_sigma_providers(monkeypatch):
    from vfe3.inference import sigma_gate as sg

    def _boom(*a, **k):
        raise AssertionError("policy_mode='none' must not inspect the sigma specification")

    monkeypatch.setattr(sg, "sigma_gate_spec_identity", _boom)
    monkeypatch.setattr(sg, "sigma_consumer_code_identity", _boom)
    monkeypatch.setattr(sg, "verify_sigma_consumer_gate", _boom)
    model = _tiny_model()                                       # policy_mode='none'
    out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, greedy=True)
    assert out.shape == (1, 5)


def test_generate_likelihood_entropy_policy_never_touches_sigma_providers(monkeypatch):
    from vfe3.inference import sigma_gate as sg

    def _boom(*a, **k):
        raise AssertionError("non-sigma_mc ambiguity must not read the sigma artifact")

    monkeypatch.setattr(sg, "sigma_gate_spec_identity", _boom)
    monkeypatch.setattr(sg, "verify_sigma_consumer_gate", _boom)
    model = _tiny_model(policy_mode="efe_one_step", policy_preference="flat", policy_top_k=4)
    out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, greedy=True)
    assert out.shape == (1, 5)


def test_generate_sigma_mc_calls_consumer_gate_and_fails_closed(tmp_path, monkeypatch):
    # A model whose ambiguity is flipped to sigma_mc after construction must fail closed at the consumer
    # boundary: the production preregistry resolves the live spec identity to FAIL, so generate() raises
    # (it never reaches the not-yet-implemented estimator). The sealed corpus is seeded into a temporary
    # cache dir behind the default_cache_dir seam (mirroring tests/test_sigma_gate.py) so the assertion
    # is deterministically the intended production-FAIL ValueError on any machine, never a
    # FileNotFoundError from a missing ambient wikitext-103 test cache.
    from vfe3.data import datasets as datasets_module
    from vfe3.data.datasets import cache_path
    corpus = cache_path("wikitext-103", "test", suffix="pt", cache_dir=tmp_path)
    corpus.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.arange(64, dtype=torch.int64), corpus)
    monkeypatch.setattr(datasets_module, "default_cache_dir", lambda: tmp_path)
    model = _tiny_model(policy_mode="efe_one_step", policy_preference="flat", policy_top_k=4)
    model.cfg.policy_ambiguity_mode = "sigma_mc"
    with pytest.raises(ValueError, match="not registered as PASS"):
        model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=1, greedy=True)


def _run_sigma_mc_synthetic_pass(tmp_path, monkeypatch, device, *, policy_mode, **overrides):
    """SYNTHETIC PASS plumbing ONLY (NOT an empirical sigma-arm validation): build a temporary governing
    identity, code identity, sealed corpus, PASS artifact, and manifest, inject the providers at
    generate()'s call sites, then run generate() so the consumer gate opens and _amb_sigma_mc actually
    runs. Mirrors tests/test_sigma_gate.py::_valid_gate, adapted to generate()'s (root-less) call sites.
    """
    import json

    from vfe3.data import datasets as datasets_module
    from vfe3.data.datasets import cache_path
    from vfe3.inference import sigma_gate as sg
    from vfe3.run_artifacts import (model_behavior_fingerprint, semantic_config_fingerprint,
                                    sigma_behavior_config)

    # Sealed corpus behind the default_cache_dir seam so sigma_measurement_context resolves off-cache.
    corpus = cache_path("wikitext-103", "test", suffix="pt", cache_dir=tmp_path)
    corpus.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.arange(64, dtype=torch.int64), corpus)
    monkeypatch.setattr(datasets_module, "default_cache_dir", lambda: tmp_path)

    # Fixed synthetic governing + code identities injected at generate()'s root-less call sites.
    spec, code = "synthetic-spec-identity", "synthetic-code-identity"
    monkeypatch.setattr(sg, "sigma_gate_spec_identity", lambda *a, **k: spec)
    monkeypatch.setattr(sg, "sigma_consumer_code_identity", lambda *a, **k: code)

    # Build the tiny live model only after injection; flip to the sigma_mc arm post-construction.
    model = _tiny_model(policy_mode=policy_mode, policy_preference="flat",
                        policy_top_k=4, **overrides).to(device)
    model.cfg.policy_ambiguity_mode = "sigma_mc"

    meas = sg.sigma_measurement_context(model.cfg)
    behavior = model_behavior_fingerprint(sigma_behavior_config(model.cfg), model.state_dict())
    ctx_fp = semantic_config_fingerprint(meas)
    record = {
        "status": "PASS", "checkpoint_id": "synthetic-generate-checkpoint",
        "model_behavior_sha256": behavior, "spec_commit": spec, "code_identity_sha256": code,
        "measurement_context": meas, "measurement_context_sha256": ctx_fp, "seeds": meas["seeds"],
        "sigma_ce_spearman": 0.5, "spearman_ci": [0.3, 0.7], "permutation_floor": 0.1,
        "stratified_ce": {"monotone": True}, "sigma_binned_ece": 0.01, "thresholds": meas["thresholds"],
    }
    art = tmp_path / "synthetic_generate_gate.json"
    art.write_text(json.dumps(record), encoding="utf-8")
    manifest = {spec: {"status": "PASS", "artifact_sha256": sg.canonical_json_sha256(art),
                       "test_only": True}}
    monkeypatch.setattr(sg, "load_sigma_gate_preregistry", lambda *a, **k: manifest)
    model.cfg.policy_sigma_gate_artifact = str(art)

    prompt = torch.tensor([[1, 2, 3]], device=device)
    before = torch.get_rng_state()
    out = model.generate(prompt, max_new_tokens=2, greedy=True)
    after = torch.get_rng_state()
    assert torch.equal(before, after)                          # local MC generator: global RNG untouched
    assert out.device.type == device.type                      # the run stays on device
    return out


def test_generate_sigma_mc_synthetic_pass_runs_estimator_cpu(tmp_path, monkeypatch):
    # SYNTHETIC PASS plumbing ONLY: the gate opens on a temporary PASS artifact so generate() runs the
    # sigma_mc estimator end-to-end on CPU. The empirical FAIL preregistry remains authoritative.
    out = _run_sigma_mc_synthetic_pass(tmp_path, monkeypatch, torch.device("cpu"),
                                       policy_mode="efe_one_step")
    assert out.shape == (1, 5)
    assert int(out.max()) < 16 and int(out.min()) >= 0
    assert torch.equal(out[:, :3], torch.tensor([[1, 2, 3]]))  # prompt preserved


@pytest.mark.skipif(_DEVICE.type != "cuda",
                    reason="RTX 5090 CUDA smoke; set VFE3_TEST_DEVICE=cuda to run")
def test_efe_rollout_sigma_mc_cuda_synthetic_pass(tmp_path, monkeypatch):
    # SYNTHETIC PASS plumbing ONLY (NOT an empirical sigma-arm validation): drives efe_rollout through
    # generate() and _amb_sigma_mc on the RTX 5090 with the consumer gate held open on a temporary PASS
    # artifact, pinning that the antithetic MC sampler keeps the run on-device and leaves the global RNG
    # untouched. SKIPS cleanly on a CPU host (VFE3_TEST_DEVICE unset).
    out = _run_sigma_mc_synthetic_pass(tmp_path, monkeypatch, _DEVICE,
                                       policy_mode="efe_rollout", policy_horizon=2)
    assert out.shape == (1, 5)
    assert int(out.max()) < 16 and int(out.min()) >= 0
